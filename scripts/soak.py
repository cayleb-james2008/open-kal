"""Forward soak: paper only, new epoch, then the gate table exactly as it stands.

`live_armed` is forced false here and is re-forced on every pass, regardless of what
`.env` says. Arming is a separate, deliberate act; a soak runner must never be able to
put real money at risk because an operator left `OPENKAL_ARM_LIVE=1` in the environment.
Real spend for this run is therefore exactly $0.00.

At the end it writes `data/soak-report.json` with the eight readiness rows, the replay
reconciliation against the ledger, the deployed sequence-gap count, and the per-lane
Brier/ECE measured against the settlement of every matched window.

    .venv/bin/python scripts/soak.py --minutes 60
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import threading
import time
from decimal import Decimal
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openkal.api import SnapshotBox, serve  # noqa: E402
from openkal.config import load_settings  # noqa: E402
from openkal.engine import Engine  # noqa: E402
from openkal.readiness import evaluate  # noqa: E402
from openkal.store import Store  # noqa: E402

REPORT = ROOT / "data" / "soak-report.json"


def lane_calibration(store: Store, since: float) -> dict:
    """Brier/ECE per lane over the epoch's recorded ticks, against settlement truth."""
    from openkal.features import brier, expected_calibration_error

    markets_path = ROOT / "data" / "settled-markets.json"
    if not markets_path.exists():
        return {}
    markets = json.loads(markets_path.read_text())
    truth = {}
    for m in markets:
        if m.get("result") in ("yes", "no") and m.get("close_time"):
            from datetime import datetime, timezone

            try:
                close = datetime.fromisoformat(m["close_time"].replace("Z", "+00:00")).timestamp()
            except ValueError:
                continue
            truth[m["ticker"]] = (1 if m["result"] == "yes" else 0, close)
    rows = store._db.execute(  # noqa: SLF001 - a read-only report, not a code path
        "SELECT series,ticker,yes_bid,yes_ask,p_model FROM ticks WHERE ts>=?", (since,)
    ).fetchall()
    out: dict[str, dict] = {}
    per_lane: dict[str, list[tuple[float, int]]] = {}
    per_lane_mid: dict[str, list[tuple[float, int]]] = {}
    for r in rows:
        item = truth.get(r["ticker"])
        if item is None or r["p_model"] is None:
            continue
        y = item[0]
        try:
            mid = (Decimal(str(r["yes_bid"])) + Decimal(str(r["yes_ask"]))) / 2
        except Exception:  # noqa: BLE001
            continue
        per_lane.setdefault(r["series"], []).append((float(r["p_model"]), y))
        per_lane_mid.setdefault(r["series"], []).append((float(mid), y))
    for lane, pairs in per_lane.items():
        out[lane] = {
            "ticks": len(pairs),
            "brier": round(brier(pairs), 4),
            "ece": round(expected_calibration_error(pairs), 4),
            "brier_mid": round(brier(per_lane_mid[lane]), 4),
            "ece_mid": round(expected_calibration_error(per_lane_mid[lane]), 4),
        }
    return out


def build_report(engine: Engine, soak_minutes: float) -> dict:
    verdict = evaluate(engine)
    epoch = engine.store.epoch_stats(engine.epoch_start)
    all_time = engine.store.stats()
    gap = None
    try:
        gap = engine.ws.seq.gaps if engine.ws else None
    except Exception:  # noqa: BLE001
        gap = None
    report = {
        "generated_ts": time.time(),
        "soak_minutes": round(soak_minutes, 2),
        "live_armed": engine.live_armed,
        "real_spend_usd": "0.00",
        "mode": "paper" if not engine.live_armed else "LIVE (unexpected)",
        "readiness": verdict,
        "gate_rows": [{"key": c["key"], "ok": c["ok"], "detail": c["detail"],
                       "label": c["label"]} for c in verdict["checks"]],
        "ws": {
            "messages": engine.ws.messages if engine.ws else 0,
            "seq_gaps": gap,
            "sync_problems": engine.ws.sync_problems if engine.ws else None,
            "reconnects": engine.ws.reconnects if engine.ws else None,
            "lag_ms": round(engine.ws.lag_ms, 1) if engine.ws else None,
            "degraded": bool(engine.ws.degraded) if engine.ws else None,
        },
        "ledger": {
            "epoch_trades": int(epoch["trades"]),
            "epoch_pnl": str(Decimal(str(epoch["pnl"])).quantize(Decimal("0.01"))),
            "all_time_trades": int(all_time["trades"]),
            "all_time_pnl": str(Decimal(str(all_time["pnl"])).quantize(Decimal("0.01"))),
            "equity": str(engine.equity.quantize(Decimal("0.01"))),
            "start_equity": str(engine.start_equity),
            "venue_balance": str(engine.venue_balance.quantize(Decimal("0.01"))),
        },
        "reproduction": verdict.get("reproduction"),
        "lanes": lane_calibration(engine.store, engine.epoch_start),
    }
    REPORT.write_text(json.dumps(report, indent=2, default=str))
    return report


async def run(minutes: float, no_dashboard: bool, db: str | None = None) -> int:
    settings = load_settings()
    # Hard paper lock. `.env` may still say OPENKAL_ARM_LIVE=1; that is a permission,
    # not an instruction, and this runner never acts on it.
    settings.allow_live = False
    settings.arm_live = False
    if db:
        settings.db_path = Path(db) if Path(db).is_absolute() else ROOT / db
    # `minutes` is how long this paper run lasts, not a readiness criterion: the
    # readiness gate is functional now and reads its own evidence (see readiness.py).

    store = Store(settings.db_path)
    now = time.time()
    # `epoch_equity` is the BASE the engine adds realised P&L to, not the net equity.
    # Feeding the net back in double-counts the ledger: 245.60 became 228.89. The only
    # honest base is the venue's own balance, read live.
    base = Decimal("0")
    try:
        from openkal.auth import KalshiClient
        from openkal.broker import Broker

        client = KalshiClient(settings)
        try:
            base = await Broker(client).balance()
        finally:
            await client.close()
    except Exception as exc:  # noqa: BLE001 - fall back to the stored base
        print(f"balance read failed: {type(exc).__name__}: {exc}", flush=True)
    if base <= 0:
        base = Decimal(str(store.get_meta("base_equity") or store.get_meta("epoch_equity") or "0"))
    if base <= 0:
        base = settings.paper_equity
    current = base.quantize(Decimal("0.01"))
    store.set_meta("base_equity", str(current))
    store.set_meta("epoch_equity", str(current))
    store.set_meta("epoch_start", now)
    store.set_meta("soak_started", now)

    engine = Engine(settings, store)
    engine.live_armed = False

    box = SnapshotBox()
    server = None
    if not no_dashboard:
        server = serve(settings.host, settings.port, box, settings.dashboard_dir)
        threading.Thread(target=server.serve_forever, name="dashboard", daemon=True).start()
        print(f"dashboard http://{settings.host}:{settings.port}", flush=True)

    print(f"soak: env={settings.env_name} paper=forced slots={len(engine.slots)} "
          f"epoch_equity={current} live_armed=False", flush=True)

    async def publish() -> None:
        while True:
            engine.live_armed = False          # belt and braces, every cycle
            box.put(engine.snapshot())
            await asyncio.sleep(0.5)

    publisher = asyncio.create_task(publish())
    started = time.time()
    report = None

    async def snapshot() -> None:
        """Write the live gate table every 30s, so an interrupted run still has evidence."""
        while True:
            await asyncio.sleep(30)
            try:
                live = evaluate(engine)
                payload = {
                    "generated_ts": time.time(),
                    "soak_minutes": round((time.time() - started) / 60.0, 2),
                    "live_armed": engine.live_armed,
                    "real_spend_usd": "0.00",
                    "gate_rows": [{"key": c["key"], "ok": c["ok"], "detail": c["detail"]}
                                  for c in live["checks"]],
                    "ws": {"messages": engine.ws.messages if engine.ws else 0,
                           "seq_gaps": engine.ws.seq.gaps if engine.ws else None,
                           "sync_problems": engine.ws.sync_problems if engine.ws else None},
                    "ledger": {"epoch_trades": int(engine.store.epoch_stats(engine.epoch_start)["trades"]),
                               "all_time_trades": int(engine.store.stats()["trades"]),
                               "all_time_pnl": str(engine.store.stats()["pnl"]),
                               "equity": str(engine.equity.quantize(Decimal("0.01")))},
                }
                (ROOT / "data" / "soak-live-report.json").write_text(
                    json.dumps(payload, indent=2, default=str))
            except Exception:  # noqa: BLE001 - a snapshot must never disturb the soak
                pass

    snapper = asyncio.create_task(snapshot())
    try:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(engine.run(), timeout=minutes * 60.0)
    finally:
        # Capture before teardown: after the loop stops, the live counters (WS gap
        # count, feed ages, fault times) are still the run's final truth. A
        # CancelledError is a BaseException, so suppressing only Exception around the
        # publisher is how the first attempt lost its report.
        try:
            report = build_report(engine, (time.time() - started) / 60.0)
        except Exception as exc:  # noqa: BLE001 - never lose the run to a report bug
            print(f"report failed: {type(exc).__name__}: {exc}", flush=True)
        engine.stop()
        publisher.cancel()
        snapper.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await publisher
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await snapper
        if server:
            server.shutdown()
    if report is None:
        print("no report written", flush=True)
        return 1
    print(json.dumps({"gate_rows": report["gate_rows"], "ws": report["ws"],
                      "ledger": report["ledger"], "lanes": report["lanes"]}, indent=2))
    print(f"wrote {REPORT}")
    return 0 if report["readiness"]["ready"] else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=15.0,
                        help="how long this paper run lasts (not a readiness criterion)")
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--db", default=None, help="scratch ledger for a verification run")
    args = parser.parse_args()
    return asyncio.run(run(args.minutes, args.no_dashboard, args.db))


if __name__ == "__main__":
    raise SystemExit(main())
