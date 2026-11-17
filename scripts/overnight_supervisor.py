#!/usr/bin/env python3
"""Overnight supervisor: watch the engine, break the switch, rebuild the tape.

This runs unattended from now until the morning report. It does three things and
nothing else:

  1. Every 15 minutes it reads the engine's own state, the venue balance and the
     ledger reconciliation, and writes one line per check to
     `data/overnight-supervisor.log`.
  2. It owns the hard drawdown circuit breaker. If the venue balance falls below
     the floor, or the reconciliation reports a discrepancy larger than the cap,
     it demotes `data/strategy-verdict.json` (promoted -> false) and restarts the
     engine flat, which disarms live trading back to paper. It never restarts
     while a position or a resting order is open: `scripts/restart_flat.sh` refuses
     that itself, and the supervisor treats the refusal as a reason to keep
     watching rather than to force anything.
  3. At the rebuild hour it rebuilds the sub-second cache from the expanded
     production tape and re-runs the frozen discovery, and at the report hour it
     writes the morning handoff.

The supervisor never arms live and never promotes a strategy. Arming is a
deliberate act with a canary in front of it; the supervisor's job is to make sure
that an armed engine cannot bleed while nobody is watching.

    .venv/bin/python scripts/overnight_supervisor.py --until 2026-11-17T04:30:00-05:00
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOG = ROOT / "data" / "overnight-supervisor.log"
VERDICT = ROOT / "data" / "strategy-verdict.json"
REPORT = ROOT / "data" / "overnight-report.md"
STATE = ROOT / "data" / "overnight-state.json"
CENTS = Decimal("0.01")


def now() -> datetime:
    return datetime.now(timezone.utc).astimezone()


def log(msg: str) -> None:
    line = f"{now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


def venue_balance() -> Decimal | None:
    """The venue's own balance, to the cent."""
    code = (
        "import asyncio,sys;sys.path.insert(0,%r)\n"
        "from openkal.auth import KalshiClient\n"
        "from openkal.config import load_settings\n"
        "async def m():\n"
        "    c=KalshiClient(load_settings())\n"
        "    try:\n"
        "        s,d=await c.get('/portfolio/balance')\n"
        "    finally:\n"
        "        await c.close()\n"
        "    print(s, (d or {}).get('balance_dollars',''))\n"
        "asyncio.run(m())\n" % str(ROOT)
    )
    try:
        out = subprocess.run([str(ROOT / ".venv/bin/python"), "-c", code],
                             capture_output=True, text=True, timeout=90).stdout.strip()
    except (subprocess.SubprocessError, OSError) as exc:
        log(f"balance read failed: {exc}")
        return None
    parts = out.split()
    if len(parts) < 2 or parts[0] != "200":
        log(f"balance read failed: {out!r}")
        return None
    try:
        return Decimal(parts[1])
    except Exception:  # noqa: BLE001 - a malformed venue number is data, not a crash
        return None


def engine_state() -> dict | None:
    try:
        out = subprocess.run(["curl", "-s", "--max-time", "10",
                              "http://127.0.0.1:8787/api/state"],
                             capture_output=True, text=True, timeout=20).stdout
        return json.loads(out) if out.strip() else None
    except (subprocess.SubprocessError, OSError, ValueError):
        return None


def reconcile(since: float | None) -> dict | None:
    cmd = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts" / "reconcile_live.py"),
           "--json", str(ROOT / "data" / "overnight-reconcile.json")]
    if since:
        cmd += ["--since", str(int(since))]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return json.loads((ROOT / "data" / "overnight-reconcile.json").read_text())
    except (subprocess.SubprocessError, OSError, ValueError):
        return None


def demote_and_restart(reason: str) -> None:
    """Break the switch: stop arming, flatten, go back to paper."""
    try:
        verdict = json.loads(VERDICT.read_text())
    except (OSError, ValueError):
        verdict = {}
    if verdict.get("promoted"):
        verdict["promoted"] = False
        verdict["disarmed_at"] = now().isoformat()
        verdict["disarm_reason"] = reason
        VERDICT.write_text(json.dumps(verdict, indent=1))
        log(f"verdict demoted: {reason}")
    else:
        log(f"verdict already demoted; disarm for: {reason}")
    # The restart exists to disarm a live engine. If the engine is already in paper
    # mode there is nothing to disarm, and restarting it every fifteen minutes
    # would churn the tape for no reason.
    st = engine_state()
    if not (st or {}).get("live_armed"):
        log("engine is already in paper mode; nothing to disarm, engine left running")
        return
    try:
        out = subprocess.run(["bash", str(ROOT / "scripts" / "restart_flat.sh")],
                             capture_output=True, text=True, timeout=300)
        log(f"restart_flat: rc={out.returncode} {out.stdout.strip()[-200:]} "
            f"{out.stderr.strip()[-200:]}")
    except (subprocess.SubprocessError, OSError) as exc:
        log(f"restart_flat failed: {exc}")


def state_write(payload: dict) -> None:
    STATE.write_text(json.dumps(payload, indent=1, default=str))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", required=True, help="ISO local time to stop")
    ap.add_argument("--interval", type=float, default=900.0)
    ap.add_argument("--balance-floor", default="105.00")
    ap.add_argument("--delta-cap", default="0.05")
    ap.add_argument("--epoch-start", default=None)
    ap.add_argument("--rebuild-at", default=None, help="ISO local time")
    ap.add_argument("--report-at", default=None, help="ISO local time")
    ap.add_argument("--no-rebuild", action="store_true")
    args = ap.parse_args()

    floor = Decimal(args.balance_floor)
    cap = Decimal(args.delta_cap)
    until = datetime.fromisoformat(args.until)
    rebuild_at = (datetime.fromisoformat(args.rebuild_at) if args.rebuild_at else None)
    report_at = (datetime.fromisoformat(args.report_at) if args.report_at else None)
    epoch_start = float(args.epoch_start) if args.epoch_start else None

    stop = {"flag": False}

    def handle(signum, frame):  # noqa: ANN001 - signal handler signature
        stop["flag"] = True
        log("signal received; supervisor will stop after this check")

    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)

    log(f"supervisor up: until {until.isoformat()}, floor ${floor}, "
        f"delta cap ${cap}, interval {args.interval:.0f}s")

    rebuilt = False
    reported = False
    armed_seen = None
    while True:
        t = now()
        bal = venue_balance()
        st = engine_state()
        armed = None
        if st:
            armed = bool(st.get("live_armed"))
            if armed_seen is None:
                armed_seen = armed
        rec = reconcile(epoch_start)
        delta = None
        if rec:
            try:
                delta = Decimal(str(rec.get("delta", "0")))
            except Exception:  # noqa: BLE001
                delta = None
        breaks = []
        if bal is not None and bal < floor:
            breaks.append(f"venue balance {bal} < floor {floor}")
        if delta is not None and abs(delta) > cap:
            breaks.append(f"reconciliation delta {delta} > cap {cap}")
        mode = (st or {}).get("mode", "?")
        log(f"check balance={bal} mode={mode} armed={armed} "
            f"delta={delta} blockers={len((st or {}).get('readiness', {}).get('blockers', []))}")
        if breaks:
            demote_and_restart("; ".join(breaks))
        state_write({"ts": t.isoformat(), "balance": str(bal) if bal is not None else None,
                     "mode": mode, "armed": armed, "delta": str(delta) if delta is not None else None,
                     "breaker": breaks, "epoch_start": epoch_start})

        if (rebuild_at and not rebuilt and not args.no_rebuild
                and t >= rebuild_at.astimezone(t.tzinfo)):
            log("rebuild hour: rebuilding the sub-second cache from the production tape")
            try:
                subprocess.run(["bash", str(ROOT / "scripts" / "rebuild_cache.sh")],
                               capture_output=True, text=True, timeout=7200)
                log("cache rebuilt; re-running the frozen discovery")
                subprocess.run([str(ROOT / ".venv/bin/python"),
                                str(ROOT / "scripts" / "subsec_discovery.py")],
                               capture_output=True, text=True, timeout=7200)
                log("discovery re-run complete")
            except (subprocess.SubprocessError, OSError) as exc:
                log(f"rebuild failed: {exc}")
            rebuilt = True

        if report_at and not reported and t >= report_at.astimezone(t.tzinfo):
            log("report hour: writing the morning handoff")
            try:
                subprocess.run([str(ROOT / ".venv/bin/python"),
                                str(ROOT / "scripts" / "overnight_report.py")],
                               capture_output=True, text=True, timeout=1800)
                log("morning report written")
            except (subprocess.SubprocessError, OSError) as exc:
                log(f"report failed: {exc}")
            reported = True

        if t >= until.astimezone(t.tzinfo):
            log("supervisor finished its window")
            break
        if stop["flag"]:
            break
        # Sleep until the next scheduled moment, never past it: the rebuild and the
        # report have to land on their hour, not a quarter-hour after it.
        nxt = until.astimezone(t.tzinfo)
        if rebuild_at is not None and not rebuilt and rebuild_at.astimezone(t.tzinfo) > t:
            nxt = min(nxt, rebuild_at.astimezone(t.tzinfo))
        if report_at is not None and not reported and report_at.astimezone(t.tzinfo) > t:
            nxt = min(nxt, report_at.astimezone(t.tzinfo))
        nap = min(args.interval, max(5.0, (nxt - t).total_seconds()))
        time.sleep(nap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
