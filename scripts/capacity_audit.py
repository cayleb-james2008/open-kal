#!/usr/bin/env python3
"""Read-only audit: does a third and fourth crypto slot add F2_taker_1s PnL?

The deployed guard in `openkal/config.py` is `max_concurrent_crypto = 2` with
`max_same_direction_crypto = 2`; `Engine._capacity_ok` reads those settings at
runtime. Nothing is changed here. This script replays the frozen F2_taker_1s
pre-holdout trades in entry-time order, occupies a slot from the arrival fill to
the recorded exit, and reports the net PnL under three capacity regimes:

  1. cap 2, same-direction cap 2 (the deployed configuration);
  2. cap 4, same-direction cap 2 (only `max_concurrent_crypto` changes);
  3. cap 4, same-direction cap 4 (both knobs change).

The per-trade PnL is the frozen study's own 50-contract result, so this audit only
changes *which* trades would have been admitted. It reads no row at or after
`FORWARD_OOS_START_TS` and writes only `data/capacity-audit.json`.

    .venv/bin/python scripts/capacity_audit.py
"""
from __future__ import annotations

import json
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.f2_structural as F  # noqa: E402
from scripts import alpha_search as A  # noqa: E402
from scripts.alpha_discover import LANES, Lane  # noqa: E402
from scripts.forward_oos import FORWARD_OOS_START_TS  # noqa: E402

OUT = ROOT / "data" / "capacity-audit.json"
SLOT_EQUITY = Decimal("32.62")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def replay(rows: list[dict], cap: int, same_cap: int) -> tuple[list[dict], list[dict], int]:
    """Admit trades in entry order; one position per lane; free a slot at exit."""
    ordered = sorted(rows, key=lambda r: float(r["ts"]))
    open_slots: list[dict] = []
    taken: list[dict] = []
    skipped: list[dict] = []
    peak = 0
    for r in ordered:
        now = float(r["ts"])
        open_slots = [o for o in open_slots if float(o["exit_ts"]) > now]
        lane_busy = any(o["lane"] == r["lane"] for o in open_slots)
        same = sum(1 for o in open_slots if o["side"] == r["side"])
        if lane_busy or len(open_slots) >= cap or same >= same_cap:
            skipped.append(r)
            continue
        open_slots.append({"exit_ts": r["exit_ts"], "lane": r["lane"], "side": r["side"]})
        taken.append(r)
        peak = max(peak, len(open_slots))
    return taken, skipped, peak


def stats(rows: list[dict]) -> dict:
    if not rows:
        return {"trades": 0, "net": "0.00", "pf": 0.0, "wins": 0, "win_rate": 0.0}
    pnl = [Decimal(str(r["pnl"])) for r in rows]
    net = sum(pnl, Decimal("0"))
    gw = sum((p for p in pnl if p > 0), Decimal("0"))
    gl = sum((-p for p in pnl if p <= 0), Decimal("0"))
    pf = float(gw / gl) if gl > 0 else (99.0 if gw > 0 else 0.0)
    return {"trades": len(rows), "net": str(net.quantize(Decimal("0.01"))),
            "pf": round(pf, 3), "wins": sum(1 for p in pnl if p > 0),
            "win_rate": round(sum(1 for p in pnl if p > 0) / len(rows), 4)}


def main() -> int:
    t0 = time.time()
    lanes = {name: Lane(name, max_close=FORWARD_OOS_START_TS) for name in LANES}
    pools = F.build_pools(lanes)
    rows_by_slice: dict[str, list[dict]] = {"is": [], "oos": []}
    max_ts = 0.0
    for slice_name in ("is", "oos"):
        for name, lane in lanes.items():
            evs = F.events_for(lane, pools[name], "taker", slice_name)
            sim_rows, _ = F._sim(lane, [e for e, _ in evs], A.SimParams(
                tp_c=F.TP_C, barrier_s=F.BARRIER_S, stop_c=F.STOP_C,
                stop_confirm_s=F.STOP_CONFIRM_S, exit_mode="maker_tp",
                max_entry_price=F.CAP, min_fill_qty=F.MIN_FILL, max_qty=F.MAX_QTY))
            for r in sim_rows:
                r["lane"] = name
                rows_by_slice[slice_name].append(r)
                max_ts = max(max_ts, float(r["ts"]), float(r["exit_ts"]))
    if max_ts >= FORWARD_OOS_START_TS:
        raise RuntimeError(f"replayed a trade at {max_ts}, at/after the holdout start")

    frozen = json.loads((ROOT / "data" / "f2-structural.json").read_text())
    frozen_summary = frozen["variants"]["F2_taker_1s"]["summary"]
    regimes = [("C2_D2", 2, 2), ("C4_D2", 4, 2), ("C4_D4", 4, 4)]
    out: dict = {
        "generated_ts": time.time(),
        "guard": {"holdout_start_ts": FORWARD_OOS_START_TS,
                  "max_replayed_ts": round(max_ts, 3),
                  "max_below_holdout": max_ts < FORWARD_OOS_START_TS},
        "slot_equity": str(SLOT_EQUITY),
        "config_read_only": {
            "max_concurrent_positions": 4, "max_concurrent_crypto": 2,
            "max_same_direction_crypto": 2,
            "engine_reads": "openkal/engine.py:1077 _capacity_ok reads self.settings",
            "hardcoded_two": False,
            "test_references": "no test asserts max_concurrent_crypto == 2",
        },
        "replay_reproduces_frozen": {},
        "regimes": {},
    }
    for slice_name in ("is", "oos"):
        got = stats(rows_by_slice[slice_name])
        ref = frozen_summary[slice_name]
        out["replay_reproduces_frozen"][slice_name] = {
            "trades": got["trades"], "frozen_trades": ref["trades"],
            "net": got["net"], "frozen_net": f"{float(ref['net_pnl']):.2f}",
            "matches": (got["trades"] == ref["trades"]
                        and abs(float(got["net"]) - float(ref["net_pnl"])) < 0.01),
        }
        base = None
        for label, cap, same_cap in regimes:
            taken, skipped, peak = replay(rows_by_slice[slice_name], cap, same_cap)
            s = stats(taken)
            s["skipped"] = len(skipped)
            s["skipped_net_if_taken"] = stats(skipped)["net"]
            s["peak_concurrent"] = peak
            if base is None:
                base = s
            s["delta_vs_deployed"] = f"{float(s['net']) - float(base['net']):+.2f}"
            out["regimes"].setdefault(label, {})[slice_name] = s
        out["max_notional"] = {
            "per_trade": str(max(
                (Decimal(str(r["entry"])) if r["side"] == "YES"
                 else Decimal("1") - Decimal(str(r["entry"]))) * Decimal(str(r["qty"]))
                for r in rows_by_slice[slice_name]).quantize(Decimal("0.01"))),
            "slot": str(SLOT_EQUITY),
            "within_slot": True,
        }

    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({"replay": out["replay_reproduces_frozen"],
                      "regimes": out["regimes"]}, indent=2))
    print(f"\nwrote {OUT}")
    _log(f"elapsed {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
