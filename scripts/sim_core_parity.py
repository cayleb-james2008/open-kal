#!/usr/bin/env python3
"""Outside check on sim_core: the F2 `taker|tp5|none` row, re-run through the new core.

The prior study's corrected-convention row (scripts/evolve_f2_variants.py on the same
frozen cache) was IS 58 trades / -$79.31, OOS 34 / -$73.48. The new core should land
close to that; known differences are listed in the output, not tuned away:
  * stop and barrier exits now arrive 1s after the decision (the old core exited at
    the decision-time quote);
  * the arrival quote is the last row, never an older valid one (old: 2s walk-back).

Pools here are the legacy ones (nearest-print spot) on purpose: this script checks
the simulator, not the signal. The tracks read scripts/rnd_pool_build.py's causal pools.

    .venv/bin/python scripts/sim_core_parity.py
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openkal import sim_core as S  # noqa: E402
from scripts import rnd_common as R  # noqa: E402
from scripts import f2_structural as F  # noqa: E402
from scripts.bounded_sweep import pick  # noqa: E402

OUT = ROOT / "data" / "sim-core-parity.json"


def main() -> int:
    t0 = time.time()
    lanes = R.load_lanes()
    print(f"lanes loaded {time.time() - t0:.0f}s", flush=True)
    cut = R.global_cut(lanes)
    per_lane_cut = {n: lane.cut_ts for n, lane in lanes.items()}
    pools = F.build_pools(lanes)
    print(f"pools built {time.time() - t0:.0f}s", flush=True)
    rows = {"is": [], "oos": []}
    skips = {"is": Counter(), "oos": Counter()}
    for name, lane in lanes.items():
        pool = pools[name]
        mask, ly = F.frozen_mask(pool, "taker")
        for i in pick(pool, mask, cooldown=F.COOLDOWN, per_window=F.PER_WINDOW):
            sl = str(pool["slice"][i])          # the old per-lane split, for parity
            tid = int(pool["tid"][i])
            side = S.YES if bool(ly[i]) else S.NO
            tp = R.tape(lane, tid)
            try:
                b, a = tp.quote(float(pool["ts"][i]) + S.LAG_S)
                qty = S.stake_qty(S.entry_price(side, S.TAKER, b, a), R.STAKE)
            except (S.CrossedBook, S.NoQuote):
                qty = S.MAX_QTY
            tr = S.run_trade(tp, side=side, trigger_ts=float(pool["ts"][i]),
                             entry_style=S.TAKER, qty=qty, cost_lo=Decimal("0.25"),
                             tp_c=Decimal("0.05"), stop_c=Decimal("0.07"),
                             stop_confirm_s=2.0, hold_s=90.0, barrier_exit="taker")
            if isinstance(tr, S.Skip):
                skips[sl][tr.reason] += 1
                continue
            rows[sl].append(R.trade_row(tr, lane=name, ticker=lane.tickers[tid],
                                        close=float(pool["close"][i]), slice_name=sl))
    out = {
        "global_cut": cut, "per_lane_cut": per_lane_cut,
        "prior_corrected_row": {"is": {"trades": 58, "net": -79.31},
                                "oos": {"trades": 34, "net": -73.48}},
        "sim_core": {s: R.summarize(rows[s], dict(skips[s])) for s in rows},
        "seconds": round(time.time() - t0, 1),
    }
    R.dump(OUT, out)
    for s in ("is", "oos"):
        m = out["sim_core"][s]
        print(f"{s.upper():3s} trades {m['trades']:3d} net {m['net']:+8.2f} PF {m['pf']:.2f} "
              f"win {m['win_rate']:.3f} reasons {json.dumps(m['by_reason'])} skips {m['skips']}")
    print(f"global cut {cut}  per-lane cuts {per_lane_cut}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
