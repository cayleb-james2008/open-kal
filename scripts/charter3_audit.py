#!/usr/bin/env python3
"""Charter 3 pre-flight: data coverage and the settlement law, before any strategy.

Measures, on the frozen cache only:

  1. per lane: settled windows, IS/OOS split at the global cut, book/print rows;
  2. the index tape: value rows, accumulator rows (finite accum and k>0), and how
     dense the accumulator is inside each window's final 60s;
  3. the settlement law: share of windows where mean(RTI over the last 60s) >= the
     published floor strike equals the recorded outcome, per lane and overall;
  4. a smoke of AsianRTI.fair: unpriced reasons and p values at secs_left 120/60/30
     on IS windows only.

    .venv/bin/python scripts/charter3_audit.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import rnd_common as RC  # noqa: E402
from scripts.asian_rti import AsianRTI  # noqa: E402
from openkal import sim_core as S  # noqa: E402


def main() -> int:
    lanes = RC.load_lanes()
    cut = RC.global_cut(lanes)
    print(f"global cut: {cut:.0f}")
    total_agree = total_windows = 0
    for name, lane in lanes.items():
        idx = lane.index
        ts, val = idx["ts"], idx["value"]
        accum, k = idx["accum"], idx["k"]
        good = np.isfinite(accum) & np.isfinite(k) & (k > 0)
        n_tickers = len(lane.meta)
        is_n = sum(1 for m in lane.meta.values() if m["close"] < cut)
        oos_n = n_tickers - is_n
        print(f"\n== {name}: {n_tickers} windows (IS {is_n} / OOS {oos_n}), "
              f"book rows {len(lane.ts):,}, prints {len(lane.trade_ts):,}")
        print(f"   index: {len(ts):,} value rows {ts.min():.0f}..{ts.max():.0f}; "
              f"accumulator rows {int(good.sum()):,}")
        # accumulator density in each window's final minute
        gaps, have = [], 0
        for ticker, m in lane.meta.items():
            close = m["close"]
            w = (ts >= close - 60) & (ts < close)
            g = good & w
            n = int(g.sum())
            gaps.append(n)
            have += int(n >= 15)
        gaps = np.array(gaps)
        print(f"   accum rows in final 60s: median {np.median(gaps):.0f}, "
              f"p10 {np.percentile(gaps, 10):.0f}, zero {(gaps == 0).sum()}/{len(gaps)}, "
              f">=15 rows in {have}/{len(gaps)}")
        # settlement law: mean(RTI[-60s:]) >= strike <=> y
        agree = 0
        n_w = 0
        worst = []
        for ticker, m in lane.meta.items():
            close, strike, y = m["close"], m["strike"], m["y"]
            w = (ts >= close - 60) & (ts < close)
            if w.sum() < 50:
                continue
            a_close = float(val[w].mean())
            pred = 1 if a_close >= strike else 0
            agree += int(pred == y)
            n_w += 1
            worst.append((abs(a_close - strike), ticker, a_close, strike, y))
        if n_w:
            worst.sort(reverse=True)
            print(f"   settlement law mean(RTI[-60s:])>=strike == y: {agree}/{n_w} "
                  f"({agree / n_w:.4f}); closest misses: " +
                  "; ".join(f"{t} avg={a:.2f} K={k2:.2f} y={yy}" for _, t, a, k2, yy in worst[:3]))
            total_agree += agree
            total_windows += n_w
    print(f"\nALL LANES settlement-law agreement: {total_agree}/{total_windows} "
          f"({total_agree / max(1, total_windows):.4f})")

    # AsianRTI smoke on IS windows of the first two lanes
    print("\n== AsianRTI smoke (IS windows)")
    for name in ("KXBTC15M", "KXSOL15M"):
        lane = lanes[name]
        model = AsianRTI(lane.index["ts"], lane.index["value"], lane.index["accum"],
                         lane.index["k"])
        reasons: Counter[str] = Counter()
        priced = 0
        seen = 0
        for ticker, m in sorted(lane.meta.items(), key=lambda kv: kv[1]["close"]):
            if m["close"] >= cut:
                continue
            tape = RC.tape(lane, lane.index_of[ticker])
            for secs in (120.0, 60.0, 30.0, 15.0):
                seen += 1
                t = m["close"] - secs
                f = model.fair(t, m["close"], m["strike"])
                if f is None:
                    # attribute the reason
                    j = int(np.searchsorted(model.ts, t, "right")) - 1
                    spot_ok = j >= 0 and t - float(model.ts[j]) <= 1.5
                    reasons["no_spot" if not spot_ok else "locked_or_sigma"] += 1
                    continue
                priced += 1
                if seen <= 8 or (ticker.endswith(tuple("13579")) and secs == 30.0 and priced <= 6):
                    try:
                        b, a = tape.quote(t)
                        q = f"book {b:.3f}/{a:.3f}"
                    except (S.CrossedBook, S.NoQuote):
                        q = "no book"
                    print(f"   {ticker} secs={secs:5.1f} p={f['p']:.3f} spot={f['spot']:.2f} "
                          f"sd={f['sd']:.3f} locked_k={f['locked_s']:.0f} {q}")
        print(f"   {name}: priced {priced}/{seen}; unpriced reasons {reasons}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
