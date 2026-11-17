#!/usr/bin/env python3
"""Prove what the settlement index actually is, before any strategy uses it.

Every lock-in strategy rests on one claim. The 15-minute crypto contracts resolve
YES when the simple average of CF Benchmarks' RTI over the last minute of the window
is at least the simple average over the *first* minute of the same window (the
published `floor_strike`). Nothing here trades. It measures the claim per lane
against the frozen tape and reports:

  * whether the result matches the two 60-second averages (exact and rounded),
  * whether Kalshi's `last_60s_windowed_average_15min` field is the running average
    of the locked part of the settlement window, so a partial sum can be trusted,
  * how `window_size` (k) tracks the seconds elapsed since the window opened,
  * the honest law for the settlement average's remaining uncertainty: with T
    seconds left, the final average moves by (T/60) * (mean of the last T seconds),
    and that mean has standard deviation sigma * sqrt(T/3), so the conditional sd of
    the settlement average scales like T^1.5 -- not like the T^2 the shipped model
    uses, which is several times too tight and therefore overconfident.

    .venv/bin/python scripts/verify_settlement.py
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "evolve-cache"
LANES = {"KXBTC15M": "BRTI", "KXETH15M": "ETHUSD_RTI", "KXSOL15M": "SOLUSD_RTI",
         "KXXRP15M": "XRPUSD_RTI", "KXDOGE15M": "DOGEUSD_RTI"}


def iso(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def main() -> int:
    markets = json.loads((CACHE / "markets.json").read_text())
    settled = {m["ticker"]: m for m in
               json.loads((ROOT / "data" / "settled-markets.json").read_text())}
    report: dict = {}
    spread_law: dict[int, list[float]] = defaultdict(list)
    sigma_by_lane: dict[str, float] = {}

    for lane, index_id in LANES.items():
        path = CACHE / f"index_{index_id}.npz"
        if not path.exists():
            continue
        d = np.load(path)
        ts, val, acc, k = d["ts"], d["value"], d["accum"], d["k"]
        n = agree_exact = agree_floor = agree_field = 0
        errs: list[float] = []
        partial_ok = partial_n = 0
        k_lag: list[float] = []
        for ticker, m in markets.items():
            if not ticker.startswith(lane + "-"):
                continue
            rec = settled.get(ticker)
            if not rec:
                continue
            close = float(m["close"])
            strike = float(m["strike"])
            y = int(m["y"])
            open_ts = iso(rec["open_time"])
            if close - 60 < ts[0] or open_ts - 60 < ts[0] or close > ts[-1]:
                continue
            w_open = (ts >= open_ts - 60) & (ts < open_ts)
            w_close = (ts >= close - 60) & (ts < close)
            if w_open.sum() < 50 or w_close.sum() < 50:
                continue
            a_open = float(val[w_open].mean())
            a_close = float(val[w_close].mean())
            n += 1
            agree_exact += int((a_close >= a_open) == (y == 1))
            agree_floor += int((a_close >= strike) == (y == 1))
            errs.append(a_open - strike)
            # Kalshi's own field at the close is the settlement value.
            sel = (ts >= close - 0.5) & (ts <= close + 0.5)
            if sel.any():
                i = int(sel.nonzero()[0][-1])
                if math.isfinite(float(acc[i])) and int(k[i]) > 0:
                    agree_field += int((float(acc[i]) >= strike) == (y == 1))
                    k_lag.append(float(k[i]) - (close - float(ts[i])))
            # Is Kalshi's field the running average of the locked part?
            win = (ts >= close - 60) & (ts <= close)
            widx = win.nonzero()[0]
            if len(widx) > 50:
                i = int(len(widx) * 0.6)
                locked_mean = float(val[widx[:i]].mean())
                partial_n += 1
                partial_ok += int(abs(locked_mean - float(acc[widx[i]]))
                                  <= max(0.5, 1e-4 * strike))
            # The conditional law: with T seconds left, how far can the settlement
            # average still move, given the locked average?
            for T in (5, 10, 15, 20, 30, 40, 50):
                cut = close - T
                w_locked = (ts >= close - 60) & (ts < cut)
                w_rest = (ts >= cut) & (ts < close)
                if w_locked.sum() < 50 or w_rest.sum() < 50:
                    continue
                a_locked = float(val[w_locked].mean())
                spread_law[T].append((a_close - a_locked) / float(val[w_locked][-1]))
        sigma_by_lane[lane] = statistics.pstdev(
            [e for e in errs]) if errs else 0.0
        report[lane] = {
            "windows": n,
            "agree_exact_open_avg": round(agree_exact / max(1, n), 4),
            "agree_floor_strike": round(agree_floor / max(1, n), 4),
            "agree_kalshi_field": round(agree_field / max(1, n), 4),
            "open_avg_minus_floor_strike_median": round(statistics.median(errs), 3)
            if errs else None,
            "kalshi_field_is_locked_partial_avg": f"{partial_ok}/{partial_n}",
            "k_lag_median_s": round(statistics.median(k_lag), 3) if k_lag else None,
        }
        r = report[lane]
        print(f"{lane:12s} n={r['windows']:4d} exact_open={r['agree_exact_open_avg']:.4f} "
              f"floor={r['agree_floor_strike']:.4f} kalshi_field={r['agree_kalshi_field']:.4f} "
              f"field_partial={r['kalshi_field_is_locked_partial_avg']} "
              f"k_lag={r['k_lag_median_s']}s")

    print("\nconditional sd of the settlement average, given the locked part "
          "(relative to spot):")
    print(f"{'T (s left)':>11s} {'n':>6s} {'sd/sp':>9s} {'sd/T^1.5':>10s} "
          f"{'sd/T^2':>9s} {'shipped/T^1.5':>15s}")
    for T in sorted(spread_law):
        arr = np.array(spread_law[T])
        sd = float(arr.std(ddof=1))
        print(f"{T:11d} {len(arr):6d} {sd:9.3e} {sd/T**1.5:10.3e} "
              f"{sd/T**2:9.3e} {(sd/T**1.5)/(T**0.5/60):15.3e}")
    print("\nsd/T^1.5 should be flat across T if the T^1.5 law is right; "
          "sd/T^2 is what the shipped model assumes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
