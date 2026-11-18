#!/usr/bin/env python3
"""Validation instrument for AsianRTI on the frozen tape (no outcomes read).

Per lane:
  * `fair-scan`  — every 1s sample at secs_left 120..15 of every in-range window,
    priced with AsianRTI.fair; refusals classified (stale_spot / sigma_pairs /
    no_venue_rows / guard) by re-deriving the model's own guards.
  * `lag-dist`   — for each in-range window, the first tape timestamp at which the
    venue reports each locked count k in 1..60, vs. the clock slot close-60+k.
    Reports p50/p90/p99/max and the fraction beyond KSEQ_TOL_S.

Usage: .venv/bin/python scripts/asian_validate.py [--lanes KXSOL15M,KXXRP15M] [--fair]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import rnd_common as RC  # noqa: E402
from scripts.asian_rti import (AsianRTI, KSEQ_TOL_S, SIG_MIN_PAIRS,  # noqa: E402
                               SPOT_TOL_S, WIN_S)


def in_range(lane) -> list[str]:
    ts = lane.index["ts"]
    return [name for name, m in lane.meta.items()
            if m["close"] - WIN_S - 1 >= ts[0] and m["close"] < ts[-1]]


def lag_dist(lane) -> dict:
    ts = lane.index["ts"]
    lags: list[float] = []
    missing = seen_windows = 0
    for name in in_range(lane):
        m = lane.meta[name]
        close = m["close"]
        lo = int(np.searchsorted(ts, close - WIN_S, "left"))
        hi = int(np.searchsorted(ts, close, "right"))
        if hi <= lo:
            continue
        acc, kk, t = (lane.index["accum"][lo:hi], lane.index["k"][lo:hi],
                      ts[lo:hi])
        lab = np.isfinite(acc) & (kk > 0)
        if not lab.any():
            continue
        kk, t = kk[lab], t[lab]
        first = {}
        for j in range(len(kk)):
            key = int(kk[j])
            if key not in first:
                first[key] = float(t[j])
        seen_windows += 1
        for k in range(1, 61):
            slot = close - WIN_S + k
            if k in first:
                lags.append(first[k] - slot)
            else:
                missing += 1
    if not lags:
        return {"windows": seen_windows, "note": "no labeled rows"}
    a = np.asarray(lags)
    return {
        "windows": seen_windows,
        "n_lags": len(a),
        "missing_k_slots": missing,
        "p50": round(float(np.percentile(a, 50)), 3),
        "p90": round(float(np.percentile(a, 90)), 3),
        "p99": round(float(np.percentile(a, 99)), 3),
        "max": round(float(a.max()), 3),
        "frac_over_0.5s": round(float((a > KSEQ_TOL_S).mean()), 4),
        "frac_over_1.0s": round(float((a > 1.0).mean()), 4),
    }


def classify_one(model: AsianRTI, lane, t: float, close: float, strike: float) -> str:
    """Refusal class by re-deriving the model's guards, in order."""
    ts = lane.index["ts"]
    j = int(np.searchsorted(ts, t, "right")) - 1
    if j < 0 or t - float(ts[j]) > SPOT_TOL_S:
        return "stale_spot"
    if model.sigma_at(t) is None:
        return "sigma_pairs"
    secs_left = close - t
    if secs_left >= WIN_S:
        return "priced"
    lo = int(np.searchsorted(ts, close - WIN_S, "left"))
    hi = int(np.searchsorted(ts, t, "right"))
    acc, kk = lane.index["accum"][lo:hi], lane.index["k"][lo:hi]
    if not (np.isfinite(acc) & (kk > 0)).any():
        return "no_venue_rows"
    if model._venue_locked(t, close) is None:
        return "guard"
    return "priced"


def fair_scan(lane) -> dict:
    model = AsianRTI(lane.index["ts"], lane.index["value"],
                     lane.index["accum"], lane.index["k"])
    skips: dict[str, int] = {}
    sd: list[float] = []
    p: list[float] = []
    n_samples = 0
    for name in in_range(lane):
        m = lane.meta[name]
        close, strike = m["close"], m["strike"]
        for secs in range(120, 14, -1):
            t = close - secs
            n_samples += 1
            cls = classify_one(model, lane, t, close, strike)
            if cls != "priced":
                skips[cls] = skips.get(cls, 0) + 1
                continue
            f = model.fair(t, close, strike)
            if f is None:                      # classification missed something
                skips["unclassified"] = skips.get("unclassified", 0) + 1
                continue
            sd.append(f["sigma"])
            p.append(f["p"])
    a_sd = np.asarray(sd)
    return {
        "windows": len(in_range(lane)),
        "samples": n_samples,
        "priced": len(p),
        "skips": dict(sorted(skips.items())),
        "sd_median": round(float(np.median(a_sd)), 5) if len(a_sd) else None,
        "p_mean": round(float(np.mean(p)), 4) if p else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lanes", default=",".join(RC.LANES))
    ap.add_argument("--fair", action="store_true",
                    help="also run the fair-value classification scan")
    args = ap.parse_args()
    names = tuple(x.strip() for x in args.lanes.split(",") if x.strip())
    lanes = RC.load_lanes(names)
    out = {}
    for name, lane in lanes.items():
        entry = {"lag_dist": lag_dist(lane)}
        if args.fair:
            entry["fair_scan"] = fair_scan(lane)
        out[name] = entry
        print(f"{name}: {json.dumps(entry)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
