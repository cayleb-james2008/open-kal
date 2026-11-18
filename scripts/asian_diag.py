#!/usr/bin/env python3
"""IS-only diagnostic: AsianRTI's p against the realized settlement average.

For every valid sample of the IS split (close < cut), both sides, cost band
[0.35, 0.60], records (p_side, cost, won, secs_left, lead, mean-sd lead vs the
window's realized settlement average). Reports:
  * y-convention audit: does the tape average over [close-60, close] reproduce
    the settled `y` for every window?
  * mean-unbiasedness: distribution of (model mean - realized avg) / model sd
    at the decision time (should centre on 0 if the mean is honest).
  * calibration: mean p vs realized win rate, in p-buckets, and split at the
    60s window boundary.

Usage: .venv/bin/python scripts/asian_diag.py [--lanes KXSOL15M,KXXRP15M]
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
from scripts.asian_pin import COST_HI, COST_LO  # noqa: E402
from scripts.asian_rti import AsianRTI, WIN_S  # noqa: E402


def realized_avg(lane, close: float) -> float | None:
    ts, val = lane.index["ts"], lane.index["value"]
    lo = int(np.searchsorted(ts, close - WIN_S, "left"))
    hi = int(np.searchsorted(ts, close, "right"))
    if hi <= lo:
        return None
    return float(np.mean(val[lo:hi]))


def diag(lane, cut: float) -> dict:
    model = AsianRTI(lane.index["ts"], lane.index["value"],
                     lane.index["accum"], lane.index["k"])
    y_bad = windows = 0
    err: list[float] = []
    secs_err: list[tuple[float, float]] = []  # (secs, err)
    stal_err: list[tuple[float, float]] = []  # (staleness s, err)
    inb: list[bool] = []                      # any side in the cost band
    fvol: list[tuple[float, float]] = []      # (secs, realized_future_vol / sigma)
    drift: list[tuple[float, float, float]] = []  # (secs, per-s drift, spot)
    rows: list[tuple[float, float, int, float, float]] = []  # p, cost, won, secs, lead
    for name, m in lane.meta.items():
        close, strike = m["close"], m["strike"]
        if close >= cut or not (lane.index["ts"][0] <= close - WIN_S
                                and close < lane.index["ts"][-1]):
            continue
        windows += 1
        ra = realized_avg(lane, close)
        if ra is None or (ra >= strike) != bool(m["y"]):
            y_bad += 1
        tape = RC.tape(lane, lane.index_of[name])
        for secs in range(120, 14, -1):
            t = close - secs
            f = model.fair(t, close, strike)
            if f is None:
                continue
            try:
                b, a = tape.quote(t)
            except Exception:
                continue
            cy_in = COST_LO <= float(b) <= COST_HI
            cn_in = COST_LO <= 1.0 - float(b) <= COST_HI
            if f["sd"] > 0 and ra is not None:
                e = (f["mean"] - ra) / f["sd"]
                err.append(e)
                secs_err.append((float(secs), e))
                stal_err.append((t - f["last_rti_ts"], e))
                # realized 1s-pair rms over (t, close] vs the trailing estimate
                ts_l, val_l = lane.index["ts"], lane.index["value"]
                a0 = int(np.searchsorted(ts_l, t, "right"))
                a1 = int(np.searchsorted(ts_l, close, "right"))
                if a1 - a0 >= 5:
                    dt = np.diff(ts_l[a0:a1])
                    ok = (dt >= 1e-3) & (dt <= 1.0 + 1e-9)
                    if ok.sum() >= 5:
                        rms = float(np.sqrt((np.diff(val_l[a0:a1])[ok] ** 2
                                             / np.maximum(dt[ok], 1e-3)).mean()))
                        fvol.append((float(secs), rms / f["sigma"]))
                drift.append((float(secs), (ra - f["mean"]) * 60.0 / f["remaining_s"],
                              float(f["spot"])))
                inb.append((cy_in or cn_in))
            for side, pside, c in ((0, float(f["p"]), float(b)),  # YES taker ~ ask
                                   (1, 1.0 - float(f["p"]), 1.0 - float(b))):
                if not (COST_LO <= c <= COST_HI):
                    continue
                lead = (f["mean"] - strike) / f["sd"] if f["sd"] > 0 else 0.0
                won = int(tape.side_won("YES" if side == 0 else "NO"))
                rows.append((pside, c, won, float(secs), lead))
    a = np.asarray(rows, dtype=np.float64)
    out: dict = {"windows": windows, "y_mismatch": y_bad, "n_pairs": len(rows)}
    if len(rows):
        p, won, secs, lead = a[:, 0], a[:, 2], a[:, 3], a[:, 4]
        out["p_mean"] = round(float(p.mean()), 4)
        out["won_rate"] = round(float(won.mean()), 4)
        out["brier"] = round(float(((p - won) ** 2).mean()), 4)
        buckets = {}
        for lo in np.arange(0.3, 0.75, 0.05):
            msk = (p >= lo) & (p < lo + 0.05)
            if msk.any():
                buckets[f"{lo:.2f}"] = {"n": int(msk.sum()),
                                        "p": round(float(p[msk].mean()), 4),
                                        "won": round(float(won[msk].mean()), 4)}
        out["p_buckets"] = buckets
        for tag, msk in (("secs_lte60", secs <= 60), ("secs_gt60", secs > 60)):
            if msk.any():
                out[tag] = {"n": int(msk.sum()), "p": round(float(p[msk].mean()), 4),
                            "won": round(float(won[msk].mean()), 4)}
        # z-score of the realized settlement average under the model's own
        # predictive distribution: mean 0 and std 1 if mean and sigma are honest
        e = np.asarray(err)
        se = np.asarray(secs_err, dtype=np.float64)
        def _est(x):
            if not len(x):
                return {"n": 0}
            return {"n": len(x), "mean": round(float(x.mean()), 3),
                    "std": round(float(x.std()), 3),
                    "p50": round(float(np.percentile(x, 50)), 3)}
        out["z_realized"] = _est(e)
        out["z_by_window"] = {
            "in": _est(se[se[:, 0] <= 60.0][:, 1]),
            "out": _est(se[se[:, 0] > 60.0][:, 1]),
        }
        ib = np.asarray(inb, dtype=bool)
        out["z_inband"] = _est(e[ib])
        st = np.asarray(stal_err, dtype=np.float64)
        out["z_by_staleness"] = {
            f"le_{hi:g}s": _est(st[(st[:, 0] > lo) & (st[:, 0] <= hi)][:, 1])
            for lo, hi in ((-0.01, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, 1.5))
        }
        fv = np.asarray(fvol, dtype=np.float64)
        if len(fv):
            out["future_vol_ratio"] = {
                "n": len(fv),
                "p50": round(float(np.percentile(fv[:, 1], 50)), 3),
                "p25": round(float(np.percentile(fv[:, 1], 25)), 3),
                "p75": round(float(np.percentile(fv[:, 1], 75)), 3),
                "in_p50": round(float(np.percentile(fv[fv[:, 0] <= 60][:, 1], 50)), 3)
                if (fv[:, 0] <= 60).any() else None,
                "out_p50": round(float(np.percentile(fv[fv[:, 0] > 60][:, 1], 50)), 3)
                if (fv[:, 0] > 60).any() else None,
            }
        dr = np.asarray(drift, dtype=np.float64)
        if len(dr):
            def _drift(msk):
                if not msk.any():
                    return None
                x = dr[msk]
                return {"n": int(msk.sum()),
                        "p50": round(float(np.percentile(x[:, 1], 50)), 5),
                        "mean": round(float(x[:, 1].mean()), 5),
                        "median_spot": round(float(np.median(x[:, 2])), 2)}
            out["drift_per_s"] = {
                "all": _drift(np.ones(len(dr), dtype=bool)),
                "in60": _drift(dr[:, 0] <= 60.0),
                "out60": _drift(dr[:, 0] > 60.0),
            }
        # model lead vs realized outcome (incremental signal beyond cost)
        out["lead_stats"] = {
            "p50": round(float(np.percentile(lead, 50)), 3),
            "mean": round(float(lead.mean()), 3),
            "std": round(float(lead.std()), 3),
            "corr_lead_won": round(float(np.corrcoef(lead, won)[0, 1]), 3),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lanes", default=",".join(RC.LANES))
    args = ap.parse_args()
    names = tuple(x.strip() for x in args.lanes.split(",") if x.strip())
    lanes = RC.load_lanes(names)
    cut = RC.global_cut(lanes)
    for name, lane in lanes.items():
        entry = diag(lane, cut)
        print(f"{name}: {json.dumps(entry)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
