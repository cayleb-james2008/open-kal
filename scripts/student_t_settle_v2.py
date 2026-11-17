#!/usr/bin/env python3
"""Track B (student_t_settle_v2): Student-t fair value, maker entry, hold to settlement. IS only by default.

Fit and fair value come from kurtosis_fv_snipe (imported, not modified). Each trigger is one
maker entry through openkal.sim_core.run_trade with no tp, stop or hold: the binary settles.

  .venv/bin/python scripts/student_t_settle_v2.py --lanes KXSOL15M     # dev, one lane, no lock
  flock data/.heavy.lock timeout 1800 nice -n 10 .venv/bin/python scripts/student_t_settle_v2.py
  scripts/student_t_settle_v2.py --score-oos                           # sealed, lead only, never run here

Volatility band: vq is the percentile of sigma(t) among the IS samples of the same lane. The
per-lane edges (101 percentiles) are stored in the IS JSON and reused by --score-oos.
A band [lo, hi) includes lo; the top band also includes 1.0.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openkal import sim_core as S  # noqa: E402
from scripts import bounded_sweep as BS  # noqa: E402
from scripts import kurtosis_fv_snipe as KF  # noqa: E402
from scripts import rnd_common as RC  # noqa: E402

OUT = ROOT / "data" / "student-t-settle-v2-is.json"
TRADES = ROOT / "data" / "student-t-settle-v2-is-trades.jsonl"
FAMILY = "student_t_settle_v2"
HURDLES = (0.03, 0.05, 0.07, 0.09)
SPREAD_MAX = (0.02, 0.04)
VOL_BANDS = ((0.0, 1.0), (0.0, 0.5), (0.5, 1.0))
PATIENCES = (10.0, 30.0)
COST_LO, COST_HI = 0.22, 0.58
COST_LO_D, COST_HI_D = Decimal("0.22"), Decimal("0.58")
COOLDOWN, PER_WINDOW = 90.0, 3
XRP = "KXXRP15M"
PCT = np.linspace(0.0, 1.0, 101)
FEATURES = ["p_side", "p_side_minus_cost", "secs", "abs_x", "spread", "vq"]


def config_name(hurdle: float, spread_max: float, vband: tuple, patience: float) -> str:
    return f"h{hurdle:.2f}_s{spread_max:.2f}_v{vband[0]:.1f}-{vband[1]:.1f}_p{patience:g}"


def grid():
    for h in HURDLES:
        for sm in SPREAD_MAX:
            for vb in VOL_BANDS:
                for p in PATIENCES:
                    yield h, sm, vb, p


def new_rec() -> dict:
    return {"rows": [], "skips": Counter(), "triggers": 0}


# ------------------------------------------------------------- candidates
def collect(lane, cut: float, params: dict, split: str) -> dict:
    """Every valid 5s sample (secs 120..780) of the tickers in one split, with maker costs."""
    before = cut if split == "is" else None    # IS never reads an index print at or after cut
    its, v = KF.clean_index(lane.index, before=before)
    tv, C, N = KF.sigma_prefix(its, v)
    table = KF.t_cdf_table(params["nu"])
    skips: Counter = Counter()
    cols: dict[str, list] = {k: [] for k in ("tid", "t", "secs", "close", "x", "p", "sig",
                                             "bid", "ask", "cym", "cnm")}
    n_tickers = 0
    for tid, name in enumerate(lane.tickers):
        m = lane.meta.get(name)
        if m is None or RC.slice_of(m["close"], cut) != split:
            continue
        n_tickers += 1
        tape = RC.tape(lane, tid)
        close, strike = m["close"], m["strike"]
        for secs in KF.SECS_GRID:
            t = close - secs
            try:
                b, a = tape.quote(t)
            except S.CrossedBook:
                skips["quote_crossed"] += 1
                continue
            except S.NoQuote:
                skips["quote_no_quote"] += 1
                continue
            spot = KF.index_last(its, v, t)
            if not math.isfinite(spot):
                skips["no_spot"] += 1
                continue
            sig = float(KF.sigma_at(t, tv, C, N))
            x = math.log(strike / spot) / (sig * math.sqrt(max(secs - KF.TAU_OFFSET, 1.0)))
            p = 1.0 - float(np.interp(x / params["s"], KF.CDF_X, table))
            vals = (tid, t, secs, close, x, p, sig, b, a,
                    float(S.entry_price(S.YES, S.MAKER, b, a)),
                    float(S.entry_price(S.NO, S.MAKER, b, a)))
            for k, val in zip(cols, vals):
                cols[k].append(val)
    pack = {k: np.asarray(val, dtype=np.float64) for k, val in cols.items()}
    pack["tid"] = pack["tid"].astype(np.int64)
    pack["spread"] = np.round(pack["ask"] - pack["bid"], 3)
    pack.update(lane=lane, split=split, tickers=n_tickers, skips=skips)
    return pack


def find_triggers(pack: dict, hurdle: float, spread_max: float, vband: tuple) -> list:
    """Deduped (index, side) maker triggers for one filter set; the larger edge wins a tie of sides."""
    lo, hi = vband
    vq = pack["vq"]
    band = (vq >= lo) & ((vq < hi) | (hi >= 1.0))
    ok_spread = band & (pack["spread"] <= spread_max)
    p, cy, cn = pack["p"], pack["cym"], pack["cnm"]
    ey = p - cy
    en = (1.0 - p) - cn
    oky = ok_spread & (cy >= COST_LO) & (cy <= COST_HI) & (ey >= hurdle)
    okn = ok_spread & (cn >= COST_LO) & (cn <= COST_HI) & (en >= hurdle)
    use_y = oky & (~okn | (ey >= en))
    keep = BS.pick({"tid": pack["tid"], "ts": pack["t"]}, oky | okn,
                   cooldown=COOLDOWN, per_window=PER_WINDOW)
    return [(i, S.YES if use_y[i] else S.NO) for i in keep]


def side_p(pack: dict, i: int, side: str) -> float:
    p = float(pack["p"][i])
    return p if side == S.YES else 1.0 - p


def trade_for(pack: dict, i: int, side: str, hurdle: float, patience: float, memo: dict):
    """One maker entry, held to settlement. Memoised: the same trigger recurs across filters."""
    tid = int(pack["tid"][i])
    t = float(pack["t"][i])
    cost_hi = min(COST_HI_D, S.px(side_p(pack, i, side) - hurdle))
    key = (tid, t, side, str(cost_hi), patience)
    if key not in memo:
        tape = RC.tape(pack["lane"], tid)
        try:
            bq, aq = tape.quote(t + S.LAG_S)
            qty = S.stake_qty(S.entry_price(side, S.MAKER, bq, aq), RC.STAKE)
        except (S.CrossedBook, S.NoQuote):
            qty = S.MAX_QTY    # run_trade reports the crossed / no-quote skip itself
        memo[key] = S.run_trade(tape, side=side, trigger_ts=t, entry_style=S.MAKER, qty=qty,
                                patience_s=patience, cost_lo=COST_LO_D, cost_hi=cost_hi)
    return memo[key]


def run_config(pack: dict, hurdle: float, spread_max: float, vband: tuple, patiences,
               out: dict, memo: dict) -> None:
    lane = pack["lane"]
    trig = find_triggers(pack, hurdle, spread_max, vband)
    for patience in patiences:
        cfg = config_name(hurdle, spread_max, vband, patience)
        rec = out[cfg]
        rec["triggers"] += len(trig)
        for i, side in trig:
            res = trade_for(pack, i, side, hurdle, patience, memo)
            if isinstance(res, S.Skip):
                rec["skips"][res.reason] += 1
                continue
            cost_s = float(pack["cym"][i] if side == S.YES else pack["cnm"][i])
            feat = [side_p(pack, i, side), side_p(pack, i, side) - cost_s,
                    float(pack["secs"][i]), abs(float(pack["x"][i])),
                    float(pack["spread"][i]), float(pack["vq"][i])]
            rec["rows"].append(RC.trade_row(
                res, lane=lane.lane, ticker=lane.tickers[int(pack["tid"][i])],
                close=float(pack["close"][i]), slice_name=pack["split"], feat=feat,
                config=cfg, family=FAMILY, hurdle=hurdle, spread_max=spread_max,
                vol_band=list(vband), patience_s=patience))


def summarize_cfg(rec: dict, hurdle, spread_max, vband, patience) -> dict:
    rows = rec["rows"]
    ex_xrp = sum((Decimal(r["pnl"]) for r in rows if r["lane"] != XRP), Decimal(0))
    return {
        "hurdle": hurdle, "spread_max": spread_max, "vol_band": list(vband),
        "patience_s": patience, "triggers": rec["triggers"],
        "fill_rate": round(len(rows) / rec["triggers"], 4) if rec["triggers"] else 0.0,
        "is": RC.summarize(rows, dict(rec["skips"])),
        "net_ex_xrp": float(ex_xrp),
    }


def champion_of(configs: dict) -> dict:
    names = list(configs)
    el = [c for c in names if KF.eligible(configs[c]["is"])]
    best = max(el or names, key=lambda c: configs[c]["is"]["net"])
    return {"family": FAMILY, "config": best, "is_eligible": bool(el), "is": configs[best]["is"]}


# ------------------------------------------------------------- modes
def run_is(lanes: dict, cut: float) -> dict:
    t0 = time.time()
    check = KF.cdf_check()
    if not check["pass"]:
        raise RuntimeError(f"Student-t CDF unit check failed: {check}")
    fits = {}
    for name, lane in lanes.items():
        its, v = KF.clean_index(lane.index, before=cut - KF.FIT_LEAD)
        fits[name] = KF.fit_lane(its, v, cut)

    out = {config_name(h, sm, vb, p): new_rec() for h, sm, vb, p in grid()}
    edges, sampling = {}, {}
    for name, lane in lanes.items():
        pack = collect(lane, cut, fits[name], "is")
        edges[name] = np.quantile(pack["sig"], PCT)
        pack["vq"] = np.interp(pack["sig"], edges[name], PCT)
        sampling[name] = {"tickers": pack["tickers"], "valid_samples": int(len(pack["t"])),
                          "skips": dict(pack["skips"]),
                          "sig_floor_share": round(float(np.mean(pack["sig"] <= KF.SIG_FLOOR)), 4)}
        memo: dict = {}
        for h in HURDLES:
            for sm in SPREAD_MAX:
                for vb in VOL_BANDS:
                    run_config(pack, h, sm, vb, PATIENCES, out, memo)

    configs = {}
    for h, sm, vb, p in grid():
        cfg = config_name(h, sm, vb, p)
        configs[cfg] = summarize_cfg(out[cfg], h, sm, vb, p)
    champ = champion_of(configs)

    champ_rows = out[champ["config"]]["rows"]
    TRADES.parent.mkdir(parents=True, exist_ok=True)
    TRADES.write_text("".join(json.dumps(r, default=RC._json_default) + "\n" for r in champ_rows))

    obj = {
        "split": "IS only: markets with close < cut; index used before cut (fit before cut - 900)",
        "cut": cut,
        "cut_source": "rnd_common.global_cut over the four lanes (cache-derived on a one-lane run)",
        "lanes": list(lanes),
        "fit": {"lanes": fits, "cdf_check": check},
        "features": FEATURES,
        "vol_band": {"edges_pct": PCT.tolist(),
                     "edges": {k: v.tolist() for k, v in edges.items()},
                     "inclusive": "vq in [lo, hi); top band also includes 1.0",
                     "edges_source": "IS samples of the same lane"},
        "sampling": sampling,
        "configs": configs,
        "champion": champ,
        "champion_trades_file": str(TRADES),
        "runtime_s": round(time.time() - t0, 1),
    }
    RC.dump(OUT, obj)
    return obj


def score_oos(lanes: dict, cut: float, is_obj: dict) -> int:
    """Sealed: score the family champion once on OOS. Never run by Track B."""
    if set(lanes) != set(RC.LANES) or set(is_obj["lanes"]) != set(RC.LANES):
        raise RuntimeError("--score-oos needs all four lanes, in the IS JSON too")
    if abs(is_obj["cut"] - cut) > 1e-9:
        raise RuntimeError("IS JSON cut does not match this cut")
    ch = is_obj["champion"]
    cfg = ch["config"]
    c = is_obj["configs"][cfg]
    vb = tuple(c["vol_band"])
    is_rows = [json.loads(ln) for ln in TRADES.read_text().splitlines() if ln.strip()]
    is_rows = [r for r in is_rows if r["config"] == cfg]

    def scorer():
        out = {cfg: new_rec()}
        for name, lane in lanes.items():
            pack = collect(lane, cut, is_obj["fit"]["lanes"][name], "oos")
            pack["vq"] = np.interp(pack["sig"], np.asarray(is_obj["vol_band"]["edges"][name]), PCT)
            run_config(pack, c["hurdle"], c["spread_max"], vb, (c["patience_s"],), out, {})
        return out[cfg]["rows"], dict(out[cfg]["skips"]), RC.fit_payoff_model(is_rows)

    entry = RC.score_oos_once(FAMILY, {"config": cfg, "hurdle": c["hurdle"],
                                       "spread_max": c["spread_max"], "vol_band": list(vb),
                                       "patience_s": c["patience_s"]}, scorer)
    print(json.dumps({"family": FAMILY, "oos": entry["oos"], "gate": entry["gate"]},
                     default=RC._json_default))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lanes", default=",".join(RC.LANES))
    ap.add_argument("--score-oos", action="store_true")
    args = ap.parse_args()
    names = tuple(x.strip() for x in args.lanes.split(",") if x.strip())
    light = KF.cut_from_cache()
    lanes = RC.load_lanes(names)
    if set(names) == set(RC.LANES):
        cut = RC.global_cut(lanes)
        if abs(cut - light) > 1e-9:
            raise RuntimeError(f"cut mismatch: global {cut} vs cache {light}")
    else:
        cut = light
    if args.score_oos:
        return score_oos(lanes, cut, json.loads(OUT.read_text()))
    obj = run_is(lanes, cut)
    c = obj["champion"]
    print(json.dumps({"out": str(OUT), "cut": cut, "runtime_s": obj["runtime_s"],
                      "configs": len(obj["configs"]),
                      "champion": {"config": c["config"], "eligible": c["is_eligible"],
                                   "trades": c["is"]["trades"], "net": c["is"]["net"],
                                   "pf": c["is"]["pf"], "win_rate": c["is"]["win_rate"]}},
                     default=RC._json_default, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
