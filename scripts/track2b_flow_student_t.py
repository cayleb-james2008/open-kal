#!/usr/bin/env python3
"""Charter 3 Track 2B - `flow_student_t_settle`: Aggressor Flow + Student-t Settlement Pin.

Mechanism:
  - IS-fitted Student-t (nu) settlement fair value ($0.35-$0.60 cost band).
  - Multi-window signed aggressor print flow (trade_ty / trade_sz imbalance at ts <= trigger_ts).
  - Enter at T+1s (taker or maker with 5s patience) and hold to $1.00 settlement ($0 exit fee).

Evaluation:
  - Scored exclusively through openkal/sim_core.py.
  - Selection strictly on chronological 60% IS (>= 30 IS trades, IS PF >= 1.25, >= 2 profitable lanes).
  - Sealed OOS scored at most ONCE logged to data/oos-ledger.jsonl.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
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

OUT = ROOT / "data" / "track2b-flow-student-t-is.json"
TRADES = ROOT / "data" / "track2b-flow-student-t-is-trades.jsonl"

FAMILY = "flow_student_t_settle"
HURDLES = (0.03, 0.05, 0.07, 0.09)
FLOW_WINDOWS = ((5.0, 15.0), (5.0, 30.0), (10.0, 30.0))
FLOW_THRESHOLDS = (0.0, 0.15, 0.30)
STYLES = ("taker", "maker")

COST_LO, COST_HI = 0.35, 0.60
COST_LO_D, COST_HI_D = Decimal("0.35"), Decimal("0.60")
COOLDOWN, PER_WINDOW = 60.0, 2
MIN_TRADES, MIN_PF, MIN_LANES = 30, 1.25, 2
FEATURES = ["p_side", "edge", "secs_left", "flow_fast", "flow_slow"]


def config_name(hurdle: float, fw: tuple[float, float], fth: float, style: str) -> str:
    return f"h{hurdle:.2f}_w{int(fw[0])}_{int(fw[1])}_th{fth:.2f}_{style}"


def parse_config(name: str) -> dict:
    parts = name.split("_")
    h = float(parts[0][1:])
    w0 = float(parts[1][1:])
    w1 = float(parts[2])
    fth = float(parts[3][2:])
    style = parts[4]
    return {"hurdle": h, "flow_windows": (w0, w1), "flow_thresh": fth, "style": style}


def fit_all_lanes(lanes: dict, cut: float) -> dict:
    """Fit IS parameters for Student-t on all four lanes."""
    fit = {}
    for name in RC.LANES:
        its, v = KF.clean_index(lanes[name].index, before=cut)
        res = KF.fit_lane(its, v, cut)
        fit[name] = res
    return fit


def collect_lane_samples(lane, fit_params: dict, cut: float, split: str) -> dict:
    """Pre-collect quotes, Student-t fair values, and cumulative trade volumes for one lane."""
    before = cut if split == "is" else None
    its, v = KF.clean_index(lane.index, before=before)
    tv, C, N = KF.sigma_prefix(its, v)

    nu = fit_params["nu"]
    s_fat = fit_params["s"]
    table = KF.t_cdf_table(nu)

    skips: Counter = Counter()
    tid_l, t_l, secs_l, close_l = [], [], [], []
    cy_taker_l, cn_taker_l = [], []
    cy_maker_l, cn_maker_l = [], []
    p_yes_l, p_no_l = [], []
    f_fast_l, f_slow_l = {}, {}
    for fw in FLOW_WINDOWS:
        f_fast_l[fw] = []
        f_slow_l[fw] = []

    n_tickers = 0
    secs_grid = np.arange(780.0, 59.0, -2.0)

    for tid, name in enumerate(lane.tickers):
        m = lane.meta.get(name)
        if m is None or RC.slice_of(m["close"], cut) != split:
            continue
        n_tickers += 1
        tape = RC.tape(lane, tid)
        close, strike = m["close"], m["strike"]

        lo, hi = int(lane.trade_bounds[tid]), int(lane.trade_bounds[tid + 1])
        tts = lane.trade_ts[lo:hi]
        sz = lane.trade_sz[lo:hi]
        signed = np.where(lane.trade_ty[lo:hi] == 1, sz, -sz)
        cs = np.concatenate(([0.0], np.cumsum(signed)))
        ct = np.concatenate(([0.0], np.cumsum(sz)))
        n_trades = len(tts)

        for secs in secs_grid:
            t = close - secs

            spot = KF.index_last(its, v, t)
            if not math.isfinite(spot) or spot <= 0:
                skips["no_spot"] += 1
                continue

            try:
                b, a = tape.quote(t)
            except S.CrossedBook:
                skips["quote_crossed"] += 1
                continue
            except S.NoQuote:
                skips["quote_no_quote"] += 1
                continue

            sig = float(KF.sigma_at(t, tv, C, N))
            tau_eff = max(secs - 40.0, 1.0)
            vol = sig * math.sqrt(tau_eff)
            if vol <= 0:
                skips["zero_vol"] += 1
                continue

            x = math.log(strike / spot) / vol
            p_yes = 1.0 - float(np.interp(x / s_fat, KF.CDF_X, table))
            p_no = 1.0 - p_yes

            cy_t = float(S.entry_price(S.YES, S.TAKER, b, a))
            cn_t = float(S.entry_price(S.NO, S.TAKER, b, a))
            cy_m = float(S.entry_price(S.YES, S.MAKER, b, a))
            cn_m = float(S.entry_price(S.NO, S.MAKER, b, a))

            tid_l.append(tid)
            t_l.append(t)
            secs_l.append(secs)
            close_l.append(close)
            cy_taker_l.append(cy_t)
            cn_taker_l.append(cn_t)
            cy_maker_l.append(cy_m)
            cn_maker_l.append(cn_m)
            p_yes_l.append(p_yes)
            p_no_l.append(p_no)

            for fw in FLOW_WINDOWS:
                w_fast, w_slow = fw
                if n_trades > 0:
                    af = int(np.searchsorted(tts, t - w_fast, "left"))
                    bf = int(np.searchsorted(tts, t, "right"))
                    denf = ct[bf] - ct[af]
                    ff = float((cs[bf] - cs[af]) / denf) if denf > 0 else 0.0

                    asl = int(np.searchsorted(tts, t - w_slow, "left"))
                    bsl = int(np.searchsorted(tts, t, "right"))
                    dens = ct[bsl] - ct[asl]
                    fs = float((cs[bsl] - cs[asl]) / dens) if dens > 0 else 0.0
                else:
                    ff, fs = 0.0, 0.0
                f_fast_l[fw].append(ff)
                f_slow_l[fw].append(fs)

    return {
        "lane": lane,
        "split": split,
        "tickers": n_tickers,
        "tid": np.asarray(tid_l, dtype=np.int64),
        "t": np.asarray(t_l),
        "secs": np.asarray(secs_l),
        "close": np.asarray(close_l),
        "cost_yes_taker": np.asarray(cy_taker_l),
        "cost_no_taker": np.asarray(cn_taker_l),
        "cost_yes_maker": np.asarray(cy_maker_l),
        "cost_no_maker": np.asarray(cn_maker_l),
        "p_yes": np.asarray(p_yes_l),
        "p_no": np.asarray(p_no_l),
        "flow_fast": {fw: np.asarray(f_fast_l[fw]) for fw in FLOW_WINDOWS},
        "flow_slow": {fw: np.asarray(f_slow_l[fw]) for fw in FLOW_WINDOWS},
        "skips": skips,
    }


def find_triggers(pack: dict, hurdle: float, fw: tuple[float, float],
                  fth: float, style: str) -> list[tuple[int, str, float, float, float, float]]:
    """Identify candidate triggers where edge >= hurdle and signed flow matches the trade direction."""
    p_yes = pack["p_yes"]
    p_no = pack["p_no"]
    ff = pack["flow_fast"][fw]
    fs = pack["flow_slow"][fw]

    if style == "taker":
        cy = pack["cost_yes_taker"]
        cn = pack["cost_no_taker"]
    else:
        cy = pack["cost_yes_maker"]
        cn = pack["cost_no_maker"]

    ey = p_yes - cy
    en = p_no - cn

    # YES condition: cost in [0.35, 0.60], edge >= hurdle, fast & slow flow >= fth
    oky = (cy >= COST_LO) & (cy <= COST_HI) & (ey >= hurdle) & (ff >= fth) & (fs >= fth)
    # NO condition: cost in [0.35, 0.60], edge >= hurdle, fast & slow flow <= -fth
    okn = (cn >= COST_LO) & (cn <= COST_HI) & (en >= hurdle) & (ff <= -fth) & (fs <= -fth)

    mask = oky | okn
    if not np.any(mask):
        return []

    use_y = oky & (~okn | (ey >= en))
    keep = BS.pick({"tid": pack["tid"], "ts": pack["t"]}, mask,
                   cooldown=COOLDOWN, per_window=PER_WINDOW)

    out = []
    for i in keep:
        side = S.YES if use_y[i] else S.NO
        p_side = float(p_yes[i]) if side == S.YES else float(p_no[i])
        cost = float(cy[i]) if side == S.YES else float(cn[i])
        edge = p_side - cost
        out.append((i, side, p_side, edge, float(ff[i]), float(fs[i])))
    return out


def run_triggers(pack: dict, triggers: list[tuple[int, str, float, float, float, float]],
                 cfg_name: str, style: str, hurdle: float, out_dict: dict) -> None:
    lane = pack["lane"]
    entry_style = S.TAKER if style == "taker" else S.MAKER
    patience = 5.0 if style == "maker" else 0.0

    for i, side, p_side, edge, ff, fs in triggers:
        tid = int(pack["tid"][i])
        t = float(pack["t"][i])
        name = lane.tickers[tid]
        feat = [p_side, edge, float(pack["secs"][i]), ff, fs]
        tape = RC.tape(lane, tid)

        try:
            bq, aq = tape.quote(t + S.LAG_S)
            cost_arr = S.entry_price(side, entry_style, bq, aq)
            qty = S.stake_qty(cost_arr, RC.STAKE)
        except (S.CrossedBook, S.NoQuote):
            qty = S.MAX_QTY

        res = S.run_trade(tape, side=side, trigger_ts=t, entry_style=entry_style, qty=qty,
                          patience_s=patience, cost_lo=COST_LO_D, cost_hi=COST_HI_D)

        if isinstance(res, S.Skip):
            out_dict[cfg_name]["skips"][res.reason] += 1
            continue

        out_dict[cfg_name]["rows"].append(RC.trade_row(
            res, lane=lane.lane, ticker=name, close=float(pack["close"][i]),
            slice_name=pack["split"], feat=feat, config=cfg_name, family=FAMILY,
            hurdle=hurdle, p_side=p_side, edge=round(edge, 6)))


def eligible(s: dict) -> bool:
    return (s["trades"] >= MIN_TRADES and s["pf"] >= MIN_PF
            and s["profitable_lanes"] >= MIN_LANES and s["crossed_fills"] == 0)


def pick_champion(configs: dict) -> dict:
    el = [c for c in configs if eligible(configs[c]["is"])]
    if el:
        best = max(el, key=lambda c: configs[c]["is"]["net"])
        return {"config": best, "is_eligible": True, "is": configs[best]["is"]}
    best = max(configs, key=lambda c: configs[c]["is"]["net"])
    return {"config": best, "is_eligible": False, "is": configs[best]["is"]}


def run_is(lanes: dict, cut: float) -> dict:
    t0 = time.time()
    fit_params = fit_all_lanes(lanes, cut)

    packs = [collect_lane_samples(lane, fit_params[lane.lane], cut, "is")
             for lane in lanes.values()]

    all_combos = list(itertools.product(HURDLES, FLOW_WINDOWS, FLOW_THRESHOLDS, STYLES))
    cfg_names = [config_name(h, fw, fth, st) for h, fw, fth, st in all_combos]
    out_dict = {cn: {"rows": [], "skips": Counter()} for cn in cfg_names}

    for pack in packs:
        for h, fw, fth, st in all_combos:
            cn = config_name(h, fw, fth, st)
            trig = find_triggers(pack, h, fw, fth, st)
            run_triggers(pack, trig, cn, st, h, out_dict)

    configs = {}
    for h, fw, fth, st in all_combos:
        cn = config_name(h, fw, fth, st)
        configs[cn] = {
            "family": FAMILY,
            "hurdle": h,
            "flow_windows": list(fw),
            "flow_thresh": fth,
            "style": st,
            "is": RC.summarize(out_dict[cn]["rows"], dict(out_dict[cn]["skips"])),
        }

    champion = pick_champion(configs)
    champ_rows = out_dict[champion["config"]]["rows"]
    TRADES.parent.mkdir(parents=True, exist_ok=True)
    TRADES.write_text("".join(json.dumps(r, default=RC._json_default) + "\n" for r in champ_rows))

    samples = {pack["lane"].lane: {"tickers": pack["tickers"],
                                   "valid_samples": int(len(pack["t"])),
                                   "skips": dict(pack["skips"])} for pack in packs}

    obj = {
        "family": FAMILY,
        "split": "IS only: markets with close < cut",
        "cut": cut,
        "lanes": list(lanes),
        "grid": {
            "hurdles": list(HURDLES),
            "flow_windows": [list(fw) for fw in FLOW_WINDOWS],
            "flow_thresholds": list(FLOW_THRESHOLDS),
            "styles": list(STYLES),
            "cost_band": [COST_LO, COST_HI],
        },
        "fit_params": fit_params,
        "sampling": samples,
        "configs": configs,
        "champion": champion,
        "champion_trades_file": str(TRADES),
        "runtime_s": round(time.time() - t0, 1),
    }
    RC.dump(OUT, obj)
    return obj


def score_oos(lanes: dict, cut: float, is_obj: dict) -> int:
    """Sealed: score the frozen champion once on OOS, logged to data/oos-ledger.jsonl."""
    if not is_obj["champion"]["is_eligible"]:
        print(f"Refusing OOS scoring: no IS configuration met the promotion eligibility bar ({MIN_TRADES}+ trades, PF >= {MIN_PF}, {MIN_LANES}+ profitable lanes). OOS look preserved unspent.")
        return 1

    cfg_name = is_obj["champion"]["config"]
    p = parse_config(cfg_name)
    h = p["hurdle"]
    fw = p["flow_windows"]
    fth = p["flow_thresh"]
    st = p["style"]
    is_rows = [json.loads(ln) for ln in TRADES.read_text().splitlines() if ln.strip()]
    fit_params = is_obj["fit_params"]

    def scorer():
        out_dict = {cfg_name: {"rows": [], "skips": Counter()}}
        for lane in lanes.values():
            pack = collect_lane_samples(lane, fit_params[lane.lane], cut, "oos")
            trig = find_triggers(pack, h, fw, fth, st)
            run_triggers(pack, trig, cfg_name, st, h, out_dict)
        return out_dict[cfg_name]["rows"], dict(out_dict[cfg_name]["skips"]), RC.fit_payoff_model(is_rows)

    entry = RC.score_oos_once(FAMILY, {"config": cfg_name, **p, "exit": "settle"}, scorer)
    print(json.dumps({"family": FAMILY, "oos": entry["oos"], "gate": entry["gate"]},
                     default=RC._json_default))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-oos", action="store_true")
    args = ap.parse_args()
    lanes = RC.load_lanes()
    cut = RC.global_cut(lanes)
    if args.score_oos:
        return score_oos(lanes, cut, json.loads(OUT.read_text()))
    obj = run_is(lanes, cut)
    print(json.dumps({"out": str(OUT), "cut": cut, "runtime_s": obj["runtime_s"],
                      "champion": {"config": obj["champion"]["config"],
                                   "eligible": obj["champion"]["is_eligible"],
                                   "trades": obj["champion"]["is"]["trades"],
                                   "net": obj["champion"]["is"]["net"],
                                   "pf": obj["champion"]["is"]["pf"],
                                   "win_rate": obj["champion"]["is"]["win_rate"],
                                   "profitable_lanes": obj["champion"]["is"]["profitable_lanes"]}},
                     default=RC._json_default, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
