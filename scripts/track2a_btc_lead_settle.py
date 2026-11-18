#!/usr/bin/env python3
"""Charter 3 Track 2A - `btc_lead_settle`: Cross-Lane BTC Lead-Lag Settlement Pin.

Mechanism:
  - Detect causal BTC index impulses (>= 10, 15, 20 bps over 3-5s, ts <= trigger_ts).
  - Shift ETH/SOL/XRP Student-t / Brownian settlement fair value by >= 0.08.
  - While the alt's Kalshi entry_ask ($0.35-$0.60) lags behind fair value by hurdle in [0.07, 0.10, 0.13].
  - Enter at T+1s and hold to $1.00 settlement ($0 exit fee, zero T+1s exit-placement lag).

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

OUT = ROOT / "data" / "track2a-btc-lead-settle-is.json"
TRADES = ROOT / "data" / "track2a-btc-lead-settle-is-trades.jsonl"

FAMILY = "btc_lead_settle"
ALT_LANES = ("KXETH15M", "KXSOL15M", "KXXRP15M")
LOOKBACKS = (3.0, 5.0)
MIN_BPS_LIST = (10.0, 15.0, 20.0)
HURDLES = (0.07, 0.10, 0.13)
MODELS = ("student_t", "brownian")
SHIFT_MODES = ("projected", "realized")

COST_LO, COST_HI = 0.35, 0.60
COST_LO_D, COST_HI_D = Decimal("0.35"), Decimal("0.60")
COOLDOWN, PER_WINDOW = 60.0, 2
MIN_TRADES, MIN_PF, MIN_LANES = 30, 1.25, 2
FEATURES = ["p_side", "edge", "secs_left", "btc_bps", "fair_shift"]


def config_name(lb: float, min_bps: float, model: str, shift_mode: str, hurdle: float) -> str:
    m_short = "t" if model == "student_t" else "b"
    s_short = "proj" if shift_mode == "projected" else "real"
    return f"lb{lb:.0f}_b{min_bps:.0f}_{m_short}_{s_short}_h{hurdle:.2f}"


def parse_config(name: str) -> dict:
    parts = name.split("_")
    lb = float(parts[0][2:])
    min_bps = float(parts[1][1:])
    model = "student_t" if parts[2] == "t" else "brownian"
    shift_mode = "projected" if parts[3] == "proj" else "realized"
    hurdle = float(parts[4][1:])
    return {"lookback": lb, "min_bps": min_bps, "model": model,
            "shift_mode": shift_mode, "hurdle": hurdle}


def fit_alt_lanes(lanes: dict, cut: float) -> dict:
    """Fit IS parameters for Student-t and Gaussian on each lane."""
    fit = {}
    for name in ALT_LANES:
        its, v = KF.clean_index(lanes[name].index, before=cut)
        res = KF.fit_lane(its, v, cut)
        fit[name] = res
    return fit


def collect_alt_samples(lane, btc_ts: np.ndarray, btc_val: np.ndarray,
                        fit_params: dict, cut: float, split: str) -> dict:
    """Collect 1s/2s candidate timestamps for the alt lane, evaluating BTC impulses and fair values."""
    before = cut if split == "is" else None
    its, v = KF.clean_index(lane.index, before=before)
    tv, C, N = KF.sigma_prefix(its, v)

    nu = fit_params["nu"]
    s_fat = fit_params["s"]
    s_g = fit_params["s_g"]
    table = KF.t_cdf_table(nu)

    skips: Counter = Counter()
    tid_l, t_l, secs_l, close_l = [], [], [], []
    cy_taker_l, cn_taker_l = [], []
    spot_now_l, spot_prev_l = {lb: [] for lb in LOOKBACKS}, {lb: [] for lb in LOOKBACKS}
    btc_bps_l = {lb: [] for lb in LOOKBACKS}
    sig_l, tau_l = [], []

    # p_t: student_t, p_b: brownian
    p_now_t_l, p_now_b_l = [], []
    p_prev_t_l = {lb: [] for lb in LOOKBACKS}
    p_prev_b_l = {lb: [] for lb in LOOKBACKS}
    p_proj_t_l = {lb: [] for lb in LOOKBACKS}
    p_proj_b_l = {lb: [] for lb in LOOKBACKS}

    n_tickers = 0
    # Step by 1s inside [60, 780]
    secs_grid = np.arange(780.0, 59.0, -1.0)

    for tid, name in enumerate(lane.tickers):
        m = lane.meta.get(name)
        if m is None or RC.slice_of(m["close"], cut) != split:
            continue
        n_tickers += 1
        tape = RC.tape(lane, tid)
        close, strike = m["close"], m["strike"]

        for secs in secs_grid:
            t = close - secs

            # Alt causal spot
            spot = KF.index_last(its, v, t)
            if not math.isfinite(spot) or spot <= 0:
                skips["no_alt_spot"] += 1
                continue

            try:
                b, a = tape.quote(t)
            except S.CrossedBook:
                skips["quote_crossed"] += 1
                continue
            except S.NoQuote:
                skips["quote_no_quote"] += 1
                continue

            cy_t = float(S.entry_price(S.YES, S.TAKER, b, a))
            cn_t = float(S.entry_price(S.NO, S.TAKER, b, a))

            sig = float(KF.sigma_at(t, tv, C, N))
            tau_eff = max(secs - 40.0, 1.0)
            vol = sig * math.sqrt(tau_eff)
            if vol <= 0:
                skips["zero_vol"] += 1
                continue

            # Fair value at current spot
            x_now = math.log(strike / spot) / vol
            p_now_t = 1.0 - float(np.interp(x_now / s_fat, KF.CDF_X, table))
            p_now_b = 1.0 - 0.5 * (1.0 + math.erf(x_now / s_g / math.sqrt(2.0)))

            # Evaluate lookbacks for BTC impulse
            lb_ok = False
            curr_btc_bps = {}
            curr_spot_prev = {}
            curr_p_prev_t, curr_p_prev_b = {}, {}
            curr_p_proj_t, curr_p_proj_b = {}, {}

            for lb in LOOKBACKS:
                # BTC causal prints
                jb1 = int(np.searchsorted(btc_ts, t, "right")) - 1
                jb0 = int(np.searchsorted(btc_ts, t - lb, "right")) - 1
                if jb1 < 0 or jb0 < 0:
                    continue
                if t - btc_ts[jb1] > 1.5 or (t - lb) - btc_ts[jb0] > 1.5:
                    continue
                b1 = btc_val[jb1]
                b0 = btc_val[jb0]
                if b0 <= 0 or b1 <= 0:
                    continue
                bps = (b1 / b0 - 1.0) * 10000.0
                curr_btc_bps[lb] = bps

                # Alt spot lb seconds ago
                sp_prev = KF.index_last(its, v, t - lb)
                curr_spot_prev[lb] = sp_prev
                if math.isfinite(sp_prev) and sp_prev > 0:
                    x_prev = math.log(strike / sp_prev) / vol
                    curr_p_prev_t[lb] = 1.0 - float(np.interp(x_prev / s_fat, KF.CDF_X, table))
                    curr_p_prev_b[lb] = 1.0 - 0.5 * (1.0 + math.erf(x_prev / s_g / math.sqrt(2.0)))
                else:
                    curr_p_prev_t[lb] = p_now_t
                    curr_p_prev_b[lb] = p_now_b

                # Alt projected spot
                sp_proj = spot * (1.0 + bps / 10000.0)
                if sp_proj > 0:
                    x_proj = math.log(strike / sp_proj) / vol
                    curr_p_proj_t[lb] = 1.0 - float(np.interp(x_proj / s_fat, KF.CDF_X, table))
                    curr_p_proj_b[lb] = 1.0 - 0.5 * (1.0 + math.erf(x_proj / s_g / math.sqrt(2.0)))
                else:
                    curr_p_proj_t[lb] = p_now_t
                    curr_p_proj_b[lb] = p_now_b

                lb_ok = True

            if not lb_ok:
                skips["no_btc_pair"] += 1
                continue

            tid_l.append(tid)
            t_l.append(t)
            secs_l.append(secs)
            close_l.append(close)
            cy_taker_l.append(cy_t)
            cn_taker_l.append(cn_t)
            p_now_t_l.append(p_now_t)
            p_now_b_l.append(p_now_b)
            sig_l.append(sig)
            tau_l.append(tau_eff)

            for lb in LOOKBACKS:
                btc_bps_l[lb].append(curr_btc_bps.get(lb, 0.0))
                spot_prev_l[lb].append(curr_spot_prev.get(lb, spot))
                p_prev_t_l[lb].append(curr_p_prev_t.get(lb, p_now_t))
                p_prev_b_l[lb].append(curr_p_prev_b.get(lb, p_now_b))
                p_proj_t_l[lb].append(curr_p_proj_t.get(lb, p_now_t))
                p_proj_b_l[lb].append(curr_p_proj_b.get(lb, p_now_b))

    return {
        "lane": lane,
        "split": split,
        "tickers": n_tickers,
        "tid": np.asarray(tid_l, dtype=np.int64),
        "t": np.asarray(t_l),
        "secs": np.asarray(secs_l),
        "close": np.asarray(close_l),
        "cost_yes": np.asarray(cy_taker_l),
        "cost_no": np.asarray(cn_taker_l),
        "p_now_t": np.asarray(p_now_t_l),
        "p_now_b": np.asarray(p_now_b_l),
        "btc_bps": {lb: np.asarray(btc_bps_l[lb]) for lb in LOOKBACKS},
        "p_prev_t": {lb: np.asarray(p_prev_t_l[lb]) for lb in LOOKBACKS},
        "p_prev_b": {lb: np.asarray(p_prev_b_l[lb]) for lb in LOOKBACKS},
        "p_proj_t": {lb: np.asarray(p_proj_t_l[lb]) for lb in LOOKBACKS},
        "p_proj_b": {lb: np.asarray(p_proj_b_l[lb]) for lb in LOOKBACKS},
        "skips": skips,
    }


def find_triggers(pack: dict, lb: float, min_bps: float, model: str,
                  shift_mode: str, hurdle: float) -> list[tuple[int, str, float, float, float]]:
    """Detect triggers where BTC impulse >= min_bps, fair value shift >= 0.08, and edge >= hurdle."""
    bps = pack["btc_bps"][lb]
    p_now = pack["p_now_t" if model == "student_t" else "p_now_b"]

    if shift_mode == "projected":
        p_eval = pack["p_proj_t" if model == "student_t" else "p_proj_b"][lb]
        shift = np.abs(p_eval - p_now)
    else:
        p_prev = pack["p_prev_t" if model == "student_t" else "p_prev_b"][lb]
        shift = np.abs(p_now - p_prev)
        p_eval = p_now

    cy = pack["cost_yes"]
    cn = pack["cost_no"]

    # Bullish trigger: BTC bps >= min_bps, shift >= 0.08, YES ask in [0.35, 0.60], p_eval - cy >= hurdle
    up = (bps >= min_bps) & (shift >= 0.08) & (cy >= COST_LO) & (cy <= COST_HI) & ((p_eval - cy) >= hurdle)

    # Bearish trigger: BTC bps <= -min_bps, shift >= 0.08, NO ask in [0.35, 0.60], (1 - p_eval) - cn >= hurdle
    down = (bps <= -min_bps) & (shift >= 0.08) & (cn >= COST_LO) & (cn <= COST_HI) & (((1.0 - p_eval) - cn) >= hurdle)

    mask = up | down
    if not np.any(mask):
        return []

    use_y = up & (~down | ((p_eval - cy) >= ((1.0 - p_eval) - cn)))
    keep = BS.pick({"tid": pack["tid"], "ts": pack["t"]}, mask,
                   cooldown=COOLDOWN, per_window=PER_WINDOW)

    out = []
    for i in keep:
        side = S.YES if use_y[i] else S.NO
        p_side = float(p_eval[i]) if side == S.YES else 1.0 - float(p_eval[i])
        cost = float(cy[i]) if side == S.YES else float(cn[i])
        edge = p_side - cost
        out.append((i, side, p_side, edge, float(shift[i])))
    return out


def run_triggers(pack: dict, triggers: list[tuple[int, str, float, float, float]],
                 cfg_name: str, lb: float, out_dict: dict) -> None:
    lane = pack["lane"]
    for i, side, p_side, edge, shift in triggers:
        tid = int(pack["tid"][i])
        t = float(pack["t"][i])
        name = lane.tickers[tid]
        feat = [p_side, edge, float(pack["secs"][i]), float(pack["btc_bps"][lb][i]), shift]
        tape = RC.tape(lane, tid)

        try:
            bq, aq = tape.quote(t + S.LAG_S)
            cost_arr = S.entry_price(side, S.TAKER, bq, aq)
            qty = S.stake_qty(cost_arr, RC.STAKE)
        except (S.CrossedBook, S.NoQuote):
            qty = S.MAX_QTY

        res = S.run_trade(tape, side=side, trigger_ts=t, entry_style=S.TAKER, qty=qty,
                          cost_lo=COST_LO_D, cost_hi=COST_HI_D)

        if isinstance(res, S.Skip):
            out_dict[cfg_name]["skips"][res.reason] += 1
            continue

        out_dict[cfg_name]["rows"].append(RC.trade_row(
            res, lane=lane.lane, ticker=name, close=float(pack["close"][i]),
            slice_name=pack["split"], feat=feat, config=cfg_name, family=FAMILY,
            lookback=lb, p_side=p_side, edge=round(edge, 6)))


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
    btc_ts, btc_val = KF.clean_index(lanes["KXBTC15M"].index, before=cut)
    fit_params = fit_alt_lanes(lanes, cut)

    packs = [collect_alt_samples(lanes[ln], btc_ts, btc_val, fit_params[ln], cut, "is")
             for ln in ALT_LANES]

    all_combos = list(itertools.product(LOOKBACKS, MIN_BPS_LIST, MODELS, SHIFT_MODES, HURDLES))
    cfg_names = [config_name(lb, mb, mod, sm, h) for lb, mb, mod, sm, h in all_combos]
    out_dict = {cn: {"rows": [], "skips": Counter()} for cn in cfg_names}

    for pack in packs:
        for lb, mb, mod, sm, h in all_combos:
            cn = config_name(lb, mb, mod, sm, h)
            trig = find_triggers(pack, lb, mb, mod, sm, h)
            run_triggers(pack, trig, cn, lb, out_dict)

    configs = {}
    for lb, mb, mod, sm, h in all_combos:
        cn = config_name(lb, mb, mod, sm, h)
        configs[cn] = {
            "family": FAMILY,
            "lookback": lb,
            "min_bps": mb,
            "model": mod,
            "shift_mode": sm,
            "hurdle": h,
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
        "alt_lanes": list(ALT_LANES),
        "grid": {
            "lookbacks": list(LOOKBACKS),
            "min_bps": list(MIN_BPS_LIST),
            "models": list(MODELS),
            "shift_modes": list(SHIFT_MODES),
            "hurdles": list(HURDLES),
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
    params = parse_config(cfg_name)
    lb = params["lookback"]
    mb = params["min_bps"]
    mod = params["model"]
    sm = params["shift_mode"]
    h = params["hurdle"]
    is_rows = [json.loads(ln) for ln in TRADES.read_text().splitlines() if ln.strip()]

    btc_ts, btc_val = KF.clean_index(lanes["KXBTC15M"].index, before=None)
    fit_params = is_obj["fit_params"]

    def scorer():
        out_dict = {cfg_name: {"rows": [], "skips": Counter()}}
        for ln in ALT_LANES:
            pack = collect_alt_samples(lanes[ln], btc_ts, btc_val, fit_params[ln], cut, "oos")
            trig = find_triggers(pack, lb, mb, mod, sm, h)
            run_triggers(pack, trig, cfg_name, lb, out_dict)
        return out_dict[cfg_name]["rows"], dict(out_dict[cfg_name]["skips"]), RC.fit_payoff_model(is_rows)

    entry = RC.score_oos_once(FAMILY, {"config": cfg_name, **params, "exit": "settle"}, scorer)
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
