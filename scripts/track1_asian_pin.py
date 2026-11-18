#!/usr/bin/env python3
"""Charter 3 Track 1 - `final_120s_asian_pin`: Asian-Option Final-120s Settlement Average Arbitrage.

Score EVERY entry and exit exclusively through openkal/sim_core.py:
  - NO_ask = 1.00 - yes_bid
  - T+1s entry lag
  - T+1s exit placement lag (if any)
  - >= 50-contract print-through for maker fills
  - entry_price <= $0.62 (and in [$0.35, $0.60])
  - zero future spot look-ahead
  - hold to $1.00 settlement ($0 exit fee, zero T+1s exit placement lag)

Grid on IS only:
  - hurdle in [0.06, 0.09, 0.12]
  - secs_left windows in [(15, 60), (15, 90), (15, 120)]
  - entry_style in ["taker", "maker"] (5s patience for maker)
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
from scripts import rnd_common as RC  # noqa: E402
from scripts.asian_rti import AsianRTI  # noqa: E402

OUT = ROOT / "data" / "track1-asian-pin-is.json"
TRADES = ROOT / "data" / "track1-asian-pin-is-trades.jsonl"

FAMILY = "final_120s_asian_pin"
HURDLES = (0.06, 0.09, 0.12)
WINDOWS = ((15, 60), (15, 90), (15, 120))
STYLES = ("taker", "maker")
COST_LO, COST_HI = 0.35, 0.60
COST_LO_D, COST_HI_D = Decimal("0.35"), Decimal("0.60")
SECS_GRID = np.arange(120.0, 14.0, -1.0)  # 120..15, descending in secs_left (ascending in time)
COOLDOWN, PER_WINDOW = 60.0, 2
MIN_TRADES, MIN_PF, MIN_LANES = 30, 1.25, 2
FEATURES = ["p_side", "edge", "secs_left", "lead_sd"]


def config_name(hurdle: float, window: tuple[int, int], style: str) -> str:
    return f"h{hurdle:.2f}_w{window[0]}_{window[1]}_{style}"


def parse_config(name: str) -> tuple[float, tuple[int, int], str]:
    parts = name.split("_")
    h = float(parts[0][1:])
    w0 = int(parts[1][1:])
    w1 = int(parts[2])
    style = parts[3]
    return h, (w0, w1), style


def collect(lane, cut: float, split: str) -> dict:
    """Collect 1s quote samples with secs_left in [15, 120] and Asian RTI model fair values."""
    model = AsianRTI(lane.index["ts"], lane.index["value"], lane.index["accum"], lane.index["k"])
    skips: Counter = Counter()
    tid_l, t_l, secs_l, close_l = [], [], [], []
    p_l, cy_taker_l, cn_taker_l = [], [], []
    cy_maker_l, cn_maker_l = [], []
    lead_l = []
    pairs: list[tuple[float, int]] = []
    n_tickers = 0

    for tid, name in enumerate(lane.tickers):
        m = lane.meta.get(name)
        if m is None or RC.slice_of(m["close"], cut) != split:
            continue
        n_tickers += 1
        tape = RC.tape(lane, tid)
        close, strike = m["close"], m["strike"]
        for secs in SECS_GRID:
            t = close - secs
            f = model.fair(t, close, strike)
            if f is None:
                skips["no_fair"] += 1
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
            cy_m = float(S.entry_price(S.YES, S.MAKER, b, a))
            cn_m = float(S.entry_price(S.NO, S.MAKER, b, a))
            p = float(f["p"])
            lead = (f["mean"] - strike) / f["sd"] if f["sd"] > 0 else (
                0.0 if f["mean"] == strike else (1e6 if f["mean"] > strike else -1e6))

            tid_l.append(tid)
            t_l.append(t)
            secs_l.append(secs)
            close_l.append(close)
            p_l.append(p)
            cy_taker_l.append(cy_t)
            cn_taker_l.append(cn_t)
            cy_maker_l.append(cy_m)
            cn_maker_l.append(cn_m)
            lead_l.append(lead)

            for side, pside, c in ((S.YES, p, cy_t), (S.NO, 1.0 - p, cn_t)):
                if COST_LO <= c <= COST_HI:
                    pairs.append((pside, int(tape.side_won(side))))

    return {
        "lane": lane,
        "split": split,
        "tickers": n_tickers,
        "tid": np.asarray(tid_l, dtype=np.int64),
        "t": np.asarray(t_l),
        "secs": np.asarray(secs_l),
        "close": np.asarray(close_l),
        "p": np.asarray(p_l),
        "cost_yes_taker": np.asarray(cy_taker_l),
        "cost_no_taker": np.asarray(cn_taker_l),
        "cost_yes_maker": np.asarray(cy_maker_l),
        "cost_no_maker": np.asarray(cn_maker_l),
        "lead": np.asarray(lead_l),
        "pairs": pairs,
        "skips": skips,
    }


def find_triggers(pack: dict, hurdle: float, window: tuple[int, int], style: str) -> list[tuple[int, str]]:
    """Find eligible triggers within the secs_left window and cost band clearing hurdle."""
    w_min, w_max = window
    in_win = (pack["secs"] >= w_min) & (pack["secs"] <= w_max)
    if not np.any(in_win):
        return []

    p = pack["p"]
    if style == "taker":
        cy = pack["cost_yes_taker"]
        cn = pack["cost_no_taker"]
    else:
        cy = pack["cost_yes_maker"]
        cn = pack["cost_no_maker"]

    ey = p - cy
    en = (1.0 - p) - cn
    oky = in_win & (cy >= COST_LO) & (cy <= COST_HI) & (ey >= hurdle)
    okn = in_win & (cn >= COST_LO) & (cn <= COST_HI) & (en >= hurdle)
    use_y = oky & (~okn | (ey >= en))
    mask = oky | okn

    keep = BS.pick({"tid": pack["tid"], "ts": pack["t"]}, mask,
                   cooldown=COOLDOWN, per_window=PER_WINDOW)
    return [(i, S.YES if use_y[i] else S.NO) for i in keep]


def run_triggers(pack: dict, triggers: list[tuple[int, str]], hurdle: float,
                 window: tuple[int, int], style: str, out_dict: dict) -> None:
    lane = pack["lane"]
    entry_style = S.TAKER if style == "taker" else S.MAKER
    patience = 5.0 if style == "maker" else 0.0
    cfg = config_name(hurdle, window, style)

    for i, side in triggers:
        tid = int(pack["tid"][i])
        t = float(pack["t"][i])
        name = lane.tickers[tid]
        p_yes = float(pack["p"][i])
        p_side = p_yes if side == S.YES else 1.0 - p_yes
        if style == "taker":
            cost_now = float(pack["cost_yes_taker" if side == S.YES else "cost_no_taker"][i])
        else:
            cost_now = float(pack["cost_yes_maker" if side == S.YES else "cost_no_maker"][i])
        edge = p_side - cost_now
        lead = float(pack["lead"][i])
        feat = [p_side, edge, float(pack["secs"][i]), lead if side == S.YES else -lead]
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
            out_dict[cfg]["skips"][res.reason] += 1
            continue

        out_dict[cfg]["rows"].append(RC.trade_row(
            res, lane=lane.lane, ticker=name, close=float(pack["close"][i]),
            slice_name=pack["split"], feat=feat, config=cfg, family=FAMILY,
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
    packs = [collect(lane, cut, "is") for lane in lanes.values()]

    all_configs = list(itertools.product(HURDLES, WINDOWS, STYLES))
    cfg_names = [config_name(h, w, st) for h, w, st in all_configs]
    out_dict = {cn: {"rows": [], "skips": Counter()} for cn in cfg_names}

    for pack in packs:
        for h, w, st in all_configs:
            trig = find_triggers(pack, h, w, st)
            run_triggers(pack, trig, h, w, st, out_dict)

    configs = {}
    for h, w, st in all_configs:
        cn = config_name(h, w, st)
        configs[cn] = {
            "family": FAMILY,
            "hurdle": h,
            "window": list(w),
            "entry_style": st,
            "is": RC.summarize(out_dict[cn]["rows"], dict(out_dict[cn]["skips"])),
        }

    champion = pick_champion(configs)
    champ_rows = out_dict[champion["config"]]["rows"]
    TRADES.parent.mkdir(parents=True, exist_ok=True)
    TRADES.write_text("".join(json.dumps(r, default=RC._json_default) + "\n" for r in champ_rows))

    pairs = [pr for pack in packs for pr in pack["pairs"]]
    from openkal import features
    calib = {
        "ece_is": round(features.expected_calibration_error(pairs), 4),
        "n_pairs": len(pairs),
        "p_side_mean": round(float(np.mean([p for p, _ in pairs])), 4) if pairs else None,
        "side_won_rate": round(float(np.mean([w for _, w in pairs])), 4) if pairs else None,
    }

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
            "windows": [list(w) for w in WINDOWS],
            "entry_styles": list(STYLES),
            "cost_band": [COST_LO, COST_HI],
        },
        "sampling": samples,
        "configs": configs,
        "champion": champion,
        "champion_trades_file": str(TRADES),
        "settlement_calibration": calib,
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
    h, w, st = parse_config(cfg_name)
    is_rows = [json.loads(ln) for ln in TRADES.read_text().splitlines() if ln.strip()]

    def scorer():
        out_dict = {cfg_name: {"rows": [], "skips": Counter()}}
        for lane in lanes.values():
            pack = collect(lane, cut, "oos")
            trig = find_triggers(pack, h, w, st)
            run_triggers(pack, trig, h, w, st, out_dict)
        return out_dict[cfg_name]["rows"], dict(out_dict[cfg_name]["skips"]), RC.fit_payoff_model(is_rows)

    entry = RC.score_oos_once(FAMILY, {"config": cfg_name, "hurdle": h, "window": list(w),
                                       "entry_style": st, "exit": "settle"}, scorer)
    print(json.dumps({"family": FAMILY, "oos": entry["oos"], "gate": entry["gate"]},
                     default=RC._json_default))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lanes", default=",".join(RC.LANES))
    ap.add_argument("--score-oos", action="store_true")
    args = ap.parse_args()
    names = tuple(x.strip() for x in args.lanes.split(",") if x.strip())
    lanes = RC.load_lanes(names)
    cut = RC.global_cut(lanes)
    if args.score_oos:
        if set(names) != set(RC.LANES):
            raise RuntimeError("--score-oos needs all four lanes")
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
