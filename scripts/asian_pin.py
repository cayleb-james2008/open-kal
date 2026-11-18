#!/usr/bin/env python3
"""Charter 3 Track 1 - `final_120s_asian_pin`: hold the final-120s Asian pin to settlement.

The 15-minute crypto binaries settle YES when the CF Benchmarks RTI average over the
window's final 60s (`mean(RTI[-60s:])`) is at or above the published floor strike.
Inside the final 120 seconds that average is a partially-locked Asian option:

  * `k = 60 - secs_left` fixings are already locked (the venue streams their running
    mean, `scripts/asian_rti.py` reads it back causally);
  * the remaining fixings start at the current spot, so the settlement average has
    mean `(locked_sum + spot * n_rem) / 60` and variance `sigma^2 * secs_left^3 / 10800`
    inside the final 60s (the T^1.5 law `scripts/verify_settlement.py` measures) and
    `sigma^2 * (secs_left - 40)` beyond it, where nothing is locked yet
    (`scripts/asian_rti.settlement_parts`; locked sums use exact `math.fsum`), with
    sigma the trailing 150s realised
    volatility of the index tape itself - nothing is fitted, everything is read at
    the decision time.

Trade: at secs_left in [15, 120], when the analytical probability exceeds the side's
taker cost by at least the hurdle and the cost sits in [0.35, 0.60], buy 50 contracts
at T+1s (sim_core arrival floor) and hold to the $1.00 settlement ($0 exit fee).

    .venv/bin/python scripts/asian_pin.py --lanes KXSOL15M   # dev, one lane
    flock data/.heavy.lock .venv/bin/python scripts/asian_pin.py   # full IS run
    scripts/asian_pin.py --score-oos                        # sealed; lead only
"""
from __future__ import annotations

import argparse
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
from scripts import bounded_sweep as BS  # noqa: E402
from scripts import rnd_common as RC  # noqa: E402
from scripts.asian_rti import AsianRTI  # noqa: E402

OUT = ROOT / "data" / "asian-pin-is.json"
TRADES = ROOT / "data" / "asian-pin-is-trades.jsonl"

FAMILY = "final_120s_asian_pin"
HURDLES = (0.06, 0.09, 0.12)
COST_LO, COST_HI = 0.35, 0.60
COST_LO_D, COST_HI_D = Decimal("0.35"), Decimal("0.60")
SECS_GRID = np.arange(120.0, 14.0, -1.0)      # 120..15, ascending in time
COOLDOWN, PER_WINDOW = 60.0, 2
MIN_TRADES, MIN_PF, MIN_LANES = 30, 1.25, 2
FEATURES = ["p_side", "edge", "secs_left", "lead_sd"]


def config_name(hurdle: float) -> str:
    return f"h{hurdle:.2f}"


# ------------------------------------------------------------------ candidates
def collect(lane, cut: float, split: str) -> dict:
    """Every 1s sample with secs_left in [15, 120] of one split, with Asian fair values."""
    model = AsianRTI(lane.index["ts"], lane.index["value"], lane.index["accum"],
                     lane.index["k"])
    skips: Counter = Counter()
    tid_l, t_l, secs_l, close_l = [], [], [], []
    p_l, cy_l, cn_l, lead_l = [], [], [], []
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
            cy = float(S.entry_price(S.YES, S.TAKER, b, a))
            cn = float(S.entry_price(S.NO, S.TAKER, b, a))
            p = float(f["p"])
            lead = (f["mean"] - strike) / f["sd"] if f["sd"] > 0 else (
                0.0 if f["mean"] == strike else math_copysign_huge(f["mean"] - strike))
            tid_l.append(tid); t_l.append(t); secs_l.append(secs); close_l.append(close)
            p_l.append(p); cy_l.append(cy); cn_l.append(cn); lead_l.append(lead)
            for side, pside, c in ((S.YES, p, cy), (S.NO, 1.0 - p, cn)):
                if COST_LO <= c <= COST_HI:
                    pairs.append((pside, int(tape.side_won(side))))
    return {
        "lane": lane, "split": split, "tickers": n_tickers,
        "tid": np.asarray(tid_l, dtype=np.int64), "t": np.asarray(t_l),
        "secs": np.asarray(secs_l), "close": np.asarray(close_l),
        "p": np.asarray(p_l), "cost_yes": np.asarray(cy_l), "cost_no": np.asarray(cn_l),
        "lead": np.asarray(lead_l), "pairs": pairs, "skips": skips,
    }


def math_copysign_huge(d: float) -> float:
    """A decided average with zero remaining variance: +/- large lead, sign carries."""
    return 1e6 if d > 0 else -1e6


def find_triggers(pack: dict, hurdle: float) -> list[tuple[int, str]]:
    """Cost-band rows whose edge clears the hurdle, deduped (BS.pick)."""
    p, cy, cn = pack["p"], pack["cost_yes"], pack["cost_no"]
    ey = p - cy
    en = (1.0 - p) - cn
    oky = (cy >= COST_LO) & (cy <= COST_HI) & (ey >= hurdle)
    okn = (cn >= COST_LO) & (cn <= COST_HI) & (en >= hurdle)
    use_y = oky & (~okn | (ey >= en))
    mask = oky | okn
    keep = BS.pick({"tid": pack["tid"], "ts": pack["t"]}, mask,
                   cooldown=COOLDOWN, per_window=PER_WINDOW)
    return [(i, S.YES if use_y[i] else S.NO) for i in keep]


def run_triggers(pack: dict, triggers: list[tuple[int, str]], hurdle: float,
                 out: dict) -> None:
    lane = pack["lane"]
    for i, side in triggers:
        tid = int(pack["tid"][i])
        t = float(pack["t"][i])
        name = lane.tickers[tid]
        p_yes = float(pack["p"][i])
        p_side = p_yes if side == S.YES else 1.0 - p_yes
        cost_now = float(pack["cost_yes" if side == S.YES else "cost_no"][i])
        edge = p_side - cost_now
        lead = float(pack["lead"][i])
        feat = [p_side, edge, float(pack["secs"][i]), lead if side == S.YES else -lead]
        tape = RC.tape(lane, tid)
        try:
            bq, aq = tape.quote(t + S.LAG_S)
            qty = S.stake_qty(S.entry_price(side, S.TAKER, bq, aq), RC.STAKE)
        except (S.CrossedBook, S.NoQuote):
            qty = S.MAX_QTY   # run_trade reports the crossed / no-quote skip itself
        res = S.run_trade(tape, side=side, trigger_ts=t, entry_style=S.TAKER, qty=qty,
                          cost_lo=COST_LO_D, cost_hi=COST_HI_D)
        cfg = config_name(hurdle)
        if isinstance(res, S.Skip):
            out[cfg]["skips"][res.reason] += 1
            continue
        out[cfg]["rows"].append(RC.trade_row(
            res, lane=lane.lane, ticker=name, close=float(pack["close"][i]),
            slice_name=pack["split"], feat=feat, config=cfg, family=FAMILY,
            hurdle=hurdle, p_side=p_side, edge=round(edge, 6)))


def eligible(s: dict) -> bool:
    return (s["trades"] >= MIN_TRADES and s["pf"] >= MIN_PF
            and s["profitable_lanes"] >= MIN_LANES and s["crossed_fills"] == 0)


def pick_champion(configs: dict) -> dict:
    el = [c for c in configs if eligible(configs[c]["is"])]
    best = max(el or list(configs), key=lambda c: configs[c]["is"]["net"])
    return {"config": best, "is_eligible": bool(el), "is": configs[best]["is"]}


# --------------------------------------------------------------------- modes
def run_is(lanes: dict, cut: float) -> dict:
    t0 = time.time()
    packs = [collect(lane, cut, "is") for lane in lanes.values()]
    out = {config_name(h): {"rows": [], "skips": Counter()} for h in HURDLES}
    trig_counts: Counter = Counter()
    for pack in packs:
        for h in HURDLES:
            trig = find_triggers(pack, h)
            trig_counts[f"h{h:.2f}"] += len(trig)
            run_triggers(pack, trig, h, out)
    configs = {config_name(h): {"family": FAMILY, "hurdle": h,
                                "is": RC.summarize(out[config_name(h)]["rows"],
                                                   dict(out[config_name(h)]["skips"]))}
               for h in HURDLES}
    champion = pick_champion(configs)
    champ_rows = out[champion["config"]]["rows"]
    TRADES.parent.mkdir(parents=True, exist_ok=True)
    TRADES.write_text("".join(json.dumps(r, default=RC._json_default) + "\n"
                              for r in champ_rows))

    calib = {}
    pairs = [pr for pack in packs for pr in pack["pairs"]]
    from openkal import features  # noqa: E402  (local: only used here)
    calib = {"ece_is": round(features.expected_calibration_error(pairs), 4),
             "n_pairs": len(pairs),
             "p_side_mean": round(float(np.mean([p for p, _ in pairs])), 4) if pairs else None,
             "side_won_rate": round(float(np.mean([w for _, w in pairs])), 4) if pairs else None,
             "cost_band": [COST_LO, COST_HI],
             "pairs_over": "every IS sampled quote (secs 120..15 step 1), both sides, cost in band"}
    samples = {pack["lane"].lane: {"tickers": pack["tickers"],
                                   "valid_samples": int(len(pack["t"])),
                                   "skips": dict(pack["skips"])} for pack in packs}
    obj = {
        "split": "IS only: markets with close < cut; the Asian model is never fitted, "
                 "it is read at each decision time",
        "cut": cut,
        "cut_source": "rnd_common.global_cut over the four lanes",
        "lanes": list(lanes),
        "hurdles": list(HURDLES),
        "cost_band": [COST_LO, COST_HI],
        "features": FEATURES,
        "dedupe": {"cooldown_s": COOLDOWN, "per_window": PER_WINDOW},
        "sampling": samples,
        "trigger_counts": dict(trig_counts),
        "configs": configs,
        "champion": champion,
        "champion_trades_file": str(TRADES),
        "settlement_calibration": calib,
        "runtime_s": round(time.time() - t0, 1),
    }
    RC.dump(OUT, obj)
    return obj


def score_oos(lanes: dict, cut: float, is_obj: dict) -> int:
    """Sealed: score the frozen champion once on OOS. Never run by Track 1 itself."""
    if abs(is_obj["cut"] - cut) > 1e-9:
        raise RuntimeError("IS JSON cut does not match this cut")
    cfg_name = is_obj["champion"]["config"]
    hurdle = is_obj["configs"][cfg_name]["hurdle"]
    is_rows = [json.loads(ln) for ln in TRADES.read_text().splitlines() if ln.strip()]

    def scorer():
        out = {cfg_name: {"rows": [], "skips": Counter()}}
        for lane in lanes.values():
            pack = collect(lane, cut, "oos")
            trig = find_triggers(pack, hurdle)
            run_triggers(pack, trig, hurdle, out)
        return out[cfg_name]["rows"], dict(out[cfg_name]["skips"]), RC.fit_payoff_model(is_rows)

    entry = RC.score_oos_once(FAMILY, {"config": cfg_name, "hurdle": hurdle,
                                       "exit": "settle"}, scorer)
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
                                   "win_rate": obj["champion"]["is"]["win_rate"]}},
                     default=RC._json_default, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
