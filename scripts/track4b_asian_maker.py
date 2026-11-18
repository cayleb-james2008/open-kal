#!/usr/bin/env python3
"""Track 4B `asian_final_min_maker`: a resting bid under the Asian-average fair value.

In the last minute of a 15-minute window the venue has locked `k` of the 60 settlement
fixings. With `k >= 15` and the exact Asian fair probability `p >= 0.70` for a side, rest a
$0-fee post-only bid at `min(0.62, round(p - hurdle, 2))` (kept inside [0.45, 0.62]) for
15 seconds and hold any fill to the $1.00 settlement. The fill is a late dump printing at
or through that bid: >= 50 contracts, via `openkal.sim_core.run_trade(entry_limit=...)`.

No parameter is fitted; the IS slice only picks the hurdle. Phases:
  default       score the IS grid (chronological first 60% of windows) and write the receipt;
  --score-oos   score the frozen IS champion once on the last 40%, logged to the OOS ledger
                (needs RND_OOS_UNSEAL=lead; refused unless the champion met the IS bar).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openkal import sim_core as S  # noqa: E402
from scripts import alpha_discover as AD  # noqa: E402
from scripts import bounded_sweep as BS  # noqa: E402
from scripts import rnd_common as RC  # noqa: E402
from scripts import track2c_true_l2_settle as C  # noqa: E402
from scripts import track4_common as T  # noqa: E402
from scripts.asian_rti import AsianRTI  # noqa: E402

FAMILY = "asian_final_min_maker"
KEY = "track4b_asian_final_min_maker"
OUT = ROOT / "data" / "track4b-asian-maker.json"
IS_TRADES = ROOT / "data" / "track4b-asian-maker-is-trades.jsonl"
OOS_TRADES = ROOT / "data" / "track4b-asian-maker-oos-trades.jsonl"

HURDLES = (0.06, 0.09, 0.12, 0.20, 0.30)   # the brief gave none: 4A's three plus two that move a capped bid
SECS_GRID = np.arange(75.0, 24.0, -1.0)    # secs_left 75..25, one decision per second
MIN_LOCKED, MIN_FAIR = 15, 0.70
BID_LO, BID_HI = Decimal("0.45"), Decimal("0.62")
PATIENCE_S = 15.0
COOLDOWN, PER_WINDOW = 60.0, 1


def config_name(hurdle: float) -> str:
    return f"h{hurdle:.2f}_maker"


def bid_for(p_side: float, hurdle: float) -> Decimal | None:
    """`min(0.62, round(p - hurdle, 2))`, or None when it falls under the $0.45 floor."""
    bid = Decimal(str(min(0.62, round(p_side - hurdle, 2))))
    return bid if bid >= BID_LO else None


def collect(lane, model, cut: float, split: str) -> dict:
    """Decision rows: every second of every window of `split` where a side clears the floor."""
    skips: Counter = Counter()
    rows = {"tid": [], "t": [], "secs": [], "close": [], "side": [], "p_side": [], "k": []}
    n_windows = 0
    for tid, name in enumerate(lane.tickers):
        m = lane.meta.get(name)
        if m is None or RC.slice_of(m["close"], cut) != split:
            continue
        n_windows += 1
        for secs in SECS_GRID:
            t = m["close"] - secs
            f = model.fair(t, m["close"], m["strike"])
            if f is None:
                skips["no_fair"] += 1
                continue
            if f["elapsed"] < MIN_LOCKED:
                skips["locked_below_15"] += 1
                continue
            if f["p"] >= MIN_FAIR:
                side, p_side = S.YES, f["p"]
            elif 1.0 - f["p"] >= MIN_FAIR:
                side, p_side = S.NO, 1.0 - f["p"]
            else:
                skips["fair_below_70"] += 1
                continue
            rows["side"].append(side)
            rows["p_side"].append(p_side)
            rows["tid"].append(tid)
            rows["t"].append(t)
            rows["secs"].append(secs)
            rows["close"].append(m["close"])
            rows["k"].append(f["elapsed"])
    pack = {k: np.asarray(v) for k, v in rows.items()}
    pack.update(lane=lane, split=split, windows=n_windows, skips=skips)
    return pack


def run_hurdle(pack: dict, hurdle: float, config: str, out: dict) -> None:
    """Pick the first decision per window, rest the bid, and price it through sim_core."""
    lane = pack["lane"]
    bids = [bid_for(float(p), hurdle) for p in pack["p_side"]]
    mask = np.array([b is not None for b in bids], dtype=bool)
    out[config]["skips"]["bid_below_floor"] += int((~mask).sum())
    for i in BS.pick({"tid": pack["tid"], "ts": pack["t"]}, mask, cooldown=COOLDOWN, per_window=PER_WINDOW):
        tid, side, bid = int(pack["tid"][i]), str(pack["side"][i]), bids[i]
        p_side = float(pack["p_side"][i])
        res = S.run_trade(RC.tape(lane, tid), side=side, trigger_ts=float(pack["t"][i]),
                          entry_style=S.MAKER, qty=S.stake_qty(bid, RC.STAKE), patience_s=PATIENCE_S,
                          cost_lo=BID_LO, cost_hi=BID_HI, entry_limit=bid)
        if isinstance(res, S.Skip):
            out[config]["skips"][res.reason] += 1
            continue
        out[config]["rows"].append(RC.trade_row(
            res, lane=lane.lane, ticker=lane.tickers[tid], close=float(pack["close"][i]),
            slice_name=pack["split"], feat=[p_side, p_side - float(bid), float(pack["secs"][i]), float(pack["k"][i])],
            config=config, family=FAMILY, hurdle=hurdle, p_side=p_side, bid=str(bid)))


def asian_model(lane) -> AsianRTI:
    return AsianRTI(lane.index["ts"], lane.index["value"], lane.index["accum"], lane.index["k"])


def run_is() -> dict:
    pin = T.pin_tape()
    closes = T.window_closes()
    if len(closes) != T.MIN_WINDOWS:
        raise RuntimeError(f"expected {T.MIN_WINDOWS} windows on the pinned tape, found {len(closes)}")
    cut = T.cut_of(closes)
    out = {config_name(h): {"rows": [], "skips": Counter()} for h in HURDLES}
    sampling = {}
    for name in RC.LANES:
        lane = AD.Lane(name)
        pack = collect(lane, asian_model(lane), cut, "is")
        sampling[name] = {"windows": pack["windows"], "decisions": int(len(pack["t"])), "skips": dict(pack["skips"])}
        for h in HURDLES:
            run_hurdle(pack, h, config_name(h), out)
        del lane, pack
    configs = {config_name(h): {"family": FAMILY, "hurdle": h, "style": "maker",
                                "is": RC.summarize(out[config_name(h)]["rows"], dict(out[config_name(h)]["skips"]))}
               for h in HURDLES}
    champion = C.pick_champion(configs)
    rows = out[champion["config"]]["rows"]
    C.atomic_text(IS_TRADES, "".join(json.dumps(r, default=RC._json_default) + "\n" for r in rows))
    obj = {"family": FAMILY, "split": "first 60% / last 40% of distinct window closes, all four lanes",
           "cut": cut, "windows": len(closes), "is_windows": sum(c < cut for c in closes),
           "oos_windows": sum(c >= cut for c in closes), "frozen_input_sha256": pin["sha256"],
           "rule": {"secs_left": [float(SECS_GRID[-1]), float(SECS_GRID[0])], "min_locked_fixings": MIN_LOCKED,
                    "min_fair_p": MIN_FAIR, "bid": "min(0.62, round(p - hurdle, 2))", "bid_band": [str(BID_LO), str(BID_HI)],
                    "patience_s": PATIENCE_S, "exit": "settle", "hurdles": list(HURDLES)},
           "sampling": sampling, "configs": configs, "champion": champion, "oos_scored": None}
    C.save_receipt(OUT, T.clean(obj), strict=True)
    return obj


def score_oos() -> dict:
    obj = json.loads(OUT.read_text())
    pin = T.pin_tape()
    if pin["sha256"] != obj["frozen_input_sha256"]:
        raise RuntimeError("receipt was frozen on a different tape")
    recovered = C.recover_scored_receipt(OUT, C.LEDGER, FAMILY, pin["sha256"])
    if recovered is not None:                       # a prior look is on the ledger: finish it, never rescore
        C.save_receipt(OUT, T.clean(recovered), strict=True)
        return recovered
    if obj["oos_scored"] is not None or not obj["champion"]["is_eligible"]:
        raise RuntimeError("OOS stays sealed: no IS champion met the bar, or it was already scored")
    config = obj["champion"]["config"]
    hurdle = obj["configs"][config]["hurdle"]
    is_rows = [json.loads(line) for line in IS_TRADES.read_text().splitlines() if line.strip()]

    def scorer():
        out = {config: {"rows": [], "skips": Counter()}}
        for name in RC.LANES:
            lane = AD.Lane(name)
            pack = collect(lane, asian_model(lane), obj["cut"], "oos")
            run_hurdle(pack, hurdle, config, out)
            del lane, pack
        C.atomic_text(OOS_TRADES, "".join(json.dumps(r, default=RC._json_default) + "\n" for r in out[config]["rows"]))
        return out[config]["rows"], dict(out[config]["skips"]), RC.fit_payoff_model(is_rows)

    C.score_with_receipt(OUT, obj, FAMILY, {"config": config, "hurdle": hurdle, **obj["rule"],
                                            "frozen_input_sha256": pin["sha256"]}, scorer)
    C.save_receipt(OUT, T.clean(obj), strict=True)
    return obj


def hand_off(obj: dict) -> None:
    best, score = obj["champion"]["is"], obj["oos_scored"]
    state = {"status": "OOS scored" if score else ("IS eligible; OOS pending" if obj["champion"]["is_eligible"]
                                                   else "IS fail; OOS unspent"),
             "champion": obj["champion"]["config"], "is": best, "is_eligible": obj["champion"]["is_eligible"],
             "oos_spent": score is not None, "oos": score["oos"] if score else None,
             "payoff_ece": score["payoff_ece"] if score else None, "gate": score["gate"] if score else None,
             "result_artifact": str(OUT.relative_to(ROOT)), "note": "Research only; no live promotion or trading."}
    final = score is not None or not obj["champion"]["is_eligible"]
    lines = [f"- {obj['windows']} windows ({obj['is_windows']} IS / {obj['oos_windows']} OOS), cut `{obj['cut']}`; "
             f"secs_left 25-75 with >= 15 locked fixings (so effectively <= 45s), fair p >= 0.70, "
             f"post-only bid min(0.62, round(p - hurdle, 2)) in [0.45, 0.62], 15s patience, hold to settlement. Priced by "
             f"`sim_core.run_trade(entry_limit=...)`, an additive maker-only argument added for this track.",
             f"- IS champion `{obj['champion']['config']}`: {T.scorecard(best)}; eligible: {obj['champion']['is_eligible']}."]
    lines.append(f"- One sealed OOS look: {T.scorecard(score['oos'])}; payoff-ECE {score['payoff_ece']}; "
                 f"gate pass: {score['gate']['pass']}." if score else "- IS bar missed; OOS left sealed, no ledger entry.")
    lines.append(T.SHARED_TAPE_NOTE)
    lines.append(f"- Receipt: `{OUT.relative_to(ROOT)}`; no live promotion.")
    T.update_handoff(KEY, state, "### Track 4B — final-minute Asian maker bid" if final else None, lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-oos", action="store_true")
    args = ap.parse_args()
    obj = score_oos() if args.score_oos else run_is()
    hand_off(obj)
    print(json.dumps({"champion": obj["champion"]["config"], "eligible": obj["champion"]["is_eligible"],
                      "is": obj["champion"]["is"], "oos": (obj["oos_scored"] or {}).get("oos")},
                     default=RC._json_default, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
