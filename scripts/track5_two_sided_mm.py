#!/usr/bin/env python3
"""Track 5 `two_sided_spread_mm`: quote both sides of a quiet window and earn the spread.

At each decision second (secs_left 240..720, step 2s) in a low-volatility regime, with a book spread of
at least `min_spread` and both legs under the $0.62 cap, `sim_core.run_two_sided_mm` rests a $0-fee
post-only YES bid at the YES bid and NO bid at 1 - the YES ask. Both fill: the locked pair holds to the
$1.00 settlement and pays the spread. One fills: the other gets `hedge_timeout_s`, or until spot moves
>= 6 bps against the filled leg, then the lone leg crosses out as a taker. Nothing is carried to expiry.

The volatility regime is the trailing-150s index sigma at the decision second; `vol_max_quantile` is a quantile
of that sigma over the lane's IS decision rows only, frozen in the receipt for OOS. Nothing else is fitted.

Phases: default = IS grid (first 60% of windows) + receipt; --score-oos = the one sealed OOS look
(RND_OOS_UNSEAL=lead; refused unless the IS champion met the bar).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
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
from scripts import kurtosis_fv_snipe as KF  # noqa: E402
from scripts import rnd_common as RC  # noqa: E402
from scripts import track2c_true_l2_settle as C  # noqa: E402
from scripts import track4_common as T  # noqa: E402

FAMILY = "two_sided_spread_mm"
KEY = "track5_two_sided_spread_mm"
OUT = ROOT / "data" / "track5-two-sided-mm.json"
IS_TRADES = ROOT / "data" / "track5-two-sided-mm-is-trades.jsonl"
OOS_TRADES = ROOT / "data" / "track5-two-sided-mm-oos-trades.jsonl"

MIN_SPREADS = (0.03, 0.04, 0.05)
VOL_QUANTILES = (0.40, 0.60)
HEDGE_TIMEOUTS = (15.0, 30.0)
SECS_GRID = np.arange(720.0, 239.0, -2.0)          # secs_left 720..240
PATIENCE_S, ADVERSE_BPS = 15.0, 6.0
CAP = float(S.CAP)
COOLDOWN, PER_WINDOW = 60.0, 2     # an attempt rests <= 1s + 15s, then <= 30s: <= 47s, so 60s apart never overlap


def config_name(min_spread: float, quantile: float, timeout: float) -> str:
    return f"s{min_spread:.2f}_q{quantile:.2f}_t{int(timeout)}"


def grid() -> list[tuple[float, float, float]]:
    return list(itertools.product(MIN_SPREADS, VOL_QUANTILES, HEDGE_TIMEOUTS))


def collect(lane, its: np.ndarray, v: np.ndarray, cut: float, split: str) -> dict:
    """Decision rows (book and trailing sigma at `t`, nothing later) for every window of `split`."""
    tv, cum, cnt = KF.sigma_prefix(its, v)
    cols = ("tid", "t", "secs", "close", "bid", "ask", "sigma")
    rows = {c: [] for c in cols}
    skips: Counter = Counter()
    windows = 0
    for tid, name in enumerate(lane.tickers):
        m = lane.meta.get(name)
        if m is None or RC.slice_of(m["close"], cut) != split:
            continue
        windows += 1
        tape = RC.tape(lane, tid)
        for secs in SECS_GRID:
            t = m["close"] - secs
            if not math.isfinite(KF.index_last(its, v, t)):
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
            for c, val in zip(cols, (tid, t, secs, m["close"], b, a, float(KF.sigma_at(t, tv, cum, cnt)))):
                rows[c].append(val)
    pack = {c: np.asarray(x, dtype=np.int64 if c == "tid" else np.float64) for c, x in rows.items()}
    pack.update(lane=lane, split=split, windows=windows, skips=skips)
    return pack


def spread_funnel(pack: dict) -> dict:
    """How often the quote is wide enough at all: decision rows at each minimum spread, and the median spread."""
    spread = pack["ask"] - pack["bid"]
    return {**{f"spread_ge_{int(ms * 100)}c": int((spread >= ms - 1e-9).sum()) for ms in MIN_SPREADS},
            "median_spread": float(np.median(spread)) if len(spread) else None}


def vol_caps(sigma: np.ndarray) -> dict[str, float]:
    """The IS sigma at each `vol_max_quantile`, from one lane's IS decision rows."""
    return {f"{q:.2f}": float(np.quantile(sigma, q)) for q in VOL_QUANTILES}


def triggers(pack: dict, min_spread: float, cap: float) -> list[int]:
    ok = ((pack["ask"] - pack["bid"] >= min_spread - 1e-9) & (pack["sigma"] <= cap)
          & (pack["bid"] <= CAP + 1e-9) & (1.0 - pack["ask"] <= CAP + 1e-9))   # sim_core makes the final cap call
    return BS.pick({"tid": pack["tid"], "ts": pack["t"]}, ok, cooldown=COOLDOWN, per_window=PER_WINDOW)


def open_until(trigger_ts: float, res) -> float:
    """When an attempt stops holding or resting anything: a lone leg's exit, a pair's second fill, a skip's last rest."""
    if isinstance(res, S.Skip):
        return trigger_ts + S.LAG_S + (PATIENCE_S if res.reason == "unfilled" else 0.0)
    return max(leg.ts for leg in res.legs if leg.role == "entry") if res.reason == "pair" else res.exit_ts


def run_config(pack: dict, spot: tuple, caps: dict, min_spread: float, quantile: float, timeout: float, out: dict) -> None:
    lane, name = pack["lane"], config_name(min_spread, quantile, timeout)
    busy: dict[int, float] = {}
    for i in triggers(pack, min_spread, caps[f"{quantile:.2f}"]):
        tid, t = int(pack["tid"][i]), float(pack["t"][i])
        if t < busy.get(tid, -math.inf):         # an exit waits only for the first valid book row, so this cannot happen
            raise RuntimeError("two-sided attempts overlap inside one window")
        res = S.run_two_sided_mm(RC.tape(lane, tid), trigger_ts=t, qty=S.MAX_QTY, hedge_timeout_s=timeout,
                                 min_spread=Decimal(str(min_spread)), spot=spot, adverse_bps=ADVERSE_BPS,
                                 patience_s=PATIENCE_S)
        busy[tid] = open_until(t, res)
        if isinstance(res, S.Skip):
            out[name]["skips"][res.reason] += 1
            continue
        spread = float(pack["ask"][i] - pack["bid"][i])
        out[name]["rows"].append(RC.trade_row(
            res, lane=lane.lane, ticker=lane.tickers[tid], close=float(pack["close"][i]), slice_name=pack["split"],
            feat=[spread, float(pack["sigma"][i]), float(pack["secs"][i]), timeout],
            config=name, family=FAMILY, min_spread=min_spread, vol_quantile=quantile, hedge_timeout_s=timeout))


def new_out() -> dict:
    return {config_name(*g): {"rows": [], "skips": Counter()} for g in grid()}


def lane_inputs(name: str):
    lane = AD.Lane(name)
    return lane, KF.clean_index(lane.index)


def pick_champion(configs: dict) -> dict:
    """The eligible config with the best net; with none eligible, the best of those that traded at all.

    A config that never traded ties at $0.00 net and would otherwise be named as the best one."""
    return C.pick_champion({c: v for c, v in configs.items() if v["is"]["trades"]} or configs)


def run_is() -> dict:
    T.refuse_if_spent(FAMILY, OUT)
    pin = T.pin_tape()
    closes = T.window_closes()
    if len(closes) != T.MIN_WINDOWS:
        raise RuntimeError(f"expected {T.MIN_WINDOWS} windows on the pinned tape, found {len(closes)}")
    cut = T.cut_of(closes)
    out, caps, sampling = new_out(), {}, {}
    for name in RC.LANES:
        lane, spot = lane_inputs(name)
        pack = collect(lane, *spot, cut, "is")
        caps[name] = vol_caps(pack["sigma"])
        sampling[name] = {"windows": pack["windows"], "decision_rows": int(len(pack["t"])), **spread_funnel(pack),
                             "skips": dict(pack["skips"])}
        for g in grid():
            run_config(pack, spot, caps[name], *g, out)
        del lane, pack
    configs = {config_name(*g): {"family": FAMILY, "min_spread": g[0], "vol_quantile": g[1], "hedge_timeout_s": g[2],
                                 "is": RC.summarize(out[config_name(*g)]["rows"], dict(out[config_name(*g)]["skips"]))}
               for g in grid()}
    champion = pick_champion(configs)
    C.atomic_text(IS_TRADES, "".join(json.dumps(r, default=RC._json_default) + "\n" for r in out[champion["config"]]["rows"]))
    obj = {"family": FAMILY, "split": "first 60% / last 40% of distinct window closes, all four lanes", "cut": cut,
           "windows": len(closes), "is_windows": sum(c < cut for c in closes), "oos_windows": sum(c >= cut for c in closes),
           "frozen_input_sha256": pin["sha256"],
           "rule": {"secs_left": [float(SECS_GRID[-1]), float(SECS_GRID[0])], "patience_s": PATIENCE_S,
                    "adverse_spot_bps": ADVERSE_BPS, "cap": CAP, "max_pair_cost": 0.97, "exit": "taker cross of a lone leg",
                    "min_spreads": list(MIN_SPREADS), "vol_max_quantiles": list(VOL_QUANTILES),
                    "hedge_timeouts_s": list(HEDGE_TIMEOUTS)},
           "vol_caps_by_lane": caps, "sampling": sampling, "configs": configs, "champion": champion, "oos_scored": None}
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
    cfg = obj["configs"][config]
    is_rows = [json.loads(line) for line in IS_TRADES.read_text().splitlines() if line.strip()]

    def scorer():
        out = {config: {"rows": [], "skips": Counter()}}
        for name in RC.LANES:
            lane, spot = lane_inputs(name)
            pack = collect(lane, *spot, obj["cut"], "oos")
            run_config(pack, spot, obj["vol_caps_by_lane"][name], cfg["min_spread"], cfg["vol_quantile"],
                       cfg["hedge_timeout_s"], out)
            del lane, pack
        C.atomic_text(OOS_TRADES, "".join(json.dumps(r, default=RC._json_default) + "\n" for r in out[config]["rows"]))
        return out[config]["rows"], dict(out[config]["skips"]), RC.fit_payoff_model(is_rows)

    C.score_with_receipt(OUT, obj, FAMILY, {"config": config, **{k: cfg[k] for k in ("min_spread", "vol_quantile", "hedge_timeout_s")},
                                            "vol_caps_by_lane": obj["vol_caps_by_lane"], "frozen_input_sha256": pin["sha256"]}, scorer)
    C.save_receipt(OUT, T.clean(obj), strict=True)
    return obj


def funnel_text(obj: dict) -> str:
    """Why so few trades: wide quotes are rare, and a one-second arrival lag finds most of them gone."""
    lanes = obj["sampling"]
    rows = "; ".join(f"{n[2:5]} {c['spread_ge_3c']}/{c['spread_ge_4c']}/{c['spread_ge_5c']} of {c['decision_rows']}"
                     f" (median {c['median_spread']:.2f})" for n, c in lanes.items())
    lost = sum(c["is"]["skips"].get("spread", 0) for c in obj["configs"].values())
    return (f"IS decision rows with spread >= 3c/4c/5c: {rows}. "
            f"{lost} attempts across the grid met a spread under the minimum one second later.")


def reasons(s: dict) -> str:
    return ", ".join(f"{k} {v['trades']} (${v['net']:+.2f})" for k, v in s["by_reason"].items()) or "none"


def hand_off(obj: dict) -> None:
    best, score = obj["champion"]["is"], obj["oos_scored"]
    eligible = obj["champion"]["is_eligible"]
    state = {"status": "OOS scored" if score else ("IS eligible; OOS pending" if eligible else "IS fail; OOS unspent"),
             "champion": obj["champion"]["config"], "is": best, "is_eligible": eligible, "oos_spent": score is not None,
             "oos": score["oos"] if score else None, "payoff_ece": score["payoff_ece"] if score else None,
             "gate": score["gate"] if score else None, "result_artifact": str(OUT.relative_to(ROOT)),
             "note": "Research only; no live promotion or trading."}
    lines = [f"- {obj['windows']} windows ({obj['is_windows']} IS / {obj['oos_windows']} OOS), cut `{obj['cut']}`; secs_left 240-720, "
             f"both legs <= $0.62, pair cost <= $0.97, {int(PATIENCE_S)}s to first fill, lone leg crossed out at the hedge timeout or a "
             f"{ADVERSE_BPS:g} bps adverse spot move; vol cap = IS per-lane sigma quantile.",
             (f"- IS champion `{obj['champion']['config']}`: {T.scorecard(best)}. Exits: {reasons(best)}." if eligible else
              f"- No IS config met the bar; best by net `{obj['champion']['config']}`: {T.scorecard(best)}. "
              f"Exits: {reasons(best)}. {funnel_text(obj)}"),
             (f"- One sealed OOS look: {T.scorecard(score['oos'])}; payoff-ECE {score['payoff_ece']}; gate pass: {score['gate']['pass']}. "
              f"Exits: {reasons(score['oos'])}." if score else "- IS bar missed; OOS left sealed, no ledger entry."),
             T.SHARED_TAPE_NOTE, f"- Receipt: `{OUT.relative_to(ROOT)}`; no live promotion."]
    T.update_handoff(KEY, state, "### Track 5 — two-sided spread market making" if score or not eligible else None, lines)


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
