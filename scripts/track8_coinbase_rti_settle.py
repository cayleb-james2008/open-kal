#!/usr/bin/env python3
"""Track 8 `coinbase_rti_settle_arb`: the settlement average repriced with the live spot for the fixings still to come.

Kalshi's KX*15M settle on the 60-second average of the CF Benchmarks RTI. In the final 90 seconds, `k` of those
60 fixings are already locked (the venue's `accum`/`k`, read through `AsianRTI`, which refuses a stale or inconsistent
accumulator), and the remaining `n = 60 - k` are priced from the live spot: the engine's tick `spot`, the latest print
from the lane's feed venues at or before the decision second.
    mean = (locked_sum + spot * n) / 60
    variance = (n / 60)^2 * sigma^2 * n / 3                       (secs_left <= 60)
             = sigma^2 * (secs_left - 40)                         (secs_left > 60: the window has not opened; same as AsianRTI)
    p = P(mean >= strike), normal.
Trigger: |spot / RTI - 1| >= 8 bps AND p - ask >= hurdle on a side whose taker ask is in [$0.35, $0.60]; buy it at
T+1s through `sim_core.run_trade`, hold to the $1.00 settlement. Grid: hurdle in {0.06, 0.09, 0.12}.
A control reprices the same rows with the RTI's own spot, to see whether the live spot adds anything.

Phases: default = IS grid + receipt; --score-oos = the one sealed OOS look (RND_OOS_UNSEAL=lead).
"""
from __future__ import annotations

import argparse
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
from scripts.asian_rti import AsianRTI  # noqa: E402
from scripts.track6_gbdt_l2_meta import brier, last_vals  # noqa: E402

FAMILY = "coinbase_rti_settle_arb"
KEY = "track8_coinbase_rti_settle_arb"
TAPE = ROOT / "data" / "track8-tape"
OUT = ROOT / "data" / "track8-coinbase-rti-settle.json"
IS_TRADES = ROOT / "data" / "track8-coinbase-rti-settle-is-trades.jsonl"
OOS_TRADES = ROOT / "data" / "track8-coinbase-rti-settle-oos-trades.jsonl"
TICKS = Path("ticks.npz")

SECS = np.arange(90.0, 24.0, -1.0)                 # secs_left 90..25, one decision per second
DIVERGENCE_BPS = 8.0
HURDLES = (0.06, 0.09, 0.12)
ASK_LO_M, ASK_HI_M = 350, 600
QUOTE_MAX_AGE_S = 5.0
QTY = S.MAX_QTY
COOLDOWN, PER_WINDOW = 60.0, 2
MIN_TRADES, MIN_PF, MIN_LANES = 20, 1.25, 2
WINDOW_S = 60.0


def config_name(hurdle: float) -> str:
    return f"h{hurdle:.2f}"


def asian_p(spot: float, locked_sum: float, locked_s: float, secs_left: float, sigma: float, strike: float) -> float:
    """P(mean of the 60 fixings >= strike) with `spot` standing in for each fixing still to come."""
    n = max(0.0, WINDOW_S - locked_s)
    mean = (locked_sum + spot * n) / WINDOW_S
    var = sigma ** 2 * (secs_left - 40.0) if secs_left > WINDOW_S else (n / WINDOW_S) ** 2 * sigma ** 2 * n / 3.0
    if var <= 0.0:
        return 1.0 if mean >= strike else 0.0
    return 0.5 * (1.0 + math.erf((mean - strike) / (math.sqrt(var) * math.sqrt(2.0))))


def lane_spot(ticks: dict, series: str) -> tuple:
    """The feed spot for one lane as a sorted (ts, price) series; zero prices are missing."""
    m = (ticks["series"] == series) & (ticks["spot"] > 0) & np.isfinite(ticks["spot"])
    order = np.argsort(ticks["ts"][m], kind="stable")
    return ticks["ts"][m][order], ticks["spot"][m][order]


def collect(lane, model: AsianRTI, spot: tuple, cut: float, split: str) -> dict:
    """Decision rows (final 90s of each window of `split`): the venue-locked RTI state and the live spot at or before t."""
    cols = ("tid", "t", "secs", "close", "y", "strike", "cb", "rti", "div_bps", "p_cb", "p_rti", "bid_m", "ask_m")
    rows = {c: [] for c in cols}
    skips: Counter = Counter()
    windows = 0
    for tid, name in enumerate(lane.tickers):
        m = lane.meta.get(name)
        if m is None or RC.slice_of(m["close"], cut) != split:
            continue
        windows += 1
        times = m["close"] - SECS
        a0, a1 = int(lane.book_bounds[tid]), int(lane.book_bounds[tid + 1])
        if a1 == a0:
            skips["no_quote"] += len(times)
            continue
        bts = lane.ts[a0:a1]
        j = np.searchsorted(bts, times, "right") - 1
        jj = np.maximum(j, 0)
        b, a = lane.bid[a0 + jj], lane.ask[a0 + jj]
        quote = (j >= 0) & np.isfinite(b) & np.isfinite(a) & (b > 0) & (a < 1) & (b < a - S.EPS)
        fresh = times - bts[jj] <= QUOTE_MAX_AGE_S
        cb = last_vals(*spot, times)
        for i, t in enumerate(times):
            if not quote[i]:
                skips["no_quote"] += 1
            elif not fresh[i]:
                skips["stale_quote"] += 1
            elif not math.isfinite(cb[i]):
                skips["no_live_spot"] += 1
            elif (f := model.fair(float(t), m["close"], m["strike"])) is None:
                skips["no_rti_state"] += 1
            else:
                args = (f["locked_sum"], f["elapsed"], float(SECS[i]), f["sigma"], m["strike"])
                vals = (tid, t, SECS[i], m["close"], m["y"], m["strike"], cb[i], f["spot"], (cb[i] / f["spot"] - 1.0) * 1e4,
                        asian_p(cb[i], *args), asian_p(f["spot"], *args), np.rint(b[i] * 1000), np.rint(a[i] * 1000))
                for c, x in zip(cols, vals):
                    rows[c].append(float(x))
    pack = {c: np.asarray(x, dtype=np.int64 if c == "tid" else np.float64) for c, x in rows.items()}
    pack.update(lane=lane.lane, split=split, windows=windows, skips=skips)
    pack["ask_y_m"], pack["ask_n_m"] = pack["ask_m"], 1000.0 - pack["bid_m"]            # NO ask = 1 - yes_bid
    return pack


def band(ask_m: np.ndarray) -> np.ndarray:
    return (ask_m >= ASK_LO_M) & (ask_m <= ASK_HI_M)


def diverged(pack: dict) -> np.ndarray:
    return np.abs(pack["div_bps"]) >= DIVERGENCE_BPS - 1e-9


def triggers(pack: dict, hurdle: float) -> list[tuple[int, str, float]]:
    ey = pack["p_cb"] - pack["ask_y_m"] / 1000.0
    en = (1.0 - pack["p_cb"]) - pack["ask_n_m"] / 1000.0
    oky = diverged(pack) & band(pack["ask_y_m"]) & (ey >= hurdle - 1e-9)
    okn = diverged(pack) & band(pack["ask_n_m"]) & (en >= hurdle - 1e-9)
    use_y = oky & (~okn | (ey >= en))
    keep = BS.pick({"tid": pack["tid"], "ts": pack["t"]}, oky | okn, cooldown=COOLDOWN, per_window=PER_WINDOW)
    return [(i, S.YES if use_y[i] else S.NO, float(ey[i] if use_y[i] else en[i])) for i in keep]


def run_hurdle(lane, pack: dict, hurdle: float, out: dict) -> None:
    name = config_name(hurdle)
    for i, side, edge in triggers(pack, hurdle):
        res = S.run_trade(RC.tape(lane, int(pack["tid"][i])), side=side, trigger_ts=float(pack["t"][i]), entry_style=S.TAKER,
                          qty=QTY, cost_lo=Decimal("0.35"), cost_hi=Decimal("0.60"))
        if isinstance(res, S.Skip):
            out[name]["skips"][res.reason] += 1
            continue
        p = float(pack["p_cb"][i] if side == S.YES else 1.0 - pack["p_cb"][i])
        out[name]["rows"].append(RC.trade_row(
            res, lane=lane.lane, ticker=lane.tickers[int(pack["tid"][i])], close=float(pack["close"][i]), slice_name=pack["split"],
            feat=[p, edge, float(pack["secs"][i]), float(pack["div_bps"][i]) * (1.0 if side == S.YES else -1.0)],
            config=name, family=FAMILY, hurdle=hurdle))


def brier_rows(packs: dict, rows) -> dict:
    """Brier of the live-spot probability, the RTI-spot probability and the side's ask on every in-band side-row of `rows`."""
    cb, rti, ask, y = [], [], [], []
    for pack in packs.values():
        sel = rows(pack)
        for side, ask_m, p_cb, p_rti, won in ((S.YES, pack["ask_y_m"], pack["p_cb"], pack["p_rti"], pack["y"]),
                                                (S.NO, pack["ask_n_m"], 1 - pack["p_cb"], 1 - pack["p_rti"], 1 - pack["y"])):
            ok = sel & band(ask_m)
            cb.append(p_cb[ok]), rti.append(p_rti[ok]), ask.append(ask_m[ok] / 1000.0), y.append(won[ok])
    cb, rti, ask, y = map(np.concatenate, (cb, rti, ask, y))
    if not len(y):
        return {"rows": 0, "cb": None, "rti": None, "ask": None, "base_rate": None}
    return {"rows": int(len(y)), "cb": brier(cb, y), "rti": brier(rti, y), "ask": brier(ask, y), "base_rate": brier(np.full(len(y), y.mean()), y)}


def eligible(s: dict) -> bool:
    return s["trades"] >= MIN_TRADES and s["pf"] >= MIN_PF and s["profitable_lanes"] >= MIN_LANES and s["crossed_fills"] == 0


def pick_champion(configs: dict) -> dict:
    """The eligible config with the best net; with none eligible, the best of those that traded, flagged ineligible."""
    el = [c for c in configs if eligible(configs[c]["is"])]
    pool = el or [c for c in configs if configs[c]["is"]["trades"]] or list(configs)
    best = max(pool, key=lambda c: configs[c]["is"]["net"])
    return {"config": best, "is_eligible": bool(el), "is": configs[best]["is"]}


def rolling_edge_bar(oos: dict) -> dict:
    return {"pass": oos["trades"] >= 12 and oos["net"] > 0 and oos["pf"] >= MIN_PF and oos["profitable_lanes"] >= MIN_LANES}


def gather(cut: float, split: str) -> tuple[dict, dict]:
    """(packs by lane, funnel stats); each lane object is released after its pass."""
    ticks = dict(np.load(TAPE / TICKS, allow_pickle=False))
    packs = {}
    for n in RC.LANES:
        lane = AD.Lane(n)
        model = AsianRTI(lane.index["ts"], lane.index["value"], lane.index["accum"], lane.index["k"])
        packs[n] = collect(lane, model, lane_spot(ticks, n), cut, split)
        del lane, model
    return packs, ticks


def run_all(packs: dict, hurdles) -> dict:
    out = {config_name(h): {"rows": [], "skips": Counter()} for h in hurdles}
    for n in RC.LANES:
        lane = AD.Lane(n)
        for h in hurdles:
            run_hurdle(lane, packs[n], h, out)
        del lane
    return out


def funnel(packs: dict) -> dict:
    """How the trigger thins out: rows, divergent rows, divergent windows, rows that also clear each hurdle."""
    res = {}
    for n, p in packs.items():
        d = diverged(p)
        both = lambda h: int((d & ((band(p["ask_y_m"]) & (p["p_cb"] - p["ask_y_m"] / 1000 >= h - 1e-9)) |
                                   (band(p["ask_n_m"]) & ((1 - p["p_cb"]) - p["ask_n_m"] / 1000 >= h - 1e-9)))).sum())
        res[n] = {"windows": p["windows"], "rows": int(len(p["t"])), "diverged_rows": int(d.sum()),
                  "diverged_windows": int(len(set(p["close"][d]))), "median_abs_div_bps": float(np.median(np.abs(p["div_bps"]))) if len(p["t"]) else None,
                  "rows_clearing_hurdle": {config_name(h): both(h) for h in HURDLES}, "skips": dict(p["skips"])}
    return res


def run_is() -> dict:
    T.refuse_if_spent(FAMILY, OUT)
    pin = T.pin_tape(TAPE)
    closes = T.window_closes(TAPE)
    cut = closes[int(len(closes) * 0.6)]
    packs, _ = gather(cut, "is")
    out = run_all(packs, HURDLES)
    configs = {c: {"hurdle": float(c[1:]), "is": RC.summarize(v["rows"], dict(v["skips"]))} for c, v in out.items()}
    champion = pick_champion(configs)
    C.atomic_text(IS_TRADES, "".join(json.dumps(r, default=RC._json_default) + "\n" for r in out[champion["config"]]["rows"]))
    obj = {"family": FAMILY, "split": "first 60% / last 40% of the distinct settled window closes, all four lanes", "cut": cut,
           "windows": len(closes), "is_windows": sum(c < cut for c in closes), "oos_windows": sum(c >= cut for c in closes),
           "frozen_input_sha256": pin["sha256"],
           "rule": {"secs_left": [float(SECS[-1]), float(SECS[0])], "divergence_bps": DIVERGENCE_BPS, "hurdles": list(HURDLES),
                    "ask_band": [ASK_LO_M / 1000, ASK_HI_M / 1000], "entry": "taker, T+1s", "exit": "settle", "qty": QTY,
                    "spot_source": "ticks.npz `spot`: the engine's latest print from the lane's feed venues (Coinbase or Kraken; venue not recorded)",
                    "eligibility": [MIN_TRADES, MIN_PF, MIN_LANES]},
           "funnel": funnel(packs), "configs": configs, "champion": champion,
           "brier_is": {"all_in_band": brier_rows(packs, lambda p: np.ones(len(p["t"]), bool)),
                        "diverged_in_band": brier_rows(packs, diverged)},
           "oos_scored": None}
    C.save_receipt(OUT, T.clean(obj), strict=True)
    return obj


def score_oos() -> dict:
    T.refuse_rescore()
    obj = json.loads(OUT.read_text())
    pin = T.pin_tape(TAPE)
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
        packs, _ = gather(obj["cut"], "oos")
        out = run_all(packs, (hurdle,))
        obj["brier_oos"] = {"all_in_band": brier_rows(packs, lambda p: np.ones(len(p["t"]), bool)),
                            "diverged_in_band": brier_rows(packs, diverged)}
        obj["funnel_oos"] = funnel(packs)
        C.atomic_text(OOS_TRADES, "".join(json.dumps(r, default=RC._json_default) + "\n" for r in out[config]["rows"]))
        return out[config]["rows"], dict(out[config]["skips"]), RC.fit_payoff_model(is_rows)

    C.score_with_receipt(OUT, obj, FAMILY, {"config": config, "hurdle": hurdle, **obj["rule"], "frozen_input_sha256": pin["sha256"]}, scorer)
    obj["rolling_edge_bar"] = rolling_edge_bar(obj["oos_scored"]["oos"])
    C.save_receipt(OUT, T.clean(obj), strict=True)
    return obj


def brier_text(label: str, b: dict) -> str:
    if not b["rows"]:
        return f"{label}: no rows"
    return (f"{label}: live-spot {b['cb']:.4f}, RTI-spot {b['rti']:.4f}, side's ask {b['ask']:.4f}, base rate {b['base_rate']:.4f} "
            f"({b['rows']} side-rows)")


def hand_off(obj: dict) -> None:
    best, score, ok = obj["champion"]["is"], obj["oos_scored"], obj["champion"]["is_eligible"]
    state = {"status": "OOS scored" if score else ("IS eligible; OOS pending" if ok else "IS fail; OOS unspent"),
             "champion": obj["champion"]["config"], "is": best, "is_eligible": ok, "oos_spent": score is not None,
             "oos": score["oos"] if score else None, "payoff_ece": score["payoff_ece"] if score else None,
             "strict_gate": score["gate"] if score else None, "rolling_edge_bar": obj.get("rolling_edge_bar"),
             "funnel": obj["funnel"], "brier_is": obj["brier_is"], "brier_oos": obj.get("brier_oos"),
             "result_artifact": str(OUT.relative_to(ROOT)), "note": "Research only; no live promotion or trading."}
    f = obj["funnel"]
    tot = lambda k: sum(v[k] for v in f.values())
    lines = [f"- {obj['windows']} windows ({obj['is_windows']} IS / {obj['oos_windows']} OOS), cut `{obj['cut']}`; secs_left 25-90; live spot "
             f"(the engine's tick `spot`: latest Coinbase-or-Kraken print, venue not recorded) vs the RTI, |divergence| >= {DIVERGENCE_BPS:g} bps, "
             f"p - ask >= hurdle, taker T+1s, hold to settlement.",
             f"- Trigger funnel (IS): {tot('rows')} decision rows, {tot('diverged_rows')} with |divergence| >= {DIVERGENCE_BPS:g} bps in "
             f"{tot('diverged_windows')} lane-windows; rows also clearing each hurdle: "
             + ", ".join(f"{h} {sum(v['rows_clearing_hurdle'][h] for v in f.values())}" for h in (config_name(x) for x in HURDLES)) + ".",
             (f"- IS champion `{obj['champion']['config']}`: {T.scorecard(best)}." if ok else
              f"- No IS config met the bar (>= {MIN_TRADES} trades, PF >= {MIN_PF}, >= {MIN_LANES} lanes); best traded `{obj['champion']['config']}`: "
              f"{T.scorecard(best)}."),
             "- All configs: " + "; ".join(f"{c} {v['is']['trades']} trades, ${v['is']['net']:+.2f}" for c, v in obj["configs"].items()) + ".",
             f"- IS Brier: {brier_text('all in-band rows', obj['brier_is']['all_in_band'])}; "
             f"{brier_text('divergent rows', obj['brier_is']['diverged_in_band'])}."]
    if score:
        lines += [f"- One sealed OOS look: {T.scorecard(score['oos'])}; payoff-ECE {score['payoff_ece']}. Rolling-edge bar pass: "
                  f"{obj['rolling_edge_bar']['pass']}; strict repo gate pass: {score['gate']['pass']}.",
                  f"- OOS Brier: {brier_text('divergent rows', obj['brier_oos']['diverged_in_band'])}."]
    else:
        lines.append("- IS bar missed; OOS left sealed, no ledger entry.")
    lines += [T.SHARED_TAPE_NOTE, f"- Receipt: `{OUT.relative_to(ROOT)}`; no live promotion."]
    T.update_handoff(KEY, state, "### Track 8 — live-spot vs RTI settlement-average arbitrage" if score or not ok else None, lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-oos", action="store_true")
    args = ap.parse_args()
    obj = score_oos() if args.score_oos else run_is()
    hand_off(obj)
    print(json.dumps({"champion": obj["champion"]["config"], "eligible": obj["champion"]["is_eligible"], "is": obj["champion"]["is"],
                      "funnel": obj["funnel"], "brier_is": obj["brier_is"], "oos": (obj["oos_scored"] or {}).get("oos")},
                     default=RC._json_default, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
