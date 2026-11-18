#!/usr/bin/env python3
"""Track 7 `cross_lane_iv_arb`: when an altcoin's binary is priced at a richer implied volatility than BTC's.

At every 5s grid second (secs_left 120..720) each lane's mid `(yes_bid + yes_ask) / 2` is inverted through the
Student-t CDF (IS-fitted nu, s) into an implied volatility, in the same units as the trailing-150s realised sigma:
    z_mkt = F^-1(mid),   iv = ln(spot/strike) / (z_mkt * s * sqrt(tau)),   tau = max(secs_left - 40, 1)
`iv_ratio = iv / realised sigma`. A binary's price is a function of vol only through |z|, so the inversion is only
defined when the mid and the spot side of the strike agree and |z_mkt| >= 0.05 (at the money a binary has no vega).
When an altcoin's iv_ratio is >= 1.35x BTC's at the same second, its binary is dearer than the market-wide premium
says it should be: high vol drags the in-the-money side's price toward 0.5, so that side is the cheap one. Buy it at
its taker ask in [$0.35, $0.60], T+1s, and hold to the $1.00 settlement through `sim_core.run_trade`.

One pre-registered configuration, no grid. A control with the same side and ask band but no dispersion trigger shows
whether the trigger adds anything over buying the favourite near the money.

Phases: default = IS scoring + receipt; --score-oos = the one sealed OOS look (RND_OOS_UNSEAL=lead).
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
from scripts.track6_gbdt_l2_meta import TAPE, brier, fit_scale, last_vals  # noqa: E402

FAMILY = "cross_lane_iv_arb"
KEY = "track7_cross_lane_iv_arb"
OUT = ROOT / "data" / "track7-cross-lane-iv.json"
IS_TRADES = ROOT / "data" / "track7-cross-lane-iv-is-trades.jsonl"
OOS_TRADES = ROOT / "data" / "track7-cross-lane-iv-oos-trades.jsonl"

SECS = np.arange(720.0, 119.0, -5.0)               # the synchronous grid: secs_left 720..120
RATIO, D_MIN = 1.35, 0.05
ASK_LO_M, ASK_HI_M = 350, 600                      # taker ask band in $0.001 units
QTY = S.MAX_QTY
COOLDOWN, PER_WINDOW = 60.0, 2
MIN_TRADES, MIN_PF, MIN_LANES = 20, 1.25, 2
ALTS = ("KXETH15M", "KXSOL15M", "KXXRP15M")
CONFIG = f"alt_iv_ratio_ge_{RATIO:.2f}x_btc_buy_itm_hold"
CONTROL = "control_buy_itm_hold_no_dispersion_trigger"


def implied_vol(mid, ln_spot_strike, tau, nu_table, s):
    """The vol that prices a binary at `mid` under the Student-t model; NaN where the inversion is undefined."""
    z = np.interp(mid, nu_table, KF.CDF_X)                        # F^-1(mid)
    with np.errstate(divide="ignore", invalid="ignore"):
        iv = ln_spot_strike / (z * s * np.sqrt(tau))
    ok = (ln_spot_strike * z > 0) & (np.abs(z) >= D_MIN) & np.isfinite(iv)
    return np.where(ok, iv, np.nan), z


def collect(lane, fit: dict, cut: float, split: str) -> dict:
    """Per grid second of every window of `split`: the quote, realised sigma and implied-vol ratio, all read at or before t."""
    its, v = KF.clean_index(lane.index)
    tv, cum, cnt = KF.sigma_prefix(its, v)
    table = KF.t_cdf_table(fit["nu"])
    cols = ("key", "tid", "t", "secs", "close", "y", "lnsk", "bid_m", "ask_m", "z", "iv", "sigma", "ratio")
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
        j = np.searchsorted(lane.ts[a0:a1], times, "right") - 1
        jj = np.maximum(j, 0)
        b, a = lane.bid[a0 + jj], lane.ask[a0 + jj]
        keep = (j >= 0) & np.isfinite(b) & np.isfinite(a) & (b > 0) & (a < 1) & (b < a - S.EPS)
        skips["no_quote"] += int((~keep).sum())
        spot = last_vals(its, v, times)
        good = np.isfinite(spot) & (spot > 0)
        skips["no_spot"] += int((keep & ~good).sum())
        keep &= good
        lnsk = np.log(np.where(keep, spot, 1.0) / m["strike"])
        tau = np.maximum(SECS - KF.TAU_OFFSET, 1.0)
        iv, z = implied_vol((b + a) / 2.0, lnsk, tau, table, fit["s"])
        sigma = np.asarray(KF.sigma_at(times, tv, cum, cnt), dtype=np.float64)
        ratio = iv / sigma
        skips["iv_undefined"] += int((keep & ~np.isfinite(ratio)).sum())
        keep &= np.isfinite(ratio)
        vals = (np.rint(m["close"]) * 1000 + SECS, np.full(len(times), tid), times, SECS, np.full(len(times), m["close"]),
                np.full(len(times), m["y"]), lnsk, np.rint(b * 1000), np.rint(a * 1000), z, iv, sigma, ratio)
        for c, x in zip(cols, vals):
            rows[c].append(np.asarray(x, dtype=np.float64)[keep])
    pack = {c: (np.concatenate(x) if x else np.zeros(0)) for c, x in rows.items()}
    pack["tid"] = pack["tid"].astype(np.int64)
    pack.update(split=split, windows=windows, skips=skips, fit=fit)
    pack["itm_yes"] = pack["lnsk"] > 0
    pack["ask_itm_m"] = np.where(pack["itm_yes"], pack["ask_m"], 1000 - pack["bid_m"])      # NO ask = 1 - yes_bid
    pack["side_won"] = np.where(pack["itm_yes"], pack["y"], 1.0 - pack["y"])
    return pack


def attach_btc(pack: dict, btc: dict) -> None:
    """The BTC iv_ratio at the very same window and grid second; NaN where BTC had none."""
    pack["btc_ratio"] = np.full(len(pack["key"]), np.nan)
    if len(btc["key"]):
        order = np.argsort(btc["key"])
        keys, ratio = btc["key"][order], btc["ratio"][order]
        i = np.minimum(np.searchsorted(keys, pack["key"]), len(keys) - 1)
        pack["btc_ratio"] = np.where(keys[i] == pack["key"], ratio[i], np.nan)
    # The alt's own price re-read with BTC's premium as the fair iv: sigma_fair = btc_ratio * realised sigma.
    table = KF.t_cdf_table(pack["fit"]["nu"])
    with np.errstate(divide="ignore", invalid="ignore"):
        z_fair = np.abs(pack["lnsk"]) / (pack["fit"]["s"] * pack["btc_ratio"] * pack["sigma"] * np.sqrt(np.maximum(pack["secs"] - KF.TAU_OFFSET, 1.0)))
    pack["p_fair"] = np.interp(z_fair, KF.CDF_X, table, right=np.nan)


def in_band(pack: dict) -> np.ndarray:
    return (pack["ask_itm_m"] >= ASK_LO_M) & (pack["ask_itm_m"] <= ASK_HI_M) & np.isfinite(pack["btc_ratio"])


def triggers(pack: dict, *, dispersion: bool) -> list[int]:
    ok = in_band(pack)
    if dispersion:
        ok &= pack["ratio"] >= RATIO * pack["btc_ratio"] - 1e-12
    return BS.pick({"tid": pack["tid"], "ts": pack["t"]}, ok, cooldown=COOLDOWN, per_window=PER_WINDOW)


def run_trades(lane, pack: dict, picks: list[int], config: str, out: dict) -> None:
    for i in picks:
        side = S.YES if pack["itm_yes"][i] else S.NO
        res = S.run_trade(RC.tape(lane, int(pack["tid"][i])), side=side, trigger_ts=float(pack["t"][i]),
                          entry_style=S.TAKER, qty=QTY, cost_lo=Decimal("0.35"), cost_hi=Decimal("0.60"))
        if isinstance(res, S.Skip):
            out[config]["skips"][res.reason] += 1
            continue
        out[config]["rows"].append(RC.trade_row(
            res, lane=lane.lane, ticker=lane.tickers[int(pack["tid"][i])], close=float(pack["close"][i]), slice_name=pack["split"],
            feat=[float(pack["ratio"][i] / pack["btc_ratio"][i]), float(pack["p_fair"][i]), float(pack["ask_itm_m"][i]) / 1000,
                  float(pack["secs"][i]), float(abs(pack["z"][i]))], config=config, family=FAMILY))


def brier_rows(packs: dict, rows) -> dict:
    """Brier of the BTC-premium fair probability and of the ITM ask on the same rows, plus the constant base rate."""
    sel = {n: rows(packs[n]) & np.isfinite(packs[n]["p_fair"]) for n in packs}
    p = np.concatenate([packs[n]["p_fair"][sel[n]] for n in packs])
    ask = np.concatenate([packs[n]["ask_itm_m"][sel[n]] / 1000 for n in packs])
    y = np.concatenate([packs[n]["side_won"][sel[n]] for n in packs])
    if not len(y):
        return {"rows": 0, "model": None, "ask": None, "base_rate": None}
    return {"rows": int(len(y)), "model": brier(p, y), "ask": brier(ask, y), "base_rate": brier(np.full(len(y), y.mean()), y)}


def eligible(s: dict) -> bool:
    return s["trades"] >= MIN_TRADES and s["pf"] >= MIN_PF and s["profitable_lanes"] >= MIN_LANES and s["crossed_fills"] == 0


def rolling_edge_bar(oos: dict) -> dict:
    return {"pass": oos["trades"] >= 12 and oos["net"] > 0 and oos["pf"] >= MIN_PF and oos["profitable_lanes"] >= MIN_LANES}


def gather(cut: float, split: str, fits: dict) -> dict:
    """Packs for all four lanes (BTC first), each lane's lane object released after its pass."""
    packs = {}
    for n in ("KXBTC15M", *ALTS):
        lane = AD.Lane(n)
        packs[n] = collect(lane, fits[n], cut, split)
        del lane
    for n in ALTS:
        attach_btc(packs[n], packs["KXBTC15M"])
    return packs


def run_split(packs: dict, configs: dict) -> dict:
    """{config: {rows, skips}} with each lane's trades priced against its own tape."""
    out = {c: {"rows": [], "skips": Counter()} for c in configs}
    for n in ALTS:
        lane = AD.Lane(n)
        for config, dispersion in configs.items():
            run_trades(lane, packs[n], triggers(packs[n], dispersion=dispersion), config, out)
        del lane
    return out


def run_is() -> dict:
    T.refuse_if_spent(FAMILY, OUT)
    pin = T.pin_tape(TAPE)
    closes = T.window_closes(TAPE)
    cut = closes[int(len(closes) * 0.6)]
    fits = {n: fit_scale(n, cut) for n in RC.LANES}
    packs = gather(cut, "is", fits)
    out = run_split(packs, {CONFIG: True, CONTROL: False})
    summ = {c: RC.summarize(v["rows"], dict(v["skips"])) for c, v in out.items()}
    ratio_rows = {n: int((in_band(p) & (p["ratio"] >= RATIO * p["btc_ratio"] - 1e-12)).sum()) for n, p in packs.items() if n in ALTS}
    C.atomic_text(IS_TRADES, "".join(json.dumps(r, default=RC._json_default) + "\n" for r in out[CONFIG]["rows"]))
    obj = {"family": FAMILY, "split": "first 60% / last 40% of the distinct settled window closes, all four lanes", "cut": cut,
           "windows": len(closes), "is_windows": sum(c < cut for c in closes), "oos_windows": sum(c >= cut for c in closes),
           "frozen_input_sha256": pin["sha256"], "student_t_scale": fits,
           "rule": {"grid_s": 5, "secs_left": [float(SECS[-1]), float(SECS[0])], "iv_ratio_vs_btc": RATIO, "min_abs_z": D_MIN,
                    "ask_band": [ASK_LO_M / 1000, ASK_HI_M / 1000], "entry": "taker, T+1s", "exit": "settle", "qty": QTY,
                    "cooldown_s": COOLDOWN, "per_window": PER_WINDOW, "eligibility": [MIN_TRADES, MIN_PF, MIN_LANES]},
           "sampling": {n: {"windows": p["windows"], "valid_iv_rows": int(len(p["t"])), "skips": dict(p["skips"]),
                            **({"triggering_rows": ratio_rows[n]} if n in ALTS else {})} for n, p in packs.items()},
           "config": CONFIG, "is": summ[CONFIG], "is_eligible": eligible(summ[CONFIG]), "control_is": summ[CONTROL],
           "brier_is": {"triggered_band_rows": brier_rows({n: packs[n] for n in ALTS}, lambda p: in_band(p) & (p["ratio"] >= RATIO * p["btc_ratio"] - 1e-12)),
                        "all_band_rows": brier_rows({n: packs[n] for n in ALTS}, in_band)},
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
    if obj["oos_scored"] is not None or not obj["is_eligible"]:
        raise RuntimeError("OOS stays sealed: the IS bar was missed, or it was already scored")
    is_rows = [json.loads(line) for line in IS_TRADES.read_text().splitlines() if line.strip()]

    def scorer():
        packs = gather(obj["cut"], "oos", obj["student_t_scale"])
        out = run_split(packs, {CONFIG: True})
        hit = lambda p: in_band(p) & (p["ratio"] >= RATIO * p["btc_ratio"] - 1e-12)
        obj["brier_oos"] = {"triggered_band_rows": brier_rows({n: packs[n] for n in ALTS}, hit),
                            "all_band_rows": brier_rows({n: packs[n] for n in ALTS}, in_band)}
        C.atomic_text(OOS_TRADES, "".join(json.dumps(r, default=RC._json_default) + "\n" for r in out[CONFIG]["rows"]))
        return out[CONFIG]["rows"], dict(out[CONFIG]["skips"]), RC.fit_payoff_model(is_rows)

    C.score_with_receipt(OUT, obj, FAMILY, {"config": CONFIG, **obj["rule"], "frozen_input_sha256": pin["sha256"]}, scorer)
    obj["rolling_edge_bar"] = rolling_edge_bar(obj["oos_scored"]["oos"])
    C.save_receipt(OUT, T.clean(obj), strict=True)
    return obj


def brier_text(label: str, b: dict) -> str:
    if not b["rows"]:
        return f"{label}: no rows"
    return f"{label}: BTC-premium fair {b['model']:.4f} vs side's own ask {b['ask']:.4f} vs base rate {b['base_rate']:.4f} ({b['rows']} rows)"


def hand_off(obj: dict) -> None:
    best, score, ok = obj["is"], obj["oos_scored"], obj["is_eligible"]
    state = {"status": "OOS scored" if score else ("IS eligible; OOS pending" if ok else "IS fail; OOS unspent"),
             "config": obj["config"], "is": best, "is_eligible": ok, "control_is": obj["control_is"], "oos_spent": score is not None,
             "oos": score["oos"] if score else None, "payoff_ece": score["payoff_ece"] if score else None,
             "strict_gate": score["gate"] if score else None, "rolling_edge_bar": obj.get("rolling_edge_bar"),
             "brier_is": obj["brier_is"], "brier_oos": obj.get("brier_oos"), "result_artifact": str(OUT.relative_to(ROOT)),
             "note": "Research only; no live promotion or trading."}
    trig = sum(s.get("triggering_rows", 0) for s in obj["sampling"].values())
    lines = [f"- {obj['windows']} windows ({obj['is_windows']} IS / {obj['oos_windows']} OOS), cut `{obj['cut']}`; 5s grid, secs_left 120-720; "
             f"alt iv_ratio >= {RATIO}x BTC's at the same second, buy the in-the-money side at its taker ask in $0.35-$0.60, hold to settlement. "
             f"{trig} IS decision rows met the trigger.",
             f"- IS: {T.scorecard(best)}; eligible (>= {MIN_TRADES} trades, PF >= {MIN_PF}, >= {MIN_LANES} lanes): {ok}.",
             f"- Control, same side and band with no dispersion trigger: {T.scorecard(obj['control_is'])}.",
             f"- IS {brier_text('triggered rows', obj['brier_is']['triggered_band_rows'])}; {brier_text('all in-band rows', obj['brier_is']['all_band_rows'])}."]
    if score:
        lines += [f"- One sealed OOS look: {T.scorecard(score['oos'])}; payoff-ECE {score['payoff_ece']}. Rolling-edge bar pass: "
                  f"{obj['rolling_edge_bar']['pass']}; strict repo gate pass: {score['gate']['pass']}.",
                  f"- OOS {brier_text('triggered rows', obj['brier_oos']['triggered_band_rows'])}."]
    else:
        lines.append("- IS bar missed; OOS left sealed, no ledger entry.")
    lines += [T.SHARED_TAPE_NOTE, f"- Receipt: `{OUT.relative_to(ROOT)}`; no live promotion."]
    T.update_handoff(KEY, state, "### Track 7 — cross-lane implied-vol dispersion" if score or not ok else None, lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-oos", action="store_true")
    args = ap.parse_args()
    obj = score_oos() if args.score_oos else run_is()
    hand_off(obj)
    print(json.dumps({"config": obj["config"], "eligible": obj["is_eligible"], "is": obj["is"], "control_is": obj["control_is"],
                      "brier_is": obj["brier_is"], "oos": (obj["oos_scored"] or {}).get("oos")}, default=RC._json_default, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
