#!/usr/bin/env python3
"""Track 4A `empirical_settle_cal`: a calibrated P(side wins) in place of the raw Student-t fair value.

For every decision second (secs_left 90..720, step 2s) and each side whose taker ask is in
[$0.35, $0.62], a ridge logistic regression predicts P(side_won) from causal features:
  z_student_t   log(spot/strike) / (sigma * sqrt(tau_eff) * s), signed to the side (IS-fitted t scale s)
  tau_eff       max(secs_left - 40, 1)
  spread        yes_ask - yes_bid
  microprice_minus_mid, flow_5s, flow_15s (signed to the side), entry_ask (the side's taker ask)
A trade fires when `p - cost >= hurdle` (cost = the style's entry price) and is priced by
`sim_core.run_trade`, held to the $1.00 settlement.

Selection is honest about in-sample fit: each IS window's probabilities come from a model fitted
on the *other* time blocks (5 blocks, keyed on window close across all four lanes, with one
window purged each side), so the IS profit factor is out-of-fold. Lambda is chosen by that same
blocked log-loss. The frozen OOS model is refitted on all IS windows only (every training row
has ts < cut - 900) and stored in the receipt.

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
from scripts.f2_structural import apply_logistic  # noqa: E402

FAMILY = "empirical_settle_cal"
KEY = "track4a_empirical_settle_cal"
OUT = ROOT / "data" / "track4a-empirical-settle.json"
IS_TRADES = ROOT / "data" / "track4a-empirical-settle-is-trades.jsonl"
OOS_TRADES = ROOT / "data" / "track4a-empirical-settle-oos-trades.jsonl"

HURDLES = (0.06, 0.09, 0.12)
STYLES = ("taker", "maker")
SECS_GRID = np.arange(720.0, 89.0, -2.0)
ASK_LO, ASK_HI = 0.35, 0.62
COST_LO, COST_HI = Decimal("0.35"), Decimal("0.62")
COOLDOWN, PER_WINDOW, MAKER_PATIENCE_S = 60.0, 2, 5.0
FEATURES = ["z_student_t", "tau_eff", "spread", "microprice_minus_mid", "flow_5s", "flow_15s", "entry_ask"]
LAMBDAS = (10.0, 1e3, 1e5)
FOLDS = 5
MAX_L2_AGE_S = 2.0


def config_name(hurdle: float, style: str) -> str:
    return f"h{hurdle:.2f}_{style}"


# ------------------------------------------------------------------ model
def fit_ridge_logistic(x: np.ndarray, y: np.ndarray, lam: float, *, iters: int = 60) -> tuple[np.ndarray, np.ndarray, dict]:
    """Newton ridge logistic: returns (w, norm) in `f2_structural.apply_logistic` form, plus fit info.

    `x` is standardised on the fit, the intercept (last in `w`) is not penalised, and the loop
    stops at the optimum (max |gradient| below 1e-6 of the row count), not after a fixed step count.
    """
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd[sd < 1e-12] = 1.0
    xs = np.column_stack([(x - mu) / sd, np.ones(len(x))])
    pen = np.r_[np.full(x.shape[1], lam), 0.0]
    w = np.zeros(xs.shape[1])
    for step in range(iters):
        p = 1.0 / (1.0 + np.exp(-(xs @ w)))
        grad = xs.T @ (p - y) + pen * w
        if np.abs(grad).max() < 1e-6 * len(x):
            break
        hess = (xs * (p * (1.0 - p))[:, None]).T @ xs + np.diag(pen) + 1e-9 * np.eye(len(w))
        w -= np.linalg.solve(hess, grad)
    return w, np.concatenate((mu, sd)), {"iters": step + 1, "max_grad_per_row": float(np.abs(grad).max() / len(x))}


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def side_x(pack: dict, side: str) -> np.ndarray:
    """Feature matrix for one side: direction-bearing columns flip sign for NO."""
    sign = 1.0 if side == S.YES else -1.0
    return np.column_stack([
        sign * pack["z"], pack["tau"], pack["spread"], sign * np.nan_to_num(pack["mpm"], nan=0.0),
        sign * pack["f5"], sign * pack["f15"], pack["ask_y"] if side == S.YES else pack["ask_n"]])


def in_domain(pack: dict, side: str) -> np.ndarray:
    ask = pack["ask_y"] if side == S.YES else pack["ask_n"]
    return np.isfinite(side_x(pack, side)).all(axis=1) & (ask >= ASK_LO) & (ask <= ASK_HI)


def side_label(pack: dict, side: str) -> np.ndarray:
    return pack["y"] if side == S.YES else 1.0 - pack["y"]


def market_logloss(packs: dict) -> float:
    """Log-loss of the side's own taker ask used as its win probability, on the model's domain rows."""
    p, y = [], []
    for pack in packs.values():
        for side in (S.YES, S.NO):
            d = in_domain(pack, side)
            p.append((pack["ask_y"] if side == S.YES else pack["ask_n"])[d])
            y.append(side_label(pack, side)[d])
    return log_loss(np.concatenate(p), np.concatenate(y))


def fold_masks(rank: np.ndarray, n_windows: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(train, held) row masks for fold k of FOLDS time blocks; the held block's neighbours are purged."""
    block_of = np.arange(n_windows) * FOLDS // n_windows
    lo, hi = np.flatnonzero(block_of == k)[[0, -1]]
    return (rank < lo - 1) | (rank > hi + 1), block_of[rank] == k


def stack(pack: dict, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Both sides' in-domain training rows for the selected pack rows."""
    xs, ys = [], []
    for side in (S.YES, S.NO):
        m = rows & in_domain(pack, side)
        xs.append(side_x(pack, side)[m])
        ys.append(side_label(pack, side)[m])
    return np.concatenate(xs), np.concatenate(ys)


def predict(model: dict, pack: dict) -> tuple[np.ndarray, np.ndarray]:
    """(p_yes, p_no) from a stored model; NaN outside the training domain."""
    w, norm = np.asarray(model["w"]), np.asarray(model["norm"])
    out = []
    for side in (S.YES, S.NO):
        p = apply_logistic(np.nan_to_num(side_x(pack, side), nan=0.0), w, norm)
        out.append(np.where(in_domain(pack, side), p, np.nan))
    return out[0], out[1]


def fit_model(packs: dict, closes_is: list[float], cut: float) -> tuple[dict, dict, dict]:
    """Choose lambda by blocked log-loss; return (frozen model, out-of-fold p per lane, cv table)."""
    rank_of = {c: i for i, c in enumerate(closes_is)}
    n_win = len(closes_is)
    for pack in packs.values():
        pack["rank"] = np.array([rank_of[c] for c in pack["close"]], dtype=np.int64)
        if len(pack["t"]) and float(pack["t"].max()) >= cut - 900.0:
            raise RuntimeError("an IS training row sits inside the final 900s before the cut")
    cv, oof_by_lam = {}, {}
    for lam in LAMBDAS:
        oof = {n: [np.full(len(p["t"]), np.nan), np.full(len(p["t"]), np.nan)] for n, p in packs.items()}
        for k in range(FOLDS):
            parts = [stack(p, fold_masks(p["rank"], n_win, k)[0]) for p in packs.values()]
            w, norm, _ = fit_ridge_logistic(np.concatenate([a for a, _ in parts]), np.concatenate([b for _, b in parts]), lam)
            for n, p in packs.items():
                held = fold_masks(p["rank"], n_win, k)[1]
                py, pn = predict({"w": w, "norm": norm}, p)
                oof[n][0][held], oof[n][1][held] = py[held], pn[held]
        ll = [log_loss(oof[n][i][in_domain(p, s)], side_label(p, s)[in_domain(p, s)])
              for n, p in packs.items() for i, s in enumerate((S.YES, S.NO))]
        cv[str(lam)] = float(np.mean(ll))
        oof_by_lam[lam] = oof
    best = min(LAMBDAS, key=lambda lam: cv[str(lam)])
    parts = [stack(p, np.ones(len(p["t"]), bool)) for p in packs.values()]
    w, norm, info = fit_ridge_logistic(np.concatenate([a for a, _ in parts]), np.concatenate([b for _, b in parts]), best)
    model = {"features": FEATURES, "lambda": best, "w": w.tolist(), "norm": norm.tolist(), "fit": info,
             "rows": int(sum(len(b) for _, b in parts))}
    return model, oof_by_lam[best], cv


# --------------------------------------------------------------- sampling
def collect(lane, micro: np.ndarray, fit: dict, cut: float, split: str) -> dict:
    """Causal decision rows for every settled window of `split`: nothing after `t` is read."""
    its, v = KF.clean_index(lane.index, before=cut if split == "is" else None)
    tv, cum, cnt = KF.sigma_prefix(its, v)
    cols = ("tid", "t", "secs", "close", "y", "z", "tau", "spread", "mpm", "f5", "f15",
            "ask_y", "ask_n", "bid_y", "bid_n")
    rows = {c: [] for c in cols}
    skips: Counter = Counter()
    windows = 0
    for tid, name in enumerate(lane.tickers):
        m = lane.meta.get(name)
        if m is None or RC.slice_of(m["close"], cut) != split:
            continue
        windows += 1
        tape = RC.tape(lane, tid)
        a0, a1 = int(lane.book_bounds[tid]), int(lane.book_bounds[tid + 1])
        lo, hi = int(lane.trade_bounds[tid]), int(lane.trade_bounds[tid + 1])
        tts, sz = lane.trade_ts[lo:hi], lane.trade_sz[lo:hi]
        cs = np.concatenate(([0.0], np.cumsum(np.where(lane.trade_ty[lo:hi] == 1, sz, -sz))))
        ct = np.concatenate(([0.0], np.cumsum(sz)))

        def flow(t, w):
            i, j = int(np.searchsorted(tts, t - w, "left")), int(np.searchsorted(tts, t, "right"))
            return float((cs[j] - cs[i]) / (ct[j] - ct[i])) if ct[j] > ct[i] else 0.0

        for secs in SECS_GRID:
            t = m["close"] - secs
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
            tau = max(secs - KF.TAU_OFFSET, 1.0)
            vol = float(KF.sigma_at(t, tv, cum, cnt)) * math.sqrt(tau)
            if vol <= 0:
                skips["zero_vol"] += 1
                continue
            j = int(np.searchsorted(lane.ts[a0:a1], t, "right")) - 1       # the row quote() used
            mp = float(micro[a0 + j]) if j >= 0 and t - float(lane.ts[a0 + j]) <= MAX_L2_AGE_S else float("nan")
            mp = mp if 0.0 < mp < 1.0 else float("nan")
            for c, val in zip(cols, (tid, t, secs, m["close"], m["y"], math.log(spot / m["strike"]) / (vol * fit["s"]),
                                     tau, a - b, mp - (a + b) / 2.0, flow(t, 5.0), flow(t, 15.0),
                                     float(S.entry_price(S.YES, S.TAKER, b, a)), float(S.entry_price(S.NO, S.TAKER, b, a)),
                                     float(S.entry_price(S.YES, S.MAKER, b, a)), float(S.entry_price(S.NO, S.MAKER, b, a)))):
                rows[c].append(val)
    pack = {c: np.asarray(v_, dtype=np.int64 if c == "tid" else np.float64) for c, v_ in rows.items()}
    pack.update(lane=lane, split=split, windows=windows, skips=skips)
    return pack


def triggers(pack: dict, p_yes: np.ndarray, p_no: np.ndarray, hurdle: float, style: str):
    cy, cn = (pack["ask_y"], pack["ask_n"]) if style == "taker" else (pack["bid_y"], pack["bid_n"])
    ey, en = p_yes - cy, p_no - cn
    oky = (cy >= ASK_LO) & (cy <= ASK_HI) & (ey >= hurdle)          # NaN p (outside the domain) is never >= hurdle
    okn = (cn >= ASK_LO) & (cn <= ASK_HI) & (en >= hurdle)
    use_y = oky & (~okn | (ey >= en))
    keep = BS.pick({"tid": pack["tid"], "ts": pack["t"]}, oky | okn, cooldown=COOLDOWN, per_window=PER_WINDOW)
    return [(i, S.YES if use_y[i] else S.NO) for i in keep]


def run_config(pack: dict, p_yes: np.ndarray, p_no: np.ndarray, hurdle: float, style: str, out: dict) -> None:
    lane, name = pack["lane"], config_name(hurdle, style)
    entry_style = S.TAKER if style == "taker" else S.MAKER
    for i, side in triggers(pack, p_yes, p_no, hurdle, style):
        tid, t = int(pack["tid"][i]), float(pack["t"][i])
        tape = RC.tape(lane, tid)
        yes = side == S.YES
        p_side = float((p_yes if yes else p_no)[i])
        cost = float((pack["ask_y"] if yes else pack["ask_n"])[i] if style == "taker" else (pack["bid_y"] if yes else pack["bid_n"])[i])
        try:
            qty = S.stake_qty(S.entry_price(side, entry_style, *tape.quote(t + S.LAG_S)), RC.STAKE)
        except (S.CrossedBook, S.NoQuote):
            qty = S.MAX_QTY
        res = S.run_trade(tape, side=side, trigger_ts=t, entry_style=entry_style, qty=qty,
                          patience_s=MAKER_PATIENCE_S, cost_lo=COST_LO, cost_hi=COST_HI)
        if isinstance(res, S.Skip):
            out[name]["skips"][res.reason] += 1
            continue
        sign = 1.0 if yes else -1.0
        out[name]["rows"].append(RC.trade_row(
            res, lane=lane.lane, ticker=lane.tickers[tid], close=float(pack["close"][i]), slice_name=pack["split"],
            feat=[p_side, p_side - cost, float(pack["secs"][i]), sign * float(pack["z"][i]), sign * float(pack["f5"][i])],
            config=name, family=FAMILY, hurdle=hurdle, p_side=p_side, edge=round(p_side - cost, 6)))


def load_lane(name: str):
    lane = AD.Lane(name)
    with np.load(T.TAPE / f"book_{name}.npz", allow_pickle=False) as data:
        micro = data["microprice"].astype(np.float64)
    return lane, micro


def fit_scale(name: str, cut: float) -> dict:
    """Student-t scale for one lane from index prints before cut - 900 only."""
    index = dict(np.load(T.TAPE / f"index_{AD.INDEX_BY_SERIES[name]}.npz"))
    its, v = KF.clean_index(index, before=cut)
    return KF.fit_lane(its, v, cut)


def new_out(configs) -> dict:
    return {c: {"rows": [], "skips": Counter()} for c in configs}


# ----------------------------------------------------------------- phases
def score_is(packs: dict, get_lane, model: dict, oof: dict) -> tuple[dict, dict]:
    """Price every config twice: `is` on the out-of-fold probabilities (what selection reads),
    `is_insample` on the all-IS refit (shown for contrast). Returns (configs, out-of-fold outputs)."""
    names = [config_name(h, s) for h in HURDLES for s in STYLES]
    oof_out, ins_out = new_out(names), new_out(names)
    for n, pack in packs.items():
        pack["lane"] = get_lane(n)
        ins = predict(model, pack)
        for h in HURDLES:
            for s in STYLES:
                run_config(pack, *oof[n], h, s, oof_out)
                run_config(pack, *ins, h, s, ins_out)
        pack["lane"] = None
    configs = {c: {"family": FAMILY, "hurdle": float(c[1:5]), "style": c.split("_")[1],
                   "is": RC.summarize(oof_out[c]["rows"], dict(oof_out[c]["skips"])),
                   "is_insample": RC.summarize(ins_out[c]["rows"], dict(ins_out[c]["skips"]))} for c in names}
    return configs, oof_out


def run_is() -> dict:
    T.refuse_if_spent(FAMILY, OUT)
    pin = T.pin_tape()
    closes = T.window_closes()
    if len(closes) != T.MIN_WINDOWS:
        raise RuntimeError(f"expected {T.MIN_WINDOWS} windows on the pinned tape, found {len(closes)}")
    cut = T.cut_of(closes)
    closes_is = [c for c in closes if c < cut]
    fits = {n: fit_scale(n, cut) for n in RC.LANES}
    packs = {}
    for n in RC.LANES:
        lane, micro = load_lane(n)
        packs[n] = collect(lane, micro, fits[n], cut, "is")
        packs[n]["lane"] = None                      # let the lane's arrays go; it is reloaded to price trades
        del lane, micro
    model, oof, cv = fit_model(packs, closes_is, cut)
    configs, oof_out = score_is(packs, lambda n: load_lane(n)[0], model, oof)
    champion = C.pick_champion(configs)
    champion["is_insample"] = configs[champion["config"]]["is_insample"]
    C.atomic_text(IS_TRADES, "".join(json.dumps(r, default=RC._json_default) + "\n" for r in oof_out[champion["config"]]["rows"]))
    mpm = float(np.mean(np.concatenate([np.isfinite(p["mpm"]) for p in packs.values()])))
    obj = {"family": FAMILY, "split": "first 60% / last 40% of distinct window closes, all four lanes", "cut": cut,
           "windows": len(closes), "is_windows": len(closes_is), "oos_windows": len(closes) - len(closes_is),
           "frozen_input_sha256": pin["sha256"], "student_t_scale": fits, "model": model, "cv_logloss_by_lambda": cv,
           "market_ask_logloss": market_logloss(packs),
           "is_scoring": f"out-of-fold ({FOLDS} close-time blocks, +-1 window purge; a block's model also trains on later IS windows, never on OOS); the all-IS refit is in is_insample",
           "microprice_finite_fraction_is": mpm,
           "sampling": {n: {"windows": p["windows"], "rows": int(len(p["t"])), "skips": dict(p["skips"])} for n, p in packs.items()},
           "configs": configs, "champion": champion, "oos_scored": None}
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
        out, coverage = new_out([config]), []
        for n in RC.LANES:
            lane, micro = load_lane(n)
            pack = collect(lane, micro, obj["student_t_scale"][n], obj["cut"], "oos")
            run_config(pack, *predict(obj["model"], pack), cfg["hurdle"], cfg["style"], out)
            coverage.append(np.isfinite(pack["mpm"]))
            del lane, micro, pack
        obj["microprice_finite_fraction_oos"] = float(np.mean(np.concatenate(coverage)))
        C.atomic_text(OOS_TRADES, "".join(json.dumps(r, default=RC._json_default) + "\n" for r in out[config]["rows"]))
        return out[config]["rows"], dict(out[config]["skips"]), RC.fit_payoff_model(is_rows)

    C.score_with_receipt(OUT, obj, FAMILY, {"config": config, "hurdle": cfg["hurdle"], "style": cfg["style"],
                                            "lambda": obj["model"]["lambda"], "frozen_input_sha256": pin["sha256"]}, scorer)
    C.save_receipt(OUT, T.clean(obj), strict=True)
    return obj


def hand_off(obj: dict) -> None:
    best, score, m = obj["champion"]["is"], obj["oos_scored"], obj["model"]
    state = {"status": "OOS scored" if score else ("IS eligible; OOS pending" if obj["champion"]["is_eligible"]
                                                   else "IS fail; OOS unspent"),
             "champion": obj["champion"]["config"], "is": best, "is_insample": obj["champion"]["is_insample"],
             "is_eligible": obj["champion"]["is_eligible"], "lambda": m["lambda"],
             "oos_spent": score is not None, "oos": score["oos"] if score else None,
             "payoff_ece": score["payoff_ece"] if score else None, "gate": score["gate"] if score else None,
             "result_artifact": str(OUT.relative_to(ROOT)), "note": "Research only; no live promotion or trading."}
    final = score is not None or not obj["champion"]["is_eligible"]
    lines = [f"- {obj['windows']} windows ({obj['is_windows']} IS / {obj['oos_windows']} OOS), cut `{obj['cut']}`; ridge logistic "
             f"(lambda {m['lambda']:g}, {m['rows']} rows) on {', '.join(FEATURES)}; microprice finite on "
             f"{obj['microprice_finite_fraction_is']:.1%} of IS rows, so that feature is inert in the fit.",
             f"- Blocked-CV log-loss {min(obj['cv_logloss_by_lambda'].values()):.4f} against {obj['market_ask_logloss']:.4f} "
             f"for the side's own taker ask used as its probability (chance 0.6931).",
             f"- IS champion `{obj['champion']['config']}` (out-of-fold): {T.scorecard(best)}; eligible: {obj['champion']['is_eligible']}. "
             f"In-sample refit: {T.scorecard(obj['champion']['is_insample'])}."]
    lines.append(f"- One sealed OOS look: {T.scorecard(score['oos'])}; payoff-ECE {score['payoff_ece']}; "
                 f"gate pass: {score['gate']['pass']}." if score else "- IS bar missed; OOS left sealed, no ledger entry.")
    lines.append(T.SHARED_TAPE_NOTE)
    lines.append(f"- Receipt: `{OUT.relative_to(ROOT)}`; no live promotion.")
    T.update_handoff(KEY, state, "### Track 4A — empirical settlement calibrator" if final else None, lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-oos", action="store_true")
    args = ap.parse_args()
    obj = score_oos() if args.score_oos else run_is()
    hand_off(obj)
    print(json.dumps({"champion": obj["champion"]["config"], "eligible": obj["champion"]["is_eligible"],
                      "lambda": obj["model"]["lambda"], "cv": obj["cv_logloss_by_lambda"],
                      "is": obj["champion"]["is"], "is_insample": obj["champion"]["is_insample"],
                      "oos": (obj["oos_scored"] or {}).get("oos")}, default=RC._json_default, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
