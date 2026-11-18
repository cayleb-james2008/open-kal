#!/usr/bin/env python3
"""Track 6 `gbdt_l2_meta`: a small gradient-boosted meta-model on the true-L2 + LayaGate tape.

Candidates are every (decision second, side) on a 2s grid, secs_left 90..720, whose taker ask is in
[$0.25, $0.62] (NO ask = 1 - yes_bid). Three HistGradientBoosting models are fitted on the IS windows only:
  win       P(side_won)
  net_<x>   P(sim_core net > 0) for exit mode x in {hold, tp5_stop7}, labelled by pricing the candidate
            through `sim_core.run_trade` (T+1s taker entry, 50 contracts, taker fee, $0.62 cap)
Features, all causal at the decision second t and signed to the side where they carry direction:
  touch imbalance, microprice - mid, LayaGate confidence, Student-t fair value and z (IS-fitted nu, s),
  BTC and ETH index impulses over 1s / 3s / 10s, spread, secs_left, entry ask.
Dropped, with the reason in the receipt: the 3-level depth ratio (the tape carries depth levels on ~0.00002% of
rows) and `gate_allow` (0 on every available tick).

Trades fire when the win model's `p - ask >= hurdle`, or the net model's `p >= threshold`, and are held to
settlement or exit on a +5c maker take-profit / 7c confirmed stop; every one is priced by `sim_core.run_trade`.
IS trades are scored on out-of-fold probabilities (blocked CV over IS windows, one window purged each side).

Run with the system python3 (scikit-learn lives there, not in .venv). Phases: default = IS grid, receipt and the
frozen models; --score-oos = the one sealed OOS look (RND_OOS_UNSEAL=lead).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import pickle
import sys
import time
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
from scripts import track2d_layagate_l2 as D  # noqa: E402
from scripts import track4_common as T  # noqa: E402
from scripts import track4a_empirical_settle as A  # noqa: E402

FAMILY = "gbdt_l2_meta"
KEY = "track6_gbdt_l2_meta"
TAPE = ROOT / "data" / "track6-tape"
OUT = ROOT / "data" / "track6-gbdt-l2-meta.json"
IS_TRADES = ROOT / "data" / "track6-gbdt-l2-meta-is-trades.jsonl"
OOS_TRADES = ROOT / "data" / "track6-gbdt-l2-meta-oos-trades.jsonl"
MODELS = ROOT / "data" / "track6-models.pkl"

SECS = np.arange(720.0, 89.0, -2.0)
ASK_LO_M, ASK_HI_M = 250, 620                       # taker ask band in $0.001 units
QTY = S.MAX_QTY
EXITS = {"hold": {}, "tp5_stop7": {"tp_c": Decimal("0.05"), "stop_c": Decimal("0.07")}}
HURDLES, NET_THRESHOLDS = (0.03, 0.06, 0.09), (0.55, 0.60, 0.65)
COOLDOWN, PER_WINDOW = 60.0, 2
MIN_IS_TRADES, MIN_PF, MIN_LANES = 15, 1.25, 2      # the brief's IS bar
BAR = {"trades": 12, "pf": 1.25, "lanes": 2}        # the brief's rolling-edge bar on OOS
MAX_L2_AGE_S = 2.0
FEATURES = ["touch_imbalance", "microprice_minus_mid", "gate_confidence", "student_t_p", "student_t_z",
            "btc_1s", "btc_3s", "btc_10s", "eth_1s", "eth_3s", "eth_10s", "spread", "secs_left", "entry_ask"]
HGB = dict(max_iter=80, learning_rate=0.05, max_depth=3, min_samples_leaf=400, l2_regularization=10.0,
           early_stopping=False, random_state=0)
ROLLING = {"spec": (24.0, 12.0), "scaled": (3.0, 1.5)}   # (train hours, test hours); the 24/12 spec needs >= 36h of L2


def config_name(kind: str, level: float, exit_mode: str) -> str:
    return f"{kind}_{level:.2f}_{exit_mode}"


def grid() -> list[tuple[str, float, str]]:
    return [("win", h, e) for h, e in itertools.product(HURDLES, EXITS)] + \
           [("net", t, e) for t, e in itertools.product(NET_THRESHOLDS, EXITS)]


# ------------------------------------------------------------------ sampling
def last_vals(its: np.ndarray, v: np.ndarray, times: np.ndarray) -> np.ndarray:
    """The last index print at or before each time, NaN when it is older than `INDEX_TOL`."""
    if len(its) == 0:
        return np.full(len(times), np.nan)
    j = np.searchsorted(its, times, "right") - 1
    jj = np.maximum(j, 0)
    return np.where((j >= 0) & (times - its[jj] <= KF.INDEX_TOL), v[jj], np.nan)


def impulses(its: np.ndarray, v: np.ndarray, times: np.ndarray) -> list[np.ndarray]:
    now = last_vals(its, v, times)
    return [np.log(now / last_vals(its, v, times - w)) for w in (1.0, 3.0, 10.0)]


def collect(lane, ex: dict, gates: dict, btc: tuple, eth: tuple, fit: dict, closes: set, cut: float, split: str) -> dict:
    """Decision rows for each window of `split`: every feature is read at or before the row's own second."""
    its, v = KF.clean_index(lane.index, before=cut if split == "is" else None)
    tv, cum, cnt = KF.sigma_prefix(its, v)
    table = KF.t_cdf_table(fit["nu"])
    cols = ("tid", "t", "secs", "close", "y", "bid", "ask", "ask_y", "ask_n", "imb", "mpm", "gconf", "pty", "z",
            "b1", "b3", "b10", "e1", "e3", "e10")
    rows = {c: [] for c in cols}
    skips: Counter = Counter()
    windows = 0
    for tid, name in enumerate(lane.tickers):
        m = lane.meta.get(name)
        if m is None or m["close"] not in closes or RC.slice_of(m["close"], cut) != split:
            continue
        windows += 1
        times = m["close"] - SECS
        a0, a1 = int(lane.book_bounds[tid]), int(lane.book_bounds[tid + 1])
        if a1 == a0:                                  # a ticker with no book rows has nothing to quote
            skips["no_quote"] += len(times)
            continue
        bts = lane.ts[a0:a1]
        j = np.searchsorted(bts, times, "right") - 1
        jj = np.maximum(j, 0)
        b, a = lane.bid[a0 + jj], lane.ask[a0 + jj]
        keep = (j >= 0) & np.isfinite(b) & np.isfinite(a) & (b > 0) & (a < 1) & (b < a - S.EPS)
        skips["no_quote"] += int((~keep).sum())
        bs, az, mp = ex["bid_sz"][a0 + jj], ex["ask_sz"][a0 + jj], ex["micro"][a0 + jj]
        l2 = (times - bts[jj] <= MAX_L2_AGE_S) & (bs > 0) & (az > 0) & (mp > 0) & (mp < 1)
        skips["no_l2"] += int((keep & ~l2).sum())
        keep &= l2
        avail, _, conf = D.gate_at(gates.get(tid), times)
        skips["no_gate"] += int((keep & ~avail).sum())
        keep &= avail
        spot = last_vals(its, v, times)
        imp = impulses(*btc, times) + impulses(*eth, times)
        fin = np.isfinite(spot) & (spot > 0) & np.isfinite(conf) & np.all([np.isfinite(x) for x in imp], axis=0)
        skips["no_spot_or_impulse"] += int((keep & ~fin).sum())
        keep &= fin
        if not keep.any():
            continue
        tau = np.maximum(SECS - KF.TAU_OFFSET, 1.0)
        vol = np.asarray(KF.sigma_at(times, tv, cum, cnt)) * np.sqrt(tau)
        z = np.log(np.where(keep, spot, 1.0) / m["strike"]) / (vol * fit["s"])
        am, bm = np.rint(a * 1000), np.rint(b * 1000)
        vals = (np.full(len(times), tid), times, SECS, np.full(len(times), m["close"]), np.full(len(times), m["y"]),
                bm / 1000, am / 1000, am / 1000, (1000 - bm) / 1000, (bs - az) / (bs + az), mp - (a + b) / 2, conf,
                np.interp(z, KF.CDF_X, table), z, *imp)
        for c, x in zip(cols, vals):
            rows[c].append(np.asarray(x, dtype=np.float64)[keep])
    pack = {c: (np.concatenate(x) if x else np.zeros(0)) for c, x in rows.items()}
    pack["tid"] = pack["tid"].astype(np.int64)
    pack.update(lane=None, split=split, windows=windows, skips=skips, net={})
    return pack


def side_x(pack: dict, side: str) -> np.ndarray:
    """Feature matrix for one side; direction-bearing columns flip sign for NO."""
    yes = side == S.YES
    sign = 1.0 if yes else -1.0
    return np.column_stack([sign * pack["imb"], sign * pack["mpm"], pack["gconf"], pack["pty"] if yes else 1 - pack["pty"],
                            sign * pack["z"], *(sign * pack[c] for c in ("b1", "b3", "b10", "e1", "e3", "e10")),
                            pack["ask"] - pack["bid"], pack["secs"], pack["ask_y"] if yes else pack["ask_n"]])


def ask_of(pack: dict, side: str) -> np.ndarray:
    return pack["ask_y"] if side == S.YES else pack["ask_n"]


def in_domain(pack: dict, side: str) -> np.ndarray:
    m = np.rint(ask_of(pack, side) * 1000)
    return (m >= ASK_LO_M) & (m <= ASK_HI_M)


def side_label(pack: dict, side: str) -> np.ndarray:
    return pack["y"] if side == S.YES else 1.0 - pack["y"]


def price(lane, pack: dict, i: int, side: str, mode: str):
    """The candidate priced by sim_core: T+1s taker entry on the side's ask, then the exit mode."""
    return S.run_trade(RC.tape(lane, int(pack["tid"][i])), side=side, trigger_ts=float(pack["t"][i]),
                       entry_style=S.TAKER, qty=QTY, cost_lo=Decimal("0.25"), cost_hi=Decimal("0.62"), **EXITS[mode])


def label_pass(lane, pack: dict) -> None:
    """net-positive labels (1/0, NaN where sim_core refuses the trade) for every in-domain candidate and exit mode."""
    for mode in EXITS:
        for side in (S.YES, S.NO):
            out = np.full(len(pack["t"]), np.nan)
            for i in np.flatnonzero(in_domain(pack, side)):
                res = price(lane, pack, i, side, mode)
                if not isinstance(res, S.Skip):
                    out[i] = float(res.pnl > 0)
            pack["net"][(mode, side)] = out


# --------------------------------------------------------------------- models
class Const:
    """Stand-in when a training set has one class or too few rows for boosting."""

    def __init__(self, p: float) -> None:
        self.p = p

    def predict_proba(self, x):
        return np.column_stack([np.full(len(x), 1 - self.p), np.full(len(x), self.p)])


def fit_one(x: np.ndarray, y: np.ndarray):
    if len(y) < 200 or y.min() == y.max():
        return Const(float(y.mean()) if len(y) else 0.5)
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(**HGB).fit(x, y)


def fit_models(packs: dict, rows: dict) -> dict:
    """The three models on the selected rows (`rows[lane]` is a boolean mask over that lane's pack)."""
    sets = {"win": ([], [])} | {f"net_{m}": ([], []) for m in EXITS}
    for n, pack in packs.items():
        for side in (S.YES, S.NO):
            dom = rows[n] & in_domain(pack, side)
            x = side_x(pack, side)
            sets["win"][0].append(x[dom])
            sets["win"][1].append(side_label(pack, side)[dom])
            for m in EXITS:
                lab = pack["net"][(m, side)]
                ok = dom & np.isfinite(lab)
                sets[f"net_{m}"][0].append(x[ok])
                sets[f"net_{m}"][1].append(lab[ok])
    return {k: fit_one(np.concatenate(xs), np.concatenate(ys)) for k, (xs, ys) in sets.items()}


def predict(models: dict, pack: dict, rows: np.ndarray | None = None) -> dict:
    """{(kind, side): probabilities}, NaN outside the candidate domain or the selected rows."""
    out = {}
    for side in (S.YES, S.NO):
        ok = in_domain(pack, side) if rows is None else in_domain(pack, side) & rows
        x = side_x(pack, side)[ok]
        for kind in ("win", *(f"net_{m}" for m in EXITS)):
            p = np.full(len(pack["t"]), np.nan)
            if ok.any():
                p[ok] = models[kind].predict_proba(x)[:, 1]
            out[(kind, side)] = p
    return out


KINDS = ("win", *(f"net_{m}" for m in EXITS))


def held_out_preds(packs: dict, splits: list, fit=None) -> dict:
    """Predictions for each split's held rows from a model that never saw them.

    `splits` is a list of {lane: (train_mask, held_mask)}; rows held out by no split stay NaN."""
    fit = fit or fit_models
    out = {n: {(k, s): np.full(len(p["t"]), np.nan) for k in KINDS for s in (S.YES, S.NO)} for n, p in packs.items()}
    for split in splits:
        models = fit(packs, {n: split[n][0] for n in packs})
        for n, p in packs.items():
            held = split[n][1]
            for key, val in predict(models, p, held).items():
                out[n][key] = np.where(held, val, out[n][key])
    return out


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def brier_vs_ask(packs: dict, preds: dict) -> dict:
    """Brier of the win model and of the side's own ask used as its probability, on the rows the model predicted."""
    p, ask, y = [], [], []
    for n, pack in packs.items():
        for side in (S.YES, S.NO):
            pw = preds[n][("win", side)]
            ok = np.isfinite(pw)
            p.append(pw[ok]), ask.append(ask_of(pack, side)[ok]), y.append(side_label(pack, side)[ok])
    p, ask, y = map(np.concatenate, (p, ask, y))
    return {"rows": int(len(y)), "model": brier(p, y), "ask": brier(ask, y), "base_rate": brier(np.full(len(y), y.mean()), y)}


# ----------------------------------------------------------------- the grid
def triggers(pack: dict, preds: dict, kind: str, level: float, mode: str):
    p_y, p_n = ((preds[("win", S.YES)], preds[("win", S.NO)]) if kind == "win"
                else (preds[(f"net_{mode}", S.YES)], preds[(f"net_{mode}", S.NO)]))
    sy, sn = (p_y - pack["ask_y"], p_n - pack["ask_n"]) if kind == "win" else (p_y, p_n)   # NaN never passes
    oky, okn = sy >= level - 1e-9, sn >= level - 1e-9
    use_y = oky & (~okn | (sy >= sn))
    keep = BS.pick({"tid": pack["tid"], "ts": pack["t"]}, oky | okn, cooldown=COOLDOWN, per_window=PER_WINDOW)
    return [(i, S.YES if use_y[i] else S.NO, float((p_y if use_y[i] else p_n)[i])) for i in keep]


def run_config(lane, pack: dict, preds: dict, kind: str, level: float, mode: str, out: dict) -> None:
    name = config_name(kind, level, mode)
    for i, side, p in triggers(pack, preds, kind, level, mode):
        res = price(lane, pack, i, side, mode)
        if isinstance(res, S.Skip):
            out[name]["skips"][res.reason] += 1
            continue
        yes = side == S.YES
        sign = 1.0 if yes else -1.0
        out[name]["rows"].append(RC.trade_row(
            res, lane=lane.lane, ticker=lane.tickers[int(pack["tid"][i])], close=float(pack["close"][i]), slice_name=pack["split"],
            feat=[p, p - float(ask_of(pack, side)[i]), float(pack["secs"][i]), sign * float(pack["imb"][i]), float(pack["gconf"][i])],
            config=name, family=FAMILY, kind=kind, level=level, exit_mode=mode))


def new_out() -> dict:
    return {config_name(*g): {"rows": [], "skips": Counter()} for g in grid()}


def run_grid(lane, pack: dict, preds: dict, out: dict) -> None:
    for kind, level, mode in grid():
        run_config(lane, pack, preds, kind, level, mode, out)


def summarize_all(out: dict) -> dict:
    return {c: {"kind": c.split("_")[0], "level": float(c.split("_")[1]), "exit_mode": c.split("_", 2)[2],
                "is": RC.summarize(v["rows"], dict(v["skips"]))} for c, v in out.items()}


def build_configs(outs: dict) -> dict:
    """`is` is the out-of-fold scorecard (what selection reads); the refit and rolling ones are shown beside it."""
    configs = summarize_all(outs["oof"])
    for c, v in configs.items():
        for tag, key in (("is_insample", "insample"), ("rolling_scaled", "rolling_scaled")):
            v[tag] = RC.summarize(outs[key][c]["rows"], dict(outs[key][c]["skips"]))
    return configs


def eligible(s: dict) -> bool:
    return s["trades"] >= MIN_IS_TRADES and s["pf"] >= MIN_PF and s["profitable_lanes"] >= MIN_LANES and s["crossed_fills"] == 0


def pick_champion(configs: dict) -> dict:
    """The eligible config with the best net; with none eligible, the best of those that traded, flagged ineligible."""
    el = [c for c in configs if eligible(configs[c]["is"])]
    pool = el or [c for c in configs if configs[c]["is"]["trades"]] or list(configs)
    best = max(pool, key=lambda c: configs[c]["is"]["net"])
    return {"config": best, "is_eligible": bool(el), "is": configs[best]["is"]}


def rolling_edge_bar(oos: dict) -> dict:
    checks = {"oos_trades_at_least_12": oos["trades"] >= BAR["trades"], "oos_net_positive": oos["net"] > 0,
              "oos_pf_at_least_1_25": oos["pf"] >= BAR["pf"], "two_profitable_lanes": oos["profitable_lanes"] >= BAR["lanes"]}
    return {**checks, "pass": all(checks.values())}


def rolling_folds(closes: list[float], train_h: float, test_h: float, *, min_train=4, min_test=2) -> list[tuple[list, list]]:
    """Time-ordered (train closes, test closes) folds; test blocks are disjoint and always after their training span."""
    folds, start = [], closes[0] if closes else 0.0
    while closes and start + (train_h + test_h) * 3600 <= closes[-1] + 900:
        tr = [c for c in closes if start <= c < start + train_h * 3600]
        te = [c for c in closes if start + train_h * 3600 <= c < start + (train_h + test_h) * 3600]
        if len(tr) >= min_train and len(te) >= min_test:
            folds.append((tr, te))
        start += test_h * 3600
    return folds


# ------------------------------------------------------------------- phases
def load_lane(name: str):
    lane = AD.Lane(name)
    with np.load(TAPE / f"book_{name}.npz", allow_pickle=False) as data:
        ex = {"bid_sz": data["yes_bid_sz"], "ask_sz": data["yes_ask_sz"], "micro": data["microprice"]}
        gates = D.load_gates(data)
    return lane, ex, gates


def spot_series(name: str) -> tuple:
    return KF.clean_index(dict(np.load(TAPE / f"index_{AD.INDEX_BY_SERIES[name]}.npz")))


def l2_closes() -> list[float]:
    """Settled BTC windows that have true-L2 rows and at least one available LayaGate tick."""
    meta = json.loads((TAPE / "markets.json").read_text())
    with np.load(TAPE / "book_KXBTC15M.npz", allow_pickle=False) as data:
        names, tid = [str(t) for t in data["tickers"]], data["tid"]
        ok = (data["yes_bid_sz"] > 0) & (data["yes_ask_sz"] > 0) & (data["microprice"] > 0) & (data["microprice"] < 1)
        have_l2 = set(np.unique(tid[ok]).tolist())
        have_gate = set(np.unique(data["tick_tid"][data["gate_available"] == 1]).tolist())
    return sorted({float(meta[n]["close"]) for i, n in enumerate(names)
                   if i in have_l2 and i in have_gate and meta.get(n, {}).get("y") in (0, 1) and meta[n].get("strike")})


def fit_from(its: np.ndarray, v: np.ndarray, cut: float) -> dict:
    """Student-t scale from index prints strictly before cut - 900 (`index_near` would otherwise reach 1.5s past it)."""
    keep = its < cut - KF.FIT_LEAD
    return KF.fit_lane(its[keep], v[keep], cut)


def fit_scale(name: str, cut: float) -> dict:
    return fit_from(*spot_series(name), cut)


def gather(closes: list[float], cut: float, split: str, fits: dict, *, label: bool) -> dict:
    """Per-lane packs for one split; `label` adds the sim_core net labels (IS training only)."""
    btc, eth = spot_series("KXBTC15M"), spot_series("KXETH15M")
    packs = {}
    for n in RC.LANES:
        lane, ex, gates = load_lane(n)
        pack = collect(lane, ex, gates, btc, eth, fits[n], set(closes), cut, split)
        if label:
            label_pass(lane, pack)
        packs[n] = pack
        del lane, ex, gates
    return packs


def score_is(packs: dict, preds_by_set: dict) -> dict:
    """Run the whole grid for each prediction set (one pass per lane); returns {set: {config: output}}."""
    outs = {k: new_out() for k in preds_by_set}
    for n in RC.LANES:
        lane = AD.Lane(n)
        for k, preds in preds_by_set.items():
            run_grid(lane, packs[n], preds[n], outs[k])
        del lane
    return outs


def run_is() -> dict:
    T.refuse_if_spent(FAMILY, OUT)
    pin = T.pin_tape(TAPE)
    import sklearn
    closes = l2_closes()
    cut = closes[int(len(closes) * 0.6)]
    closes_is = [c for c in closes if c < cut]
    fits = {n: fit_scale(n, cut) for n in RC.LANES}
    packs = gather(closes_is, cut, "is", fits, label=True)
    rank_of = {c: i for i, c in enumerate(closes_is)}
    for pack in packs.values():
        pack["rank"] = np.array([rank_of[c] for c in pack["close"]], dtype=np.int64)
        if len(pack["t"]) and float(pack["t"].max()) >= cut - 900.0:
            raise RuntimeError("an IS row sits inside the final 900s before the cut")
    t0 = time.process_time()
    splits = [{n: A.fold_masks(p["rank"], len(closes_is), k) for n, p in packs.items()} for k in range(A.FOLDS)]
    oof = held_out_preds(packs, splits)
    cv_cpu = time.process_time() - t0
    t0 = time.process_time()
    final = fit_models(packs, {n: np.ones(len(p["t"]), bool) for n, p in packs.items()})
    final_cpu = time.process_time() - t0
    ins = {n: predict(final, p) for n, p in packs.items()}
    roll = {}
    for tag, (tr_h, te_h) in ROLLING.items():
        folds = rolling_folds(closes_is, tr_h, te_h)
        splits = [{n: (np.isin(p["close"], tr), np.isin(p["close"], te)) for n, p in packs.items()} for tr, te in folds]
        roll[tag] = (folds, held_out_preds(packs, splits))
    sets = {"oof": oof, "insample": ins, "rolling_scaled": roll["scaled"][1]}
    outs = score_is(packs, sets)
    configs = build_configs(outs)
    champion = pick_champion(configs)
    champion["is_insample"] = configs[champion["config"]]["is_insample"]
    C.atomic_text(IS_TRADES, "".join(json.dumps(r, default=RC._json_default) + "\n" for r in outs["oof"][champion["config"]]["rows"]))
    blob = pickle.dumps({"models": final, "features": FEATURES, "hgb": HGB})
    MODELS.write_bytes(blob)                          # its sha256 is in the receipt; OOS refuses a different file
    obj = {"family": FAMILY, "split": "chronological 60% IS / 40% OOS of the L2-and-gate-enabled window closes", "cut": cut,
           "l2_closes": closes, "is_windows": len(closes_is), "oos_windows": len(closes) - len(closes_is),
           "frozen_input_sha256": pin["sha256"], "student_t_scale": fits, "features": FEATURES,
           "dropped_features": {"depth3_ratio": "depth_bid_sz/depth_ask_sz are finite on ~0.00002% of true-L2 rows",
                                "gate_allow": "0 on every available LayaGate tick"},
           "model": {"hgb": HGB, "models_sha256": hashlib.sha256(blob).hexdigest(), "sklearn": sklearn.__version__,
                     "python": sys.version.split()[0], "cv_cpu_s": round(cv_cpu, 2), "final_fit_cpu_s": round(final_cpu, 2)},
           "brier_is_oof": brier_vs_ask(packs, oof), "brier_is_insample": brier_vs_ask(packs, ins),
           "rolling": {"spec_24h_12h": {"is_span_hours": (closes_is[-1] - closes_is[0]) / 3600, "folds": len(roll["spec"][0])},
                       "scaled_3h_1p5h": {"folds": len(roll["scaled"][0]), "note": "scaled 2:1 substitute, IS windows only"}},
           "sampling": {n: {"windows": p["windows"], "rows": int(len(p["t"])), "skips": dict(p["skips"])} for n, p in packs.items()},
           "configs": configs, "champion": champion, "oos_scored": None}
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
    blob = MODELS.read_bytes()
    if hashlib.sha256(blob).hexdigest() != obj["model"]["models_sha256"]:
        raise RuntimeError("the frozen models changed since the IS receipt")
    models = pickle.loads(blob)["models"]
    cfg = obj["champion"]["config"]
    kind, level, mode = cfg.split("_")[0], float(cfg.split("_")[1]), cfg.split("_", 2)[2]
    closes = [c for c in obj["l2_closes"] if c >= obj["cut"]]
    is_rows = [json.loads(line) for line in IS_TRADES.read_text().splitlines() if line.strip()]

    def scorer():
        packs = gather(closes, obj["cut"], "oos", obj["student_t_scale"], label=False)
        preds = {n: predict(models, p) for n, p in packs.items()}
        out = new_out()
        for n in RC.LANES:
            lane = AD.Lane(n)
            run_config(lane, packs[n], preds[n], kind, level, mode, out)
            del lane
        obj["brier_oos"] = brier_vs_ask(packs, preds)
        obj["oos_windows_scored"] = len(closes)
        C.atomic_text(OOS_TRADES, "".join(json.dumps(r, default=RC._json_default) + "\n" for r in out[cfg]["rows"]))
        return out[cfg]["rows"], dict(out[cfg]["skips"]), RC.fit_payoff_model(is_rows)

    C.score_with_receipt(OUT, obj, FAMILY, {"config": cfg, "models_sha256": obj["model"]["models_sha256"],
                                            "frozen_input_sha256": pin["sha256"]}, scorer)
    obj["rolling_edge_bar"] = rolling_edge_bar(obj["oos_scored"]["oos"])
    C.save_receipt(OUT, T.clean(obj), strict=True)
    return obj


def rolling_text(obj: dict) -> str:
    """The scaled rolling fold beside the out-of-fold IS scoring that selection uses, config by config."""
    cfgs, folds = obj["configs"], obj["rolling"]["scaled_3h_1p5h"]["folds"]
    traded = [v["rolling_scaled"]["net"] for v in cfgs.values() if v["rolling_scaled"]["trades"]]
    passing = [c for c, v in cfgs.items() if rolling_edge_bar(v["rolling_scaled"])["pass"]]
    vs = "; ".join(f"`{c}` rolling ${cfgs[c]['rolling_scaled']['net']:+.2f}, out-of-fold ${cfgs[c]['is']['net']:+.2f} (PF {cfgs[c]['is']['pf']:.2f})"
                   for c in passing)
    return (f"Scaled rolling ({folds} fold, 3h train / 1.5h test, IS windows only, not used for selection): net "
            f"${min(traded, default=0):+.2f} to ${max(traded, default=0):+.2f} across {len(traded)} configs that traded; "
            f"{len(passing)} meet the rolling-edge bar" + (f": {vs}. One fold is noise-sized against 12 configs." if passing else "."))


def hand_off(obj: dict) -> None:
    best, score, eligible_ = obj["champion"]["is"], obj["oos_scored"], obj["champion"]["is_eligible"]
    state = {"status": "OOS scored" if score else ("IS eligible; OOS pending" if eligible_ else "IS fail; OOS unspent"),
             "champion": obj["champion"]["config"], "is": best, "is_eligible": eligible_, "oos_spent": score is not None,
             "oos": score["oos"] if score else None, "payoff_ece": score["payoff_ece"] if score else None,
             "strict_gate": score["gate"] if score else None, "rolling_edge_bar": obj.get("rolling_edge_bar"),
             "rolling_scaled_bar_pass": [c for c, v in obj["configs"].items() if rolling_edge_bar(v["rolling_scaled"])["pass"]],
             "brier_is_oof": obj["brier_is_oof"], "brier_oos": obj.get("brier_oos"),
             "result_artifact": str(OUT.relative_to(ROOT)), "note": "Research only; no live promotion or trading."}
    b = obj["brier_is_oof"]
    lines = [f"- {len(obj['l2_closes'])} L2-and-gate windows ({obj['is_windows']} IS / {obj['oos_windows']} OOS), cut `{obj['cut']}`; "
             f"HistGradientBoosting on {len(FEATURES)} causal features (depth3 ratio and gate_allow unusable: see receipt); "
             f"IS trades scored out-of-fold.",
             f"- IS Brier, win model {b['model']:.4f} vs side's own ask {b['ask']:.4f} vs base rate {b['base_rate']:.4f} ({b['rows']} rows).",
             (f"- IS champion `{obj['champion']['config']}`: {T.scorecard(best)}." if eligible_ else
              f"- No IS config met the bar (>= {MIN_IS_TRADES} trades, PF >= {MIN_PF}, >= {MIN_LANES} lanes); best by net "
              f"`{obj['champion']['config']}`: {T.scorecard(best)}."),
             f"- Rolling 24h-train/12h-test: {obj['rolling']['spec_24h_12h']['folds']} folds (IS span "
             f"{obj['rolling']['spec_24h_12h']['is_span_hours']:.1f}h, below the 36h needed). {rolling_text(obj)}",
             f"- Refit on all IS windows and scored in-sample, the best config nets "
             f"${max(v['is_insample']['net'] for v in obj['configs'].values()):+.2f}; out-of-fold the best PF is "
             f"{max(v['is']['pf'] for v in obj['configs'].values()):.2f}: the boosted models memorise the IS windows."]
    if score:
        o, bo = score["oos"], obj["brier_oos"]
        lines += [f"- One sealed OOS look: {T.scorecard(o)}; payoff-ECE {score['payoff_ece']}. Rolling-edge bar pass: "
                  f"{obj['rolling_edge_bar']['pass']}; strict repo gate pass: {score['gate']['pass']}.",
                  f"- OOS Brier, win model {bo['model']:.4f} vs side's own ask {bo['ask']:.4f} ({bo['rows']} rows)."]
    else:
        lines.append("- IS bar missed; OOS left sealed, no ledger entry.")
    lines += [T.SHARED_TAPE_NOTE, f"- Receipt: `{OUT.relative_to(ROOT)}`; no live promotion."]
    T.update_handoff(KEY, state, "### Track 6 — gradient-boosted L2 + LayaGate meta-model" if score or not eligible_ else None, lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-oos", action="store_true")
    args = ap.parse_args()
    obj = score_oos() if args.score_oos else run_is()
    hand_off(obj)
    print(json.dumps({"champion": obj["champion"]["config"], "eligible": obj["champion"]["is_eligible"], "is": obj["champion"]["is"],
                      "brier_is_oof": obj["brier_is_oof"], "oos": (obj["oos_scored"] or {}).get("oos")},
                     default=RC._json_default, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
