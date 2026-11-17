#!/usr/bin/env python3
"""Track 3 (kurtosis_fv_snipe): Student-t fair value vs a Gaussian control, IS only by default.

Fit (IS index only): per lane, a scaled Student-t fitted to 15-minute index log-returns
scaled by trailing realised vol. Trade: taker entries the fair-value model prices cheap,
every fill through openkal.sim_core.run_trade.

  .venv/bin/python scripts/kurtosis_fv_snipe.py --lanes KXSOL15M     # dev, one lane
  flock data/.heavy.lock ... scripts/kurtosis_fv_snipe.py            # full IS run (4 lanes)
  scripts/kurtosis_fv_snipe.py --score-oos                           # sealed, lead only, never run here
"""
from __future__ import annotations

import argparse
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

from openkal import features  # noqa: E402
from openkal import sim_core as S  # noqa: E402
from scripts import bounded_sweep as BS  # noqa: E402
from scripts import rnd_common as RC  # noqa: E402

OUT = ROOT / "data" / "kurtosis-fv-is.json"
TRADES = ROOT / "data" / "kurtosis-fv-is-trades.jsonl"

FAMILY = {"fat": "kurtosis_fv_snipe", "gauss": "gauss_fv_control"}
MODELS = ("fat", "gauss")
HURDLES = (0.06, 0.09, 0.12)
EXITS = {"tp6": Decimal("0.06"), "tp8": Decimal("0.08"), "settle": None}
FEATURES = ["p_side", "p_side_minus_cost", "secs", "abs_x"]

COST_LO, COST_HI = 0.22, 0.58
COST_LO_D, COST_HI_D = Decimal("0.22"), Decimal("0.58")
SIG_WINDOW, SIG_FLOOR, SIG_MIN_N = 150.0, 5.0e-5, 5
INDEX_TOL = 1.5
FIT_LEAD = 900.0
FIT_STEP = 15.0
HORIZONS = (60, 120, 240, 480, 780)
NU_GRID = (2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0)
S_GRID = np.linspace(0.3, 1.6, 131)
CDF_X = np.linspace(-80.0, 80.0, 320001)          # step 0.0005
SECS_GRID = np.arange(780.0, 119.0, -5.0)         # 780..120, ascending in time
TAU_OFFSET = 40.0                                 # 60s average: variance uses secs - 40
COOLDOWN, PER_WINDOW = 90.0, 3
MIN_TRADES, MIN_PF = 30, 1.20
CHUNK = 50_000


def config_name(model: str, hurdle: float, exit_name: str) -> str:
    return f"{model}_h{hurdle:.2f}_{exit_name}"


# ----------------------------------------------------------------- split and cut
def cut_from_cache() -> float:
    """The global cut from the frozen cache alone, so a one-lane dev run uses the same cut.

    Same rule as rnd_common.global_cut over all four lanes: markets with a strike, all closes.
    """
    meta = json.loads((RC.CACHE / "markets.json").read_text())
    closes: set[float] = set()
    for ln in RC.LANES:
        for name in np.load(RC.CACHE / f"book_{ln}.npz", allow_pickle=False)["tickers"]:
            m = meta.get(str(name))
            if m is None or not m.get("strike"):
                continue
            closes.add(float(m["close"]))
    cs = sorted(closes)
    return cs[int(len(cs) * 0.6)]


# ------------------------------------------------------------------ index vol
def clean_index(index: dict, before: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    ts = np.asarray(index["ts"], dtype=np.float64)
    v = np.asarray(index["value"], dtype=np.float64)
    good = np.isfinite(ts) & np.isfinite(v) & (v > 0)
    ts, v = ts[good], v[good]
    if before is not None:
        keep = ts < before
        ts, v = ts[keep], v[keep]
    if len(ts) > 1 and np.any(np.diff(ts) < 0):
        raise ValueError("index tape is not time-sorted")
    return ts, v


def sigma_prefix(its: np.ndarray, v: np.ndarray):
    """Cumulative sums for the trailing estimator of alpha_search.trailing_sigma.

    Pair j (points j and j+1, 1e-3 <= dt <= 5s) contributes log(v1/v0)^2 / dt, stamped at
    the later point. A query at t only reads pairs stamped at or before t.
    """
    dt = np.diff(its)
    ok = (dt >= 1e-3) & (dt <= 5.0)
    lr = np.log(v[1:] / v[:-1])
    rate = np.where(ok, lr * lr / np.maximum(dt, 1e-3), 0.0)
    C = np.concatenate(([0.0], np.cumsum(rate)))
    N = np.concatenate(([0.0], np.cumsum(ok.astype(np.float64))))
    return its[1:], C, N


def sigma_at(tq, tv: np.ndarray, C: np.ndarray, N: np.ndarray) -> np.ndarray:
    """Trailing per-root-second sigma at time(s) tq, floored at SIG_FLOOR."""
    tq = np.asarray(tq, dtype=np.float64)
    hi = np.searchsorted(tv, tq, "right")                 # pairs stamped <= t
    lo = np.searchsorted(tv, tq - SIG_WINDOW, "right")    # ... and > t - 150
    cnt = N[hi] - N[lo]
    tot = C[hi] - C[lo]
    est = np.where(cnt >= SIG_MIN_N, tot / np.maximum(cnt, 1.0), 0.0)
    return np.maximum(np.sqrt(est), SIG_FLOOR)


def index_near(its: np.ndarray, v: np.ndarray, tq) -> np.ndarray:
    """Nearest index print within INDEX_TOL seconds (fit only), NaN otherwise."""
    tq = np.asarray(tq, dtype=np.float64)
    j = np.searchsorted(its, tq, "left")
    jm = np.clip(j - 1, 0, len(its) - 1)
    jp = np.clip(j, 0, len(its) - 1)
    dm = np.abs(its[jm] - tq)
    dp = np.abs(its[jp] - tq)
    use_m = dm <= dp
    out = np.where(use_m, v[jm], v[jp])
    gap = np.where(use_m, dm, dp)
    return np.where(gap <= INDEX_TOL, out, np.nan)


def index_last(its: np.ndarray, v: np.ndarray, t: float) -> float:
    """Causal spot: the last index print at or before t, if it is at most INDEX_TOL old."""
    j = int(np.searchsorted(its, t, "right")) - 1
    if j < 0 or t - float(its[j]) > INDEX_TOL:
        return float("nan")
    return float(v[j])


# ------------------------------------------------------------- Student-t
def t_cdf_table(nu: float) -> np.ndarray:
    """Standard t CDF on CDF_X: cumulative trapezoid of the pdf, normalised to end at 1."""
    h = CDF_X[1] - CDF_X[0]
    const = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi)
    pdf = np.exp(const - (nu + 1) / 2 * np.log1p(CDF_X * CDF_X / nu))
    cum = np.concatenate(([0.0], np.cumsum((pdf[1:] + pdf[:-1]) * (0.5 * h))))
    return cum / cum[-1]


def cdf_check() -> dict:
    worst0 = max(abs(float(np.interp(0.0, CDF_X, t_cdf_table(nu))) - 0.5) for nu in NU_GRID)
    t100 = float(np.interp(1.96, CDF_X, t_cdf_table(100.0)))
    ok = worst0 <= 1e-6 and abs(t100 - 0.975) <= 2e-3
    return {"cdf0_worst_abs_err_over_nu_grid": worst0, "nu100_cdf_at_1_96": t100,
            "target_0_975": 0.975, "tolerance_cdf0": 1e-6, "tolerance_1_96": 2e-3, "pass": ok}


def fit_student_t(z: np.ndarray) -> tuple[float, float, float]:
    """Scaled Student-t MLE (nu, s) on a grid; returns (nu, s, loglik)."""
    n = len(z)
    best = None
    for nu in NU_GRID:
        c = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi)
        acc = np.zeros(len(S_GRID))
        for k in range(0, n, CHUNK):
            zc = z[k:k + CHUNK]
            acc += np.log1p((zc[:, None] / S_GRID[None, :]) ** 2 / nu).sum(axis=0)
        ll = n * c - n * np.log(S_GRID) - (nu + 1) / 2 * acc
        j = int(np.argmax(ll))
        if best is None or ll[j] > best[2]:
            best = (float(nu), float(S_GRID[j]), float(ll[j]))
    return best


def fit_lane(its: np.ndarray, v: np.ndarray, cut: float) -> dict:
    """Fit one lane on index points before cut - 900 only (no price inside or after OOS)."""
    tv, C, N = sigma_prefix(its, v)
    end = cut - FIT_LEAD
    grid = np.arange(float(its[0]), end, FIT_STEP)
    zs = []
    for tau in HORIZONS:
        t = grid[grid + tau < end]
        v0 = index_near(its, v, t)
        v1 = index_near(its, v, t + tau)
        sig = sigma_at(t, tv, C, N)
        z = (np.log(v1) - np.log(v0)) / (sig * math.sqrt(tau))
        zs.append(z[np.isfinite(z)])
    z = np.concatenate(zs)
    nu, s, ll_t = fit_student_t(z)
    sg = float(np.std(z))
    ll_g = float(np.sum(-0.5 * math.log(2 * math.pi * sg * sg) - z * z / (2 * sg * sg)))
    m = float(z.mean())
    var = float(((z - m) ** 2).mean())
    kurt = float(((z - m) ** 4).mean() / (var * var) - 3.0)
    return {"nu": nu, "s": s, "s_g": sg, "n": int(len(z)), "ll_t": ll_t, "ll_gauss": ll_g,
            "excess_kurtosis": kurt, "index_points": int(len(its))}


# ------------------------------------------------------------- candidates
def collect(lane, cut: float, params: dict, split: str) -> dict:
    """Every valid 5s sample (secs 120..780) of the tickers in one split, with both models."""
    before = cut if split == "is" else None   # IS never reads an index print at or after cut
    its, v = clean_index(lane.index, before=before)
    tv, C, N = sigma_prefix(its, v)
    s_fat, s_g = params["s"], params["s_g"]
    table = t_cdf_table(params["nu"])
    skips: Counter = Counter()
    tid_l, t_l, secs_l, close_l, x_l, cy_l, cn_l = [], [], [], [], [], [], []
    p_l = {m: [] for m in MODELS}
    pairs: dict[str, list[tuple[float, int]]] = {m: [] for m in MODELS}
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
            try:
                b, a = tape.quote(t)
            except S.CrossedBook:
                skips["quote_crossed"] += 1
                continue
            except S.NoQuote:
                skips["quote_no_quote"] += 1
                continue
            spot = index_last(its, v, t)
            if not math.isfinite(spot):
                skips["no_spot"] += 1
                continue
            sig = float(sigma_at(t, tv, C, N))
            tau_eff = max(secs - TAU_OFFSET, 1.0)
            x = math.log(strike / spot) / (sig * math.sqrt(tau_eff))
            p_fat = 1.0 - float(np.interp(x / s_fat, CDF_X, table))
            p_gauss = 1.0 - 0.5 * (1.0 + math.erf(x / s_g / math.sqrt(2.0)))
            cy = float(S.entry_price(S.YES, S.TAKER, b, a))
            cn = float(S.entry_price(S.NO, S.TAKER, b, a))
            tid_l.append(tid); t_l.append(t); secs_l.append(secs); close_l.append(close)
            x_l.append(x); cy_l.append(cy); cn_l.append(cn)
            p_l["fat"].append(p_fat); p_l["gauss"].append(p_gauss)
            for model, p in (("fat", p_fat), ("gauss", p_gauss)):
                for side, pside, c in ((S.YES, p, cy), (S.NO, 1.0 - p, cn)):
                    if COST_LO <= c <= COST_HI:
                        pairs[model].append((pside, int(tape.side_won(side))))
    return {
        "lane": lane, "split": split, "tickers": n_tickers,
        "tid": np.asarray(tid_l, dtype=np.int64), "t": np.asarray(t_l),
        "secs": np.asarray(secs_l), "close": np.asarray(close_l), "x": np.asarray(x_l),
        "cost_yes": np.asarray(cy_l), "cost_no": np.asarray(cn_l),
        "p": {m: np.asarray(p_l[m]) for m in MODELS}, "pairs": pairs, "skips": skips,
    }


def find_triggers(pack: dict, model: str, hurdle: float) -> list[tuple[int, str]]:
    """Trigger rows after the per-ticker dedupe (bounded_sweep.pick: 90s apart, max 3)."""
    p = pack["p"][model]
    cy, cn = pack["cost_yes"], pack["cost_no"]
    ey = p - cy
    en = (1.0 - p) - cn
    oky = (cy >= COST_LO) & (cy <= COST_HI) & (ey >= hurdle)
    okn = (cn >= COST_LO) & (cn <= COST_HI) & (en >= hurdle)
    use_y = oky & (~okn | (ey >= en))
    mask = oky | okn
    keep = BS.pick({"tid": pack["tid"], "ts": pack["t"]}, mask,
                   cooldown=COOLDOWN, per_window=PER_WINDOW)
    return [(i, S.YES if use_y[i] else S.NO) for i in keep]


def run_triggers(pack: dict, triggers, model: str, hurdle: float, out: dict,
                 exits: dict) -> None:
    lane = pack["lane"]
    for i, side in triggers:
        tid = int(pack["tid"][i])
        t = float(pack["t"][i])
        name = lane.tickers[tid]
        p_yes = float(pack["p"][model][i])
        p_side = p_yes if side == S.YES else 1.0 - p_yes
        cost_now = float(pack["cost_yes" if side == S.YES else "cost_no"][i])
        feat = [p_side, p_side - cost_now, float(pack["secs"][i]), abs(float(pack["x"][i]))]
        cost_hi = min(COST_HI_D, S.px(p_side - hurdle))
        tape = RC.tape(lane, tid)
        try:
            bq, aq = tape.quote(t + S.LAG_S)
            qty = S.stake_qty(S.entry_price(side, S.TAKER, bq, aq), RC.STAKE)
        except (S.CrossedBook, S.NoQuote):
            qty = S.MAX_QTY   # run_trade reports the crossed / no-quote skip itself
        for ex, tp in exits.items():
            cfg = config_name(model, hurdle, ex)
            res = S.run_trade(tape, side=side, trigger_ts=t, entry_style=S.TAKER, qty=qty,
                              cost_lo=COST_LO_D, cost_hi=cost_hi, tp_c=tp)
            if isinstance(res, S.Skip):
                out[cfg]["skips"][res.reason] += 1
                continue
            out[cfg]["rows"].append(RC.trade_row(
                res, lane=lane.lane, ticker=name, close=float(pack["close"][i]),
                slice_name=pack["split"], feat=feat, config=cfg, family=FAMILY[model],
                model=model, hurdle=hurdle, exit=ex))


def eligible(s: dict) -> bool:
    return s["trades"] >= MIN_TRADES and s["pf"] >= MIN_PF and s["crossed_fills"] == 0


def pick_champion(family: str, configs: dict) -> dict:
    names = [c for c, v in configs.items() if v["family"] == family]
    el = [c for c in names if eligible(configs[c]["is"])]
    best = max(el or names, key=lambda c: configs[c]["is"]["net"])
    return {"config": best, "is_eligible": bool(el), "is": configs[best]["is"]}


# ------------------------------------------------------------- modes
def run_is(lanes: dict, cut: float) -> dict:
    t0 = time.time()
    check = cdf_check()
    if not check["pass"]:
        raise RuntimeError(f"Student-t CDF unit check failed: {check}")
    fits = {}
    for name, lane in lanes.items():
        its, v = clean_index(lane.index, before=cut - FIT_LEAD)
        fits[name] = fit_lane(its, v, cut)
    packs = [collect(lane, cut, fits[name], "is") for name, lane in lanes.items()]

    out = {config_name(m, h, e): {"rows": [], "skips": Counter()}
           for m in MODELS for h in HURDLES for e in EXITS}
    trig_counts: Counter = Counter()
    for pack in packs:
        for model in MODELS:
            for h in HURDLES:
                trig = find_triggers(pack, model, h)
                trig_counts[f"{model}_h{h:.2f}"] += len(trig)
                run_triggers(pack, trig, model, h, out, EXITS)

    configs = {}
    for model in MODELS:
        for h in HURDLES:
            for e in EXITS:
                cfg = config_name(model, h, e)
                configs[cfg] = {"family": FAMILY[model], "model": model, "hurdle": h,
                                "exit": e,
                                "is": RC.summarize(out[cfg]["rows"], dict(out[cfg]["skips"]))}
    champions = {fam: pick_champion(fam, configs) for fam in FAMILY.values()}

    calib = {}
    for model in MODELS:
        pairs = [pr for pack in packs for pr in pack["pairs"][model]]
        calib[model] = {
            "ece_is": round(features.expected_calibration_error(pairs), 4),
            "n_pairs": len(pairs),
            "p_side_mean": round(float(np.mean([p for p, _ in pairs])), 4) if pairs else None,
            "side_won_rate": round(float(np.mean([w for _, w in pairs])), 4) if pairs else None,
            "cost_band": [COST_LO, COST_HI],
            "pairs_over": "every IS sampled quote (secs 120..780 step 5), both sides, cost in band",
        }

    champ_rows = []
    for fam, ch in champions.items():
        champ_rows.extend(out[ch["config"]]["rows"])
    TRADES.parent.mkdir(parents=True, exist_ok=True)
    TRADES.write_text("".join(json.dumps(r, default=RC._json_default) + "\n" for r in champ_rows))

    samples = {}
    for pack in packs:
        samples[pack["lane"].lane] = {"tickers": pack["tickers"],
                                      "valid_samples": int(len(pack["t"])),
                                      "skips": dict(pack["skips"])}
    obj = {
        "split": "IS only: markets with close < cut; index used before cut (fit before cut - 900)",
        "cut": cut,
        "cut_source": "rnd_common.global_cut over the four lanes",
        "lanes": list(lanes),
        "fit": {"lanes": fits, "cdf_check": check},
        "features": FEATURES,
        "spot_rule": "last index print at or before t within 1.5s (causal, not nearest)",
        "sampling": samples,
        "trigger_counts": dict(trig_counts),
        "configs": configs,
        "champions": champions,
        "champion_trades_file": str(TRADES),
        "settlement_calibration": calib,
        "runtime_s": round(time.time() - t0, 1),
    }
    RC.dump(OUT, obj)
    return obj


def score_oos(lanes: dict, cut: float, is_obj: dict) -> int:
    """Sealed: score each family champion once on OOS. Never run by Track 3."""
    if abs(is_obj["cut"] - cut) > 1e-9:
        raise RuntimeError("IS JSON cut does not match this cut")
    trades = [json.loads(ln) for ln in TRADES.read_text().splitlines() if ln.strip()]
    for family in FAMILY.values():
        ch = is_obj["champions"][family]
        cfg_name = ch["config"]
        c = is_obj["configs"][cfg_name]
        is_rows = [r for r in trades if r["config"] == cfg_name]

        def scorer(c=c, cfg_name=cfg_name, is_rows=is_rows):
            out = {cfg_name: {"rows": [], "skips": Counter()}}
            for name, lane in lanes.items():
                pack = collect(lane, cut, is_obj["fit"]["lanes"][name], "oos")
                trig = find_triggers(pack, c["model"], c["hurdle"])
                run_triggers(pack, trig, c["model"], c["hurdle"], out,
                             {c["exit"]: EXITS[c["exit"]]})
            return out[cfg_name]["rows"], dict(out[cfg_name]["skips"]), RC.fit_payoff_model(is_rows)

        entry = RC.score_oos_once(family, {"config": cfg_name, "model": c["model"],
                                           "hurdle": c["hurdle"], "exit": c["exit"]}, scorer)
        print(json.dumps({"family": family, "oos": entry["oos"], "gate": entry["gate"]},
                         default=RC._json_default))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lanes", default=",".join(RC.LANES))
    ap.add_argument("--score-oos", action="store_true")
    args = ap.parse_args()
    names = tuple(x.strip() for x in args.lanes.split(",") if x.strip())
    light = cut_from_cache()
    lanes = RC.load_lanes(names)
    if set(names) == set(RC.LANES):
        cut = RC.global_cut(lanes)
        if abs(cut - light) > 1e-9:
            raise RuntimeError(f"cut mismatch: global {cut} vs cache {light}")
    else:
        cut = light
    if args.score_oos:
        if set(names) != set(RC.LANES):
            raise RuntimeError("--score-oos needs all four lanes")
        return score_oos(lanes, cut, json.loads(OUT.read_text()))
    obj = run_is(lanes, cut)
    print(json.dumps({"out": str(OUT), "cut": cut, "runtime_s": obj["runtime_s"],
                      "configs": len(obj["configs"]),
                      "champions": {f: {"config": c["config"], "eligible": c["is_eligible"],
                                        "trades": c["is"]["trades"], "net": c["is"]["net"],
                                        "pf": c["is"]["pf"], "win_rate": c["is"]["win_rate"]}
                                    for f, c in obj["champions"].items()}},
                     default=RC._json_default, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
