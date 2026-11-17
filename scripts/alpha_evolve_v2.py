"""Track 4: multi-generation evolutionary search over the F2 rule family.

Default mode is IS-only. Every genome is evaluated on the IS windows (close < the
global cut), each family's champion is picked on IS, and no OOS row is built or
simulated. `--score-oos` scores the frozen champions once on OOS through
rnd_common.score_oos_once. It refuses without RND_OOS_UNSEAL=lead and is not run
by the research track.
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
from scripts import rnd_common as RC  # noqa: E402
from scripts.bounded_sweep import pick  # noqa: E402

SEED = 20261117
COOLDOWN, PER_WINDOW = 90.0, 3
POOL_DIR = ROOT / "data" / "rnd-pool"
OUT_JSON = ROOT / "data" / "alpha-evolve-v2.json"
OUT_TRADES = ROOT / "data" / "alpha-evolve-v2-is-trades.jsonl"
FEAT_NAMES = ["abs_z", "flow_d", "rev_d", "delta_bps", "spread", "secs"]

# Numeric genes step +-1 on these grids (None = gene off where it is allowed).
GRID = {
    "z_min": [0.10, 0.20, 0.30, 0.40, 0.50],
    "delta_bps_min": [3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
    "flow_min": [-0.20, -0.10, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30],
    "rev_min": [-0.20, -0.10, 0.0, 0.05, 0.10, 0.20],
    "shift_min": [None, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03],
    "spread_max": [0.01, 0.02, 0.03, 0.04],
    "secs_lo": [180.0, 240.0, 300.0, 360.0, 420.0],
    "secs_hi": [540.0, 600.0, 660.0, 720.0, 780.0],
    "vol_q_lo": [0.0, 0.2, 0.4, 0.6],
    "vol_q_hi": [0.4, 0.6, 0.8, 1.0],
    "div_min": [None, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
    "cost_lo": [0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
    "cost_hi": [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.62],
    "tp_c": [None, 0.04, 0.05, 0.06, 0.08],
    "stop_c": [None, 0.05, 0.07, 0.10],
    "hold_s": [None, 90.0, 180.0],
}
# Categorical genes resample.
CAT = {"entry": ["taker", "maker3", "maker5", "maker10"], "barrier_exit": ["settle", "taker"]}
ENTRY = {"taker": (S.TAKER, 5.0), "maker3": (S.MAKER, 3.0), "maker5": (S.MAKER, 5.0),
         "maker10": (S.MAKER, 10.0)}

SEED_GENOME = {
    "z_min": 0.30, "delta_bps_min": 3.0, "flow_min": 0.10, "rev_min": 0.0,
    "shift_min": None, "spread_max": 0.04, "secs_lo": 180.0, "secs_hi": 780.0,
    "blocks": [0, 1, 2, 3], "vol_q_lo": 0.0, "vol_q_hi": 1.0, "div_min": None,
    "entry": "taker", "cost_lo": 0.25, "cost_hi": 0.60, "tp_c": 0.05, "stop_c": 0.07,
    "hold_s": 90.0, "barrier_exit": "taker",
}
FAMILY_GENES = {
    "depletion": {"flow_min", "rev_min", "shift_min"},
    "divergence": {"div_min"},
    "spread_regime": {"spread_max", "secs_lo", "secs_hi"},
    "tod_vol": {"blocks", "vol_q_lo", "vol_q_hi"},
}
FAMILY_SEED = {"divergence": {"div_min": 0.0}}
COMMON_GENES = {"z_min", "delta_bps_min", "entry", "cost_lo", "cost_hi", "tp_c", "stop_c",
                "hold_s", "barrier_exit"}


def _dec(x):
    return None if x is None else Decimal(str(x))


# ------------------------------------------------------------------ data
def light_cut() -> float:
    """The global cut without loading books: same closes as rnd_common.global_cut."""
    meta = json.loads((RC.CACHE / "markets.json").read_text())
    closes = set()
    for name in RC.LANES:
        tickers = np.load(RC.CACHE / f"book_{name}.npz", allow_pickle=False)["tickers"]
        for t in tickers:
            m = meta.get(str(t))
            if m and m.get("strike"):
                closes.add(float(m["close"]))
    closes = sorted(closes)
    return closes[int(len(closes) * 0.6)]


def load_ticks() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Engine spot feed per ticker: ticker -> (ts, spot), both sorted by time."""
    tk = np.load(RC.CACHE / "ticks.npz", allow_pickle=False)
    good = tk["spot"] > 0            # 6 engine ticks carry spot 0: they fake a 10000 bps gap
    names, inv = np.unique(tk["ticker"][good], return_inverse=True)
    order = np.lexsort((tk["ts"][good], inv))
    ts = tk["ts"][good][order]
    sp = tk["spot"][good][order].astype(np.float64)
    bounds = np.searchsorted(inv[order], np.arange(len(names) + 1))
    return {str(n): (ts[bounds[i]:bounds[i + 1]], sp[bounds[i]:bounds[i + 1]])
            for i, n in enumerate(names)}


def divergence(P: dict, lane, ticks: dict) -> np.ndarray:
    """Signed bps gap of the engine tick (last at or before ts, within 1.5s) vs pool spot."""
    tid = P["tid"]
    out = np.full(len(tid), np.nan)
    if not len(tid):
        return out
    cuts = np.flatnonzero(np.diff(tid)) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [len(tid)]))
    for a, b in zip(starts, ends):
        series = ticks.get(lane.tickers[int(tid[a])])
        if series is None:
            continue
        tts, tsp = series
        t = P["ts"][a:b]
        j = np.searchsorted(tts, t, "right") - 1
        jj = np.maximum(j, 0)
        ok = (j >= 0) & (t - tts[jj] <= 1.5)
        out[a:b] = np.where(ok, (tsp[jj] / P["spot"][a:b] - 1.0) * 1e4 * P["d"][a:b], np.nan)
    return out


def prep(name: str, lane, cut: float, which: str, ticks: dict):
    """Derived per-row features for one lane, restricted to `which` ("is" or "oos")."""
    raw = dict(np.load(POOL_DIR / f"slim_{name}.npz", allow_pickle=False))
    # Quotes are float32 values held in float64: snap to the $0.001 tick so that
    # spread and cost thresholds on the tick (0.04, 0.60) compare exactly.
    raw["bid"], raw["ask"] = np.round(raw["bid"], 3), np.round(raw["ask"], 3)
    tid, z, spot = raw["tid"], raw["z"], raw["spot"]
    strike = np.array([float(lane.meta[t]["strike"]) if t in lane.meta else np.nan
                       for t in lane.tickers])[tid]
    secs = raw["secs"].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.abs(np.log(spot / strike)) / (np.abs(z) * np.sqrt(secs))
    is_m = raw["close"] < cut
    # Vol-regime quantile edges come from IS rows of this lane only.
    edges = np.sort(sigma[is_m & (z != 0) & np.isfinite(sigma)])
    vq = np.searchsorted(edges, sigma, "right") / len(edges)
    sel = (is_m if which == "is" else ~is_m) & (z != 0)
    ts = raw["ts"][sel]
    d = np.sign(z[sel])
    P = {
        "tid": tid[sel], "ts": ts, "close": raw["close"][sel], "z": z[sel], "d": d,
        "spot": spot[sel], "spread": np.round(raw["ask"] - raw["bid"], 3)[sel], "secs": secs[sel],
        "flow_d": raw["flow"][sel] * d, "rev_d": raw["rev"][sel] * d,
        "shift_d": raw["shift"][sel] * d,
        "delta_bps": raw["delta"][sel] / spot[sel] * 1e4, "vq": vq[sel],
        # Entry cost of the side the index favours, as frozen_mask prices it (pre-dedupe).
        "cost_taker": np.round(np.where(z[sel] > 0, raw["ask"][sel], 1.0 - raw["bid"][sel]), 3),
        "cost_maker": np.round(np.where(z[sel] > 0, raw["bid"][sel], 1.0 - raw["ask"][sel]), 3),
        "block": (ts % 86400 // 3600).astype(np.int64) // 6,
    }
    P["div"] = divergence(P, lane, ticks)
    return lane, P


# ------------------------------------------------------------------ evaluation
def mask(g: dict, P: dict) -> np.ndarray:
    m = ((np.abs(P["z"]) >= g["z_min"]) & (P["flow_d"] >= g["flow_min"])
         & (P["rev_d"] >= g["rev_min"]) & (P["delta_bps"] >= g["delta_bps_min"])
         & (P["spread"] <= g["spread_max"]) & (P["secs"] >= g["secs_lo"])
         & (P["secs"] <= g["secs_hi"]) & (P["vq"] >= g["vol_q_lo"])
         & (P["vq"] <= g["vol_q_hi"]) & np.isin(P["block"], g["blocks"]))
    cost = P["cost_taker"] if g["entry"] == "taker" else P["cost_maker"]
    m &= (cost >= g["cost_lo"] - 1e-9) & (cost <= g["cost_hi"] + 1e-9)
    if g["shift_min"] is not None:
        m &= P["shift_d"] >= g["shift_min"]
    if g["div_min"] is not None:
        m &= P["div"] >= g["div_min"]          # NaN never passes
    return m


def trade_one(lane, P: dict, i: int, g: dict, slice_name: str):
    tid = int(P["tid"][i])
    tape = RC.tape(lane, tid)
    side = S.YES if P["z"][i] > 0 else S.NO
    style, patience = ENTRY[g["entry"]]
    trig = float(P["ts"][i])
    try:
        cost = S.entry_price(side, style, *tape.quote(trig + S.LAG_S))
        qty = S.stake_qty(cost, RC.STAKE)
    except (S.CrossedBook, S.NoQuote):
        qty = S.MAX_QTY                      # run_trade reports the skip
    res = S.run_trade(tape, side=side, trigger_ts=trig, entry_style=style, qty=qty,
                      patience_s=patience, cost_lo=_dec(g["cost_lo"]),
                      cost_hi=_dec(g["cost_hi"]), tp_c=_dec(g["tp_c"]),
                      stop_c=_dec(g["stop_c"]), stop_confirm_s=2.0, hold_s=g["hold_s"],
                      barrier_exit=g["barrier_exit"])
    if isinstance(res, S.Skip):
        return None, res.reason
    feat = [float(abs(P["z"][i])), float(P["flow_d"][i]), float(P["rev_d"][i]),
            float(P["delta_bps"][i]), float(P["spread"][i]), float(P["secs"][i])]
    row = RC.trade_row(res, lane=lane.lane, ticker=lane.tickers[tid],
                       close=float(P["close"][i]), slice_name=slice_name, feat=feat)
    return row, None


def evaluate_all(g: dict, prepared: dict, slice_name: str):
    rows: list[dict] = []
    skips: Counter = Counter()
    for lane, P in prepared.values():
        for i in pick(P, mask(g, P), cooldown=COOLDOWN, per_window=PER_WINDOW):
            row, why = trade_one(lane, P, i, g, slice_name)
            if row is None:
                skips[why] += 1
            else:
                rows.append(row)
    return rows, dict(skips)


# ------------------------------------------------------------------ search
def valid(g: dict) -> bool:
    return (g["secs_hi"] - g["secs_lo"] >= 120 and g["vol_q_hi"] - g["vol_q_lo"] >= 0.4 - 1e-9
            and g["cost_hi"] >= g["cost_lo"] + 0.05 - 1e-9 and len(g["blocks"]) > 0)


def _mutate_gene(g: dict, name: str, rng) -> None:
    if name in GRID:
        vals = GRID[name]
        i = vals.index(g[name])
        step = 1 if rng.integers(2) else -1
        g[name] = vals[min(max(i + step, 0), len(vals) - 1)]
    elif name == "blocks":
        g[name] = [b for b in range(4) if rng.integers(2)]
    else:
        g[name] = CAT[name][int(rng.integers(len(CAT[name])))]


def mutate(parent: dict, family: str, rng) -> dict:
    genes = sorted(FAMILY_GENES[family] | COMMON_GENES)
    for _ in range(100):
        child = {**parent, "blocks": list(parent["blocks"])}
        k = int(rng.integers(1, 3))
        for j in rng.choice(len(genes), size=k, replace=False):
            _mutate_gene(child, genes[int(j)], rng)
        if child != parent and valid(child):
            return child
    return {**parent, "blocks": list(parent["blocks"])}


def eligible(s: dict) -> bool:
    return s["trades"] >= 30 and s["pf"] >= 1.20 and s["crossed_fills"] == 0


def pick_parent(evals: list[dict]) -> dict:
    el = [e for e in evals if eligible(e["is"])]
    return max(el or evals, key=lambda e: e["is"]["net"])


def champion(evals: list[dict]) -> tuple[dict, bool]:
    el = [e for e in evals if eligible(e["is"])]
    if el:
        return max(el, key=lambda e: e["is"]["net"]), True
    return max(evals, key=lambda e: e["is"]["net"]), False


def search(family: str, prepared: dict, G: int, M: int, memo: dict) -> list[dict]:
    rng = np.random.default_rng(SEED)
    evals: list[dict] = []

    def run(genome: dict, gen: int, idx: int, parent_id: str | None) -> None:
        key = json.dumps(genome, sort_keys=True)
        if key not in memo:
            rows, skips = evaluate_all(genome, prepared, "is")
            memo[key] = RC.summarize(rows, skips)
        rec = {"id": f"{family}:{gen}.{idx}", "family": family, "generation": gen,
               "index": idx, "parent_id": parent_id, "genome": genome, "is": memo[key]}
        evals.append(rec)
        s = rec["is"]
        print(f"[{family}] {rec['id']} trades={s['trades']} net={s['net']:.2f} "
              f"pf={s['pf']:.3f} crossed={s['crossed_fills']} elig={eligible(s)}",
              file=sys.stderr, flush=True)

    run({**SEED_GENOME, **FAMILY_SEED.get(family, {}), "blocks": list(SEED_GENOME["blocks"])},
        0, 0, None)
    for gen in range(1, G + 1):
        parent = pick_parent(evals)
        for m in range(1, M + 1):
            run(mutate(parent["genome"], family, rng), gen, m, parent["id"])
    return evals


# ------------------------------------------------------------------ OOS (never run here)
def score_oos(args, names: list[str], cut: float, ticks: dict, lanes: dict) -> None:
    doc = json.loads(args.out.read_text())
    if doc["cut"] != cut:
        raise SystemExit(f"cut mismatch: IS json {doc['cut']} vs {cut}")
    trades = [json.loads(x) for x in args.trades.read_text().splitlines() if x.strip()]
    prepared = {n: prep(n, lanes[n], cut, "oos", ticks) for n in names}
    for fam, ch in doc["champions"].items():
        model = RC.fit_payoff_model([r for r in trades if r["family"] == fam])

        def scorer(g=ch["genome"], m=model):
            rows, skips = evaluate_all(g, prepared, "oos")
            return rows, skips, m

        entry = RC.score_oos_once(fam, {"genome": ch["genome"],
                                        "evaluation_id": ch["evaluation_id"]}, scorer)
        print(f"[{fam}] OOS gate pass={entry['gate']['pass']} net={entry['oos']['net']}",
              file=sys.stderr, flush=True)


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lanes", default=",".join(RC.LANES))
    ap.add_argument("--generations", type=int, default=4)
    ap.add_argument("--mutations", type=int, default=12)
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    ap.add_argument("--trades", type=Path, default=OUT_TRADES)
    ap.add_argument("--score-oos", action="store_true",
                    help="score the IS champions once on OOS (refused without RND_OOS_UNSEAL)")
    args = ap.parse_args()
    names = [n for n in args.lanes.split(",") if n]

    cut = light_cut()
    lanes = RC.load_lanes(names)
    if set(names) == set(RC.LANES):
        assert cut == RC.global_cut(lanes), "light cut differs from rnd_common.global_cut"
    ticks = load_ticks()
    if args.score_oos:
        score_oos(args, names, cut, ticks, lanes)
        return 0

    prepared = {n: prep(n, lanes[n], cut, "is", ticks) for n in names}
    rows_is = {n: int(len(P["tid"])) for n, (_, P) in prepared.items()}
    coverage = {n: float(np.isfinite(P["div"]).mean()) if len(P["div"]) else 0.0
                for n, (_, P) in prepared.items()}

    memo: dict = {}
    all_evals: list[dict] = []
    champs: dict = {}
    trades_out: list[dict] = []
    for fam in FAMILY_GENES:
        evals = search(fam, prepared, args.generations, args.mutations, memo)
        all_evals += evals
        best, elig = champion(evals)
        rows, skips = evaluate_all(best["genome"], prepared, "is")
        summary = RC.summarize(rows, skips)
        assert summary["net"] == best["is"]["net"], "champion re-evaluation drifted"
        champs[fam] = {"genome": best["genome"], "is_eligible": elig, "is": summary,
                       "evaluation_id": best["id"]}
        trades_out += [dict(r, family=fam) for r in rows]

    doc = {
        "cut": cut,
        "lanes": names,
        "mode": "IS-only",
        "oos_scored": False,
        "seed": SEED,
        "generations": args.generations,
        "mutations": args.mutations,
        "n_evaluations": len(all_evals),
        "n_configs": len(memo),
        "rows_is": rows_is,
        "divergence_coverage": coverage,
        "feat": FEAT_NAMES,
        "notes": [
            "IS only: close < cut. No OOS row was built or simulated.",
            "Tape is about two days: hour-of-day blocks have few independent samples, "
            "so tod_vol results are weak evidence.",
            "shift_min None and div_min None mean off (the brief's -1 for shift).",
            "Numeric genes step +-1 on the grid; blocks resample; entry, barrier_exit resample.",
            "A genome already evaluated reuses its IS summary; every child is still logged.",
            "Quotes snap to the $0.001 tick; engine ticks with spot <= 0 are dropped before the "
            "divergence lookup (they fall back to the last valid tick within 1.5s).",
            "The cost band is applied in the mask before pick(), as in f2_structural.frozen_mask, "
            "so out-of-band triggers do not use up the cooldown; run_trade still enforces it.",
        ],
        "families": {f: {"primitive_genes": sorted(FAMILY_GENES[f]),
                         "seed": {**SEED_GENOME, **FAMILY_SEED.get(f, {})}}
                     for f in FAMILY_GENES},
        "evaluations": all_evals,
        "champions": champs,
    }
    RC.dump(args.out, doc)
    args.trades.write_text("".join(json.dumps(r, default=RC._json_default) + "\n"
                                   for r in trades_out))
    for fam, c in champs.items():
        s = c["is"]
        print(f"{fam}: {c['evaluation_id']} eligible={c['is_eligible']} trades={s['trades']} "
              f"net={s['net']:.2f} pf={s['pf']:.3f} win={s['win_rate']} "
              f"skips={s['skips']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
