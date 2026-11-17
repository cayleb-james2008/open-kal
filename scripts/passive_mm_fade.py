"""Track A: passive book-imbalance fade (`passive_mm_fade`).

Rest a $0-fee maker bid that fades an extreme imbalance skew, in a bounded-volatility
regime. Exits are $0-fee resting legs only (maker take-profit, maker scratch) or
settlement. No taker leg anywhere and no stop. Every fill goes through
openkal.sim_core.run_trade with the side and the style; prices are never computed here.

The book cache has no sizes, so the imbalance is the pool's `flow` (10s signed print
imbalance) or `rev` (5s touch-revision imbalance), each z-scored on that lane's IS rows.

Default mode is IS-ONLY: OOS rows are never simulated. `--score-oos` reads the frozen
champion from the IS JSON and scores it once through rnd_common.score_oos_once. It
refuses without RND_OOS_UNSEAL=lead and is not run by the research track.

Usage (all four lanes, under the shared lock):
    flock data/.heavy.lock timeout 1800 nice -n 10 .venv/bin/python scripts/passive_mm_fade.py
Debug on one lane (no lock needed):
    .venv/bin/python scripts/passive_mm_fade.py --lanes KXSOL15M --out X.json --trades-out X.jsonl
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
from scripts import alpha_evolve_v2 as AE  # noqa: E402  (global cut without loading books)
from scripts import rnd_common as RC  # noqa: E402
from scripts.bounded_sweep import pick  # noqa: E402

FAMILY = "passive_mm_fade"
POOL_DIR = ROOT / "data" / "rnd-pool"
OUT_JSON = ROOT / "data" / "passive-mm-fade-is.json"
OUT_TRADES = ROOT / "data" / "passive-mm-fade-is-trades.jsonl"
FEAT_NAMES = ["z_obi_source", "abs_pool_z", "spread", "secs", "vq"]
COOLDOWN, PER_WINDOW = 90.0, 3
PATIENCE_S = 10.0
COST_HI = Decimal("0.55")
SOURCES = ("flow", "rev")
THRS = (1.5, 2.0)
VOL_BANDS = ((0.0, 1.0), (0.2, 0.8))
TP_CS = (Decimal("0.04"), Decimal("0.06"))
EXITS = {
    "settle": {},
    "scratch_hold180": {"hold_s": 180.0, "barrier_exit": "scratch"},
    "scratch_pre120": {"scratch_by_s": 120.0, "barrier_exit": "scratch"},
}
PROXY_NOTE = (
    "The book cache has top-of-book prices only (no sizes), so depth OBI and microprice "
    "are not computable. Proxies: flow = 10s signed print-flow imbalance (+ = takers buying "
    "YES); rev = 5s touch-revision imbalance (+ = bid upticks dominate). Each is z-scored "
    "per lane on that lane's IS rows (population std, valid rows only: z != 0, finite "
    "sigma, flow and rev finite): z_obi = (x - mean) / std, applied to every row."
)


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------ configs
def all_configs() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for src in SOURCES:
        for thr in THRS:
            for lo, hi in VOL_BANDS:
                for tp in TP_CS:
                    for ex in EXITS:
                        cfg = {"source": src, "thr": thr, "vol_band": [lo, hi],
                               "tp_c": str(tp), "exit": ex}
                        out[config_name(cfg)] = cfg
    if len(out) != 48:
        raise RuntimeError(f"expected 48 configs, built {len(out)}")
    return out


def config_name(cfg: dict) -> str:
    lo, hi = cfg["vol_band"]
    return f"{cfg['source']}_thr{cfg['thr']:g}_vq{lo:g}-{hi:g}_tp{cfg['tp_c']}_{cfg['exit']}"


# ------------------------------------------------------------------ data
def prepare(lane, name: str, cut: float, which: str, norm: dict | None = None):
    """Derived per-row features for one lane, restricted to `which` ("is" or "oos").

    The z-score and the vol-quantile edges always come from IS rows only. `norm` is
    passed in when scoring OOS, so the frozen IS values are reused.
    """
    raw = dict(np.load(POOL_DIR / f"slim_{name}.npz", allow_pickle=False))

    def col(k: str) -> np.ndarray:
        return raw[k].astype(np.float64)

    tid = raw["tid"].astype(np.int64)
    if np.any(np.diff(tid) < 0):
        raise RuntimeError(f"{name}: pool tids are not sorted; pick() needs sorted tids")
    bid = np.round(col("bid"), 3)
    ask = np.round(col("ask"), 3)
    z, spot, secs = col("z"), col("spot"), col("secs")
    close, ts = col("close"), col("ts")
    flow, rev = col("flow"), col("rev")
    strike = np.array([float(lane.meta[t]["strike"]) if t in lane.meta else np.nan
                       for t in lane.tickers])[tid]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.abs(np.log(spot / strike)) / (np.abs(z) * np.sqrt(secs))
    is_m = close < cut
    base = (z != 0) & np.isfinite(sigma) & np.isfinite(flow) & np.isfinite(rev)
    base &= np.isfinite(bid) & np.isfinite(ask)
    edges = np.sort(sigma[is_m & base])
    vq = np.searchsorted(edges, sigma, "right") / len(edges)
    if norm is None:
        norm = {}
        for src, x in (("flow", flow), ("rev", rev)):
            v = x[is_m & base]
            norm[src] = {"mean": float(v.mean()), "std": float(v.std()), "n_is": int(v.size)}
    sel = base & (is_m if which == "is" else ~is_m)
    zo = {src: ((x - norm[src]["mean"]) / norm[src]["std"])[sel]
          for src, x in (("flow", flow), ("rev", rev))}
    P = {
        "tid": tid[sel], "ts": ts[sel], "close": close[sel], "secs": secs[sel],
        "zo": zo, "absz": np.abs(z[sel]), "vq": vq[sel],
        "spread": np.round(ask - bid, 3)[sel],
        "bid": bid[sel], "cost_no": np.round(1.0 - ask, 3)[sel],
    }
    n_is = int((is_m & base).sum())
    return P, norm, n_is


def trigger_mask(P: dict, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    zo = P["zo"][cfg["source"]]
    lo, hi = cfg["vol_band"]
    m = (np.abs(zo) >= cfg["thr"]) & (P["vq"] >= lo) & (P["vq"] <= hi)
    # Fade side cost at the trigger row: YES fade is a bid on YES, NO fade a bid on NO.
    cost = np.where(zo > 0, P["cost_no"], P["bid"])
    m &= cost <= float(COST_HI) + 1e-9
    return m, zo


# ------------------------------------------------------------------ evaluation
def trade_one(lane, cfg: dict, P: dict, i: int, zo: np.ndarray, which: str):
    tid = int(P["tid"][i])
    tape = RC.tape(lane, tid)
    side = S.NO if zo[i] > 0 else S.YES
    trig = float(P["ts"][i])
    try:
        cost = S.entry_price(side, S.MAKER, *tape.quote(trig + S.LAG_S))
        qty = S.stake_qty(cost, RC.STAKE)
    except (S.CrossedBook, S.NoQuote):
        qty = S.MAX_QTY                      # run_trade reports the skip
    res = S.run_trade(tape, side=side, trigger_ts=trig, entry_style=S.MAKER, qty=qty,
                      patience_s=PATIENCE_S, cost_hi=COST_HI,
                      tp_c=Decimal(cfg["tp_c"]), **EXITS[cfg["exit"]])
    if isinstance(res, S.Skip):
        return None, res.reason
    if any(leg.style == S.TAKER for leg in res.legs) or res.crossed:
        raise RuntimeError("taker or crossed leg in passive_mm_fade; fill model violated")
    feat = [float(zo[i]), float(P["absz"][i]), float(P["spread"][i]),
            float(P["secs"][i]), float(P["vq"][i])]
    row = RC.trade_row(res, lane=lane.lane, ticker=lane.tickers[tid],
                       close=float(P["close"][i]), slice_name=which, feat=feat)
    return row, None


def run_config(cfg: dict, preps: list, which: str):
    rows: list[dict] = []
    skips: Counter = Counter()
    triggers = 0
    for lane, P in preps:
        m, zo = trigger_mask(P, cfg)
        picked = pick({"tid": P["tid"], "ts": P["ts"]}, m, cooldown=COOLDOWN,
                      per_window=PER_WINDOW)
        triggers += len(picked)
        for i in picked:
            row, why = trade_one(lane, cfg, P, i, zo, which)
            if row is None:
                skips[why] += 1
            else:
                rows.append(row)
    return rows, dict(skips), triggers


def exit_mix(rows: list[dict]) -> dict:
    mix = {k: {"count": 0, "net": Decimal(0)}
           for k in ("tp", "scratch", "settle_won", "settle_lost")}
    for r in rows:
        key = r["reason"]
        if key == "settle":
            key = "settle_won" if Decimal(r["exit_value"]) > 0 else "settle_lost"
        if key not in mix:
            raise RuntimeError(f"unexpected exit reason {r['reason']!r}")
        mix[key]["count"] += 1
        mix[key]["net"] += Decimal(r["pnl"])
    return {k: {"count": v["count"], "net": float(v["net"])} for k, v in mix.items()}


def evaluate(cfg: dict, preps: list) -> dict:
    rows, skips, triggers = run_config(cfg, preps, "is")
    fills = len(rows)
    return {
        **cfg,
        "is": RC.summarize(rows, skips),
        "triggers": triggers,
        "fills": fills,
        "fill_rate": round(fills / triggers, 4) if triggers else 0.0,
        "mix": exit_mix(rows),
    }


def choose(configs: dict[str, dict]) -> tuple[str, bool]:
    """IS rule: eligible = trades >= 30, PF >= 1.20, zero crossed fills; max net wins."""
    elig = {k: v for k, v in configs.items()
            if v["is"]["trades"] >= 30 and v["is"]["pf"] >= 1.20
            and v["is"]["crossed_fills"] == 0}
    pool = elig or configs
    return max(pool, key=lambda k: pool[k]["is"]["net"]), bool(elig)


# ------------------------------------------------------------------ modes
def run_is(args) -> None:
    names = [n.strip() for n in args.lanes.split(",") if n.strip()]
    bad = [n for n in names if n not in RC.LANES]
    if bad:
        raise SystemExit(f"unknown lanes {bad}")
    lanes = RC.load_lanes(tuple(names))
    cut = AE.light_cut()          # global cut over all four lanes, even for a subset run
    if len(names) == len(RC.LANES) and cut != RC.global_cut(lanes):
        raise RuntimeError("light_cut disagrees with rnd_common.global_cut")
    preps, norms, rows_is = [], {}, {}
    for name in names:
        P, norm, n_is = prepare(lanes[name], name, cut, "is")
        preps.append((lanes[name], P))
        norms[name] = norm
        rows_is[name] = n_is
        _log(f"{name}: IS rows {n_is}, norm {norm}")
    configs = all_configs()
    results: dict[str, dict] = {}
    for cname, cfg in configs.items():
        results[cname] = evaluate(cfg, preps)
        r = results[cname]["is"]
        _log(f"{cname}: trades {r['trades']} net {r['net']:.2f} pf {r['pf']:.3f} "
             f"fill {results[cname]['fill_rate']}")
    champ, eligible = choose(results)
    champ_rows, _, _ = run_config(configs[champ], preps, "is")
    args.trades_out.parent.mkdir(parents=True, exist_ok=True)
    with args.trades_out.open("w") as fh:
        for row in champ_rows:
            fh.write(json.dumps(row, default=RC._json_default) + "\n")
    doc = {
        "family": FAMILY,
        "cut": cut,
        "lanes": names,
        "stake": str(RC.STAKE),
        "proxy_note": PROXY_NOTE,
        "features": FEAT_NAMES,
        "z_obi_norm": norms,
        "rows_is": rows_is,
        "n_configs": len(results),
        "configs": results,
        "champion": {"family": FAMILY, "config": champ, "is_eligible": eligible,
                     "is": results[champ]["is"]},
        "champion_trades_file": str(args.trades_out),
    }
    RC.dump(args.out, doc)
    _log(f"champion {champ} eligible={eligible}; wrote {args.out} and {args.trades_out}")


def run_score_oos(args) -> dict:
    """Score the IS champion once on OOS. Implemented, never run by the research track."""
    doc = json.loads(args.out.read_text())
    champ = doc["champion"]["config"]
    cfg = doc["configs"][champ]
    names = list(doc["z_obi_norm"])
    cut = doc["cut"]
    is_rows = [json.loads(line) for line in args.trades_out.read_text().splitlines()
               if line.strip()]
    lanes = RC.load_lanes(tuple(names))

    def scorer():
        preps = []
        for name in names:
            P, _, _ = prepare(lanes[name], name, cut, "oos", norm=doc["z_obi_norm"][name])
            preps.append((lanes[name], P))
        rows, skips, _ = run_config(cfg, preps, "oos")
        return rows, skips, RC.fit_payoff_model(is_rows)

    entry_cfg = {"name": champ, **{k: cfg[k] for k in ("source", "thr", "vol_band",
                                                       "tp_c", "exit")}}
    return RC.score_oos_once(FAMILY, entry_cfg, scorer)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lanes", default=",".join(RC.LANES))
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    ap.add_argument("--trades-out", type=Path, default=OUT_TRADES)
    ap.add_argument("--score-oos", action="store_true",
                    help="score the frozen IS champion once on OOS (sealed; not run)")
    args = ap.parse_args()
    if args.score_oos:
        entry = run_score_oos(args)
        print(json.dumps({"family": FAMILY, "gate": entry["gate"]}, default=RC._json_default))
        return
    run_is(args)


if __name__ == "__main__":
    main()
