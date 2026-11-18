#!/usr/bin/env python3
"""Track 10: re-score the old champion families at 1, 2 and 5 s latency, with taker fees, on windows nobody chose them on.

Registered in scripts/track10_prereg.json (sha pinned in tests/test_track10_rescore.py) before any trade was scored. Every trade
is run by openkal.sim_core.run_trade; the scenario latency is its `lag_s`. Nothing is refit: the rules, the fade's z-score
normaliser and the exits are frozen, so the walk-forward folds show stability over time, and only the post-freeze fold is tested.

    flock data/.heavy.lock nice -n 10 .venv/bin/python scripts/track10_champion_rescore.py --build-tape
    flock data/.heavy.lock nice -n 10 .venv/bin/python scripts/track10_champion_rescore.py --build-pools
    flock data/.heavy.lock nice -n 10 .venv/bin/python scripts/track10_champion_rescore.py            # score, ledger, receipt
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openkal import sim_core as S  # noqa: E402
from scripts import alpha_search as A  # noqa: E402
from scripts import f2_structural as F  # noqa: E402
from scripts import passive_mm_fade as PM  # noqa: E402
from scripts import rnd_common as RC  # noqa: E402
from scripts import rnd_pool_build as RB  # noqa: E402
from scripts import track2c_true_l2_settle as C  # noqa: E402
from scripts import track4_common as T  # noqa: E402
from scripts.bounded_sweep import pick  # noqa: E402

SPEC_PATH = Path(__file__).with_name("track10_prereg.json")
SPEC_BYTES = SPEC_PATH.read_bytes()
SPEC = json.loads(SPEC_BYTES)
SPEC_SHA256 = hashlib.sha256(SPEC_BYTES).hexdigest()
TAPE = ROOT / "data" / "track10-tape"
POOL = ROOT / "data" / "track10-pool"
LANES = RC.LANES
LATENCIES = tuple(SPEC["latencies_s"])
SCORED = tuple(k for k, v in SPEC["families"].items() if v["scored"])
M, ALPHA = SPEC["holm"]["m"], SPEC["holm"]["alpha"]
FREEZE_CLOSE = float(SPEC["freeze"]["freeze_close_epoch"])
DRAWS, SEED, MIN_TRADES, FOLDS_PRE = 10_000, 0, 30, 5            # the registered bootstrap, trade floor and fold count
FADE_RECEIPT = ROOT / "data" / "passive-mm-fade-is.json"
FADE_IS_CUT = 1794881700.0                                        # the fade's own IS/OOS cut, from its receipt
POOL_KEYS = tuple(RB.KEEP)
POOL_PIN = "pool-pin.json"                                         # which pinned tape a pool directory was built from
REASON = "Track 10: frozen champion re-scored on post-freeze windows at a scenario latency (not an OOS look of the family)"
# recorded before this track existed; the new pools must rebuild them (a validity gate, not a registered threshold)
RECORDED = {"F2_taker_1s_pre_freeze_triggers": 66 + 45, "passive_mm_fade_is_triggers": 56}
# F2's forward look (scripts/forward_oos.py, the 'forward' block of data/strategy-verdict.json): the frozen rule was scored once, nothing
# fitted, on every settled window from 2026-11-17 05:00. The artifact data/forward-oos.json is not on this machine; the end is inferred
# from the block's 27 trades at 2.7 trades an hour, about 10 hours, i.e. to about 15:00, which is after the freeze.
F2_FORWARD_LOOK = {"slice_start": 1794909600.0, "inferred_hours": 10.0, "inferred_end_close": 1794909600.0 + 10 * 3600.0}
EVIDENCE = {
    "F2_taker_1s": [("/home/cayleb/open-kal/data/f2-structural.json", "frozen rule, IS and OOS on the frozen 60/40 split"),
                    ("data/evolve-f2-variants.json", "peg, take-profit and ETH variants, IS and OOS"),
                    ("data/sim-core-parity.json", "sim_core rerun of the rule, IS and OOS (unlogged)"),
                    ("data/alpha-evolve-v2.json", "seed of the evolutionary search (IS)"),
                    ("/home/cayleb/open-kal/data/strategy-verdict.json", "promotion verdict: payoff calibrator fitted, forward slice scored once by scripts/forward_oos.py")],
    "passive_mm_fade": [("data/passive-mm-fade-is.json", "IS grid, champion frozen")],
    "zobi_xrp_carry": [("/home/cayleb/open-kal/data/strategy-verdict.prev-rejected.json", "z_obi_flow verdict on the frozen split")],
    "zobi_band_084": [("/home/cayleb/open-kal/data/strategy-verdict.prev-rejected.json", "z_obi_flow verdict on the frozen split")],
}


def build_tape(dest: Path = TAPE, snapshot=C.snapshot_cache) -> dict:
    """Pin the tape: a consistency-checked copy of the live evolve cache (only read), digests in pin.json."""
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "pin.json").exists():
        raise RuntimeError(f"{dest} is already pinned")
    digests = snapshot(dest)
    if digests is None:
        raise RuntimeError("the evolve cache changed while it was copied; try again")
    (dest / "pin.json").write_text(json.dumps({"sha256": digests, "source": str(C.EVOLVE_CACHE)}) + "\n")
    return digests


def build_pools(tape: Path = TAPE, out: Path = POOL, lanes=LANES) -> dict:
    """Causal-spot trigger pools for the pinned tape, one lane at a time, written to `out` (never to data/rnd-pool)."""
    if out.resolve() == RB.POOL.resolve():
        raise RuntimeError("refusing to overwrite data/rnd-pool: earlier receipts depend on it")
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"{out} already holds pools; build a later pin into a new directory")
    pin = T.pin_tape(tape)
    A.index_at = RB.causal_index_at                      # obi_pool resolves A.index_at at call time
    out.mkdir(parents=True, exist_ok=True)
    rows = {}
    for name in lanes:
        t0 = time.time()
        lane_set = RC.load_lanes((name,))
        pool = F.build_pools(lane_set)[name]
        idx = np.flatnonzero(np.abs(pool["z"]) >= RB.Z_FLOOR)
        assert np.all(np.diff(pool["tid"][idx]) >= 0), "pool rows must stay grouped by tid"
        np.savez(out / f"{RB.PREFIX}{name}.npz", **{k: np.asarray(pool[k][idx], dtype=dt) for k, dt in RB.KEEP.items()})
        rows[name] = int(len(idx))
        print(f"{name}: {len(idx):,} rows ({time.time() - t0:.0f}s)", flush=True)
        del lane_set, pool
        gc.collect()
    (out / POOL_PIN).write_text(json.dumps({"tape_sha256": pin["sha256"]}) + "\n")
    return rows


# ----------------------------------------------------------------------------------------------------- triggers
def load_pool(name: str, pool_dir: Path) -> dict:
    with np.load(pool_dir / f"{RB.PREFIX}{name}.npz", allow_pickle=False) as f:
        return {k: f[k] for k in POOL_KEYS}


def f2_triggers(name: str, lane, pool: dict) -> list[dict]:
    """The frozen F2 rule's picked rows for one lane (f2_structural's own mask, bounded_sweep.pick)."""
    mask, long_yes = F.frozen_mask(pool, "taker")
    out = []
    for i in pick(pool, mask, cooldown=F.COOLDOWN, per_window=F.PER_WINDOW):
        tid = int(pool["tid"][i])
        out.append({"family": "F2_taker_1s", "lane": name, "tid": tid, "ticker": lane.tickers[tid], "close": float(pool["close"][i]),
                    "ts": float(pool["ts"][i]), "side": S.YES if bool(long_yes[i]) else S.NO, "feat": F.feature_row(pool, i)})
    return out


def fade_norm() -> dict:
    """The per-lane z-score normaliser frozen in the fade's IS receipt."""
    receipt = json.loads(FADE_RECEIPT.read_text())
    if receipt["champion"]["config"] != "flow_thr2_vq0-1_tp0.06_scratch_hold180":
        raise RuntimeError("the fade receipt's champion is not the registered one")
    return receipt["z_obi_norm"]


def fade_triggers(name: str, lane, pool_dir: Path, norm: dict) -> list[dict]:
    """passive_mm_fade's champion triggers: prepare() with every row 'in sample' and the frozen normaliser, so no row is refit."""
    old, PM.POOL_DIR = PM.POOL_DIR, pool_dir
    try:
        P, _, _ = PM.prepare(lane, name, math.inf, "is", norm=norm[name])
    finally:
        PM.POOL_DIR = old
    mask, zo = PM.trigger_mask(P, {"source": "flow", "thr": 2.0, "vol_band": (0.0, 1.0)})
    out = []
    for i in pick({"tid": P["tid"], "ts": P["ts"]}, mask, cooldown=PM.COOLDOWN, per_window=PM.PER_WINDOW):
        tid = int(P["tid"][i])
        out.append({"family": "passive_mm_fade", "lane": name, "tid": tid, "ticker": lane.tickers[tid], "close": float(P["close"][i]),
                    "ts": float(P["ts"][i]), "side": S.NO if zo[i] > 0 else S.YES})      # fade: flow buying YES buys NO
    return out


# ---------------------------------------------------------------------------------------------------------- trades
def stake_qty(tape: S.Tape, side: str, ts: float, lag: float) -> int:
    """Contracts the stake buys at the cost the order meets on arrival; a crossed or empty book leaves MAX_QTY for run_trade to skip."""
    try:
        b, a = tape.quote(ts + lag)
        return S.stake_qty(S.entry_price(side, S.TAKER, b, a), RC.STAKE)
    except (S.CrossedBook, S.NoQuote):
        return S.MAX_QTY


def run_one(t: dict, tape: S.Tape, lag: float):
    qty = stake_qty(tape, t["side"], t["ts"], lag)
    common = dict(side=t["side"], trigger_ts=t["ts"], entry_style=S.TAKER, qty=qty, lag_s=lag)
    if t["family"] == "F2_taker_1s":
        return S.run_trade(tape, cost_lo=Decimal("0.25"), tp_c=Decimal("0.05"), stop_c=Decimal("0.07"), stop_confirm_s=2.0,
                           hold_s=90.0, barrier_exit="taker", **common)
    return S.run_trade(tape, cost_hi=Decimal("0.55"), tp_c=Decimal("0.06"), hold_s=180.0, barrier_exit="scratch", **common)


def score_cell(triggers: list[dict], tapes: dict, lag: float) -> tuple[list[dict], dict]:
    """Every trigger of one family at one latency through sim_core; (trade rows, skip counts by slice)."""
    rows, skips = [], {"pre": Counter(), "post": Counter()}
    for t in triggers:
        slice_name = "post" if t["close"] > FREEZE_CLOSE else "pre"
        res = run_one(t, tapes[(t["lane"], t["tid"])], lag)
        if isinstance(res, S.Skip):
            skips[slice_name][res.reason] += 1
            continue
        rows.append(RC.trade_row(res, lane=t["lane"], ticker=t["ticker"], close=t["close"], slice_name=slice_name, feat=t.get("feat"),
                                 family=t["family"], latency_s=lag))
    return rows, {k: dict(v) for k, v in skips.items()}


# --------------------------------------------------------------------------------------------------- statistics
def folds_of(closes: list[float]) -> dict[float, int]:
    """Fold number (1-5) of every pre-freeze window, equal counts with the remainder to the earliest folds; post-freeze windows are fold 6."""
    out = {c: 6 for c in closes if c > FREEZE_CLOSE}
    pre = sorted(c for c in closes if c <= FREEZE_CLOSE)
    for k, chunk in enumerate(np.array_split(np.array(pre), FOLDS_PRE), start=1):
        out.update({float(c): k for c in chunk})
    return out


def bootstrap_p(windows: list[float], rows: list[dict], draws: int = DRAWS, seed: int = SEED) -> dict:
    """One-sided percentile bootstrap of the mean net per trade over post-freeze windows; H0 mean <= 0 (registered)."""
    index = {c: i for i, c in enumerate(sorted(set(windows)))}
    net, n = np.zeros(len(index)), np.zeros(len(index))
    for r in rows:
        net[index[r["close"]]] += float(r["pnl"])
        n[index[r["close"]]] += 1
    if n.sum() == 0:
        return {"mean": None, "p": 1.0, "draws": 0}
    chosen = np.random.default_rng(seed).integers(0, len(index), size=(draws, len(index)))
    count = n[chosen].sum(axis=1)
    mean = np.where(count > 0, net[chosen].sum(axis=1) / np.maximum(count, 1), 0.0)
    return {"mean": float(net.sum() / n.sum()), "p": (1 + int((mean <= 0).sum())) / (draws + 1), "draws": draws}


def holm(p_by_cell: dict[str, float], m: int = M, alpha: float = ALPHA) -> dict[str, dict]:
    """Holm step-down over m hypotheses; the cells that were not scored are the m - len(p_by_cell) hypotheses with p = 1 at the end."""
    ordered = sorted(p_by_cell.items(), key=lambda kv: kv[1])
    out, still = {}, True
    for i, (cell, p) in enumerate(ordered, start=1):
        limit = alpha / (m - i + 1)
        still = still and p <= limit
        out[cell] = {"p": p, "threshold": limit, "rejected": still}
    return out


def verdict(post_trades: int, rejected: bool) -> str:
    if post_trades < MIN_TRADES:
        return "underpowered"
    return "edge_confirmed" if rejected else "no_edge_shown"


def short(summary: dict) -> dict:
    return {k: summary[k] for k in ("trades", "win_rate", "net", "pf")}


# ----------------------------------------------------------------------------------------------------------- ledger
def iso(path: Path) -> str | None:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat() if path.exists() else None


def window_ledger(closes: list[float], triggers: dict, scored: dict) -> dict:
    """Which windows each family has seen: pre-freeze windows with the files that show the looks, post-freeze windows and what this run scored."""
    pre = [c for c in closes if c <= FREEZE_CLOSE]
    post = [c for c in closes if c > FREEZE_CLOSE]
    ledger_lines = []
    if RC.LEDGER.exists():
        ledger_lines = [json.loads(x) for x in RC.LEDGER.read_text().splitlines() if x.strip()]
    out = {}
    for fam, files in EVIDENCE.items():
        own = [{"ts": x["ts"], "status": x.get("status")} for x in ledger_lines if x["family"] == fam]
        out[fam] = {
            "pre_freeze": {"windows": len(pre), "first": pre[0], "last": pre[-1], "seen": True,
                           "evidence": [{"path": f, "what": what, "mtime": iso(ROOT / f)} for f, what in files],
                           "oos_ledger_lines": own},
            "post_freeze": {"windows": len(post), "first": post[0], "last": post[-1], "seen_before_this_run": False,
                            "scored_this_run": fam in SCORED, "windows_with_a_trade": sorted({t for t in scored.get(fam, set())})},
        }
    early = [c for c in post if c <= F2_FORWARD_LOOK["inferred_end_close"]]
    out["F2_taker_1s"]["post_freeze"].update({
        "seen_before_this_run": bool(early),
        "forward_look": {**F2_FORWARD_LOOK, "windows_possibly_scored_by_that_look": early, "fitted_on": "nothing",
                         "note": "inferred end, see F2_FORWARD_LOOK; these windows were scored once for F2 before this run but never selected or fitted on"}})
    return out


def ledger_append(entry: dict) -> None:
    RC.LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with RC.LEDGER.open("a") as fh:
        fh.write(json.dumps(entry, default=RC._json_default) + "\n")


def cell_name(family: str, lag: float) -> str:
    return f"{family}@{lag:g}s"


# --------------------------------------------------------------------------------------------------------------- run
def run(tape: Path = TAPE, pool_dir: Path = POOL, out_dir: Path = ROOT / "data") -> dict:
    t0 = time.time()
    pin = T.pin_tape(tape)
    closes = T.window_closes(tape)
    stamp = int(closes[-1])
    receipt_path = out_dir / f"track10-rescore-{stamp}.json"
    if receipt_path.exists():
        raise RuntimeError(f"{receipt_path.name} exists: this pin was already scored; pin a later tape to extend the test")
    post = [c for c in closes if c > FREEZE_CLOSE]
    if not post:
        raise RuntimeError("the pinned tape has no post-freeze window")
    pool_pin = pool_dir / POOL_PIN
    if not pool_pin.exists() or json.loads(pool_pin.read_text())["tape_sha256"] != pin["sha256"]:
        raise RuntimeError(f"the pools in {pool_dir} were not built from this pinned tape; build them with --build-pools into a new directory")
    lanes, norm = RC.load_lanes(), fade_norm()
    triggers = {fam: [] for fam in SCORED}
    for name in LANES:
        pool = load_pool(name, pool_dir)
        triggers["F2_taker_1s"] += f2_triggers(name, lanes[name], pool)
        del pool
        triggers["passive_mm_fade"] += fade_triggers(name, lanes[name], pool_dir, norm)
        gc.collect()
    check = {"F2_taker_1s_pre_freeze_triggers": sum(t["close"] <= FREEZE_CLOSE for t in triggers["F2_taker_1s"]),
             "passive_mm_fade_is_triggers": sum(t["close"] < FADE_IS_CUT for t in triggers["passive_mm_fade"])}
    if check != RECORDED:
        raise RuntimeError(f"the rebuilt pools do not reproduce the recorded triggers: {check} against {RECORDED}")
    tapes = {(t["lane"], t["tid"]): RC.tape(lanes[t["lane"]], t["tid"]) for fam in SCORED for t in triggers[fam]}
    fold_of = folds_of(closes)
    cells = {fam: {} for fam in SCORED}
    for fam in SCORED:                                                       # every look is on the ledger before any scoring
        for lag in LATENCIES:
            ledger_append({"ts": datetime.now(timezone.utc).isoformat(), "family": f"rescore10:{cell_name(fam, lag)}", "status": "started",
                           "config": {"family": fam, "latency_s": lag, "spec_sha256": SPEC_SHA256}, "rescore": True, "rescore_reason": REASON})
    all_rows, p_by_cell = [], {}
    for fam in SCORED:
        for lag in LATENCIES:
            rows, skips = score_cell(triggers[fam], tapes, lag)
            all_rows += rows
            pre_rows, post_rows = [r for r in rows if r["slice"] == "pre"], [r for r in rows if r["slice"] == "post"]
            by_fold = {k: short(RC.summarize([r for r in rows if fold_of[r["close"]] == k])) for k in range(1, FOLDS_PRE + 2)}
            test = bootstrap_p(post, post_rows)
            p_by_cell[cell_name(fam, lag)] = test["p"]
            cells[fam][lag] = {"triggers": {"pre": sum(t["close"] <= FREEZE_CLOSE for t in triggers[fam]),
                                            "post": sum(t["close"] > FREEZE_CLOSE for t in triggers[fam])},
                               "skips": skips, "pre": RC.summarize(pre_rows, skips["pre"]), "post": RC.summarize(post_rows, skips["post"]),
                               "folds": by_fold, "test": test}
    holm_out = holm(p_by_cell)
    for fam in SCORED:
        for lag in LATENCIES:
            c, h = cells[fam][lag], holm_out[cell_name(fam, lag)]
            c["holm"] = h
            c["verdict"] = verdict(c["post"]["trades"], h["rejected"])
            ledger_append({"ts": datetime.now(timezone.utc).isoformat(), "family": f"rescore10:{cell_name(fam, lag)}", "status": "scored",
                           "config": {"family": fam, "latency_s": lag, "spec_sha256": SPEC_SHA256}, "rescore": True, "rescore_reason": REASON,
                           "post": short(c["post"]), "pre": short(c["pre"]), "p": h["p"], "verdict": c["verdict"], "cache": str(tape)})
    scored_windows = {fam: {r["close"] for r in all_rows if r["family"] == fam and r["slice"] == "post"} for fam in SCORED}
    out = {"track": "champion_rescore", "spec_sha256": SPEC_SHA256, "pin_sha256": pin["sha256"], "tape": str(tape),
           "freeze_close": FREEZE_CLOSE, "windows": {"total": len(closes), "post_freeze": len(post), "first_post": post[0], "last": closes[-1]},
           "fold_sizes": dict(sorted(Counter(fold_of.values()).items())), "pool_check": check,
           "triggers": {fam: len(triggers[fam]) for fam in SCORED},
           "not_scored": {k: v["reason_not_scored"] for k, v in SPEC["families"].items() if not v["scored"]},
           "holm": {"m": M, "alpha": ALPHA, "scored_cells": len(p_by_cell)},
           "cells": {fam: {f"{lag:g}": v for lag, v in cells[fam].items()} for fam in SCORED}, "seconds": round(time.time() - t0, 1)}
    wl = window_ledger(closes, triggers, scored_windows)
    RC.dump(out_dir / f"track10-window-ledger-{stamp}.json", wl)
    with (out_dir / f"track10-rescore-trades-{stamp}.jsonl").open("w") as fh:
        for r in all_rows:
            fh.write(json.dumps(r, default=RC._json_default) + "\n")
    RC.dump(receipt_path, out)
    return out


def rebuild_window_ledger(tape: Path = TAPE, out_dir: Path = ROOT / "data") -> dict:
    """Rewrite only the window ledger from the pin and the scored trades (no scoring, no OOS ledger line)."""
    T.pin_tape(tape)
    closes = T.window_closes(tape)
    stamp = int(closes[-1])
    rows = [json.loads(x) for x in (out_dir / f"track10-rescore-trades-{stamp}.jsonl").read_text().splitlines() if x.strip()]
    traded = {fam: {r["close"] for r in rows if r["family"] == fam and r["slice"] == "post"} for fam in SCORED}
    wl = window_ledger(closes, {}, traded)
    RC.dump(out_dir / f"track10-window-ledger-{stamp}.json", wl)
    return wl


def table(out: dict) -> str:
    lines = [f"{'cell':22s} {'pre-freeze n / net / PF':26s} {'post-freeze n / win / net / PF':32s} {'p':>7s} {'Holm limit':>10s}  verdict"]
    for fam, by_lag in out["cells"].items():
        for lag, c in by_lag.items():
            pre, post = c["pre"], c["post"]
            lines.append(f"{cell_name(fam, float(lag)):22s} {pre['trades']:4d} {pre['net']:+9.2f} {pre['pf']:6.2f}      "
                         f"{post['trades']:4d} {post['win_rate']:.3f} {post['net']:+9.2f} {post['pf']:6.2f}   "
                         f"{c['test']['p']:7.4f} {c['holm']['threshold']:10.5f}  {c['verdict']}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tape", type=Path, default=TAPE)
    ap.add_argument("--pool", type=Path, default=POOL)
    ap.add_argument("--build-tape", action="store_true", help="pin the tape (cache snapshot) and stop")
    ap.add_argument("--build-pools", action="store_true", help="build the causal pools for the pinned tape and stop")
    ap.add_argument("--rebuild-ledger", action="store_true", help="rewrite only the window ledger from the pin and the scored trades")
    args = ap.parse_args()
    if args.build_tape:
        print(json.dumps(build_tape(args.tape), indent=1))
        return 0
    if args.build_pools:
        print(json.dumps(build_pools(args.tape, args.pool)))
        return 0
    if args.rebuild_ledger:
        print(json.dumps({k: v["post_freeze"]["seen_before_this_run"] for k, v in rebuild_window_ledger(args.tape).items()}))
        return 0
    out = run(args.tape, args.pool)
    print(table(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
