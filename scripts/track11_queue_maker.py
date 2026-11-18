#!/usr/bin/env python3
"""Track 11: a favourite-side bid joined behind a short queue, filled first-in-first-out, scored in sample only.

Registered in scripts/track11_prereg.json (v2, sha pinned in tests/test_track11_queue_maker.py) before any L2 price or outcome was
read. Every order goes through openkal.sim_core.run_trade(entry_style=maker, queue_model=True). The 19 out-of-sample L2 windows are
never indexed, built into a tape, simulated or counted; the receipt says oos_scored null and nothing here writes the OOS ledger.

    flock data/.heavy.lock nice -n 10 .venv/bin/python scripts/track11_queue_maker.py
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
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
from scripts import rnd_common as RC  # noqa: E402
from scripts import track10_champion_rescore as R10  # noqa: E402
from scripts import track4_common as T  # noqa: E402

SPEC_PATH = Path(__file__).with_name("track11_prereg.json")
SPEC_BYTES = SPEC_PATH.read_bytes()
SPEC = json.loads(SPEC_BYTES)
SPEC_SHA256 = hashlib.sha256(SPEC_BYTES).hexdigest()
FAMILY = "queue_short_favourite_bid"
TAPE = ROOT / "data" / "track10-tape"
OUT = ROOT / "data" / "track11-queue-maker.json"
TRADES_OUT = ROOT / "data" / "track11-queue-maker-is-trades.jsonl"
LANES = RC.LANES
DECISION_SECS = (660.0, 540.0, 420.0, 300.0)
BAND_LO, BAND_HI = Decimal("0.40"), Decimal("0.60")
Q_MAX, PATIENCE_S = 100.0, 100.0
N_L2, CUT_EPOCH = SPEC["split"]["is_windows"] + SPEC["split"]["oos_windows"], float(SPEC["split"]["cut_epoch"])
MIN_FILLS, MIN_PF, MIN_LANES, ALPHA = 30, 1.25, 2, 0.05
ARMS = {                                      # name: (queue filter, queue model on)
    "primary": ("short", True),
    "legacy_fill": ("short", False),
    "long_queue": ("long", True),
}


# ---------------------------------------------------------------------------------------------------------- windows
def l2_closes(tape: Path) -> list[float]:
    """Settled window closes where every lane has finite touch sizes on more than half of the window's book rows."""
    meta = json.loads((tape / "markets.json").read_text())
    per_lane = []
    for lane in LANES:
        with np.load(tape / f"book_{lane}.npz", allow_pickle=False) as z:
            names = [str(t) for t in z["tickers"]]
            tid = z["tid"]
            finite = np.isfinite(z["yes_bid_sz"].astype(np.float64)) & np.isfinite(z["yes_ask_sz"].astype(np.float64))
        bounds = np.searchsorted(tid, np.arange(len(names) + 1))
        ok = set()
        for i, name in enumerate(names):
            m = meta.get(name)
            a, b = int(bounds[i]), int(bounds[i + 1])
            if m and m.get("y") in (0, 1) and m.get("strike") and b > a and finite[a:b].mean() > 0.5:
                ok.add(float(m["close"]))
        per_lane.append(ok)
    return sorted(set.intersection(*per_lane))


def split_of(closes: list[float]) -> tuple[float, list[float]]:
    """(cut, in-sample closes) by the registered 60% rule."""
    cut = closes[int(0.6 * len(closes))]
    return cut, [c for c in closes if c < cut]


def load_sizes(tape: Path, lane_name: str) -> tuple[np.ndarray, np.ndarray]:
    """The lane's displayed YES bid and ask sizes, row for row with the book arrays the lane was loaded from."""
    with np.load(tape / f"book_{lane_name}.npz", allow_pickle=False) as z:
        return z["yes_bid_sz"].astype(np.float32), z["yes_ask_sz"].astype(np.float32)


def sized_tape(lane, tid: int, sizes: tuple[np.ndarray, np.ndarray]) -> S.Tape:
    """The lane's tape for one ticker with its touch sizes attached (sliced exactly like the book rows)."""
    a, b = int(lane.book_bounds[tid]), int(lane.book_bounds[tid + 1])
    return dataclasses.replace(RC.tape(lane, tid), yes_bid_sz=sizes[0][a:b], yes_ask_sz=sizes[1][a:b])


# ---------------------------------------------------------------------------------------------------------- orders
def decide(tape: S.Tape, secs: float, arm: str):
    """One decision point: ('skip', reason) or ('order', side, qty, trigger_ts)."""
    trigger = tape.close_ts - secs
    try:
        bid, ask = tape.quote(trigger + S.LAG_S)
    except S.CrossedBook:
        return "skip", "crossed"
    except S.NoQuote:
        return "skip", "no_quote"
    yes_bid, no_bid = S.px(bid), S.ONE - S.px(ask)
    if yes_bid == no_bid:
        return "skip", "tie"
    side, fav = (S.YES, yes_bid) if yes_bid > no_bid else (S.NO, no_bid)
    if not BAND_LO <= fav <= BAND_HI:
        return "skip", "band"
    ahead = tape.queue_ahead(trigger + S.LAG_S, side, fav)
    if ahead is None:
        return "skip", "queue_unknown"
    wanted = ARMS[arm][0]
    if (ahead <= Q_MAX) != (wanted == "short"):
        return "skip", "queue_long" if wanted == "short" else "queue_short"
    return "order", side, S.stake_qty(fav, RC.STAKE), trigger


def run_window(tape: S.Tape, arm: str, counts: Counter):
    """The four decision points of one lane-window; at most one position (the first order that fills)."""
    for secs in DECISION_SECS:
        d = decide(tape, secs, arm)
        if d[0] == "skip":
            counts[d[1]] += 1
            continue
        _, side, qty, trigger = d
        counts["placed"] += 1
        res = S.run_trade(tape, side=side, trigger_ts=trigger, entry_style=S.MAKER, qty=qty, patience_s=PATIENCE_S,
                          queue_model=ARMS[arm][1])
        if isinstance(res, S.Skip):
            counts[res.reason] += 1
            continue
        counts["filled"] += 1
        return res, secs
    return None, None


# ---------------------------------------------------------------------------------------------------------- verdict
def bar(summary: dict, p: float) -> dict:
    """The registered in-sample bar; every check must hold. Clearing it spends no OOS look."""
    checks = {"at_least_30_fills": summary["trades"] >= MIN_FILLS, "net_positive": summary["net"] > 0,
              "pf_at_least_1_25": summary["pf"] >= MIN_PF, "two_profitable_lanes": summary["profitable_lanes"] >= MIN_LANES,
              "bootstrap_p_below_0_05": p < ALPHA}
    checks["eligible"] = all(checks.values())
    return checks


def arm_report(rows: list[dict], counts: Counter, closes: list[float]) -> dict:
    s = RC.summarize(rows, {k: v for k, v in counts.items() if k not in ("placed", "filled")})
    test = R10.bootstrap_p(closes, rows)
    return {"summary": s, "decision_points": dict(sorted(counts.items())), "test": test, "bar": bar(s, test["p"])}


# --------------------------------------------------------------------------------------------------------------- run
def run(tape: Path = TAPE, out: Path = OUT, trades_out: Path = TRADES_OUT) -> dict:
    t0 = time.time()
    T.refuse_if_spent(FAMILY, out)
    T.refuse_rescore()
    if out.exists():
        raise RuntimeError(f"{out.name} exists: the registered run is a single configuration and was already made")
    pin = T.pin_tape(tape)
    closes = l2_closes(tape)
    cut, is_closes = split_of(closes)
    if (len(closes), cut, len(is_closes)) != (N_L2, CUT_EPOCH, SPEC["split"]["is_windows"]):
        raise RuntimeError(f"the pinned tape's L2 windows ({len(closes)}, cut {cut}) are not the registered ones ({N_L2}, cut {CUT_EPOCH})")
    in_sample = set(is_closes)
    lanes = RC.load_lanes()
    arms = {name: {"rows": [], "counts": Counter()} for name in ARMS}
    for lane_name in LANES:
        lane = lanes[lane_name]
        sizes = load_sizes(tape, lane_name)
        for tid, ticker in enumerate(lane.tickers):
            meta = lane.meta.get(ticker)
            if meta is None or meta["close"] not in in_sample:
                continue                                              # out-of-sample and non-L2 windows are never indexed into a tape
            tp = sized_tape(lane, tid, sizes)
            for name, store in arms.items():
                res, secs = run_window(tp, name, store["counts"])
                if res is not None:
                    store["rows"].append(RC.trade_row(res, lane=lane_name, ticker=ticker, close=float(meta["close"]), slice_name="is",
                                                      family=FAMILY, arm=name, decision_secs=secs))
        del sizes
        gc.collect()
    report = {name: arm_report(store["rows"], store["counts"], is_closes) for name, store in arms.items()}
    result = {"family": FAMILY, "spec_sha256": SPEC_SHA256, "pin_sha256": pin["sha256"], "cut": cut,
              "windows": {"l2": len(closes), "in_sample": len(is_closes), "oos_sealed": len(closes) - len(is_closes)},
              "arms": report, "primary_eligible": report["primary"]["bar"]["eligible"],
              "oos_scored": None, "seconds": round(time.time() - t0, 1)}
    with trades_out.open("w") as fh:
        for store in arms.values():
            for r in store["rows"]:
                fh.write(json.dumps(r, default=RC._json_default) + "\n")
    RC.dump(out, T.clean(result))
    return result


def table(result: dict) -> str:
    lines = [f"{'arm':12s} {'placed':>6s} {'fills':>5s} {'win':>5s} {'net':>9s} {'PF':>6s} {'lanes+':>6s} {'p':>7s}  eligible"]
    for name, a in result["arms"].items():
        s, d = a["summary"], a["decision_points"]
        lines.append(f"{name:12s} {d.get('placed', 0):6d} {s['trades']:5d} {s['win_rate']:5.2f} {s['net']:+9.2f} {s['pf']:6.2f} "
                     f"{s['profitable_lanes']:6d} {a['test']['p']:7.4f}  {a['bar']['eligible']}")
    return "\n".join(lines)


def main() -> int:
    argparse.ArgumentParser().parse_args()
    out = run()
    print(table(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
