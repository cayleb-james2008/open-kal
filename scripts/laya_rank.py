"""Does the RLCD-trained Laya checkpoint isolate a high-win-rate subset, out of sample?

The checkpoint in `data/laya-rlcd/model` was fine-tuned on 13,755 labels and reported
0.985 accuracy / ECE 0.009 on its 400-item calibration slice. That number is not
evidence of skill, because the training text contains the outcome:

    "lane KXBTC15M, model probability 0.006, book mid 0.235, kernel edge +0.2244
     dollars, realised 30s post-fee return +0.0326 dollars per contract."

The label is `enter` when that last number is positive, so the model can read the
answer off the prompt. This script tests the checkpoint the only way that means
anything:

  1. build a *fresh* labelled slice from ticks recorded after the training file was
     written (genuinely unseen), with a leakage-free state (no realised return);
  2. score it in batch with the fine-tuned checkpoint;
  3. fit Platt scaling on the first 400 items (the "calibration" slice) and evaluate
     rank deciles on the remaining items (the frozen test slice);
  4. report whether the top decile has a higher win rate and positive post-fee P&L.

Latency is measured per item: a 1.7s CPU forward pass cannot sit in the entry path,
whatever the ranking says.

    .venv/bin/python scripts/laya_rank.py --probe     # 8 items, print + timing
    .venv/bin/python scripts/laya_rank.py             # full 400 + test split
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openkal import features  # noqa: E402
from openkal.fees import taker_fee  # noqa: E402

DB = ROOT / "data" / "openkal-prod.db"
MODEL = ROOT / "data" / "laya-rlcd" / "model"
OUT = ROOT / "data" / "laya-rank.json"
# `labels.csv` is the exact file the fine-tune read, so its write time is exactly the
# boundary of what the model could have seen. Taken from the file, not pasted: a
# hard-coded wall clock in the wrong timezone silently turns training rows into
# "out-of-sample" rows, which is the one mistake this study cannot afford.
TRAIN_CUTOFF = (ROOT / "data" / "laya-rlcd" / "labels.csv").stat().st_mtime

QUESTION = {
    "open": {
        "type": "choice",
        "instructions": (
            "A market-making bot is considering opening the position described in the "
            "state. Should it open now, or skip this trade?"
        ),
        "criteria": {
            "enter": "Open the position now: the model edge is real and the book supports it.",
            "skip": "Skip this trade: the edge looks like noise, the spread or the timing is bad.",
        },
    }
}


@dataclass
class Item:
    ts: float
    series: str
    ticker: str
    p_model: float
    mid: float
    spread: float
    edge: float
    secs_left: float
    ret_30s: float
    label: int
    state: str


def load_items(cutoff: float, per_lane: int) -> list[Item]:
    """Fresh, leakage-free items: every input is knowable at the decision tick."""
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=30)
    db.row_factory = sqlite3.Row
    rows = db.execute(
        "SELECT ts,series,ticker,yes_bid,yes_ask,spot,p_model,edge,secs_left,spread"
        " FROM ticks WHERE ts>? AND yes_ask IS NOT NULL ORDER BY series,ts", (cutoff,)
    ).fetchall()
    db.close()
    by_series: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        by_series.setdefault(row["series"], []).append(row)
    out: list[Item] = []
    for series, series_rows in sorted(by_series.items()):
        picked = 0
        i = 0
        while i < len(series_rows) and picked < per_lane:
            row = series_rows[i]
            try:
                bid, ask = Decimal(row["yes_bid"]), Decimal(row["yes_ask"])
                if not (Decimal("0") < bid < ask < Decimal("1")):
                    i += 1
                    continue
                mid = (bid + ask) / 2
                p = float(row["p_model"] or 0.5)
                side = "YES" if p >= float(mid) else "NO"
                ask_side = ask if side == "YES" else (Decimal(1) - bid)
                if ask_side <= Decimal("0.02"):
                    i += 1
                    continue
                later = next((r for r in series_rows[i:] if r["ts"] - row["ts"] >= 30.0), None)
                if later is None:
                    break
                mid_later = (Decimal(later["yes_bid"]) + Decimal(later["yes_ask"])) / 2
                mark = mid_later if side == "YES" else (Decimal(1) - mid_later)
                fee = taker_fee(ask_side, 1)
                net = float(mark - ask_side - fee)
            except Exception:  # noqa: BLE001 - a malformed row is not a training item
                i += 1
                continue
            state = (
                f"lane {series}, model probability {p:.3f}, book mid {float(mid):.3f}, "
                f"spread {float(ask - bid):.3f} dollars, kernel edge {float(row['edge'] or 0):+.4f} "
                f"dollars, {float(row['secs_left'] or 0):.0f} seconds until expiry. "
                f"Recent lane record: 0 wins and 0 losses, calibration error 0.000."
            )
            out.append(Item(ts=float(row["ts"]), series=series, ticker=row["ticker"],
                            p_model=p, mid=float(mid), spread=float(ask - bid),
                            edge=float(row["edge"] or 0), secs_left=float(row["secs_left"] or 0),
                            ret_30s=net, label=1 if net > 0 else 0, state=state))
            picked += 1
            # One item per 30s per lane: consecutive ticks are the same trade, and
            # 5Hz rows would make a 400-item set eight minutes long.
            while i < len(series_rows) and series_rows[i]["ts"] - row["ts"] < 30.0:
                i += 1
    out.sort(key=lambda x: x.ts)
    return out


def score(items: list[Item], batch_size: int) -> list[float]:
    from laya import Agent

    agent = Agent(str(MODEL), device="cpu")
    states = [it.state for it in items]
    started = time.perf_counter()
    results = agent.predict_batch(states, QUESTION, batch_size=batch_size)
    elapsed = time.perf_counter() - started
    print(f"scored {len(items)} items in {elapsed:.1f}s "
          f"({elapsed / max(1, len(items)) * 1000:.0f} ms/item)", flush=True)
    out: list[float] = []
    for res in results:
        answer = (res.get("answers") or {}).get("open") or {}
        probs = answer.get("probabilities") or {}
        out.append(float(probs.get("enter") or 0.0))
    return out


def fit_platt(ps: list[float], ys: list[int]) -> tuple[float, float]:
    """Newton/IRLS on logit(p) -> y; fixed iteration count, no randomness."""
    xs = [math.log(min(1 - 1e-6, max(1e-6, p)) / (1 - min(1 - 1e-6, max(1e-6, p)))) for p in ps]
    a, b = 1.0, 0.0
    for _ in range(50):
        ga = gb = haa = hab = hbb = 0.0
        for x, y in zip(xs, ys):
            z = max(-30.0, min(30.0, a * x + b))
            p = 1.0 / (1.0 + math.exp(-z))
            w = max(p * (1 - p), 1e-6)
            ga += (p - y) * x
            gb += p - y
            haa += w * x * x
            hab += w * x
            hbb += w
        det = haa * hbb - hab * hab
        if abs(det) < 1e-12:
            break
        a -= max(-0.5, min(0.5, (hbb * ga - hab * gb) / det))
        b -= max(-0.5, min(0.5, (haa * gb - hab * ga) / det))
        a = max(0.05, min(12.0, a))
        b = max(-8.0, min(8.0, b))
    return a, b


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x)))) if x < 30 else 1.0


def apply_platt(p: float, a: float, b: float) -> float:
    q = min(1 - 1e-6, max(1e-6, p))
    return sigmoid(a * math.log(q / (1 - q)) + b)


def deciles(ps: list[float], items: list[Item]) -> list[dict]:
    order = np.argsort(np.array(ps))
    out = []
    n = len(order)
    for d in range(10):
        lo = int(d * n / 10)
        hi = int((d + 1) * n / 10)
        idx = order[lo:hi]
        if len(idx) == 0:
            continue
        sel = [items[i] for i in idx]
        wins = sum(it.label for it in sel)
        pnl = sum((Decimal(str(it.ret_30s)) for it in sel), Decimal("0"))
        out.append({"decile": d + 1, "n": len(sel), "wins": wins,
                    "win_rate": round(wins / len(sel), 3),
                    "mean_p": round(float(np.mean([ps[i] for i in idx])), 3),
                    "net_per_contract": round(float(pnl / len(sel)), 4)})
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--per-lane", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    items = load_items(TRAIN_CUTOFF, args.per_lane)
    print(f"{len(items)} unseen items after the training cutoff "
          f"({len({i.series for i in items})} lanes)", flush=True)
    if not items:
        print("no post-training ticks: cannot score out of sample")
        return 1
    if args.probe:
        items = items[:8]
    ps = score(items, args.batch_size)
    ys = [it.label for it in items]

    if args.probe:
        for it, p in zip(items, ps):
            print(f"{it.series} p_model={it.p_model:.3f} laya_enter={p:.3f} "
                  f"ret30={it.ret_30s:+.4f} label={it.label}")
        return 0

    cut = min(400, len(items) // 2)
    calib, test = items[:cut], items[cut:]
    a, b = fit_platt(ps[:cut], ys[:cut])
    rescaled = [apply_platt(p, a, b) for p in ps]
    # Brier sits next to ECE because they fail differently: Platt scaling can
    # improve the binned gap while making the per-item error worse, and the
    # frozen slice is the only place that shows up.
    brier_raw = features.brier(list(zip(ps, ys)))
    brier_raw_test = features.brier(list(zip(ps[cut:], ys[cut:])))
    brier_rescaled_test = features.brier(list(zip(rescaled[cut:], ys[cut:])))
    payload = {
        "generated_ts": time.time(),
        "model": str(MODEL),
        "training_cutoff": TRAIN_CUTOFF,
        "items": len(items),
        "calibration_items": len(calib),
        "test_items": len(test),
        "base_rate": round(sum(ys) / len(ys), 3),
        "raw": {
            "accuracy": round(sum(1 for p, y in zip(ps, ys) if (p >= 0.5) == bool(y)) / len(ys), 3),
            "ece": round(features.expected_calibration_error(list(zip(ps, ys))), 4),
            "brier": round(brier_raw, 4),
            "mean_p": round(float(np.mean(ps)), 3),
        },
        "platt": {"a": round(a, 4), "b": round(b, 4)},
        "test": {
            "raw": {
                "accuracy": round(sum(1 for p, y in zip(ps[cut:], ys[cut:])
                                    if (p >= 0.5) == bool(y)) / max(1, len(test)), 3),
                "ece": round(features.expected_calibration_error(
                    list(zip(ps[cut:], ys[cut:]))), 4),
                "brier": round(brier_raw_test, 4),
                "deciles": deciles(ps[cut:], test),
            },
            "rescaled": {
                "ece": round(features.expected_calibration_error(
                    list(zip(rescaled[cut:], ys[cut:]))), 4),
                "brier": round(brier_rescaled_test, 4),
                "deciles": deciles(rescaled[cut:], test),
            },
            "brier_lift": round(brier_raw_test - brier_rescaled_test, 4),
        },
        "verdict": None,
    }
    top = payload["test"]["rescaled"]["deciles"][-1]
    payload["verdict"] = {
        "top_decile_win_rate": top["win_rate"],
        "top_decile_net_per_contract": top["net_per_contract"],
        "usable_as_gate": bool(top["win_rate"] > payload["base_rate"] + 0.05
                               and top["net_per_contract"] > 0),
        "latency_note": "CPU forward pass measured per item; see stdout",
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
