"""Build the sub-second book + print cache used by `scripts/micro_study.py`.

The 1 Hz `ticks` table is a decision log, not a fill tape: when the websocket book
loses its ask side the engine used to fall back to the REST snapshot, which can be
40 seconds stale. Every honest measurement therefore has to read `ws_book` (the
venue's own ladder, every message) and `ws_trade` (its print tape) directly.

    .venv/bin/python scripts/book_cache.py
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.micro_study import LANES, MARKETS, connect  # noqa: E402

CACHE = ROOT / "data" / "evolve-cache"


def _f(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def build(only: list[str] | None = None) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    db = connect()
    markets = {}
    for m in json.loads(MARKETS.read_text()):
        if m.get("result") not in ("yes", "no"):
            continue
        markets[m["ticker"]] = m
    tickers = [r[0] for r in db.execute("SELECT DISTINCT ticker FROM ticks")]
    for lane in LANES:
        if only and lane not in only:
            continue
        if (CACHE / f"book_{lane}.npz").exists():
            print(f"{lane}: cache exists, skipping", flush=True)
            continue
        names = sorted(t for t in tickers if t.startswith(lane + "-") and t in markets)
        rows_ts: list[np.ndarray] = []
        rows_bid: list[np.ndarray] = []
        rows_ask: list[np.ndarray] = []
        rows_tid: list[np.ndarray] = []
        tr_ts: list[np.ndarray] = []
        tr_px: list[np.ndarray] = []
        tr_sz: list[np.ndarray] = []
        tr_ty: list[np.ndarray] = []
        tr_tid: list[np.ndarray] = []
        for i, name in enumerate(names):
            book = db.execute(
                "SELECT ts,yes_bid,yes_ask FROM ws_book WHERE ticker=? ORDER BY ts", (name,)
            ).fetchall()
            # A ticker's rows are the peak of this loop: hand them to numpy and let
            # the row objects go before the next ticker is pulled.
            gc.collect()
            if not book:
                continue
            rows_ts.append(np.array([r["ts"] for r in book], dtype=np.float64))
            rows_bid.append(np.array([_f(r["yes_bid"]) for r in book], dtype=np.float32))
            rows_ask.append(np.array([_f(r["yes_ask"]) for r in book], dtype=np.float32))
            rows_tid.append(np.full(len(book), i, dtype=np.int16))
            trades = db.execute(
                "SELECT ts,yes_price,count,taker_side FROM ws_trade WHERE ticker=? ORDER BY ts",
                (name,)).fetchall()
            if trades:
                tr_ts.append(np.array([r["ts"] for r in trades], dtype=np.float64))
                tr_px.append(np.array([_f(r["yes_price"]) for r in trades], dtype=np.float32))
                tr_sz.append(np.array([_f(r["count"]) for r in trades], dtype=np.float32))
                tr_ty.append(np.array([1 if r["taker_side"] == "yes" else 0 for r in trades],
                                      dtype=np.int8))
                tr_tid.append(np.full(len(trades), i, dtype=np.int16))
            if i % 10 == 0:
                print(f"{lane}: {i}/{len(names)} tickers", flush=True)
        np.savez_compressed(
            CACHE / f"book_{lane}.npz",
            tickers=np.array(names),
            ts=np.concatenate(rows_ts), bid=np.concatenate(rows_bid),
            ask=np.concatenate(rows_ask), tid=np.concatenate(rows_tid),
            trade_ts=np.concatenate(tr_ts), trade_px=np.concatenate(tr_px),
            trade_sz=np.concatenate(tr_sz), trade_ty=np.concatenate(tr_ty),
            trade_tid=np.concatenate(tr_tid),
        )
        print(f"{lane}: {len(names)} tickers, {sum(len(x) for x in rows_ts):,} book rows, "
              f"{sum(len(x) for x in tr_ts):,} prints", flush=True)
        del rows_ts, rows_bid, rows_ask, rows_tid, tr_ts, tr_px, tr_sz, tr_ty, tr_tid
        gc.collect()
    db.close()


if __name__ == "__main__":
    build(sys.argv[1:] or None)
