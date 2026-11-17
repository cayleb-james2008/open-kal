#!/usr/bin/env python3
"""Minute candles (yes bid/ask/price) for settled 15-minute altcoin markets, into
data/candles-15m.jsonl. Only windows whose previous BTC 15-minute move was at least
--btc-min-bp are fetched. That move is known when the alt window opens, so the filter
adds no look-ahead, and BTC's own candles are not needed. Read-only and resumable (a
rerun skips what is already on disk). Deliberately slow: the live engine shares this
account's rate limit.

    .venv/bin/python scripts/fetch_candles.py --btc-min-bp 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from openkal.auth import KalshiClient  # noqa: E402
from openkal.config import load_settings  # noqa: E402

OUT = Path("data/candles-15m.jsonl")


def ts_of(iso: str) -> int:
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())


async def fetch(client: KalshiClient, tk: str, close: int) -> tuple[int, dict]:
    series = tk.split("-")[0]
    status, d = 0, {}
    for attempt in range(4):
        try:
            status, d = await client.get(f"/series/{series}/markets/{tk}/candlesticks",
                                         {"start_ts": close - 900, "end_ts": close,
                                          "period_interval": 1})
        except Exception:  # noqa: BLE001 - a dropped request is retried, not fatal
            status = 0
        if status not in (0, 429):
            break
        await asyncio.sleep(5 * (attempt + 1))
    return status, d


async def main(btc_min_bp: float) -> None:
    markets = json.loads(Path("data/settled-markets.json").read_text())
    btc_bp: dict[int, float] = {}
    for m in markets:
        if m["ticker"].startswith("KXBTC15M"):
            try:
                btc_bp[ts_of(m["close_time"])] = (
                    float(m["expiration_value"]) / float(m["floor_strike"]) - 1) * 1e4
            except (TypeError, ValueError, ZeroDivisionError):
                pass
    done: set[str] = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            try:
                done.add(json.loads(line)["ticker"])
            except Exception:  # noqa: BLE001 - a torn line is just refetched
                pass
    todo = []
    for m in markets:
        if m["ticker"].startswith("KXBTC15M") or m["ticker"] in done:
            continue
        prev = btc_bp.get(ts_of(m["close_time"]) - 900)
        if prev is not None and abs(prev) >= btc_min_bp:
            todo.append(m)
    # newest first, so a partial file already covers the most recent hours
    todo.sort(key=lambda m: m["close_time"], reverse=True)
    print("queued", len(todo), flush=True)
    client = KalshiClient(load_settings())
    n = 0
    try:
        with OUT.open("a") as fh:
            for m in todo:
                tk, close = m["ticker"], ts_of(m["close_time"])
                status, d = await fetch(client, tk, close)
                if status != 200:
                    continue
                fh.write(json.dumps({"ticker": tk, "close": close, "strike": m.get("floor_strike"),
                                     "value": m.get("expiration_value"),
                                     "result": m.get("result"),
                                     "candles": d.get("candlesticks", [])}) + "\n")
                fh.flush()
                n += 1
                if n % 250 == 0:
                    print(n, "fetched", flush=True)
                await asyncio.sleep(0.2)
    finally:
        await client.close()
    print("done", n, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--btc-min-bp", type=float, default=10.0)
    asyncio.run(main(ap.parse_args().btc_min_bp))
