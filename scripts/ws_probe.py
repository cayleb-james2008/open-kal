"""Probe the authenticated Kalshi WebSocket plane before building on it.

Answers three questions with evidence, not documentation:
  1. does `/indexlist` expose the crypto RTI indices we need?
  2. does the signed handshake to `wss://external-api-ws.kalshi.com/trade-api/ws/v2` connect?
  3. which channels deliver what, and what do their messages look like?
"""
from __future__ import annotations

import asyncio
import base64
import json
import sys
import time
from pathlib import Path

import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openkal.auth import KalshiClient  # noqa: E402
from openkal.config import load_settings  # noqa: E402

WS_URL = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
WS_PATH = "/trade-api/ws/v2"


def ws_headers(client: KalshiClient) -> dict[str, str]:
    ts = str(int(time.time() * 1000))
    sig = base64.b64encode(
        client._sign((ts + "GET" + WS_PATH).encode())   # noqa: SLF001 - same signer as REST
    ).decode()
    return {
        "KALSHI-ACCESS-KEY": client.settings.api_key_id,
        "KALSHI-ACCESS-SIGNATURE": sig,
        "KALSHI-ACCESS-TIMESTAMP": ts,
    }


async def main() -> int:
    settings = load_settings()
    client = KalshiClient(settings)
    for path in ("/indexlist", "/indices", "/series/indexlist", "/cfbenchmarks/indexlist"):
        status, payload = await client.get(path)
        print(path, status, json.dumps(payload)[:200])

    st, m = await client.get("/markets", {"series_ticker": "KXBTC15M", "status": "open", "limit": 1})
    ticker = (m.get("markets") or [{}])[0].get("ticker")
    print("ticker", ticker)

    async with websockets.connect(WS_URL, additional_headers=ws_headers(client),
                                  open_timeout=15, ping_interval=20) as ws:
        print("connected")
        subs = [
            {"id": 1, "cmd": "subscribe", "params": {"channels": ["cfbenchmarks_value_5hz"],
                                                     "market_tickers": [ticker]}},
            {"id": 2, "cmd": "subscribe", "params": {"channels": ["cfbenchmarks_value"],
                                                     "market_tickers": [ticker]}},
            {"id": 3, "cmd": "subscribe", "params": {"channels": ["orderbook_delta"],
                                                     "market_tickers": [ticker]}},
            {"id": 4, "cmd": "subscribe", "params": {"channels": ["trade"],
                                                     "market_tickers": [ticker]}},
            {"id": 5, "cmd": "subscribe", "params": {"channels": ["fill", "user_orders",
                                                                  "market_positions"]}},
            {"id": 6, "cmd": "subscribe", "params": {"channels": ["market_lifecycle_v2"]}},
        ]
        for s in subs:
            await ws.send(json.dumps(s))
        seen: dict[str, int] = {}
        deadline = time.time() + 25
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
            except asyncio.TimeoutError:
                print("... wait")
                continue
            msg = json.loads(raw)
            t = msg.get("type", "?")
            seen[t] = seen.get(t, 0) + 1
            if seen[t] <= 2:
                print(t, json.dumps(msg)[:420])
            if msg.get("type") == "error":
                print("ERROR", json.dumps(msg)[:300])
        print("counts", seen)
    await client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
