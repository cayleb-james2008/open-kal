"""Record the Kalshi WebSocket tape into SQLite (books, indices, trades).

Run through a quarter-hour close to capture `last_60s_windowed_average_15min`, which is
the settlement accumulator the final-minute model needs.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openkal.auth import KalshiClient  # noqa: E402
from openkal.config import load_settings  # noqa: E402
from openkal.kalshi_ws import KalshiWS  # noqa: E402
from openkal.store import Store  # noqa: E402

SERIES = ["KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M"]


async def current_tickers(client: KalshiClient) -> list[str]:
    out: list[str] = []
    for series in SERIES:
        status, payload = await client.get("/markets", {"series_ticker": series,
                                                        "status": "open", "limit": 5})
        if status != 200:
            continue
        markets = [m for m in (payload.get("markets") or [])
                   if m.get("status") in {"active", "initialized", "open"}]
        if markets:
            out.append(max(markets, key=lambda m: m.get("close_time", ""))["ticker"])
    return out


async def main(minutes: float = 20.0) -> int:
    settings = load_settings()
    client = KalshiClient(settings)
    store = Store(settings.db_path)
    ws = KalshiWS(settings, client, store=store)
    ws.watch(await current_tickers(client))
    await ws.start()
    deadline = time.time() + minutes * 60
    last_report = 0.0
    while time.time() < deadline:
        await asyncio.sleep(20)
        ws.watch(await current_tickers(client))     # follow the roll to the next window
        if time.time() - last_report > 120:
            last_report = time.time()
            finals = [i for i in ws.indices.values() if i.final_minute is not None]
            print(f"{time.strftime('%H:%M:%S')} msgs={ws.messages} lag={ws.lag_ms:.0f}ms "
                  f"books={len(ws.books)} indices={len(ws.indices)} finals={len(finals)} "
                  f"reconnects={ws.reconnects} degraded={ws.degraded}", flush=True)
    await ws.stop()
    await client.close()
    return 0


if __name__ == "__main__":
    mins = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    raise SystemExit(asyncio.run(main(mins)))
