"""Spot feeds: Coinbase primary, Kraken cross-confirmation, REST fallback.

BRTI's constituents include Coinbase and Kraken (not Binance), so the lead-lag
signal has to come from those two and not from a composite that includes a
venue the index ignores.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import math
import time
from typing import Callable

import httpx
import websockets

COINBASE_WS = "wss://ws-feed.exchange.coinbase.com"
KRAKEN_WS = "wss://ws.kraken.com"
COINBASE_REST = "https://api.exchange.coinbase.com/products/{}/ticker"

KRAKEN_PAIRS = {
    "BTC-USD": "XBT/USD",
    "ETH-USD": "ETH/USD",
    "SOL-USD": "SOL/USD",
    "XRP-USD": "XRP/USD",
}


def _num(value: object) -> float | None:
    """A venue field as a positive finite float, or None when it is not one."""
    try:
        out = float(value)          # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) and out > 0 else None


class SpotFeed:
    """Two sockets and a REST fallback: a last price and a touch per product."""

    def __init__(self, symbols: list[str],
                 on_price: Callable[..., None]) -> None:
        self.symbols = symbols
        self.on_price = on_price
        # `on_price` historically took (symbol, price, source). The venue's touch is
        # part of a spot print now (Coinbase `best_bid`/`best_ask`, Kraken `b`/`a`),
        # so a five-argument consumer gets it and a three-argument one keeps working.
        try:
            self._wide = len(inspect.signature(on_price).parameters) >= 5
        except (TypeError, ValueError):
            self._wide = False
        self.last: dict[str, tuple[float, float]] = {}      # symbol -> (epoch, price)
        # symbol -> source -> (epoch, price). Fair value must be priced off the
        # freshest constituent print, not whichever socket wrote last.
        self.stamps: dict[str, dict[str, tuple[float, float]]] = {}
        # symbol -> source -> (epoch, bid, ask), for persistence and basis studies.
        self.touches: dict[str, dict[str, tuple[float, float | None, float | None]]] = {}
        self.kraken: dict[str, float] = {}
        self.connected = {"coinbase": False, "kraken": False}
        self.messages = 0
        self.errors: list[str] = []
        self._tasks: list[asyncio.Task] = []
        self._stop = asyncio.Event()

    def price(self, symbol: str) -> tuple[float, float] | None:
        return self.last.get(symbol)

    def freshest(self, symbol: str) -> tuple[float, float, str] | None:
        """Newest (price, age_s, source) across Coinbase, Kraken and the REST backstop."""
        rows = self.stamps.get(symbol) or {}
        if not rows:
            row = self.last.get(symbol)
            return None if row is None else (row[1], max(0.0, time.time() - row[0]), "unknown")
        source, (ts, price) = max(rows.items(), key=lambda kv: kv[1][0])
        return price, max(0.0, time.time() - ts), source

    def age(self, symbol: str) -> float:
        row = self.last.get(symbol)
        return 1e9 if row is None else max(0.0, time.time() - row[0])

    def disagreement_bps(self, symbol: str) -> float | None:
        rows = self.stamps.get(symbol) or {}
        cb, kr = rows.get("coinbase"), rows.get("kraken")
        if not cb or not kr or cb[1] <= 0:
            return None
        return abs(cb[1] - kr[1]) / cb[1] * 10_000

    async def start(self) -> None:
        self._tasks = [
            asyncio.create_task(self._coinbase_loop(), name="coinbase-ws"),
            asyncio.create_task(self._kraken_loop(), name="kraken-ws"),
            asyncio.create_task(self._rest_loop(), name="spot-rest"),
        ]

    async def stop(self) -> None:
        self._stop.set()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _rest_loop(self) -> None:
        """Backstop: if both sockets are silent, poll Coinbase's public ticker."""
        async with httpx.AsyncClient(timeout=6.0) as client:
            while not self._stop.is_set():
                try:
                    stale = [s for s in self.symbols if self.age(s) > 12.0]
                    for symbol in stale:
                        r = await client.get(COINBASE_REST.format(symbol))
                        if r.status_code == 200:
                            price = float(r.json()["price"])
                            self._set(symbol, price, "rest")
                except Exception as exc:  # noqa: BLE001 - a backstop must never kill the loop
                    self.errors.append(f"rest {type(exc).__name__}")
                    self.errors = self.errors[-8:]
                await asyncio.sleep(3.0)

    def _set(self, symbol: str, price: float, source: str,
             bid: float | None = None, ask: float | None = None) -> None:
        if not (price > 0 and math.isfinite(price)):
            return
        now = time.time()
        self.last[symbol] = (now, price)
        self.stamps.setdefault(symbol, {})[source] = (now, price)
        self.touches.setdefault(symbol, {})[source] = (now, bid, ask)
        self.messages += 1
        if self._wide:
            self.on_price(symbol, price, source, bid, ask)
        else:
            self.on_price(symbol, price, source)

    def touch(self, symbol: str, source: str,
              ) -> tuple[float, float | None, float | None] | None:
        """Newest (epoch, bid, ask) one venue published for `symbol`."""
        return (self.touches.get(symbol) or {}).get(source)

    async def _coinbase_loop(self) -> None:
        while not self._stop.is_set():
            try:
                async with websockets.connect(COINBASE_WS, open_timeout=10, ping_interval=20) as ws:
                    await ws.send(json.dumps({
                        "type": "subscribe",
                        "product_ids": self.symbols,
                        "channels": ["ticker"],
                    }))
                    self.connected["coinbase"] = True
                    async for raw in ws:
                        msg = json.loads(raw)
                        if msg.get("type") == "ticker":
                            self._set(msg["product_id"], float(msg["price"]), "coinbase",
                                      _num(msg.get("best_bid")), _num(msg.get("best_ask")))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self.connected["coinbase"] = False
                self.errors.append(f"coinbase {type(exc).__name__}")
                self.errors = self.errors[-8:]
                await asyncio.sleep(2.0)

    async def _kraken_loop(self) -> None:
        pairs = [KRAKEN_PAIRS[s] for s in self.symbols if s in KRAKEN_PAIRS]
        if not pairs:
            return
        reverse = {v: k for k, v in KRAKEN_PAIRS.items()}
        while not self._stop.is_set():
            try:
                async with websockets.connect(KRAKEN_WS, open_timeout=10) as ws:
                    await ws.send(json.dumps({"event": "subscribe", "pair": pairs,
                                              "subscription": {"name": "ticker"}}))
                    self.connected["kraken"] = True
                    async for raw in ws:
                        msg = json.loads(raw)
                        # A Kraken ticker frame is [channelID, data, name, pair] on this
                        # endpoint. The shipped parser assumed the pair came before the
                        # name, so it never matched a single message: Kraken has been
                        # invisible to `stamps` and `freshest` since day one, which is
                        # why its readiness row read 1e9 ms. Accept either order rather
                        # than betting on which one the venue sends.
                        if (isinstance(msg, list) and len(msg) >= 4
                                and "ticker" in (msg[-2], msg[-1])):
                            pair = msg[-2] if msg[-1] == "ticker" else msg[-1]
                            symbol = reverse.get(pair)
                            if symbol:
                                frame = msg[1]
                                price = float(frame["c"][0])
                                self.kraken[symbol] = price
                                # `b`/`a` are [price, whole_lot_volume, lot_volume].
                                # The touch is carried with the print so the spot
                                # tape can be joined to a venue's book, not only to
                                # its last trade.
                                try:
                                    bid, ask = _num(frame["b"][0]), _num(frame["a"][0])
                                except (KeyError, IndexError, TypeError):
                                    bid = ask = None
                                # Publish it like every other source. Kraken is a BRTI
                                # constituent, and `freshest()` reads `stamps`: recording
                                # the price only in `self.kraken` kept the venue out of the
                                # spot tape and made its freshness unmeasurable.
                                self._set(symbol, price, "kraken", bid, ask)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self.connected["kraken"] = False
                self.errors.append(f"kraken {type(exc).__name__}")
                self.errors = self.errors[-8:]
                await asyncio.sleep(3.0)
