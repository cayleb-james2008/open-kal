"""Live venue orders, behind the readiness gate.

Only `openkal.engine` may construct a broker, and only after `readiness.evaluate`
returns ready. Every order is post-only and carries a `client_order_id` we can
reconcile, plus `cancel_order_on_pause` so a paused exchange cancels rather than
leaves a live resting bid.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from .auth import KalshiClient


def _count(value: Any) -> int:
    """Venue counts arrive as fixed-point strings like "3.00"."""
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _cents(price: Decimal) -> str:
    return f"{Decimal(price).quantize(Decimal('0.0001')):f}"


@dataclass
class VenueOrder:
    order_id: str
    client_order_id: str
    ticker: str
    side: str
    action: str
    price: Decimal
    contracts: int
    status: str
    filled: int = 0


class Broker:
    def __init__(self, client: KalshiClient) -> None:
        self.client = client
        self.group_id: str | None = None
        self.errors: list[str] = []

    async def balance(self) -> Decimal:
        status, payload = await self.client.get("/portfolio/balance")
        if status != 200:
            return Decimal("0")
        value = payload.get("balance_dollars") or payload.get("balance")
        if value is None:
            return Decimal("0")
        return Decimal(str(value)) if "." in str(value) else Decimal(str(value)) / Decimal(100)

    async def ensure_group(self) -> str | None:
        # Order groups are v1-only and the v1 write surface is retired (410). The
        # per-order `cancel_order_on_pause` flag gives the same protection.
        return None

    async def place(self, ticker: str, side: str, action: str, price: Decimal,
                    contracts: int, post_only: bool = True,
                    tif: str = "good_till_canceled",
                    reduce_only: bool = False) -> VenueOrder | None:
        """`side` is the position side (YES/NO); `action` is entry or exit."""
        bid_ask = self._bid_ask(side, action)
        group = await self.ensure_group()
        body: dict[str, Any] = {
            "ticker": ticker,
            "client_order_id": str(uuid.uuid4()),
            "side": bid_ask,
            "count": str(int(contracts)),
            "price": _cents(price),
            "time_in_force": tif,
            "post_only": post_only,
            "self_trade_prevention_type": "taker_at_cross",
            "cancel_order_on_pause": True,
        }
        if reduce_only:
            body["reduce_only"] = True
        if group:
            body["order_group_id"] = group
        status, payload = await self.client.post("/portfolio/events/orders", body)
        if status not in (200, 201):
            self.errors.append(f"{status} {str(payload)[:160]}")
            return None
        order = payload.get("order") or payload
        return VenueOrder(
            order_id=order.get("order_id", ""),
            client_order_id=body["client_order_id"], ticker=ticker, side=side,
            action=action, price=price, contracts=contracts,
            status=order.get("status", "resting"), filled=_count(order.get("fill_count")),
        )

    async def cancel(self, order_id: str, market_ticker: str | None = None) -> bool:
        """V2 cancel. The shard is ambiguous without market_ticker, so always pass it."""
        params = {"exchange_index": -1}
        if market_ticker:
            params["market_ticker"] = market_ticker
        status, payload = await self.client.delete(f"/portfolio/events/orders/{order_id}", params)
        if status in (200, 201, 204):
            return True
        self.errors.append(f"cancel {status} {str(payload)[:120]}")
        return False

    async def open_orders(self) -> dict[str, dict[str, Any]]:
        """Resting orders by id. This is the read surface that actually resolves on V2."""
        out: dict[str, dict[str, Any]] = {}
        status, payload = await self.client.get("/portfolio/orders", {"status": "resting", "limit": 200})
        if status == 200:
            for o in payload.get("orders", []) or []:
                if o.get("order_id"):
                    out[o["order_id"]] = o
        return out

    async def fills(self, order_id: str | None = None, ticker: str | None = None,
                    limit: int = 100) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit}
        if order_id:
            params["order_id"] = order_id
        if ticker:
            params["ticker"] = ticker
        status, payload = await self.client.get("/portfolio/fills", params)
        if status != 200:
            return []
        return payload.get("fills", []) or []

    async def position(self, ticker: str) -> int:
        for pos in await self.positions():
            if pos.get("ticker") == ticker:
                return _count(pos.get("position_fp") or pos.get("position"))
        return 0

    async def positions(self) -> list[dict[str, Any]]:
        """Every non-zero venue position, as the venue reports it."""
        status, payload = await self.client.get("/portfolio/positions", {"limit": 200})
        if status != 200:
            return []
        return payload.get("market_positions", []) or []

    async def order(self, order_id: str) -> VenueOrder | None:
        status, payload = await self.client.get(f"/portfolio/orders/{order_id}")
        if status != 200:
            return None
        o = payload.get("order") or payload
        return VenueOrder(
            order_id=o.get("order_id", order_id), client_order_id=o.get("client_order_id", ""),
            ticker=o.get("ticker", ""), side="", action="",
            price=Decimal(str(o.get("yes_price_dollars") or o.get("yes_price") or "0")),
            contracts=_count(o.get("count") or o.get("initial_count")),
            status=o.get("status", ""), filled=_count(o.get("fill_count")),
        )

    async def flatten_all(self) -> None:
        status, payload = await self.client.get("/portfolio/orders", {"status": "resting"})
        if status == 200:
            for o in payload.get("orders", []):
                await self.cancel(o.get("order_id", ""), o.get("ticker"))
        status, payload = await self.client.get("/portfolio/positions")
        if status != 200:
            return
        for pos in payload.get("market_positions", []):
            qty = int(float(pos.get("position_fp") or pos.get("position") or 0))
            if qty == 0:
                continue
            ticker = pos.get("ticker", "")
            side = "YES" if qty > 0 else "NO"
            # A YES exit sells YES, so 0.01 crosses any bid. A NO exit *buys* YES,
            # so it has to reach the other side of the book: 0.01 is a NO order that
            # can never fill and would leave the position open.
            price = Decimal("0.01") if side == "YES" else Decimal("0.99")
            await self.place(ticker, side, "exit", price, abs(qty),
                             post_only=False, tif="immediate_or_cancel", reduce_only=True)

    @staticmethod
    def _bid_ask(side: str, action: str) -> str:
        """Kalshi books are YES-centric: buying NO is selling YES."""
        if side == "YES":
            return "bid" if action == "entry" else "ask"
        return "ask" if action == "entry" else "bid"
