"""Authenticated recorder for Kalshi's `communications` channel.

The channel is a firehose: a 22-second production capture on 2026-11-18 carried
~35,000 RFQs.  This recorder therefore writes normalized rows in batched
transactions instead of one synchronous insert per message.  Every RFQ request is
kept; combo legs are zlib-compressed because a multi-leg RFQ can otherwise add
several kilobytes per row.

Visibility boundary: `rfq_created` / `rfq_deleted` are broadcast to every
authenticated listener, but `quote_created`, `quote_accepted`, `quote_executed`
and the rest of the quote lifecycle are private to the RFQ creator and the quote
creator.  "Every quote" therefore means every quote this account can see: its
own quotes and quotes on its own RFQs.  The REST reconciler asks for both with
`user_filter=self` and `rfq_user_filter=self`.

Run:

    .venv/bin/python -m openkal.maker.rfq_recorder record --seconds 60
    .venv/bin/python -m openkal.maker.rfq_recorder record --max-events 5000
    .venv/bin/python -m openkal.maker.rfq_recorder summary
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import time
import zlib
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import websockets

from ..auth import KalshiClient
from ..config import ROOT, Settings, load_settings
from ..kalshi_ws import WS_URL, KalshiWS

DEFAULT_DB = ROOT / "data" / "rfq.db"
SOCKET_ID = 41


def parse_ts(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    if isinstance(value, (int, float)):
        number = float(value)
        return number / 1000.0 if number > 1e12 else number
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def pack(value: Any) -> bytes | None:
    if not value:
        return None
    return zlib.compress(json.dumps(value, separators=(",", ":")).encode(), 6)


def unpack(value: bytes | None) -> Any:
    if not value:
        return None
    try:
        return json.loads(zlib.decompress(value))
    except (ValueError, zlib.error):
        return None


class RFQRecorder:
    def __init__(self, settings: Settings, db_path: Path | str = DEFAULT_DB, *,
                 client: KalshiClient | None = None,
                 market_prefix: str | None = None) -> None:
        self.settings = settings
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.db_path))
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self._init_db()
        self.client = client or KalshiClient(settings)
        self.market_prefix = market_prefix or None
        self.types: Counter[str] = Counter()
        self.events = 0
        self.skipped = 0
        self.dropped = 0
        self.last_ts = 0.0
        self.last_error: str | None = None
        self._events: list[tuple] = []
        self._rfqs: dict[str, dict[str, Any]] = {}
        self._batches: dict[str, dict[str, Any]] = {}
        self._quotes: dict[str, dict[str, Any]] = {}
        self._quote_events: list[tuple] = []
        self._executions: dict[str, dict[str, Any]] = {}
        self._settlements: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ schema
    def _init_db(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
              id INTEGER PRIMARY KEY AUTOINCREMENT, ws_seq INTEGER, ts REAL,
              type TEXT, rfq_id TEXT, batch_id TEXT, quote_id TEXT, ticker TEXT,
              creator_id TEXT, target_cost TEXT, contracts TEXT, status TEXT,
              payload TEXT
            );
            CREATE TABLE IF NOT EXISTS rfqs (
              id TEXT PRIMARY KEY, created_ts REAL, deleted_ts REAL, updated_ts REAL,
              market_ticker TEXT, event_ticker TEXT, creator_id TEXT,
              target_cost TEXT, contracts TEXT, status TEXT, batch_id TEXT,
              mve_collection TEXT, legs BLOB, leg_count INTEGER
            );
            CREATE TABLE IF NOT EXISTS batches (
              id TEXT PRIMARY KEY, created_ts REAL, deleted_ts REAL, creator_id TEXT,
              legs BLOB, leg_count INTEGER
            );
            CREATE TABLE IF NOT EXISTS quotes (
              id TEXT PRIMARY KEY, rfq_id TEXT, market_ticker TEXT, creator_id TEXT,
              rfq_creator_id TEXT, yes_bid TEXT, no_bid TEXT, yes_contracts TEXT,
              no_contracts TEXT, status TEXT, created_ts REAL, updated_ts REAL,
              accepted_side TEXT, accepted_ts REAL, confirmed_ts REAL,
              executed_ts REAL, cancelled_ts REAL, rest_remainder INTEGER,
              post_only INTEGER, creator_order_id TEXT, cancellation_reason TEXT
            );
            CREATE TABLE IF NOT EXISTS quote_events (
              id INTEGER PRIMARY KEY AUTOINCREMENT, ws_seq INTEGER, ts REAL,
              type TEXT, quote_id TEXT, rfq_id TEXT, ticker TEXT, status TEXT,
              payload TEXT
            );
            CREATE TABLE IF NOT EXISTS executions (
              quote_id TEXT PRIMARY KEY, rfq_id TEXT, order_id TEXT,
              market_ticker TEXT, contracts TEXT, filled TEXT, price TEXT,
              fee TEXT, executed_ts REAL, raw_json TEXT
            );
            CREATE TABLE IF NOT EXISTS settlements (
              market_ticker TEXT PRIMARY KEY, result TEXT, settlement_value TEXT,
              settled_ts REAL, raw_json TEXT
            );
            CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
            CREATE INDEX IF NOT EXISTS events_type ON events(type);
            CREATE INDEX IF NOT EXISTS events_rfq ON events(rfq_id);
            CREATE INDEX IF NOT EXISTS quote_events_quote ON quote_events(quote_id);
            """
        )
        self.db.commit()

    # ------------------------------------------------------------------ ingest
    def _merge_rfq(self, rfq_id: str, **fields: Any) -> None:
        row = self._rfqs.setdefault(rfq_id, {"id": rfq_id})
        for key, value in fields.items():
            if value is not None and value != "":
                row[key] = value
        if row.get("deleted_ts"):
            row["status"] = "deleted"
        elif not row.get("status"):
            row["status"] = "open"
        row["updated_ts"] = self.last_ts or time.time()

    def _merge_batch(self, batch_id: str, **fields: Any) -> None:
        row = self._batches.setdefault(batch_id, {"id": batch_id})
        for key, value in fields.items():
            if value is not None and value != "":
                row[key] = value

    def _merge_quote(self, quote_id: str, **fields: Any) -> None:
        row = self._quotes.setdefault(quote_id, {"id": quote_id})
        explicit = fields.get("status")
        for key, value in fields.items():
            if value is not None and value != "":
                row[key] = value
        if row.get("cancelled_ts"):
            row["status"] = "cancelled"
        elif row.get("executed_ts"):
            row["status"] = "executed"
        elif row.get("confirmed_ts"):
            row["status"] = "confirmed"
        elif row.get("accepted_ts"):
            row["status"] = "accepted"
        elif explicit in {"executed", "cancelled"}:
            row["status"] = explicit
        elif not row.get("status"):
            row["status"] = "open"

    def ingest(self, msg: dict[str, Any]) -> None:
        kind = str(msg.get("type") or "unknown")
        body = msg.get("msg") or {}
        if not isinstance(body, dict):
            body = {}
        seq = int(msg.get("seq") or 0)
        now = time.time()
        self.types[kind] += 1
        self.last_ts = now
        if kind == "subscribed":
            return  # socket control frame, not a market event

        ticker = str(body.get("market_ticker") or body.get("ticker") or "")
        if self.market_prefix and ticker and not ticker.startswith(self.market_prefix):
            self.skipped += 1
            self.types["skipped"] += 1
            return

        rfq_id: str | None = None
        batch_id: str | None = None
        quote_id: str | None = None
        status = ""

        if kind in {"rfq_created", "rfq_deleted"}:
            rfq_id = str(body.get("id") or body.get("rfq_id") or "")
            if not rfq_id:
                return
            if kind == "rfq_created":
                status = "open"
                legs = body.get("mve_selected_legs") or []
                self._merge_rfq(
                    rfq_id,
                    created_ts=parse_ts(body.get("created_ts")) or now,
                    market_ticker=body.get("market_ticker"),
                    event_ticker=body.get("event_ticker"),
                    creator_id=body.get("creator_id"),
                    target_cost=body.get("target_cost_dollars"),
                    contracts=body.get("contracts_fp"),
                    status="open",
                    batch_id=body.get("rfq_batch_id"),
                    mve_collection=body.get("mve_collection_ticker"),
                    legs=pack(legs), leg_count=len(legs),
                )
            else:
                status = "deleted"
                self._merge_rfq(
                    rfq_id,
                    deleted_ts=parse_ts(body.get("deleted_ts")) or now,
                    market_ticker=body.get("market_ticker"),
                    creator_id=body.get("creator_id"),
                    target_cost=body.get("target_cost_dollars"),
                    status="deleted",
                )
        elif kind in {"rfq_batch_created", "rfq_batch_deleted"}:
            batch_id = str(body.get("id") or body.get("rfq_batch_id") or "")
            if not batch_id:
                return
            if kind == "rfq_batch_created":
                status = "open"
                legs = body.get("legs") or []
                self._merge_batch(
                    batch_id,
                    created_ts=parse_ts(body.get("created_ts")) or now,
                    creator_id=body.get("creator_id"),
                    legs=pack(legs), leg_count=len(legs),
                )
            else:
                status = "deleted"
                self._merge_batch(batch_id,
                                  deleted_ts=parse_ts(body.get("deleted_ts")) or now)
                for child in body.get("rfq_ids") or []:
                    self._merge_rfq(str(child), deleted_ts=now, status="deleted")
        elif kind.startswith("quote_"):
            quote_id = str(body.get("quote_id") or body.get("id") or "")
            rfq_id = str(body.get("rfq_id") or "") or None
            if not quote_id:
                return
            ticker = str(body.get("market_ticker") or ticker)
            if kind == "quote_created":
                status = "open"
                self._merge_quote(
                    quote_id, rfq_id=rfq_id, market_ticker=body.get("market_ticker"),
                    creator_id=body.get("creator_id"),
                    rfq_creator_id=body.get("rfq_creator_id"),
                    yes_bid=body.get("yes_bid_dollars") or body.get("yes_bid"),
                    no_bid=body.get("no_bid_dollars") or body.get("no_bid"),
                    yes_contracts=body.get("yes_contracts_fp"),
                    no_contracts=body.get("no_contracts_fp"),
                    created_ts=parse_ts(body.get("created_ts")) or now,
                    rest_remainder=body.get("rest_remainder"),
                    post_only=body.get("post_only"),
                    status="open",
                )
            elif kind == "quote_accepted":
                status = "accepted"
                self._merge_quote(
                    quote_id, rfq_id=rfq_id, market_ticker=body.get("market_ticker"),
                    accepted_side=body.get("accepted_side"),
                    accepted_ts=parse_ts(body.get("accepted_ts")) or now,
                    status="accepted",
                )
            elif kind == "quote_confirmed":
                status = "confirmed"
                self._merge_quote(
                    quote_id, rfq_id=rfq_id, market_ticker=body.get("market_ticker"),
                    confirmed_ts=parse_ts(body.get("confirmed_ts")) or now,
                    status="confirmed",
                )
            elif kind == "quote_executed":
                status = "executed"
                order_id = body.get("creator_order_id") or body.get("order_id")
                self._merge_quote(
                    quote_id, rfq_id=rfq_id, market_ticker=body.get("market_ticker"),
                    executed_ts=parse_ts(body.get("executed_ts")) or now,
                    creator_order_id=order_id,
                    yes_bid=body.get("yes_bid_dollars") or body.get("yes_bid"),
                    no_bid=body.get("no_bid_dollars") or body.get("no_bid"),
                    status="executed",
                )
                if order_id:
                    self._executions[quote_id] = {
                        "quote_id": quote_id, "rfq_id": rfq_id,
                        "order_id": order_id,
                        "market_ticker": body.get("market_ticker") or ticker,
                        "contracts": body.get("contracts_fp") or body.get("count_fp"),
                        "filled": body.get("fill_count_fp") or body.get("filled"),
                        "price": body.get("yes_price_dollars") or body.get("price_dollars"),
                        "fee": body.get("fee_cost"),
                        "executed_ts": parse_ts(body.get("executed_ts")) or now,
                        "raw_json": json.dumps(body)[:2000],
                    }
            elif kind == "quote_cancelled":
                status = "cancelled"
                self._merge_quote(
                    quote_id, rfq_id=rfq_id, market_ticker=body.get("market_ticker"),
                    cancelled_ts=parse_ts(body.get("cancelled_ts")) or now,
                    cancellation_reason=body.get("cancellation_reason"),
                    status="cancelled",
                )
            else:
                status = str(body.get("status") or kind.removeprefix("quote_"))
                self._merge_quote(quote_id, rfq_id=rfq_id,
                                  market_ticker=body.get("market_ticker"),
                                  status=status)
            self._quote_events.append(
                (seq, now, kind, quote_id, rfq_id, ticker, status,
                 json.dumps(body, separators=(",", ":"))[:2000]))
        else:
            # An error frame or a future message type: keep the event, not a schema guess.
            payload = json.dumps(body, separators=(",", ":"))[:1000] if body else None
            self._events.append(
                (seq, now, kind, None, None, None, ticker, body.get("creator_id"),
                 None, None, None, payload))
            self.events += 1
            return

        self._events.append(
            (seq, now, kind, rfq_id, batch_id, quote_id, ticker,
             body.get("creator_id"), body.get("target_cost_dollars"),
             body.get("contracts_fp"), status, None))
        self.events += 1

    def record_settlement(self, ticker: str, result: str,
                          settlement_value: Any = None, ts: float | None = None) -> None:
        self._settlements[ticker] = {
            "market_ticker": ticker, "result": result,
            "settlement_value": None if settlement_value is None else str(settlement_value),
            "settled_ts": time.time() if ts is None else float(ts),
            "raw_json": None,
        }

    # ------------------------------------------------------------------ flush
    def flush(self) -> None:
        if not (self._events or self._rfqs or self._batches or self._quotes
                or self._quote_events or self._executions or self._settlements):
            return
        with self.db:
            if self._events:
                self.db.executemany(
                    """INSERT INTO events
                       (ws_seq,ts,type,rfq_id,batch_id,quote_id,ticker,creator_id,
                        target_cost,contracts,status,payload)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""", self._events)
            for row in self._rfqs.values():
                self.db.execute(
                    """INSERT INTO rfqs
                       (id,created_ts,deleted_ts,updated_ts,market_ticker,event_ticker,
                        creator_id,target_cost,contracts,status,batch_id,mve_collection,
                        legs,leg_count)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(id) DO UPDATE SET
                         created_ts=COALESCE(excluded.created_ts,rfqs.created_ts),
                         deleted_ts=COALESCE(excluded.deleted_ts,rfqs.deleted_ts),
                         updated_ts=excluded.updated_ts,
                         market_ticker=COALESCE(excluded.market_ticker,rfqs.market_ticker),
                         event_ticker=COALESCE(excluded.event_ticker,rfqs.event_ticker),
                         creator_id=COALESCE(excluded.creator_id,rfqs.creator_id),
                         target_cost=COALESCE(excluded.target_cost,rfqs.target_cost),
                         contracts=COALESCE(excluded.contracts,rfqs.contracts),
                         status=CASE WHEN rfqs.status='deleted' THEN 'deleted'
                                     ELSE excluded.status END,
                         batch_id=COALESCE(excluded.batch_id,rfqs.batch_id),
                         mve_collection=COALESCE(excluded.mve_collection,rfqs.mve_collection),
                         legs=COALESCE(excluded.legs,rfqs.legs),
                         leg_count=COALESCE(excluded.leg_count,rfqs.leg_count)""",
                    (row.get("id"), row.get("created_ts"), row.get("deleted_ts"),
                     row.get("updated_ts"), row.get("market_ticker"),
                     row.get("event_ticker"), row.get("creator_id"),
                     row.get("target_cost"), row.get("contracts"), row.get("status"),
                     row.get("batch_id"), row.get("mve_collection"),
                     row.get("legs"), row.get("leg_count")))
            for row in self._batches.values():
                self.db.execute(
                    """INSERT INTO batches (id,created_ts,deleted_ts,creator_id,legs,leg_count)
                       VALUES (?,?,?,?,?,?)
                       ON CONFLICT(id) DO UPDATE SET
                         created_ts=COALESCE(excluded.created_ts,batches.created_ts),
                         deleted_ts=COALESCE(excluded.deleted_ts,batches.deleted_ts),
                         creator_id=COALESCE(excluded.creator_id,batches.creator_id),
                         legs=COALESCE(excluded.legs,batches.legs),
                         leg_count=COALESCE(excluded.leg_count,batches.leg_count)""",
                    (row.get("id"), row.get("created_ts"), row.get("deleted_ts"),
                     row.get("creator_id"), row.get("legs"), row.get("leg_count")))
            for row in self._quotes.values():
                self.db.execute(
                    """INSERT INTO quotes
                       (id,rfq_id,market_ticker,creator_id,rfq_creator_id,yes_bid,no_bid,
                        yes_contracts,no_contracts,status,created_ts,updated_ts,
                        accepted_side,accepted_ts,confirmed_ts,executed_ts,cancelled_ts,
                        rest_remainder,post_only,creator_order_id,cancellation_reason)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(id) DO UPDATE SET
                         rfq_id=COALESCE(excluded.rfq_id,quotes.rfq_id),
                         market_ticker=COALESCE(excluded.market_ticker,quotes.market_ticker),
                         creator_id=COALESCE(excluded.creator_id,quotes.creator_id),
                         rfq_creator_id=COALESCE(excluded.rfq_creator_id,quotes.rfq_creator_id),
                         yes_bid=COALESCE(excluded.yes_bid,quotes.yes_bid),
                         no_bid=COALESCE(excluded.no_bid,quotes.no_bid),
                         yes_contracts=COALESCE(excluded.yes_contracts,quotes.yes_contracts),
                         no_contracts=COALESCE(excluded.no_contracts,quotes.no_contracts),
                         status=CASE WHEN quotes.status IN ('executed','cancelled')
                                     THEN quotes.status ELSE excluded.status END,
                         created_ts=COALESCE(excluded.created_ts,quotes.created_ts),
                         updated_ts=COALESCE(excluded.updated_ts,quotes.updated_ts),
                         accepted_side=COALESCE(excluded.accepted_side,quotes.accepted_side),
                         accepted_ts=COALESCE(excluded.accepted_ts,quotes.accepted_ts),
                         confirmed_ts=COALESCE(excluded.confirmed_ts,quotes.confirmed_ts),
                         executed_ts=COALESCE(excluded.executed_ts,quotes.executed_ts),
                         cancelled_ts=COALESCE(excluded.cancelled_ts,quotes.cancelled_ts),
                         rest_remainder=COALESCE(excluded.rest_remainder,quotes.rest_remainder),
                         post_only=COALESCE(excluded.post_only,quotes.post_only),
                         creator_order_id=COALESCE(excluded.creator_order_id,quotes.creator_order_id),
                         cancellation_reason=COALESCE(excluded.cancellation_reason,quotes.cancellation_reason)""",
                    (row.get("id"), row.get("rfq_id"), row.get("market_ticker"),
                     row.get("creator_id"), row.get("rfq_creator_id"),
                     row.get("yes_bid"), row.get("no_bid"), row.get("yes_contracts"),
                     row.get("no_contracts"), row.get("status") or "open",
                     row.get("created_ts"), row.get("updated_ts") or time.time(),
                     row.get("accepted_side"), row.get("accepted_ts"),
                     row.get("confirmed_ts"), row.get("executed_ts"),
                     row.get("cancelled_ts"),
                     None if row.get("rest_remainder") is None else int(bool(row["rest_remainder"])),
                     None if row.get("post_only") is None else int(bool(row["post_only"])),
                     row.get("creator_order_id"), row.get("cancellation_reason")))
            if self._quote_events:
                self.db.executemany(
                    """INSERT INTO quote_events
                       (ws_seq,ts,type,quote_id,rfq_id,ticker,status,payload)
                       VALUES (?,?,?,?,?,?,?,?)""", self._quote_events)
            for row in self._executions.values():
                self.db.execute(
                    """INSERT INTO executions
                       (quote_id,rfq_id,order_id,market_ticker,contracts,filled,price,
                        fee,executed_ts,raw_json)
                       VALUES (?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(quote_id) DO UPDATE SET
                         order_id=COALESCE(excluded.order_id,executions.order_id),
                         market_ticker=COALESCE(excluded.market_ticker,executions.market_ticker),
                         contracts=COALESCE(excluded.contracts,executions.contracts),
                         filled=COALESCE(excluded.filled,executions.filled),
                         price=COALESCE(excluded.price,executions.price),
                         fee=COALESCE(excluded.fee,executions.fee),
                         executed_ts=COALESCE(excluded.executed_ts,executions.executed_ts),
                         raw_json=COALESCE(excluded.raw_json,executions.raw_json)""",
                    (row.get("quote_id"), row.get("rfq_id"), row.get("order_id"),
                     row.get("market_ticker"), row.get("contracts"), row.get("filled"),
                     row.get("price"), row.get("fee"), row.get("executed_ts"),
                     row.get("raw_json")))
            for row in self._settlements.values():
                self.db.execute(
                    """INSERT INTO settlements
                       (market_ticker,result,settlement_value,settled_ts,raw_json)
                       VALUES (?,?,?,?,?)
                       ON CONFLICT(market_ticker) DO UPDATE SET
                         result=excluded.result,
                         settlement_value=COALESCE(excluded.settlement_value,settlements.settlement_value),
                         settled_ts=excluded.settled_ts,
                         raw_json=COALESCE(excluded.raw_json,settlements.raw_json)""",
                    (row.get("market_ticker"), row.get("result"),
                     row.get("settlement_value"), row.get("settled_ts"),
                     row.get("raw_json")))
        self._events = []
        self._rfqs = {}
        self._batches = {}
        self._quotes = {}
        self._quote_events = []
        self._executions = {}
        self._settlements = {}

    # ------------------------------------------------------------------ reconcile
    async def reconcile(self) -> int:
        """Pull private quote snapshots this account is allowed to see."""
        seen = 0
        for params in ({"user_filter": "self"}, {"rfq_user_filter": "self"}):
            code, payload = await self.client.get(
                "/communications/quotes", {**params, "limit": 500})
            if code != 200:
                continue
            for quote in payload.get("quotes") or []:
                quote_id = str(quote.get("id") or "")
                if not quote_id:
                    continue
                status = str(quote.get("status") or "unknown")
                self._merge_quote(
                    quote_id, rfq_id=quote.get("rfq_id"),
                    market_ticker=quote.get("market_ticker"),
                    creator_id=quote.get("creator_id"),
                    rfq_creator_id=quote.get("rfq_creator_id"),
                    yes_bid=quote.get("yes_bid_dollars"),
                    no_bid=quote.get("no_bid_dollars"),
                    yes_contracts=quote.get("yes_contracts_fp"),
                    no_contracts=quote.get("no_contracts_fp"),
                    created_ts=parse_ts(quote.get("created_ts")) or None,
                    updated_ts=parse_ts(quote.get("updated_ts")) or time.time(),
                    accepted_side=quote.get("accepted_side"),
                    accepted_ts=parse_ts(quote.get("accepted_ts")) or None,
                    confirmed_ts=parse_ts(quote.get("confirmed_ts")) or None,
                    executed_ts=parse_ts(quote.get("executed_ts")) or None,
                    cancelled_ts=parse_ts(quote.get("cancelled_ts")) or None,
                    rest_remainder=quote.get("rest_remainder"),
                    post_only=quote.get("post_only"),
                    creator_order_id=quote.get("creator_order_id"),
                    cancellation_reason=quote.get("cancellation_reason"),
                    status=status,
                )
                self._quote_events.append(
                    (0, time.time(), "quote_snapshot", quote_id,
                     quote.get("rfq_id"), quote.get("market_ticker"), status,
                     json.dumps(quote, separators=(",", ":"))[:2000]))
                self.events += 1
                seen += 1
        self.flush()
        return seen

    async def settle(self) -> int:
        """Resolve fills and market results for our executed quotes, best effort."""
        rows = self.db.execute(
            """SELECT e.quote_id,e.order_id,e.market_ticker,e.filled,e.price,e.fee
               FROM executions e
               LEFT JOIN settlements s ON s.market_ticker=e.market_ticker
               WHERE s.market_ticker IS NULL LIMIT 50""").fetchall()
        resolved = 0
        for quote_id, order_id, ticker, filled, price, fee in rows:
            if order_id and (not filled or not price):
                code, payload = await self.client.get(f"/portfolio/orders/{order_id}")
                if code == 200:
                    order = payload.get("order") or payload
                    new_filled = order.get("fill_count_fp") or order.get("fill_count")
                    new_price = order.get("yes_price_dollars") or order.get("yes_price")
                    if new_filled:
                        filled = str(new_filled)
                    if new_price:
                        price = str(new_price)
                code, payload = await self.client.get(
                    "/portfolio/fills", {"order_id": order_id, "limit": 100})
                if code == 200:
                    fills = payload.get("fills") or []
                    total = sum(float(f.get("count_fp") or f.get("count") or 0)
                                for f in fills)
                    if total > 0:
                        filled = str(total)
                        notional = sum(
                            float(f.get("count_fp") or f.get("count") or 0)
                            * float(f.get("yes_price_dollars") or f.get("yes_price") or 0)
                            for f in fills)
                        price = str(notional / total)
                        fee = str(sum(float(f.get("fee_cost") or 0) for f in fills))
                with self.db:
                    self.db.execute(
                        "UPDATE executions SET filled=?,price=?,fee=? WHERE quote_id=?",
                        (filled, price, fee, quote_id))
            if not ticker:
                continue
            code, payload = await self.client.get(f"/markets/{ticker}")
            if code != 200:
                continue
            market = payload.get("market") or payload
            result = market.get("result")
            if result:
                self.record_settlement(ticker, str(result),
                                       market.get("settlement_value_dollars")
                                       or market.get("expiration_value"))
                resolved += 1
        self.flush()
        return resolved

    # ------------------------------------------------------------------ run
    async def run(self, *, seconds: float | None = None,
                  max_events: int | None = None) -> None:
        helper = KalshiWS(self.settings, self.client)
        started = time.time()
        backoff = 2.0
        try:
            await self.reconcile()
        except Exception as exc:  # noqa: BLE001 - a REST snapshot is optional
            self.last_error = f"reconcile {type(exc).__name__}: {exc}"
        while True:
            try:
                async with websockets.connect(
                        WS_URL, **helper.connect_kwargs()) as socket:
                    await socket.send(json.dumps({
                        "id": SOCKET_ID, "cmd": "subscribe",
                        "params": {"channels": ["communications"]},
                    }))
                    backoff = 2.0
                    last_flush = last_reconcile = last_settle = time.time()
                    while True:
                        try:
                            raw = await asyncio.wait_for(socket.recv(), timeout=1.0)
                        except asyncio.TimeoutError:
                            raw = None
                        if raw is not None:
                            try:
                                msg = json.loads(raw)
                            except ValueError:
                                msg = {}
                            if msg:
                                self.ingest(msg)
                        now = time.time()
                        if now - last_flush >= 0.5:
                            self.flush()
                            last_flush = now
                        if now - last_reconcile >= 60.0:
                            try:
                                await self.reconcile()
                            except Exception as exc:  # noqa: BLE001
                                self.last_error = f"reconcile {type(exc).__name__}: {exc}"
                            last_reconcile = now
                        if now - last_settle >= 60.0:
                            try:
                                await self.settle()
                            except Exception as exc:  # noqa: BLE001
                                self.last_error = f"settle {type(exc).__name__}: {exc}"
                            last_settle = now
                        if seconds and now - started >= seconds:
                            return
                        if max_events and self.events >= max_events:
                            return
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the socket reconnects
                self.last_error = f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(backoff)
                backoff = min(30.0, backoff * 1.8)
            finally:
                self.flush()

    def summary(self) -> dict[str, Any]:
        self.flush()
        def one(sql: str) -> int:
            row = self.db.execute(sql).fetchone()
            return int(row[0] or 0) if row else 0
        last = self.db.execute("SELECT MAX(ts) FROM events").fetchone()
        types = dict(self.db.execute(
            "SELECT type,COUNT(*) FROM events GROUP BY type ORDER BY type").fetchall())
        return {
            "db": str(self.db_path),
            "events": one("SELECT COUNT(*) FROM events"),
            "rfqs": one("SELECT COUNT(*) FROM rfqs"),
            "open_rfqs": one("SELECT COUNT(*) FROM rfqs WHERE status='open'"),
            "combo_rfqs": one("SELECT COUNT(*) FROM rfqs WHERE leg_count > 0"),
            "batches": one("SELECT COUNT(*) FROM batches"),
            "quotes": one("SELECT COUNT(*) FROM quotes"),
            "accepted_quotes": one("SELECT COUNT(*) FROM quotes WHERE status IN ('accepted','confirmed','executed')"),
            "executed_quotes": one("SELECT COUNT(*) FROM quotes WHERE status='executed'"),
            "executions": one("SELECT COUNT(*) FROM executions"),
            "settlements": one("SELECT COUNT(*) FROM settlements"),
            "last_event_ts": float(last[0]) if last and last[0] is not None else 0.0,
            "types": types,
            "skipped": self.skipped,
            "last_error": self.last_error,
        }

    def close(self) -> None:
        self.flush()
        self.db.commit()
        self.db.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rfq_recorder", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record", help="record the communications channel")
    rec.add_argument("--db", default=str(DEFAULT_DB))
    rec.add_argument("--seconds", type=float, default=0.0)
    rec.add_argument("--max-events", type=int, default=0)
    rec.add_argument("--market-prefix", default=None,
                     help="optional ticker prefix (default: every RFQ)")
    rec.add_argument("--json", action="store_true")
    summ = sub.add_parser("summary", help="show the recorder's totals")
    summ.add_argument("--db", default=str(DEFAULT_DB))
    args = parser.parse_args(argv)

    if args.command == "summary":
        recorder = RFQRecorder(load_settings(), args.db)
        print(json.dumps(recorder.summary(), indent=2))
        recorder.close()
        return 0

    async def run() -> int:
        settings = load_settings()
        recorder = RFQRecorder(settings, args.db, market_prefix=args.market_prefix)
        try:
            await recorder.run(
                seconds=args.seconds or None,
                max_events=args.max_events or None)
        finally:
            result = recorder.summary()
            recorder.close()
            await recorder.client.close()
        print(json.dumps(result, indent=2))
        return 0

    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
