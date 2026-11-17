"""The authenticated Kalshi WebSocket plane.

This is the market-data source the bot was missing. It carries four things REST could
never give us at the right speed:

  * `cfbenchmarks_value` — the BRTI/RTI print every second, with `avg_60s_data` and,
    in the final minute before :00/:15/:30/:45, `last_60s_windowed_average_15min`,
    which is the live settlement accumulator (k of 60 seconds already locked).
  * `cfbenchmarks_value_5hz` — the same indices at 200ms.
  * `orderbook_delta` — the real ladder, so we stop polling `/markets` snapshots and
    stop being 3 seconds behind the book.
  * `trade`, `fill`, `user_orders`, `market_positions`, `market_lifecycle_v2`.

Handshake headers are generated fresh on every connect, because a stale timestamp is a
401 and the socket reconnects on its own schedule, not ours.

Received messages are handed to callbacks synchronously and persisted through a queue,
so a slow SQLite write can never stall the receive loop.
"""
from __future__ import annotations

import asyncio
import base64
import inspect
import json
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable

import websockets

from .auth import KalshiClient
from .config import Settings

WS_URL = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
WS_PATH = "/trade-api/ws/v2"

# The indices each crypto 15-minute series settles against.
INDEX_BY_SERIES = {
    "KXBTC15M": "BRTI",
    "KXETH15M": "ETHUSD_RTI",
    "KXSOL15M": "SOLUSD_RTI",
    "KXXRP15M": "XRPUSD_RTI",
    "KXDOGE15M": "DOGEUSD_RTI",
}


class SeqTracker:
    """Sequence continuity per subscription.

    Kalshi restarts `seq` at 1 on every re-subscribe for the same `sid`, so a snapshot
    is a *new base*, not a higher number. Treating it as `max(old, new)` — which is what
    this code did before — makes every delta after a re-subscribe look like a gap, and
    the bot re-fetches every ladder from REST forever. That bug produced 502,414 bogus
    gaps in one run.
    """

    def __init__(self) -> None:
        self.bases: dict[int, int] = {}
        self.gaps = 0

    def on_snapshot(self, sid: int, seq: int) -> None:
        self.bases[sid] = seq

    def on_delta(self, sid: int, seq: int) -> bool:
        """True when this delta continues the stream; False when a message was missed."""
        last = self.bases.get(sid)
        ok = last is None or seq in (last + 1, last)
        if not ok:
            self.gaps += 1
        self.bases[sid] = max(last or 0, seq)
        return ok


def _dec(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception:  # noqa: BLE001 - venue numbers are not always parseable
        return Decimal("0")


@dataclass
class Book:
    """Local ladder built from `orderbook_snapshot` + `orderbook_delta`."""

    ticker: str
    yes: dict[Decimal, Decimal] = field(default_factory=dict)
    no: dict[Decimal, Decimal] = field(default_factory=dict)
    seq: int = 0
    synced: bool = False
    updated_ts: float = 0.0

    def apply_snapshot(self, msg: dict[str, Any], seq: int) -> None:
        self.yes = {_dec(p): _dec(c) for p, c in msg.get("yes_dollars_fp") or [] if _dec(c) > 0}
        self.no = {_dec(p): _dec(c) for p, c in msg.get("no_dollars_fp") or [] if _dec(c) > 0}
        # A snapshot redefines the sequence base: a new subscription restarts at seq 1,
        # so keeping the old seq would make every following delta look like a gap.
        self.seq = seq
        self.synced = bool(self.yes or self.no)
        self.updated_ts = time.time()

    def apply_delta(self, msg: dict[str, Any], seq: int) -> bool:
        """Apply one delta. Sequence gaps are tracked per subscription, not per book."""
        side = self.yes if msg.get("side") == "yes" else self.no
        price = _dec(msg.get("price_dollars"))
        size = side.get(price, Decimal("0")) + _dec(msg.get("delta_fp"))
        if size <= 0:
            side.pop(price, None)
        else:
            side[price] = size
        self.seq = seq
        self.updated_ts = time.time()
        return True

    @property
    def yes_bid(self) -> Decimal | None:
        return max(self.yes) if self.yes else None

    @property
    def yes_ask(self) -> Decimal | None:
        """Asks are the other side of the book: a NO bid at p is a YES ask at 1-p."""
        return (Decimal(1) - max(self.no)) if self.no else None

    @property
    def no_ask(self) -> Decimal | None:
        return (Decimal(1) - max(self.yes)) if self.yes else None


@dataclass
class IndexPrint:
    index_id: str
    value: Decimal
    avg_60s: Decimal | None
    final_minute: Decimal | None       # last_60s_windowed_average_15min.value
    final_minute_ticks: int            # window_size of that average = k seconds locked
    received_ts: float
    channel: str = ""


class KalshiWS:
    """One socket, one reconnect loop, three callbacks."""

    def __init__(self, settings: Settings, client: KalshiClient,
                 on_book: Callable[[Book], None] | None = None,
                 on_index: Callable[[IndexPrint], None] | None = None,
                 on_event: Callable[[dict[str, Any]], None] | None = None,
                 store: Any = None) -> None:
        self.settings = settings
        self.client = client
        self.on_book = on_book
        self.on_index = on_index
        self.on_event = on_event
        self.store = store
        self.books: dict[str, Book] = {}
        self.indices: dict[str, IndexPrint] = {}
        self.lag_ms: float = 0.0
        self.degraded = False
        self.messages = 0
        self.reconnects = 0
        self.last_error: str | None = None
        self._queue: asyncio.Queue[tuple] = asyncio.Queue(maxsize=20000)
        self._writer: asyncio.Task | None = None
        # Strong reference to the receive loop. `create_task` alone leaves the task
        # reachable only through a reference cycle, and a collected socket task looks
        # exactly like a healthy one from the snapshot: 0 messages, 0 errors.
        self.task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._tickers: list[str] = []
        self._resync_pending: set[str] = set()
        self.sync_problems = 0
        # Readiness distinguishes a gap that was repaired from one that was not. A gap
        # triggers a REST resync, so `gaps - resync_ok` is the honest unrecovered count.
        self.resync_ok = 0
        self.resync_fail = 0
        # `seq` is per subscription (sid) and shared by every ticker on it, so a
        # per-ticker counter would report a gap on every message.
        self.seq = SeqTracker()

    # ------------------------------------------------------------------ auth
    def headers(self) -> dict[str, str]:
        ts = str(int(time.time() * 1000))
        sig = base64.b64encode(
            self.client._sign((ts + "GET" + WS_PATH).encode())   # noqa: SLF001
        ).decode()
        return {
            "KALSHI-ACCESS-KEY": self.settings.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": sig,
            "KALSHI-ACCESS-TIMESTAMP": ts,
        }

    # ------------------------------------------------------------------ lifecycle
    def watch(self, tickers: list[str]) -> None:
        """Called by the engine when the 15-minute windows roll."""
        new = [t for t in tickers if t and t not in self._tickers]
        self._tickers.extend(new)
        if new and getattr(self, "_ws", None) is not None:
            asyncio.create_task(self._subscribe_markets(new))

    async def start(self) -> None:
        if self._writer is None or self._writer.done():
            self._writer = asyncio.create_task(self._persist_loop(), name="ws-persist")
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._run(), name="kalshi-ws")

    async def stop(self) -> None:
        self._stop.set()
        for task in (self.task, self._writer):
            if task is not None:
                task.cancel()
        await asyncio.gather(*[t for t in (self.task, self._writer) if t],
                             return_exceptions=True)

    def connect_kwargs(self) -> dict[str, Any]:
        """Handshake kwargs for whichever `websockets` major is installed.

        `websockets` 13 renamed the header argument to `additional_headers`; 12 and
        older take `extra_headers`. Passing the wrong name raises a `TypeError` that
        the reconnect loop treats as a transient socket failure, so the plane retries
        forever at 0 messages. Pick the name the installed client actually declares.
        """
        try:
            params = inspect.signature(websockets.connect).parameters
        except (TypeError, ValueError):  # builtins / C callables
            params = {}
        key = "additional_headers" if "additional_headers" in params else "extra_headers"
        return {key: self.headers(), "open_timeout": 15, "ping_interval": 20,
                "max_size": 8_000_000}

    async def _run(self) -> None:
        backoff = 2.0
        while not self._stop.is_set():
            try:
                async with websockets.connect(WS_URL, **self.connect_kwargs()) as ws:
                    self._ws = ws
                    self.reconnects += 1
                    backoff = 2.0
                    await self._subscribe_all(ws)
                    await self._receive(ws)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - a socket failure is routine
                self.last_error = f"{type(exc).__name__}: {exc}"
                # Say it once. A silent retry loop is how a wrong handshake looked
                # like a quiet venue for eighteen minutes.
                if self.reconnects == 0 and self.messages == 0:
                    print(f"kalshi ws connect failed: {self.last_error}", flush=True)
                await asyncio.sleep(backoff)
                backoff = min(30.0, backoff * 1.8)
            finally:
                self._ws = None

    async def _subscribe_all(self, ws) -> None:
        cmds = [
            {"id": 1, "cmd": "subscribe", "params": {
                "channels": ["cfbenchmarks_value"],
                "index_ids": sorted(set(INDEX_BY_SERIES.values())),
            }},
            {"id": 2, "cmd": "subscribe", "params": {
                "channels": ["cfbenchmarks_value_5hz"],
                "index_ids": sorted(set(INDEX_BY_SERIES.values())),
            }},
            {"id": 3, "cmd": "indexlist", "params": {"channel": "cfbenchmarks_value"}},
            {"id": 4, "cmd": "subscribe", "params": {
                "channels": ["orderbook_delta", "trade", "market_lifecycle_v2"],
                "market_tickers": self._tickers,
            }} if self._tickers else None,
            {"id": 5, "cmd": "subscribe", "params": {
                "channels": ["fill", "user_orders", "market_positions"],
            }},
        ]
        for cmd in cmds:
            if cmd:
                await ws.send(json.dumps(cmd))

    async def _subscribe_markets(self, tickers: list[str]) -> None:
        ws = getattr(self, "_ws", None)
        if ws is None:
            return
        try:
            await ws.send(json.dumps({"id": 9, "cmd": "subscribe", "params": {
                "channels": ["orderbook_delta", "trade"], "market_tickers": tickers}}))
        except Exception:  # noqa: BLE001 - the reconnect loop resubscribes everything
            pass

    async def _receive(self, ws) -> None:
        async for raw in ws:
            self.messages += 1
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            self._track_lag(msg)
            self._handle(msg)

    def _track_lag(self, msg: dict[str, Any]) -> None:
        sent = msg.get("sending_ts_ms")
        body = msg.get("msg") or {}
        ts = body.get("ts_ms") or body.get("received_at")
        if not sent or not ts:
            return
        lag = max(0.0, float(sent) - float(ts))
        self.lag_ms = lag if self.lag_ms == 0 else self.lag_ms * 0.9 + lag * 0.1
        # A feed this far behind cannot be trusted to front-run anything.
        self.degraded = self.lag_ms > 2000.0

    def _handle(self, msg: dict[str, Any]) -> None:
        kind = msg.get("type")
        seq = int(msg.get("seq") or 0)
        body = msg.get("msg") or {}
        sid = int(msg.get("sid") or 0)
        if kind == "orderbook_snapshot":
            book = self._book(body.get("market_ticker", ""))
            book.apply_snapshot(body, seq)
            # A snapshot redefines the subscription's base. Taking max() here made every
            # re-subscribe (which restarts seq at 1) look like a gap forever: 502k bogus
            # resyncs over six hours.
            self.seq.on_snapshot(sid, seq)
            self._publish_book(book)
        elif kind == "orderbook_delta":
            ticker = body.get("market_ticker", "")
            book = self._book(ticker)
            # A gap on the subscription means this book (and every book on it) missed
            # messages, so it is refetched from REST rather than trusted.
            if not self.seq.on_delta(sid, seq):
                self.sync_problems += 1
                asyncio.create_task(self._resync(ticker))
            book.apply_delta(body, seq)
            self._publish_book(book)
        elif kind in {"cfbenchmarks_value", "cfbenchmarks_value_5hz"}:
            self._handle_index(kind, body)
        elif kind in {"trade", "fill", "user_orders", "market_positions",
                      "market_lifecycle_v2", "cfbenchmarks_indexlist", "error"}:
            if kind == "cfbenchmarks_indexlist":
                self._put(("indexlist", time.time(), json.dumps(body)[:4000]))
            elif self.on_event:
                self.on_event(msg)
            if kind in {"trade", "fill"}:
                self._put(("trade", time.time(), body.get("market_ticker", ""),
                           body.get("yes_price_dollars"), body.get("count_fp"),
                           body.get("taker_side"), json.dumps(body)[:600]))

    def _book(self, ticker: str) -> Book:
        return self.books.setdefault(ticker, Book(ticker=ticker))

    def _publish_book(self, book: Book) -> None:
        if self.on_book:
            self.on_book(book)
        self._put(("book", book.updated_ts, book.ticker,
                   str(book.yes_bid), str(book.yes_ask), book.seq))

    def _handle_index(self, kind: str, body: dict[str, Any]) -> None:
        index_id = body.get("index_id") or body.get("index") or ""
        raw = body.get("data")
        value = None
        if isinstance(raw, str):
            try:
                frame = json.loads(raw)
                value = frame.get("value") or frame.get("index_value")
            except ValueError:
                value = None
        if value is None:
            value = body.get("value")
        avg = (body.get("avg_60s_data") or {}).get("value")
        final = body.get("last_60s_windowed_average_15min") or {}
        print_ = IndexPrint(
            index_id=index_id,
            value=_dec(value),
            avg_60s=_dec(avg) if avg is not None else None,
            final_minute=_dec(final.get("value")) if final.get("value") is not None else None,
            final_minute_ticks=int(final.get("window_size") or 0),
            received_ts=float(body.get("received_at") or time.time() * 1000) / 1000.0,
            channel=kind,
        )
        # The 5Hz channel carries no averages; the once-per-second channel does. Keep
        # the freshest value but never let a lean tick erase the settlement accumulator.
        prev = self.indices.get(index_id)
        if (print_.final_minute is None and prev is not None
                and prev.final_minute is not None
                and time.time() - prev.received_ts < 5.0):
            print_ = IndexPrint(print_.index_id, print_.value, print_.avg_60s or prev.avg_60s,
                                prev.final_minute, prev.final_minute_ticks,
                                print_.received_ts)
        self.indices[index_id] = print_
        if self.on_index:
            self.on_index(print_)
        self._put(("index", print_.received_ts, index_id, str(print_.value),
                   str(print_.avg_60s), str(print_.final_minute), print_.final_minute_ticks, kind))

    async def _resync(self, ticker: str) -> None:
        """A sequence gap means the ladder is wrong; replace it with a REST snapshot."""
        if ticker in self._resync_pending:
            return
        self._resync_pending.add(ticker)
        try:
            status, payload = await self.client.get(f"/markets/{ticker}/orderbook", {"depth": 20})
            if status == 200:
                ob = payload.get("orderbook_fp") or {}
                book = self._book(ticker)
                # Depth snapshot has no sequence of its own: restart the base so the
                # next delta is accepted rather than immediately flagged.
                book.apply_snapshot({"yes_dollars_fp": ob.get("yes_dollars"),
                                     "no_dollars_fp": ob.get("no_dollars")}, 0)
                self._publish_book(book)
                self.resync_ok += 1
            else:
                self.resync_fail += 1
        except Exception as exc:  # noqa: BLE001
            self.resync_fail += 1
            self.last_error = f"resync {type(exc).__name__}: {exc}"
        finally:
            self._resync_pending.discard(ticker)

    # ------------------------------------------------------------------ persistence
    def _put(self, row: tuple) -> None:
        try:
            self._queue.put_nowait(row)
        except asyncio.QueueFull:
            pass  # dropping a persisted row is better than stalling the receiver

    async def _persist_loop(self) -> None:
        while True:
            batch: list[tuple] = []
            try:
                batch.append(await self._queue.get())
                deadline = time.time() + 0.5
                while time.time() < deadline and len(batch) < 400:
                    try:
                        batch.append(self._queue.get_nowait())
                    except asyncio.QueueEmpty:
                        await asyncio.sleep(0.02)
            except asyncio.CancelledError:
                return
            if self.store is not None and batch:
                try:
                    self.store.ws_batch(batch)
                except Exception:  # noqa: BLE001 - persistence must not kill the writer
                    pass
