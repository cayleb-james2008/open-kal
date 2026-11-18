"""Favourite-side maker lane: gas plus the top liquidity-reward markets.

Rule (frozen for this lane):

  * universe: national AAA gas (`KXAAAGASD`) and the top reward markets by
    dollars per day, excluding multi-leg combos;
  * only markets that settle within seven days, and only where at most 2,000
    contracts are ahead of us at our price -- otherwise quote one tick inside
    the spread when the spread allows, and skip the market when it does not;
  * quote the **favourite** side (the side whose mid is at or above 50c), resting
    a bid at 85-99c; join the touch, or improve it by one tick while staying
    inside the spread and inside the band;
  * size 1-10 contracts, capped by 1% of equity and by the program's reward rate;
  * gas positions are pure favourite carry; reward markets additionally accrue a
    documented-formula *estimate*, never a claim of payment;
  * every quote, fill, markout, reward estimate, settlement and halt is logged.

The paper lane replays live trades against its resting quotes with a queue-aware
fill model: a print at the touch only fills the part beyond the size already
ahead of us; a print through the touch sweeps the queue.  That is deliberately
conservative -- the gas touch often has 10,000+ contracts ahead.

Live quotes carry an order-group id.  The group's rolling 15-second contracts
limit is the automatic kill switch; a trigger cancels every quote in the group.
`is_auto_cancel_enabled` is false while active and true after a trigger.

Commands:

    .venv/bin/python -m openkal.lanes.fav_maker ready
    .venv/bin/python -m openkal.lanes.fav_maker paper --seconds 120
    .venv/bin/python -m openkal.lanes.fav_maker status

    # only after the operator arms this lane:
    .venv/bin/python scripts/arm_live.py --lane fav-maker --by "NAME"
    .venv/bin/python -m openkal.lanes.fav_maker canary --live
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import time
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import ROUND_DOWN, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import websockets

from .. import arming
from ..auth import KalshiClient
from ..broker import Broker
from ..config import ROOT, Settings, load_settings
from ..fees import maker_fee
from ..kalshi_ws import WS_URL, KalshiWS
from ..maker import rewards
from ..maker.order_groups import OrderGroupKillSwitch, alt_get

LANE = "fav-maker"
GAS_SERIES = "KXAAAGASD"
TICK = Decimal("0.01")
PRICE_MIN = Decimal("0.85")
PRICE_MAX = Decimal("0.99")
SIZE_MIN, SIZE_MAX = 1, 10
MAX_RISK_USD = Decimal("10.00")
GAS_SIZE = 5
TOP_REWARDS = 10
REWARD_SCAN = 100
# A reward rate earned on a market that settles after the capital is needed
# elsewhere is not a reward; seven days is the frozen carrying horizon.  And a
# quote behind 2,000 contracts is a number in a queue, not a fill: improve by a
# tick when the spread allows, otherwise do not quote it.
MAX_DAYS_TO_SETTLE = 7.0
SETTLE_HORIZON_S = MAX_DAYS_TO_SETTLE * 86400.0
MAX_QUEUE_AHEAD = Decimal("2000")
GAS_QUOTES = 3
QUOTE_REFRESH_S = 15.0
REWARD_INTERVAL_S = 15.0
SETTLE_INTERVAL_S = 30.0
MARK_HORIZONS = (5.0, 30.0, 60.0)
GROUP_LIMIT = 20
DEFAULT_DB = ROOT / "data" / "fav-maker.db"
DEFAULT_PAPER_EQUITY = Decimal("1000.00")
ZERO = Decimal("0")


def dec(value: Any, default: str = "0") -> Decimal:
    if value in (None, ""):
        return Decimal(default)
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal(default)


def parse_ts(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    if isinstance(value, (int, float)):
        number = float(value)
        # Trade timestamps can arrive in milliseconds; markouts are in seconds.
        return number / 1000.0 if number > 1e12 else number
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def price_field(row: dict[str, Any], dollars: str, cents: str) -> Decimal | None:
    value = row.get(dollars)
    if value not in (None, ""):
        return dec(value)
    value = row.get(cents)
    if value not in (None, ""):
        return dec(value) / Decimal("100")
    return None


@dataclass(frozen=True)
class Snap:
    ticker: str
    series: str
    market_type: str
    status: str
    yes_bid: Decimal
    yes_ask: Decimal
    no_bid: Decimal
    no_ask: Decimal
    close_ts: float
    strike: Decimal
    mve: bool = False
    program: rewards.Program | None = None


@dataclass(frozen=True)
class Quote:
    snap: Snap
    side: str
    price: Decimal
    contracts: int
    best_bid: Decimal
    best_ask: Decimal
    queue_ahead: Decimal
    kind: str

    @property
    def ticker(self) -> str:
        return self.snap.ticker

    @property
    def reward_usd_per_day(self) -> Decimal:
        return self.snap.program.usd_per_day if self.snap.program else ZERO


@dataclass
class PaperQuote:
    db_id: int
    quote: Quote
    remaining: int
    queue_ahead: Decimal
    created_ts: float
    filled: int = 0


@dataclass
class Fill:
    db_id: int
    ts: float
    quote: Quote
    contracts: int
    yes_price: Decimal
    taker_side: str
    mid_at_fill: Decimal | None


def parse_market(row: dict[str, Any], program: rewards.Program | None = None) -> Snap | None:
    ticker = str(row.get("ticker") or "")
    if not ticker:
        return None
    yes_bid = price_field(row, "yes_bid_dollars", "yes_bid")
    yes_ask = price_field(row, "yes_ask_dollars", "yes_ask")
    if yes_bid is None or yes_ask is None:
        return None
    no_bid = price_field(row, "no_bid_dollars", "no_bid")
    no_ask = price_field(row, "no_ask_dollars", "no_ask")
    if no_bid is None:
        no_bid = Decimal(1) - yes_ask
    if no_ask is None:
        no_ask = Decimal(1) - yes_bid
    return Snap(
        ticker=ticker,
        series=str(row.get("series_ticker") or ticker.split("-")[0]),
        market_type=str(row.get("market_type") or ""),
        status=str(row.get("status") or "").lower(),
        yes_bid=yes_bid, yes_ask=yes_ask, no_bid=no_bid, no_ask=no_ask,
        close_ts=parse_ts(row.get("close_time")),
        strike=dec(row.get("floor_strike")),
        mve=bool(row.get("mve_collection_ticker") or ticker.startswith("KXMVE")),
        program=program,
    )


def favourite(snap: Snap) -> tuple[str, Decimal, Decimal] | None:
    """The side with the higher mid and its own bid/ask, if it is quoteable."""
    # A zero YES bid is normal for a deep favourite: the NO bid is 1 - yes_ask.
    # Requiring a YES bid here excluded every far-strike gas market.
    if snap.yes_ask <= 0:
        return None
    if (snap.yes_bid + snap.yes_ask) / 2 >= Decimal("0.5"):
        side, bid, ask = "YES", snap.yes_bid, snap.yes_ask
    else:
        side, bid, ask = "NO", snap.no_bid, snap.no_ask
    if bid < PRICE_MIN or bid > PRICE_MAX or ask <= bid:
        return None
    return side, bid, ask


def quote_price(bid: Decimal, ask: Decimal) -> Decimal | None:
    """Join the touch, or improve by a tick without crossing or leaving 85-99c."""
    if bid < PRICE_MIN or bid > PRICE_MAX or ask <= bid:
        return None
    improved = bid + TICK
    price = improved if improved <= PRICE_MAX and improved < ask else bid
    if price < PRICE_MIN or price > PRICE_MAX or price >= ask:
        return None
    return price


def size_for(price: Decimal, usd_per_day: Decimal, equity: Decimal) -> int:
    """1-10 contracts, bounded by 1% of equity and by the reward rate."""
    budget = min(MAX_RISK_USD, max(ZERO, equity) * Decimal("0.01"))
    risk_cap = int((budget / price).to_integral_value(rounding=ROUND_DOWN)) if price > 0 else 0
    desired = GAS_SIZE if usd_per_day <= 0 else 1 + int(min(Decimal("9"), usd_per_day / Decimal("20")))
    return max(0, min(SIZE_MAX, risk_cap, desired))


async def fetch_depth(client: KalshiClient, ticker: str) -> rewards.Depth | None:
    code, payload = await client.get(f"/markets/{ticker}/orderbook", {"depth": 20})
    if code != 200:
        return None
    return rewards.Depth.from_payload(payload)


async def quote_for(client: KalshiClient, snap: Snap, equity: Decimal,
                    depth: rewards.Depth | None = None) -> tuple[Quote | None, rewards.Depth | None]:
    if depth is None:
        depth = await fetch_depth(client, snap.ticker)
    if depth is None:
        return None, None
    # The market row can be seconds old.  Rebuild the top of book from the depth
    # we just fetched so a refresh cannot quote through the current market.
    if depth.yes_ask is not None and depth.no_bid is not None:
        snap = replace(
            snap,
            yes_bid=depth.yes_bid if depth.yes_bid is not None else ZERO,
            yes_ask=depth.yes_ask,
            no_bid=depth.no_bid,
            no_ask=depth.no_ask if depth.no_ask is not None else Decimal(1) - snap.yes_bid,
        )
    fav = favourite(snap)
    if fav is None:
        return None, depth
    side, bid, ask = fav
    price = quote_price(bid, ask)
    if price is None:
        return None, depth
    queue = depth.best_size(side, price) if price == bid else ZERO
    price = queue_price(bid, ask, price, queue)
    if price is None:
        return None, depth
    queue = depth.best_size(side, price) if price == bid else ZERO
    usd_per_day = snap.program.usd_per_day if snap.program else ZERO
    contracts = size_for(price, usd_per_day, equity)
    if contracts < SIZE_MIN:
        return None, depth
    return Quote(snap, side, price, contracts, bid, ask, queue,
                 "reward" if snap.program else "gas"), depth


def eligible(snap: Snap, now: float) -> bool:
    """Open, binary, non-combo, and settling inside the seven-day horizon."""
    return (snap.status in {"active", "open"} and snap.market_type == "binary"
            and not snap.mve
            and 0 < snap.close_ts <= now + SETTLE_HORIZON_S)


def queue_price(bid: Decimal, ask: Decimal, price: Decimal,
                queue: Decimal) -> Decimal | None:
    """The price to quote against the resting size already at it.

    At or under the cap, join as planned.  Above it, step one tick inside the
    spread -- only a price nobody else is resting at has an empty queue.  If the
    spread is already one tick wide there is nowhere to step, and the market is
    skipped rather than queued behind 2,000 contracts.
    """
    if queue <= MAX_QUEUE_AHEAD:
        return price
    improved = bid + TICK
    if improved < ask and improved <= PRICE_MAX and improved >= PRICE_MIN:
        return improved
    return None


async def build_universe(client: KalshiClient, now: float,
                         equity: Decimal) -> list[Quote]:
    """Gas candidates plus the top reward markets that already sit in the band.

    A failed program or market fetch is raised, not reported as "no markets":
    an empty universe is only legitimate when the data actually arrived.
    """
    programs = await rewards.fetch_programs(client)
    if not programs:
        raise RuntimeError("incentive_programs returned no programs")
    ranked = rewards.rank(programs, now, REWARD_SCAN)
    markets = await rewards.bulk_markets(client, [p.ticker for p in ranked])
    if ranked and not markets:
        raise RuntimeError("bulk market snapshot returned no markets")
    candidates: list[Snap] = []
    for program in ranked:
        market = markets.get(program.ticker)
        if not market:
            continue
        snap = parse_market(market, program)
        if snap and eligible(snap, now) and favourite(snap) is not None:
            candidates.append(snap)
        if len(candidates) >= 30:
            break

    code, payload = await client.get(
        "/markets", {"series_ticker": GAS_SERIES, "status": "open", "limit": 200})
    if code != 200:
        raise RuntimeError(f"gas markets HTTP {code}: {str(payload)[:160]}")
    gas: list[Snap] = []
    for market in payload.get("markets") or []:
        if str(market.get("strike_type") or "") != "greater":
            continue
        snap = parse_market(market)
        if snap and eligible(snap, now) and favourite(snap) is not None:
            gas.append(snap)
    gas.sort(key=lambda s: (s.close_ts, -s.strike))
    candidates.extend(gas[:10])

    semaphore = asyncio.Semaphore(5)

    async def one(snap: Snap) -> tuple[Snap, Quote | None, rewards.Depth | None]:
        async with semaphore:
            quote, depth = await quote_for(client, snap, equity)
        return snap, quote, depth

    results = await asyncio.gather(*(one(s) for s in candidates))
    quotes: list[Quote] = []
    seen: set[str] = set()
    reward_count = gas_count = 0
    for snap, quote, _depth in results:
        if quote is None or quote.ticker in seen:
            continue
        if snap.program is not None and reward_count >= TOP_REWARDS:
            continue
        if snap.program is None and gas_count >= GAS_QUOTES:
            continue
        quotes.append(quote)
        seen.add(quote.ticker)
        if snap.program is not None:
            reward_count += 1
        else:
            gas_count += 1
    return quotes


# --------------------------------------------------------------------------- store
class FavStore:
    def __init__(self, path: Path | str = DEFAULT_DB) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path))
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
              id INTEGER PRIMARY KEY AUTOINCREMENT, started_ts REAL, ended_ts REAL,
              mode TEXT, equity TEXT, quotes INTEGER, fills INTEGER, pnl TEXT, halted TEXT
            );
            CREATE TABLE IF NOT EXISTS quotes (
              id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, mode TEXT, ticker TEXT,
              series TEXT, side TEXT, price TEXT, contracts INTEGER, best_bid TEXT,
              best_ask TEXT, queue_ahead TEXT, program_id TEXT, reward_usd_per_day TEXT,
              kind TEXT, status TEXT, reason TEXT, order_id TEXT
            );
            CREATE TABLE IF NOT EXISTS fills (
              id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, quote_id INTEGER,
              ticker TEXT, side TEXT, price TEXT, contracts INTEGER, yes_price TEXT,
              taker_side TEXT, queue_ahead TEXT, mid_at_fill TEXT
            );
            CREATE TABLE IF NOT EXISTS marks (
              id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, fill_id INTEGER,
              horizon_s REAL, mid TEXT, markout_cents TEXT
            );
            CREATE TABLE IF NOT EXISTS rewards (
              id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, ticker TEXT,
              program_id TEXT, side TEXT, price TEXT, contracts INTEGER, reference TEXT,
              our_score TEXT, total_score TEXT, share TEXT, accrual_usd TEXT,
              target_ok INTEGER, reason TEXT, method TEXT
            );
            CREATE TABLE IF NOT EXISTS settlements (
              id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, fill_id INTEGER UNIQUE,
              ticker TEXT, side TEXT, contracts INTEGER, entry TEXT, result TEXT,
              won INTEGER, pnl_usd TEXT, fee_usd TEXT
            );
            CREATE TABLE IF NOT EXISTS halts (
              id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, reason TEXT, detail TEXT
            );
            CREATE INDEX IF NOT EXISTS quotes_ticker ON quotes(ticker);
            CREATE INDEX IF NOT EXISTS fills_quote ON fills(quote_id);
            """
        )
        self.db.commit()

    def start_run(self, mode: str, equity: Decimal) -> int:
        cur = self.db.execute(
            "INSERT INTO runs (started_ts,mode,equity,quotes,fills,pnl) VALUES (?,?,?,0,0,'0')",
            (time.time(), mode, str(equity)))
        self.db.commit()
        return int(cur.lastrowid)

    def finish_run(self, run_id: int, *, quotes: int, fills: int,
                   pnl: Decimal, halted: str | None) -> None:
        self.db.execute(
            "UPDATE runs SET ended_ts=?,quotes=?,fills=?,pnl=?,halted=? WHERE id=?",
            (time.time(), quotes, fills, str(pnl), halted, run_id))
        self.db.commit()

    def log_quote(self, mode: str, quote: Quote, status: str = "resting",
                  reason: str = "", order_id: str | None = None) -> int:
        cur = self.db.execute(
            """INSERT INTO quotes
               (ts,mode,ticker,series,side,price,contracts,best_bid,best_ask,
                queue_ahead,program_id,reward_usd_per_day,kind,status,reason,order_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (time.time(), mode, quote.ticker, quote.snap.series, quote.side,
             str(quote.price), quote.contracts, str(quote.best_bid),
             str(quote.best_ask), str(quote.queue_ahead),
             quote.snap.program.id if quote.snap.program else None,
             str(quote.reward_usd_per_day), quote.kind, status, reason, order_id))
        self.db.commit()
        return int(cur.lastrowid)

    def cancel_quote(self, quote_id: int, reason: str) -> None:
        self.db.execute("UPDATE quotes SET status='cancelled',reason=? WHERE id=?",
                        (reason, quote_id))
        self.db.commit()

    def log_fill(self, quote: Quote, quote_id: int, contracts: int,
                 yes_price: Decimal, taker_side: str, queue_ahead: Decimal,
                 mid_at_fill: Decimal | None) -> int:
        cur = self.db.execute(
            """INSERT INTO fills
               (ts,quote_id,ticker,side,price,contracts,yes_price,taker_side,
                queue_ahead,mid_at_fill) VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (time.time(), quote_id, quote.ticker, quote.side, str(quote.price),
             contracts, str(yes_price), taker_side, str(queue_ahead),
             None if mid_at_fill is None else str(mid_at_fill)))
        self.db.commit()
        return int(cur.lastrowid)

    def log_mark(self, fill_id: int, horizon: float, mid: Decimal,
                 markout_cents: Decimal) -> None:
        self.db.execute(
            "INSERT INTO marks (ts,fill_id,horizon_s,mid,markout_cents) VALUES (?,?,?,?,?)",
            (time.time(), fill_id, horizon, str(mid), str(markout_cents)))
        self.db.commit()

    def log_reward(self, ticker: str, estimate: rewards.Estimate,
                   contracts: int) -> None:
        self.db.execute(
            """INSERT INTO rewards
               (ts,ticker,program_id,side,price,contracts,reference,our_score,
                total_score,share,accrual_usd,target_ok,reason,method)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (time.time(), ticker, estimate.program_id, estimate.side,
             str(estimate.price), contracts,
             None if estimate.reference is None else str(estimate.reference),
             str(estimate.our_score), str(estimate.total_score), str(estimate.share),
             str(estimate.accrual_usd), int(estimate.target_ok), estimate.reason,
             estimate.method))
        self.db.commit()

    def log_settlement(self, fill: Fill, result: str, won: bool,
                       pnl: Decimal, fee: Decimal) -> None:
        self.db.execute(
            """INSERT OR IGNORE INTO settlements
               (ts,fill_id,ticker,side,contracts,entry,result,won,pnl_usd,fee_usd)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (time.time(), fill.db_id, fill.quote.ticker, fill.quote.side,
             fill.contracts, str(fill.quote.price), result, int(won),
             str(pnl), str(fee)))
        self.db.commit()

    def log_halt(self, reason: str, detail: str) -> None:
        self.db.execute("INSERT INTO halts (ts,reason,detail) VALUES (?,?,?)",
                        (time.time(), reason, detail))
        self.db.commit()

    def summary(self) -> dict[str, Any]:
        def count(sql: str, *args: Any) -> int:
            row = self.db.execute(sql, args).fetchone()
            return int(row[0] or 0) if row else 0

        def total(sql: str, *args: Any) -> Decimal:
            row = self.db.execute(sql, args).fetchone()
            return dec(row[0]) if row and row[0] is not None else ZERO

        marks = {}
        for horizon in MARK_HORIZONS:
            row = self.db.execute(
                "SELECT AVG(CAST(markout_cents AS REAL)) FROM marks WHERE horizon_s=?",
                (horizon,)).fetchone()
            marks[str(int(horizon))] = None if not row or row[0] is None else round(row[0], 3)
        return {
            "db": str(self.path),
            "runs": count("SELECT COUNT(*) FROM runs"),
            "quotes": count("SELECT COUNT(*) FROM quotes"),
            "resting_quotes": count("SELECT COUNT(*) FROM quotes WHERE status='resting'"),
            "fills": count("SELECT COUNT(*) FROM fills"),
            "fill_contracts": count("SELECT COALESCE(SUM(contracts),0) FROM fills"),
            "reward_estimate_usd": str(total("SELECT SUM(CAST(accrual_usd AS REAL)) FROM rewards")),
            "settlements": count("SELECT COUNT(*) FROM settlements"),
            "wins": count("SELECT COUNT(*) FROM settlements WHERE won=1"),
            "losses": count("SELECT COUNT(*) FROM settlements WHERE won=0"),
            "realized_pnl_usd": str(total("SELECT SUM(CAST(pnl_usd AS REAL)) FROM settlements")),
            "average_markout_cents": marks,
            "halts": count("SELECT COUNT(*) FROM halts"),
        }

    def close(self) -> None:
        self.db.commit()
        self.db.close()


# --------------------------------------------------------------------------- paper
class PaperBook:
    def __init__(self, client: KalshiClient, store: FavStore,
                 templates: list[Quote], equity: Decimal,
                 group_limit: int = GROUP_LIMIT) -> None:
        self.client = client
        self.store = store
        self.templates = templates
        self.equity = equity
        self.group_limit = int(group_limit)
        self.quotes: dict[str, list[PaperQuote]] = {}
        self.books: dict[str, rewards.Depth] = {}
        self.fills: list[Fill] = []
        self.fill_events: list[tuple[float, int]] = []
        self.pending_marks: list[tuple[float, Fill, float, Decimal | None]] = []
        self.settled_fill_ids: set[int] = set()
        self.halted: str | None = None
        self.tape_error: str | None = None

    def add(self, quote: Quote, now: float | None = None,
            queue_ahead: Decimal | None = None) -> PaperQuote:
        now = time.time() if now is None else now
        queue = quote.queue_ahead if queue_ahead is None else queue_ahead
        db_id = self.store.log_quote("paper", quote)
        paper = PaperQuote(db_id, quote, quote.contracts, queue, now)
        self.quotes.setdefault(quote.ticker, []).append(paper)
        return paper

    async def refresh(self) -> None:
        now = time.time()
        self.cancel_all(now, "refresh")
        for template in self.templates:
            quote, depth = await quote_for(self.client, template.snap, self.equity)
            if depth is not None:
                self.books[template.ticker] = depth
            if quote is None:
                continue
            self.add(quote, now)

    def cancel_all(self, now: float, reason: str) -> None:
        for rows in self.quotes.values():
            for paper in rows:
                self.store.cancel_quote(paper.db_id, reason)
        self.quotes = {}

    def _halt(self, now: float, detail: str) -> None:
        self.halted = detail
        self.cancel_all(now, "order_group_trigger")
        self.store.log_halt("order_group_trigger", detail)

    def on_trade(self, ticker: str, yes_price: Decimal, count: Decimal,
                 taker_side: str, ts: float | None = None) -> None:
        if self.halted or count <= 0:
            return
        now = time.time() if ts is None else float(ts)
        for paper in list(self.quotes.get(ticker, [])):
            quote = paper.quote
            line = quote.price if quote.side == "YES" else Decimal(1) - quote.price
            if quote.side == "YES":
                hit = taker_side.lower() == "no" and yes_price <= quote.price
            else:
                hit = taker_side.lower() == "yes" and yes_price <= line
            if not hit:
                continue
            sweep = yes_price < line
            available = count if sweep else max(ZERO, count - paper.queue_ahead)
            quantity = int(min(Decimal(paper.remaining), available))
            if quantity <= 0:
                if not sweep:
                    paper.queue_ahead = max(ZERO, paper.queue_ahead - count)
                continue
            book = self.books.get(ticker)
            mid = book.mid(quote.side) if book else None
            fill_id = self.store.log_fill(
                quote, paper.db_id, quantity, yes_price, taker_side.lower(),
                paper.queue_ahead, mid)
            fill = Fill(fill_id, now, quote, quantity, yes_price,
                        taker_side.lower(), mid)
            self.fills.append(fill)
            paper.remaining -= quantity
            paper.filled += quantity
            if not sweep:
                paper.queue_ahead = max(ZERO, paper.queue_ahead - count)
            self.fill_events.append((now, quantity))
            for horizon in MARK_HORIZONS:
                self.pending_marks.append((now + horizon, fill, horizon, mid))
            if paper.remaining <= 0:
                self.quotes[ticker].remove(paper)
            recent = sum(c for t, c in self.fill_events if now - t <= 15.0)
            if recent > self.group_limit:
                self._halt(now, f"filled {recent} contracts in 15s > group limit {self.group_limit}")
                return

    async def process_marks(self, now: float) -> None:
        due = [m for m in self.pending_marks if m[0] <= now]
        for _due_ts, fill, horizon, _mid_at_fill in due:
            depth = await fetch_depth(self.client, fill.quote.ticker)
            mid = depth.mid(fill.quote.side) if depth else None
            if mid is not None:
                markout = (mid - fill.quote.price) * Decimal("100")
                self.store.log_mark(fill.db_id, horizon, mid, markout)
            self.pending_marks.remove((_due_ts, fill, horizon, _mid_at_fill))

    async def reward_estimates(self, now: float) -> None:
        for rows in self.quotes.values():
            for paper in rows:
                program = paper.quote.snap.program
                if program is None:
                    continue
                depth = await fetch_depth(self.client, paper.quote.ticker)
                if depth is None:
                    continue
                estimate = rewards.estimate(
                    program, depth, paper.quote.side, paper.quote.price,
                    Decimal(paper.remaining))
                self.store.log_reward(paper.quote.ticker, estimate, paper.remaining)

    async def settle_positions(self, now: float) -> None:
        tickers = {f.quote.ticker for f in self.fills
                   if f.db_id not in self.settled_fill_ids}
        for ticker in tickers:
            code, payload = await self.client.get(f"/markets/{ticker}")
            if code != 200:
                continue
            market = payload.get("market") or payload
            result = str(market.get("result") or "").lower()
            if result not in {"yes", "no"}:
                continue
            for fill in [f for f in self.fills if f.quote.ticker == ticker
                         and f.db_id not in self.settled_fill_ids]:
                won = result == fill.quote.side.lower()
                pnl = (Decimal(fill.contracts) * (Decimal(1) - fill.quote.price)
                       if won else -Decimal(fill.contracts) * fill.quote.price)
                fee = maker_fee(fill.quote.price, fill.contracts)
                self.store.log_settlement(fill, result, won, pnl, fee)
                self.settled_fill_ids.add(fill.db_id)

    async def run(self, seconds: float) -> dict[str, Any]:
        run_id = self.store.start_run("paper", self.equity)
        await self.refresh()
        stop = asyncio.Event()
        tape = TradeTape(self.client, self.client.settings,
                         [q.ticker for q in self.templates], self.on_trade)
        tape_task = asyncio.create_task(tape.run(stop))
        started = time.time()
        end = started + max(1.0, seconds)
        last_refresh = last_reward = last_settle = time.time()
        try:
            while time.time() < end and not self.halted:
                await asyncio.sleep(1.0)
                now = time.time()
                await self.process_marks(now)
                if now - last_refresh >= QUOTE_REFRESH_S:
                    await self.refresh()
                    last_refresh = now
                if now - last_reward >= REWARD_INTERVAL_S:
                    await self.reward_estimates(now)
                    last_reward = now
                if now - last_settle >= SETTLE_INTERVAL_S:
                    await self.settle_positions(now)
                    last_settle = now
                self.tape_error = tape.error
        finally:
            stop.set()
            tape_task.cancel()
            await asyncio.gather(tape_task, return_exceptions=True)
            self.cancel_all(time.time(), "run_end")
        result = self.store.summary()
        result["paper"] = {
            "quotes_at_end": sum(len(v) for v in self.quotes.values()),
            "fills": len(self.fills),
            "halted": self.halted,
            "tape_error": self.tape_error,
        }
        self.store.finish_run(
            run_id, quotes=result["quotes"], fills=result["fills"],
            pnl=dec(result["realized_pnl_usd"]), halted=self.halted)
        return result


class TradeTape:
    def __init__(self, client: KalshiClient, settings: Settings,
                 tickers: list[str], on_trade) -> None:
        self.client = client
        self.settings = settings
        self.tickers = sorted(set(tickers))
        self.on_trade = on_trade
        self.error: str | None = None

    async def run(self, stop: asyncio.Event) -> None:
        helper = KalshiWS(self.settings, self.client)
        backoff = 2.0
        while not stop.is_set():
            try:
                async with websockets.connect(WS_URL, **helper.connect_kwargs()) as socket:
                    await socket.send(json.dumps({
                        "id": 42, "cmd": "subscribe",
                        "params": {"channels": ["trade"],
                                   "market_tickers": self.tickers}}))
                    backoff = 2.0
                    self.error = None
                    async for raw in socket:
                        if stop.is_set():
                            return
                        try:
                            msg = json.loads(raw)
                        except ValueError:
                            continue
                        if msg.get("type") == "error":
                            self.error = json.dumps(msg.get("msg") or {})[:300]
                            continue
                        if msg.get("type") != "trade":
                            continue
                        body = msg.get("msg") or {}
                        yes_price = price_field(body, "yes_price_dollars", "yes_price")
                        count = dec(body.get("count_fp") or body.get("count"))
                        if yes_price is None or count <= 0:
                            continue
                        self.on_trade(str(body.get("market_ticker") or ""),
                                      yes_price, count,
                                      str(body.get("taker_side") or ""),
                                      parse_ts(body.get("ts")) or time.time())
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the tape reconnects
                self.error = f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(backoff)
                backoff = min(30.0, backoff * 1.8)


# --------------------------------------------------------------------------- live
def require_operator_arm(settings: Settings) -> dict[str, Any]:
    if settings.env_name != "prod" or not settings.allow_live:
        raise RuntimeError("live requires KALSHI_ENV=prod and ALLOW_LIVE=1")
    path = arming.arm_path(LANE)
    record = arming.consume(path)
    if not arming.fresh_operator_act(record, now=time.time(), lane=LANE):
        raise RuntimeError(
            f"no fresh arm at {path}; run scripts/arm_live.py --lane {LANE} --by NAME")
    return record or {}


async def advanced_or_error(client: KalshiClient) -> str | None:
    code, payload = await alt_get(client, "/account/limits")
    if code != 200:
        return f"/account/limits HTTP {code}: {str(payload)[:160]}"
    tier = str((payload or {}).get("usage_tier") or "").lower()
    if tier != "advanced":
        return f"Advanced tier required, venue reports {tier or 'unknown'}"
    return None


async def canary(settings: Settings) -> dict[str, Any]:
    client = KalshiClient(settings)
    tier_error = await advanced_or_error(client)
    if tier_error:
        await client.close()
        return {"ok": False, "error": tier_error}
    record = require_operator_arm(settings)
    broker = Broker(client)
    group = OrderGroupKillSwitch(client, 1)
    created = await group.create()
    if created is None or created.triggered:
        await group.delete()
        await client.close()
        return {"ok": False, "error": group.error or "group not active"}
    try:
        equity = await broker.balance()
        quotes = await build_universe(client, time.time(), equity)
        gas = [q for q in quotes if q.kind == "gas"]
        if not gas:
            return {"ok": False, "error": "no gas quote in the band"}
        quote = gas[0]
        order = await broker.place_fav_quote(
            quote.ticker, quote.side, quote.price, 1, created.id)
        if order is None:
            return {"ok": False, "error": broker.errors[-1] if broker.errors else "order rejected"}
        await asyncio.sleep(5.0)
        fills = await broker.fills(order_id=order.order_id)
        await broker.cancel(order.order_id, quote.ticker)
        await group.trigger()
        return {
            "ok": True, "armed_by": record.get("by"), "ticker": quote.ticker,
            "side": quote.side, "price": str(quote.price), "order_id": order.order_id,
            "fills": len(fills), "group_id": created.id,
        }
    finally:
        await group.trigger()
        await group.delete()
        await client.close()


async def live(settings: Settings, seconds: float = 3600.0) -> dict[str, Any]:
    client = KalshiClient(settings)
    tier_error = await advanced_or_error(client)
    if tier_error:
        await client.close()
        return {"ok": False, "error": tier_error}
    record = require_operator_arm(settings)
    broker = Broker(client)
    store = FavStore(DEFAULT_DB)
    group = OrderGroupKillSwitch(client, GROUP_LIMIT)
    created = await group.create()
    if created is None or created.triggered:
        await group.delete()
        await client.close()
        return {"ok": False, "error": group.error or "group not active"}
    equity = await broker.balance()
    quotes = await build_universe(client, time.time(), equity)
    placed: list[tuple[str, str]] = []
    run_id = store.start_run("live", equity)
    halted: str | None = None
    try:
        for quote in quotes:
            order = await broker.place_fav_quote(
                quote.ticker, quote.side, quote.price, quote.contracts, created.id)
            if order is not None:
                store.log_quote("live", quote, order_id=order.order_id)
                placed.append((order.order_id, quote.ticker))
        if not placed:
            return {"ok": False, "error": "no quote was accepted by the venue",
                    "quotes": len(quotes), "broker_errors": broker.errors[-5:]}
        end = time.time() + max(1.0, seconds)
        while time.time() < end:
            await asyncio.sleep(5.0)
            status = await group.status()
            if status is not None and status.triggered:
                halted = "venue reported order group triggered"
                store.log_halt("order_group_trigger", halted)
                break
        return {"ok": True, "armed_by": record.get("by"), "group_id": created.id,
                "placed": len(placed), "quotes": len(quotes), "equity": str(equity)}
    finally:
        for order_id, ticker in placed:
            await broker.cancel(order_id, ticker)
        await group.trigger()
        await group.delete()
        summary = store.summary()
        store.finish_run(run_id, quotes=summary["quotes"], fills=summary["fills"],
                         pnl=dec(summary["realized_pnl_usd"]), halted=halted)
        store.close()
        await client.close()


def paper_equity(settings: Settings) -> Decimal:
    """The paper bankroll; `.env` sets 0 for the live engine, not for this lane."""
    return settings.paper_equity if settings.paper_equity > 0 else DEFAULT_PAPER_EQUITY


async def ready(settings: Settings) -> dict[str, Any]:
    client = KalshiClient(settings)
    try:
        tier_code, tier = await alt_get(client, "/account/limits")
        group = OrderGroupKillSwitch(client, GROUP_LIMIT)
        cycle_ok, problems, cycle = await group.verify_cycle()
        equity = paper_equity(settings)
        try:
            quotes = await build_universe(client, time.time(), equity)
            universe_error = None
        except Exception as exc:  # noqa: BLE001 - report, do not pretend empty is healthy
            quotes = []
            universe_error = f"{type(exc).__name__}: {exc}"
        record = arming.read(arming.arm_path(LANE))
        fresh = arming.fresh_operator_act(record, now=time.time(), lane=LANE) if record else False
        tier_ok = (tier_code == 200
                   and str((tier or {}).get("usage_tier") or "").lower() == "advanced")
        return {
            "lane": LANE,
            "tier": tier if tier_code == 200 else {"status": tier_code, "body": tier},
            "tier_ok": tier_ok,
            "group_cycle": {"ok": cycle_ok, "problems": problems, **cycle},
            "universe": [
                {"ticker": q.ticker, "kind": q.kind, "side": q.side,
                 "price": str(q.price), "contracts": q.contracts,
                 "best_bid": str(q.best_bid), "best_ask": str(q.best_ask),
                 "queue_ahead": str(q.queue_ahead),
                 "reward_usd_per_day": str(q.reward_usd_per_day)}
                for q in quotes
            ],
            "paper_equity": str(equity),
            "universe_error": universe_error,
            "paper_ready": bool(quotes),
            "live_ready": bool(quotes) and cycle_ok and tier_ok,
            "arm_record": record,
            "arm_fresh": fresh,
            "arm_command": f"scripts/arm_live.py --lane {LANE} --by NAME",
            "note": ("Group auto-cancel is inherent: active groups report "
                     "is_auto_cancel_enabled=false; a trigger flips it true."),
        }
    finally:
        await client.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fav_maker", description="Favourite-side maker lane")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ready", help="verify tier, group kill switch and quote plans")
    paper = sub.add_parser("paper", help="run the paper lane")
    paper.add_argument("--seconds", type=float, default=120.0)
    paper.add_argument("--loop", action="store_true",
                       help="repeat --seconds runs forever (use 3600 for a day loop)")
    paper.add_argument("--db", default=str(DEFAULT_DB))
    status = sub.add_parser("status", help="show the lane database totals")
    status.add_argument("--db", default=str(DEFAULT_DB))
    can = sub.add_parser("canary", help="one-contract live canary (requires arm)")
    can.add_argument("--live", action="store_true")
    live_cmd = sub.add_parser("live", help="live quoter (requires arm)")
    live_cmd.add_argument("--live", action="store_true")
    live_cmd.add_argument("--seconds", type=float, default=3600.0)
    args = parser.parse_args(argv)

    if args.command == "ready":
        print(json.dumps(asyncio.run(ready(load_settings())), indent=2))
        return 0
    if args.command == "status":
        store = FavStore(args.db)
        print(json.dumps(store.summary(), indent=2))
        store.close()
        return 0
    if args.command == "paper":
        async def run_paper() -> dict[str, Any]:
            settings = load_settings()
            client = KalshiClient(settings)
            store = FavStore(args.db)
            equity = paper_equity(settings)
            try:
                quotes = await build_universe(client, time.time(), equity)
            except Exception as exc:  # noqa: BLE001 - the command reports the real cause
                store.close()
                await client.close()
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            if not quotes:
                store.close()
                await client.close()
                return {"ok": False, "error": "no quoteable markets in the 85-99c band"}
            book = PaperBook(client, store, quotes, equity)
            print(json.dumps({"plans": [
                {"ticker": q.ticker, "kind": q.kind, "side": q.side,
                 "price": str(q.price), "contracts": q.contracts,
                 "queue_ahead": str(q.queue_ahead)}
                for q in quotes]}, indent=2), flush=True)
            result = await book.run(args.seconds)
            store.close()
            await client.close()
            return result
        while True:
            print(json.dumps(asyncio.run(run_paper()), indent=2), flush=True)
            if not args.loop:
                break
            time.sleep(1.0)
        return 0
    if not args.live:
        print(json.dumps({"ok": False, "error": "add --live after arming"}, indent=2))
        return 2
    try:
        if args.command == "canary":
            result = asyncio.run(canary(load_settings()))
        else:
            result = asyncio.run(live(load_settings(), args.seconds))
    except Exception as exc:  # noqa: BLE001 - the operator gets a refusal, not a traceback
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
