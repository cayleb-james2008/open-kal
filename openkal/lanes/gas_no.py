"""AAA gas/diesel far-strike NO lane -- a separate module from the crypto engine.

Frozen rule (do not tune):

  * daily AAA gas (`KXAAAGASD`) and diesel (`KXDIESELD`) markets;
  * look once, at 6h after the market opens (14:00 ET for an 08:00 open);
  * buy NO on a strike whose NO price is 95-99c, only when at least 10 contracts
    rest at that price on the live book;
  * 10 contracts per signal, at most 4 open positions and 2 per event, farthest
    strikes first;
  * a limit at the seen price with `immediate_or_cancel`, never chasing;
  * hold to settlement.

Pauses (never a hard stop -- log and wait for the operator):

  * 2 losses in one ET day;
  * a 20% drawdown on this lane's realized equity;
  * a day whose AAA daily change is >= 2c national gas, or >= 3c for state gas
    and diesel.  An unavailable change also pauses that series: the rule exists
    to avoid large-move days, so an unknown move is not permission to trade.

`Kalshi V2` quotes every order price on the YES leg.  Buying NO at 0.95 is
`side="ask"` with a wire price of 0.05; `Broker.place_no_entry` exists for exactly
that reason.  The lane never uses the crypto broker's NO path.

Run:

    .venv/bin/python -m openkal.lanes.gas_no dry
    .venv/bin/python -m openkal.lanes.gas_no status
    .venv/bin/python -m openkal.lanes.gas_no settle
    .venv/bin/python -m openkal.lanes.gas_no canary --live
    .venv/bin/python -m openkal.lanes.gas_no execute --live

Live commands require `ALLOW_LIVE=1`, `KALSHI_ENV=prod`, no `OPENKAL_FORCE_PAPER`,
and a fresh record from
`scripts/arm_live.py --lane gas-no --by "the operator's name"`.
"""
from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from decimal import ROUND_DOWN, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo

from .. import arming
from ..auth import KalshiClient
from ..broker import Broker
from ..config import ROOT, load_settings
from ..fees import taker_fee

LANE = "gas-no"
ET = ZoneInfo("America/New_York")

# --- the frozen rule -------------------------------------------------------
NO_MIN = Decimal("0.95")
NO_MAX = Decimal("0.99")
MIN_DEPTH = Decimal("10")
SIZE = 10
CANARY_SIZE = 1
MAX_OPEN = 4
MAX_PER_EVENT = 2
DECISION_DELAY_S = 6 * 3600
# The 6h mark is a moment, not an all-day invitation.  An hour is enough for a
# scheduled process to start and read the book; after it, the look is spent.
DECISION_GRACE_S = 3600.0

LOSS_PAUSE_COUNT = 2
LANE_DRAWDOWN = Decimal("0.20")
DEFAULT_EQUITY = Decimal("100.00")

NATIONAL_GAS = "KXAAAGASD"
DIESEL = "KXDIESELD"
# The lane as specified: national AAA gas and national diesel.  State gas
# (`KXAAAGASD<ST>`) is recognised by the pause table below if a state series is
# ever added, but it is not scanned by default.
SERIES = (NATIONAL_GAS, DIESEL)

STATE_PATH = ROOT / "data" / "gas-no-state.json"
LOG_PATH = ROOT / "data" / "gas-no.log"


def _dec(value: Any, default: str = "0") -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal(default)


def _parse_ts(value: Any) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _et(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, ET)


def _fmt_ts(ts: float) -> str:
    return _et(ts).strftime("%Y-%m-%d %H:%M:%S %Z")


def _et_day(ts: float) -> str:
    return _et(ts).date().isoformat()


def series_kind(series: str) -> str:
    """`national_gas`, `state_gas`, `diesel`, or `unknown` (strictest pause)."""
    name = (series or "").upper()
    if name == NATIONAL_GAS:
        return "national_gas"
    if name.startswith("KXAAAGAS"):
        return "state_gas"
    if name.startswith("KXDIESEL"):
        return "diesel"
    return "unknown"


PAUSE_CHANGE = {
    "national_gas": Decimal("0.02"),
    "state_gas": Decimal("0.03"),
    "diesel": Decimal("0.03"),
    "unknown": Decimal("0.02"),
}


def pause_threshold(series: str) -> Decimal:
    return PAUSE_CHANGE[series_kind(series)]


# --- market data -----------------------------------------------------------
@dataclass(frozen=True)
class Market:
    ticker: str
    event: str
    series: str
    strike: Decimal
    open_ts: float
    yes_bid: Decimal
    yes_bid_size: Decimal

    @property
    def no_price(self) -> Decimal:
        """What buying NO costs: the NO ask is `1 - best YES bid`."""
        return Decimal(1) - self.yes_bid

    @property
    def decision_ts(self) -> float:
        return self.open_ts + DECISION_DELAY_S


@dataclass(frozen=True)
class DailyChange:
    series: str
    latest: Decimal
    prior: Decimal
    change: Decimal
    asof: str


@dataclass(frozen=True)
class Candidate:
    market: Market
    no_price: Decimal
    depth: Decimal
    distance: Decimal | None


@dataclass(frozen=True)
class PlannedOrder:
    ticker: str
    event: str
    series: str
    strike: Decimal
    side: str
    price: Decimal
    contracts: int
    tif: str
    depth: Decimal
    distance: Decimal | None
    reason: str

    @property
    def premium(self) -> Decimal:
        return self.price * self.contracts

    @property
    def fee(self) -> Decimal:
        return taker_fee(self.price, self.contracts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "event": self.event,
            "series": self.series,
            "strike": str(self.strike),
            "side": self.side,
            "price": str(self.price),
            "contracts": self.contracts,
            "time_in_force": self.tif,
            "premium": str(self.premium),
            "fee": str(self.fee),
            "seen_depth": str(self.depth),
            "distance": None if self.distance is None else str(self.distance),
            "reason": self.reason,
        }


def book_top(payload: dict[str, Any]) -> tuple[Decimal, Decimal] | None:
    """Best YES bid and the size resting at that price.

    Kalshi publishes bids only; a NO ask at 95c is a YES bid at 5c.  The legacy
    `yes` shape is in cents, `yes_dollars` in dollars.
    """
    ob = payload.get("orderbook_fp") or payload.get("orderbook") or {}
    rows = ob.get("yes_dollars")
    scale = Decimal("1")
    if rows is None:
        rows = ob.get("yes") or []
        scale = Decimal("100")
    best: tuple[Decimal, Decimal] | None = None
    for row in rows or []:
        try:
            price = _dec(row[0]) / scale
            size = _dec(row[1])
        except (IndexError, TypeError):
            continue
        if size <= 0:
            continue
        if best is None or price > best[0]:
            best = (price, size)
        elif price == best[0]:
            best = (price, best[1] + size)
    return best


async def fetch_open_markets(client: KalshiClient, series_list: Sequence[str]) -> list[Market]:
    """Every open `greater` market in the lane's series, with its live top bid."""
    markets: list[Market] = []
    for series in series_list:
        status, payload = await client.get(
            "/events", {"series_ticker": series, "status": "open", "limit": 20})
        if status != 200:
            continue
        for event in payload.get("events") or []:
            event_ticker = event.get("event_ticker")
            if not event_ticker:
                continue
            code, body = await client.get(
                "/markets", {"event_ticker": event_ticker, "limit": 100})
            if code != 200:
                continue
            for row in body.get("markets") or []:
                if (row.get("strike_type") or "") != "greater":
                    continue
                if (row.get("status") or "").lower() in {"settled", "finalized", "closed"}:
                    continue
                open_ts = _parse_ts(row.get("open_time"))
                strike = _dec(row.get("floor_strike"))
                ticker = row.get("ticker") or ""
                if not ticker or open_ts <= 0 or strike <= 0:
                    continue
                code, book = await client.get(
                    f"/markets/{ticker}/orderbook", {"depth": 100})
                if code != 200:
                    continue
                top = book_top(book)
                if top is None:
                    continue
                markets.append(Market(
                    ticker=ticker, event=event_ticker, series=series, strike=strike,
                    open_ts=open_ts, yes_bid=top[0], yes_bid_size=top[1]))
    markets.sort(key=lambda m: (m.event, -m.strike))
    return markets


def parse_event_date(event_ticker: str) -> date | None:
    try:
        return datetime.strptime(event_ticker.rsplit("-", 1)[-1], "%y%b%d").date()
    except (IndexError, ValueError):
        return None


async def fetch_daily_change(client: KalshiClient, series: str) -> DailyChange | None:
    """Today's AAA change from the venue's own settled values.

    The series settles with `expiration_value` set to the AAA print, so the two
    most recent settled events are today's and yesterday's values.  No scraping:
    the contract's own settlement source is the number that resolves it.
    """
    status, payload = await client.get(
        "/events", {"series_ticker": series, "status": "settled", "limit": 10})
    if status != 200:
        return None
    events: list[tuple[date, str]] = []
    for event in payload.get("events") or []:
        day = parse_event_date(event.get("event_ticker") or "")
        if day is not None:
            events.append((day, event.get("event_ticker") or ""))
    events.sort(reverse=True)
    values: list[tuple[date, Decimal]] = []
    for day, event_ticker in events:
        code, body = await client.get(
            "/markets", {"event_ticker": event_ticker, "limit": 100})
        if code != 200:
            continue
        found = {str(row.get("expiration_value"))
                 for row in body.get("markets") or []
                 if row.get("expiration_value") not in (None, "")}
        if len(found) != 1:
            continue
        values.append((day, _dec(found.pop())))
        if len(values) == 2:
            break
    if len(values) < 2:
        return None
    (new_day, new_value), (old_day, old_value) = values
    return DailyChange(series=series, latest=new_value, prior=old_value,
                       change=new_value - old_value, asof=new_day.isoformat())


async def fetch_changes(client: KalshiClient,
                        series_list: Sequence[str]) -> dict[str, DailyChange | None]:
    results = await asyncio.gather(
        *(fetch_daily_change(client, series) for series in series_list),
        return_exceptions=True)
    return {series: (result if isinstance(result, DailyChange) else None)
            for series, result in zip(series_list, results)}


# --- the rule --------------------------------------------------------------
def candidate_for(market: Market, reference: Decimal | None = None) -> Candidate | None:
    """The strike qualifies only inside 95-99c *and* with 10 resting contracts."""
    price = market.no_price
    if not (NO_MIN <= price <= NO_MAX):
        return None
    if market.yes_bid_size < MIN_DEPTH:
        return None
    distance = abs(market.strike - reference) if reference is not None else None
    return Candidate(market=market, no_price=price,
                     depth=market.yes_bid_size, distance=distance)


def ineligible_reason(market: Market) -> str:
    price = market.no_price
    if not (NO_MIN <= price <= NO_MAX):
        return f"NO {price:.2f} outside 95-99c"
    if market.yes_bid_size < MIN_DEPTH:
        return f"depth {market.yes_bid_size:g} < {MIN_DEPTH:g}"
    return ""


def candidates_for(markets: Iterable[Market],
                   references: dict[str, Decimal] | None = None
                   ) -> list[Candidate]:
    references = references or {}
    out: list[Candidate] = []
    for market in markets:
        candidate = candidate_for(market, references.get(market.series))
        if candidate is not None:
            out.append(candidate)
    return out


def _far_key(candidate: Candidate) -> tuple[Decimal, Decimal, Decimal, str]:
    distance = candidate.distance if candidate.distance is not None else Decimal("0")
    return (-distance, -candidate.no_price, -candidate.market.strike,
            candidate.market.ticker)


def plan_orders(candidates: Sequence[Candidate],
                open_positions: Sequence[dict[str, Any]] = (),
                *, size: int = SIZE, max_open: int = MAX_OPEN,
                per_event: int = MAX_PER_EVENT) -> list[PlannedOrder]:
    """Farthest strikes first, inside the per-event and lane-wide caps.

    A candidate already held is skipped rather than doubled: the rule's cap is
    on open positions, and the same strike is the same position.
    """
    held = {pos.get("ticker") for pos in open_positions if pos.get("open", True)}
    counts: Counter[str] = Counter(
        pos.get("event") for pos in open_positions if pos.get("open", True))
    total = sum(counts.values())
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.market.ticker in held:
            continue
        grouped[candidate.market.event].append(candidate)

    orders: list[PlannedOrder] = []
    for event in sorted(grouped,
                        key=lambda e: (min(c.market.decision_ts for c in grouped[e]), e)):
        room = min(per_event - counts.get(event, 0), max_open - total)
        if room <= 0:
            continue
        for candidate in sorted(grouped[event], key=_far_key)[:room]:
            orders.append(PlannedOrder(
                ticker=candidate.market.ticker,
                event=event,
                series=candidate.market.series,
                strike=candidate.market.strike,
                side="NO",
                price=candidate.no_price,
                contracts=size,
                tif="immediate_or_cancel",
                depth=candidate.depth,
                distance=candidate.distance,
                reason=(f"NO {candidate.no_price:.2f} with {candidate.depth:g} "
                        f"resting, farthest-first"),
            ))
            total += 1
    return orders


def actionable(market: Market, now: float) -> bool:
    """The 6h look has arrived and has not yet expired."""
    return market.decision_ts <= now <= market.decision_ts + DECISION_GRACE_S


# --- pause arithmetic ------------------------------------------------------
def settlement_pnl(entry: Decimal, contracts: int, fee: Decimal, *, won: bool) -> Decimal:
    """A NO held to settlement: a win pays $1, a loss pays nothing.

    The entry fee is the only fee; settlement itself is not a trade.
    """
    if contracts <= 0:
        return -_dec(fee)
    gross = Decimal(contracts) * (Decimal(1) - entry) if won else -Decimal(contracts) * entry
    return gross - _dec(fee)


def resolve_no(*, result: str | None, value: Any, strike: Decimal) -> bool | None:
    """Whether a NO won.  The venue's result is authoritative; the value is a fallback."""
    verdict = (result or "").strip().lower()
    if verdict == "no":
        return True
    if verdict == "yes":
        return False
    if value not in (None, ""):
        return _dec(value) <= strike
    return None


def losses_today(positions: Sequence[dict[str, Any]], now: float) -> int:
    day = _et_day(now)
    losses = 0
    for pos in positions:
        if pos.get("open", True) or pos.get("pnl") in (None, ""):
            continue
        settled = pos.get("settled_ts")
        if settled is None or _et_day(float(settled)) != day:
            continue
        if _dec(pos.get("pnl")) < 0:
            losses += 1
    return losses


def drawdown_tripped(equity: Decimal, peak: Decimal) -> bool:
    if peak <= 0:
        return False
    return equity <= peak * (Decimal(1) - LANE_DRAWDOWN)


def series_pause_reason(change: DailyChange | None, series: str) -> str | None:
    """The day's move, or a refusal to trade a series whose move is unknown."""
    if change is None:
        return f"{series}: daily change unavailable"
    threshold = pause_threshold(series)
    if abs(change.change) >= threshold:
        return (f"{series} daily change {abs(change.change) * 100:.2f}c "
                f">= {threshold * 100:.0f}c")
    return None


def lane_pause_reasons(state: dict[str, Any], now: float) -> list[str]:
    """Reasons that stop the whole lane, independent of any one series."""
    reasons: list[str] = []
    losses = losses_today(state.get("positions") or [], now)
    if losses >= LOSS_PAUSE_COUNT:
        reasons.append(f"{losses} losses today")
    equity = lane_equity(state)
    peak = _dec(state.get("peak") or equity)
    if drawdown_tripped(equity, peak):
        reasons.append(f"lane drawdown {(peak - equity) / peak * 100:.1f}% "
                       f">= {LANE_DRAWDOWN * 100:.0f}%")
    return reasons


def lane_equity(state: dict[str, Any]) -> Decimal:
    start = _dec(state.get("equity_start") or DEFAULT_EQUITY)
    realized = sum((_dec(pos.get("pnl")) for pos in state.get("positions") or []
                    if not pos.get("open", True) and pos.get("pnl") not in (None, "")),
                   Decimal("0"))
    return start + realized


# --- state -----------------------------------------------------------------
def new_state(equity_start: Decimal | None = None) -> dict[str, Any]:
    start = equity_start if equity_start is not None else DEFAULT_EQUITY
    return {
        "lane": LANE,
        "created_ts": time.time(),
        "equity_start": str(start),
        "peak": str(start),
        "positions": [],
        "canary": None,
        "pause": None,
        "series_pause": {},
        "evaluated": {},
    }


class LaneLedger:
    """Small durable ledger for one lane: positions, pauses, canary, log."""

    def __init__(self, path: Path | None = None, log_path: Path | None = None,
                 *, equity_start: Decimal | None = None) -> None:
        self.path = Path(path) if path is not None else STATE_PATH
        self.log_path = Path(log_path) if log_path is not None else LOG_PATH
        self.data = self._load(equity_start)

    def _load(self, equity_start: Decimal | None) -> dict[str, Any]:
        data = new_state(equity_start)
        try:
            raw = json.loads(self.path.read_text())
        except (OSError, ValueError):
            raw = {}
        if isinstance(raw, dict):
            data.update(raw)
        if not isinstance(data.get("positions"), list):
            data["positions"] = []
        if not isinstance(data.get("series_pause"), dict):
            data["series_pause"] = {}
        if not isinstance(data.get("evaluated"), dict):
            data["evaluated"] = {}
        return data

    # -- reads
    @property
    def equity_start(self) -> Decimal:
        return _dec(self.data.get("equity_start") or DEFAULT_EQUITY)

    def equity(self) -> Decimal:
        return lane_equity(self.data)

    def peak(self) -> Decimal:
        return _dec(self.data.get("peak") or self.equity_start)

    def realized(self) -> Decimal:
        return self.equity() - self.equity_start

    def open_positions(self) -> list[dict[str, Any]]:
        return [p for p in self.data.get("positions") or [] if p.get("open", True)]

    def canary_ok(self) -> bool:
        return bool((self.data.get("canary") or {}).get("ok"))

    def pause(self) -> dict[str, Any] | None:
        return self.data.get("pause")

    def series_pause(self, series: str, now: float) -> str | None:
        row = (self.data.get("series_pause") or {}).get(series)
        if not row:
            return None
        if row.get("day") != _et_day(now):
            return None  # a daily-move pause is a condition for that day only
        return row.get("reason")

    def evaluated(self, event: str) -> bool:
        return event in (self.data.get("evaluated") or {})

    # -- writes
    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, default=str) + "\n")
        os.replace(tmp, self.path)

    def log(self, kind: str, **fields: Any) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        row = {"ts": time.time(), "lane": LANE, "kind": kind}
        row.update(fields)
        with self.log_path.open("a") as handle:
            handle.write(json.dumps(row, default=str) + "\n")

    def latch_pause(self, reason: str, now: float) -> dict[str, Any]:
        existing = self.data.get("pause")
        if existing:
            return existing
        row = {"reason": reason, "ts": now, "at": _fmt_ts(now)}
        self.data["pause"] = row
        self.log("pause", **row)
        self.save()
        return row

    def resume(self, by: str, now: float) -> bool:
        name = (by or "").strip()
        if not name:
            raise ValueError("resuming requires the operator's name (--by)")
        had = self.data.get("pause")
        self.data["pause"] = None
        self.log("resume", by=name, previous=had)
        self.save()
        return bool(had)

    def pause_series(self, series: str, reason: str, now: float) -> None:
        day = _et_day(now)
        row = (self.data.get("series_pause") or {}).get(series)
        if row and row.get("day") == day and row.get("reason") == reason:
            return
        self.data.setdefault("series_pause", {})[series] = {
            "reason": reason, "ts": now, "day": day}
        self.log("series_pause", series=series, reason=reason, day=day)
        self.save()

    def mark_evaluated(self, event: str, now: float) -> None:
        self.data.setdefault("evaluated", {})[event] = now
        self.save()

    def record_position(self, *, ticker: str, event: str, series: str, strike: Decimal,
                        contracts: int, entry: Decimal, fee: Decimal, ts: float,
                        order_id: str = "", canary: bool = False) -> dict[str, Any]:
        row = {
            "ticker": ticker, "event": event, "series": series, "strike": str(strike),
            "side": "NO", "contracts": contracts, "entry": str(entry), "fee": str(fee),
            "opened_ts": ts, "order_id": order_id, "canary": canary, "open": True,
        }
        self.data.setdefault("positions", []).append(row)
        self.log("entry", **{k: row[k] for k in
                             ("ticker", "series", "contracts", "entry", "fee", "order_id")},
                 canary=canary)
        self.save()
        return row

    def settle_position(self, ticker: str, pnl: Decimal, result: str, ts: float) -> dict | None:
        for pos in self.data.get("positions") or []:
            if pos.get("ticker") != ticker or not pos.get("open", True):
                continue
            pos["open"] = False
            pos["pnl"] = str(pnl)
            pos["result"] = result
            pos["settled_ts"] = ts
            self.data["peak"] = str(max(self.peak(), self.equity()))
            self.log("settle", ticker=ticker, pnl=str(pnl), result=result,
                     equity=str(self.equity()))
            self.save()
            return pos
        return None

    def record_canary(self, payload: dict[str, Any]) -> None:
        self.data["canary"] = payload
        self.log("canary", **payload)
        self.save()


# --- venue operations ------------------------------------------------------
async def wait_fills(broker: Broker, order_id: str, *, timeout: float = 8.0,
                     expected: int | None = None) -> list[dict[str, Any]]:
    """The order's fills, once they have stopped arriving.

    An IOC fills in pieces, and reading the first print booked 1 of 59 contracts on
    this account before. A whole order returns at once; a partial one returns after
    its total has been stable for `FILL_SETTLE_S` (or at the deadline).
    """
    deadline = time.monotonic() + timeout
    seen = Decimal("0")
    stable_since: float | None = None
    fills: list[dict[str, Any]] = []
    while True:
        fills = await broker.fills(order_id=order_id, limit=50)
        total = sum((_dec(f.get("count_fp")) for f in fills), Decimal("0"))
        if expected is not None and total >= Decimal(expected) - Decimal("0.5"):
            return fills
        now = time.monotonic()
        if total > seen:
            seen, stable_since = total, now
        elif total > 0 and stable_since is not None and now - stable_since >= 0.8:
            return fills
        if now >= deadline:
            return fills
        await asyncio.sleep(0.3)


def summarize_fills(fills: Sequence[dict[str, Any]], side: str = "NO"
                    ) -> tuple[int, Decimal, Decimal]:
    """(whole contracts, volume-weighted price, fees) from the venue's own tape."""
    exact = sum((_dec(f.get("count_fp")) for f in fills), Decimal("0"))
    if exact <= 0:
        return 0, Decimal("0"), Decimal("0")
    key = "no_price_dollars" if side == "NO" else "yes_price_dollars"
    notional = sum((_dec(f.get("count_fp")) * _dec(f.get(key)) for f in fills),
                   Decimal("0"))
    fee = sum((_dec(f.get("fee_cost")) for f in fills), Decimal("0"))
    qty = int(exact.to_integral_value(rounding=ROUND_DOWN))
    return qty, (notional / exact).quantize(Decimal("0.0001")), fee


async def settle_positions(client: KalshiClient, ledger: LaneLedger,
                           now: float | None = None) -> list[dict[str, Any]]:
    """Book every open position whose market has settled.  No orders are sent."""
    now = time.time() if now is None else now
    booked: list[dict[str, Any]] = []
    for pos in list(ledger.open_positions()):
        status, payload = await client.get(f"/markets/{pos['ticker']}")
        if status != 200:
            continue
        row = payload.get("market") or payload
        result = (row.get("result") or "").strip().lower()
        value = row.get("expiration_value")
        if not result and value in (None, ""):
            continue
        won = resolve_no(result=result, value=value,
                         strike=_dec(pos.get("strike")))
        if won is None:
            continue
        pnl = settlement_pnl(_dec(pos.get("entry")), int(pos.get("contracts") or 0),
                             _dec(pos.get("fee")), won=won)
        settled = ledger.settle_position(pos["ticker"], pnl, result or "value", now)
        if settled is not None:
            booked.append(settled)
    return booked


def market_status(market: Market, now: float) -> str:
    if now < market.decision_ts:
        return "scheduled"
    if now > market.decision_ts + DECISION_GRACE_S:
        return "missed"
    return "due"


# --- dry run ---------------------------------------------------------------
async def dry_run(settings: Any, ledger: LaneLedger, *, at: float | None = None,
                  client: KalshiClient | None = None,
                  series_list: Sequence[str] = SERIES) -> dict[str, Any]:
    """Paper only: read the live book and print what the rule would place."""
    now = time.time() if at is None else float(at)
    own = client is None
    client = client or KalshiClient(settings)
    try:
        markets = await fetch_open_markets(client, series_list)
        changes = await fetch_changes(client, series_list)
    finally:
        if own:
            await client.close()

    references = {series: change.latest for series, change in changes.items()
                  if change is not None}
    lane_reasons = lane_pause_reasons(ledger.data, now)
    series_reasons = {series: reason for series in series_list
                      if (reason := series_pause_reason(changes.get(series), series))}

    live_markets = [m for m in markets if market_status(m, now) != "missed"]
    candidates = [c for c in candidates_for(live_markets, references)
                  if c.market.series not in series_reasons]
    orders = [] if lane_reasons else plan_orders(candidates, ledger.open_positions())

    events: dict[str, dict[str, Any]] = {}
    for market in live_markets:
        event = events.setdefault(market.event, {
            "event": market.event,
            "series": market.series,
            "decision_ts": market.decision_ts,
            "decision_at": _fmt_ts(market.decision_ts),
            "status": market_status(market, now),
            "series_paused": market.series in series_reasons,
            "markets": [],
        })
        candidate = candidate_for(market, references.get(market.series))
        event["markets"].append({
            "ticker": market.ticker,
            "strike": str(market.strike),
            "no_price": str(market.no_price),
            "depth": str(market.yes_bid_size),
            "eligible": candidate is not None,
            "why": "" if candidate is not None else ineligible_reason(market),
        })

    for event in events.values():
        event["markets"].sort(key=lambda row: _dec(row["no_price"]), reverse=True)
        event["markets"] = event["markets"][:3]

    return {
        "lane": LANE,
        "mode": "paper",
        "as_of": now,
        "as_of_local": _fmt_ts(now),
        "equity": str(ledger.equity()),
        "equity_start": str(ledger.equity_start),
        "peak": str(ledger.peak()),
        "realized": str(ledger.realized()),
        "pause": ledger.pause() or ({"reason": "; ".join(lane_reasons)} if lane_reasons else None),
        "series": {
            series: {
                "change": None if changes.get(series) is None
                else str(changes[series].change),
                "change_cents": None if changes.get(series) is None
                else f"{changes[series].change * 100:+.2f}",
                "asof": None if changes.get(series) is None else changes[series].asof,
                "threshold": str(pause_threshold(series)),
                "paused": series in series_reasons,
                "reason": series_reasons.get(series, ""),
            }
            for series in series_list
        },
        "events": sorted(events.values(), key=lambda e: (e["decision_ts"], e["event"])),
        "orders": [order.to_dict() for order in orders],
        "canary": ledger.data.get("canary"),
        "open_positions": ledger.open_positions(),
    }


# --- live: canary and full size -------------------------------------------
def _connected(settings: Any) -> tuple[KalshiClient, Broker]:
    client = KalshiClient(settings)
    return client, Broker(client)


async def _prepare_live(client: KalshiClient, ledger: LaneLedger, now: float,
                        series_list: Sequence[str]
                        ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Settle, then evaluate pauses.  Returns (blocked, context)."""
    booked = await settle_positions(client, ledger, now)
    lane_reasons = lane_pause_reasons(ledger.data, now)
    if lane_reasons:
        pause = ledger.latch_pause("; ".join(lane_reasons), now)
        return {"error": "lane paused", "pause": pause, "settled": booked}, {}

    changes = await fetch_changes(client, series_list)
    series_reasons: dict[str, str] = {}
    for series in series_list:
        reason = series_pause_reason(changes.get(series), series)
        if reason:
            series_reasons[series] = reason
            ledger.pause_series(series, reason, now)

    markets = await fetch_open_markets(client, series_list)
    references = {series: change.latest for series, change in changes.items()
                  if change is not None}
    due = [m for m in markets
           if m.series not in series_reasons
           and actionable(m, now)
           and not ledger.evaluated(m.event)]
    candidates = candidates_for(due, references)
    return None, {"settled": booked, "changes": changes,
                  "series_pauses": series_reasons, "due": due,
                  "candidates": candidates}


async def canary(settings: Any, ledger: LaneLedger, *, live: bool,
                 at: float | None = None, client: KalshiClient | None = None,
                 broker: Broker | None = None,
                 series_list: Sequence[str] = SERIES) -> dict[str, Any]:
    """One contract on one signal: place, fill, reconcile, verify on the venue."""
    if not live:
        raise ValueError("the canary places a real order and requires --live")
    now = time.time() if at is None else float(at)
    own = client is None
    client = client or KalshiClient(settings)
    broker = broker or Broker(client)
    try:
        if ledger.canary_ok():
            return {"ok": False, "error": "canary already passed; use execute"}
        blocked, context = await _prepare_live(client, ledger, now, series_list)
        if blocked:
            return blocked
        orders = plan_orders(context["candidates"], ledger.open_positions(),
                             size=CANARY_SIZE)
        if not orders:
            return {"ok": False, "error": "no eligible signal right now",
                    "settled": context["settled"],
                    "series_pauses": context["series_pauses"]}

        order = orders[0]
        balance_before = await broker.balance()
        venue = await broker.place_no_entry(order.ticker, order.price, CANARY_SIZE)
        if venue is None:
            return {"ok": False, "error": "venue rejected the order",
                    "detail": broker.errors[-1] if broker.errors else "unknown"}
        fills = await wait_fills(broker, venue.order_id,
                                 expected=venue.filled or CANARY_SIZE)
        if not fills:
            return {"ok": False, "error": "IOC unfilled; canary not established",
                    "ticker": order.ticker, "order_id": venue.order_id,
                    "requested": order.price}
        qty, price, fee = summarize_fills(fills, side="NO")
        position = await broker.position(order.ticker)
        balance_after = await broker.balance()
        model_fee = taker_fee(price, qty)
        expected_move = -(price * qty + fee) if qty else Decimal("0")
        balance_move = balance_after - balance_before

        checks = {
            "filled_one": qty == CANARY_SIZE,
            "at_or_below_limit": price <= order.price,
            "fee_matches_model": abs(fee - model_fee) <= Decimal("0.01"),
            "venue_position": position == -CANARY_SIZE,
        }
        ok = all(checks.values())
        payload = {
            "ok": ok,
            "ts": now,
            "at": _fmt_ts(now),
            "ticker": order.ticker,
            "event": order.event,
            "price": str(price),
            "limit": str(order.price),
            "contracts": qty,
            "fee": str(fee),
            "model_fee": str(model_fee),
            "order_id": venue.order_id,
            "venue_position": position,
            "checks": checks,
            # Informational: the crypto lanes share this account and may move cash
            # at the same moment, so the fills tape and the position are the
            # authoritative reconciliation, not the balance delta.
            "balance_before": str(balance_before),
            "balance_after": str(balance_after),
            "balance_move": str(balance_move),
            "expected_move": str(expected_move),
            "balance_delta": str(abs(balance_move - expected_move)),
        }
        ledger.record_canary(payload)
        if qty > 0:
            ledger.record_position(
                ticker=order.ticker, event=order.event, series=order.series,
                strike=order.strike, contracts=qty, entry=price, fee=fee,
                ts=now, order_id=venue.order_id, canary=True)
        return payload
    finally:
        if own:
            await client.close()


async def execute(settings: Any, ledger: LaneLedger, *, live: bool,
                  at: float | None = None, client: KalshiClient | None = None,
                  broker: Broker | None = None,
                  series_list: Sequence[str] = SERIES) -> dict[str, Any]:
    """Place the day's full-size signals (must follow a passing canary)."""
    if not live:
        raise ValueError("execute places real orders and requires --live")
    now = time.time() if at is None else float(at)
    own = client is None
    client = client or KalshiClient(settings)
    broker = broker or Broker(client)
    try:
        if not ledger.canary_ok():
            return {"ok": False,
                    "error": "canary has not passed; run `canary --live` first"}
        blocked, context = await _prepare_live(client, ledger, now, series_list)
        if blocked:
            return {"ok": False, **blocked}

        orders = plan_orders(context["candidates"], ledger.open_positions())
        results: list[dict[str, Any]] = []
        for order in orders:
            venue = await broker.place_no_entry(order.ticker, order.price,
                                                order.contracts)
            if venue is None:
                results.append({"order": order.to_dict(), "ok": False,
                                "error": broker.errors[-1] if broker.errors else "rejected"})
                ledger.log("reject", ticker=order.ticker,
                           error=broker.errors[-1] if broker.errors else "rejected")
                continue
            fills = await wait_fills(broker, venue.order_id,
                                     expected=venue.filled or order.contracts)
            if not fills:
                results.append({"order": order.to_dict(), "ok": False,
                                "order_id": venue.order_id, "error": "IOC unfilled"})
                ledger.log("unfilled", ticker=order.ticker,
                           order_id=venue.order_id)
                continue
            qty, price, fee = summarize_fills(fills, side="NO")
            if qty > 0:
                ledger.record_position(
                    ticker=order.ticker, event=order.event, series=order.series,
                    strike=order.strike, contracts=qty, entry=price, fee=fee,
                    ts=now, order_id=venue.order_id)
            results.append({"order": order.to_dict(), "ok": qty > 0,
                            "order_id": venue.order_id, "filled": qty,
                            "fill_price": str(price), "fee": str(fee)})

        for event in sorted({m.event for m in context["due"]}):
            ledger.mark_evaluated(event, now)
        return {"ok": True, "placed": len([r for r in results if r.get("filled")]),
                "settled": context["settled"],
                "series_pauses": context["series_pauses"],
                "orders": results}
    finally:
        if own:
            await client.close()


# --- reporting -------------------------------------------------------------
def format_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"AAA gas/diesel far-strike NO -- lane {report['lane']} "
                 f"({report['mode']} run)")
    lines.append(f"as of {report['as_of_local']}")
    lines.append(f"lane equity {report['equity']} "
                 f"(start {report['equity_start']}, peak {report['peak']}, "
                 f"realized {report['realized']})")
    pause = report.get("pause")
    lines.append(f"lane pause: {pause['reason'] if pause else 'none'}")
    lines.append("series:")
    for series, row in report["series"].items():
        change = ("unavailable" if row["change"] is None
                  else f"{row['change_cents']}c ({row['asof']})")
        state = "PAUSED" if row["paused"] else "active"
        lines.append(f"  {series:10s} change {change:22s} "
                     f"threshold {_dec(row['threshold']) * 100:.0f}c  {state}"
                     + (f"  [{row['reason']}]" if row["reason"] else ""))
    for event in report["events"]:
        lines.append(f"event {event['event']} ({event['series']}) "
                     f"decision {event['decision_at']} [{event['status']}]"
                     + (" series paused" if event["series_paused"] else ""))
        for row in event["markets"]:
            suffix = "" if row["eligible"] else f"  -- {row['why']}"
            lines.append(f"    {row['ticker']}  NO {_dec(row['no_price']):.2f}  "
                         f"depth {_dec(row['depth']):g}{suffix}")
    orders = report["orders"]
    lines.append(f"orders ({len(orders)}):")
    for order in orders:
        lines.append(f"  BUY NO {order['contracts']}x {order['ticker']} "
                     f"@ {_dec(order['price']):.2f} limit {order['time_in_force']}, "
                     f"seen depth {_dec(order['seen_depth']):g}; "
                     f"premium {order['premium']}, fee {order['fee']}")
    if not orders:
        lines.append("  none")
    canary = report.get("canary")
    lines.append("canary: " + ("passed" if canary and canary.get("ok")
                               else "not passed" if canary else "not run"))
    open_positions = report.get("open_positions") or []
    lines.append(f"open positions: {len(open_positions)}")
    for pos in open_positions:
        lines.append(f"  {pos['ticker']} NO {pos['contracts']} @ {pos['entry']}")
    lines.append("paper dry run: no order was sent and no arm record was read.")
    return "\n".join(lines)


def status_text(ledger: LaneLedger) -> str:
    lines = [
        f"lane {LANE}  equity {ledger.equity()}  peak {ledger.peak()}  "
        f"realized {ledger.realized()}",
        f"canary: {'passed' if ledger.canary_ok() else 'not passed'}",
    ]
    pause = ledger.pause()
    lines.append(f"lane pause: {pause['reason'] if pause else 'none'}")
    for series, row in (ledger.data.get("series_pause") or {}).items():
        lines.append(f"series pause {series}: {row.get('reason')} ({row.get('day')})")
    for pos in ledger.open_positions():
        lines.append(f"open {pos['ticker']} NO {pos['contracts']} @ {pos['entry']}")
    record = arming.read(arming.arm_path(LANE))
    if record:
        lines.append(f"arm record: {record.get('by')} at "
                     f"{_fmt_ts(float(record.get('ts') or 0.0))} "
                     f"for lane {record.get('lane') or 'crypto'}")
    else:
        lines.append(f"arm record: none at {arming.arm_path(LANE).relative_to(ROOT)}")
    return "\n".join(lines)


def _parse_at(value: str | None) -> float | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ET)
    return parsed.timestamp()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="openkal.lanes.gas_no",
        description="AAA gas/diesel far-strike NO lane (separate from the crypto engine)")
    parser.add_argument("command",
                        choices=("dry", "status", "canary", "execute", "settle", "resume"))
    parser.add_argument("--live", action="store_true",
                        help="place real orders (requires the gas-no operator arm record)")
    parser.add_argument("--at", default=None,
                        help="dry run only: evaluate as of an ISO timestamp")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--by", default=os.environ.get("OPENKAL_OPERATOR")
                        or getpass.getuser(),
                        help="the operator's name, recorded on resume")
    parser.add_argument("--note", default="")
    args = parser.parse_args(argv)
    if args.at and args.command != "dry":
        print("refused: --at is a dry-run simulation and cannot be used to place orders",
              file=sys.stderr)
        return 2

    settings = load_settings()
    ledger = LaneLedger()

    if args.command == "dry":
        report = asyncio.run(dry_run(settings, ledger, at=_parse_at(args.at)))
        print(json.dumps(report, indent=2, default=str) if args.json
              else format_report(report))
        return 0

    if args.command == "status":
        print(status_text(ledger))
        return 0

    if args.command == "resume":
        try:
            had = ledger.resume(args.by, time.time())
        except ValueError as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 2
        print("pause cleared" if had else "no pause to clear")
        return 0

    if args.command == "settle":
        async def _settle() -> list[dict[str, Any]]:
            client = KalshiClient(settings)
            try:
                return await settle_positions(client, ledger)
            finally:
                await client.close()
        booked = asyncio.run(_settle())
        if not booked:
            print("no open position has settled")
            return 0
        for row in booked:
            print(f"settled {row['ticker']} NO {row['contracts']} -> "
                  f"{row['result']} pnl {row['pnl']}")
        print(f"lane equity {ledger.equity()}  peak {ledger.peak()}")
        return 0

    # ---- live commands: the lane's own operator record, nothing else ----
    if not args.live:
        print(f"refused: `{args.command}` places real orders and requires --live",
              file=sys.stderr)
        return 2
    if not settings.live:
        print("refused: live execution is disabled (needs ALLOW_LIVE=1, "
              "KALSHI_ENV=prod and no OPENKAL_FORCE_PAPER)", file=sys.stderr)
        return 2
    record = arming.consume(arming.arm_path(LANE))
    if not arming.fresh_operator_act(record, now=time.time(), lane=LANE):
        print(f"refused: no fresh {LANE} operator record at "
              f"{arming.arm_path(LANE).relative_to(ROOT)}; run "
              f"`scripts/arm_live.py --lane {LANE} --by \"your name\"` first",
              file=sys.stderr)
        return 2

    if args.command == "canary":
        result = asyncio.run(canary(settings, ledger, live=True,
                                    at=_parse_at(args.at)))
    else:
        result = asyncio.run(execute(settings, ledger, live=True,
                                     at=_parse_at(args.at)))
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
