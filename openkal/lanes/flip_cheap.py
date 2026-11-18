"""Paper lane: buy the cheap side after a 95c favourite flips in the last three minutes.

Frozen rule for this lane:

  * universe: the nine coin 15-minute series (BTC, ETH, SOL, XRP, DOGE, BNB,
    HYPE, NEAR, ZEC), one live market per series;
  * a side whose **mid** was at least 95c at some point in the last 60 seconds is
    now at least 20c below that high;
  * the decision is in the final 180 seconds of the 15-minute window;
  * contracts traded in the current market over the last 60 seconds are at or
    above that coin's 75th percentile of completed 60-second buckets;
  * buy **the opposite side** -- the one that rose from <=5c to roughly 20-30c
    -- with a 10 contract IOC at the live ask, all-or-nothing on displayed
    depth; the entry must stay under 50c;
  * hold to settlement.  P&L is settlement payout minus premium and the Kalshi
    taker fee; there is no exit.

The cheap side is the side that **rose**: when a 95c favourite collapses, the
other side is repriced from a few cents to 20-30c and that is what this lane
buys.  Buying the fallen favourite was the earlier reading and it lost money:
that leg averaged a 25.3c entry and won only 30%.  `side` is therefore the
traded (rose) side, `fell_side` is the favourite that dropped, and both sides'
mids are stored on every signal.  A candidate is refused unless the traded
side's mid is strictly below 50c, and `entry_reason` refuses any ask at or
above 50c, so a late re-quote cannot turn the cheap side into a favourite.

Volume percentile: the lane records every completed 60-second bucket per coin
(including zero-volume buckets) while it is running.  At a signal it compares
the live 60-second volume with the nearest-rank 75th percentile of completed
buckets from the trailing seven days, excluding the current bucket.  At least 30
completed buckets are required, so a fresh database needs 30 minutes of warm-up
before it can signal.

Paper fills are deliberately conservative: the displayed size at the best ask
must be at least 10 contracts.  If it is not, the signal is recorded as a
no-fill.  There is no partial fill and no retry in the same window.

The daily report is UTC and prints every coin separately, with ETH and ZEC also
called out in a focus block.  The gate is:

    150 paper fills, EV/contract > +1c, and the 95% lower confidence bound > 0

`canary --live` is the only live command.  It refuses unless the gate passes, a
fresh lane-specific arm record exists, and ALLOW_LIVE/prod is set.  It buys one
contract, holds to settlement, and consumes the arm.  No other command places an
order.

Commands:

    .venv/bin/python -m openkal.lanes.flip_cheap run --seconds 3600
    .venv/bin/python -m openkal.lanes.flip_cheap report --date 2026-11-18
    .venv/bin/python -m openkal.lanes.flip_cheap gate
    .venv/bin/python -m openkal.lanes.flip_cheap status

    # only after the gate and a fresh operator arm:
    .venv/bin/python scripts/arm_live.py --lane flip-cheap --by "NAME"
    .venv/bin/python -m openkal.lanes.flip_cheap canary --live
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sqlite3
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .. import arming
from ..auth import KalshiClient
from ..broker import Broker
from ..config import ROOT, Settings, load_settings
from ..fees import taker_fee
from ..kalshi_ws import Book, KalshiWS

LANE = "flip-cheap"
TICK = Decimal("0.01")
FAV_MIN = Decimal("0.95")
DROP_MIN = Decimal("0.20")
# The cheap side is only cheap below a half dollar.  Backtest entries averaged
# 25.3c; anything at or above 50c is the other trade (the fallen favourite).
MAX_ENTRY = Decimal("0.50")
SIGNAL_WINDOW_S = 60.0
LAST_MINUTES_S = 180.0
CONTRACTS = 10
CANARY_CONTRACTS = 1
MIN_VOLUME_BUCKETS = 30
TRAILING_VOLUME_DAYS = 7
P75 = 0.75
GATE_FILLS = 150
GATE_EV = Decimal("0.01")
CI_Z = 1.96
EVAL_S = 0.5
REFRESH_S = 5.0
SETTLE_S = 5.0
STATUS_S = 60.0
MAX_MID_AGE_S = 5.0
DEFAULT_DB = ROOT / "data" / "flip-cheap.db"


@dataclass(frozen=True)
class Coin:
    series: str
    asset: str
    display: str


# The nine coins on Kalshi's 15-minute crypto page.  NEAR and ZEC are included
# even though they are not crypto-engine lanes; this is a separate paper lane.
COINS: tuple[Coin, ...] = (
    Coin("KXBTC15M", "BTC", "Bitcoin"),
    Coin("KXETH15M", "ETH", "Ethereum"),
    Coin("KXSOL15M", "SOL", "Solana"),
    Coin("KXXRP15M", "XRP", "XRP"),
    Coin("KXDOGE15M", "DOGE", "Dogecoin"),
    Coin("KXBNB15M", "BNB", "BNB"),
    Coin("KXHYPE15M", "HYPE", "Hyperliquid"),
    Coin("KXNEAR15M", "NEAR", "NEAR"),
    Coin("KXZEC15M", "ZEC", "Zcash"),
)
COIN_BY_SERIES = {coin.series: coin for coin in COINS}
ASSETS = tuple(coin.asset for coin in COINS)


def dec(value: Any, default: str = "0") -> Decimal:
    if value in (None, ""):
        return Decimal(default)
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal(default)


def dstr(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


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


def day_bounds(day: date) -> tuple[float, float]:
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp()
    return start, start + 86400.0


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile, with no interpolation."""
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    index = max(0, math.ceil(q * len(ordered)) - 1)
    return ordered[min(index, len(ordered) - 1)]


def mean_se(values: list[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    mean = sum(values) / len(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, se, mean - CI_Z * se


def mid_yes(book: Book | None) -> Decimal | None:
    """Top-of-book YES mid.  A one-sided book has no trustworthy mid."""
    if book is None or not book.synced:
        return None
    bid, ask = book.yes_bid, book.yes_ask
    if bid is None or ask is None:
        return None
    return (bid + ask) / Decimal(2)


def side_mid(yes_mid: Decimal, side: str) -> Decimal:
    return yes_mid if side == "YES" else Decimal(1) - yes_mid


def side_ask(book: Book | None, side: str) -> tuple[Decimal | None, Decimal | None]:
    """Best executable ask in the side's own terms plus displayed size there."""
    if book is None:
        return None, None
    if side == "YES":
        return book.yes_ask, book.yes_ask_sz
    return book.no_ask, book.yes_bid_sz


def entry_reason(ask: Decimal | None, depth: Decimal | None,
                 contracts: int = CONTRACTS) -> str:
    """Why an IOC at the live ask is not a clean all-or-nothing fill."""
    if ask is None:
        return "no_ask"
    if not (Decimal("0") < ask < Decimal("1")):
        return "ask_out_of_range"
    if ask >= MAX_ENTRY:
        return "ask_at_or_above_50"
    if depth is None:
        return "no_depth"
    if depth < Decimal(contracts):
        return f"depth_below_{contracts}"
    return ""


def settlement_pnl(side: str, result: str, contracts: int,
                   entry: Decimal, fee: Decimal) -> tuple[Decimal, Decimal, Decimal]:
    """Payout, dollar P&L and P&L per contract for a hold-to-settlement binary."""
    if contracts <= 0:
        return Decimal("0"), Decimal("0"), Decimal("0")
    won = ((side == "YES" and result == "yes")
           or (side == "NO" and result == "no"))
    payout = Decimal(contracts) if won else Decimal("0")
    pnl = payout - Decimal(entry) * Decimal(contracts) - Decimal(fee)
    return payout, pnl, pnl / Decimal(contracts)


@dataclass(frozen=True)
class DropCandidate:
    side: str
    fell_side: str
    high_ts: float
    high_mid: Decimal
    current_mid: Decimal
    drop: Decimal


@dataclass
class MarketWindow:
    coin: Coin
    ticker: str
    close_ts: float
    strike: float
    mid_hist: deque[tuple[float, Decimal]] = field(default_factory=deque)
    trades: deque[tuple[float, float]] = field(default_factory=deque)
    last_mid_ts: float = 0.0
    book: Book | None = None
    signaled: bool = False
    last_reject: str = ""

    def observe(self, book: Book, now: float) -> None:
        mid = mid_yes(book)
        if mid is None:
            return
        self.book = book
        self.last_mid_ts = now
        # Record every distinct mid, but do not append the same unchanged value
        # more than four times a second.
        if (not self.mid_hist or self.mid_hist[-1][1] != mid
                or now - self.mid_hist[-1][0] >= 0.25):
            self.mid_hist.append((now, mid))
        while self.mid_hist and self.mid_hist[0][0] < now - SIGNAL_WINDOW_S - 5.0:
            self.mid_hist.popleft()

    def add_trade(self, now: float, contracts: float) -> None:
        if contracts <= 0:
            return
        self.trades.append((now, float(contracts)))
        while self.trades and self.trades[0][0] < now - 120.0:
            self.trades.popleft()

    def rolling_volume(self, now: float) -> float:
        while self.trades and self.trades[0][0] < now - SIGNAL_WINDOW_S:
            self.trades.popleft()
        return sum(count for _, count in self.trades)

    def drop_candidate(self, now: float) -> DropCandidate | None:
        if not (0.0 < self.close_ts - now <= LAST_MINUTES_S):
            return None
        if not self.mid_hist or now - self.last_mid_ts > MAX_MID_AGE_S:
            return None
        latest = self.mid_hist[-1][1]
        best: DropCandidate | None = None
        for fell_side in ("YES", "NO"):
            values = [(ts, side_mid(mid, fell_side)) for ts, mid in self.mid_hist
                      if ts >= now - SIGNAL_WINDOW_S]
            if not values:
                continue
            high_ts, high = max(values, key=lambda row: (row[1], row[0]))
            current = side_mid(latest, fell_side)
            drop = high - current
            if high < FAV_MIN or drop < DROP_MIN:
                continue
            # `current` is the fallen favourite's mid, so the side that rose is
            # priced at 1 - current.  Refuse the trade if that is not below 50c.
            if current <= Decimal(1) - MAX_ENTRY:
                continue
            side = "NO" if fell_side == "YES" else "YES"
            candidate = DropCandidate(side, fell_side, high_ts, high, current, drop)
            if best is None or candidate.drop > best.drop:
                best = candidate
        return best


@dataclass
class Signal:
    ts: float
    coin: Coin
    ticker: str
    close_ts: float
    side: str
    fell_side: str
    high_ts: float
    high_mid: Decimal
    current_mid: Decimal
    drop: Decimal
    volume_60: float
    p75: float
    yes_mid: Decimal | None
    no_mid: Decimal | None
    ask: Decimal | None
    depth: Decimal | None
    filled: bool
    contracts: int
    fee: Decimal
    fee_source: str
    no_fill_reason: str
    mode: str
    order_id: str = ""


SCHEMA = """
CREATE TABLE IF NOT EXISTS windows (
    ticker TEXT PRIMARY KEY, series TEXT NOT NULL, asset TEXT NOT NULL,
    close_ts REAL NOT NULL, strike TEXT, result TEXT, status TEXT,
    settled_ts REAL, updated_ts REAL
);

CREATE TABLE IF NOT EXISTS volume_buckets (
    series TEXT NOT NULL, bucket_ts REAL NOT NULL, volume REAL NOT NULL,
    PRIMARY KEY (series, bucket_ts)
);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, series TEXT NOT NULL, asset TEXT NOT NULL, ticker TEXT NOT NULL,
    close_ts REAL NOT NULL, side TEXT NOT NULL, fell_side TEXT,
    high_mid TEXT NOT NULL, high_ts REAL NOT NULL, current_mid TEXT NOT NULL,
    drop_mid TEXT NOT NULL, volume_60 REAL NOT NULL, p75 REAL NOT NULL,
    yes_mid TEXT, no_mid TEXT, ask TEXT, depth TEXT,
    filled INTEGER NOT NULL DEFAULT 0, contracts INTEGER NOT NULL DEFAULT 0,
    fee TEXT, fee_source TEXT, order_id TEXT,
    result TEXT, payout TEXT, pnl TEXT, pnl_per_contract TEXT,
    settled_ts REAL, no_fill_reason TEXT, mode TEXT NOT NULL DEFAULT 'paper'
);
CREATE INDEX IF NOT EXISTS signals_ts ON signals(ts);
CREATE INDEX IF NOT EXISTS signals_asset_ts ON signals(asset, ts);

CREATE TABLE IF NOT EXISTS rejections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, series TEXT NOT NULL, asset TEXT NOT NULL, ticker TEXT NOT NULL,
    side TEXT NOT NULL, high_mid TEXT, current_mid TEXT, drop_mid TEXT,
    volume_60 REAL, p75 REAL, reason TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mode TEXT NOT NULL, started_ts REAL NOT NULL, ended_ts REAL,
    signals INTEGER, fills INTEGER, contracts INTEGER, pnl TEXT, note TEXT
);
"""


class FlipStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(str(path), check_same_thread=False, timeout=10.0)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        # The first builds stored only the traded side.  Add the fallen side so
        # an existing paper database keeps its history and gains the new column.
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(signals)")}
        if "fell_side" not in columns:
            self.db.execute("ALTER TABLE signals ADD COLUMN fell_side TEXT")
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def record_window(self, window: MarketWindow) -> None:
        self.db.execute(
            "INSERT INTO windows(ticker,series,asset,close_ts,strike,status,updated_ts) "
            "VALUES(?,?,?,?,?,?,?) ON CONFLICT(ticker) DO UPDATE SET "
            "close_ts=excluded.close_ts, strike=excluded.strike, updated_ts=excluded.updated_ts",
            (window.ticker, window.coin.series, window.coin.asset, window.close_ts,
             str(window.strike), "open", time.time()),
        )
        self.db.commit()

    def record_bucket(self, series: str, bucket_ts: float, volume: float) -> None:
        self.db.execute(
            "INSERT INTO volume_buckets(series,bucket_ts,volume) VALUES(?,?,?) "
            "ON CONFLICT(series,bucket_ts) DO UPDATE SET volume=volume+excluded.volume",
            (series, float(bucket_ts), float(volume)),
        )
        self.db.commit()

    def p75(self, series: str, now: float) -> float | None:
        bucket = math.floor(now / 60.0) * 60.0
        since = bucket - TRAILING_VOLUME_DAYS * 86400.0
        rows = self.db.execute(
            "SELECT volume FROM volume_buckets WHERE series=? AND bucket_ts>=? "
            "AND bucket_ts<?", (series, since, bucket),
        ).fetchall()
        if len(rows) < MIN_VOLUME_BUCKETS:
            return None
        return percentile([float(row[0]) for row in rows], P75)

    def log_rejection(self, now: float, window: MarketWindow, candidate: DropCandidate,
                      volume: float, p75: float | None, reason: str) -> None:
        self.db.execute(
            "INSERT INTO rejections(ts,series,asset,ticker,side,high_mid,current_mid,"
            "drop_mid,volume_60,p75,reason) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (now, window.coin.series, window.coin.asset, window.ticker, candidate.side,
             str(candidate.high_mid), str(candidate.current_mid), str(candidate.drop),
             float(volume), None if p75 is None else float(p75), reason),
        )
        self.db.commit()

    def log_signal(self, signal: Signal) -> int:
        cursor = self.db.execute(
            "INSERT INTO signals(ts,series,asset,ticker,close_ts,side,fell_side,high_mid,high_ts,"
            "current_mid,drop_mid,volume_60,p75,yes_mid,no_mid,ask,depth,filled,contracts,"
            "fee,fee_source,order_id,no_fill_reason,mode) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (signal.ts, signal.coin.series, signal.coin.asset, signal.ticker,
             signal.close_ts, signal.side, signal.fell_side,
             str(signal.high_mid), signal.high_ts,
             str(signal.current_mid), str(signal.drop), float(signal.volume_60),
             float(signal.p75), dstr(signal.yes_mid), dstr(signal.no_mid),
             dstr(signal.ask), dstr(signal.depth), 1 if signal.filled else 0,
             int(signal.contracts), str(signal.fee), signal.fee_source,
             signal.order_id, signal.no_fill_reason, signal.mode),
        )
        self.db.commit()
        return int(cursor.lastrowid)

    def set_fill(self, signal_id: int, filled: bool, contracts: int, fee: Decimal,
                 ask: Decimal | None, depth: Decimal | None, reason: str, *,
                 order_id: str = "", fee_source: str = "") -> None:
        self.db.execute(
            "UPDATE signals SET filled=?, contracts=?, fee=?, fee_source=?, order_id=?, "
            "no_fill_reason=?, ask=COALESCE(?,ask), depth=COALESCE(?,depth) WHERE id=?",
            (1 if filled else 0, int(contracts), str(fee), fee_source, order_id, reason,
             dstr(ask), dstr(depth), signal_id),
        )
        self.db.commit()

    def settle(self, signal_id: int, result: str, payout: Decimal, pnl: Decimal,
               per_contract: Decimal, now: float) -> None:
        self.db.execute(
            "UPDATE signals SET result=?, payout=?, pnl=?, pnl_per_contract=?, "
            "settled_ts=? WHERE id=?",
            (result, str(payout), str(pnl), str(per_contract), now, signal_id),
        )
        self.db.execute(
            "UPDATE windows SET result=?, status='settled', settled_ts=? "
            "WHERE ticker=(SELECT ticker FROM signals WHERE id=?)",
            (result, now, signal_id),
        )
        self.db.commit()

    def pending(self) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM signals WHERE filled=1 AND pnl_per_contract IS NULL "
            "ORDER BY close_ts, id",
        ).fetchall()

    def count(self, sql: str, args: tuple = ()) -> int:
        row = self.db.execute(sql, args).fetchone()
        return int(row[0] if row else 0)

    def counts(self) -> dict[str, Any]:
        last = self.db.execute("SELECT MAX(ts) FROM signals").fetchone()
        return {
            "signals": self.count("SELECT COUNT(*) FROM signals"),
            "fills": self.count("SELECT COUNT(*) FROM signals WHERE filled=1"),
            "pending": self.count(
                "SELECT COUNT(*) FROM signals WHERE filled=1 AND pnl_per_contract IS NULL"),
            "volume_buckets": self.count("SELECT COUNT(*) FROM volume_buckets"),
            "last_signal_ts": float(last[0]) if last and last[0] is not None else None,
            "db": str(self.path),
        }

    def start_run(self, mode: str) -> int:
        cursor = self.db.execute(
            "INSERT INTO runs(mode,started_ts) VALUES(?,?)", (mode, time.time()))
        self.db.commit()
        return int(cursor.lastrowid)

    def finish_run(self, run_id: int, counts: dict[str, Any], note: str = "") -> None:
        self.db.execute(
            "UPDATE runs SET ended_ts=?, signals=?, fills=?, contracts=?, note=? WHERE id=?",
            (time.time(), counts.get("signals", 0), counts.get("fills", 0),
             self.count("SELECT COALESCE(SUM(contracts),0) FROM signals"), note, run_id),
        )
        self.db.commit()

    @staticmethod
    def _stats(rows: list[sqlite3.Row]) -> dict[str, Any]:
        filled = [row for row in rows if int(row["filled"] or 0) == 1]
        settled = [row for row in filled if row["pnl_per_contract"] is not None]
        contracts = sum(int(row["contracts"] or 0) for row in filled)
        settled_contracts = sum(int(row["contracts"] or 0) for row in settled)
        pnls = [float(row["pnl"] or 0) for row in settled]
        per = [float(row["pnl_per_contract"] or 0) for row in settled]
        ev = (sum(pnls) / settled_contracts) if settled_contracts else None
        _, se, lower = mean_se(per)
        if ev is not None and se is not None:
            lower = ev - CI_Z * se
        wins = sum(1 for pnl in pnls if pnl > 0)
        asks = [float(row["ask"]) for row in filled
                if row["ask"] not in (None, "")]
        depths = [float(row["depth"]) for row in rows
                  if row["depth"] not in (None, "")]
        no_fills = len(rows) - len(filled)
        return {
            "signals": len(rows),
            "fills": len(filled),
            "settled_fills": len(settled),
            "contracts": contracts,
            "settled_contracts": settled_contracts,
            "pending": len(filled) - len(settled),
            "no_fills": no_fills,
            "no_fill_rate": (no_fills / len(rows)) if rows else None,
            "ev_per_contract": ev,
            "se": se,
            "lower_95": lower,
            "win_rate": (wins / len(settled)) if settled else None,
            "fees": sum(float(row["fee"] or 0) for row in filled),
            "avg_ask": (sum(asks) / len(asks)) if asks else None,
            "avg_depth": (sum(depths) / len(depths)) if depths else None,
        }

    def daily(self, day: date) -> dict[str, Any]:
        start, end = day_bounds(day)
        rows = self.db.execute(
            "SELECT * FROM signals WHERE ts>=? AND ts<? AND mode='paper' ORDER BY ts,id",
            (start, end),
        ).fetchall()
        by_coin: dict[str, dict[str, Any]] = {}
        for asset in ASSETS:
            by_coin[asset] = self._stats([row for row in rows if row["asset"] == asset])
        return {
            "date": day.isoformat(),
            "overall": self._stats(list(rows)),
            "by_coin": by_coin,
            "focus": {"ETH": by_coin["ETH"], "ZEC": by_coin["ZEC"]},
        }

    def gate(self) -> dict[str, Any]:
        rows = self.db.execute(
            "SELECT * FROM signals WHERE mode='paper' AND filled=1 "
            "AND pnl_per_contract IS NOT NULL ORDER BY ts,id",
        ).fetchall()
        contracts = sum(int(row["contracts"] or 0) for row in rows)
        pnls = [float(row["pnl"] or 0) for row in rows]
        per = [float(row["pnl_per_contract"] or 0) for row in rows]
        ev = (sum(pnls) / contracts) if contracts else None
        _, se, lower = mean_se(per)
        if ev is not None and se is not None:
            lower = ev - CI_Z * se
        pending = self.count(
            "SELECT COUNT(*) FROM signals WHERE mode='paper' AND filled=1 "
            "AND pnl_per_contract IS NULL")
        eligible = bool(
            len(rows) >= GATE_FILLS and ev is not None and Decimal(str(ev)) > GATE_EV
            and lower is not None and lower > 0.0)
        reasons: list[str] = []
        if len(rows) < GATE_FILLS:
            reasons.append(f"{len(rows)}/{GATE_FILLS} paper fills")
        if ev is None:
            reasons.append("no settled paper fills")
        elif Decimal(str(ev)) <= GATE_EV:
            reasons.append(f"EV/contract {ev * 100:+.2f}c is not above +1.00c")
        if lower is None:
            reasons.append("no lower confidence bound")
        elif lower <= 0.0:
            reasons.append(f"95% lower bound {lower * 100:+.2f}c is not above 0")
        return {
            "fills": len(rows),
            "contracts": contracts,
            "pending": pending,
            "ev_per_contract": ev,
            "se": se,
            "lower_95": lower,
            "eligible": eligible,
            "reasons": reasons,
        }


class VolumeTracker:
    """Completed 60-second contract-volume buckets per coin."""

    def __init__(self, store: FlipStore) -> None:
        self.store = store
        self.current: dict[str, tuple[float, float]] = {}

    def ensure(self, now: float) -> None:
        bucket = math.floor(now / 60.0) * 60.0
        for coin in COINS:
            self.current.setdefault(coin.series, (bucket, 0.0))

    def add(self, series: str, now: float, contracts: float) -> None:
        if contracts <= 0:
            return
        bucket = math.floor(now / 60.0) * 60.0
        current_bucket, volume = self.current.get(series, (bucket, 0.0))
        if bucket < current_bucket:
            return
        while bucket > current_bucket:
            self.store.record_bucket(series, current_bucket, volume)
            current_bucket += 60.0
            volume = 0.0
        self.current[series] = (current_bucket, volume + float(contracts))

    def flush(self, now: float) -> None:
        bucket = math.floor(now / 60.0) * 60.0
        for series, (current_bucket, volume) in list(self.current.items()):
            while current_bucket < bucket:
                self.store.record_bucket(series, current_bucket, volume)
                current_bucket += 60.0
                volume = 0.0
            self.current[series] = (current_bucket, volume)


class FlipRunner:
    def __init__(self, settings: Settings, store: FlipStore, *, mode: str = "paper",
                 seconds: float = 0.0) -> None:
        self.settings = settings
        self.store = store
        self.mode = mode
        self.seconds = seconds
        self.client = KalshiClient(settings)
        self.broker = Broker(self.client) if mode == "canary" else None
        self.windows: dict[str, MarketWindow] = {}
        self.active: dict[str, str] = {}
        self.volume = VolumeTracker(store)
        self.ws: KalshiWS | None = None
        self.ws_tickers: set[str] = set()
        self.pending_tickers: tuple[str, ...] | None = None
        self.stop = asyncio.Event()
        self.canary_done = False
        self.started = time.time()
        self.run_id: int | None = None
        self.last_ws_error = ""
        self.tasks: list[asyncio.Task] = []

    async def run(self) -> int:
        self.run_id = self.store.start_run(self.mode)
        self.volume.ensure(self.started)
        note = ""
        try:
            await self._refresh(force=True)
            self.tasks = [
                asyncio.create_task(self._refresh_loop(), name="flip-refresh"),
                asyncio.create_task(self._eval_loop(), name="flip-eval"),
                asyncio.create_task(self._settle_loop(), name="flip-settle"),
                asyncio.create_task(self._status_loop(), name="flip-status"),
            ]
            while not self.stop.is_set():
                await asyncio.sleep(1.0)
                if self.seconds > 0 and time.time() - self.started >= self.seconds:
                    note = "time limit reached"
                    break
                if (self.mode == "canary" and self.canary_done
                        and not self.store.pending()):
                    note = "canary trade settled" if self.store.count(
                        "SELECT COUNT(*) FROM signals WHERE mode='canary' AND filled=1"
                    ) else "canary no-fill"
                    break
            return 0
        finally:
            self.stop.set()
            for task in self.tasks:
                task.cancel()
            if self.tasks:
                await asyncio.gather(*self.tasks, return_exceptions=True)
            if self.ws is not None:
                try:
                    await self.ws.stop()
                except Exception:  # noqa: BLE001 - shutdown must not mask the run result
                    pass
            await self.client.close()
            counts = self.store.counts()
            if self.run_id is not None:
                self.store.finish_run(self.run_id, counts, note)
            self.store.close()

    async def _refresh_loop(self) -> None:
        while not self.stop.is_set():
            await asyncio.sleep(REFRESH_S)
            try:
                await self._refresh()
            except Exception as exc:  # noqa: BLE001 - keep the recorder alive
                print(f"refresh error: {type(exc).__name__}: {exc}", flush=True)

    async def _refresh(self, *, force: bool = False) -> None:
        now = time.time()
        found: dict[str, str] = {}
        for coin in COINS:
            status, payload = await self.client.get(
                "/markets", {"series_ticker": coin.series, "status": "open", "limit": 10})
            if status != 200:
                continue
            rows = []
            for market in payload.get("markets") or []:
                close_ts = parse_ts(market.get("close_time"))
                if (market.get("status") in {"active", "open", "initialized"}
                        and close_ts > now - 30.0):
                    rows.append((close_ts, market))
            if not rows:
                continue
            close_ts, market = max(rows, key=lambda row: row[0])
            ticker = str(market.get("ticker") or "")
            if not ticker:
                continue
            window = self.windows.get(ticker)
            if window is None:
                try:
                    strike = float(market.get("floor_strike") or 0.0)
                except (TypeError, ValueError):
                    strike = 0.0
                window = MarketWindow(coin, ticker, close_ts, strike)
                self.windows[ticker] = window
                self.store.record_window(window)
                print(f"OPEN {coin.asset} {ticker} close={datetime.fromtimestamp(close_ts, timezone.utc):%H:%M:%S}Z",
                      flush=True)
            found[coin.series] = ticker

        self.active = found
        # A closed window is no longer needed once its ticker has rolled: the
        # signal and settlement rows are in SQLite, not in this object.  Keeping
        # every 15-minute book for a multi-day run would grow memory without
        # bound.
        for ticker, window in list(self.windows.items()):
            if self.active.get(window.coin.series) == ticker:
                continue
            if now - window.close_ts > 600.0:
                self.windows.pop(ticker, None)
        tickers = tuple(sorted(found.values()))
        if not tickers:
            return
        if force or tickers == self.pending_tickers:
            if tickers != tuple(sorted(self.ws_tickers)):
                await self._restart_ws(tickers)
            self.pending_tickers = None
        else:
            self.pending_tickers = tickers

    async def _restart_ws(self, tickers: tuple[str, ...]) -> None:
        if self.ws is not None:
            try:
                await self.ws.stop()
            except Exception:  # noqa: BLE001
                pass
            self.ws = None
        ws = KalshiWS(self.settings, self.client, on_book=self._on_book,
                      on_event=self._on_event)
        ws.watch(list(tickers))
        await ws.start()
        self.ws = ws
        self.ws_tickers = set(tickers)
        for _ in range(100):
            if all(t in ws.books and ws.books[t].synced for t in tickers):
                break
            await asyncio.sleep(0.1)
        synced = sum(1 for t in tickers if t in ws.books and ws.books[t].synced)
        print(f"WS {synced}/{len(tickers)} books synced messages={ws.messages}", flush=True)
        if synced == 0:
            # Let the next refresh retry instead of sitting on a dead socket.
            self.ws_tickers = set()

    def _on_book(self, book: Book) -> None:
        window = self.windows.get(book.ticker)
        if window is not None and self.active.get(window.coin.series) == book.ticker:
            window.observe(book, time.time())

    def _on_event(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        body = message.get("msg") or {}
        if kind == "trade":
            ticker = str(body.get("market_ticker") or "")
            window = self.windows.get(ticker)
            if window is None or self.active.get(window.coin.series) != ticker:
                return
            count = dec(body.get("count_fp") or body.get("count"))
            if count <= 0:
                return
            now = time.time()
            window.add_trade(now, float(count))
            self.volume.add(window.coin.series, now, float(count))
        elif kind == "error":
            text = str(body.get("msg") or body.get("message") or "")
            if "Unknown command" not in text:
                self.last_ws_error = text[:180]

    async def _eval_loop(self) -> None:
        while not self.stop.is_set():
            await asyncio.sleep(EVAL_S)
            try:
                await self._evaluate()
            except Exception as exc:  # noqa: BLE001 - one bad book must not stop the lane
                print(f"eval error: {type(exc).__name__}: {exc}", flush=True)

    async def _evaluate(self) -> None:
        now = time.time()
        self.volume.flush(now)
        for series, ticker in list(self.active.items()):
            window = self.windows.get(ticker)
            if window is None or window.signaled or now >= window.close_ts:
                continue
            candidate = window.drop_candidate(now)
            if candidate is None:
                continue
            volume = window.rolling_volume(now)
            p75 = self.store.p75(series, now)
            if p75 is None:
                self._reject(window, candidate, volume, None, "warmup")
                continue
            if volume < p75:
                self._reject(window, candidate, volume, p75, "volume_below_p75")
                continue
            await self._emit(window, candidate, volume, p75, now)

    def _reject(self, window: MarketWindow, candidate: DropCandidate, volume: float,
                p75: float | None, reason: str) -> None:
        if window.last_reject == reason:
            return
        window.last_reject = reason
        self.store.log_rejection(time.time(), window, candidate, volume, p75, reason)
        shown = "warmup" if p75 is None else f"p75={p75:.1f}"
        print(f"REJECT {window.coin.asset} {candidate.side} drop={candidate.drop:.3f} "
              f"vol60={volume:.1f} {shown} {reason}", flush=True)

    async def _emit(self, window: MarketWindow, candidate: DropCandidate, volume: float,
                    p75: float, now: float) -> None:
        window.signaled = True
        ask, depth = side_ask(window.book, candidate.side)
        paper_reason = entry_reason(ask, depth, CONTRACTS)
        paper_filled = not paper_reason
        fee = (taker_fee(ask, CONTRACTS)
               if paper_filled and ask is not None else Decimal("0"))
        yes_mid = mid_yes(window.book)
        no_mid = (Decimal(1) - yes_mid) if yes_mid is not None else None
        signal = Signal(
            ts=now, coin=window.coin, ticker=window.ticker, close_ts=window.close_ts,
            side=candidate.side, fell_side=candidate.fell_side,
            high_ts=candidate.high_ts, high_mid=candidate.high_mid,
            current_mid=candidate.current_mid, drop=candidate.drop, volume_60=volume,
            p75=p75, yes_mid=yes_mid, no_mid=no_mid, ask=ask, depth=depth,
            filled=paper_filled, contracts=CONTRACTS if paper_filled else 0, fee=fee,
            fee_source="kalshi_taker_formula" if paper_filled else "",
            no_fill_reason=paper_reason, mode=self.mode,
        )
        signal_id = self.store.log_signal(signal)
        ask_txt = "n/a" if ask is None else f"{ask:.3f}"
        shown = ("CANARY SIGNAL" if self.mode == "canary" else
                 ("PAPER FILL" if paper_filled else "PAPER NO-FILL " + paper_reason))
        print(f"SIGNAL {window.coin.asset} {candidate.side} drop={candidate.drop:.3f} "
              f"vol60={volume:.1f} p75={p75:.1f} ask={ask_txt} depth={depth} {shown}",
              flush=True)
        if self.mode == "canary":
            canary_reason = entry_reason(ask, depth, CANARY_CONTRACTS)
            if canary_reason:
                self.store.set_fill(signal_id, False, 0, Decimal("0"), ask, depth,
                                    canary_reason, fee_source="")
                print(f"CANARY WAIT {canary_reason}; arm stays consumed", flush=True)
                return
            await self._canary_fill(signal_id, window, candidate, ask, depth)
            return
        if not paper_filled:
            self.store.set_fill(signal_id, False, 0, Decimal("0"), ask, depth, paper_reason,
                                fee_source="")
            return
        self.store.set_fill(signal_id, True, CONTRACTS, fee, ask, depth, "",
                            fee_source="kalshi_taker_formula")
        print(f"FILL {window.coin.asset} {candidate.side} {CONTRACTS} @ {ask_txt} "
              f"fee=${fee:.2f}", flush=True)

    async def _canary_fill(self, signal_id: int, window: MarketWindow,
                           candidate: DropCandidate, ask: Decimal,
                           depth: Decimal) -> None:
        if self.canary_done or self.broker is None:
            return
        self.canary_done = True
        if depth < Decimal(CANARY_CONTRACTS):
            reason = "depth_below_1"
            self.store.set_fill(signal_id, False, 0, Decimal("0"), ask, depth, reason)
            print(f"CANARY NO-FILL {reason}", flush=True)
            self.stop.set()
            return
        if candidate.side == "YES":
            order = await self.broker.place(
                window.ticker, "YES", "entry", ask, CANARY_CONTRACTS,
                post_only=False, tif="immediate_or_cancel")
        else:
            order = await self.broker.place_no_entry(window.ticker, ask, CANARY_CONTRACTS)
        if order is None:
            reason = "venue_rejected"
            self.store.set_fill(signal_id, False, 0, Decimal("0"), ask, depth, reason)
            print(f"CANARY NO-FILL {reason} errors={self.broker.errors[-3:]}", flush=True)
            self.stop.set()
            return

        rows = []
        for _ in range(10):
            rows = await self.broker.fills(order_id=order.order_id)
            if rows:
                break
            await asyncio.sleep(0.5)
        if rows:
            quantity = sum(dec(row.get("count_fp") or row.get("count")) for row in rows)
            held = int(quantity)
            if held > 0:
                key = "no_price_dollars" if candidate.side == "NO" else "yes_price_dollars"

                def fill_price(row: dict[str, Any]) -> Decimal:
                    if row.get(key) not in (None, ""):
                        return dec(row.get(key))
                    if candidate.side == "NO":
                        return Decimal(1) - dec(row.get("yes_price_dollars"))
                    return dec(row.get("yes_price_dollars"))

                notional = sum(dec(row.get("count_fp") or row.get("count"))
                               * fill_price(row) for row in rows)
                fee = sum(dec(row.get("fee_cost")) for row in rows)
                entry = notional / Decimal(held)
                self.store.set_fill(signal_id, True, held, fee, entry, depth, "",
                                    order_id=order.order_id, fee_source="venue_fill")
                print(f"CANARY FILL {candidate.side} {held} @ {entry:.3f} "
                      f"fee=${fee:.4f} order={order.order_id}", flush=True)
                return
        if order.filled > 0:
            held = int(order.filled)
            fee = taker_fee(ask, held)
            self.store.set_fill(signal_id, True, held, fee, ask, depth, "",
                                order_id=order.order_id, fee_source="estimate")
            print(f"CANARY FILL {candidate.side} {held} @ {ask:.3f} "
                  f"fee~${fee:.4f} order={order.order_id}", flush=True)
            return
        reason = "ioc_no_fill"
        self.store.set_fill(signal_id, False, 0, Decimal("0"), ask, depth, reason)
        print(f"CANARY NO-FILL {reason} order={order.order_id}", flush=True)
        self.stop.set()

    async def _settle_loop(self) -> None:
        while not self.stop.is_set():
            await asyncio.sleep(SETTLE_S)
            try:
                await self._settle()
            except Exception as exc:  # noqa: BLE001
                print(f"settle error: {type(exc).__name__}: {exc}", flush=True)

    async def _settle(self) -> None:
        for row in self.store.pending():
            status, payload = await self.client.get(f"/markets/{row['ticker']}")
            if status != 200:
                continue
            market = payload.get("market") or payload
            result = str(market.get("result") or "").lower()
            if result not in {"yes", "no"}:
                continue
            contracts = int(row["contracts"] or 0)
            if contracts <= 0:
                continue
            payout, pnl, per = settlement_pnl(
                str(row["side"]), result, contracts, dec(row["ask"]), dec(row["fee"]))
            self.store.settle(int(row["id"]), result, payout, pnl, per, time.time())
            print(f"SETTLE {row['asset']} {row['side']} {contracts} @ {row['ask']} "
                  f"result={result} pnl=${pnl:.2f} per={per:.3f}", flush=True)

    async def _status_loop(self) -> None:
        while not self.stop.is_set():
            await asyncio.sleep(STATUS_S)
            counts = self.store.counts()
            synced = sum(1 for ticker in self.active.values()
                         if ticker in (self.ws.books if self.ws else {})
                         and self.ws.books[ticker].synced)
            print(f"STATUS active={len(self.active)} synced={synced} "
                  f"signals={counts['signals']} fills={counts['fills']} "
                  f"pending={counts['pending']} buckets={counts['volume_buckets']} "
                  f"ws_msgs={self.ws.messages if self.ws else 0}", flush=True)


def _c(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:+.2f}c"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def print_report(report: dict[str, Any]) -> None:
    overall = report["overall"]
    print(f"FLIP_CHEAP paper — {report['date']} UTC")
    print(f"signals {overall['signals']}  fills {overall['fills']}  "
          f"contracts {overall['contracts']}  pending {overall['pending']}  "
          f"no-fill {overall['no_fills']} ({_pct(overall['no_fill_rate'])})")
    print(f"EV/contract {_c(overall['ev_per_contract'])}  "
          f"SE {_c(overall['se'])}  95% lower {_c(overall['lower_95'])}  "
          f"win {_pct(overall['win_rate'])}  fees ${overall['fees']:.2f}")
    print(f"{'coin':5} {'sig':>4} {'fill':>4} {'settled':>7} {'EV/ct':>9} "
          f"{'SE':>8} {'low':>9} {'win':>6} {'avg ask':>8} {'avg depth':>10}")
    for coin in COINS:
        row = report["by_coin"][coin.asset]
        ask = "n/a" if row["avg_ask"] is None else f"{row['avg_ask']:.3f}"
        depth = "n/a" if row["avg_depth"] is None else f"{row['avg_depth']:.1f}"
        print(f"{coin.asset:5} {row['signals']:4d} {row['fills']:4d} "
              f"{row['settled_fills']:7d} {_c(row['ev_per_contract']):>9} "
              f"{_c(row['se']):>8} {_c(row['lower_95']):>9} "
              f"{_pct(row['win_rate']):>6} {ask:>8} {depth:>10}")
    print("ETH/ZEC focus")
    for asset in ("ETH", "ZEC"):
        row = report["focus"][asset]
        print(f"  {asset}: signals {row['signals']} fills {row['fills']} "
              f"settled {row['settled_fills']} EV/contract {_c(row['ev_per_contract'])} "
              f"SE {_c(row['se'])} lower {_c(row['lower_95'])} "
              f"win {_pct(row['win_rate'])}")


def print_gate(gate: dict[str, Any]) -> None:
    print("FLIP_CHEAP live-canary gate (paper evidence only)")
    print(f"paper fills {gate['fills']}/{GATE_FILLS}  contracts {gate['contracts']}  "
          f"pending {gate['pending']}")
    print(f"EV/contract {_c(gate['ev_per_contract'])}  SE {_c(gate['se'])}  "
          f"95% lower {_c(gate['lower_95'])}")
    print("ELIGIBLE" if gate["eligible"] else f"NOT ELIGIBLE: {'; '.join(gate['reasons'])}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flip_cheap", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="paper recorder/simulator")
    run.add_argument("--seconds", type=float, default=0.0, help="0 = run until stopped")
    run.add_argument("--db", default=str(DEFAULT_DB))
    report = sub.add_parser("report", help="daily paper report")
    report.add_argument("--date", default=datetime.now(timezone.utc).date().isoformat())
    report.add_argument("--db", default=str(DEFAULT_DB))
    report.add_argument("--json", action="store_true")
    gate = sub.add_parser("gate", help="paper-evidence gate for the live canary")
    gate.add_argument("--db", default=str(DEFAULT_DB))
    gate.add_argument("--json", action="store_true")
    status = sub.add_parser("status", help="database counters and gate")
    status.add_argument("--db", default=str(DEFAULT_DB))
    status.add_argument("--json", action="store_true")
    canary = sub.add_parser("canary", help="one-contract live canary, gated and armed")
    canary.add_argument("--live", action="store_true")
    canary.add_argument("--seconds", type=float, default=4 * 3600.0)
    canary.add_argument("--db", default=str(DEFAULT_DB))
    return parser


async def _paper(args: argparse.Namespace) -> int:
    store = FlipStore(Path(args.db))
    return await FlipRunner(load_settings(), store, mode="paper",
                            seconds=args.seconds).run()


async def _canary(args: argparse.Namespace) -> int:
    if not args.live:
        print("refused: canary is live; pass --live")
        return 2
    settings = load_settings()
    if not settings.live:
        print("refused: ALLOW_LIVE=true and KALSHI_ENV=prod are required")
        return 2
    store = FlipStore(Path(args.db))
    try:
        gate = store.gate()
        print_gate(gate)
        if not gate["eligible"]:
            print("refused: paper gate not met; no arm was consumed")
            return 2
        path = arming.arm_path(LANE)
        record = arming.read(path)
        if not arming.fresh_operator_act(record, now=time.time(), lane=LANE):
            print(f"refused: no fresh lane arm at {path}; "
                  f"run scripts/arm_live.py --lane {LANE} --by NAME")
            return 2
        arming.consume(path)
    finally:
        store.close()
    print("ARM CONSUMED: waiting for one signal; this canary buys one contract and holds")
    store = FlipStore(Path(args.db))
    return await FlipRunner(settings, store, mode="canary", seconds=args.seconds).run()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.cmd == "run":
        return asyncio.run(_paper(args))
    if args.cmd == "canary":
        return asyncio.run(_canary(args))
    store = FlipStore(Path(args.db))
    try:
        if args.cmd == "report":
            day = date.fromisoformat(args.date)
            report = store.daily(day)
            if args.json:
                print(json.dumps(report, indent=2, default=str))
            else:
                print_report(report)
            return 0
        if args.cmd == "gate":
            gate = store.gate()
            if args.json:
                print(json.dumps(gate, indent=2, default=str))
            else:
                print_gate(gate)
            return 0
        counts = store.counts()
        gate = store.gate()
        if args.json:
            print(json.dumps({"counts": counts, "gate": gate}, indent=2, default=str))
        else:
            print(json.dumps(counts, indent=2, default=str))
            print_gate(gate)
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
