"""Liquidity-reward recorder and the documented Kalshi LIP scoring model.

Kalshi snapshots the book once a second.  A snapshot counts only when resting
size reaches `target_size` on **both** the YES and NO sides.  The reference price
is the first level where cumulative size reaches one fifth of the target, walking
down from the best bid.  Orders at or better than that price get full size
credit; each tick below it is multiplied by the discount factor.  The day's pool
is split by each participant's share of the total score.

The recorder stores the raw depth and program rates.  `estimate()` applies the
documented formula to our hypothetical order and labels the result an estimate:
we cannot see other participants' orders outside the published depth, and Kalshi
pays only after the program ends.

Run:

    .venv/bin/python -m openkal.maker.rewards record --once --top 25
    .venv/bin/python -m openkal.maker.rewards record --loop --interval 60
    .venv/bin/python -m openkal.maker.rewards summary
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

from ..auth import KalshiClient
from ..config import ROOT, Settings, load_settings

ZERO = Decimal("0")
TICK = Decimal("0.01")
DEFAULT_DB = ROOT / "data" / "rewards.db"
METHOD = "kalshi_lip_v1"


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
        return number / 1000.0 if number > 1e12 else number
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


@dataclass(frozen=True)
class Program:
    id: str
    ticker: str
    series: str
    reward_usd: Decimal
    start_ts: float
    end_ts: float
    target: Decimal
    discount: Decimal
    description: str = ""

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_ts - self.start_ts)

    @property
    def usd_per_day(self) -> Decimal:
        if self.duration_s <= 0:
            return ZERO
        return self.reward_usd / (Decimal(str(self.duration_s)) / Decimal("86400"))

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Program | None":
        ticker = str(row.get("market_ticker") or "")
        start = parse_ts(row.get("start_date"))
        end = parse_ts(row.get("end_date"))
        if not ticker or start <= 0 or end <= start:
            return None
        reward = dec(row.get("period_reward")) / Decimal("10000")  # centi-cents
        target = dec(row.get("target_size_fp"))
        discount = dec(row.get("discount_factor_bps"), "10000") / Decimal("10000")
        program_id = str(row.get("id") or row.get("program_id")
                         or f"{ticker}:{int(start)}")
        return cls(
            id=program_id, ticker=ticker,
            series=str(row.get("series_ticker") or ticker.split("-")[0]),
            reward_usd=reward, start_ts=start, end_ts=end,
            target=target, discount=discount,
            description=str(row.get("incentive_description") or ""),
        )


async def fetch_programs(client: KalshiClient, *, status: str = "active",
                         incentive_type: str = "liquidity",
                         max_pages: int = 12) -> list[Program]:
    """Every active liquidity program, paged.  The endpoint needs no auth."""
    out: list[Program] = []
    cursor: str | None = None
    for _ in range(max_pages):
        params: dict[str, Any] = {"status": status, "type": incentive_type, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        code, payload = await client.get("/incentive_programs", params)
        if code != 200:
            break
        rows = payload.get("incentive_programs") or []
        for row in rows:
            program = Program.from_row(row)
            if program is not None:
                out.append(program)
        cursor = payload.get("cursor") or payload.get("next_cursor")
        if not cursor or not rows:
            break
    return out


def rank(programs: Iterable[Program], now: float, limit: int = 100) -> list[Program]:
    """Active programs by reward per day, with Kalshi's target-size bounds."""
    rows = [p for p in programs
            if p.start_ts <= now < p.end_ts
            and Decimal("100") < p.target < Decimal("20000")
            and p.reward_usd > 0]
    rows.sort(key=lambda p: (p.usd_per_day, p.reward_usd), reverse=True)
    return rows[:limit]


@dataclass(frozen=True)
class Depth:
    yes: dict[Decimal, Decimal]
    no: dict[Decimal, Decimal]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Depth":
        book = payload.get("orderbook_fp") or payload.get("orderbook") or payload
        yes_rows = book.get("yes_dollars")
        no_rows = book.get("no_dollars")
        scale = Decimal("1")
        if yes_rows is None and no_rows is None:
            yes_rows, no_rows = book.get("yes") or [], book.get("no") or []
            scale = Decimal("100")
        return cls(yes=_ladder(yes_rows or [], scale), no=_ladder(no_rows or [], scale))

    def side_levels(self, side: str) -> dict[Decimal, Decimal]:
        return self.yes if side.upper() == "YES" else self.no

    def total(self, side: str) -> Decimal:
        return sum(self.side_levels(side).values(), ZERO)

    def best(self, side: str) -> Decimal | None:
        levels = self.side_levels(side)
        return max(levels) if levels else None

    def best_size(self, side: str, price: Decimal | None = None) -> Decimal:
        levels = self.side_levels(side)
        target = price if price is not None else (max(levels) if levels else None)
        return levels.get(target, ZERO) if target is not None else ZERO

    def levels(self, side: str, count: int = 10) -> list[tuple[Decimal, Decimal]]:
        return sorted(self.side_levels(side).items(), reverse=True)[:count]

    @property
    def yes_bid(self) -> Decimal | None:
        return self.best("YES")

    @property
    def yes_ask(self) -> Decimal | None:
        bid = self.best("NO")
        return (Decimal(1) - bid) if bid is not None else None

    @property
    def no_bid(self) -> Decimal | None:
        return self.best("NO")

    @property
    def no_ask(self) -> Decimal | None:
        bid = self.best("YES")
        return (Decimal(1) - bid) if bid is not None else None

    @property
    def mid_yes(self) -> Decimal | None:
        if self.yes_bid is None or self.yes_ask is None:
            return None
        return (self.yes_bid + self.yes_ask) / Decimal("2")

    def mid(self, side: str) -> Decimal | None:
        bid = self.best(side)
        if bid is None:
            return None
        other = self.best("NO" if side.upper() == "YES" else "YES")
        if other is None:
            return None
        ask = Decimal(1) - other
        return (bid + ask) / Decimal("2")


def _ladder(rows: Iterable[Iterable[Any]], scale: Decimal) -> dict[Decimal, Decimal]:
    out: dict[Decimal, Decimal] = {}
    for row in rows:
        try:
            price, size = Decimal(str(row[0])) / scale, Decimal(str(row[1]))
        except (IndexError, TypeError, InvalidOperation, ValueError):
            continue
        if size > 0:
            out[price] = out.get(price, ZERO) + size
    return out


def reference_price(levels: dict[Decimal, Decimal], target: Decimal) -> Decimal | None:
    """First price where cumulative size reaches one fifth of the target."""
    need = Decimal(target) / Decimal("5")
    if need <= 0:
        return None
    cumulative = ZERO
    for price in sorted(levels, reverse=True):
        cumulative += levels[price]
        if cumulative >= need:
            return price
    return None


def score_at(price: Decimal, size: Decimal, reference: Decimal | None,
             discount: Decimal, tick: Decimal = TICK) -> Decimal:
    if reference is None or size <= 0:
        return ZERO
    if price >= reference:
        return size
    steps = ((reference - price) / tick).to_integral_value(rounding=ROUND_HALF_UP)
    ticks = max(0, int(steps))
    return size * (discount ** ticks)


@dataclass(frozen=True)
class Estimate:
    program_id: str
    ticker: str
    side: str
    price: Decimal
    size: Decimal
    target: Decimal
    target_ok: bool
    reference: Decimal | None
    our_score: Decimal
    total_score: Decimal
    share: Decimal
    accrual_usd: Decimal
    reason: str = ""
    method: str = METHOD


def estimate(program: Program, depth: Depth, side: str, price: Decimal,
             size: Decimal, tick: Decimal = TICK) -> Estimate:
    """One-second reward estimate under the documented LIP formula."""
    side = side.upper()
    target_ok = (depth.total("YES") >= program.target
                 and depth.total("NO") >= program.target)
    levels = dict(depth.side_levels(side))
    levels[price] = levels.get(price, ZERO) + size
    reference = reference_price(levels, program.target)
    reason = ""
    if not target_ok:
        reason = "target_size_not_met"
    elif reference is None:
        reason = "no_reference_price"
    our_score = score_at(price, size, reference, program.discount, tick)
    total_score = sum((score_at(p, s, reference, program.discount, tick)
                       for p, s in levels.items()), ZERO)
    if reason:
        share = ZERO
        accrual = ZERO
    else:
        share = (our_score / total_score) if total_score > 0 else ZERO
        accrual = (program.reward_usd * share / Decimal(str(program.duration_s))
                   if program.duration_s > 0 else ZERO)
    return Estimate(
        program_id=program.id, ticker=program.ticker, side=side, price=price,
        size=size, target=program.target, target_ok=target_ok,
        reference=reference, our_score=our_score, total_score=total_score,
        share=share, accrual_usd=accrual, reason=reason,
    )


async def bulk_markets(client: KalshiClient, tickers: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for start in range(0, len(tickers), 100):
        chunk = tickers[start:start + 100]
        code, payload = await client.get(
            "/markets", {"tickers": ",".join(chunk), "limit": len(chunk)})
        if code != 200:
            continue
        for market in payload.get("markets") or []:
            if market.get("ticker"):
                out[str(market["ticker"])] = market
    return out


class RewardRecorder:
    """Record the top reward markets' rates and book depth on each sweep."""

    def __init__(self, db_path: Path | str = DEFAULT_DB,
                 client: KalshiClient | None = None,
                 top_n: int = 25, scan_n: int = 100, depth_levels: int = 10) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.client = client
        self.top_n = max(1, int(top_n))
        self.scan_n = max(self.top_n, int(scan_n))
        self.depth_levels = max(1, int(depth_levels))
        self.db = sqlite3.connect(str(self.db_path))
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self._init()

    def _init(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS programs (
              id TEXT PRIMARY KEY, ticker TEXT, series TEXT, reward_usd TEXT,
              start_ts REAL, end_ts REAL, target TEXT, discount TEXT,
              usd_per_day TEXT, description TEXT, updated_ts REAL
            );
            CREATE TABLE IF NOT EXISTS sweeps (
              id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, programs_seen INTEGER,
              markets INTEGER, total_usd_per_day TEXT, errors TEXT
            );
            CREATE TABLE IF NOT EXISTS books (
              id INTEGER PRIMARY KEY AUTOINCREMENT, sweep_id INTEGER, ticker TEXT,
              program_id TEXT, reward_usd TEXT, usd_per_day TEXT, target TEXT,
              discount TEXT, yes_bid TEXT, yes_ask TEXT, no_bid TEXT, no_ask TEXT,
              yes_total TEXT, no_total TEXT, mid_yes TEXT, spread TEXT,
              fav_side TEXT, fav_bid TEXT, fav_ask TEXT, estimate_usd TEXT
            );
            CREATE TABLE IF NOT EXISTS levels (
              sweep_id INTEGER, ticker TEXT, side TEXT, price TEXT, size TEXT
            );
            CREATE TABLE IF NOT EXISTS estimates (
              id INTEGER PRIMARY KEY AUTOINCREMENT, sweep_id INTEGER, ts REAL,
              ticker TEXT, program_id TEXT, side TEXT, price TEXT, size TEXT,
              reference TEXT, our_score TEXT, total_score TEXT, share TEXT,
              accrual_usd TEXT, target_ok INTEGER, reason TEXT, method TEXT
            );
            CREATE INDEX IF NOT EXISTS books_sweep ON books(sweep_id);
            CREATE INDEX IF NOT EXISTS levels_sweep ON levels(sweep_id);
            """
        )
        self.db.commit()

    def _upsert_program(self, program: Program, now: float) -> None:
        self.db.execute(
            """INSERT INTO programs
               (id,ticker,series,reward_usd,start_ts,end_ts,target,discount,
                usd_per_day,description,updated_ts)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 reward_usd=excluded.reward_usd, start_ts=excluded.start_ts,
                 end_ts=excluded.end_ts, target=excluded.target,
                 discount=excluded.discount, usd_per_day=excluded.usd_per_day,
                 description=excluded.description, updated_ts=excluded.updated_ts""",
            (program.id, program.ticker, program.series, str(program.reward_usd),
             program.start_ts, program.end_ts, str(program.target),
             str(program.discount), str(program.usd_per_day),
             program.description, now))

    async def sweep(self, now: float | None = None) -> dict[str, Any]:
        """One pass: rates, market snapshots, depth, and hypothetical estimates."""
        if self.client is None:
            raise RuntimeError("RewardRecorder needs a KalshiClient")
        now = time.time() if now is None else float(now)
        programs = await fetch_programs(self.client)
        ranked = rank(programs, now, self.scan_n)
        for program in ranked:
            self._upsert_program(program, now)
        markets = await bulk_markets(self.client, [p.ticker for p in ranked])
        chosen: list[tuple[Program, dict[str, Any]]] = []
        for program in ranked:
            market = markets.get(program.ticker)
            if not market:
                continue
            status = str(market.get("status") or "").lower()
            close_ts = parse_ts(market.get("close_time"))
            if status not in {"active", "open"} or (close_ts and close_ts <= now):
                continue
            if market.get("mve_collection_ticker") or program.ticker.startswith("KXMVE"):
                continue
            chosen.append((program, market))
            if len(chosen) >= self.top_n:
                break

        semaphore = asyncio.Semaphore(5)
        errors: list[str] = []

        async def depth_for(program: Program) -> Depth | None:
            async with semaphore:
                code, payload = await self.client.get(
                    f"/markets/{program.ticker}/orderbook", {"depth": self.depth_levels})
            if code != 200:
                errors.append(f"{program.ticker}: depth {code}")
                return None
            return Depth.from_payload(payload)

        depths = await asyncio.gather(*(depth_for(p) for p, _ in chosen))
        cursor = self.db.execute(
            "INSERT INTO sweeps (ts,programs_seen,markets,total_usd_per_day,errors) "
            "VALUES (?,?,?,?,?)",
            (now, len(programs), len(chosen),
             str(sum((p.usd_per_day for p, _ in chosen), ZERO)),
             json.dumps(errors[:20])))
        sweep_id = int(cursor.lastrowid)
        total_est = ZERO
        for (program, market), depth in zip(chosen, depths):
            if depth is None:
                continue
            yes_bid = dec(market.get("yes_bid_dollars"), "0")
            yes_ask = dec(market.get("yes_ask_dollars"), "0")
            no_bid = dec(market.get("no_bid_dollars"), str(Decimal(1) - yes_ask))
            no_ask = dec(market.get("no_ask_dollars"), str(Decimal(1) - yes_bid))
            mid = (yes_bid + yes_ask) / 2 if yes_bid and yes_ask else None
            fav_side, fav_bid, fav_ask = favourite_quote(yes_bid, yes_ask, no_bid, no_ask)
            estimate_usd = ZERO
            if fav_side and fav_bid is not None and fav_ask is not None and fav_ask > fav_bid:
                estimate_row = estimate(program, depth, fav_side, fav_bid, Decimal("5"))
                estimate_usd = estimate_row.accrual_usd
                total_est += estimate_usd
                self.db.execute(
                    """INSERT INTO estimates
                       (sweep_id,ts,ticker,program_id,side,price,size,reference,
                        our_score,total_score,share,accrual_usd,target_ok,reason,method)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (sweep_id, now, program.ticker, program.id, fav_side,
                     str(fav_bid), "5", None if estimate_row.reference is None else str(estimate_row.reference),
                     str(estimate_row.our_score), str(estimate_row.total_score),
                     str(estimate_row.share), str(estimate_row.accrual_usd),
                     int(estimate_row.target_ok), estimate_row.reason, estimate_row.method))
            self.db.execute(
                """INSERT INTO books
                   (sweep_id,ticker,program_id,reward_usd,usd_per_day,target,discount,
                    yes_bid,yes_ask,no_bid,no_ask,yes_total,no_total,mid_yes,spread,
                    fav_side,fav_bid,fav_ask,estimate_usd)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (sweep_id, program.ticker, program.id, str(program.reward_usd),
                 str(program.usd_per_day), str(program.target), str(program.discount),
                 str(yes_bid), str(yes_ask), str(no_bid), str(no_ask),
                 str(depth.total("YES")), str(depth.total("NO")),
                 None if mid is None else str(mid),
                 None if mid is None else str((yes_ask - yes_bid) if yes_ask and yes_bid else 0),
                 fav_side, None if fav_bid is None else str(fav_bid),
                 None if fav_ask is None else str(fav_ask), str(estimate_usd)))
            for side in ("YES", "NO"):
                for price, size in depth.levels(side, self.depth_levels):
                    self.db.execute(
                        "INSERT INTO levels (sweep_id,ticker,side,price,size) VALUES (?,?,?,?,?)",
                        (sweep_id, program.ticker, side, str(price), str(size)))
        self.db.commit()
        return {
            "sweep_id": sweep_id,
            "programs_seen": len(programs),
            "markets": len(chosen),
            "estimate_usd_per_second": str(total_est),
            "errors": errors,
        }

    def summary(self) -> dict[str, Any]:
        row = self.db.execute(
            "SELECT id,ts,programs_seen,markets,total_usd_per_day,errors "
            "FROM sweeps ORDER BY id DESC LIMIT 1").fetchone()
        if row is None:
            return {"sweeps": 0}
        sweep_id, ts, programs_seen, markets, total, errors = row
        top = self.db.execute(
            "SELECT ticker,usd_per_day,fav_side,fav_bid,fav_ask,estimate_usd "
            "FROM books WHERE sweep_id=? ORDER BY CAST(usd_per_day AS REAL) DESC LIMIT 5",
            (sweep_id,)).fetchall()
        return {
            "sweeps": self.db.execute("SELECT COUNT(*) FROM sweeps").fetchone()[0],
            "last_sweep_id": sweep_id,
            "last_ts": ts,
            "programs_seen": programs_seen,
            "markets": markets,
            "total_usd_per_day": total,
            "errors": json.loads(errors or "[]"),
            "top": [
                {"ticker": r[0], "usd_per_day": r[1], "fav_side": r[2],
                 "fav_bid": r[3], "fav_ask": r[4], "estimate_usd": r[5]}
                for r in top
            ],
        }

    def close(self) -> None:
        self.db.commit()
        self.db.close()


def favourite_quote(yes_bid: Decimal, yes_ask: Decimal, no_bid: Decimal,
                    no_ask: Decimal) -> tuple[str | None, Decimal | None, Decimal | None]:
    """The high-probability side's own bid/ask."""
    if yes_ask <= 0:
        return None, None, None
    if (yes_bid + yes_ask) / 2 >= Decimal("0.5"):
        return "YES", yes_bid, yes_ask
    return "NO", no_bid, no_ask


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rewards", description="Liquidity reward recorder")
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record", help="record one sweep or loop")
    rec.add_argument("--db", default=str(DEFAULT_DB))
    rec.add_argument("--top", type=int, default=25)
    rec.add_argument("--scan", type=int, default=100)
    rec.add_argument("--once", action="store_true")
    rec.add_argument("--loop", action="store_true")
    rec.add_argument("--interval", type=float, default=60.0)
    rec.add_argument("--seconds", type=float, default=0.0)
    rec.add_argument("--json", action="store_true")
    sub.add_parser("summary", help="show the latest sweep").add_argument(
        "--db", default=str(DEFAULT_DB))
    args = parser.parse_args(argv)

    if args.command == "summary":
        recorder = RewardRecorder(args.db)
        print(json.dumps(recorder.summary(), indent=2))
        recorder.close()
        return 0

    async def run() -> int:
        settings: Settings = load_settings()
        client = KalshiClient(settings)
        recorder = RewardRecorder(args.db, client, top_n=args.top, scan_n=args.scan)
        started = time.time()
        try:
            while True:
                result = await recorder.sweep()
                print(json.dumps(result, indent=2), flush=True)
                if args.once or not args.loop:
                    break
                if args.seconds and time.time() - started + args.interval > args.seconds:
                    break
                await asyncio.sleep(max(1.0, args.interval))
        finally:
            recorder.close()
            await client.close()
        return 0

    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
