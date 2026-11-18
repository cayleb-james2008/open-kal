"""Prove the live path with real money, end to end.

This is the acceptance test for `LiveExecutor`, not a strategy backtest:

  1. open a real position through `LiveExecutor` (maker first, taker if the maker
     bid does not fill), on 1 contract
  2. run the same `decide`/`Executor` loop the engine runs until the position closes
  3. reconcile the local SQLite ledger against the venue's own fills tape
  4. place a resting bid on a second lane, fire the kill switch, and assert the venue
     has no resting orders and no open positions afterwards

Nothing here estimates a fill or a fee. Every number printed is either from
`/portfolio/fills` or from the ledger row written when the venue reported the fill.

    .venv/bin/python scripts/live_probe.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openkal import arming  # noqa: E402
from openkal.config import SLOTS, load_settings  # noqa: E402
from openkal.engine import Engine  # noqa: E402
from openkal.live_exec import LiveExecutor  # noqa: E402
from openkal.store import Store  # noqa: E402
from openkal.strategy import Decision, Params, contracts_for, decide  # noqa: E402


async def venue_state(broker) -> tuple[dict, list]:
    return await broker.open_orders(), await broker.fills(limit=200)


def log(message: str) -> None:
    print(f"[probe] {message}", flush=True)


async def open_position(engine: Engine, slot, taker: bool, seconds: float = 120.0) -> bool:
    """Drive LiveExecutor until it holds a real position, or give up cleanly."""
    deadline = time.time() + seconds
    ex: LiveExecutor = engine.live
    while time.time() < deadline:
        await engine._refresh_market(slot)
        quote = slot.quote
        if quote is None:
            await asyncio.sleep(2)
            continue
        seconds_left = max(0.0, quote.close_ts - time.time())
        if seconds_left < slot.params.flatten_before_close_s + 20:
            await asyncio.sleep(2)
            continue
        slot.bankroll = quote.yes_ask if taker else quote.yes_bid   # exactly one contract
        decision = Decision(
            "entry", f"{'taker' if taker else 'maker'} probe", side="YES",
            price=quote.yes_ask if taker else quote.yes_bid, p_model=slot.p_up,
        )
        venue = {"orders": await engine.broker.open_orders(),
                 "fills": await engine.broker.fills(limit=200)}
        await ex.step(slot, decision, quote, seconds_left, time.time(), venue)
        if slot.position is not None:
            log(f"FILLED {slot.position.contracts} x {slot.position.side} at {slot.position.entry} "
                f"({slot.position.ticker})")
            return True
        await asyncio.sleep(3)
    if ex.entry_order is not None and slot.quote is not None:
        await engine.broker.cancel(ex.entry_order.order_id, slot.quote.ticker)
        ex.entry_order = None
        slot.order = None
        log("maker bid did not fill; cancelled instead of waiting")
    return False


async def close_position(engine: Engine, slot, seconds: float = 240.0) -> bool:
    """Run the engine's own live loop until the position is closed by the venue."""
    deadline = time.time() + seconds
    ex: LiveExecutor = engine.live
    while time.time() < deadline and slot.position is not None:
        await engine._refresh_market(slot)
        quote = slot.quote
        if quote is None:
            await asyncio.sleep(2)
            continue
        seconds_left = max(0.0, quote.close_ts - time.time())
        decision = decide(quote=quote, p_up=slot.p_up, seconds_left=seconds_left, spot_age=0.0,
                          params=slot.params, position=slot.position,
                          edge_hurdle=slot.params.edge_hurdle, can_open=False)
        venue = {"orders": await engine.broker.open_orders(),
                 "fills": await engine.broker.fills(limit=200)}
        await ex.step(slot, decision, quote, seconds_left, time.time(), venue)
        if slot.position is not None and seconds_left < 60:
            # Never carry a probe into the settlement print.
            await ex.flatten_everything(engine.slots)
            break
        await asyncio.sleep(3)
    return slot.position is None


async def reconcile(engine: Engine, slot, ticker: str) -> dict:
    """Ledger row vs the venue's own fills tape. To the cent."""
    rows = [r for r in engine.store.recent_trades(50) if r["ticker"] == ticker and r["source"] == "live"]
    fills = [f for f in await engine.broker.fills(limit=200) if f.get("market_ticker") == ticker]
    venue_pnl = Decimal("0")
    venue_fees = Decimal("0")
    for f in fills:
        qty = Decimal(str(f.get("count_fp") or "0"))
        yes = Decimal(str(f.get("yes_price_dollars") or "0"))
        fee = Decimal(str(f.get("fee_cost") or "0"))
        venue_fees += fee
        if f.get("action") == "sell":
            venue_pnl += yes * qty - fee
        else:
            venue_pnl -= yes * qty + fee
    local = rows[0] if rows else None
    local_pnl = Decimal(str(local["pnl"])) if local else Decimal("0")
    # The ledger's exit leg is net of the exit fee; add the entry fee the venue charged
    # so both sides describe the same cash flow.
    local_total = local_pnl
    if local:
        local_total = (Decimal(str(local["exit_price"])) - Decimal(str(local["entry"]))) * int(local["contracts"]) \
            - Decimal(str(local["fee"])) - entry_fee(fills)
    return {
        "ticker": ticker,
        "venue_fills": len(fills),
        "venue_pnl": venue_pnl,
        "venue_fees": venue_fees,
        "local_pnl": local_pnl,
        "local_total_net": local_total,
        "delta": local_total - venue_pnl,
        "ledger_row": local,
    }


def entry_fee(fills: list) -> Decimal:
    total = Decimal("0")
    for f in fills:
        qty = Decimal(str(f.get("count_fp") or "0"))
        if f.get("action") == "buy":
            total += Decimal(str(f.get("fee_cost") or "0"))
    return total


async def main() -> int:
    # Real orders, so the same operator record the engine requires: the arm lock is
    # only real if no script can bypass it.
    record = arming.consume()
    if record is None:
        log("refusing: no operator arm record; run scripts/arm_live.py --by <name> first")
        return 2
    log(f"operator {record.get('by')} armed this run")
    settings = load_settings()
    store = Store(Path(settings.db_path))
    engine = Engine(settings, store)
    await engine._sync_account()
    log(f"venue {settings.env_name} · balance {engine.venue_balance}")
    engine.live = LiveExecutor(engine, engine.broker)
    engine.live_armed = True

    slot = next(s for s in engine.slots if s.cfg.series == "KXBTC15M")
    other = next(s for s in engine.slots if s.cfg.series == "KXETH15M")
    params = Params()

    opened = await open_position(engine, slot, taker=False, seconds=75)
    if not opened:
        log("maker probe produced no fill; taking the position with an IOC at the ask")
        opened = await open_position(engine, slot, taker=True, seconds=45)
    if not opened:
        log("FAIL: LiveExecutor could not open a position")
        return 2
    ticker = slot.position.ticker

    # A resting bid on a second lane, so the kill switch has something to cancel.
    await engine._refresh_market(other)
    if other.quote is not None and other.quote.yes_bid >= Decimal("0.05"):
        other.bankroll = other.quote.yes_bid
        await engine.broker.place(other.quote.ticker, "YES", "entry", other.quote.yes_bid, 1)
        log(f"resting probe bid on {other.cfg.series} at {other.quote.yes_bid}")

    closed = await close_position(engine, slot, seconds=240)
    log(f"position closed by the venue: {closed}")

    report = await reconcile(engine, slot, ticker)
    log(f"reconciliation: {report['venue_fills']} venue fills, venue net "
        f"{report['venue_pnl']:.4f}, local net {report['local_total_net']:.4f}, "
        f"delta {report['delta']:.4f}")

    orders, _ = await venue_state(engine.broker)
    log(f"before kill: {len(orders)} resting orders, position {await engine.broker.position(ticker)}")

    await engine.kill("probe: verify kill switch")
    orders, _ = await venue_state(engine.broker)
    pos = await engine.broker.position(ticker)
    log(f"after kill: {len(orders)} resting orders, position {pos}")

    status, payload = await engine.client.get("/portfolio/positions", {"limit": 100})
    open_positions = [p for p in (payload.get("market_positions") or [])
                      if int(float(p.get("position_fp") or p.get("position") or 0)) != 0]
    log(f"open positions after kill: {len(open_positions)}")
    ok = abs(report["delta"]) <= Decimal("0.01") and len(orders) == 0 and not open_positions
    await engine.client.close()
    log("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
