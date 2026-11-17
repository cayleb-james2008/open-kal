"""Kill-switch acceptance test against the live venue.

Puts real state on the book — a resting bid and an open position — then calls
`Engine.kill` and asserts the venue comes back flat. No simulation of either step.
"""
from __future__ import annotations

import asyncio
import sys
import time
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openkal.config import load_settings  # noqa: E402
from openkal.engine import Engine  # noqa: E402
from openkal.live_exec import LiveExecutor  # noqa: E402
from openkal.store import Store  # noqa: E402
from openkal.strategy import Decision  # noqa: E402


def log(m: str) -> None:
    print(f"[kill] {m}", flush=True)


async def wait_flat(engine: Engine, timeout: float = 25.0) -> tuple[int, int]:
    """The venue reflects cancels and IOCs asynchronously; poll instead of sampling once."""
    deadline = time.time() + timeout
    last = (-1, -1)
    while time.time() < deadline:
        last = await counts(engine)
        if last == (0, 0):
            return last
        await asyncio.sleep(1.5)
    return last


async def counts(engine: Engine) -> tuple[int, int]:
    orders = await engine.broker.open_orders()
    status, payload = await engine.client.get("/portfolio/positions", {"limit": 100})
    open_pos = [p for p in (payload.get("market_positions") or [])
                if int(float(p.get("position_fp") or p.get("position") or 0)) != 0]
    return len(orders), len(open_pos)


async def main() -> int:
    settings = load_settings()
    store = Store(Path(settings.db_path))
    engine = Engine(settings, store)
    await engine._sync_account()
    engine.live = LiveExecutor(engine, engine.broker)
    engine.live_armed = True
    log(f"balance {engine.venue_balance}")

    trade_slot = engine.slots[0]
    rest_slot = engine.slots[1]
    await engine._refresh_market(trade_slot)
    await engine._refresh_market(rest_slot)
    q, rq = trade_slot.quote, rest_slot.quote
    if q is None or rq is None:
        log("FAIL: no live market to test on")
        return 2

    # 1. a resting bid that cannot fill (deep below the book)
    rest_price = max(Decimal("0.02"), rq.yes_bid - Decimal("0.25"))
    order = await engine.broker.place(rq.ticker, "YES", "entry", rest_price, 1)
    log(f"resting bid {rq.ticker} at {rest_price}: {'placed' if order else 'rejected'}")

    # 2. a real position through LiveExecutor
    trade_slot.bankroll = q.yes_ask
    trade_slot.params = trade_slot.params.__class__(entry_style="taker")
    decision = Decision("entry", "kill test", side="YES", price=q.yes_ask, p_model=trade_slot.p_up)
    for _ in range(10):
        await engine._refresh_market(trade_slot)
        q = trade_slot.quote
        if q is None:
            break
        venue = {"orders": await engine.broker.open_orders(),
                 "fills": await engine.broker.fills(limit=100)}
        await engine.live.step(trade_slot, decision, q, max(0.0, q.close_ts - time.time()),
                               time.time(), venue)
        if trade_slot.position is not None:
            break
        await asyncio.sleep(2)
    if trade_slot.position is None:
        log("FAIL: could not open the position to test with")
        await engine.broker.cancel(order.order_id, rq.ticker) if order else None
        return 2
    log(f"position {trade_slot.position.contracts} x {trade_slot.position.side} "
        f"at {trade_slot.position.entry}")

    before = await counts(engine)
    log(f"before kill: {before[0]} resting orders, {before[1]} open positions")

    await engine.kill("kill-switch acceptance test")
    after = await wait_flat(engine)
    log(f"after kill: {after[0]} resting orders, {after[1]} open positions")

    ok = before[0] >= 1 and before[1] >= 1 and after == (0, 0)
    await engine.client.close()
    log("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
