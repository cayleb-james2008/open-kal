"""The live exit path, exercised against a broker that behaves like the venue.

Three failures were observed live before these existed:

  * a take-profit sell at 0.88 on a book bidding 0.922 was rejected as
    `post only cross` and retried forever — the position never closed;
  * each retry reset the fallback deadline, so the IOC that was supposed to save it
    could never fire;
  * a rejected placement left `slot.exit` set, so nothing was ever retried either.

The incident this file now also pins down: an unhandled `NameError` in the live
exit path fired on every tick after 85 SOL contracts filled at 0.77, so the
take-profit and the scratch were never placed and the position went to settlement
unguarded. Multi-fill aggregation (42.75 + 41.25 on one order) is covered for the
same reason: reading only the first print booked half a position.

Run: .venv/bin/python tests/test_live_exit.py
"""
from __future__ import annotations

import asyncio
import sys
import time
import types
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openkal.config import (COMPOUND_STAGES, HARD_MAX_ENTRY_PRICE,  # noqa: E402
                            peak_drawdown_tripped, stage_for, staged_risk_cap)
from openkal import decision_gate, features  # noqa: E402
from openkal.decision_gate import LayaGate, Verdict  # noqa: E402
from openkal.engine import Engine  # noqa: E402
from openkal.execution import Executor, RestingExit, RestingOrder  # noqa: E402
from openkal.live_exec import (DEPTH_CLIP_FRACTION, LiveExecutor,  # noqa: E402
                               depth_clipped, sized_contracts)
from openkal.strategy import Decision, Params, Position, Quote, decide  # noqa: E402

TICKER = "KXSOL15M-26NOV161300-00"
NOW = 1_000_000.0
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))


class FakeBroker:
    """Accepts or rejects post-only exits the way the venue does, and fills IOC."""

    def __init__(self, *, post_only_ok: bool = True, fill_qty: float = 10.0,
                 fill_price: str | None = None, reject_all: bool = False) -> None:
        self.post_only_ok = post_only_ok
        self.reject_all = reject_all
        self.fill_qty = fill_qty
        self.fill_price = fill_price
        self.placed: list[dict] = []
        self.errors: list[str] = []
        self.cancelled: list[str] = []
        self.resting: dict[str, dict] = {}

    async def place(self, ticker, side, action, price, contracts, post_only=True,
                    tif="good_till_canceled", reduce_only=False):
        self.placed.append(dict(ticker=ticker, side=side, action=action, price=price,
                                contracts=contracts, post_only=post_only, tif=tif,
                                reduce_only=reduce_only))
        if self.reject_all or (post_only and not self.post_only_ok):
            self.errors.append("400 {'error': {'details': 'post only cross'}}")
            return None
        order_id = f"ord{len(self.placed)}"
        if tif != "immediate_or_cancel":
            self.resting[order_id] = {"order_id": order_id, "ticker": ticker}
        return SimpleNamespace(order_id=order_id)

    async def cancel(self, order_id, ticker=None):
        self.cancelled.append(order_id)
        self.resting.pop(order_id, None)
        return True

    async def open_orders(self):
        return dict(self.resting)

    async def fills(self, order_id=None, ticker=None, limit=100):
        sent = self.placed[-1] if self.placed else {"price": Decimal("0.90")}
        price = self.fill_price or str(sent["price"])
        return [{"order_id": order_id, "no_price_dollars": price,
                 "yes_price_dollars": price, "count_fp": str(self.fill_qty),
                 "fee_cost": "0.015", "is_taker": True}]


def quote(bid: str, ask: str, bid_sz: str | None = None,
          ask_sz: str | None = None) -> Quote:
    return Quote(ticker=TICKER, series="KXSOL15M", strike=70.0, close_ts=NOW + 200,
                 yes_bid=Decimal(bid), yes_ask=Decimal(ask),
                 last=Decimal(bid), updated_ts=NOW, received_ts=NOW,
                 yes_bid_sz=None if bid_sz is None else Decimal(bid_sz),
                 yes_ask_sz=None if ask_sz is None else Decimal(ask_sz))


def slot_for(q: Quote) -> SimpleNamespace:
    pos = Position(ticker=TICKER, side="NO", entry=Decimal("0.83"), contracts=10,
                   opened_ts=NOW - 50, p_entry=0.6, stake=Decimal("8.30"),
                   tp=Decimal("0.88"), sl=Decimal("0.58"), signal="z_obi_flow")
    return SimpleNamespace(cfg=SimpleNamespace(series="KXSOL15M"), params=SimpleNamespace(
        zobi_barrier_s=45.0,
        zobi_scratch_patience_s=20.0, flatten_before_close_s=45.0,
        zobi_entry_patience_s=15.0, take_profit=Decimal("0.05"),
        stop_loss=Decimal("0.25"), edge_hurdle=Decimal("0.015")),
        bankroll=Decimal("65.58"), peak_bankroll=Decimal("65.58"), order=None,
        position=pos, exit=None, p_up=0.6, spot=70.0, quote=q, window_entries=1,
        pnl_today=Decimal("0"), wins=0, losses=0, consecutive_losses=0, ece=0.0,
        pairs=[], last_trade=None, window_halted=False, halted=False)


def ctx() -> SimpleNamespace:
    return SimpleNamespace(cash=Decimal("100"), realized=Decimal("0"), fills=[],
                           store=None, supervisor=None, source="live")


def run(slot, ex, decision, q, seconds_left=200.0) -> None:
    asyncio.run(ex._decide_exit(slot, decision, q, seconds_left, NOW))


def venue(fills: list[dict], order_ids: tuple[str, ...] = ()) -> dict:
    return {"orders": {oid: {} for oid in order_ids}, "fills": fills}


# --------------------------------------------------------------------------
# 1. Multi-fill aggregation: one order, several prints.
# --------------------------------------------------------------------------
def control_multi_fill() -> None:
    ex = LiveExecutor(ctx(), FakeBroker())
    rows = [
        {"order_id": "o1", "count_fp": "40.05", "yes_price_dollars": "0.77",
         "fee_cost": "0.000000", "is_taker": False},
        {"order_id": "o1", "count_fp": "44.95", "yes_price_dollars": "0.77",
         "fee_cost": "0.000000", "is_taker": False},
    ]
    summary = ex._fill_summary(venue(rows), "o1", "YES")
    check("fills: 40.05 + 44.95 on one order is 85 contracts, not 40",
          summary is not None and summary[0] == 85,
          f"summary={summary}")
    check("fills: the aggregated price is the size-weighted fill price",
          summary is not None and summary[1] == Decimal("0.7700")
          and summary[2] == Decimal("0.000000"),
          f"price/fee={summary[1:]}")

    # And the entry poll books the whole position from that summary.
    slot = slot_for(quote("0.13", "0.15"))
    slot.position = None
    slot.order = RestingOrder(ticker=TICKER, side="YES", price=Decimal("0.77"),
                              contracts=85, placed_ts=NOW, reserved=Decimal("65.45"))
    ex = LiveExecutor(ctx(), FakeBroker())
    ex.ctx.cash = Decimal("934.55")
    slot.order.order_id = "o1"
    asyncio.run(ex._poll_entry(slot, slot.quote, NOW, venue(rows, ("o1",))))
    check("fills: the live position is the venue's 85 contracts, not a half fill",
          slot.position is not None and slot.position.contracts == 85
          and slot.position.entry == Decimal("0.7700"),
          f"position={slot.position.contracts if slot.position else None}")

    # A take-profit exit that prints in three pieces reconciles the same way.
    pos = slot.position
    pos.tp = Decimal("0.82")
    slot.exit = RestingExit(ticker=TICKER, side="YES", price=Decimal("0.82"),
                            contracts=85, deadline=NOW + 8, reason="take-profit reached",
                            order_id="o2")
    ex.exit_reason = "take-profit reached"
    exits = [
        {"order_id": "o2", "count_fp": "40.05", "yes_price_dollars": "0.84", "fee_cost": "0.0",
         "is_taker": False},
        {"order_id": "o2", "count_fp": "44.95", "yes_price_dollars": "0.84", "fee_cost": "0.0",
         "is_taker": False},
    ]
    asyncio.run(ex._poll_exit(slot, slot.quote, NOW, venue(exits, ("o2",))))
    check("fills: a three-print take-profit closes the whole 85-lot",
          slot.position is None and slot.exit is None,
          f"position={slot.position} exit={slot.exit}")
    pnl = Decimal(str(ex.ctx.fills[-1]["pnl"])) if ex.ctx.fills else Decimal("0")
    # 85 * (0.84 - 0.77) = 5.95, no fee on either maker leg.
    check("fills: the P&L is the full position at the aggregated price",
          pnl == Decimal("5.9500"), f"pnl={pnl}")


# --------------------------------------------------------------------------
# 1b. Partial fills: book what the venue filled, when it has finished filling.
# --------------------------------------------------------------------------
def control_partial_fills() -> None:
    # Live, 17:31: a 59-lot ETH bid printed 0.22 + 22.88 + 35.90 inside 270ms. The
    # first poll saw 0.22, booked 1 contract and dropped the order.
    def row(oid, qty, px="0.58", taker=False, fee="0.0"):
        return {"order_id": oid, "count_fp": qty, "yes_price_dollars": px,
                "fee_cost": fee, "is_taker": taker}

    slot = slot_for(quote("0.57", "0.59"))
    slot.position = None
    slot.order = RestingOrder(ticker=TICKER, side="YES", price=Decimal("0.58"),
                              contracts=59, placed_ts=NOW, reserved=Decimal("34.22"),
                              order_id="e1")
    ex = LiveExecutor(ctx(), FakeBroker())
    asyncio.run(ex._poll_entry(slot, slot.quote, NOW, venue([row("e1", "0.22")])))
    check("partial: the first 0.22 print of a 59-lot entry books nothing yet",
          slot.position is None and slot.order is not None,
          f"position={slot.position.contracts if slot.position else None}")
    prints = [row("e1", "0.22"), row("e1", "22.88"), row("e1", "35.90")]
    asyncio.run(ex._poll_entry(slot, slot.quote, NOW + 1.3, venue(prints)))
    check("partial: the completed entry books all 59 contracts",
          slot.position is not None and slot.position.contracts == 59 and slot.order is None,
          f"position={slot.position.contracts if slot.position else None}")

    # An entry that stops short: booked at its settled size, not dropped, not 1.
    slot = slot_for(quote("0.57", "0.59"))
    slot.position = None
    slot.order = RestingOrder(ticker=TICKER, side="YES", price=Decimal("0.58"),
                              contracts=59, placed_ts=NOW, reserved=Decimal("34.22"),
                              order_id="e2")
    ex = LiveExecutor(ctx(), FakeBroker())
    broker = FakeBroker()
    ex = LiveExecutor(ctx(), broker)
    part = venue([row("e2", "20.00")])
    live = venue(part["fills"], ("e2",))
    asyncio.run(ex._poll_entry(slot, slot.quote, NOW, live))
    asyncio.run(ex._poll_entry(slot, slot.quote, NOW + 61, live))      # stale: pulled
    asyncio.run(ex._poll_entry(slot, slot.quote, NOW + 62.3, part))    # off the book
    asyncio.run(ex._poll_entry(slot, slot.quote, NOW + 32, venue([]))) # failed read
    early, kept = slot.position, slot.order
    asyncio.run(ex._poll_entry(slot, slot.quote, NOW + 65.5, part))
    check("partial: a stale part-filled entry is cancelled, and survives an empty read",
          broker.cancelled == ["e2"] and early is None and kept is not None,
          f"cancelled={broker.cancelled} early={early} kept={kept is not None}")
    check("partial: a cancelled 20-of-59 entry is booked as 20 once the total settles",
          early is None and slot.position is not None and slot.position.contracts == 20,
          f"early={early} position={slot.position.contracts if slot.position else None}")

    # Live, 17:01: a 45-lot XRP IOC exit filled 15.35 and was booked as a full close.
    slot = slot_for(quote("0.13", "0.15"))
    slot.position.contracts = 45
    slot.exit = RestingExit(ticker=TICKER, side="NO", price=Decimal("0.85"), contracts=45,
                            deadline=NOW + 8, reason="take-profit reached", order_id="x1")
    ex = LiveExecutor(ctx(), FakeBroker())
    part = venue([{"order_id": "x1", "count_fp": "15.35", "no_price_dollars": "0.88",
                   "yes_price_dollars": "0.12", "fee_cost": "0.2445", "is_taker": True}])
    asyncio.run(ex._poll_exit(slot, slot.quote, NOW, part))
    held = slot.position.contracts if slot.position else None
    asyncio.run(ex._poll_exit(slot, slot.quote, NOW + 3.5, part))
    check("partial: a 15.35-of-45 exit closes 15 and keeps a 30-lot position",
          held == 45 and slot.position is not None and slot.position.contracts == 30
          and slot.exit is None and ex.ctx.fills[-1]["contracts"] == 15,
          f"held={held} left={slot.position.contracts if slot.position else None}")

    # A part-filled resting exit that reaches its deadline is pulled, not forgotten.
    slot = slot_for(quote("0.13", "0.15"))
    slot.position.contracts = 45
    slot.exit = RestingExit(ticker=TICKER, side="NO", price=Decimal("0.88"), contracts=45,
                            deadline=NOW - 1, reason="take-profit reached",
                            retry_on_timeout=True, order_id="x2")
    broker = FakeBroker()
    ex = LiveExecutor(ctx(), broker)
    part["fills"][0]["order_id"] = "x2"
    asyncio.run(ex._poll_exit(slot, slot.quote, NOW, venue(part["fills"], ("x2",))))
    check("partial: a part-filled exit at its deadline is cancelled and still tracked",
          broker.cancelled == ["x2"] and slot.exit is not None and slot.exit.order_id == "x2"
          and slot.position.contracts == 45,
          f"cancelled={broker.cancelled} order_id={slot.exit.order_id if slot.exit else None}")
    gone = venue(part["fills"])
    asyncio.run(ex._poll_exit(slot, slot.quote, NOW + 1.3, gone))
    asyncio.run(ex._poll_exit(slot, slot.quote, NOW + 2, venue([])))   # failed read
    mid = (slot.position.contracts, slot.exit is not None)
    asyncio.run(ex._poll_exit(slot, slot.quote, NOW + 3.5, gone))
    check("partial: it books 15 only after the cancel settles, through an empty read",
          mid == (45, True) and slot.position is not None
          and slot.position.contracts == 30 and slot.exit is None,
          f"mid={mid} left={slot.position.contracts if slot.position else None}")


# --------------------------------------------------------------------------
# 2. Take-profit, scratch and settlement reconciliation.
# --------------------------------------------------------------------------
def control_tp_scratch_settle() -> None:
    # The BTC trade: 84 @ 0.78 out at 0.84, taker on the exit, fees on the tape.
    q = quote("0.13", "0.15")
    slot = slot_for(q)
    slot.position = Position(ticker=TICKER, side="YES", entry=Decimal("0.78"),
                             contracts=84, opened_ts=NOW - 60, p_entry=0.62,
                             stake=Decimal("65.52"), tp=Decimal("0.83"),
                             sl=Decimal("0.53"), signal="z_obi_flow")
    ex = LiveExecutor(ctx(), FakeBroker())
    ex.ctx.cash = Decimal("0")
    ex.exit_reason = "take-profit reached"
    slot.exit = RestingExit(ticker=TICKER, side="YES", price=Decimal("0.83"),
                            contracts=84, deadline=NOW + 8, reason="take-profit reached",
                            order_id="o3")
    fills = [
        {"order_id": "o3", "count_fp": "35.75", "yes_price_dollars": "0.84",
         "fee_cost": "0.336400", "is_taker": True},
        {"order_id": "o3", "count_fp": "6.25", "yes_price_dollars": "0.84",
         "fee_cost": "0.058800", "is_taker": True},
        {"order_id": "o3", "count_fp": "42.00", "yes_price_dollars": "0.84",
         "fee_cost": "0.395200", "is_taker": True},
    ]
    asyncio.run(ex._poll_exit(slot, q, NOW, venue(fills, ("o3",))))
    pnl = Decimal(str(ex.ctx.fills[-1]["pnl"])) if ex.ctx.fills else Decimal("0")
    check("reconcile: BTC 84 @ 0.78 -> 0.84 nets the venue's +$4.25",
          pnl == Decimal("4.2496"), f"pnl={pnl}")

    # The scratch misses: keep the binary, place no taker leg.
    slot, broker = slot_for(quote("0.13", "0.15")), FakeBroker()
    ex = LiveExecutor(ctx(), broker)
    scratch = Decision("scratch", "time barrier 45s", side="NO", price=Decimal("0.83"))
    run(slot, ex, scratch, slot.quote)
    placed_before = len(broker.placed)
    slot.exit.deadline = NOW - 1
    asyncio.run(ex._poll_exit(slot, slot.quote, NOW,
                              venue([], tuple(broker.resting))))     # pulled
    asyncio.run(ex._poll_exit(slot, slot.quote, NOW + 1.3, venue([])))  # seen off the book
    early_hold = slot.position.hold_to_settle
    asyncio.run(ex._poll_exit(slot, slot.quote, NOW + 4.4, venue([])))  # and settled
    check("reconcile: an unfilled scratch carries the binary and fires no IOC",
          slot.position is not None and slot.position.hold_to_settle
          and not early_hold and len(broker.placed) == placed_before,
          f"hold={slot.position.hold_to_settle if slot.position else None} "
          f"placed={len(broker.placed)}")

    # Settlement books 1/0, at the position's own contract, through Engine._roll_slot.
    class FakeStore:
        def __init__(self) -> None:
            self.ticks: list[dict] = []
            self.trades: list[dict] = []

        def tick(self, **kw):
            self.ticks.append(kw)

        def trade(self, **kw):
            self.trades.append(kw)

        def fill(self, **kw):
            pass

    store = FakeStore()
    engine = Engine.__new__(Engine)
    engine.cash = Decimal("0")
    engine.store = store
    engine.broker = None
    engine.slots = []
    engine.venue_balance = Decimal("0")
    engine._venue_flat = True
    engine._last_window_sync = 0.0
    engine.settings = SimpleNamespace(market_poll_s=3.0)
    engine.executor = Executor(engine)
    engine.executor.ctx = SimpleNamespace(cash=Decimal("0"), realized=Decimal("0"),
                                          fills=[], store=store, supervisor=None,
                                          source="live")
    engine.supervisor = SimpleNamespace(triggered=lambda slot: None)
    old = quote("0.13", "0.15")
    slot = slot_for(old)
    slot.position = Position(ticker=TICKER, side="YES", entry=Decimal("0.77"),
                             contracts=85, opened_ts=NOW - 900, p_entry=0.8,
                             stake=Decimal("65.45"), tp=Decimal("0.82"),
                             sl=Decimal("0.52"), signal="z_obi_flow",
                             hold_to_settle=True)
    slot.lock_in_settlement = (TICKER, 68.0, 70.0)     # 68 < 70: YES loses
    slot.window_entries = 1
    asyncio.run(types.MethodType(Engine._roll_slot, engine)(slot, old, NOW))
    settled = Decimal(str(store.trades[-1]["pnl"])) if store.trades else Decimal("0")
    check("reconcile: a carried 85-lot YES settles at 0, -$65.45 exactly",
          settled == Decimal("-65.45") and slot.position is None,
          f"pnl={settled}")

    # The promoted program bounds the tail with the cost band, not a stop: a 0.61
    # entry is refused, 0.60 is the widest allowed, and a carried contract can lose
    # no more than the 0.60 it cost. This replaced the earlier 0.55 cap / 6c stop
    # checks when the frozen study promoted the 0.30-0.60 carry rule.
    params = Params()
    # Re-aligned 2026-11-16 to the verified detector: the band is 0.15-0.84 and the
    # window is 90-420s left, so the refusal is tested at 0.85 inside that window.
    q61 = quote("0.85", "0.86")
    too_high = decide(quote=q61, p_up=0.95, seconds_left=300.0, spot_age=0.1,
                      params=params, position=None, edge_hurdle=Decimal("0"),
                      can_open=True, signal="z_obi_flow", now=NOW, spot=70.0)
    check("tail: a 0.85 flow entry is refused by the 0.15-0.84 cost band",
          too_high.action != "entry" and "band" in too_high.reason,
          f"{too_high.action}: {too_high.reason}")
    in_band = decide(quote=quote("0.84", "0.85"), p_up=0.95, seconds_left=300.0, spot_age=0.1,
                     params=params, position=None, edge_hurdle=Decimal("0"),
                     can_open=True, signal="z_obi_flow", now=NOW, spot=70.0)
    early = decide(quote=quote("0.50", "0.51"), p_up=0.62, seconds_left=600.0, spot_age=0.1,
                   params=params, position=None, edge_hurdle=Decimal("0"),
                   can_open=True, signal="z_obi_flow", now=NOW, spot=70.0)
    check("tail: 0.84 is allowed, and 600s left is outside the 90-420s flow window",
          in_band.action == "entry" and early.action != "entry",
          f"in_band={in_band.action} early={early.action}: {early.reason}")
    held = Position(ticker=TICKER, side="YES", entry=Decimal("0.60"), contracts=60,
                    opened_ts=NOW - 60, p_entry=0.8, stake=Decimal("36.00"),
                    tp=Decimal("0.65"), sl=Decimal("0.35"), signal="z_obi_flow")
    carried = decide(quote=quote("0.48", "0.49"), p_up=0.8, seconds_left=300.0,
                     spot_age=0.1, params=params, position=held,
                     edge_hurdle=Decimal("0"), can_open=False, signal="z_obi_flow",
                     now=NOW, spot=70.0)
    # Deployed frozen behaviour (strategy-verdict.json, exit_mode=maker_scratch,
    # barrier_s=45): once the barrier elapsed the lane rests a post-only scratch at
    # breakeven (px >= entry, so the round trip can never lose more than the entry cost),
    # and only an unfilled scratch carries the binary to settlement. zobi_hold_to_settle
    # is False in the promoted program.
    check("tail: a held flow contract rests a scratch at breakeven; max loss is its entry cost (band cap 0.84)",
          carried.action == "scratch" and params.zobi_max_price == Decimal("0.84")
          and params.zobi_max_loss == 0, f"{carried.action}: {carried.reason}")


# --------------------------------------------------------------------------
# 3. Fail-safe: an exception while live cannot leave a position unguarded.
# --------------------------------------------------------------------------
def control_fail_safe() -> None:
    q = quote("0.13", "0.15")
    slot, broker = slot_for(q), FakeBroker(fill_qty=10.0, fill_price="0.14")
    slot.exit = None
    ex = LiveExecutor(ctx(), broker)
    broker.resting["stale"] = {"order_id": "stale", "ticker": TICKER}
    flat = asyncio.run(ex.emergency_flatten(slot, "fail-safe NameError"))
    sent = broker.placed[-1] if broker.placed else {}
    check("fail-safe: resting orders on the ticker are cancelled first",
          "stale" in broker.cancelled, f"cancelled={broker.cancelled}")
    check("fail-safe: the flatten is IOC, reduce_only and not post-only",
          sent.get("tif") == "immediate_or_cancel" and sent.get("reduce_only") is True
          and sent.get("post_only") is False,
          f"sent={ {k: sent.get(k) for k in ('tif', 'reduce_only', 'post_only')} }")
    check("fail-safe: a filled flatten leaves the slot flat", flat and slot.position is None,
          f"flat={flat} position={slot.position}")

    # A rejected flatten must report open, never silently claim flat.
    slot, broker = slot_for(q), FakeBroker(reject_all=True)
    ex = LiveExecutor(ctx(), broker)
    rejected = asyncio.run(ex.emergency_flatten(slot, "fail-safe"))
    check("fail-safe: a rejected IOC reports the position still open",
          rejected is False and slot.position is not None, f"flat={rejected}")

    # The engine wrapper must invoke it when the lane body throws.
    calls: list[str] = []

    class FakeLive:
        async def emergency_flatten(self, slot, reason="fail-safe"):
            calls.append(reason)
            slot.position = None
            return True

    notes: list[tuple[str, str]] = []
    engine = Engine.__new__(Engine)
    engine.fault_times = []
    engine.live_armed = True
    engine.live = FakeLive()
    engine.supervisor = SimpleNamespace(note=lambda k, v: notes.append((k, v)))
    slot = SimpleNamespace(cfg=SimpleNamespace(series="KXSOL15M"), position=object(),
                           order=None, halted=False, window_halted=False,
                           reason="z+flow working")

    async def boom(slot, now, venue=None):
        raise NameError("name 'ROUND_HALF_UP' is not defined")

    engine._step_slot_inner = boom   # type: ignore[assignment]
    raised = False
    try:
        asyncio.run(engine._step_slot(slot, NOW, None))
    except NameError:
        raised = True
    check("fail-safe: a lane exception with a live position triggers the flatten",
          bool(calls) and slot.halted and not slot.position,
          f"calls={calls} halted={slot.halted}")
    check("fail-safe: the fault is on the supervisor tape, not re-raised",
          any(k == "FAIL_SAFE" for k, _ in notes) and not raised,
          f"notes={[k for k, _ in notes]} raised={raised}")


# --------------------------------------------------------------------------
# 4. F2 execution: pinned-ask IOC entry, $0 maker TP, 2s-confirmed stop.
# --------------------------------------------------------------------------
def f2_slot(q: Quote, *, entry: str = "0.47", contracts: int = 40) -> SimpleNamespace:
    """A slot running the frozen F2 rule, with the real Params (not a subset)."""
    params = Params()
    pos = Position(ticker=TICKER, side="YES", entry=Decimal(entry), contracts=contracts,
                   opened_ts=NOW - 10, p_entry=0.62, stake=Decimal(entry) * contracts,
                   tp=Decimal(entry) + params.f2_take_profit,
                   sl=Decimal(entry) - params.f2_stop_loss, signal="F2_taker_1s")
    return SimpleNamespace(cfg=SimpleNamespace(series="KXSOL15M"), params=params,
                           bankroll=Decimal("100"), peak_bankroll=Decimal("100"),
                           order=None, position=pos, exit=None, p_up=0.62, spot=70.0,
                           quote=q, window_entries=0, pnl_today=Decimal("0"),
                           wins=0, losses=0, consecutive_losses=0, ece=0.0,
                           pairs=[], last_trade=None, window_halted=False, halted=False)


def control_f2_execution() -> None:
    entry = Decision("entry", "F2 IOC at the YES ask", side="YES", price=Decimal("0.47"),
                     p_model=0.62, signal="F2_taker_1s")

    # (a) The limit is the best ask, not a price above it, and it is an IOC.
    q = quote("0.45", "0.47")
    slot, broker = f2_slot(q), FakeBroker()
    slot.position = None
    ex = LiveExecutor(ctx(), broker)
    asyncio.run(ex._place_entry(slot, entry, q, NOW, 300.0))
    sent = broker.placed[-1] if broker.placed else {}
    check("F2: the entry is pinned to the best ask as a limit IOC",
          len(broker.placed) == 1 and sent.get("price") == Decimal("0.47")
          and sent.get("tif") == "immediate_or_cancel" and sent.get("post_only") is False,
          f"price={sent.get('price')} tif={sent.get('tif')} post_only={sent.get('post_only')}")
    check("F2: the resting order records its IOC time-in-force",
          slot.order is not None and slot.order.tif == "immediate_or_cancel",
          f"tif={getattr(slot.order, 'tif', None)}")

    # (b) The ask moves up between the decision and the order: the pin follows the
    #     book (the study priced arrival, T+1s) and never exceeds it.
    slot, broker = f2_slot(quote("0.47", "0.49")), FakeBroker()
    slot.position = None
    ex = LiveExecutor(ctx(), broker)
    asyncio.run(ex._place_entry(slot, entry, slot.quote, NOW, 300.0))
    sent = broker.placed[-1] if broker.placed else {}
    check("F2: a higher ask is paid at the ask, never above it",
          sent.get("price") == Decimal("0.49"), f"price={sent.get('price')}")

    # (c) The ask moves down: paying the decided price would be the upward walk the
    #     program is measured without, so no order is sent.
    slot, broker = f2_slot(quote("0.43", "0.45")), FakeBroker()
    slot.position = None
    ex = LiveExecutor(ctx(), broker)
    asyncio.run(ex._place_entry(slot, entry, slot.quote, NOW, 300.0))
    check("F2: a lower ask refuses the stale, higher limit (zero upward walk)",
          not broker.placed and slot.order is None and "no upward price walk" in (ex.note or ""),
          f"placed={len(broker.placed)} note={ex.note!r}")

    # (d) A NO entry pins the NO ask, which is `1 - best bid`.
    no_entry = Decision("entry", "F2 IOC at the NO ask", side="NO", price=Decimal("0.55"),
                        p_model=0.38, signal="F2_taker_1s")
    slot, broker = f2_slot(quote("0.45", "0.47")), FakeBroker()
    slot.position = None
    ex = LiveExecutor(ctx(), broker)
    asyncio.run(ex._place_entry(slot, no_entry, slot.quote, NOW, 300.0))
    sent = broker.placed[-1] if broker.placed else {}
    check("F2: a NO entry is pinned to 1 - best bid, not to the YES ask",
          sent.get("price") == Decimal("0.55") and sent.get("side") == "NO",
          f"price={sent.get('price')} side={sent.get('side')}")

    # (e) A partial IOC books the whole contracts the venue delivered, floored.
    slot, broker = f2_slot(quote("0.45", "0.47"), contracts=40), FakeBroker()
    slot.position = None
    slot.order = RestingOrder(ticker=TICKER, side="YES", price=Decimal("0.47"),
                              contracts=40, placed_ts=NOW, reserved=Decimal("18.80"),
                              signal="F2_taker_1s", order_id="f1",
                              tif="immediate_or_cancel")
    ex = LiveExecutor(ctx(), broker)
    ex.ctx.cash = Decimal("81.20")
    part = venue([{"order_id": "f1", "count_fp": "12.40", "yes_price_dollars": "0.47",
                   "fee_cost": "0.1745", "is_taker": True}])
    asyncio.run(ex._poll_entry(slot, slot.quote, NOW, part))
    early = slot.position
    # The order is off the book (an IOC cannot rest), so the total is booked once it
    # has stopped moving for the settle window -- 12 whole contracts, not 1 and not 40.
    asyncio.run(ex._poll_entry(slot, slot.quote, NOW + 3.2, part))
    check("F2: a 12.40-of-40 IOC print settles into 12 contracts at the fill",
          early is None and slot.position is not None and slot.position.contracts == 12
          and slot.position.entry == Decimal("0.4700") and slot.order is None,
          f"early={early} position={slot.position.contracts if slot.position else None}")
    check("F2: the partial entry carries the venue's fee into the position",
          slot.position is not None and slot.position.entry_fee == Decimal("0.1745"),
          f"entry_fee={slot.position.entry_fee if slot.position else None}")

    # (f) An IOC that prints nothing is finished within the fill-settle window, so
    #     the lane can act on the next window instead of waiting a minute.
    slot, broker = f2_slot(quote("0.45", "0.47"), contracts=40), FakeBroker()
    slot.position = None
    slot.order = RestingOrder(ticker=TICKER, side="YES", price=Decimal("0.47"),
                              contracts=40, placed_ts=NOW, reserved=Decimal("18.80"),
                              signal="F2_taker_1s", order_id="f2",
                              tif="immediate_or_cancel")
    ex = LiveExecutor(ctx(), broker)
    asyncio.run(ex._poll_entry(slot, slot.quote, NOW + 3.2, venue([])))
    check("F2: an unfilled IOC clears in 3s instead of blocking the lane for 60",
          slot.order is None and slot.position is None, f"order={slot.order}")

    # (g) The take-profit rests immediately, post-only, at entry_fill + 0.05.
    q = quote("0.45", "0.47")
    slot, broker = f2_slot(q, entry="0.47"), FakeBroker()
    ex = LiveExecutor(ctx(), broker)
    asyncio.run(ex._rest_take_profit(
        slot, Decision("hold", "F2 working", side="YES"), q, NOW))
    sent = broker.placed[-1] if broker.placed else {}
    check("F2: the take-profit is a post-only ask at entry_fill + 0.05",
          len(broker.placed) == 1 and sent.get("price") == Decimal("0.52")
          and sent.get("post_only") is True and slot.exit.retry_on_timeout,
          f"price={sent.get('price')} post_only={sent.get('post_only')}")

    # (h) The stop is confirmed for two seconds, then crosses reduce-only at the bid.
    pos = slot.position
    pos.entry, pos.tp = Decimal("0.47"), Decimal("0.52")
    down = quote("0.39", "0.41")            # bid 0.39 <= entry - 0.07 = 0.40
    params = Params()
    first = decide(quote=down, p_up=0.3, seconds_left=300.0, spot_age=0.1,
                   params=params, position=pos, edge_hurdle=Decimal("0"),
                   can_open=False, signal="F2_taker_1s", now=NOW, spot=70.0)
    mid = decide(quote=down, p_up=0.3, seconds_left=300.0, spot_age=0.1,
                 params=params, position=pos, edge_hurdle=Decimal("0"),
                 can_open=False, signal="F2_taker_1s", now=NOW + 1.9, spot=70.0)
    fired = decide(quote=down, p_up=0.3, seconds_left=300.0, spot_age=0.1,
                   params=params, position=pos, edge_hurdle=Decimal("0"),
                   can_open=False, signal="F2_taker_1s", now=NOW + 2.1, spot=70.0)
    check("F2: one wick through the 7c stop is not a stop; 2s of breach is",
          first.action == "hold" and mid.action == "hold" and fired.action == "exit_sl"
          and fired.price == Decimal("0.39"),
          f"{first.action}/{mid.action}/{fired.action} price={fired.price}")
    slot.exit = None
    asyncio.run(ex._decide_exit(slot, fired, down, 300.0, NOW + 2.1))
    sent = broker.placed[-1] if broker.placed else {}
    check("F2: the confirmed stop is a reduce-only IOC at the best bid",
          sent.get("price") == Decimal("0.39") and sent.get("reduce_only") is True
          and sent.get("post_only") is False and sent.get("tif") == "immediate_or_cancel",
          f"price={sent.get('price')} reduce_only={sent.get('reduce_only')}")


def control_paper_f2() -> None:
    """The paper executor must book the same F2 entry/TP/stop as the live rule."""
    q = quote("0.45", "0.47")
    entry = Decision("entry", "F2 IOC at the YES ask", side="YES", price=Decimal("0.47"),
                     p_model=0.62, signal="F2_taker_1s")
    slot = f2_slot(q)
    slot.position = None
    slot.window_entries = 0
    ex = Executor(ctx())
    ex.ctx.cash = Decimal("1000")
    ex.place_entry(slot, entry, q, NOW, 300.0)
    pos = slot.position
    check("paper F2: taker entry books the ask, +5c TP and one window entry",
          pos is not None and pos.entry == Decimal("0.47")
          and pos.tp == Decimal("0.52") and pos.signal == "F2_taker_1s"
          and pos.entry_fee > 0 and slot.window_entries == 1,
          f"pos={pos and (pos.entry, pos.tp, pos.entry_fee)} entries={slot.window_entries}")

    up = quote("0.52", "0.54")
    ex.execute(slot, Decision("exit_tp", "take-profit reached", side="YES",
                              price=Decimal("0.52")), up, 300.0, NOW + 1.0)
    fill = ex.ctx.fills[-1] if ex.ctx.fills else {}
    check("paper F2: the +5c take-profit closes maker at entry+0.05 with no fee",
          slot.position is None and fill.get("price") == "0.52"
          and Decimal(str(fill.get("fee", "0"))) == 0,
          f"price={fill.get('price')} fee={fill.get('fee')} position={slot.position}")

    slot = f2_slot(q)
    ex = Executor(ctx())
    ex.ctx.cash = Decimal("1000")
    down = quote("0.39", "0.41")
    ex.execute(slot, Decision("exit_sl", "F2 confirmed stop", side="YES",
                              price=Decimal("0.39")), down, 300.0, NOW + 1.0)
    fill = ex.ctx.fills[-1] if ex.ctx.fills else {}
    check("paper F2: the confirmed stop closes taker at the best bid, not as a maker rest",
          slot.position is None and slot.exit is None and fill.get("price") == "0.39"
          and Decimal(str(fill.get("fee", "0"))) > 0,
          f"price={fill.get('price')} fee={fill.get('fee')} exit={slot.exit}")


def control_sizing() -> None:
    """The live lane may not buy more than the study scored, or risk more than $15.

    The 16:00 trade that lost $42.97: a 24c NO on a $41.11 lane sized 170
    contracts ($40.80 of premium). `forward_oos.py` scored 50; the stop cannot
    fill at a 0.00 bid, so the premium, not the 7c stop, is the real risk.
    """
    params = Params()
    qty = sized_contracts(Decimal("41.11"), Decimal("0.24"), params)
    premium = Decimal("0.24") * qty
    check("sizing: a 24c fill on a $41.11 lane is capped at 50 contracts",
          qty <= 50, f"qty={qty} (the live fill was 170)")
    check("sizing: worst-case premium on that fill stays at or under $15",
          premium <= Decimal("15.00"), f"premium={premium} qty={qty}")
    mid = sized_contracts(Decimal("41.11"), Decimal("0.50"), params)
    check("sizing: a 50c entry is bounded by the $15 risk cap, not the slot",
          Decimal("0.50") * mid <= Decimal("15.00"), f"qty={mid} premium={Decimal('0.50') * mid}")


def control_compounding() -> None:
    """Staged fractional-Kelly sizing: six tiers, six ceilings, two hard bounds.

    The 16:00 trade lost $42.97 because a $41 lane bought 170 contracts; the fix was
    a flat $15 / 50-contract cap, which by design cannot compound. These checks pin
    the stage table that replaces it: the cap still binds at stage 1, stages 2-6
    lift the contract ceiling as the account grows (so a validated lane can compound
    to $1,000, $10,000 and $300,000 rather than stalling at 250 contracts), and
    neither the $0.62 entry ceiling nor the 20% trailing-peak breaker is ever
    stage-dependent.

    A hold-to-settlement binary loses its whole premium, so no stage may risk 20% of
    equity: the fractions are 10/12/12/12/10/8%, and `sized_contracts` clips that
    again to half the remaining distance to `0.80 * peak`. At a fresh high-water mark
    the binding cap is therefore 10% of equity, which is what most of these numbers
    are computed from.
    """
    tiers = ((Decimal("121.46"), "stage1", Decimal("0.10"), 50),
             (Decimal("174.99"), "stage1", Decimal("0.10"), 50),
             (Decimal("175.00"), "stage2", Decimal("0.12"), 120),
             (Decimal("349.99"), "stage2", Decimal("0.12"), 120),
             (Decimal("350.00"), "stage3", Decimal("0.12"), 300),
             (Decimal("999.99"), "stage3", Decimal("0.12"), 300),
             (Decimal("1000.00"), "stage4", Decimal("0.12"), 1_000),
             (Decimal("4999.99"), "stage4", Decimal("0.12"), 1_000),
             (Decimal("5000.00"), "stage5", Decimal("0.10"), 2_500),
             (Decimal("24999.99"), "stage5", Decimal("0.10"), 2_500),
             (Decimal("25000.00"), "stage6", Decimal("0.08"), 5_000),
             (Decimal("300000.00"), "stage6", Decimal("0.08"), 5_000))
    for equity, name, pct, ceiling in tiers:
        stage = stage_for(equity)
        check(f"compounding: ${equity} resolves to {name} ({pct * 100:g}%, {ceiling} contracts)",
              stage.name == name and stage.max_risk_pct == pct
              and stage.max_contracts == ceiling,
              f"{stage.name} {stage.max_risk_pct} {stage.max_contracts}")

    check("compounding: stage 1 at $121.46 risks $12.14, 10% of equity",
          staged_risk_cap(Decimal("121.46")) == Decimal("12.14"),
          f"cap={staged_risk_cap(Decimal('121.46'))}")
    check("compounding: stages 2-6 risk 12%, 12%, 12%, 10% and 8% of equity",
          staged_risk_cap(Decimal("175.00")) == Decimal("21.00")
          and staged_risk_cap(Decimal("350.00")) == Decimal("42.00")
          and staged_risk_cap(Decimal("1000.00")) == Decimal("120.00")
          and staged_risk_cap(Decimal("5000.00")) == Decimal("500.00")
          and staged_risk_cap(Decimal("25000.00")) == Decimal("2000.00"),
          f"{staged_risk_cap(Decimal('175.00'))} / {staged_risk_cap(Decimal('350.00'))} / "
          f"{staged_risk_cap(Decimal('25000.00'))}")
    check("compounding: no stage allows 20% of equity at risk",
          all(stage.max_risk_pct < Decimal("0.20") for stage in COMPOUND_STAGES)
          and max(stage.max_risk_pct for stage in COMPOUND_STAGES) <= Decimal("0.12"),
          f"{[str(s.max_risk_pct) for s in COMPOUND_STAGES]}")

    # The stage's contract ceiling still binds at stage 1, while the buffer clip
    # binds before the stage fraction everywhere the account is near its peak.
    params = Params()
    s1 = sized_contracts(Decimal("30.00"), Decimal("0.24"), params, equity=Decimal("121.46"),
                         peak_equity=Decimal("121.46"))
    check("compounding: a 24c entry at $121.46 is still capped at 50 contracts",
          s1 == 50 and Decimal("0.24") * s1 <= Decimal("12.14"), f"qty={s1}")
    s2 = sized_contracts(Decimal("50.00"), Decimal("0.30"), params, equity=Decimal("200.00"),
                         peak_equity=Decimal("200.00"))
    check("compounding: stage 2 at a fresh peak is clipped to half its buffer, not 12%",
          s2 == 66 and Decimal("0.30") * s2 <= Decimal("20.00"), f"qty={s2}")
    s3 = sized_contracts(Decimal("87.50"), Decimal("0.30"), params, equity=Decimal("350.00"),
                         peak_equity=Decimal("350.00"))
    check("compounding: stage 3 at $350 risks $34.80, not the $42 its 12% allows",
          s3 == 116 and Decimal("0.30") * s3 <= Decimal("35.00"), f"qty={s3}")
    drawn = sized_contracts(Decimal("400"), Decimal("0.30"), params,
                            equity=Decimal("300.00"), peak_equity=Decimal("350.00"))
    check("compounding: 14% under the peak, only half the $20 buffer may be risked",
          drawn == 33 and Decimal("0.30") * drawn <= Decimal("10.00"), f"qty={drawn}")
    cheap = sized_contracts(Decimal("400"), Decimal("0.02"), params,
                            equity=Decimal("350.00"), peak_equity=Decimal("350.00"))
    check("compounding: the 300-contract stage 3 ceiling binds on a cheap book",
          cheap == 300, f"qty={cheap}")

    # Stages 4-6: the ladder keeps lifting the ceiling instead of hitting a 250 wall.
    s4 = sized_contracts(Decimal("1000"), Decimal("0.30"), params, equity=Decimal("1000.00"),
                         peak_equity=Decimal("1000.00"))
    check("compounding: stage 4 risks $100 of a $1,000 account (333, not 833)",
          s4 == 333, f"qty={s4}")
    s5 = sized_contracts(Decimal("5000"), Decimal("0.30"), params, equity=Decimal("5000.00"),
                         peak_equity=Decimal("5000.00"))
    check("compounding: stage 5 sizes to 10% of equity, under its 2,500 ceiling",
          s5 == 1666 and Decimal("0.30") * s5 <= Decimal("500.00"), f"qty={s5}")
    s6 = sized_contracts(Decimal("25000"), Decimal("0.30"), params, equity=Decimal("25000.00"),
                         peak_equity=Decimal("25000.00"))
    check("compounding: stage 6 binds its 5,000-contract ceiling at 8% of equity",
          s6 == 5000 and Decimal("0.30") * s6 <= Decimal("2000.00"), f"qty={s6}")

    boundaries = [
        (Decimal("174.99"), "stage1"), (Decimal("175.00"), "stage2"),
        (Decimal("999.99"), "stage3"), (Decimal("1000.00"), "stage4"),
        (Decimal("4999.99"), "stage4"), (Decimal("5000.00"), "stage5"),
        (Decimal("24999.99"), "stage5"), (Decimal("25000.00"), "stage6"),
        (Decimal("300000.00"), "stage6"),
    ]
    wrong = [(str(eq), stage_for(eq).name, want) for eq, want in boundaries
             if stage_for(eq).name != want]
    check("compounding: the six stages break exactly at 175/350/1k/5k/25k", not wrong,
          f"wrong={wrong}")
    sized_by_risk = sized_contracts(Decimal("100.00"), Decimal("0.47"), params,
                                    equity=Decimal("200.00"),
                                    peak_equity=Decimal("200.00"))
    check("compounding: the drawdown buffer binds before the contract ceiling",
          sized_by_risk == 42 and Decimal("0.47") * sized_by_risk <= Decimal("20.00"),
          f"qty={sized_by_risk}")

    # The invariant itself: a total-premium loss can never cross the floor.
    tail = []
    for equity, peak in ((Decimal("200.00"), Decimal("200.00")),
                         (Decimal("190.00"), Decimal("200.00")),
                         (Decimal("165.00"), Decimal("200.00"))):
        qty = sized_contracts(Decimal("1000"), Decimal("0.62"), params,
                              equity=equity, peak_equity=peak)
        premium = Decimal("0.62") * qty
        tail.append((str(equity), qty, str(premium),
                     equity - premium >= peak * Decimal("0.80")))
    check("compounding: a total-premium loss can never cross the 20% floor",
          all(row[3] for row in tail), f"{tail}")

    # $0.62 is tradeable at every stage; one cent above it never is.
    check("compounding: 0.62 is still tradeable at the top stage",
          sized_contracts(Decimal("500"), Decimal("0.62"), params,
                          equity=Decimal("400.00"),
                          peak_equity=Decimal("400.00")) > 0,
          f"qty={sized_contracts(Decimal('500'), Decimal('0.62'), params, equity=Decimal('400'), peak_equity=Decimal('400'))}")
    above = [sized_contracts(Decimal("500"), Decimal("0.63"), params, equity=equity)
             for equity in (Decimal("100"), Decimal("200"), Decimal("400"))]
    check("compounding: 0.63 is refused at every stage",
          above == [0, 0, 0] and HARD_MAX_ENTRY_PRICE == Decimal("0.62"), f"qty={above}")

    check("compounding: 20% below the peak trips the breaker; 19.99% does not",
          peak_drawdown_tripped(Decimal("80.00"), Decimal("100.00"))
          and not peak_drawdown_tripped(Decimal("80.01"), Decimal("100.00"))
          and not peak_drawdown_tripped(Decimal("121.46"), Decimal("0")),
          f"at80={peak_drawdown_tripped(Decimal('80.00'), Decimal('100.00'))}")

    # The live entry path enforces both invariants itself, after the F2 re-pin.
    q = quote("0.60", "0.63")
    slot, broker = f2_slot(q), FakeBroker()
    slot.position = None
    ex = LiveExecutor(ctx(), broker)
    over = Decision("entry", "0.63 ask", side="YES", price=Decimal("0.63"),
                    signal="z_obi_flow")
    asyncio.run(ex._place_entry(slot, over, q, NOW, 300.0))
    check("compounding: the live path refuses a 0.63 entry",
          not broker.placed and slot.order is None and "ceiling" in (ex.note or ""),
          f"placed={len(broker.placed)} note={ex.note!r}")

    down = SimpleNamespace(cash=Decimal("100"), realized=Decimal("0"), fills=[], store=None,
                           supervisor=None, source="live", equity=Decimal("79.99"),
                           peak_equity=Decimal("100.00"))
    q = quote("0.45", "0.47")
    slot, broker = f2_slot(q), FakeBroker()
    slot.position = None
    ex = LiveExecutor(down, broker)
    entry = Decision("entry", "z+flow at 0.47", side="YES", price=Decimal("0.47"),
                     signal="z_obi_flow")
    asyncio.run(ex._place_entry(slot, entry, q, NOW, 300.0))
    check("compounding: a 20% drawdown stops new entries before the order",
          not broker.placed and slot.order is None and "breaker" in (ex.note or ""),
          f"placed={len(broker.placed)} note={ex.note!r}")

    growing = SimpleNamespace(cash=Decimal("100"), realized=Decimal("0"), fills=[], store=None,
                              supervisor=None, source="live", equity=Decimal("200.00"),
                              peak_equity=Decimal("200.00"))
    slot, broker = f2_slot(q), FakeBroker()
    slot.position = None
    slot.bankroll = Decimal("50.00")
    ex = LiveExecutor(growing, broker)
    asyncio.run(ex._place_entry(slot, entry, q, NOW, 300.0))
    check("compounding: stage 2 sizes the live order to half the drawdown buffer",
          slot.order is not None and slot.order.contracts == 42
          and broker.placed and broker.placed[-1]["contracts"] == 42
          and Decimal("0.47") * 42 <= Decimal("20.00"),
          f"contracts={slot.order.contracts if slot.order else None}")


def control_depth_clipping() -> None:
    """An order may take at most half the size resting at the ask it would pay."""
    check("depth: the clip fraction is 50% of the resting touch",
          DEPTH_CLIP_FRACTION == Decimal("0.50"), f"fraction={DEPTH_CLIP_FRACTION}")
    check("depth: a YES entry clips to half the resting YES ask size",
          depth_clipped(85, "YES", quote("0.45", "0.47", ask_sz="40")) == 20,
          f"qty={depth_clipped(85, 'YES', quote('0.45', '0.47', ask_sz='40'))}")
    check("depth: a NO entry clips to half the resting YES bid (its NO ask)",
          depth_clipped(85, "NO", quote("0.45", "0.47", bid_sz="12")) == 6,
          f"qty={depth_clipped(85, 'NO', quote('0.45', '0.47', bid_sz='12'))}")
    check("depth: an odd touch rounds down, never up",
          depth_clipped(85, "YES", quote("0.45", "0.47", ask_sz="7")) == 3,
          f"qty={depth_clipped(85, 'YES', quote('0.45', '0.47', ask_sz='7'))}")
    check("depth: a one-lot touch supports no order at all",
          depth_clipped(85, "YES", quote("0.45", "0.47", ask_sz="1")) == 0,
          f"qty={depth_clipped(85, 'YES', quote('0.45', '0.47', ask_sz='1'))}")
    check("depth: an unknown touch size clips nothing",
          depth_clipped(85, "YES", quote("0.45", "0.47")) == 85
          and depth_clipped(85, "NO", quote("0.45", "0.47")) == 85,
          f"qty={depth_clipped(85, 'YES', quote('0.45', '0.47'))}")
    check("depth: a clip never grows a small order",
          depth_clipped(4, "YES", quote("0.45", "0.47", ask_sz="400")) == 4,
          f"qty={depth_clipped(4, 'YES', quote('0.45', '0.47', ask_sz='400'))}")

    # The live entry path applies the clip, and a thin book refuses the entry
    # instead of sending a zero-lot order.
    thin = quote("0.45", "0.47", bid_sz="60", ask_sz="9")
    slot, broker = f2_slot(thin), FakeBroker()
    slot.position = None
    slot.bankroll = Decimal("50.00")
    ex = LiveExecutor(SimpleNamespace(cash=Decimal("50"), realized=Decimal("0"), fills=[],
                                      store=None, supervisor=None, source="live",
                                      equity=Decimal("350.00"),
                                      peak_equity=Decimal("350.00")), broker)
    entry = Decision("entry", "z+flow at 0.47", side="YES", price=Decimal("0.47"),
                     signal="z_obi_flow")
    asyncio.run(ex._place_entry(slot, entry, thin, NOW, 300.0))
    check("depth: the live order is clipped to 4 of the 9 resting at the ask",
          slot.order is not None and slot.order.contracts == 4
          and broker.placed and broker.placed[-1]["contracts"] == 4,
          f"contracts={slot.order.contracts if slot.order else None}")

    slot, broker = f2_slot(quote("0.45", "0.47", ask_sz="1")), FakeBroker()
    slot.position = None
    slot.bankroll = Decimal("50.00")
    ex = LiveExecutor(ctx(), broker)
    asyncio.run(ex._place_entry(slot, entry, slot.quote, NOW, 300.0))
    check("depth: a one-lot book refuses the order rather than sizing it to zero",
          not broker.placed and slot.order is None and "too thin" in (ex.note or ""),
          f"placed={len(broker.placed)} note={ex.note!r}")

    # The clip is a size cap, not a price exception: the 0.62 ceiling and the 20%
    # breaker still refuse a deep book that prices above them.
    deep = quote("0.61", "0.63", ask_sz="400")
    slot, broker = f2_slot(deep), FakeBroker()
    slot.position = None
    ex = LiveExecutor(ctx(), broker)
    asyncio.run(ex._place_entry(slot, Decision("entry", "0.63 ask", side="YES",
                                              price=Decimal("0.63"),
                                              signal="z_obi_flow"), deep, NOW, 300.0))
    check("depth: deep liquidity does not lift the $0.62 entry ceiling",
          not broker.placed and slot.order is None, f"placed={len(broker.placed)}")


def main() -> int:
    tp = Decision("exit_tp", "take-profit reached", side="NO", price=Decimal("0.88"))

    # 1. The book already pays more than the take-profit. A post-only sell at the
    #    target would cross (the venue rejects it), so the ask is rested instead:
    #    $0 fee if it is lifted, and the bid is taken only if nobody lifts it.
    q = quote("0.07", "0.08")          # NO bid = 0.92 >= 0.88, NO ask = 0.93
    slot, broker = slot_for(q), FakeBroker()
    ex = LiveExecutor(ctx(), broker)
    run(slot, ex, tp, q)
    sent = broker.placed[-1] if broker.placed else {}
    check("exit: a take-profit below the bid rests post-only at the ask first",
          len(broker.placed) == 1 and sent.get("post_only") is True
          and sent.get("price") == Decimal("0.93") and slot.exit.take_on_timeout,
          f"placed={len(broker.placed)} post_only={sent.get('post_only')} price={sent.get('price')}")
    lifted = venue([{"order_id": "ord1", "count_fp": "10.00", "no_price_dollars": "0.93",
                     "yes_price_dollars": "0.07", "fee_cost": "0.0", "is_taker": False}])
    asyncio.run(ex._poll_exit(slot, q, NOW + 1, lifted))
    pnl = Decimal(str(ex.ctx.fills[-1]["pnl"])) if ex.ctx.fills else Decimal("0")
    check("exit: lifted at the ask, it closes as a maker with no fee",
          slot.position is None and pnl == Decimal("1.0000") and ex.ctx.fills[-1]["fee"] == 0,
          f"pnl={pnl} fee={ex.ctx.fills[-1]['fee'] if ex.ctx.fills else None}")

    # 1b. Nobody lifts it: pulled, seen off the book, then the bid is taken (IOC).
    slot, broker = slot_for(q), FakeBroker()
    ex = LiveExecutor(ctx(), broker)
    run(slot, ex, tp, q)
    asyncio.run(ex._poll_exit(slot, q, NOW + 3.1, venue([], tuple(broker.resting))))
    asyncio.run(ex._poll_exit(slot, q, NOW + 3.6, venue([], tuple(broker.resting) or ("ord1",))))
    asyncio.run(ex._poll_exit(slot, q, NOW + 4.4, venue([])))
    waiting = (len(broker.placed), slot.position is not None)
    asyncio.run(ex._poll_exit(slot, q, NOW + 7.5, venue([])))
    sent = broker.placed[-1]
    pnl = Decimal(str(ex.ctx.fills[-1]["pnl"])) if ex.ctx.fills else Decimal("0")
    check("exit: an unlifted gap take-profit is pulled, then taken as a reduce-only IOC",
          broker.cancelled == ["ord1"] and waiting == (1, True)  # one cancel, no early IOC
          and sent.get("post_only") is False and sent.get("reduce_only") is True
          and slot.position is None,
          f"cancelled={broker.cancelled} waiting={waiting} post_only={sent.get('post_only')}")
    # 0.92 - 0.83 = 9c on 10 contracts, minus the exact taker fee.
    check("exit: the booked P&L is net of the taker fee",
          pnl > Decimal("0.80") and pnl < Decimal("0.90"),
          f"pnl={pnl} (gross 0.90)")

    # 1c. The bid falls back under the target while it rests: no taker leg, and the
    #     position goes back to the decision loop.
    slot, broker = slot_for(q), FakeBroker()
    ex = LiveExecutor(ctx(), broker)
    run(slot, ex, tp, q)
    back = quote("0.13", "0.15")       # NO bid = 0.85 < 0.88
    asyncio.run(ex._poll_exit(slot, back, NOW + 3.1, venue([], tuple(broker.resting))))
    asyncio.run(ex._poll_exit(slot, back, NOW + 4.4, venue([])))
    asyncio.run(ex._poll_exit(slot, back, NOW + 7.5, venue([])))
    check("exit: if the bid retreats under the target, nothing is taken",
          len(broker.placed) == 1 and slot.position is not None and slot.exit is None,
          f"placed={len(broker.placed)} exit={slot.exit}")

    # 1c'. Bid at 0.99: nothing above it to rest at, so the bid is taken at once.
    top = quote("0.00", "0.01")        # NO bid = 0.99, NO ask = 1.00
    slot, broker = slot_for(top), FakeBroker()
    ex = LiveExecutor(ctx(), broker)
    run(slot, ex, tp, top)
    check("exit: at a 0.99 bid the take-profit goes straight out as an IOC",
          len(broker.placed) == 1 and broker.placed[0]["post_only"] is False,
          f"placed={len(broker.placed)} post_only={broker.placed[0]['post_only'] if broker.placed else None}")

    # 1c". A stop pulls the resting take-profit instead of waiting out the barrier.
    q2 = quote("0.15", "0.16")
    slot, broker = slot_for(q2), FakeBroker()
    slot.position.opened_ts = NOW - 5
    ex = LiveExecutor(ctx(), broker)
    asyncio.run(ex._rest_take_profit(slot, Decision("hold", "z+flow working", side="NO"), q2, NOW))
    run(slot, ex, Decision("exit_sl", "confirmed stop", side="NO", price=Decimal("0.84")), q2)
    asyncio.run(ex._poll_exit(slot, q2, NOW, venue([], tuple(broker.resting))))
    check("exit: a stop decision pulls the resting take-profit at once",
          slot.exit is not None and slot.exit.deadline <= NOW and broker.cancelled == ["ord1"],
          f"deadline_now={slot.exit.deadline <= NOW if slot.exit else None} cancelled={broker.cancelled}")

    # 1d. A crossed or locked book never gets the resting ask.
    for label, cq in (("crossed", quote("0.60", "0.55")), ("locked", quote("0.08", "0.08"))):
        slot, broker = slot_for(cq), FakeBroker()
        ex = LiveExecutor(ctx(), broker)
        asyncio.run(ex._rest_take_profit(
            slot, Decision("hold", "z+flow working", side="NO"), cq, NOW - 40))
        check(f"exit: no take-profit is rested on a {label} book",
              not broker.placed, f"placed={len(broker.placed)}")

    # 1e. From the fill, the take-profit rests as a maker ask at entry+5c to the barrier.
    q = quote("0.15", "0.16")          # NO bid = 0.84 < 0.88
    slot, broker = slot_for(q), FakeBroker()
    slot.position.opened_ts = NOW - 5
    ex = LiveExecutor(ctx(), broker)
    hold = Decision("hold", "z+flow working", side="NO")
    asyncio.run(ex._rest_take_profit(slot, hold, q, NOW))
    asyncio.run(ex._rest_take_profit(slot, hold, q, NOW + 1))
    sent = broker.placed[-1] if broker.placed else {}
    check("exit: the flow take-profit rests post-only at the target from the fill, once",
          len(broker.placed) == 1 and sent.get("post_only") is True
          and sent.get("price") == Decimal("0.88") and slot.exit.retry_on_timeout
          and slot.exit.deadline == NOW + 40,
          f"placed={len(broker.placed)} price={sent.get('price')}")
    slot.position.hold_to_settle = True
    slot.exit = None
    asyncio.run(ex._rest_take_profit(slot, hold, q, NOW + 2))
    check("exit: a position carried to settlement gets no new resting take-profit",
          len(broker.placed) == 1, f"placed={len(broker.placed)}")

    # 2. The take-profit is above the bid: it rests post-only, once, and the next
    #    tick does not place a second order or reset the deadline.
    q = quote("0.13", "0.15")          # NO bid = 0.85 < 0.88 -> the target rests above it
    slot, broker = slot_for(q), FakeBroker()
    ex = LiveExecutor(ctx(), broker)
    run(slot, ex, tp, q)
    deadline = slot.exit.deadline if slot.exit else None
    run(slot, ex, tp, q)
    check("exit: a resting take-profit is placed once, post-only",
          len(broker.placed) == 1 and broker.placed[0]["post_only"] is True,
          f"placed={len(broker.placed)} post_only={broker.placed[0]['post_only'] if broker.placed else None}")
    check("exit: a second decision tick does not reset the fallback deadline",
          slot.exit is not None and slot.exit.deadline == deadline and len(broker.placed) == 1,
          f"deadline_moved={bool(slot.exit) and slot.exit.deadline != deadline} placed={len(broker.placed)}")

    # 3. The venue rejects the rest: the marker is cleared so the next tick can retry,
    #    and nothing is left pretending to be in flight.
    q = quote("0.13", "0.15")
    slot, broker = slot_for(q), FakeBroker(post_only_ok=False)
    ex = LiveExecutor(ctx(), broker)
    run(slot, ex, tp, q)
    check("exit: a rejected rest clears the in-flight marker",
          slot.exit is None and bool(ex.note) and "post only" in ex.note,
          f"exit={slot.exit} note={ex.note!r}")

# --------------------------------------------------------------------------
# 10. The LayaGate: a cached verdict read in 0ms, and a filter a promoted
#     strategy can turn on without a code edit.
# --------------------------------------------------------------------------
def gate_engine(gate: LayaGate) -> Engine:
    engine = Engine.__new__(Engine)
    engine.gate = gate
    return engine


def control_layagate_cache() -> None:
    """The entry path reads the gate's opinion from memory, never from the wire.

    A 610-650ms forward pass used to sit on the entry hot path, so every tick
    that wanted a second opinion paid for it in latency. The verdict is now
    cached per ticker and the decision reads that cache.
    """
    gate = LayaGate("http://127.0.0.1:1", mode="shadow")
    verdict = Verdict(True, False, 0.31, "skip: the edge looks like noise", "laya", 640.0)
    verdict.ticker = TICKER
    verdict.ts = time.time()
    gate.cache[TICKER] = verdict
    gate.mid_ref[TICKER] = 0.46

    started = time.perf_counter()
    got = gate.cached(TICKER)
    elapsed_ms = (time.perf_counter() - started) * 1000
    check("gate: the cached verdict is a dictionary read, not a forward pass",
          got is verdict and elapsed_ms < 1.0,
          f"elapsed={elapsed_ms:.4f}ms")

    # The whole entry path, with the HTTP client booby-trapped: a connection
    # attempt here would raise, so a clean return proves nothing was opened.
    class Boom:
        def __init__(self, *a, **kw):
            raise AssertionError("the entry path opened a connection")

    original = decision_gate.httpx.AsyncClient
    decision_gate.httpx.AsyncClient = Boom
    try:
        engine = gate_engine(gate)
        slot = SimpleNamespace(params=Params(require_layagate_allow=True))
        entry = Decision("entry", "kernel entry", side="YES", price=Decimal("0.47"),
                         p_model=0.62, edge=Decimal("0.06"))
        started = time.perf_counter()
        out = engine._gate_veto(slot, quote("0.45", "0.47"), entry)
        elapsed_ms = (time.perf_counter() - started) * 1000
    finally:
        decision_gate.httpx.AsyncClient = original
    check("gate: the entry path reads the cache in 0ms and opens no connection",
          out.action == "abstain" and elapsed_ms < 1.0 and "laya gate" in out.reason,
          f"action={out.action} elapsed={elapsed_ms:.4f}ms reason={out.reason!r}")

    # Staleness: the cache is refreshed on a timer, and sooner when the book moves.
    check("gate: a fresh verdict for this ticker is not stale",
          not gate.stale(TICKER, mid=0.46), f"stale={gate.stale(TICKER, mid=0.46)}")
    check("gate: a 2c mid move makes the cached verdict stale",
          gate.stale(TICKER, mid=0.48), f"stale={gate.stale(TICKER, mid=0.48)}")
    check("gate: a ticker with no verdict is stale",
          gate.stale("KXSOL15M-26NOV161315-00", mid=0.46), "no verdict for that ticker")
    verdict.ts = time.time() - 30.0
    check("gate: a verdict older than the refresh window is stale",
          gate.stale(TICKER, mid=0.46), f"stale={gate.stale(TICKER, mid=0.46)}")


def control_layagate_filter() -> None:
    """`require_layagate_allow` is the switch that makes the gate binding."""
    gate = LayaGate("http://127.0.0.1:1", mode="shadow")
    engine = gate_engine(gate)
    entry = Decision("entry", "kernel entry", side="YES", price=Decimal("0.47"),
                     p_model=0.62, edge=Decimal("0.06"))

    def cached(allow: bool, confidence: float, available: bool = True) -> None:
        verdict = Verdict(available, allow, confidence, "note", "laya", 640.0)
        verdict.ticker = TICKER
        verdict.ts = time.time()
        gate.cache[TICKER] = verdict

    # Off by default: the shipped build has no fine-tuned model, so the gate is a
    # shadow and the kernel's numeric edge decides.
    cached(allow=False, confidence=0.2)
    off = engine._gate_veto(SimpleNamespace(params=Params()), quote("0.45", "0.47"), entry)
    check("gate: with the filter off the kernel's entry stands",
          off.action == "entry", f"action={off.action}")

    # On: the cached opinion is binding, with no code edit anywhere else.
    on = engine._gate_veto(SimpleNamespace(params=Params(require_layagate_allow=True)),
                           quote("0.45", "0.47"), entry)
    check("gate: with the filter on an allow=False verdict refuses the entry",
          on.action == "abstain" and on.regime == "gate" and "laya gate" in on.reason,
          f"action={on.action} reason={on.reason!r}")

    # An allow verdict does not block, and a minimum confidence is honoured.
    cached(allow=True, confidence=0.9)
    allowed = engine._gate_veto(SimpleNamespace(params=Params(require_layagate_allow=True)),
                                quote("0.45", "0.47"), entry)
    check("gate: an allow=True verdict lets the entry through",
          allowed.action == "entry", f"action={allowed.action}")
    cached(allow=True, confidence=0.40)
    weak = engine._gate_veto(SimpleNamespace(
        params=Params(require_layagate_allow=True, layagate_min_confidence=0.50)),
        quote("0.45", "0.47"), entry)
    check("gate: a minimum confidence refuses a weak allow",
          weak.action == "abstain", f"action={weak.action} reason={weak.reason!r}")

    # A gate that is offline never blocks: the kernel's edge decides.
    cached(allow=False, confidence=0.1, available=False)
    offline = engine._gate_veto(SimpleNamespace(params=Params(require_layagate_allow=True)),
                                quote("0.45", "0.47"), entry)
    check("gate: an offline gate never blocks the kernel",
          offline.action == "entry", f"action={offline.action}")


# --------------------------------------------------------------------------
# 11. The settlement program: Student-t value, true-L2 imbalance, signed flow,
#     and the cost ceiling that bounds a carry.
# --------------------------------------------------------------------------
def settlement_quote(bid: str, ask: str) -> Quote:
    return Quote(ticker=TICKER, series="KXSOL15M", strike=70.0, close_ts=NOW + 300,
                 yes_bid=Decimal(bid), yes_ask=Decimal(ask), last=Decimal(bid),
                 updated_ts=NOW, received_ts=NOW,
                 yes_bid_sz=Decimal("500"), yes_ask_sz=Decimal("200"))


def control_settlement_program() -> None:
    params = Params()

    # The Student-t CDF is exact, not a table: it has to reproduce the published
    # 0.975 quantiles or the settlement probability is a lookup error.
    worst = max(abs(features.student_t_cdf(q, nu) - 0.975)
                for nu, q in ((1.0, 12.7062), (2.0, 4.3027), (5.0, 2.5706),
                              (8.0, 2.3060), (10.0, 2.2281), (100.0, 1.9840)))
    check("settlement: the Student-t CDF reproduces the published quantiles",
          worst < 5e-6, f"worst_error={worst:.2e}")
    check("settlement: the Student-t CDF is symmetric about zero",
          abs(features.student_t_cdf(1.3, 8.0)
              + features.student_t_cdf(-1.3, 8.0) - 1.0) < 1e-12,
          f"sum={features.student_t_cdf(1.3, 8.0) + features.student_t_cdf(-1.3, 8.0)}")

    # The imbalance needs real depth: an unknown size is no book, not a neutral one.
    check("settlement: an unknown resting size yields no imbalance, not a neutral one",
          features.l2_imbalance(None, 100) is None
          and features.l2_imbalance(100, None) is None
          and features.l2_imbalance(0, 100) is None,
          "None for unknown or empty depth")
    check("settlement: the imbalance is (bid - ask) / (bid + ask)",
          abs(features.l2_imbalance(500, 200) - 0.6) < 1e-12,
          f"obi={features.l2_imbalance(500, 200)}")

    # The signal: no depth, no thesis; outside the window, no thesis.
    sig = features.student_t_l2_settle(
        spot=70.0, strike=70.0, seconds_left=300.0, sigma=2.0e-4, nu=8.0, s=1.13,
        l2_obi=0.6, flow_fast=0.5, flow_slow=0.4)
    check("settlement: the signal carries the Student-t probability and the book",
          sig is not None and sig.name == "student_t_l2_settle"
          and abs(sig.l2_obi - 0.6) < 1e-12 and abs(sig.flow_fast - 0.5) < 1e-12
          and 0.0 <= sig.p_up <= 1.0,
          f"sig={sig}")
    check("settlement: a spot above the strike prices above a half",
          sig is not None and features.student_t_settlement_p_up(
              spot=71.0, strike=70.0, seconds_left=300.0, sigma=2.0e-4,
              nu=8.0, s=1.13) > 0.5,
          "a spot above the strike settles YES more often than not")
    check("settlement: no depth means no signal",
          features.student_t_l2_settle(
              spot=70.0, strike=70.0, seconds_left=300.0, sigma=2.0e-4, nu=8.0, s=1.13,
              l2_obi=None, flow_fast=0.5, flow_slow=0.4) is None,
          "no resting size to read an imbalance from")
    check("settlement: outside the program's window there is no signal",
          features.student_t_l2_settle(
              spot=70.0, strike=70.0, seconds_left=30.0, sigma=2.0e-4, nu=8.0, s=1.13,
              l2_obi=0.6, flow_fast=0.5, flow_slow=0.4) is None,
          "the window is 780s to 60s before the close")

    def entry(q: Quote, *, p_up: float = 0.70, obi: float = 0.6,
              flow_fast: float = 0.5, flow_slow: float = 0.4,
              seconds_left: float = 300.0) -> Decision:
        return decide(quote=q, p_up=p_up, seconds_left=seconds_left, spot_age=0.1,
                      params=params, position=None, edge_hurdle=Decimal("0.015"),
                      can_open=True, signal="student_t_l2_settle", now=NOW, spot=70.0,
                      l2_obi=obi, flow_fast=flow_fast, flow_slow=flow_slow)

    # The happy path: inside the band, the book leans to the bid, the aggressors
    # lean the same way, and the edge is real.
    good = entry(settlement_quote("0.46", "0.48"))
    check("settlement: a YES entry inside the band with the book and flow agreeing is taken",
          good.action == "entry" and good.side == "YES" and good.price == Decimal("0.48"),
          f"action={good.action} side={good.side} price={good.price} reason={good.reason!r}")

    # The band is 35c-60c, and the 62c ceiling is on top of it.
    over_band = entry(settlement_quote("0.59", "0.61"))
    check("settlement: a 61c cost is outside the 60c band",
          over_band.action == "abstain" and "outside" in over_band.reason,
          f"action={over_band.action} reason={over_band.reason!r}")
    under_band = entry(settlement_quote("0.32", "0.34"))
    check("settlement: a 34c cost is below the 35c band floor",
          under_band.action == "abstain" and "outside" in under_band.reason,
          f"action={under_band.action} reason={under_band.reason!r}")
    over_cap = entry(settlement_quote("0.36", "0.40"), obi=-0.6, flow_fast=-0.5,
                      flow_slow=-0.4)
    check("settlement: a side whose cost is over the 0.62 cap is refused",
          over_cap.action == "abstain" and f"over the {HARD_MAX_ENTRY_PRICE} cap"
          in over_cap.reason,
          f"action={over_cap.action} reason={over_cap.reason!r}")

    # The book and the flow have to lean the way the side does.
    wrong_book = entry(settlement_quote("0.46", "0.48"), obi=0.05)
    check("settlement: a book that does not lean to the side refuses the entry",
          wrong_book.action == "abstain" and "imbalance" in wrong_book.reason,
          f"action={wrong_book.action} reason={wrong_book.reason!r}")
    wrong_flow = entry(settlement_quote("0.46", "0.48"), flow_fast=0.10)
    check("settlement: an aggressor flow that does not lean to the side refuses the entry",
          wrong_flow.action == "abstain" and "flow" in wrong_flow.reason,
          f"action={wrong_flow.action} reason={wrong_flow.reason!r}")
    half_flow = entry(settlement_quote("0.46", "0.48"), flow_fast=0.5, flow_slow=0.10)
    check("settlement: both flow windows have to agree",
          half_flow.action == "abstain" and "flow" in half_flow.reason,
          f"action={half_flow.action} reason={half_flow.reason!r}")

    # The edge is the settlement probability over the premium plus the entry fee.
    thin = entry(settlement_quote("0.46", "0.48"), p_up=0.50)
    check("settlement: a trade with no edge over its premium is refused",
          thin.action == "abstain" and "under" in thin.reason,
          f"action={thin.action} reason={thin.reason!r}")

    # The exit is the settlement: there is no take-profit and no stop.
    pos = Position(ticker=TICKER, side="YES", entry=Decimal("0.48"), contracts=10,
                   opened_ts=NOW - 300, p_entry=0.70, stake=Decimal("4.80"),
                   tp=Decimal("0.53"), sl=Decimal("0.41"), signal="student_t_l2_settle")
    held = decide(quote=settlement_quote("0.46", "0.48"), p_up=0.70, seconds_left=120.0,
                  spot_age=0.1, params=params, position=pos, edge_hurdle=Decimal("0.015"),
                  can_open=True, signal="student_t_l2_settle", now=NOW, spot=70.0)
    check("settlement: the position is held to expiry, never taken off early",
          held.action == "hold" and held.regime == "settle",
          f"action={held.action} regime={held.regime} reason={held.reason!r}")



    control_partial_fills()
    control_tp_scratch_settle()
    control_fail_safe()
    control_f2_execution()
    control_paper_f2()
    control_sizing()
    control_compounding()
    control_depth_clipping()
    control_layagate_cache()
    control_layagate_filter()
    control_settlement_program()

    width = max(len(n) for n, _, _ in RESULTS)
    failed = 0
    for name, ok, detail in RESULTS:
        failed += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} live-exit checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
