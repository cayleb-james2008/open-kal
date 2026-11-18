"""Live execution against the venue, with venue truth as the only fill source.

Paper fills simulate a maker book. Once armed, a position exists only when the
exchange's fills tape says it filled, at the exchange's fill price, and the ledger is
written from that number. The decision, the sizing, the fee model and `close()` are
the same code the tape replay drives.
"""
from __future__ import annotations

import time
from dataclasses import replace
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from typing import Any

from .broker import VenueOrder
from .config import (HARD_MAX_ENTRY_PRICE, peak_drawdown_tripped, stage_for,
                     staged_risk_cap)
from .execution import Executor, RestingExit, RestingOrder
from .strategy import Decision, Position, Quote, contracts_for


def sized_contracts(bankroll: Decimal, price: Decimal, params: Any,
                    equity: Decimal | None = None,
                    peak_equity: Decimal | None = None) -> int:
    """Contracts one entry may hold: the slot, the stage's risk and its ceiling.

    `contracts_for` spends the whole slot, which is how a 24c NO became 170
    contracts ($40.80 of premium) against a study scored at 50. A binary that
    settles to zero loses the premium, not the 7c stop distance, so the
    worst-case loss has to be bounded here rather than at the exit.

    `equity` is the *account* equity the compounding stage is read from; the lane's
    `bankroll` stays the outer bound (a lane never spends more than its own slice).
    The stage table in `openkal.config` replaces the flat MAX_CONTRACTS /
    MAX_TRADE_RISK_USD pair: at stage 1 it reproduces them ($12.14 at $121.46) and
    at stages 2-6 it lifts the contract ceiling as the account compounds.
    `params` still carries the study's own ceilings for replay and research; sizing
    here is owned by config so a mutated strategy parameter cannot widen it.

    `peak_equity` is the account's high-water mark. A carried binary loses its whole
    premium, so the staged fraction alone can still put a loss through the 20%
    trailing-peak floor: at a fresh peak a 12% stage trade would leave 88% of peak,
    and a second one taken 10% down would leave 79%. The premium at risk is therefore
    clipped to half the remaining distance to `0.80 * peak_equity`, so even a total
    loss cannot carry equity through the floor from any equity the breaker allows.
    When the peak is unknown the current equity is used as the peak, which is the
    conservative reading: no caller may claim more than a fresh high-water buffer.
    """
    if price <= 0 or price > HARD_MAX_ENTRY_PRICE:
        return 0
    account = bankroll if equity is None else Decimal(str(equity))
    stage = stage_for(account)
    peak = account if peak_equity is None else Decimal(str(peak_equity))
    if peak <= 0:
        peak = account
    buffer_risk = (Decimal("0.50")
                   * max(Decimal("0"), account - peak * Decimal("0.80"))
                   ).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
    effective_risk = min(staged_risk_cap(account), buffer_risk)
    by_risk = int(effective_risk / price)
    return max(0, min(contracts_for(bankroll, price), stage.max_contracts, by_risk))


# A single order may take at most half of the size resting at the ask it pays. The
# ladder in `data/openkal-prod.db` says how thin these books are: a 300-lot stage 3
# order into a touch holding 40 contracts is a market impact the study never priced,
# and on a 24c NO the rest of the book is 40c away.
DEPTH_CLIP_FRACTION = Decimal("0.50")


def depth_clipped(qty: int, side: str, quote: Any) -> int:
    """Clip `qty` to half the size resting on the ask this order would pay.

    A YES entry pays the YES ask, so its depth is `yes_ask_sz`; a NO entry buys NO
    at `1 - yes_bid`, so its depth is the resting YES bid, `yes_bid_sz`. Rounded
    down, never up: a 3-lot touch allows 1 contract, and a 1-lot touch allows none.

    When the touch size is unknown -- a quote from a REST snapshot, or a moment
    with no socket ladder -- nothing is clipped, because the alternative is
    refusing every entry on a tape that does not record depth.
    """
    if qty <= 0:
        return 0
    size = getattr(quote, "yes_ask_sz" if side == "YES" else "yes_bid_sz", None)
    if size is None:
        return qty
    try:
        resting = Decimal(str(size))
    except (ArithmeticError, TypeError, ValueError):
        return qty
    if resting <= 0:
        return qty
    allowed = int((resting * DEPTH_CLIP_FRACTION).to_integral_value(rounding=ROUND_DOWN))
    return max(0, min(qty, allowed))


def _dec(value: Any) -> Decimal:
    try:
        return Decimal(str(value or "0"))
    except Exception:  # noqa: BLE001 - a malformed venue number must not kill the loop
        return Decimal("0")


class LiveExecutor(Executor):
    def __init__(self, ctx: Any, broker: Any) -> None:
        super().__init__(ctx)
        self.broker = broker
        # Per-slot state lives on the slot (`RestingOrder.order_id` /
        # `RestingExit.order_id`). One executor serves four independent lanes; holding
        # a single `entry_order` here meant any lane's resting bid blocked every other
        # lane's entry, and the id was overwritten by whichever lane spoke last.
        self.note: str = ""

    async def step(self, slot: Any, decision: Decision, quote: Quote, seconds_left: float,
                   now: float, venue: dict[str, Any]) -> None:
        await self._poll_entry(slot, quote, now, venue)
        await self._poll_exit(slot, quote, now, venue)

        if slot.position is not None:
            await self._rest_take_profit(slot, decision, quote, now)
            await self._decide_exit(slot, decision, quote, seconds_left, now)
            return
        if slot.order is not None:
            return
        if decision.action == "entry":
            await self._place_entry(slot, decision, quote, now, seconds_left)

    # ------------------------------------------------------------------ entry
    async def _place_entry(self, slot: Any, decision: Decision, quote: Quote,
                           now: float, seconds_left: float) -> None:
        price = decision.price or Decimal("0")
        if price <= 0 or seconds_left < slot.params.flatten_before_close_s + 10:
            return
        # Last gate before real money: `bid < ask` or no order. `decide` already
        # refuses a crossed ladder, but the live path must not depend on an upstream
        # caller remembering to. A crossed book is a stale side, and resting a bid
        # into it books a spread the venue never offered.
        if not self.book_open(quote):
            self.note = f"crossed book refused: bid {quote.yes_bid} >= ask {quote.yes_ask}"
            self.record(slot, "cancel", decision.side or "YES", price, 0, Decimal("0"),
                        "crossed book refused")
            return
        # F2 is a taker rule and its fill price is part of the frozen evidence:
        # the order is a limit IOC pinned to the best ask *of the side held*, and
        # nothing above it. If the ask has moved down since the decision, paying
        # the decided price would be paying above the book (the upward walk this
        # program is measured without), so the entry is refused; if it has moved
        # up, the pin follows the book, because the study priced the order at its
        # arrival (T+1s) and never at the trigger's quote.
        if decision.signal == "F2_taker_1s":
            pin = (quote.yes_ask if decision.side == "YES"
                   else (Decimal(1) - quote.yes_bid))
            if price > pin:
                self.note = (f"F2 entry refused: ask moved down {price} -> {pin}; "
                             f"no upward price walk")
                self.record(slot, "cancel", decision.side or "YES", price, 0,
                            Decimal("0"), "F2 pin: ask moved down")
                return
            price = pin
            if not (slot.params.f2_min_price <= price <= slot.params.f2_max_price):
                self.note = f"F2 entry refused: pinned cost {price} outside the band"
                return
        if decision.signal == "student_t_l2_settle":
            # The settlement program is a taker rule on the same terms: the order is
            # an IOC at the ask of the side held, the band is its own, and the cost
            # ceiling below still refuses a pin that has walked out of the band.
            pin = (quote.yes_ask if decision.side == "YES"
                   else (Decimal(1) - quote.yes_bid))
            if price > pin:
                self.note = (f"settlement entry refused: ask moved down {price} -> {pin}; "
                             "no upward price walk")
                self.record(slot, "cancel", decision.side or "YES", price, 0,
                            Decimal("0"), "settlement pin: ask moved down")
                return
            price = pin
            if not (slot.params.sl2_min_price <= price <= slot.params.sl2_max_price):
                self.note = f"settlement entry refused: pinned cost {price} outside the band"
                return
        # Two invariants outrank every stage: no entry above the $0.62 cost
        # ceiling, and no new risk once the account is 20% below its high-water
        # mark. Both are checked after the F2 re-pin, so a price that moved up into
        # the ceiling is refused rather than paid.
        if price > HARD_MAX_ENTRY_PRICE:
            self.note = f"entry refused: {price} is above the {HARD_MAX_ENTRY_PRICE} ceiling"
            self.record(slot, "cancel", decision.side or "YES", price, 0, Decimal("0"),
                        "above the 0.62 cost ceiling")
            return
        equity = getattr(self.ctx, "equity", None)
        if equity is None:
            equity = getattr(self.ctx, "cash", slot.bankroll)
        peak = getattr(self.ctx, "peak_equity", None)
        if peak_drawdown_tripped(equity, peak):
            self.note = (f"compounding breaker: equity {equity} is 20% below peak {peak}")
            self.record(slot, "cancel", decision.side or "YES", price, 0, Decimal("0"),
                        "20% trailing peak drawdown")
            return
        qty = sized_contracts(slot.bankroll, price, slot.params, equity=equity,
                              peak_equity=peak)
        qty = depth_clipped(qty, decision.side or "YES", quote)
        if qty <= 0:
            self.note = "entry skipped: the resting ask is too thin to size against"
            self.record(slot, "cancel", decision.side or "YES", price, 0, Decimal("0"),
                        "resting depth below one contract")
            return
        reserved = price * qty
        if reserved > self.ctx.cash:
            return
        # Post-only for the flow program: the frozen evidence is a maker entry at
        # the touch, and a taker cross would pay a fee the study never charged.
        # F2 is taker by construction, independent of a generic entry-style default.
        taker = (decision.signal == "F2_taker_1s"
                 or (slot.params.entry_style == "taker" and decision.signal != "z_obi_flow"))
        tif = "immediate_or_cancel" if taker else "good_till_canceled"
        order = await self.broker.place(
            quote.ticker, decision.side or "YES", "entry", price, qty,
            post_only=not taker, tif=tif,
        )
        if order is None:
            self.note = self.broker.errors[-1] if self.broker.errors else "venue rejected the bid"
            return
        self.ctx.cash -= reserved
        # The signal tag is not decoration: it decides how long the order rests (the
        # flow program waits its 15s patience instead of dying on a weak model tick)
        # and, once filled, which exit machine the position runs. Dropping it here made
        # every live flow order behave like a generic one.
        slot.order = RestingOrder(ticker=quote.ticker, side=decision.side or "YES", price=price,
                                  signal=decision.signal,
                                  contracts=qty, placed_ts=now, reserved=reserved,
                                  print_cursor=now, order_id=order.order_id, tif=tif)
        self.record(slot, "quote", decision.side or "YES", price, qty, Decimal("0"),
                    f"live {'IOC' if taker else 'post_only'} {order.order_id[:8]}")

    def _fill_for(self, venue: dict[str, Any], order_id: str) -> dict[str, Any] | None:
        for f in venue.get("fills", []) or []:
            if f.get("order_id") == order_id:
                return f
        return None

    def _fill_summary(self, venue: dict[str, Any], order_id: str,
                      side: str) -> tuple[int, Decimal, Decimal] | None:
        """Every print on this order, summed. An order fills in pieces.

        Reading only the first print booked 42 of the 84 contracts the venue actually
        delivered (42.75 + 41.25 on one order), and the ledger then held half a
        position while the exchange held the whole one.
        """
        rows = [f for f in (venue.get("fills") or []) if f.get("order_id") == order_id]
        if not rows:
            return None
        exact = sum((_dec(f.get("count_fp")) for f in rows), Decimal("0"))
        if exact <= 0:
            return None
        notional = sum((_dec(f.get("count_fp")) * self._fill_price(f, side) for f in rows),
                       Decimal("0"))
        fee = sum((_dec(f.get("fee_cost")) for f in rows), Decimal("0"))
        # Whole contracts actually delivered, floored: rounding *up* booked a 13-lot
        # position out of 12.7 contracts the venue delivered. Flooring keeps a
        # 15.35-of-45 exit a 15-lot exit and a 20.00-of-59 a 20-lot one, which is
        # what the venue reports. A sub-contract total still leaves a real position
        # at the exchange, so it books the smallest this ledger can manage (one)
        # rather than dropping it; reconcile_live.py then sees the size difference
        # instead of an orphan nobody is watching.
        qty = int(exact.to_integral_value(rounding=ROUND_DOWN))
        if qty <= 0:
            qty = 1
        return qty, (notional / exact).quantize(Decimal("0.0001")), fee

    # A venue snapshot is cached for 1.2s, so "unchanged" has to span fresh reads.
    FILL_SETTLE_S = 3.0

    def _fill_final(self, venue: dict[str, Any], tracker: Any, order_id: str,
                    contracts: int, resting: bool, now: float) -> bool:
        """True once the fills seen on this order are everything it will get.

        An order fills in pieces and the tape can trail the book. Booking the first
        print booked 1 of 59 ETH contracts (0.22 + 22.88 + 35.90 inside 270ms) and
        15.35 of a 45-lot exit as a full close, leaving the rest at the venue
        unmanaged. Final means: the whole order printed, or the order is off the
        book and its total has not grown for `FILL_SETTLE_S`.
        """
        exact = sum((_dec(f.get("count_fp")) for f in (venue.get("fills") or [])
                     if f.get("order_id") == order_id), Decimal("0"))
        if exact >= Decimal(contracts) - Decimal("0.5"):
            return True
        if resting or exact != tracker.seen_fill:
            # The settle clock runs only once the order is off the book: a print can
            # race the cancel of an order that had rested part-filled for a while.
            tracker.seen_fill, tracker.seen_ts = max(exact, tracker.seen_fill), now
            return False
        return now - tracker.seen_ts >= self.FILL_SETTLE_S

    @staticmethod
    def _fill_price(fill: dict[str, Any], side: str) -> Decimal:
        if side == "YES":
            return _dec(fill.get("yes_price_dollars"))
        return _dec(fill.get("no_price_dollars"))

    async def _poll_entry(self, slot: Any, quote: Quote, now: float, venue: dict[str, Any]) -> None:
        if slot.order is None or not slot.order.order_id:
            return
        order_id = slot.order.order_id
        summary = self._fill_summary(venue, order_id, slot.order.side)
        resting = order_id in (venue.get("orders") or {})
        if summary is None and slot.order.seen_fill > 0:
            # Fills were on the tape and now are not: a failed read, not a reversal.
            return
        if summary is not None and not self._fill_final(
                venue, slot.order, order_id, slot.order.contracts, resting, now):
            # Part-filled. Past its patience the remainder is pulled; either way the
            # position is booked only when the total has stopped moving.
            flow = getattr(slot.order, "signal", "") == "z_obi_flow"
            if resting and now - slot.order.placed_ts > (
                    slot.params.zobi_entry_patience_s if flow else 60):
                await self.broker.cancel(order_id, quote.ticker)
            return
        if summary is not None:
            side = slot.order.side
            filled_qty, filled_px, filled_fee = summary
            price = filled_px or slot.order.price
            qty = max(1, min(filled_qty, slot.order.contracts))
            sig = getattr(slot.order, "signal", "")
            self.ctx.cash += slot.order.reserved - price * qty - filled_fee
            tp_price = price + (slot.params.f2_take_profit
                                if sig == "F2_taker_1s" else slot.params.take_profit)
            slot.position = Position(
                ticker=slot.order.ticker, side=side, entry=price, contracts=qty,
                opened_ts=now, p_entry=slot.p_up, stake=price * qty,
                tp=min(Decimal("0.99"), tp_price),
                sl=max(Decimal("0.01"), price - slot.params.stop_loss),
                signal=sig, entry_fee=filled_fee,
                entry_spot=slot.spot,
            )
            if sig in ("z_obi_flow", "F2_taker_1s", "student_t_l2_settle"):
                slot.window_entries = int(getattr(slot, "window_entries", 0)) + 1
            self.record(slot, "entry", side, price, qty, filled_fee, "live fill")
            slot.order = None
            return
        # An IOC entry that is off the book with no print on the tape has expired
        # unfilled: that is what immediate-or-cancel means. Waiting the generic
        # 60s would block this lane's next entry behind an order the venue has
        # already killed, which on a 90s-barrier rule is most of the window.
        if (getattr(slot.order, "tif", "") == "immediate_or_cancel" and not resting
                and now - slot.order.placed_ts >= self.FILL_SETTLE_S):
            self.ctx.cash += slot.order.reserved
            self.record(slot, "cancel", slot.order.side, slot.order.price,
                        slot.order.contracts, Decimal("0"), "IOC unfilled")
            slot.order = None
            return
        # The flow program's entry rests for its patience and is never cancelled on a
        # weak model tick: its gates are z/flow/revision, not edge.
        flow_order = getattr(slot.order, "signal", "") == "z_obi_flow"
        stale = now - slot.order.placed_ts > (
            slot.params.zobi_entry_patience_s if flow_order else 60)
        p_side = slot.p_up if slot.order.side == "YES" else 1 - slot.p_up
        edge_left = Decimal(str(p_side)) - slot.order.price
        if not resting and not stale:
            return
        if not resting or stale or ((not flow_order)
                                    and edge_left < slot.params.edge_hurdle / 2):
            if resting:
                await self.broker.cancel(order_id, quote.ticker)
            self.ctx.cash += slot.order.reserved
            self.record(slot, "cancel", slot.order.side, slot.order.price, slot.order.contracts,
                        Decimal("0"), "thesis gone" if not stale else "rested 60s")
            slot.order = None

    # ------------------------------------------------------------------ exit
    # How long a take-profit rests above a bid that already pays it before the
    # bid is taken instead.
    TP_GAP_MAKER_S = 3.0

    async def _rest_take_profit(self, slot: Any, decision: Decision, quote: Quote,
                                now: float) -> None:
        """Rest the flow take-profit as a maker ask from the fill, as the study scored it.

        The frozen structure is an ask at entry+5c that is lifted for a $0 fee. Waiting
        for the bid to reach the target first meant the ask could only ever cross, so
        every live take-profit left as a taker (trade 65: $1.05 on a $5.40 gain).
        """
        pos = slot.position
        if (pos is None or slot.exit is not None or pos.hold_to_settle
                or decision.action != "hold" or not self.book_open(quote)):
            return
        if pos.signal == "F2_taker_1s":
            # entry_fill + 0.05, exactly, from the venue's own fill price: the
            # take-profit is the trade, and it costs nothing when it is lifted.
            target = pos.entry + slot.params.f2_take_profit
            left = slot.params.f2_barrier_s - max(0.0, now - pos.opened_ts)
            if left < 1.0 or target <= pos.side_bid(quote):
                return
            await self._start_exit(slot, quote, target, "F2 take-profit", now,
                                   patience_s=left, retry_on_timeout=True)
            return
        if pos.signal != "z_obi_flow":
            return
        left = slot.params.zobi_barrier_s - max(0.0, now - pos.opened_ts)
        if left < 1.0 or pos.tp <= pos.side_bid(quote):
            return
        await self._start_exit(slot, quote, pos.tp, "take-profit reached", now,
                               patience_s=left, retry_on_timeout=True)

    async def _decide_exit(self, slot: Any, decision: Decision, quote: Quote,
                           seconds_left: float, now: float) -> None:
        # `slot.exit` is the in-flight marker, and the venue order id lives on it. The
        # old executor-level `exit_order` was shared by four lanes, so one lane's exit
        # blocked (and overwrote) every other lane's.
        if slot.exit is not None:
            if decision.action in {"exit_sl", "flatten"} and slot.exit.retry_on_timeout:
                # A stop or a flatten outranks the resting take-profit: pull it now
                # so the next tick can act, instead of waiting out the barrier.
                slot.exit.deadline = min(slot.exit.deadline, now)
            return
        pos = slot.position
        if pos is None:
            return
        if decision.action == "exit_tp":
            # The flow take-profit is a post-only ask that rests *to the 45s barrier*,
            # not the 8s generic fallback: the frozen exit structure scores the TP and
            # the scratch against the same barrier, and an IOC after 8s would pay a
            # taker fee the study never charged.
            flow = pos.signal == "z_obi_flow"
            f2 = pos.signal == "F2_taker_1s"
            barrier = (slot.params.zobi_barrier_s if flow
                       else slot.params.f2_barrier_s if f2 else 0.0)
            patience = (max(1.0, barrier - max(0.0, now - pos.opened_ts))
                        if (flow or f2) else None)
            await self._start_exit(slot, quote, pos.tp, "take-profit reached", now,
                                   patience_s=patience, retry_on_timeout=flow or f2)
        elif decision.action == "scratch":
            # Post-only at breakeven, and if it misses, the binary is kept rather
            # than flattened with a taker order.
            await self._start_exit(slot, quote, decision.price or pos.side_bid(quote),
                                   decision.reason, now, hold_on_timeout=True,
                                   patience_s=slot.params.zobi_scratch_patience_s)
        elif decision.action in {"exit_sl", "flatten"}:
            await self._start_exit(slot, quote, pos.side_bid(quote), decision.reason, now)

    async def _start_exit(self, slot: Any, quote: Quote, price: Decimal,
                          reason: str, now: float, *, hold_on_timeout: bool = False,
                          patience_s: float | None = None,
                          retry_on_timeout: bool = False) -> None:
        pos = slot.position
        if pos is None:
            return
        target = max(Decimal("0.01"), min(Decimal("0.99"), price))
        wait = self.flatten_maker_s if patience_s is None else float(patience_s)
        if not hold_on_timeout and target <= pos.side_bid(quote):
            # The book already pays at least the target. A post-only sell at or below
            # the bid IS the cross the venue rejects (`post only cross`, observed three
            # times live), so take the bid instead: the fill is no worse than target.
            ask = pos.side_ask(quote)
            if not (retry_on_timeout and self.book_open(quote) and ask > pos.side_bid(quote)):
                await self._ioc_out(slot, quote, reason, now)
                return
            # The flow take-profit: rest at the ask first ($0 fee, and any further
            # gap is ours), and take the bid only if nobody lifts it.
            target, wait = min(Decimal("0.99"), ask), self.TP_GAP_MAKER_S
            if target <= pos.side_bid(quote):
                # 0.99 bid: there is no price above it to rest at.
                await self._ioc_out(slot, quote, reason, now)
                return
            take_on_timeout, retry_on_timeout = True, False
        else:
            take_on_timeout = False
        slot.exit = RestingExit(ticker=quote.ticker, side=pos.side, price=target,
                                contracts=pos.contracts, deadline=now + wait,
                                reason=reason, hold_on_timeout=hold_on_timeout,
                                retry_on_timeout=retry_on_timeout,
                                take_on_timeout=take_on_timeout)
        order = await self.broker.place(pos.ticker, pos.side, "exit", target, pos.contracts)
        if order is None:
            # Nothing is resting at the venue. Clear the marker so the next tick can
            # retry at a fresh price instead of waiting on a deadline nobody honours.
            slot.exit = None
            self.note = self.broker.errors[-1] if self.broker.errors else "exit rejected"
            return
        slot.exit.order_id = order.order_id

    async def _poll_exit(self, slot: Any, quote: Quote, now: float, venue: dict[str, Any]) -> None:
        ex = slot.exit
        pos = slot.position
        if pos is None or ex is None:
            return
        if ex.order_id:
            rows = [f for f in (venue.get("fills") or []) if f.get("order_id") == ex.order_id]
            summary = self._fill_summary(venue, ex.order_id, pos.side)
            resting = ex.order_id in (venue.get("orders") or {})
            if summary is None and ex.seen_fill > 0:
                # Fills were on the tape and now are not: a failed read, not a reversal.
                return
            if summary is not None and self._fill_final(
                    venue, ex, ex.order_id, pos.contracts, resting, now):
                # Every print, not just the first: a 42-lot exit that fills 35.75 then
                # 6.25 is one 42-lot exit.
                qty, price, fee = summary
                maker = not any(bool(f.get("is_taker")) for f in rows)
                # The venue's fee, not the model's: the tape that gave us the price
                # gave us the fee on the same row.
                self._close_filled(slot, qty, price or ex.price, ex.reason, maker, fee)
                slot.exit = None
                return
            if summary is not None and not resting:
                # Off the book with a partial total still settling: wait for it.
                return
            if now >= ex.deadline and resting:
                # Keep the id until the order is seen off the book: a fill that races
                # the cancel is then booked instead of being forgotten.
                if now - ex.pulled_ts >= 2.0:
                    ex.pulled_ts = now
                    await self.broker.cancel(ex.order_id, quote.ticker)
                return
            elif not resting and now < ex.deadline:
                # Gone from the book without a fill we can see yet: give it one tick.
                return
            # Off the book past its deadline with no fill on the tape. The tape can
            # trail the book, and these exits are not reduce-only: acting on a fill
            # that has not shown yet would sell the position twice.
            ex.off_ts = ex.off_ts or now
            if now - ex.off_ts < self.FILL_SETTLE_S:
                return
        if now >= ex.deadline:
            if ex.take_on_timeout:
                slot.exit = None
                if pos.side_bid(quote) >= pos.tp:
                    await self._ioc_out(slot, quote, ex.reason, now)
                # Otherwise the bid fell back under the target: the decision loop
                # owns the position again (resting take-profit, then the scratch).
                return
            if ex.hold_on_timeout:
                # The scratch missed: keep the binary, cancel the order, no taker leg.
                pos.hold_to_settle = True
                self.record(slot, "cancel", pos.side, ex.price, pos.contracts,
                            Decimal("0"), "scratch missed: hold to settlement")
                slot.exit = None
                return
            if ex.retry_on_timeout:
                # The take-profit rested to the barrier unlifted. Hand the position back
                # to the decision loop, which places the 45s post-only scratch; crossing
                # out here would be a taker exit the frozen structure never scored.
                self.record(slot, "cancel", pos.side, ex.price, pos.contracts,
                            Decimal("0"), "take-profit expired at the barrier")
                slot.exit = None
                return
            await self._ioc_out(slot, quote, ex.reason or "exit fallback", now)

    def _close_filled(self, slot: Any, qty: int, price: Decimal, reason: str,
                      maker: bool, fee: Decimal) -> None:
        """Close the contracts the venue actually took; keep the rest as a position."""
        pos = slot.position
        left = pos.contracts - qty
        if left <= 0:
            self.close(slot, price, reason, maker=maker, exit_fee=fee)
            return
        rest = replace(pos, contracts=left, stake=pos.entry * left, entry_fee=Decimal("0"))
        pos.contracts, pos.stake = qty, pos.entry * qty
        self.close(slot, price, reason, maker=maker, exit_fee=fee)
        slot.position = rest

    async def _ioc_out(self, slot: Any, quote: Quote, reason: str, now: float) -> None:
        pos = slot.position
        if pos is None:
            return
        bid = pos.side_bid(quote)
        order = await self.broker.place(pos.ticker, pos.side, "exit", bid, pos.contracts,
                                        post_only=False, tif="immediate_or_cancel",
                                        reduce_only=True)
        if order is None:
            self.note = "IOC exit rejected; position still open"
            slot.exit = None
            return
        # A taker order usually fills on the ack, but the fill tape can lag it. Poll
        # this order like any other instead of firing a fresh IOC on the next tick:
        # live, one 42-lot position was sold twice (two 42-lot IOC orders) because the
        # first fill was not visible yet when the decision re-ran.
        slot.exit = RestingExit(ticker=quote.ticker, side=pos.side, price=bid,
                                contracts=pos.contracts,
                                deadline=now + self.flatten_maker_s,
                                reason=reason, hold_on_timeout=True,
                                order_id=order.order_id)
        fills = await self.broker.fills(order_id=order.order_id)
        exact = sum((_dec(f.get("count_fp")) for f in fills), Decimal("0"))
        # Only a whole fill is booked on the ack. A partial one is left to
        # `_poll_exit`, which waits for the total to settle and keeps the remainder.
        if exact >= Decimal(pos.contracts) - Decimal("0.5"):
            notional = sum((_dec(f.get("count_fp")) * self._fill_price(f, pos.side)
                            for f in fills), Decimal("0"))
            fee = sum((_dec(f.get("fee_cost")) for f in fills), Decimal("0"))
            price = (notional / exact).quantize(Decimal("0.0001")) if exact > 0 else bid
            self.close(slot, price or bid, reason, maker=False, exit_fee=fee)
            slot.exit = None

    async def flatten_everything(self, slots: list[Any]) -> None:
        """Kill switch: cancel every resting bid, then take out every position."""
        for slot in slots:
            if (slot.order is not None and slot.order.order_id
                    and slot.quote is not None):
                await self.broker.cancel(slot.order.order_id, slot.quote.ticker)
            if slot.order is not None:
                self.ctx.cash += slot.order.reserved
                slot.order = None
            if slot.exit is not None and slot.exit.order_id and slot.quote is not None:
                await self.broker.cancel(slot.exit.order_id, slot.quote.ticker)
            if slot.position is not None and slot.quote is not None:
                await self._ioc_out(slot, slot.quote, "kill switch")
                slot.exit = None
        await self.broker.flatten_all()

    async def emergency_flatten(self, slot: Any, reason: str = "fail-safe") -> bool:
        """One slot's fail-safe: pull its orders, then cross out reduce-only.

        Called when the lane threw an unhandled exception while holding a live
        position. The alternative -- leaving the position for the next tick -- is
        how a +$4.25 take-profit became a -$65.45 settlement: the exit machine
        never got a chance to run and the binary expired unguarded. Returns True
        when the slot is flat.
        """
        pos = slot.position
        quote = slot.quote
        ticker = pos.ticker if pos is not None else (quote.ticker if quote is not None else "")
        # Cancel by ticker at the venue, not only the ids we still remember: an
        # order whose id was lost to the exception is still a resting order.
        try:
            resting = await self.broker.open_orders()
        except Exception:  # noqa: BLE001 - the cross-out matters more than the list
            resting = {}
        for oid, row in (resting or {}).items():
            order_ticker = row.get("ticker") or ticker
            if ticker and order_ticker != ticker:
                continue
            await self.broker.cancel(oid, order_ticker)
        if slot.order is not None:
            self.ctx.cash += slot.order.reserved
            self.record(slot, "cancel", slot.order.side, slot.order.price,
                        slot.order.contracts, Decimal("0"), f"{reason}: resting bid pulled")
            slot.order = None
        slot.exit = None
        if pos is None:
            self.note = f"{reason}: no position, resting orders pulled"
            return True
        # A YES exit sells YES, so 0.01 crosses any bid; a NO exit buys YES, so it
        # needs the other end of the book. Using 0.01 for both is a NO order that
        # can never fill.
        bid = pos.side_bid(quote) if quote is not None else (
            Decimal("0.01") if pos.side == "YES" else Decimal("0.99"))
        order = await self.broker.place(pos.ticker, pos.side, "exit", bid, pos.contracts,
                                        post_only=False, tif="immediate_or_cancel",
                                        reduce_only=True)
        if order is None:
            self.note = f"{reason}: IOC rejected; position still open"
            return False
        fills = await self.broker.fills(order_id=order.order_id)
        exact = sum((_dec(f.get("count_fp")) for f in fills), Decimal("0"))
        if exact >= Decimal(pos.contracts) - Decimal("0.5"):
            notional = sum((_dec(f.get("count_fp")) * self._fill_price(f, pos.side)
                            for f in fills), Decimal("0"))
            fee = sum((_dec(f.get("fee_cost")) for f in fills), Decimal("0"))
            price = (notional / exact).quantize(Decimal("0.0001")) if exact > 0 else bid
            self.close(slot, price or bid, reason, maker=False, exit_fee=fee)
            self.note = f"{reason}: flattened {pos.contracts} {pos.side} at {price}"
            return slot.position is None
        # The ack can lead the tape. Poll this order on the next tick instead of
        # firing a second IOC at a position that may already be gone.
        slot.exit = RestingExit(ticker=pos.ticker, side=pos.side, price=bid,
                                contracts=pos.contracts,
                                deadline=time.time() + self.flatten_maker_s,
                                reason=reason, hold_on_timeout=False,
                                order_id=order.order_id)
        self.note = f"{reason}: IOC acked, awaiting the fill tape"
        return False
