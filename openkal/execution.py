"""One execution core, used by the live paper engine and by the tape replay.

The live loop and the replay must not disagree about what a fill is, what a fee is,
or when an exit happens. Everything that touches money lives here:

  * a resting maker entry at the best bid (0 fee)
  * a maker-first stop and a maker-first flatten, with an IOC fallback after a deadline
  * a resting maker take-profit
  * the exact Decimal fee model on every taker leg

The strategy's *decision* lives in `strategy.decide`; this module only obeys it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Protocol

from .fees import maker_fee, taker_fee
from .strategy import Decision, Position, Quote, contracts_for

TICK = Decimal("0.01")
# Contracts that must print at or through a passive limit before the order counts as
# filled. Same rule, same number as `scripts/alpha_search.py`'s `MIN_PASSIVE_QTY`.
MIN_PASSIVE_QTY = Decimal("20")


@dataclass
class RestingOrder:
    """A maker bid waiting for a seller."""

    ticker: str
    side: str
    price: Decimal
    contracts: int
    placed_ts: float
    reserved: Decimal
    last_seen: Decimal = Decimal("0")
    signal: str = ""
    # Print-tape confirmation of this order's fill, accumulated as the tape arrives:
    # `print_cursor` is the last timestamp examined and `printed` the contracts that
    # traded at or through `price` since placement.
    print_cursor: float = 0.0
    printed: Decimal = Decimal("0")
    # Venue order id once the live executor has placed it. It belongs to the slot's
    # own order, not to the executor: one executor serves all four lanes.
    order_id: str = ""
    # The time-in-force the venue was asked for. An IOC order cannot rest: once it
    # is off the book with no fill on the tape it is finished, and a lane that
    # keeps waiting on it blocks every later entry for a minute.
    tif: str = "good_till_canceled"
    # Venue fill total last seen on this order and when it was first seen at that
    # size. A partial total is only booked once it has stopped growing.
    seen_fill: Decimal = Decimal("0")
    seen_ts: float = 0.0


def passive_prints(store: Any, order: RestingOrder, now: float) -> Decimal:
    """New contracts printed at/through this order's limit since it was last read.

    `store` is the read-only print tape. The replay passes the same ledger it reads
    ticks from but keeps `ctx.store` empty, so a parity run can confirm a maker fill
    without writing one into the ledger it is checking.
    """
    if store is None or now <= order.print_cursor:
        return Decimal("0")
    read = getattr(store, "printed_qty", None)
    if read is None:
        return Decimal("0")
    try:
        return Decimal(str(read(ticker=order.ticker, t0=order.print_cursor, t1=now,
                                price=order.price, side=order.side)))
    except Exception:  # noqa: BLE001 - an unreadable tape is no fill, never a guess
        return Decimal("0")


@dataclass
class RestingExit:
    """A maker ask trying to close a position before the IOC fallback."""

    ticker: str
    side: str
    price: Decimal
    contracts: int
    deadline: float
    reason: str
    # `hold_on_timeout`: when the resting exit misses, keep the position and let the
    # binary settle instead of paying a taker fee to flatten it. The promoted flow
    # program's scratch works this way; every other exit keeps the IOC fallback.
    hold_on_timeout: bool = False
    # `retry_on_timeout`: the resting exit was only the *first* step of the exit
    # structure. Drop the order and hand the position back to the decision loop
    # instead of crossing out -- the flow take-profit rests to its 45s barrier and
    # is then replaced by the scratch, exactly as the frozen study scored it.
    retry_on_timeout: bool = False
    # `take_on_timeout`: a take-profit resting above a bid that already pays the
    # target. If nobody lifts it, take the bid while it still pays the target.
    take_on_timeout: bool = False
    order_id: str = ""
    # When the cancel was last sent, and when the order was first seen off the book
    # after its deadline: the timeout only acts once a racing fill would have shown.
    pulled_ts: float = 0.0
    off_ts: float = 0.0
    seen_fill: Decimal = Decimal("0")
    seen_ts: float = 0.0


class ExecutionContext(Protocol):
    cash: Decimal
    realized: Decimal
    fills: list[dict[str, Any]]
    store: Any
    supervisor: Any


@dataclass
class Executor:
    """Stateless operations over a slot; the context owns cash and the ledger."""

    ctx: ExecutionContext
    flatten_maker_s: float = 8.0
    # Read-only venue print tape for maker-fill confirmation. Defaults to the ledger
    # store; the replay overrides it so a fill can be *proven* without being written.
    tape: Any = None
    venue_fills: list[dict[str, Any]] = field(default_factory=list)
    # True for a live epoch: `venue_fills` is then the *complete* set of maker entries
    # the exchange matched, so a resting bid that is not in it did not fill and must
    # not be invented from a print that cannot show the queue ahead of it.
    venue_anchored: bool = False
    # Venue-reported exit prints for a live epoch. The exchange is the only authority
    # on what it actually closed us at; a crash-recovery flatten or a carry settlement
    # is not something the tape can re-derive. Empty for paper fixtures.
    venue_exits: list[dict[str, Any]] = field(default_factory=list)
    # Venue-confirmed entry fills (`fills.note == 'live fill'`) for a live epoch. The
    # exchange is the only authority on whether a resting bid was matched; the print
    # rule cannot see the queue ahead of us. Empty for paper/fixture epochs, where the
    # print tape is the only evidence there is.

    # ---------------------------------------------------------------- entry
    def review_order(self, slot: Any, quote: Quote, now: float) -> None:
        order = slot.order
        if order is None:
            return
        p_side = slot.p_up if order.side == "YES" else 1 - slot.p_up
        edge_left = Decimal(str(p_side)) - order.price
        # The flow program is a microstructure entry, not an edge-versus-book trade:
        # its entry conditions are the z/flow/revision gates, so the weak-tick
        # cancel would silently delete trades the frozen study counted. Its order
        # also expires on the study's 15s entry patience, not the generic 60s.
        flow_order = getattr(order, "signal", "") == "z_obi_flow"
        stale = now - order.placed_ts > (
            slot.params.zobi_entry_patience_s if flow_order else 60)
        # The fair value is a digital-option delta on a per-second spot print, so a
        # single tick can swing it. One weak tick is noise, not a dead thesis; three
        # in a row is a thesis. Cancelling on the first one is how a maker book ends
        # up with a zero fill rate.
        weak = (not flow_order) and edge_left < slot.params.edge_hurdle / 2
        slot.weak_ticks = getattr(slot, "weak_ticks", 0) + 1 if weak else 0
        if slot.halted or quote.ticker != order.ticker or stale or slot.weak_ticks >= 3:
            self.ctx.cash += order.reserved
            self.record(slot, "cancel", order.side, order.price, order.contracts, Decimal("0"),
                        "thesis gone" if not stale else "rested 60s")
            slot.order = None

    def manage_order(self, slot: Any, quote: Quote, seconds_left: float) -> None:
        order = slot.order
        if order is None:
            return
        if quote.ticker != order.ticker or seconds_left < slot.params.flatten_before_close_s / 2:
            self.ctx.cash += order.reserved
            self.record(slot, "cancel", order.side, order.price, order.contracts, Decimal("0"), "expired")
            slot.order = None
            return
        # The venue's own tape is the first word on a live epoch's maker fills.
        confirmed = self._venue_fill(order, quote)
        if confirmed is not None:
            price, qty = confirmed
            self._fill_entry(slot, order, quote, quote.received_ts, price, qty, "live fill")
            return
        if self.venue_anchored:
            # The exchange has already told us every maker entry it matched. This bid is
            # not among them: it rests until patience pulls it, exactly as it did live.
            return
        # Otherwise a maker fill is a trade, not a touch. `ask <= our bid` is a
        # *possibility* of a fill, not evidence of one: live, five ETH/SOL/XRP bids were
        # cancelled unfilled while this code booked them, and the replay then
        # "reproduced" trades the ledger never had. The order fills only once the print
        # tape shows MIN_PASSIVE_QTY contracts traded at or through its limit since it
        # was placed -- `scripts/alpha_search.py`'s rule for a passive candidate.
        order.printed += passive_prints(
            self.tape if self.tape is not None else self.ctx.store, order,
            quote.received_ts)
        order.print_cursor = max(order.print_cursor, quote.received_ts)
        if order.printed >= MIN_PASSIVE_QTY:
            self._fill_entry(slot, order, quote, quote.received_ts, order.price,
                             order.contracts, "maker")

    def _fill_entry(self, slot: Any, order: RestingOrder, quote: Quote, now: float,
                    price: Decimal, contracts: int, note: str) -> None:
        """Book a passive entry at the price the fill actually happened at."""
        qty = max(1, min(int(contracts), order.contracts))
        self.ctx.cash += order.reserved - price * qty
        slot.position = Position(
            ticker=order.ticker, side=order.side, entry=price, contracts=qty,
            opened_ts=now, p_entry=slot.p_up, stake=price * qty,
            tp=min(Decimal("0.99"), price + slot.params.take_profit),
            sl=max(Decimal("0.01"), price - slot.params.stop_loss),
            signal=getattr(order, "signal", ""),
            entry_spot=slot.spot,
        )
        if getattr(order, "signal", "") in ("z_obi_flow", "F2_taker_1s"):
            slot.window_entries = int(getattr(slot, "window_entries", 0)) + 1
        self.record(slot, "entry", order.side, price, qty, Decimal("0"), note)
        slot.order = None

    def _venue_taker_fill(self, quote: Quote, now: float):
        """The venue's IOC fill for this decision, if it matched one.

        Matching is by contract and the decision's working moment. The exchange is
        authoritative for the whole entry -- side, price and size: at the fill tick
        the tape's spot/quote can sit a fraction of a second behind the venue, and
        the recorded decision can land on the other side of the strike. The exit,
        fees and P&L are still re-derived, so a wrong exit or ledger row shows up.
        """
        for index, row in enumerate(self.venue_fills):
            if row.get("ticker") != quote.ticker:
                continue
            try:
                ts = float(row.get("ts") or 0.0)
                price = Decimal(str(row.get("price") or "0"))
                contracts = int(float(row.get("contracts") or 0))
            except (TypeError, ValueError):
                continue
            if contracts <= 0 or ts < now - 5.0 or ts > now + 5.0:
                continue
            side = str(row.get("side") or "")
            if side not in ("YES", "NO"):
                continue
            self.venue_fills.pop(index)
            return side, price, contracts
        return None

    def _venue_fill(self, order: RestingOrder, quote: Quote):
        """Pop the venue's report for this order, if it filled inside this tick.

        Matching is by contract, side, limit (one tick of tolerance for the venue's own
        price rounding) and the order's working life, so a late fill cannot be
        attributed to a different window's bid.
        """
        for index, row in enumerate(self.venue_fills):
            if row.get("ticker") != order.ticker or row.get("side") != order.side:
                continue
            try:
                price = Decimal(str(row.get("price") or "0"))
                ts = float(row.get("ts") or 0.0)
                contracts = int(float(row.get("contracts") or 0))
            except (TypeError, ValueError):
                continue
            if contracts <= 0 or ts < order.placed_ts - 0.5 or ts > quote.received_ts:
                continue
            if (not self.venue_anchored
                    and abs(price - order.price) > Decimal("0.005")):
                # Paper epochs stand in for the exchange, so the fill must land at the
                # resting price. A live epoch's intra-window tape can lag the venue's
                # own match; the venue fill is authoritative for in-window matches.
                continue
            self.venue_fills.pop(index)
            return price, contracts
        return None

    def place_entry(self, slot: Any, decision: Decision, quote: Quote, now: float,
                    seconds_left: float = 0.0) -> None:
        if slot.order is not None or slot.position is not None:
            return
        # No entry inside the flatten window. The live executor has always refused
        # these (55s at the shipped params), so a replay that took them was booking
        # trades the venue was never asked to place: the `final_minute` thesis fires at
        # T-60..T-12 and every one of its entries was refused live.
        if seconds_left < slot.params.flatten_before_close_s + 10:
            return
        # Same rule the live executor enforces: bid < ask, or no entry. The paper
        # path is what produces the study numbers, so it cannot be the loose one.
        if not self.book_open(quote):
            self.record(slot, "cancel", decision.side or "YES",
                        decision.price or Decimal("0"), 0, Decimal("0"),
                        "crossed book refused")
            return
        price = decision.price or Decimal("0")
        if price <= 0:
            return
        qty = contracts_for(slot.bankroll, price)
        if qty <= 0:
            return
        reserved = price * qty
        if reserved > self.ctx.cash:
            return
        # The flow program is post-only by construction: its frozen evidence is a
        # maker entry at the touch, and a taker cross at that price would fill at a
        # price the book never offered.
        taker = (decision.signal == "F2_taker_1s"
                 or (slot.params.entry_style == "taker" and decision.signal != "z_obi_flow"))
        entry_side = None
        if taker:
            if self.venue_anchored:
                # A live taker entry is an IOC: the exchange, not the signal, says
                # whether it filled. Replaying every signal as filled booked seven
                # XRP trades against the one the venue actually matched, so the
                # epoch's repro control could never agree with the ledger.
                confirmed = self._venue_taker_fill(quote, now)
                if confirmed is None:
                    self.record(slot, "cancel", decision.side or "YES", price, qty,
                                Decimal("0"), "IOC unfilled")
                    return
                entry_side, price, qty = confirmed
                reserved = price * qty
                if reserved > self.ctx.cash:
                    return
            fee = taker_fee(price, qty)
            if reserved + fee > self.ctx.cash:
                return
            self.ctx.cash -= reserved + fee
            tp_price = price + (slot.params.f2_take_profit
                                if decision.signal == "F2_taker_1s"
                                else slot.params.take_profit)
            side = entry_side if self.venue_anchored else (decision.side or "YES")
            slot.position = Position(
                ticker=quote.ticker, side=side, entry=price, contracts=qty,
                opened_ts=now, p_entry=slot.p_up, stake=reserved,
                tp=min(Decimal("0.99"), tp_price),
                sl=max(Decimal("0.01"), price - slot.params.stop_loss),
                signal=decision.signal, entry_fee=fee,
                entry_spot=slot.spot, impulse_bps=decision.impulse_bps,
            )
            if decision.signal == "F2_taker_1s":
                slot.window_entries = int(getattr(slot, "window_entries", 0)) + 1
            self.record(slot, "entry", side, price, qty, fee,
                        "live fill" if self.venue_anchored else "taker cross")
            return
        self.ctx.cash -= reserved
        side_last = quote.last if decision.side == "YES" else Decimal(1) - quote.last
        slot.order = RestingOrder(ticker=quote.ticker, side=decision.side or "YES", price=price,
                                  signal=decision.signal,
                                  contracts=qty, placed_ts=now, reserved=reserved, last_seen=side_last,
                                  print_cursor=now)
        self.record(slot, "quote", decision.side or "YES", price, qty, Decimal("0"), "post_only rest")

    # ---------------------------------------------------------------- exit
    def maybe_venue_exit(self, slot: Any, quote: Quote, now: float) -> bool:
        """Close a venue-anchored position at the venue's reported exit print.

        The venue is the only authority on what it actually closed us at. A crash-
        recovery flatten, a maker take-profit or a carry settlement left in the boot
        ledger cannot be re-derived from the tape; without this the replay would close
        the survivor at the next window's quote and book a price the venue never gave.
        """
        if not self.venue_exits or slot.position is None:
            return False
        pos = slot.position
        for i, row in enumerate(self.venue_exits):
            if row.get("ticker") != pos.ticker or row.get("side") != pos.side:
                continue
            try:
                qty = float(row.get("contracts") or 0)
                price = Decimal(str(row.get("price") or "0"))
            except (TypeError, ValueError):
                continue
            if abs(qty - pos.contracts) >= 0.5:
                continue
            self.venue_exits.pop(i)
            reason = str(row.get("note") or "exit")
            self.close(slot, price, reason, maker=True,
                       exit_fee=Decimal(str(row.get("fee") or "0")))
            return True
        return False

    def begin_exit(self, slot: Any, quote: Quote, price: Decimal, reason: str, now: float) -> None:
        """Ask at the bid first; the IOC fallback is in `poll_exit`."""
        pos = slot.position
        if pos is None or slot.exit is not None:
            return
        side_bid = pos.side_bid(quote)
        target = max(Decimal("0.01"), min(Decimal("0.99"), side_bid))
        slot.exit = RestingExit(ticker=quote.ticker, side=pos.side, price=target,
                                contracts=pos.contracts, deadline=now + self.flatten_maker_s, reason=reason)

    def begin_scratch(self, slot: Any, quote: Quote, price: Decimal, now: float) -> None:
        """Rest a post-only exit at breakeven; if it misses, keep the binary."""
        pos = slot.position
        if pos is None or slot.exit is not None:
            return
        slot.exit = RestingExit(
            ticker=quote.ticker, side=pos.side, price=price, contracts=pos.contracts,
            deadline=now + float(slot.params.zobi_scratch_patience_s),
            reason="scratch", hold_on_timeout=True)

    def poll_exit(self, slot: Any, quote: Quote, now: float) -> None:
        ex = slot.exit
        pos = slot.position
        if ex is None or pos is None:
            return
        if quote.ticker != ex.ticker:
            self.close(slot, pos.side_bid(quote), "window rolled", maker=False)
            slot.exit = None
            return
        side_bid = pos.side_bid(quote)
        # Our ask is lifted when the market's bid reaches it.
        if side_bid >= ex.price:
            self.close(slot, ex.price, ex.reason, maker=True)
            slot.exit = None
            return
        if now >= ex.deadline:
            if ex.hold_on_timeout:
                pos.hold_to_settle = True
                self.record(slot, "cancel", pos.side, ex.price, pos.contracts,
                            Decimal("0"), "scratch missed: hold to settlement")
                slot.exit = None
                return
            if ex.retry_on_timeout:
                # First leg of a multi-step exit, not the last: pull it and let the
                # decision loop place the next one (take-profit -> scratch).
                self.record(slot, "cancel", pos.side, ex.price, pos.contracts,
                            Decimal("0"), "exit rested to the barrier")
                slot.exit = None
                return
            self.close(slot, side_bid, ex.reason, maker=False)
            slot.exit = None

    @staticmethod
    def book_open(quote: Quote) -> bool:
        """`bid < ask`, the only book state that may be traded.

        A crossed ladder means one side is stale; a locked one (bid == ask) has no
        spread to earn. Both are refusals, on the paper path and the live path alike.
        """
        return quote.yes_bid < quote.yes_ask

    def execute(self, slot: Any, decision: Decision, quote: Quote,
                seconds_left: float, now: float) -> None:
        if decision.action == "entry":
            self.place_entry(slot, decision, quote, now, seconds_left)
            return
        if slot.position is None:
            return
        if decision.action == "exit_tp":
            # The take-profit is a resting maker ask; it fills the moment the bid reaches it.
            self.close(slot, slot.position.tp, "take-profit reached", maker=True)
            slot.exit = None
            return
        if decision.action == "scratch":
            self.begin_scratch(slot, quote,
                               decision.price or slot.position.side_bid(quote), now)
            return
        if decision.action in {"exit_sl", "flatten"}:
            if slot.position.signal == "F2_taker_1s":
                # F2's confirmed stop and 90s barrier are reduce-only IOC exits at
                # the best bid. Resting them as maker asks understates the fee and
                # lets a disappearing bid postpone the exit the live path takes now.
                self.close(slot, slot.position.side_bid(quote), decision.reason, maker=False)
                slot.exit = None
                return
            self.begin_exit(slot, quote, decision.price or slot.position.side_bid(quote),
                            decision.reason, now)

    # ---------------------------------------------------------------- ledger
    def close(self, slot: Any, price: Decimal, reason: str, maker: bool,
              exit_fee: Decimal | None = None) -> None:
        """Book the close. `exit_fee` is the venue's own tape figure when known.

        The model fee is the right number for paper and replay, but a live close
        has the exchange's fee on the same fill rows that give the price; using the
        model instead leaves cents of drift between this ledger and
        `/portfolio/fills`, and reconciliation is the whole point of live.
        """
        pos = slot.position
        if pos is None:
            return
        if exit_fee is None:
            exit_fee = (maker_fee if maker else taker_fee)(price, pos.contracts)
        # Net of both legs. The entry fee was already taken from cash at fill time;
        # leaving it out of `realized` would overstate every taker round trip.
        total_fee = exit_fee + pos.entry_fee
        pnl = (price - pos.entry) * pos.contracts - total_fee
        self.ctx.cash += price * pos.contracts - exit_fee
        self.ctx.realized += pnl
        slot.bankroll += pnl
        slot.peak_bankroll = max(slot.peak_bankroll, slot.bankroll)
        slot.pnl_today += pnl
        won = pnl > 0
        if won:
            slot.wins += 1
            slot.consecutive_losses = 0
        else:
            slot.losses += 1
            slot.consecutive_losses += 1
        # The flow program halts its window after the first loss: re-entering a
        # trending 15m contract after it has already hurt us is how the study's
        # settlement tails got built.
        if pos.signal == "z_obi_flow" and pnl <= 0 and hasattr(slot, "window_halted"):
            slot.window_halted = True
        # F2's stored probability is already `P(this trade wins)`, per side; every
        # other program stores `P(YES)`, which has to be flipped for a short.
        p_side = (pos.p_entry if pos.side == "YES" or pos.signal == "F2_taker_1s"
                  else 1 - pos.p_entry)
        slot.pairs.append((p_side, 1 if won else 0))
        slot.pairs = slot.pairs[-200:]
        from .features import expected_calibration_error
        slot.ece = expected_calibration_error(slot.pairs) if len(slot.pairs) >= 8 else 0.0
        slot.last_trade = {
            "ts": pos.opened_ts, "ticker": pos.ticker, "side": pos.side, "entry": str(pos.entry),
            "exit": str(price), "contracts": pos.contracts, "fee": str(total_fee),
            "pnl": str(pnl), "reason": reason, "p_entry": pos.p_entry,
            "signal": pos.signal,
        }
        if self.ctx.store is not None:
            self.ctx.store.trade(
                series=slot.cfg.series, ticker=pos.ticker, side=pos.side, opened_ts=pos.opened_ts,
                closed_ts=slot.quote.received_ts if slot.quote else pos.opened_ts,
                entry=pos.entry, exit_price=price,
                contracts=pos.contracts, fee=total_fee, pnl=pnl, exit_reason=reason,
                p_model=pos.p_entry, spot=slot.spot, strike=slot.quote.strike if slot.quote else None,
                source=getattr(self.ctx, "source", "live"),
            )
        self.record(slot, "exit", pos.side, price, pos.contracts, exit_fee, reason, pnl=pnl)
        slot.position = None
        if self.ctx.supervisor is not None and not slot.halted:
            reason_hit = self.ctx.supervisor.triggered(slot)
            if reason_hit:
                self.ctx.supervisor.start(slot, reason_hit)

    def record(self, slot: Any, kind: str, side: str, price: Decimal, contracts: int,
               fee: Decimal, note: str, pnl: Decimal = Decimal("0")) -> None:
        self.ctx.fills.append({
            "ts": slot.quote.received_ts if slot.quote else 0.0, "series": slot.cfg.series,
            "side": side, "kind": kind, "price": str(price), "contracts": contracts,
            "pnl": str(pnl), "reason": note, "fee": fee,
        })
        self.ctx.fills = self.ctx.fills[-200:]
        if self.ctx.store is not None and kind in {"entry", "exit"}:
            self.ctx.store.fill(
                ts=slot.quote.received_ts if slot.quote else 0.0, series=slot.cfg.series,
                ticker=slot.quote.ticker if slot.quote else "", side=side, kind=kind,
                price=str(price), contracts=contracts, fee=str(fee), note=note,
            )
