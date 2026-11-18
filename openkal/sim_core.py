"""Canonical sub-second fill simulator for the 15-minute Kalshi binaries.

Every research backtest prices fills through this module, so the four rules that
earlier sweeps each got wrong at least once live in one place:

  * Book symmetry. A NO contract is priced from the YES ladder: NO ask = 1 - yes_bid,
    NO bid = 1 - yes_ask. Callers pass the side, the style and the YES quote; they
    never pass a price. That is what removes the "limit in YES terms" bug class.
  * Arrival. An order exists `lag_s` (>= 1s) after its trigger and is priced on the
    last book row at or before arrival. A crossed row raises `CrossedBook`, an empty
    or one-sided row raises `NoQuote`; `run_trade` turns both into a counted skip.
  * Passive fills. A resting order fills only once >= 50 contracts have printed at
    or through its limit after it arrived, on the side of the tape that can hit it.
  * Fees. Every taker leg pays `openkal.fees.taker_fee` in Decimal; maker legs pay $0.

Prices are contract prices in dollars (0..1), quantised to the venue's $0.001 tick.
`entry` on a trade is what one contract of the held side cost, so it is directly the
quantity the $0.62 cap bounds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal

import numpy as np

from openkal.fees import taker_fee

YES, NO = "YES", "NO"
TAKER, MAKER = "taker", "maker"
BUY, SELL = "buy", "sell"

LAG_S = 1.0                 # trigger-to-order arrival, the floor for every order
MIN_PRINT = 50.0            # contracts that must print at/through a passive limit
CAP = Decimal("0.62")       # hard ceiling on the cost of one contract of the held side
MAX_QTY = 50
EXIT_BUFFER_S = 20.0        # nothing is placed or filled in the last 20s of a window
EPS = 1e-6                  # float32 tolerance for at/through compares
ONE = Decimal("1")
ZERO = Decimal("0")
CENT = Decimal("0.01")


class CrossedBook(ValueError):
    """yes_bid >= yes_ask: no order may be priced or filled against this row."""


class NoQuote(LookupError):
    """No two-sided book at the requested time."""


def px(value: float | Decimal) -> Decimal:
    """A tape float as an exact contract price on the $0.001 tick."""
    return Decimal(str(round(float(value), 3)))


def opposite(side: str) -> str:
    return NO if side == YES else YES


def check_book(yes_bid: float, yes_ask: float) -> tuple[Decimal, Decimal]:
    """Validate one YES quote. Crossed raises first; then empty or one-sided."""
    if math.isfinite(yes_bid) and math.isfinite(yes_ask) and yes_bid >= yes_ask - EPS:
        raise CrossedBook(f"crossed book {yes_bid}/{yes_ask}")
    if not (math.isfinite(yes_bid) and math.isfinite(yes_ask)
            and 0.0 < yes_bid and yes_ask < 1.0):
        raise NoQuote(f"no two-sided book {yes_bid}/{yes_ask}")
    return px(yes_bid), px(yes_ask)


def entry_price(side: str, style: str, yes_bid: float, yes_ask: float) -> Decimal:
    """Cost of buying one contract of `side`: taker crosses, maker joins its own bid."""
    b, a = check_book(yes_bid, yes_ask)
    if side == YES:
        return a if style == TAKER else b
    if side == NO:
        return ONE - b if style == TAKER else ONE - a
    raise ValueError(side)


def exit_price(side: str, style: str, yes_bid: float, yes_ask: float) -> Decimal:
    """Proceeds of selling one contract of `side`: taker hits, maker joins its own ask."""
    b, a = check_book(yes_bid, yes_ask)
    if side == YES:
        return b if style == TAKER else a
    if side == NO:
        return ONE - a if style == TAKER else ONE - b
    raise ValueError(side)


def print_rule(side: str, action: str, limit: Decimal) -> tuple[bool, float, bool]:
    """Which prints trade against a resting order: (taker_yes, yes_level, at_or_above).

    Kalshi stamps every print with the YES price and the aggressor's side. A resting
    NO bid at L is the same order as a resting YES offer at 1 - L, so it is lifted by
    YES-buying aggressors at a YES price >= 1 - L; the other three cases mirror that.
    """
    lim = float(limit)
    if side == YES and action == BUY:
        return False, lim, False          # hit by YES sellers at <= L
    if side == YES and action == SELL:
        return True, lim, True            # lifted by YES buyers at >= L
    if side == NO and action == BUY:
        return True, 1.0 - lim, True      # lifted by YES buyers at >= 1 - L
    if side == NO and action == SELL:
        return False, 1.0 - lim, False    # hit by YES sellers at <= 1 - L
    raise ValueError((side, action))


def post_only_ok(side: str, action: str, limit: Decimal, yes_bid: float,
                 yes_ask: float) -> bool:
    """True when a resting order at `limit` would not cross the book it is sent into."""
    b, a = check_book(yes_bid, yes_ask)
    if side == YES:
        return limit < a if action == BUY else limit > b
    no_bid, no_ask = ONE - a, ONE - b
    return limit < no_ask if action == BUY else limit > no_bid


@dataclass(frozen=True)
class Tape:
    """One contract's top-of-book rows and print tape, sorted by time."""

    book_ts: np.ndarray
    yes_bid: np.ndarray
    yes_ask: np.ndarray
    trade_ts: np.ndarray
    trade_px: np.ndarray        # YES price of each print
    trade_sz: np.ndarray
    trade_taker_yes: np.ndarray  # 1 when the aggressor bought YES
    close_ts: float
    result_yes: int | None = None  # 1 when YES settled in the money

    @classmethod
    def from_lane(cls, lane, tid: int, close_ts: float, result_yes: int | None) -> "Tape":
        """Slice one ticker out of an `alpha_discover.Lane`-shaped object (no copy)."""
        a, b = int(lane.book_bounds[tid]), int(lane.book_bounds[tid + 1])
        c, d = int(lane.trade_bounds[tid]), int(lane.trade_bounds[tid + 1])
        return cls(lane.ts[a:b], lane.bid[a:b], lane.ask[a:b], lane.trade_ts[c:d],
                   lane.trade_px[c:d], lane.trade_sz[c:d], lane.trade_ty[c:d],
                   float(close_ts), result_yes)

    def row_at(self, ts: float) -> tuple[float, float]:
        """The last book row at or before `ts`, unvalidated."""
        j = int(np.searchsorted(self.book_ts, ts, "right")) - 1
        if j < 0:
            raise NoQuote(f"no book row at or before {ts}")
        return float(self.yes_bid[j]), float(self.yes_ask[j])

    def quote(self, ts: float) -> tuple[float, float]:
        """The book an order arriving at `ts` meets. Never walks back past a bad row."""
        b, a = self.row_at(ts)
        check_book(b, a)
        return b, a

    def next_valid_quote(self, ts: float, until: float) -> tuple[float, tuple[float, float]] | None:
        """First valid two-sided row at or after `ts` (forward only, for forced exits)."""
        j = max(0, int(np.searchsorted(self.book_ts, ts, "right")) - 1)
        while j < len(self.book_ts) and float(self.book_ts[j]) <= until:
            b, a = float(self.yes_bid[j]), float(self.yes_ask[j])
            try:
                check_book(b, a)
                return max(ts, float(self.book_ts[j])), (b, a)
            except (CrossedBook, NoQuote):
                j += 1
        return None

    def print_fill(self, t0: float, t1: float, side: str, action: str, limit: Decimal,
                   need: float = MIN_PRINT) -> float | None:
        """Time the `need`-th contract printed at/through a resting order in (t0, t1]."""
        if need < MIN_PRINT:
            raise ValueError(f"print-through below {MIN_PRINT:g} contracts")
        if t1 <= t0:
            return None
        lo = int(np.searchsorted(self.trade_ts, t0, "right"))
        hi = int(np.searchsorted(self.trade_ts, t1, "right"))
        if hi <= lo:
            return None
        taker_yes, level, above = print_rule(side, action, limit)
        p = self.trade_px[lo:hi]
        hit = (self.trade_taker_yes[lo:hi] == 1) == taker_yes
        hit &= (p >= level - EPS) if above else (p <= level + EPS)
        if not hit.any():
            return None
        cum = np.cumsum(np.where(hit, self.trade_sz[lo:hi], 0.0))
        if cum[-1] < need - EPS:
            return None
        return float(self.trade_ts[lo + int(np.argmax(cum >= need - EPS))])

    def side_won(self, side: str) -> bool:
        if self.result_yes is None:
            raise ValueError("unsettled tape")
        return bool(self.result_yes) == (side == YES)


@dataclass
class Leg:
    role: str           # entry / tp / lock / stop / barrier / settle
    side: str
    action: str
    style: str
    price: Decimal
    qty: int
    ts: float
    fee: Decimal


@dataclass
class Trade:
    side: str
    entry_style: str
    qty: int
    trigger_ts: float
    order_ts: float
    entry: Decimal               # cost of one contract of `side`
    reason: str                  # tp / lock / stop / barrier_taker / settle
    pnl: Decimal
    fees: Decimal
    exit_ts: float
    exit_value: Decimal          # per-contract value realised on the way out
    legs: list[Leg] = field(default_factory=list)
    crossed: int = 0             # fills against a crossed book: 0 by construction
    requotes: int = 0            # forced exits that waited for a valid row

    @property
    def won(self) -> bool:
        return self.pnl > 0


@dataclass(frozen=True)
class Skip:
    reason: str     # crossed / no_quote / cost_band / unfilled / too_late / post_only


def _confirmed_breach(ts: np.ndarray, bad: np.ndarray, confirm_s: float,
                      end_ts: float) -> float | None:
    """Time a run of adverse rows has lasted `confirm_s` unbroken; None if never.

    A run lasts from its first adverse row until the next non-adverse row, or until
    `end_ts` when the book never recovers inside the segment.
    """
    idx = np.flatnonzero(bad)
    if not len(idx):
        return None
    if confirm_s <= 0:
        return float(ts[idx[0]])
    edges = np.flatnonzero(np.diff(np.concatenate(([0], bad.astype(np.int8), [0]))))
    for s0, e0 in zip(edges[::2], edges[1::2]):
        start = float(ts[s0])
        # The run holds until the next non-adverse row (or the end of the segment).
        end = float(ts[e0]) if e0 < len(ts) else end_ts
        if end - start >= confirm_s - EPS:
            return start + confirm_s
    return None


def run_trade(tape: Tape, *, side: str, trigger_ts: float, entry_style: str, qty: int,
              lag_s: float = LAG_S, patience_s: float = 5.0, max_cost: Decimal = CAP,
              cost_lo: Decimal | None = None, cost_hi: Decimal | None = None,
              tp_c: Decimal | None = None, lock_c: Decimal | None = None,
              stop_c: Decimal | None = None, stop_confirm_s: float = 2.0,
              hold_s: float | None = None, barrier_exit: str = "settle",
              scratch_by_s: float | None = None, scratch_c: Decimal = ZERO,
              min_print: float = MIN_PRINT, entry_limit: Decimal | None = None) -> Trade | Skip:
    """One trade from trigger to flat, under the canonical fill rules.

    Entry: taker crosses the book at arrival (fee charged), maker joins the side's
    bid at arrival and waits up to `patience_s` for the print-through. A maker may
    instead rest at `entry_limit`, a price in held-side terms: it must not cross the
    book it arrives into (else Skip("post_only")), costs that price when it fills,
    and still has to clear the same cap, band and 50-contract print-through.
    Exits, raced on the same tape, earliest wins, take-profit and lock are
    one-cancels-other:
      tp_c    $0-fee post-only sell of the held side at entry + tp_c;
      lock_c  $0-fee post-only buy of the opposite side at 1 - entry - lock_c, which
              locks YES + NO = $1 at settlement; like every buy leg it must cost
              <= CAP, or it is never placed (an unfilled lock leaves full expiry risk);
              both are sent on the entry fill and arrive `lag_s` after it;
      stop_c  confirmed adverse move of the side's taker exit price; the exit order
              arrives `lag_s` later and pays the taker fee;
      hold_s  time barrier at fill + hold_s; `scratch_by_s` also puts one at
              close - scratch_by_s. At the barrier, `barrier_exit` is:
                settle   hold the binary ($0 exit fee);
                taker    cross out at the book `lag_s` later (taker fee);
                scratch  a $0-fee post-only sell of the held side, arriving `lag_s`
                         later, at max(entry - scratch_c, side's bid + 1c), filled on
                         the usual print-through; unfilled, the position settles.
    Unfilled at the end, the position settles: $1 or $0 a contract, no exit fee.
    """
    if lag_s < LAG_S - EPS:
        raise ValueError(f"arrival lag {lag_s}s is below the {LAG_S:g}s floor")
    if min_print < MIN_PRINT:
        raise ValueError(f"print-through {min_print} is below {MIN_PRINT:g}")
    if max_cost > CAP:
        raise ValueError(f"cost ceiling {max_cost} is above {CAP}")
    if side not in (YES, NO) or entry_style not in (TAKER, MAKER):
        raise ValueError((side, entry_style))
    if barrier_exit not in ("settle", "taker", "scratch"):
        raise ValueError(f"barrier_exit {barrier_exit!r}")
    if entry_limit is not None and entry_style != MAKER:
        raise ValueError("entry_limit is a resting bid: it needs entry_style='maker'")
    qty = min(int(qty), MAX_QTY)
    if qty <= 0:
        return Skip("qty")
    order_ts = trigger_ts + lag_s
    last_ts = tape.close_ts - EXIT_BUFFER_S
    if order_ts >= last_ts:
        return Skip("too_late")
    try:
        b, a = tape.quote(order_ts)
    except CrossedBook:
        return Skip("crossed")
    except NoQuote:
        return Skip("no_quote")
    cost = entry_price(side, entry_style, b, a)
    if entry_limit is not None:
        if not post_only_ok(side, BUY, entry_limit, b, a):
            return Skip("post_only")
        cost = entry_limit
    if cost <= 0 or cost > max_cost:
        return Skip("cost_band")
    if (cost_lo is not None and cost < cost_lo) or (cost_hi is not None and cost > cost_hi):
        return Skip("cost_band")

    fee_in = ZERO
    if entry_style == TAKER:
        fill_ts = order_ts
        fee_in = taker_fee(cost, qty)
    else:
        fill_ts = tape.print_fill(order_ts, min(order_ts + patience_s, last_ts), side, BUY,
                                  cost, need=min_print)
        if fill_ts is None:
            return Skip("unfilled")
    legs = [Leg("entry", side, BUY, entry_style, cost, qty, fill_ts, fee_in)]

    # Resting exits are orders too: sent on the entry fill, they arrive `lag_s` later and
    # are checked post-only against the book they arrive into.
    deadline = last_ts if hold_s is None else min(last_ts, fill_ts + hold_s)
    if scratch_by_s is not None:
        deadline = min(deadline, tape.close_ts - scratch_by_s)
    place_ts = fill_ts + lag_s
    try:
        fb, fa = tape.quote(place_ts) if place_ts < deadline else (None, None)
    except (CrossedBook, NoQuote):
        fb = fa = None
    resting: list[tuple[str, str, str, Decimal]] = []
    if tp_c is not None:
        tp_px = cost + tp_c
        if tp_px < ONE:
            resting.append(("tp", side, SELL, tp_px))
    if lock_c is not None:
        lock_px = ONE - cost - lock_c
        if 0 < lock_px <= CAP:
            resting.append(("lock", opposite(side), BUY, lock_px))
    events: list[tuple[float, int, str, object]] = []
    for prio, (role, s, act, lim) in enumerate(resting):
        if fb is None or not post_only_ok(s, act, lim, fb, fa):
            continue  # post-only would reject it: the leg is never live
        t = tape.print_fill(place_ts, deadline, s, act, lim, need=min_print)
        if t is not None:
            events.append((t, prio, role, (s, act, lim)))

    if stop_c is not None:
        lo = int(np.searchsorted(tape.book_ts, fill_ts, "right"))
        hi = int(np.searchsorted(tape.book_ts, deadline, "right"))
        seg_ts = tape.book_ts[lo:hi]
        sb = tape.yes_bid[lo:hi].astype(np.float64)
        sa = tape.yes_ask[lo:hi].astype(np.float64)
        valid = np.isfinite(sb) & np.isfinite(sa) & (sb > 0) & (sa < 1) & (sb < sa - EPS)
        mark = sb if side == YES else 1.0 - sa      # the side's taker exit price
        bad = valid & (mark <= float(cost - stop_c) + EPS)
        # An invalid row neither starts nor breaks a run: it is carried as "unknown"
        # by dropping it from the segment before the run test.
        t_conf = _confirmed_breach(seg_ts[valid], bad[valid], stop_confirm_s, deadline)
        if t_conf is not None and t_conf + lag_s < deadline:
            events.append((t_conf, 9, "stop", None))

    events.sort(key=lambda e: (e[0], e[1]))
    requotes = 0
    if events:
        t, _prio, role, payload = events[0]
        if role in ("tp", "lock"):
            s, act, lim = payload
            legs.append(Leg(role, s, act, MAKER, lim, qty, t, ZERO))
            value = lim if role == "tp" else ONE - lim   # lock: the pair pays $1
            pnl = (value - cost) * qty - fee_in
            return Trade(side, entry_style, qty, trigger_ts, order_ts, cost, role, pnl,
                         fee_in, t, value, legs)
        arrive = t + lag_s
        q = _forced_quote(tape, arrive, last_ts)
        if q is not None:
            t_x, (qb, qa), waited = q
            value = exit_price(side, TAKER, qb, qa)
            fee_out = taker_fee(value, qty)
            legs.append(Leg("stop", side, SELL, TAKER, value, qty, t_x, fee_out))
            pnl = (value - cost) * qty - fee_in - fee_out
            return Trade(side, entry_style, qty, trigger_ts, order_ts, cost, "stop", pnl,
                         fee_in + fee_out, t_x, value, legs, requotes=int(waited))
    if deadline < last_ts and barrier_exit == "scratch":
        place2 = deadline + lag_s
        try:
            qb, qa = tape.quote(place2)
            px = max(cost - scratch_c, exit_price(side, TAKER, qb, qa) + CENT)
            ok = px < ONE and post_only_ok(side, SELL, px, qb, qa)
        except (CrossedBook, NoQuote):
            ok = False
        t = tape.print_fill(place2, last_ts, side, SELL, px, need=min_print) if ok else None
        if t is not None:
            legs.append(Leg("scratch", side, SELL, MAKER, px, qty, t, ZERO))
            pnl = (px - cost) * qty - fee_in
            return Trade(side, entry_style, qty, trigger_ts, order_ts, cost, "scratch", pnl,
                         fee_in, t, px, legs)
    if deadline < last_ts and barrier_exit == "taker":
        q = _forced_quote(tape, deadline + lag_s, last_ts)
        if q is not None:
            t_x, (qb, qa), waited = q
            value = exit_price(side, TAKER, qb, qa)
            fee_out = taker_fee(value, qty)
            legs.append(Leg("barrier", side, SELL, TAKER, value, qty, t_x, fee_out))
            pnl = (value - cost) * qty - fee_in - fee_out
            return Trade(side, entry_style, qty, trigger_ts, order_ts, cost, "barrier_taker",
                         pnl, fee_in + fee_out, t_x, value, legs, requotes=int(waited))
    value = ONE if tape.side_won(side) else ZERO
    legs.append(Leg("settle", side, SELL, "settle", value, qty, tape.close_ts, ZERO))
    pnl = (value - cost) * qty - fee_in
    return Trade(side, entry_style, qty, trigger_ts, order_ts, cost, "settle", pnl, fee_in,
                 tape.close_ts, value, legs, requotes=requotes)


def _forced_quote(tape: Tape, ts: float, until: float):
    """Book for a taker exit arriving at `ts`; waits forward past a bad row, never back."""
    try:
        return ts, tape.quote(ts), False
    except (CrossedBook, NoQuote):
        nxt = tape.next_valid_quote(ts, until)
        if nxt is None:
            return None
        return nxt[0], nxt[1], True


SPOT_STALE_S = 5.0          # a spot reference older than this cannot arm the adverse-move exit


def _adverse_spot(spot, side: str, t0: float, t1: float, bps: float) -> float | None:
    """First spot print in (t0, t1] at least `bps` against a held YES (down) or NO (up), else None.

    The reference is the last print at or before `t0`. Without a recent reference there is no
    adverse-move exit; the timeout still applies.
    """
    if spot is None:
        return None
    ts, value = spot
    j = int(np.searchsorted(ts, t0, "right")) - 1
    if j < 0 or t0 - float(ts[j]) > SPOT_STALE_S:
        return None
    hi = int(np.searchsorted(ts, t1, "right"))
    move = np.asarray(value[j + 1:hi], dtype=np.float64) / float(value[j]) - 1.0
    bad = move <= -bps * 1e-4 + 1e-12 if side == YES else move >= bps * 1e-4 - 1e-12
    return float(ts[j + 1 + int(np.argmax(bad))]) if bad.any() else None


def run_two_sided_mm(tape: Tape, *, trigger_ts: float, qty: int, hedge_timeout_s: float,
                     min_spread: Decimal = Decimal("0.03"), spot=None, adverse_bps: float = 6.0,
                     lag_s: float = LAG_S, patience_s: float = 15.0, max_cost: Decimal = CAP,
                     max_pair_cost: Decimal = Decimal("0.97"), min_print: float = MIN_PRINT) -> Trade | Skip:
    """Quote both sides of one window: a YES bid at the YES bid and a NO bid at 1 - the YES ask.

    Both $0-fee post-only legs arrive `lag_s` after the trigger on the book they meet, each at the
    price the side's own bid already has, so neither can cross. Each fills on its own >= 50-contract
    print-through (`min_print`); the first fill must come within `patience_s`, else nothing trades.
      * both fill: the pair is locked, holds to settlement and pays the spread (pair cost <= `max_pair_cost`);
      * one fills: the other leg gets `hedge_timeout_s` from that fill, or until `spot` (an index
        tape `(ts, value)`) has moved `adverse_bps` against the filled leg, whichever comes first;
        a second fill by then locks the pair. If it has not filled, the cancel and the exit order
        land `lag_s` later and the lone leg crosses out as a taker (exact Decimal fee). A second-leg
        fill inside that one-lag window is not counted: the pessimistic reading of the race.
    The exit waits forward for the first valid book row if the arrival book is crossed or empty
    (`requotes`). A lone leg is never left to settle unless no exit can be placed before the last
    order second (reason "carried"). `Trade.entry` is the held leg's cost for a lone leg and the
    dearer leg for a pair: the $0.62 cap bounds each contract held.
    """
    if lag_s < LAG_S - EPS:
        raise ValueError(f"arrival lag {lag_s}s is below the {LAG_S:g}s floor")
    if min_print < MIN_PRINT:
        raise ValueError(f"print-through {min_print} is below {MIN_PRINT:g}")
    if max_cost > CAP:
        raise ValueError(f"cost ceiling {max_cost} is above {CAP}")
    if hedge_timeout_s <= 0:
        raise ValueError("hedge_timeout_s must be positive")
    qty = min(int(qty), MAX_QTY)
    if qty <= 0:
        return Skip("qty")
    order_ts = trigger_ts + lag_s
    last_ts = tape.close_ts - EXIT_BUFFER_S
    if order_ts >= last_ts:
        return Skip("too_late")
    try:
        b, a = tape.quote(order_ts)
    except CrossedBook:
        return Skip("crossed")
    except NoQuote:
        return Skip("no_quote")
    cost = {YES: entry_price(YES, MAKER, b, a), NO: entry_price(NO, MAKER, b, a)}
    if px(a) - px(b) < min_spread:
        return Skip("spread")
    if max(cost.values()) > max_cost:
        return Skip("cost_band")
    if sum(cost.values()) > max_pair_cost:
        return Skip("pair_cost")
    if not all(post_only_ok(s, BUY, cost[s], b, a) for s in (YES, NO)):
        return Skip("post_only")

    first_by = min(order_ts + patience_s, last_ts)
    fills = {s: tape.print_fill(order_ts, first_by, s, BUY, cost[s], need=min_print) for s in (YES, NO)}
    found = {s: t for s, t in fills.items() if t is not None}
    if not found:
        return Skip("unfilled")
    first = min(found, key=lambda s: (found[s], s != YES))
    other, t1 = opposite(first), found[first]
    timeout_ts = t1 + hedge_timeout_s
    breach_ts = _adverse_spot(spot, first, t1, timeout_ts, adverse_bps)
    event_ts = timeout_ts if breach_ts is None else breach_ts
    reason = "hedge_timeout" if breach_ts is None else "spot_breach"
    arrive = event_ts + lag_s                       # the cancel and the exit order land one lag after the decision
    t2 = tape.print_fill(order_ts, min(event_ts, last_ts), other, BUY, cost[other], need=min_print)
    legs = [Leg("entry", first, BUY, MAKER, cost[first], qty, t1, ZERO)]
    entry = max(cost.values())
    if t2 is not None:
        legs.append(Leg("entry", other, BUY, MAKER, cost[other], qty, t2, ZERO))
        yes_wins = tape.side_won(YES)
        legs += [Leg("settle", YES, SELL, "settle", ONE if yes_wins else ZERO, qty, tape.close_ts, ZERO),
                 Leg("settle", NO, SELL, "settle", ZERO if yes_wins else ONE, qty, tape.close_ts, ZERO)]
        pnl = (ONE - cost[YES] - cost[NO]) * qty
        return Trade(first, MAKER, qty, trigger_ts, order_ts, entry, "pair", pnl, ZERO, tape.close_ts, ONE, legs)
    q = _forced_quote(tape, arrive, last_ts) if arrive < last_ts else None
    if q is None:                                    # no exit can be placed: the lone leg settles
        value = ONE if tape.side_won(first) else ZERO
        legs.append(Leg("settle", first, SELL, "settle", value, qty, tape.close_ts, ZERO))
        return Trade(first, MAKER, qty, trigger_ts, order_ts, cost[first], "carried", (value - cost[first]) * qty,
                     ZERO, tape.close_ts, value, legs)
    t_x, (qb, qa), waited = q
    value = exit_price(first, TAKER, qb, qa)
    fee_out = taker_fee(value, qty)
    legs.append(Leg("hedge", first, SELL, TAKER, value, qty, t_x, fee_out))
    return Trade(first, MAKER, qty, trigger_ts, order_ts, cost[first], reason, (value - cost[first]) * qty - fee_out,
                 fee_out, t_x, value, legs, requotes=int(waited))


def stake_qty(cost: Decimal, stake: Decimal) -> int:
    """Contracts a dollar stake buys at `cost`, capped at MAX_QTY."""
    if cost <= 0:
        return 0
    return min(MAX_QTY, int(stake // cost))
