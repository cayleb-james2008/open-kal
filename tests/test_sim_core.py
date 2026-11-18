"""The canonical simulator's invariants, on hand-built tapes.

The books sit far from 0.50 (0.60 / 0.63) on purpose: at a 0.50 mid a NO priced at
`yes_bid` and a NO priced at `1 - yes_bid` cost the same, and the bug that inflated
the F2 study would pass unnoticed.

Run: .venv/bin/python -m unittest tests/test_sim_core.py
"""
from __future__ import annotations

import sys
import unittest
from decimal import Decimal
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openkal import sim_core as S  # noqa: E402
from openkal.fees import taker_fee  # noqa: E402

D = Decimal
T0 = 1_000.0
CLOSE = T0 + 600.0


def tape(book, prints=(), result_yes=1, close=CLOSE) -> S.Tape:
    """book: [(ts, bid, ask)], prints: [(ts, yes_px, size, taker_yes)]."""
    bt = np.array([r[0] for r in book], dtype=np.float64)
    pr = list(prints)
    return S.Tape(
        bt, np.array([r[1] for r in book], dtype=np.float32),
        np.array([r[2] for r in book], dtype=np.float32),
        np.array([p[0] for p in pr], dtype=np.float64),
        np.array([p[1] for p in pr], dtype=np.float32),
        np.array([p[2] for p in pr], dtype=np.float32),
        np.array([p[3] for p in pr], dtype=np.int8), close, result_yes)


FLAT = [(T0 - 10.0, 0.60, 0.63)]


class BookSymmetry(unittest.TestCase):
    def test_entry_prices(self):
        self.assertEqual(S.entry_price(S.YES, S.TAKER, 0.60, 0.63), D("0.63"))
        self.assertEqual(S.entry_price(S.YES, S.MAKER, 0.60, 0.63), D("0.60"))
        self.assertEqual(S.entry_price(S.NO, S.TAKER, 0.60, 0.63), D("0.40"))
        self.assertEqual(S.entry_price(S.NO, S.MAKER, 0.60, 0.63), D("0.37"))

    def test_no_taker_entry_is_never_the_yes_bid(self):
        self.assertNotEqual(S.entry_price(S.NO, S.TAKER, 0.60, 0.63), D("0.60"))

    def test_exit_prices(self):
        self.assertEqual(S.exit_price(S.YES, S.TAKER, 0.60, 0.63), D("0.60"))
        self.assertEqual(S.exit_price(S.YES, S.MAKER, 0.60, 0.63), D("0.63"))
        self.assertEqual(S.exit_price(S.NO, S.TAKER, 0.60, 0.63), D("0.37"))
        self.assertEqual(S.exit_price(S.NO, S.MAKER, 0.60, 0.63), D("0.40"))

    def test_float32_tape_values_quantise_exactly(self):
        f = float(np.float32(0.23))
        self.assertNotEqual(f, 0.23)
        self.assertEqual(S.px(f), D("0.23"))
        self.assertEqual(S.px(float(np.float32(0.097))), D("0.097"))

    def test_no_taker_trade_charges_one_minus_bid(self):
        t = tape(FLAT, result_yes=0)
        tr = S.run_trade(t, side=S.NO, trigger_ts=T0, entry_style=S.TAKER, qty=50)
        self.assertIsInstance(tr, S.Trade)
        self.assertEqual(tr.entry, D("0.40"))
        self.assertEqual(tr.fees, taker_fee(D("0.40"), 50))
        self.assertEqual(tr.fees, D("0.84"))
        self.assertEqual(tr.pnl, (D("1") - D("0.40")) * 50 - D("0.84"))
        self.assertEqual(tr.reason, "settle")

    def test_no_taker_stop_sells_at_one_minus_ask(self):
        book = FLAT + [(T0 + 5.0, 0.66, 0.68), (T0 + 30.0, 0.60, 0.63)]
        t = tape(book, result_yes=0)
        tr = S.run_trade(t, side=S.NO, trigger_ts=T0, entry_style=S.TAKER, qty=10,
                         stop_c=D("0.07"), stop_confirm_s=2.0)
        self.assertEqual(tr.reason, "stop")
        self.assertEqual(tr.exit_value, D("0.32"))          # 1 - 0.68, not 0.66 or 0.68
        self.assertEqual(tr.exit_ts, T0 + 5.0 + 2.0 + 1.0)  # confirmed, then 1s arrival
        fee_in, fee_out = taker_fee(D("0.40"), 10), taker_fee(D("0.32"), 10)
        self.assertEqual(tr.fees, fee_in + fee_out)
        self.assertEqual(tr.pnl, (D("0.32") - D("0.40")) * 10 - fee_in - fee_out)

    def test_short_breach_does_not_stop(self):
        book = FLAT + [(T0 + 5.0, 0.66, 0.68), (T0 + 6.0, 0.60, 0.63)]
        tr = S.run_trade(tape(book, result_yes=0), side=S.NO, trigger_ts=T0,
                         entry_style=S.TAKER, qty=10, stop_c=D("0.07"), stop_confirm_s=2.0)
        self.assertEqual(tr.reason, "settle")


class Arrival(unittest.TestCase):
    def test_order_is_priced_one_second_after_the_trigger(self):
        book = FLAT + [(T0 + 0.5, 0.50, 0.53)]
        tr = S.run_trade(tape(book), side=S.YES, trigger_ts=T0, entry_style=S.TAKER, qty=1)
        self.assertEqual(tr.order_ts, T0 + 1.0)
        self.assertEqual(tr.entry, D("0.53"))   # the book at arrival, not at the trigger

    def test_lag_below_one_second_is_refused(self):
        with self.assertRaises(ValueError):
            S.run_trade(tape(FLAT), side=S.YES, trigger_ts=T0, entry_style=S.TAKER,
                        qty=1, lag_s=0.5)

    def test_cap_and_print_floor_cannot_be_loosened(self):
        with self.assertRaises(ValueError):
            S.run_trade(tape(FLAT), side=S.YES, trigger_ts=T0, entry_style=S.TAKER,
                        qty=1, max_cost=D("0.70"))
        with self.assertRaises(ValueError):
            S.run_trade(tape(FLAT), side=S.YES, trigger_ts=T0, entry_style=S.MAKER,
                        qty=1, min_print=20)

    def test_entry_above_cap_is_skipped(self):
        r = S.run_trade(tape(FLAT), side=S.YES, trigger_ts=T0, entry_style=S.TAKER, qty=1)
        self.assertEqual(r, S.Skip("cost_band"))        # YES ask 0.63 > 0.62

    def test_qty_is_capped(self):
        tr = S.run_trade(tape(FLAT), side=S.NO, trigger_ts=T0, entry_style=S.TAKER, qty=500)
        self.assertEqual(tr.qty, S.MAX_QTY)


class CrossedBooks(unittest.TestCase):
    def test_pricing_raises(self):
        for b, a in ((0.60, 0.60), (0.61, 0.60)):
            with self.assertRaises(S.CrossedBook):
                S.entry_price(S.NO, S.TAKER, b, a)
            with self.assertRaises(S.CrossedBook):
                S.exit_price(S.YES, S.MAKER, b, a)

    def test_trade_skips_and_never_walks_back(self):
        book = FLAT + [(T0 + 0.9, 0.62, 0.61)]
        r = S.run_trade(tape(book), side=S.NO, trigger_ts=T0, entry_style=S.TAKER, qty=5)
        self.assertEqual(r, S.Skip("crossed"))

    def test_empty_side_is_no_quote(self):
        r = S.run_trade(tape([(T0, float("nan"), 0.63)]), side=S.NO, trigger_ts=T0,
                        entry_style=S.TAKER, qty=5)
        self.assertEqual(r, S.Skip("no_quote"))


class PrintThrough(unittest.TestCase):
    def _maker_yes(self, prints):
        return S.run_trade(tape(FLAT, prints), side=S.YES, trigger_ts=T0,
                           entry_style=S.MAKER, qty=10, patience_s=10.0)

    def test_49_contracts_do_not_fill(self):
        r = self._maker_yes([(T0 + 2.0, 0.60, 30, 0), (T0 + 3.0, 0.59, 19, 0)])
        self.assertEqual(r, S.Skip("unfilled"))

    def test_50_contracts_fill_at_the_50th(self):
        tr = self._maker_yes([(T0 + 2.0, 0.60, 30, 0), (T0 + 3.0, 0.59, 20, 0)])
        self.assertEqual(tr.entry, D("0.60"))
        self.assertEqual(tr.legs[0].ts, T0 + 3.0)
        self.assertEqual(tr.legs[0].fee, D("0"))

    def test_wrong_aggressor_and_wrong_level_do_not_count(self):
        r = self._maker_yes([(T0 + 2.0, 0.60, 100, 1),     # a YES buyer cannot hit a bid
                             (T0 + 3.0, 0.61, 100, 0)])    # above the bid: not through it
        self.assertEqual(r, S.Skip("unfilled"))

    def test_prints_before_arrival_do_not_count(self):
        r = self._maker_yes([(T0 + 0.5, 0.60, 100, 0)])
        self.assertEqual(r, S.Skip("unfilled"))

    def test_no_maker_bid_is_filled_by_yes_buyers(self):
        # NO bid at 1 - 0.63 = 0.37 is a YES offer at 0.63: YES buyers at >= 0.63 lift it.
        good = S.run_trade(tape(FLAT, [(T0 + 2.0, 0.63, 50, 1)]), side=S.NO,
                           trigger_ts=T0, entry_style=S.MAKER, qty=10)
        self.assertEqual(good.entry, D("0.37"))
        bad = S.run_trade(tape(FLAT, [(T0 + 2.0, 0.63, 50, 0), (T0 + 2.5, 0.62, 80, 1)]),
                          side=S.NO, trigger_ts=T0, entry_style=S.MAKER, qty=10)
        self.assertEqual(bad, S.Skip("unfilled"))

    def test_print_fill_refuses_a_lower_floor(self):
        with self.assertRaises(ValueError):
            tape(FLAT).print_fill(T0, T0 + 5, S.YES, S.BUY, D("0.60"), need=49)


class EntryLimit(unittest.TestCase):
    """A caller-chosen resting bid, in held-side cost terms, instead of joining the bid."""

    def _run(self, prints, side, limit, book=FLAT, **kw):
        return S.run_trade(tape(book, prints), side=side, trigger_ts=T0, entry_style=S.MAKER,
                           qty=10, patience_s=10.0, entry_limit=D(limit), **kw)

    def test_yes_limit_below_the_bid_fills_only_when_the_tape_trades_through_it(self):
        self.assertEqual(self._run([(T0 + 2.0, 0.60, 100, 0)], S.YES, "0.55"), S.Skip("unfilled"))
        tr = self._run([(T0 + 2.0, 0.55, 30, 0), (T0 + 3.0, 0.54, 20, 0)], S.YES, "0.55")
        self.assertEqual((tr.entry, tr.legs[0].ts, tr.legs[0].fee), (D("0.55"), T0 + 3.0, D("0")))
        self.assertEqual(tr.pnl, (D("1") - D("0.55")) * 10)

    def test_no_limit_is_filled_by_yes_buyers_at_or_above_one_minus_the_limit(self):
        # A NO bid at 0.36 is a YES offer at 0.64: YES buyers at 0.63 do not reach it.
        self.assertEqual(self._run([(T0 + 2.0, 0.63, 100, 1)], S.NO, "0.36"), S.Skip("unfilled"))
        self.assertEqual(self._run([(T0 + 2.0, 0.64, 100, 0)], S.NO, "0.36"), S.Skip("unfilled"))
        tr = self._run([(T0 + 2.0, 0.64, 50, 1)], S.NO, "0.36")
        self.assertEqual((tr.entry, tr.side), (D("0.36"), S.NO))

    def test_a_limit_that_would_cross_the_book_is_refused_as_post_only(self):
        self.assertEqual(self._run([], S.YES, "0.63"), S.Skip("post_only"))     # = the YES ask
        self.assertEqual(self._run([], S.NO, "0.40"), S.Skip("post_only"))      # = the NO ask 1 - bid
        self.assertNotEqual(self._run([], S.YES, "0.62"), S.Skip("post_only"))

    def test_the_cost_cap_still_binds_a_limit(self):
        book = [(T0 - 10.0, 0.70, 0.75)]
        self.assertEqual(self._run([], S.YES, "0.65", book=book), S.Skip("cost_band"))

    def test_a_limit_is_maker_only(self):
        with self.assertRaises(ValueError):
            S.run_trade(tape(FLAT), side=S.YES, trigger_ts=T0, entry_style=S.TAKER, qty=10,
                        entry_limit=D("0.55"))


class TwoSidedMM(unittest.TestCase):
    """Rest a YES bid at the YES bid and a NO bid at 1 - the YES ask; a lone fill never rides to expiry."""

    BOOK = [(T0 - 10.0, 0.48, 0.53)]            # YES bid 0.48, NO bid 1 - 0.53 = 0.47: pair cost 0.95
    YES_FILL = (T0 + 2.0, 0.48, 50, 0)          # YES sellers at or through the YES bid
    NO_FILL = (T0 + 4.0, 0.53, 50, 1)           # YES buyers at or above the YES ask lift the NO bid

    def _run(self, prints, book=BOOK, result_yes=1, spot=None, **kw):
        args = dict(trigger_ts=T0, qty=10, hedge_timeout_s=15.0)
        args.update(kw)
        return S.run_two_sided_mm(tape(book, prints, result_yes=result_yes), spot=spot, **args)

    def test_both_legs_fill_and_the_locked_pair_pays_the_spread_whoever_wins(self):
        for result in (1, 0):
            tr = self._run([self.YES_FILL, self.NO_FILL], result_yes=result)
            self.assertEqual((tr.reason, tr.fees, tr.pnl), ("pair", D("0"), (D("1") - D("0.95")) * 10))
            self.assertEqual(tr.exit_ts, CLOSE)
        yes_leg, no_leg = [leg for leg in tr.legs if leg.role == "entry"]
        self.assertEqual((yes_leg.side, yes_leg.price), (S.YES, D("0.48")))
        self.assertEqual((no_leg.side, no_leg.price), (S.NO, D("0.47")))      # 1 - the YES ask, never the YES bid
        self.assertTrue(all(leg.fee == 0 for leg in tr.legs))
        self.assertEqual((tr.entry, tr.order_ts), (D("0.48"), T0 + S.LAG_S))

    def test_each_leg_needs_its_own_fifty_contracts(self):
        tr = self._run([(T0 + 2.0, 0.48, 49, 0), self.NO_FILL])               # YES leg is one contract short
        self.assertEqual((tr.side, tr.reason), (S.NO, "hedge_timeout"))
        self.assertEqual(self._run([(T0 + 2.0, 0.48, 49, 0), (T0 + 3.0, 0.53, 49, 1)]), S.Skip("unfilled"))
        self.assertEqual(self._run([(T0 + 0.5, 0.48, 500, 0)]), S.Skip("unfilled"))   # before arrival

    def test_a_lone_leg_is_crossed_out_at_the_timeout_with_the_exact_taker_fee(self):
        book = self.BOOK + [(T0 + 10.0, 0.45, 0.50)]
        tr = self._run([self.YES_FILL], book=book)
        self.assertEqual((tr.side, tr.reason, tr.exit_ts), (S.YES, "hedge_timeout", T0 + 2.0 + 15.0 + S.LAG_S))
        self.assertEqual(tr.exit_value, D("0.45"))                            # the YES bid it can hit
        self.assertEqual(tr.fees, taker_fee(D("0.45"), 10))
        self.assertEqual(tr.pnl, (D("0.45") - D("0.48")) * 10 - taker_fee(D("0.45"), 10))

    def test_a_lone_no_leg_exits_at_one_minus_the_yes_ask(self):
        book = self.BOOK + [(T0 + 10.0, 0.45, 0.50)]
        tr = self._run([self.NO_FILL], book=book, result_yes=0)               # NO would have won: still exits
        self.assertEqual((tr.side, tr.exit_value, tr.reason), (S.NO, D("0.50"), "hedge_timeout"))
        self.assertEqual(tr.pnl, (D("0.50") - D("0.47")) * 10 - taker_fee(D("0.50"), 10))

    def test_a_spot_move_of_six_bps_against_the_leg_exits_early(self):
        spot = (np.array([T0, T0 + 1, T0 + 3, T0 + 5, T0 + 8]), np.array([10000.0, 10000.0, 10005.9, 10006.0, 10011.0]))
        tr = self._run([self.NO_FILL], spot=spot)                             # NO filled at T0+4: the reference is 10005.9
        self.assertEqual(tr.reason, "hedge_timeout")                          # +5.1 bps from there (11 bps from the start)
        up = (np.array([T0, T0 + 3, T0 + 6]), np.array([10000.0, 10000.0, 10006.0]))
        tr = self._run([(T0 + 2.0, 0.53, 50, 1)], spot=up)                    # filled at T0+2, ref 10000
        self.assertEqual((tr.reason, tr.exit_ts), ("spot_breach", T0 + 6.0 + S.LAG_S))
        short = (np.array([T0, T0 + 3, T0 + 6]), np.array([10000.0, 10000.0, 10005.9]))
        self.assertEqual(self._run([(T0 + 2.0, 0.53, 50, 1)], spot=short).reason, "hedge_timeout")
        down = (np.array([T0, T0 + 3]), np.array([10000.0, 9000.0]))
        self.assertEqual(self._run([(T0 + 2.0, 0.53, 50, 1)], spot=down).reason, "hedge_timeout")  # helps a NO leg

    def test_a_falling_spot_hurts_a_yes_leg(self):
        down = (np.array([T0, T0 + 3]), np.array([10000.0, 9994.0]))
        tr = self._run([self.YES_FILL], spot=down)
        self.assertEqual((tr.side, tr.reason), (S.YES, "spot_breach"))

    def test_the_second_leg_must_fill_by_the_hedge_decision_not_during_the_cancel_lag(self):
        timeout = T0 + 2.0 + 15.0                                             # first fill + hedge_timeout_s
        self.assertEqual(self._run([self.YES_FILL, (timeout - 0.5, 0.53, 50, 1)]).reason, "pair")
        self.assertEqual(self._run([self.YES_FILL, (timeout, 0.53, 50, 1)]).reason, "pair")
        late = self._run([self.YES_FILL, (timeout + 0.5, 0.53, 50, 1)])       # the cancel has not landed yet: not counted
        self.assertEqual((late.reason, late.exit_ts), ("hedge_timeout", timeout + S.LAG_S))

    def test_a_stale_spot_reference_cannot_arm_the_adverse_exit(self):
        stale = (np.array([T0 - 100.0, T0 + 3.0]), np.array([10000.0, 9990.0]))     # last print before the fill is 102s old
        self.assertEqual(self._run([self.YES_FILL], spot=stale).reason, "hedge_timeout")
        fresh = (np.array([T0 + 1.0, T0 + 3.0]), np.array([10000.0, 9990.0]))
        self.assertEqual(self._run([self.YES_FILL], spot=fresh).reason, "spot_breach")

    def test_an_exit_into_a_crossed_book_waits_for_the_first_valid_row(self):
        book = self.BOOK + [(T0 + 15.0, 0.55, 0.50), (T0 + 40.0, 0.45, 0.50)]    # crossed from T0+15 to T0+40
        tr = self._run([self.YES_FILL], book=book)
        self.assertEqual((tr.reason, tr.exit_ts, tr.requotes, tr.exit_value), ("hedge_timeout", T0 + 40.0, 1, D("0.45")))

    def test_the_first_fill_must_come_inside_the_patience_window(self):
        self.assertEqual(self._run([(T0 + 17.0, 0.48, 50, 0)]), S.Skip("unfilled"))
        self.assertEqual(self._run([(T0 + 17.0, 0.48, 50, 0)], patience_s=20.0).side, S.YES)

    def test_skips(self):
        self.assertEqual(self._run([], book=[(T0 - 10.0, 0.50, 0.52)]), S.Skip("spread"))
        self.assertEqual(self._run([], book=[(T0 - 10.0, 0.50, 0.52)], min_spread=D("0")), S.Skip("pair_cost"))
        self.assertEqual(self._run([], book=[(T0 - 10.0, 0.65, 0.70)]), S.Skip("cost_band"))
        self.assertEqual(self._run([], book=[(T0 - 10.0, 0.55, 0.50)]), S.Skip("crossed"))
        self.assertEqual(self._run([], book=[(T0 + 5.0, 0.48, 0.53)]), S.Skip("no_quote"))
        self.assertEqual(self._run([], qty=0), S.Skip("qty"))
        self.assertEqual(S.run_two_sided_mm(tape(self.BOOK, close=T0 + 15.0), trigger_ts=T0, qty=10,
                                            hedge_timeout_s=15.0), S.Skip("too_late"))

    def test_floors_cannot_be_loosened(self):
        for kw in ({"lag_s": 0.5}, {"min_print": 49.0}, {"max_cost": D("0.63")}, {"hedge_timeout_s": 0.0}):
            with self.assertRaises(ValueError):
                self._run([], **kw)

    def test_a_lone_leg_with_no_time_to_exit_is_reported_as_carried(self):
        close = T0 + 25.0                                                      # last order second is close - 20 = T0 + 5
        tr = S.run_two_sided_mm(tape(self.BOOK, [(T0 + 2.0, 0.48, 50, 0)], close=close), trigger_ts=T0, qty=10,
                                hedge_timeout_s=15.0)
        self.assertEqual((tr.reason, tr.exit_ts), ("carried", close))
        self.assertEqual(tr.pnl, (D("1") - D("0.48")) * 10)


class PairLock(unittest.TestCase):
    """A NO bid at 1 - c - L is the same resting order as a YES offer at c + L."""

    PRINTS = [(T0 + 4.0, 0.44, 50, 1), (T0 + 8.0, 0.46, 50, 1)]
    BOOK = [(T0 - 10.0, 0.39, 0.40)]   # YES taker entry 0.40

    def _run(self, **kw):
        return S.run_trade(tape(self.BOOK, self.PRINTS, result_yes=0), side=S.YES,
                           trigger_ts=T0, entry_style=S.TAKER, qty=20, **kw)

    def test_lock_at_l_equals_take_profit_at_l(self):
        lock = self._run(lock_c=D("0.06"))
        tp = self._run(tp_c=D("0.06"))
        self.assertEqual((lock.exit_ts, lock.pnl), (tp.exit_ts, tp.pnl))
        self.assertEqual(lock.legs[-1].side, S.NO)
        self.assertEqual(lock.legs[-1].price, D("0.54"))   # 1 - 0.40 - 0.06

    def test_lower_lock_pre_empts_the_take_profit(self):
        tr = self._run(lock_c=D("0.04"), tp_c=D("0.06"))
        self.assertEqual(tr.reason, "lock")
        self.assertEqual(tr.exit_ts, T0 + 4.0)
        self.assertEqual(tr.pnl, D("0.04") * 20 - taker_fee(D("0.40"), 20))
        self.assertEqual(tr.legs[-1].fee, D("0"))

    def test_unfilled_lock_settles_without_an_exit_fee(self):
        tr = self._run(lock_c=D("0.08"))
        self.assertEqual(tr.reason, "settle")
        self.assertEqual(tr.fees, taker_fee(D("0.40"), 20))
        self.assertEqual(tr.pnl, -D("0.40") * 20 - tr.fees)


class ExitLegArrival(unittest.TestCase):
    """A take-profit or lock is an order too: it arrives 1s after the fill that sends it."""

    BOOK = [(T0 - 10.0, 0.39, 0.40)]   # YES taker entry 0.40, fill at T0 + 1

    def _run(self, prints, **kw):
        return S.run_trade(tape(self.BOOK, prints, result_yes=0), side=S.YES,
                           trigger_ts=T0, entry_style=S.TAKER, qty=20, **kw)

    def test_print_inside_the_first_second_after_the_fill_does_not_count(self):
        for kw in ({"tp_c": D("0.05")}, {"lock_c": D("0.05")}):
            tr = self._run([(T0 + 1.5, 0.45, 50, 1)], **kw)
            self.assertEqual(tr.reason, "settle", kw)

    def test_print_after_the_leg_arrives_fills_it(self):
        tr = self._run([(T0 + 1.5, 0.45, 50, 1), (T0 + 2.5, 0.45, 50, 1)], tp_c=D("0.05"))
        self.assertEqual((tr.reason, tr.exit_ts), ("tp", T0 + 2.5))


class LockLegCap(unittest.TestCase):
    """Owner ruling 2026-11-17: the $0.62 ceiling binds every buy leg, the lock too."""

    BOOK = [(T0 - 10.0, 0.29, 0.30)]   # YES taker entry 0.30

    def _run(self, lock_c, prints):
        return S.run_trade(tape(self.BOOK, prints, result_yes=0), side=S.YES,
                           trigger_ts=T0, entry_style=S.TAKER, qty=10, lock_c=D(lock_c))

    def test_lock_priced_above_the_cap_is_never_placed(self):
        tr = self._run("0.04", [(T0 + 5.0, 0.34, 50, 1)])   # lock NO bid 0.66 > 0.62
        self.assertEqual(tr.reason, "settle")

    def test_lock_at_the_cap_is_placed(self):
        tr = self._run("0.08", [(T0 + 5.0, 0.38, 50, 1)])   # lock NO bid 0.62
        self.assertEqual((tr.reason, tr.legs[-1].price), ("lock", D("0.62")))


class MakerScratch(unittest.TestCase):
    """A $0-fee pre-expiry exit: a post-only sell at entry (or one tick over the bid)."""

    def _yes(self, prints, **kw):
        return S.run_trade(tape([(T0 - 10.0, 0.39, 0.40)], prints, result_yes=0),
                           side=S.YES, trigger_ts=T0, entry_style=S.TAKER, qty=10,
                           barrier_exit="scratch", **kw)

    def test_scratch_rests_after_the_barrier_and_fills_at_breakeven(self):
        prints = [(T0 + 11.5, 0.40, 50, 1),    # before the scratch arrives (T0 + 12)
                  (T0 + 13.0, 0.40, 50, 1)]
        tr = self._yes(prints, hold_s=10.0)
        self.assertEqual((tr.reason, tr.exit_ts, tr.exit_value), ("scratch", T0 + 13.0, D("0.40")))
        self.assertEqual(tr.legs[-1].fee, D("0"))
        self.assertEqual(tr.pnl, -taker_fee(D("0.40"), 10))

    def test_unfilled_scratch_settles(self):
        tr = self._yes([(T0 + 11.5, 0.40, 50, 1)], hold_s=10.0)
        self.assertEqual(tr.reason, "settle")

    def test_scratch_by_close(self):
        tr = self._yes([(CLOSE - 55.0, 0.40, 50, 1)], scratch_by_s=60.0)
        self.assertEqual((tr.reason, tr.exit_ts), ("scratch", CLOSE - 55.0))

    def test_no_scratch_is_a_yes_bid_hit_by_yes_sellers(self):
        # NO bought at 1 - 0.60 = 0.40; NO taker exit is 1 - 0.63 = 0.37, so the scratch
        # rests at 0.40 = a YES bid at 0.60, hit by YES sellers at <= 0.60.
        t = tape([(T0 - 10.0, 0.60, 0.63)], [(T0 + 13.0, 0.60, 50, 0)], result_yes=1)
        tr = S.run_trade(t, side=S.NO, trigger_ts=T0, entry_style=S.TAKER, qty=10,
                         hold_s=10.0, barrier_exit="scratch")
        self.assertEqual((tr.reason, tr.exit_value), ("scratch", D("0.40")))

    def test_unknown_barrier_exit_is_refused(self):
        with self.assertRaises(ValueError):
            S.run_trade(tape(FLAT), side=S.NO, trigger_ts=T0, entry_style=S.TAKER, qty=1,
                        hold_s=10.0, barrier_exit="market")


if __name__ == "__main__":
    unittest.main()
