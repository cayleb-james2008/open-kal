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


if __name__ == "__main__":
    unittest.main()
