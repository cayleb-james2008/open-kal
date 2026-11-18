"""Synthetic RTI tape tests; no market files or project entry points needed."""

import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from asian_rti import AsianRTI


def exact_tape():
    t = np.arange(600., 902.)
    # Two-unit moves up to window open, then fixes 101, 102, ..., 160.
    v = np.where(t <= 840, 100 + 2 * (t % 2), 100 + t - 840)
    return t, v


def venue_tape():
    t = np.arange(600., 841.) + .2
    history = [(s, 98 + i % 2, np.nan, 0) for i, s in enumerate(t)]
    # Missing raw prices in the window. Only the venue knows the whole sum.
    rows = history + [
        (841.05, 98, 98, 1), (841.05, 98, np.nan, 0),
        (855.08, 100, 100, 15), (869.07, 109, 100, 29),
        (870.02, 115, 100, 29),       # new spot, old count
        (870.07, 115, 100.5, 30), (870.07, 115, np.nan, 0),
        (870.81, 112, 100.5, 30),     # carried aggregate; NOT a new fix
        (871.02, 120, 100.5, 30), (871.05, 120, 3135 / 31, 31),
    ]
    return np.array(rows).T


class AsianRTITest(unittest.TestCase):
    def test_before_and_at_window_open(self):
        model = AsianRTI(*exact_tape())
        for trigger, variance in [(780, 320.), (840, 80.)]:  # r=120, 60
            with self.subTest(trigger=trigger):
                f = model.fair(trigger, 900, 100 + math.sqrt(variance))
                self.assertIsNotNone(f)
                self.assertEqual(f['sigma'], 2.)
                self.assertEqual(f['variance'], variance)
                self.assertEqual((f['mean'], f['spot'], f['last_rti_ts']), (100, 100, trigger))
                self.assertEqual((f['elapsed'], f['locked_sum']), (0, 0))
                self.assertAlmostEqual(f['p'], .15865525393145707, places=14)  # one-sigma tail

    def test_half_locked_right_endpoints_and_continuous_variance(self):
        t, v = exact_tape()
        # Recording duplicates must not make expensive ticks count more.
        copies = np.where(v > 120, 7, 1)
        model = AsianRTI(np.repeat(t, copies), np.repeat(v, copies))
        for trigger in (870, 870.25):
            with self.subTest(trigger=trigger):
                f = model.fair(trigger, 900, 122.75)
                self.assertIsNotNone(f)
                self.assertEqual((f['elapsed'], f['locked_sum']), (30, 3465))  # 101..130
                self.assertEqual((f['spot'], f['mean'], f['p']), (130, 122.75, .5))
                self.assertAlmostEqual(f['sigma'] ** 2, 3.4)  # 120 two-unit, 30 one-unit moves
                self.assertAlmostEqual(f['variance'], 3.4 * (900 - trigger)**3 / 10800)

    def test_venue_aggregate_covers_missing_prices_only_once_received(self):
        model = AsianRTI(*venue_tape())
        self.assertIsNone(model.fair(870.06, 900, 100))  # #30 has not arrived
        f = model.fair(870.07, 900, 100)
        self.assertIsNotNone(f)
        self.assertEqual((f['elapsed'], f['locked_sum'], f['spot']), (30, 3015, 115))
        f = model.fair(870.83, 900, 106.25)
        self.assertEqual((f['mean'], f['spot'], f['p']), (106.25, 112, .5))
        self.assertEqual(f['last_rti_ts'], 870.81)
        self.assertAlmostEqual(f['variance'], f['sigma']**2 * 29.17**3 / 10800)
        self.assertIsNone(model.fair(871.03, 900, 100))  # still #30, now need #31

    def test_suffix_invariance_even_within_same_second(self):
        tape = venue_tape()
        prefix = tape[:, tape[0] <= 870.83]
        expected = AsianRTI(*prefix).fair(870.83, 900, 105)
        self.assertIsNotNone(expected)
        # Contradictory aggregate and large RTI move one millisecond AFTER decision.
        shock = np.array([[870.831, 10000., 300., 30.]]).T
        extended = np.hstack((prefix, shock, tape[:, tape[0] > 870.83]))
        self.assertEqual(AsianRTI(*extended).fair(870.83, 900, 105), expected)
        t, v = exact_tape()
        mask = t <= 870.25
        self.assertEqual(AsianRTI(t[mask], v[mask]).fair(870.25, 900, 125),
                         AsianRTI(t, np.where(mask, v, 1e6)).fair(870.25, 900, 125))

    def test_raw_fixing_gap_and_conflicting_duplicate_are_unavailable(self):
        t, v = exact_tape()
        keep = t != 854
        # Just before/after a missing fix cannot substitute for the fixing.
        order = np.argsort(np.r_[t[keep], 853.99, 854.01])
        model = AsianRTI(np.r_[t[keep], 853.99, 854.01][order], np.r_[v[keep], 13, 14][order])
        self.assertIsNone(model.fair(870, 900, 100))
        i = np.searchsorted(t, 854)
        model = AsianRTI(np.insert(t, i, 854), np.insert(v, i, 999))
        self.assertIsNone(model.fair(870, 900, 100))

    def test_venue_window_mismatch_carry_and_conflict(self):
        tape = venue_tape()
        self.assertIsNone(AsianRTI(*tape).fair(871.03, 901, 100))  # not a venue quarter close
        wrong = tape.copy()
        wrong[2:, (wrong[0] >= 869.07) & (wrong[0] <= 870.81)] = [[100.5], [30]]
        self.assertIsNone(AsianRTI(*wrong).fair(870.83, 900, 105))  # #30 first seen too early
        wrong = tape.copy()
        wrong[2:, wrong[0] == 870.81] = [[200.], [30]]  # two different averages for #30
        self.assertIsNone(AsianRTI(*wrong).fair(870.83, 900, 105))
        # On a merged tape, unlabeled raw rows are not a replacement for absent venue fixes.
        t, v = exact_tape()
        self.assertIsNone(AsianRTI(t, v, np.full_like(t, np.nan), t * 0).fair(870, 900, 105))

    def test_stale_spot_sparse_history_and_gap_not_bridged_for_sigma(self):
        t, v = exact_tape()
        keep = (t <= 778) | (t >= 781)
        self.assertIsNone(AsianRTI(t[keep], v[keep]).fair(780, 900, 100))
        self.assertIsNone(AsianRTI(np.repeat(t[::2], 20), np.repeat(v[::2], 20)).fair(780, 900, 100))
        keep = (t < 695) | (t >= 711)
        v = np.where(t < 695, 100., 1000.)
        f = AsianRTI(t[keep], v[keep]).fair(780, 900, 1000)
        self.assertIsNotNone(f)
        self.assertEqual(f['sigma'], 0.)  # no one-second return across this jump

    def test_zero_variance_uses_greater_than_or_equal(self):
        t, v = exact_tape()
        x = 100.01
        flat = AsianRTI(t, np.full_like(t, x))
        for trigger in (780, 870, 900):
            with self.subTest(trigger=trigger):
                self.assertEqual(flat.fair(trigger, 900, x)['p'], 1.)
                self.assertEqual(flat.fair(trigger, 900, math.nextafter(x, math.inf))['p'], 0.)
        at_close = AsianRTI(t, v).fair(900, 900, 130.5)
        self.assertGreater(at_close['sigma'], 0.)
        self.assertEqual((at_close['elapsed'], at_close['variance'], at_close['p']), (60, 0., 1.))
        self.assertIsNone(flat.fair(900.01, 900, 100))

    def test_unusable_inputs(self):
        self.assertIsNone(AsianRTI([], []).fair(780, 900, 100))
        self.assertIsNone(AsianRTI([781], [100]).fair(780, 900, 100))
        t, v = exact_tape()
        self.assertIsNone(AsianRTI(t, v).fair(np.nan, 900, 100))
        with self.assertRaises(ValueError):
            AsianRTI(t, v[:-1])
        with self.assertRaises(ValueError):
            AsianRTI(t, v, k=t)


if __name__ == '__main__':
    unittest.main()
