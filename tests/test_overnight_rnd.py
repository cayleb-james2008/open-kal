import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import overnight_rnd_loop as loop, rnd_common as common
from scripts import track2c_true_l2_settle as l2


class TrueDepth(unittest.TestCase):
    def test_counts_only_distinct_settled_closes_with_joint_depth_and_microprice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = {f"BTC-{i}": {"close": 100 + i * 900, "y": y}
                    for i, y in enumerate((1, 0, 0, None, 1))}
            (root / "markets.json").write_text(json.dumps(rows))
            np.savez(root / "book_KXBTC15M.npz", tickers=list(rows), tid=[0, 0, 1, 2, 3, 4],
                     yes_bid_sz=[2, 2, 2, np.nan, 2, 2], yes_ask_sz=[2] * 6,
                     microprice=[.5, .5, np.nan, .5, .5, .5])
            got = l2.audit_l2_depth(root, as_of=3000)
            self.assertEqual(got["valid_l2_rows"], 4)
            self.assertEqual(got["l2_window_closes"], [100])

    def test_z_uses_past_window_and_ignores_future_or_unavailable_depth(self):
        from scripts.track2c_true_l2_settle import causal_z_l2_obi

        t = np.arange(7) * 2.
        obi = np.array([-.5, 0, .5, .25, -.25, .5, -.75])
        z = causal_z_l2_obi(t, obi)
        self.assertTrue(np.isnan(z[:5]).all())
        self.assertAlmostEqual(z[5], math.sqrt(2))
        obi[6] = .95
        self.assertEqual(causal_z_l2_obi(t, obi)[5], z[5])
        obi[2] = np.nan
        self.assertTrue(np.isnan(causal_z_l2_obi(t, obi)[5]))

    def test_check_only_does_not_spend_a_ready_holdout(self):
        with patch.object(l2, "audit_l2_depth", return_value={"windows_with_l2": 24}), \
                patch.object(l2, "run_full_l2_sweep") as sweep, \
                patch("sys.argv", ["track2c_true_l2_settle.py", "--check-only"]):
            self.assertEqual(l2.main(), 0)
            sweep.assert_not_called()

    def test_xrp_cannot_carry_a_passing_promotion(self):
        # The three other lanes together lose $1 even though two are profitable.
        result = {"net": 12, "pf": 1.8, "trades": 30, "profitable_lanes": 3,
                  "net_ex_best_lane": -1, "net_ex_best_window": 2,
                  "crossed_fills": 0, "max_entry_cost": .56,
                  "by_lane": {"KXBTC15M": {"net": 4}, "KXETH15M": {"net": 3},
                              "KXSOL15M": {"net": -8}, "KXXRP15M": {"net": 13}},
                  "net_ex_xrp": -1}
        gate = common.promotion_gate(result, .03)
        self.assertFalse(gate["net_ex_xrp_positive"])
        self.assertFalse(gate["pass"])


class ShiftTiming(unittest.TestCase):
    def test_slot_alignment_and_inclusive_deadline(self):
        # At :00..:05 the 300s slots stay on the same clock grid.
        with patch.object(loop.time, "time", side_effect=[900.7, 900.7, 900.7, 900.7, 1200]), \
                patch.object(loop.time, "sleep") as sleep, patch.object(loop, "log"), \
                patch.object(loop, "check_once", return_value={"status": "accumulating"}) as check:
            loop.run_loop(check_only=True, until=1200)
            self.assertEqual(check.call_count, 2)
            sleep.assert_called_once()
            self.assertAlmostEqual(sleep.call_args.args[0], 299.3)

    def test_expired_shift_never_reads_the_cache(self):
        with patch.object(loop.time, "time", return_value=1201), patch.object(loop, "log"), \
                patch.object(loop, "check_once") as check:
            loop.run_loop(check_only=True, until=1200)
            check.assert_not_called()


if __name__ == "__main__":
    unittest.main()
