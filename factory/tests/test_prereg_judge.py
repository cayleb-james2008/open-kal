"""The factory judge's registration is pinned: the owner's bars, the corrections added to them, and the leak guard."""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT / "prereg_judge.json"
PREREG_SHA256 = "5c3017ce8ccf76692fb6c17b8dc22a2b1f062819085e751506e7ed5bd874dea4"
SPEC = json.loads(SPEC_PATH.read_text())


class Registration(unittest.TestCase):
    def test_the_registered_file_has_not_drifted(self):
        self.assertEqual(hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest(), PREREG_SHA256)
        self.assertEqual(SPEC["version"], 2)
        (v2,) = SPEC["amendments"]
        self.assertEqual((v2["version"], v2["supersedes_commit"]), (2, "5e5154b"))


    def test_the_owners_bars_are_all_there(self):
        b = SPEC["owner_bars"]
        self.assertIn("> 0 with the lower confidence bound > 0", b["oos_ev"])
        self.assertEqual(b["oos_trades"], ">= 200")
        self.assertIn("20%", b["max_drawdown"])
        self.assertIn("164", b["max_drawdown"])
        self.assertIn("doubled (10 s)", b["stress"])
        self.assertIn("1 cent", b["stress"])

    def test_the_multiple_testing_correction_and_the_leak_guard_are_written_down(self):
        a = SPEC["additions_to_the_bars"]
        self.assertIn("0.05 / N", a["corrected_lower_bound"])
        self.assertIn("window-clustered", a["corrected_lower_bound"])
        self.assertTrue(a["both_bounds_logged"])
        self.assertIn("never e itself", SPEC["walk_forward"]["evolution_leak_guard"])
        self.assertIn("never refit", SPEC["walk_forward"]["epoch"])

    def test_execution_numbers_are_the_ones_the_canonical_simulator_uses(self):
        from openkal import sim_core as S
        e = SPEC["execution_model"]
        self.assertEqual((e["latency_s"], e["stress_latency_s"], e["stress_slippage_usd"]), (5, 10, 0.01))
        self.assertEqual((e["entry_cost_ceiling"], e["max_contracts"]), (float(S.CAP), S.MAX_QTY))
        self.assertEqual(e["capital_usd"], 164.0)
        self.assertIn("0.07", e["fee"])

    def test_no_llm_and_the_deferred_families_are_declared(self):
        g = SPEC["generator"]
        self.assertTrue(g["no_llm"])
        self.assertIn("maker_quoting", g["deferred_families"])
        self.assertIn("PROVISIONAL", SPEC["indicator_registry"]["status"])
        self.assertEqual(SPEC["walk_forward"]["min_train_blocks"], 4)


if __name__ == "__main__":
    unittest.main()


class AmendmentV2(unittest.TestCase):
    def test_the_second_brief_adds_the_overfitting_controls_and_the_two_tiers(self):
        a = SPEC["judge_additions_v2"]
        self.assertIn("embargo = 1 window", a["purged_cv"])
        self.assertIn("above 0.5 cannot promote", a["pbo"])
        self.assertIn("DSR >= 0.95", a["deflated_sharpe"])
        self.assertIn(">= 50 trades", a["fresh_range_retest"])
        self.assertIn("protocol violation", a["count_every_candidate"])
        self.assertIn("top 100 candidates per hour", SPEC["screening_and_replay"]["tier_2_replay"])
        self.assertIn("only the replay may promote", SPEC["screening_and_replay"]["tier_2_replay"])
        self.assertIn("no usable driver", SPEC["layers"]["gpu"])
