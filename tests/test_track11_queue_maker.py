"""Track 11, the queue-aware maker idea: the registration is pinned and its numbers are the ones the simulator and the rule use."""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from openkal import sim_core as S
from scripts import rnd_common as RC

ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT / "scripts" / "track11_prereg.json"
PREREG_SHA256 = "dfc85d2d0453f5bfa60109a2abac3a175767fe093a95fc7f52d097b3804e76f4"
SPEC = json.loads(SPEC_PATH.read_text())


class Registration(unittest.TestCase):
    def test_the_registered_file_has_not_drifted(self):
        self.assertEqual(hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest(), PREREG_SHA256)
        self.assertEqual(SPEC["version"], 1)
        self.assertIsNone(SPEC["oos_scored"])
        self.assertIn("no trade of this track has been simulated", SPEC["status"])

    def test_the_split_is_sixty_percent_of_the_46_l2_windows_and_the_rest_is_sealed(self):
        sp = SPEC["split"]
        self.assertEqual((sp["is_windows"], sp["oos_windows"], sp["is_windows"] + sp["oos_windows"]), (27, 19, 46))
        self.assertEqual(int(0.6 * 46), 27)
        self.assertIn("l2_closes[int(0.6 * 46)]", sp["rule"])
        self.assertIn("never loaded, simulated or counted", sp["oos"])
        self.assertEqual(sp["cut_epoch"], 1795001400.0)

    def test_the_rule_numbers_are_in_the_text_and_agree_with_the_canonical_simulator(self):
        r = SPEC["rule"]
        self.assertIn("660, 540, 420 and 300 seconds", r["decision_points"])
        self.assertIn("[0.40, 0.60]", r["price_band"])
        self.assertIn("at most 100 contracts", r["queue_band"])
        self.assertIn("patience 100 s", r["order"])
        self.assertIn("65.58", r["order"])
        self.assertEqual(str(RC.STAKE), "65.58")
        self.assertEqual((S.MAX_QTY, S.LAG_S, str(S.CAP), S.MIN_PRINT), (50, 1.0, "0.62", 50.0))
        self.assertIn("at most 50 contracts", r["order"])
        self.assertIn("queue_model=True", r["all_orders_through"])
        self.assertIn("held to settlement", r["exit"])

    def test_the_patience_is_shorter_than_the_gap_between_decisions_so_orders_never_overlap(self):
        secs = [660, 540, 420, 300]
        self.assertTrue(all(a - b >= 100 for a, b in zip(secs, secs[1:])))

    def test_the_queue_model_text_matches_what_sim_core_implements(self):
        q = SPEC["queue_position_model"]
        for text in ("yes_bid_sz", "yes_ask_sz", "inside the spread has 0 ahead", "below the touch"):
            self.assertIn(text, q["queue_ahead"])
        self.assertIn("queue_ahead + own contracts", q["fill"])
        self.assertIn("strictly through", q["fill"])
        self.assertIn("50 contracts", q["fill"])
        self.assertTrue(hasattr(S.Tape, "queue_ahead") and hasattr(S.Tape, "queue_fill"))
        self.assertIn("queue_model", S.run_trade.__code__.co_varnames)

    def test_the_bar_is_the_earlier_tracks_bar_and_clearing_it_spends_nothing(self):
        bar = " ".join(SPEC["is_bar"]["needs_all_of"])
        for text in ("at least 30 fills", "net > 0", "profit factor >= 1.25", "at least 2 profitable lanes", "p < 0.05", "10000 draws, seed 0"):
            self.assertIn(text, bar)
        self.assertIn("never an OOS look", SPEC["is_bar"]["meaning"])
        self.assertEqual(len(SPEC["controls_descriptive"]), 3)


if __name__ == "__main__":
    unittest.main()
