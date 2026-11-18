"""Track 10, the champion re-score: the registration is pinned, and every number in it is the one the old code uses."""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT / "scripts" / "track10_prereg.json"
PREREG_SHA256 = "0e55a88d67b47252cebe8c6c5ce9fa2fa907dcfe34389e2dc6fa6a64e654cd73"
SPEC = json.loads(SPEC_PATH.read_text())


class Registration(unittest.TestCase):
    def test_the_registered_file_has_not_drifted(self):
        self.assertEqual(hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest(), PREREG_SHA256)
        self.assertEqual(SPEC["version"], 1)
        self.assertNotIn("amendments", SPEC)

    def test_holm_counts_every_family_and_latency_including_the_two_that_cannot_be_scored(self):
        fams = SPEC["families"]
        self.assertEqual(sorted(fams), ["F2_taker_1s", "passive_mm_fade", "zobi_band_084", "zobi_xrp_carry"])
        self.assertEqual(SPEC["latencies_s"], [1, 2, 5])
        self.assertEqual(SPEC["holm"]["m"], len(fams) * len(SPEC["latencies_s"]))
        self.assertEqual([k for k, v in fams.items() if not v["scored"]], ["zobi_xrp_carry", "zobi_band_084"])
        self.assertIn("p = 1", SPEC["holm"]["meaning"])
        self.assertEqual(SPEC["holm"]["alpha"], 0.05)

    def test_the_two_unscored_families_are_blocked_by_the_canonical_ceiling_not_by_choice(self):
        from openkal import sim_core as S
        self.assertEqual(str(S.CAP), "0.62")
        for name in ("zobi_xrp_carry", "zobi_band_084"):
            self.assertIn("0.62", SPEC["families"][name]["reason_not_scored"])
        self.assertIn("0.84", SPEC["families"]["zobi_band_084"]["defined_by"])

    def test_f2_numbers_are_the_frozen_rule_of_f2_structural(self):
        from scripts import f2_structural as F
        self.assertEqual((F.Z_MIN, F.FLOW_MIN, F.REV_MIN, F.DELTA_BPS_MIN), (0.30, 0.10, 0.0, 3.0))
        self.assertEqual((F.COST_LO, F.COST_HI, F.CAP, F.T_LO, F.T_HI), (0.25, 0.60, 0.62, 180.0, 780.0))
        self.assertEqual((F.TP_C, F.STOP_C, F.STOP_CONFIRM_S, F.BARRIER_S, F.COOLDOWN, F.PER_WINDOW), (0.05, 0.07, 2.0, 90.0, 90.0, 3))
        f2 = SPEC["families"]["F2_taker_1s"]
        for text in ("|z| >= 0.30", "flow >= 0.10", "revision >= 0.0", "3 bp", "[0.25, 0.60]", "[180, 780]", "cooldown 90 s", "at most 3 per window"):
            self.assertIn(text, f2["trigger"])
        self.assertIn("[0.25, 0.62]", f2["entry"])
        for text in ("0.05", "0.07 confirmed for 2 s", "90 s"):
            self.assertIn(text, f2["exits"])

    def test_the_fade_is_the_frozen_champion_and_its_numbers_are_the_ones_passive_mm_fade_uses(self):
        from scripts import passive_mm_fade as P
        receipt = ROOT / "data" / "passive-mm-fade-is.json"
        if not receipt.exists():
            self.skipTest("the IS receipt is not on this machine")
        champion = json.loads(receipt.read_text())["champion"]["config"]
        self.assertEqual(champion, "flow_thr2_vq0-1_tp0.06_scratch_hold180")
        self.assertEqual((P.COOLDOWN, P.PER_WINDOW, str(P.COST_HI)), (90.0, 3, "0.55"))
        self.assertEqual(P.EXITS["scratch_hold180"], {"hold_s": 180.0, "barrier_exit": "scratch"})
        self.assertEqual((P.THRS[1], P.VOL_BANDS[0], str(P.TP_CS[1])), (2.0, (0.0, 1.0), "0.06"))
        fade = SPEC["families"]["passive_mm_fade"]
        for text in ("|z_obi| >= 2.0", "(NO: 1 - ask, YES: bid) at most 0.55", "cooldown 90 s, at most 3 per window"):
            self.assertIn(text, fade["trigger"])
        self.assertIn("scratch", fade["exits"])
        self.assertIn("180 s", fade["exits"])
        self.assertIn("0.06", fade["exits"])
        self.assertIn("assumption", fade)                                                  # the family name is an assumption, written down

    def test_the_freeze_is_the_last_window_of_the_frozen_cache(self):
        frozen = ROOT / "data" / "frozen-cache"
        if not (frozen / "markets.json").exists():
            self.skipTest("the frozen cache is not on this machine")
        from scripts import track4_common as T
        closes = T.window_closes(frozen)
        self.assertEqual((len(closes), closes[-1]), (SPEC["freeze"]["frozen_cache_windows"], SPEC["freeze"]["freeze_close_epoch"]))

    def test_the_decision_rules_are_written_before_any_data(self):
        self.assertEqual((SPEC["test"]["null"], SPEC["verdict"]["edge_confirmed"]),
                         ("mean <= 0", "Holm-rejected and at least 30 post-freeze trades"))
        self.assertIn("draws 10000, seed 0", SPEC["test"]["p_value"])
        self.assertIn("fewer than 30", SPEC["verdict"]["underpowered"])
        self.assertIn("5 folds", SPEC["walk_forward"]["folds"])
        self.assertTrue(SPEC["walk_forward"]["nothing_refit"].startswith("all thresholds, the fade's z-score normaliser and every exit are frozen"))


if __name__ == "__main__":
    unittest.main()
