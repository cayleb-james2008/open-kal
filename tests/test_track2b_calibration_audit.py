import unittest

import numpy as np

from scripts import track2b_calibration_audit as audit


class CalibrationAudit(unittest.TestCase):
    def test_one_parameter_platt_preserves_complements_and_fits_is(self):
        # 3/4 wins at 0.6 and 1/4 at 0.4 have the same optimum slope.
        p = np.array([.6] * 4 + [.4] * 4)
        y = np.array([1, 1, 1, 0, 1, 0, 0, 0])
        slope = audit.fit_platt(p, y)
        np.testing.assert_allclose(audit.platt(np.array([.4, .5, .6]), slope), [.25, .5, .75])
        self.assertEqual(audit.fit_platt(p, 1 - y), 0)

    def test_ece_and_bucket_counts_reflect_flipped_settlement(self):
        p = np.array([.42, .49, .55, .8])
        y = np.array([0, 1, 1, 1])
        observed = audit.metrics(p, y)
        self.assertAlmostEqual(observed["ece"], .185)
        self.assertEqual([(b["bin"], b["trades"], b["wins"]) for b in observed["buckets"]],
                         [(4, 2, 1), (5, 1, 1), (8, 1, 1)])
        y[2] = 0
        self.assertAlmostEqual(audit.metrics(p, y)["ece"], .21)

    def test_flat_null_has_only_sampling_noise(self):
        counts = np.array([2, 2])
        uniforms = np.array([[.1, .6, .1, .6], [.1, .2, .8, .9]])
        flat = audit.simulated_ece(np.full(4, .5), np.repeat([1, 2], counts), uniforms)
        np.testing.assert_allclose(flat, [0, .5])

    def test_reproduces_frozen_baseline_without_refitting_student_t(self):
        r = audit.audit_frozen()
        self.assertEqual((r["economics"]["is"]["trades"], r["economics"]["oos"]["trades"]), (44, 36))
        self.assertEqual(r["economics"]["oos"]["net_ex_xrp"], 131.71)
        self.assertEqual(round(r["models"]["historical_ridge_logistic_5f_control"]["oos"]["ece"], 4), .1581)


if __name__ == "__main__":
    unittest.main()
