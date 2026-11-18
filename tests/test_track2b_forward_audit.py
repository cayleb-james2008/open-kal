import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import track2b_forward_holdout as prior


class ForwardAudit(unittest.TestCase):
    def test_verdict_refuses_other_ticker_future_and_unavailable_rows(self):
        from scripts.track2b_forward_audit import gate_at

        ts = np.array([9.1, 9.4, 9.7, 9.8, 10.1])
        ids = np.array([0, 0, 1, 0, 0])
        available = np.array([1, 1, 1, 0, 1])
        allow = np.array([1, 0, 1, 1, 1])
        self.assertEqual(gate_at(ts, ids, available, allow, 0, 9.7), (False, 9.4))
        self.assertEqual(gate_at(ts, ids, available, allow, 0, 10.0), (None, 9.8))
        self.assertEqual(gate_at(ts, ids, available, allow, 2, 10.0), (None, None))

    def test_spent_forward_look_refuses_before_accessing_the_cache(self):
        with patch.object(prior, "EVOLVE_CACHE", Path("/nonexistent-cache")), self.assertRaisesRegex(
            RuntimeError, "already scored"
        ):
            prior.run_forward_holdout()


if __name__ == "__main__":
    unittest.main()
