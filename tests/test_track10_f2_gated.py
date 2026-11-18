"""The exploratory gated-F2 row: the calibrator is fitted on pre-freeze trades only and the desk's entry test decides what is kept."""
from __future__ import annotations

import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import numpy as np

from openkal.strategy import Params, exit_cost, taker_fee_per_contract
from scripts import track10_f2_gated as G


def rows(n, seed, shift=0.0, entry="0.40"):
    """Trades whose first feature predicts the win: positive z wins about 80% of the time, negative z about 20%."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        z = float(rng.normal(0.0, 1.0))
        win = rng.random() < (0.8 if z > 0 else 0.2) + shift
        out.append({"feat": [z, 0.1, 0.1, 5.0], "pnl": "1.5" if win else "-2.0", "entry": entry, "close": float(i), "lane": "KXBTC15M",
                    "reason": "settle", "fee_entry": "0", "fee_exit": "0", "crossed": 0, "requotes": 0})
    return out


class Gate(unittest.TestCase):
    def test_a_trade_the_calibrator_likes_passes_and_one_it_dislikes_does_not(self):
        pre = rows(400, 1)
        g = G.gate(pre, [{"feat": [2.0, 0.1, 0.1, 5.0], "pnl": "1", "entry": "0.40", "close": 0.0, "lane": "L"},
                         {"feat": [-2.0, 0.1, 0.1, 5.0], "pnl": "1", "entry": "0.40", "close": 1.0, "lane": "L"}])
        self.assertTrue(g["fitted"])
        self.assertEqual([r["feat"][0] for r in g["kept"]], [2.0])
        self.assertGreater(g["kept"][0]["p_side"], 0.6)

    def test_the_calibrator_is_fitted_on_the_pre_rows_only(self):
        pre, post = rows(300, 2), rows(100, 3)
        a = G.gate(pre, post)["kept"]
        flipped = [{**r, "pnl": "-2.0" if Decimal(r["pnl"]) > 0 else "1.5"} for r in post]            # the post outcomes are not read at all
        b = G.gate(pre, flipped)["kept"]
        self.assertEqual([r["p_side"] for r in a], [r["p_side"] for r in b])
        self.assertEqual([r["close"] for r in a], [r["close"] for r in b])

    def test_the_desks_entry_test_is_p_less_ask_less_exit_cost_less_the_taker_fee_against_the_larger_hurdle(self):
        pre = rows(300, 4)
        params = Params()
        g = G.gate(pre, rows(60, 5), params)
        self.assertEqual(g["hurdle"], float(max(Decimal(str(params.edge_hurdle)), Decimal(str(params.min_net_edge)))))
        for r in g["kept"]:
            ask = Decimal(r["entry"])
            net = Decimal(str(r["p_side"])) - ask - exit_cost(ask, params) - taker_fee_per_contract(ask)
            self.assertGreaterEqual(net, Decimal(str(g["hurdle"])))
            self.assertAlmostEqual(r["gate_net"], float(net))
        dearer = G.gate(pre, [{**r, "entry": "0.58"} for r in rows(60, 5)], params)["kept"]              # a dearer ask leaves less edge
        self.assertLess(len(dearer), len(g["kept"]))

    def test_the_hurdle_is_the_larger_of_the_two(self):
        self.assertEqual(G.gate(rows(300, 6), [], Params(edge_hurdle=0.05, min_net_edge=0.01))["hurdle"], 0.05)
        self.assertEqual(G.gate(rows(300, 6), [], Params(edge_hurdle=0.01, min_net_edge=0.03))["hurdle"], 0.03)

    def test_with_fewer_than_twenty_pre_trades_nothing_is_fitted_and_nothing_is_kept(self):
        g = G.gate(rows(19, 7), rows(50, 8))
        self.assertEqual((g["fitted"], g["kept"], g["fit_rows"]), (False, [], 19))
        self.assertTrue(G.gate(rows(20, 7), rows(50, 8))["fitted"])

    def test_a_losing_row_is_a_zero_target_and_a_zero_pnl_row_too(self):
        pre = rows(100, 9)
        z = [dict(r, pnl="0") for r in pre]                                                             # pnl exactly 0 is not a win
        self.assertEqual(G.gate(z, rows(10, 9))["kept"], [])                                            # an all-loss target gives p near 0: nothing passes

    def test_the_module_says_it_is_not_registered_and_the_family_names_are_marked(self):
        self.assertIn("EXPLORATORY and not registered", G.__doc__)
        self.assertIn("outside Holm", G.REASON)
        self.assertEqual(G.MIN_FIT, 20)


class Run(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.lines = []

    def tearDown(self):
        self.tmp.cleanup()

    def go(self, registered_net, registered_trades=40):
        pre = [dict(r, slice="pre") for r in rows(30, 11)]
        post = [dict(r, slice="post", close=100.0 + i) for i, r in enumerate(rows(40, 12))]
        reg = self.dir / "reg.json"
        reg.write_text(json.dumps({"cells": {"F2_taker_1s": {f"{lag:g}": {"post": {"trades": registered_trades, "net": registered_net}}
                                                             for lag in G.R.LATENCIES}}}))
        with patch.object(G.T, "pin_tape", lambda t: {}), patch.object(G.T, "window_closes", lambda t: [1.0, G.R.FREEZE_CLOSE + 900.0]), \
                patch.object(G.RC, "load_lanes", lambda: {n: object() for n in G.R.LANES}), patch.object(G.R, "load_pool", lambda n, d: {}), \
                patch.object(G.R, "f2_triggers", lambda n, l, p: []), patch.object(G.RC, "tape", lambda l, t: None), \
                patch.object(G.R, "score_cell", lambda trig, tapes, lag: (pre + post, {})), \
                patch.object(G.R, "ledger_append", lambda e: self.lines.append((e["status"], e["family"]))):
            return G.run(tape=self.dir / "t", pool_dir=self.dir / "p", out=self.dir / "o.json", registered=reg), post

    def test_the_run_is_once_only(self):
        out = self.dir / "o.json"
        out.write_text("{}")

        def never(*a, **k):
            raise AssertionError("the run went past its once-only check and touched real data")

        with patch.object(G.T, "pin_tape", never), patch.object(G.R, "ledger_append", never), self.assertRaisesRegex(RuntimeError, "made once"):
            G.run(out=out)

    def test_it_refuses_when_the_ungated_score_does_not_reproduce_the_registered_cell(self):
        post = [dict(r, slice="post") for r in rows(40, 12)]
        net = float(G.RC.summarize(post)["net"])
        with self.assertRaisesRegex(RuntimeError, "does not reproduce"):
            self.go(net + 1.0)
        with self.assertRaisesRegex(RuntimeError, "does not reproduce"):
            self.go(net, registered_trades=39)
        self.assertFalse((self.dir / "o.json").exists())

    def test_a_matching_run_logs_started_before_scored_marks_itself_exploratory_and_writes_the_receipt(self):
        post = [dict(r, slice="post") for r in rows(40, 12)]
        net = float(G.RC.summarize(post)["net"])
        result, _ = self.go(net)
        self.assertEqual([x[0] for x in self.lines], ["started"] * 3 + ["scored"] * 3)
        self.assertTrue(all(x[1].startswith("rescore10x:F2_taker_1s_gated@") for x in self.lines))
        self.assertEqual((result["registered"], result["outside_holm"]), (False, True))
        self.assertEqual(sorted(result["cells"]), ["1", "2", "5"])
        self.assertTrue((self.dir / "o.json").exists())


if __name__ == "__main__":
    unittest.main()
