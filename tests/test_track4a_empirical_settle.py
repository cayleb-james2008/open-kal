"""Track 4A: the fit reaches an optimum, features are causal, folds are purged, trades go through sim_core."""
from __future__ import annotations

import json
import tempfile
import unittest
from decimal import Decimal as D
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from openkal import sim_core as S
from scripts import track2c_true_l2_settle as C
from scripts import track4_common as T
from scripts import track4a_empirical_settle as A

CLOSE, STRIKE, CUT = 2000.0, 100.0, 2900.0
FIT = {"s": 1.0, "nu": 10.0}


def base_arrays(seed=7):
    rng = np.random.default_rng(seed)
    book_ts = np.arange(CLOSE - 900.0, CLOSE, 1.0)
    idx_ts = np.arange(CLOSE - 1200.0, CLOSE + 1.0, 1.0)
    return {
        "book_ts": book_ts, "bid": np.full(len(book_ts), 0.55), "ask": np.full(len(book_ts), 0.58),
        "micro": np.where(book_ts > CLOSE - 500.0, 0.575, np.nan),
        "idx_ts": idx_ts, "idx": 100.0 + np.cumsum(rng.normal(0, 0.02, len(idx_ts))),
        "tr_ts": np.sort(rng.uniform(CLOSE - 900.0, CLOSE, 400)), "tr_sz": rng.uniform(1, 30, 400),
        "tr_ty": rng.integers(0, 2, 400).astype(np.int8)}


def lane_of(d, y=1):
    name = "KXBTC15M-TEST"
    lane = SimpleNamespace(
        lane="KXBTC15M", tickers=[name], meta={name: {"strike": STRIKE, "close": CLOSE, "y": y}},
        ts=d["book_ts"], bid=d["bid"], ask=d["ask"], book_bounds=np.array([0, len(d["book_ts"])]),
        trade_ts=d["tr_ts"], trade_px=np.full(len(d["tr_ts"]), 0.57), trade_sz=d["tr_sz"], trade_ty=d["tr_ty"],
        trade_bounds=np.array([0, len(d["tr_ts"])]), index={"ts": d["idx_ts"], "value": d["idx"]})
    return lane, d["micro"]


def pack_of(d, y=1):
    lane, micro = lane_of(d, y)
    return A.collect(lane, micro, FIT, CUT, "is")


class Fit(unittest.TestCase):
    def test_newton_reaches_the_optimum_and_recovers_the_signal(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(40000, 3))
        x[:, 2] = x[:, 0] * 0.8 + rng.normal(size=40000) * 0.6      # correlated, like z and entry_ask
        true = 1.0 / (1.0 + np.exp(-(1.0 * x[:, 0] - 0.5 * x[:, 1] + 0.3)))
        y = (rng.random(40000) < true).astype(float)
        w, norm, info = A.fit_ridge_logistic(x, y, 1.0)
        self.assertLess(info["max_grad_per_row"], 1e-6)
        from scripts.f2_structural import apply_logistic
        self.assertLess(float(np.abs(apply_logistic(x, w, norm) - true).mean()), 0.01)
        heavy = A.fit_ridge_logistic(x, y, 1e6)[0]
        self.assertLess(np.abs(heavy[:3]).max(), 0.2 * np.abs(w[:3]).max())     # lambda really shrinks

    def test_no_side_flips_direction_and_uses_its_own_ask(self):
        pack = {"z": np.array([1.0]), "tau": np.array([100.0]), "spread": np.array([0.03]), "mpm": np.array([np.nan]),
                "f5": np.array([0.4]), "f15": np.array([-0.2]), "ask_y": np.array([0.58]), "ask_n": np.array([0.45])}
        yes, no = A.side_x(pack, S.YES)[0], A.side_x(pack, S.NO)[0]
        np.testing.assert_allclose(yes, [1.0, 100.0, 0.03, 0.0, 0.4, -0.2, 0.58])
        np.testing.assert_allclose(no, [-1.0, 100.0, 0.03, 0.0, -0.4, 0.2, 0.45])


class Folds(unittest.TestCase):
    def test_every_window_is_held_once_and_neighbours_are_purged(self):
        rank = np.arange(130)
        held_count = np.zeros(130)
        for k in range(A.FOLDS):
            train, held = A.fold_masks(rank, 130, k)
            held_count += held
            lo, hi = np.flatnonzero(held)[[0, -1]]
            self.assertFalse(train[max(lo - 1, 0):hi + 2].any())
            self.assertTrue(train[:max(lo - 1, 0)].all() and train[hi + 2:].all())
        self.assertTrue((held_count == 1).all())

    def test_a_training_row_inside_the_last_900s_before_the_cut_is_refused(self):
        pack = {"t": np.array([CUT - 899.0]), "close": np.array([1.0])}
        with self.assertRaises(RuntimeError):
            A.fit_model({"KXBTC15M": pack}, [1.0], CUT)


class Causality(unittest.TestCase):
    def test_a_decision_row_never_reads_anything_after_its_own_time(self):
        d = base_arrays()
        pack = pack_of(d)
        i = int(np.flatnonzero(pack["secs"] == 300.0)[0])
        t = float(pack["t"][i])
        e = {k: v.copy() for k, v in d.items()}
        late_i, late_b, late_t = e["idx_ts"] > t, e["book_ts"] > t, e["tr_ts"] > t
        e["idx"][late_i] += 5.0
        e["bid"][late_b], e["ask"][late_b], e["micro"][late_b] = 0.10, 0.20, 0.15
        e["tr_sz"][late_t] *= 50.0
        e["tr_ty"][late_t] = 1 - e["tr_ty"][late_t]
        other = pack_of(e)
        for col in ("z", "tau", "spread", "mpm", "f5", "f15", "ask_y", "ask_n", "bid_y", "bid_n"):
            self.assertEqual(pack[col][i], other[col][i], col)
        later = int(np.flatnonzero(pack["secs"] == 120.0)[0])
        self.assertNotEqual(pack["z"][later], other["z"][later])               # the perturbation is real

    def test_cost_columns_follow_the_book_symmetry_for_each_side_and_style(self):
        pack = pack_of(base_arrays())                      # YES book 0.55 / 0.58
        got = {c: set(np.round(pack[c], 3)) for c in ("ask_y", "ask_n", "bid_y", "bid_n")}
        self.assertEqual(got, {"ask_y": {0.58}, "ask_n": {0.45}, "bid_y": {0.55}, "bid_n": {0.42}})

    def test_microprice_is_missing_where_the_tape_has_no_sizes(self):
        pack = pack_of(base_arrays())
        old = pack["secs"] >= 500.0
        self.assertTrue(np.isnan(pack["mpm"][old]).all() and np.isfinite(pack["mpm"][~old]).all())


class Trades(unittest.TestCase):
    def run_forced(self, p_yes, p_no, style, y):
        pack = pack_of(base_arrays(), y=y)
        n = len(pack["t"])
        py, pn = np.full(n, p_yes), np.full(n, p_no)
        out = A.new_out(["c"])
        with patch.object(A, "config_name", return_value="c"):
            A.run_config(pack, py, pn, 0.06, style, out)
        return out["c"]

    def test_yes_taker_pays_the_ask_and_no_taker_pays_one_minus_the_bid(self):
        yes = self.run_forced(0.95, np.nan, "taker", 1)["rows"][0]
        no = self.run_forced(np.nan, 0.95, "taker", 0)["rows"][0]
        self.assertEqual((yes["side"], yes["entry"]), ("YES", "0.58"))
        self.assertEqual((no["side"], no["entry"]), ("NO", "0.45"))            # 1 - 0.55, never the YES bid
        self.assertEqual(D(no["pnl"]) + D(no["fees"]), (D(1) - D("0.45")) * D(no["qty"]))
        self.assertEqual(no["order_ts"], no["trigger_ts"] + S.LAG_S)
        self.assertLessEqual(D(yes["entry"]), D("0.62"))

    def test_probability_below_cost_plus_hurdle_never_trades(self):
        res = self.run_forced(0.62, 0.40, "taker", 1)
        self.assertEqual(res["rows"], [])

    def test_a_side_whose_ask_is_outside_the_band_is_not_in_the_domain(self):
        d = base_arrays()
        d["ask"][:] = 0.70
        d["bid"][:] = 0.66
        pack = pack_of(d)
        self.assertFalse(A.in_domain(pack, S.YES).any())


class Sealed(unittest.TestCase):
    def test_an_ineligible_champion_never_touches_the_ledger(self):
        with tempfile.TemporaryDirectory() as dd:
            out, ledger = Path(dd) / "r.json", Path(dd) / "ledger.jsonl"
            out.write_text(json.dumps({"frozen_input_sha256": {"x": "1"}, "oos_scored": None,
                                       "champion": {"config": "h0.06_taker", "is_eligible": False}}))
            with patch.object(A, "OUT", out), patch.object(C, "LEDGER", ledger), \
                    patch.object(A.T, "pin_tape", return_value={"sha256": {"x": "1"}}):
                with self.assertRaises(RuntimeError):
                    A.score_oos()
            self.assertFalse(ledger.exists())

    def test_strict_json_can_hold_an_infinite_profit_factor(self):
        text = json.dumps(T.clean({"pf": float("inf"), "a": [float("nan"), 1.5]}), allow_nan=False)
        self.assertEqual(json.loads(text), {"pf": "inf", "a": ["nan", 1.5]})


if __name__ == "__main__":
    unittest.main()
