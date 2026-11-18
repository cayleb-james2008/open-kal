"""Track 5 on hand-built tapes: the regime filter, trigger spacing, and pricing through sim_core."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from collections import Counter
from decimal import Decimal as D
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from openkal import sim_core as S
from scripts import track2c_true_l2_settle as C
from scripts import track5_two_sided_mm as M
from tests.test_sim_core import tape as sim_tape

CLOSE, STRIKE, CUT = 2000.0, 100.0, 2900.0
OPEN_CAPS = {"0.40": np.inf, "0.60": np.inf}


def lane_of(prints=(), bid=0.48, ask=0.53, y=1, idx_end=CLOSE):
    name = "KXBTC15M-TEST"
    book_ts = np.arange(CLOSE - 900.0, CLOSE, 1.0)
    idx_ts = np.arange(CLOSE - 1200.0, idx_end + 1.0, 1.0)
    rng = np.random.default_rng(3)
    idx = 10000.0 + np.cumsum(rng.normal(0, 0.5, len(idx_ts)))
    lane = SimpleNamespace(
        lane="KXBTC15M", tickers=[name], meta={name: {"strike": STRIKE, "close": CLOSE, "y": y}},
        ts=book_ts, bid=np.full(len(book_ts), bid), ask=np.full(len(book_ts), ask),
        book_bounds=np.array([0, len(book_ts)]), trade_ts=np.array([p[0] for p in prints], float),
        trade_px=np.array([p[1] for p in prints], float), trade_sz=np.array([p[2] for p in prints], float),
        trade_ty=np.array([p[3] for p in prints], np.int8), trade_bounds=np.array([0, len(prints)]))
    return lane, idx_ts, idx


def pack_of(**kw):
    lane, its, v = lane_of(**kw)
    return M.collect(lane, its, v, CUT, "is"), (its, v)


class Sampling(unittest.TestCase):
    def test_the_grid_is_the_twelve_requested_configs(self):
        self.assertEqual(len(M.grid()), 12)
        self.assertEqual(len({M.config_name(*g) for g in M.grid()}), 12)

    def test_decision_rows_span_240_to_720_seconds_left_and_the_book_is_read_at_t(self):
        pack, _ = pack_of()
        self.assertEqual((float(pack["secs"].min()), float(pack["secs"].max()), len(pack["t"])), (240.0, 720.0, 241))
        self.assertEqual((set(pack["bid"]), set(pack["ask"])), ({0.48}, {0.53}))

    def test_rows_without_a_fresh_index_print_are_dropped_not_called_quiet(self):
        pack, _ = pack_of(idx_end=CLOSE - 500.0)
        self.assertGreater(pack["skips"]["no_spot"], 0)
        self.assertLessEqual(float(pack["t"].max()), CLOSE - 500.0 + 2.0)

    def test_the_other_split_is_not_sampled(self):
        lane, its, v = lane_of()
        self.assertEqual(len(M.collect(lane, its, v, CUT, "oos")["t"]), 0)

    def test_the_spread_funnel_counts_rows_at_each_minimum(self):
        pack = {"bid": np.array([0.50, 0.48, 0.47, 0.45]), "ask": np.array([0.51, 0.51, 0.51, 0.50])}
        got = M.spread_funnel(pack)
        self.assertEqual((got["spread_ge_3c"], got["spread_ge_4c"], got["spread_ge_5c"]), (3, 2, 1))
        self.assertAlmostEqual(got["median_spread"], 0.035)
        text = M.funnel_text({"sampling": {"KXBTC15M": {**got, "decision_rows": 4}},
                              "configs": {"c": {"is": {"skips": {"spread": 7}}}}})
        self.assertIn("BTC 3/2/1 of 4", text)
        self.assertIn("7 attempts", text)

    def test_vol_caps_are_the_is_quantiles(self):
        sigma = np.arange(1.0, 101.0)
        self.assertEqual(M.vol_caps(sigma), {"0.40": float(np.quantile(sigma, 0.4)), "0.60": float(np.quantile(sigma, 0.6))})


class Champion(unittest.TestCase):
    @staticmethod
    def cfg(trades, net, pf=0.5, lanes=1):
        return {"is": {"trades": trades, "net": net, "pf": pf, "profitable_lanes": lanes, "crossed_fills": 0}}

    def test_a_config_that_never_traded_is_not_the_best_of_the_losers(self):
        got = M.pick_champion({"idle": self.cfg(0, 0.0), "loser": self.cfg(10, -3.3), "worse": self.cfg(9, -4.8)})
        self.assertEqual((got["config"], got["is_eligible"]), ("loser", False))

    def test_an_eligible_config_still_wins_and_all_idle_still_returns_something(self):
        got = M.pick_champion({"ok": self.cfg(40, 5.0, pf=1.4, lanes=3), "loser": self.cfg(10, -3.3)})
        self.assertEqual((got["config"], got["is_eligible"]), ("ok", True))
        self.assertEqual(M.pick_champion({"a": self.cfg(0, 0.0)})["config"], "a")


class Triggers(unittest.TestCase):
    def pack(self, **cols):
        n = len(next(iter(cols.values())))
        base = {"tid": np.zeros(n, np.int64), "t": np.arange(n) * 100.0, "bid": np.full(n, 0.48), "ask": np.full(n, 0.53),
                "sigma": np.full(n, 1.0)}
        base.update({k: np.asarray(v, float) for k, v in cols.items()})
        return base

    def test_spread_regime_and_cost_cap_each_exclude(self):
        p = self.pack(ask=[0.51, 0.52, 0.53, 0.53, 0.53], bid=[0.48, 0.48, 0.48, 0.48, 0.65], sigma=[1, 1, 5.0, 1, 1])
        got = M.triggers(p, 0.04, 2.0)
        self.assertEqual(got, [1, 3])                       # row0 spread .03, row2 too volatile, row4 YES bid .65 > cap
        self.assertEqual(M.triggers(self.pack(bid=[0.2], ask=[0.25]), 0.03, 2.0), [])     # NO leg 0.75 > cap

    def test_attempts_are_spaced_and_capped_per_window(self):
        p = self.pack(sigma=[1] * 6)
        p["t"] = np.array([0.0, 10.0, 70.0, 80.0, 140.0, 150.0])
        self.assertEqual(M.triggers(p, 0.03, 2.0), [0, 2])


class Trades(unittest.TestCase):
    T0 = CLOSE - 720.0                                       # the first decision of the window

    def go(self, prints, spot=None, timeout=15.0, **kw):
        pack, its_v = pack_of(prints=prints, **kw)
        out = M.new_out()
        M.run_config(pack, spot if spot is not None else its_v, OPEN_CAPS, 0.03, 0.40, timeout, out)
        return out[M.config_name(0.03, 0.40, timeout)]

    def test_a_filled_pair_is_priced_by_sim_core_and_pays_the_spread(self):
        res = self.go([(self.T0 + 2.0, 0.48, 50, 0), (self.T0 + 4.0, 0.53, 50, 1)])
        row = res["rows"][0]
        self.assertEqual((row["reason"], row["entry"], D(row["pnl"]), D(row["fees"])), ("pair", "0.48", D("2.50"), D(0)))
        self.assertEqual(row["order_ts"], self.T0 + S.LAG_S)
        self.assertEqual((row["qty"], len(row["feat"])), (S.MAX_QTY, 4))
        self.assertEqual(res["skips"]["unfilled"], 1)         # the second attempt, 60s later, saw no prints

    def test_a_lone_leg_is_crossed_out_and_pays_the_taker_fee(self):
        res = self.go([(self.T0 + 2.0, 0.48, 50, 0)])
        row = res["rows"][0]
        fee = S.taker_fee(D("0.48"), S.MAX_QTY)
        self.assertEqual((row["reason"], row["side"], D(row["fees"])), ("hedge_timeout", "YES", fee))
        self.assertEqual(D(row["pnl"]), -fee)                 # flat book: out at the entry price, minus the fee

    def test_the_spot_tape_reaches_the_simulator(self):
        spot = (np.array([self.T0, self.T0 + 3.0]), np.array([10000.0, 9990.0]))
        row = self.go([(self.T0 + 2.0, 0.48, 50, 0)], spot=spot)["rows"][0]       # YES filled, spot -10 bps
        self.assertEqual(row["reason"], "spot_breach")


class Overlap(unittest.TestCase):
    def test_open_until_for_each_outcome(self):
        book, prints = [(990.0, 0.48, 0.53)], [(1002.0, 0.48, 50, 0), (1004.0, 0.53, 50, 1)]
        pair = S.run_two_sided_mm(sim_tape(book, prints), trigger_ts=1000.0, qty=10, hedge_timeout_s=15.0)
        self.assertEqual((pair.reason, M.open_until(1000.0, pair)), ("pair", 1004.0))
        self.assertEqual(M.open_until(1000.0, S.Skip("unfilled")), 1000.0 + S.LAG_S + M.PATIENCE_S)
        self.assertEqual(M.open_until(1000.0, S.Skip("spread")), 1000.0 + S.LAG_S)

    def test_overlapping_attempts_in_one_window_are_refused_loudly(self):
        pack, its_v = pack_of(prints=[(CLOSE - 718.0, 0.48, 50, 0)])
        out = M.new_out()
        with patch.object(M, "triggers", return_value=[0, 5]):     # two attempts 10s apart: the first is still open
            with self.assertRaises(RuntimeError):
                M.run_config(pack, its_v, OPEN_CAPS, 0.03, 0.40, 15.0, out)


class Sealed(unittest.TestCase):
    def test_an_eligible_champion_still_needs_the_lead_seal(self):
        with tempfile.TemporaryDirectory() as dd:
            d = Path(dd)
            (d / "is.jsonl").write_text("")
            (d / "r.json").write_text(json.dumps({"frozen_input_sha256": {"x": "1"}, "oos_scored": None, "cut": 1.0,
                                                  "vol_caps_by_lane": {}, "configs": {"c": {"min_spread": .03, "vol_quantile": .4, "hedge_timeout_s": 15.0}},
                                                  "champion": {"config": "c", "is_eligible": True}}))
            with patch.object(M, "OUT", d / "r.json"), patch.object(M, "IS_TRADES", d / "is.jsonl"), \
                    patch.object(C, "LEDGER", d / "ledger.jsonl"), patch.object(M.RC, "LEDGER", d / "ledger.jsonl"), \
                    patch.object(M.T, "pin_tape", return_value={"sha256": {"x": "1"}}), \
                    patch.dict("os.environ", {}, clear=False):
                os.environ.pop("RND_OOS_UNSEAL", None)
                with self.assertRaisesRegex(RuntimeError, "sealed"):
                    M.score_oos()
            self.assertFalse((d / "ledger.jsonl").exists())

    def test_an_is_rerun_is_refused_once_the_family_has_a_ledger_line_or_a_scored_receipt(self):
        with tempfile.TemporaryDirectory() as dd:
            d = Path(dd)
            with patch.object(C, "LEDGER", d / "ledger.jsonl"):
                M.T.refuse_if_spent(M.FAMILY, d / "r.json")                         # nothing yet: allowed
                (d / "r.json").write_text(json.dumps({"oos_scored": {"x": 1}}))
                with self.assertRaises(RuntimeError):
                    M.T.refuse_if_spent(M.FAMILY, d / "r.json")
                (d / "r.json").write_text(json.dumps({"oos_scored": None}))
                (d / "ledger.jsonl").write_text(json.dumps({"family": "other"}) + "\n" + json.dumps({"family": M.FAMILY}) + "\n")
                with self.assertRaises(RuntimeError):
                    M.T.refuse_if_spent(M.FAMILY, d / "r.json")

    def test_an_ineligible_champion_never_touches_the_ledger(self):
        with tempfile.TemporaryDirectory() as dd:
            out, ledger = Path(dd) / "r.json", Path(dd) / "ledger.jsonl"
            out.write_text(json.dumps({"frozen_input_sha256": {"x": "1"}, "oos_scored": None,
                                       "champion": {"config": "s0.03_q0.40_t15", "is_eligible": False}}))
            with patch.object(M, "OUT", out), patch.object(C, "LEDGER", ledger), \
                    patch.object(M.T, "pin_tape", return_value={"sha256": {"x": "1"}}):
                with self.assertRaises(RuntimeError):
                    M.score_oos()
            self.assertFalse(ledger.exists())


if __name__ == "__main__":
    unittest.main()
