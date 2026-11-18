"""Track 7 on hand-built tapes: the vol inversion, the cross-lane alignment, the trigger, and sim_core pricing."""
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
from scripts import kurtosis_fv_snipe as KF
from scripts import track2c_true_l2_settle as C
from scripts import track7_cross_lane_iv as M

CLOSE, STRIKE, CUT = 2000.0, 100.0, 2900.0
FIT = {"nu": 10.0, "s": 1.0}
TABLE = KF.t_cdf_table(FIT["nu"])


def arrays(spot0=100.3, bid=0.55, ask=0.58, seed=7):
    rng = np.random.default_rng(seed)
    ts = np.arange(CLOSE - 900.0, CLOSE, 1.0)
    idx_ts = np.arange(CLOSE - 1200.0, CLOSE + 1.0, 1.0)
    return {"ts": ts, "bid": np.full(len(ts), bid), "ask": np.full(len(ts), ask), "idx_ts": idx_ts,
            "idx": spot0 + np.cumsum(rng.normal(0, 0.003, len(idx_ts)))}


def lane_of(d, y=1):
    name = "KXETH15M-TEST"
    return SimpleNamespace(
        lane="KXETH15M", tickers=[name], meta={name: {"strike": STRIKE, "close": CLOSE, "y": y}},
        ts=d["ts"], bid=d["bid"], ask=d["ask"], book_bounds=np.array([0, len(d["ts"])]),
        trade_ts=np.zeros(0), trade_px=np.zeros(0), trade_sz=np.zeros(0), trade_ty=np.zeros(0, np.int8),
        trade_bounds=np.array([0, 0]), index={"ts": d["idx_ts"], "value": d["idx"]})


def pack_of(d=None, y=1):
    lane = lane_of(d or arrays(), y)
    return lane, M.collect(lane, FIT, CUT, "is")


class Inversion(unittest.TestCase):
    def mid_for(self, sigma, ln_sk, tau):
        z = ln_sk / (sigma * FIT["s"] * np.sqrt(tau))
        return float(np.interp(z, KF.CDF_X, TABLE))

    def test_the_implied_vol_recovers_the_vol_that_priced_the_binary(self):
        for ln_sk in (0.003, -0.003):                                   # spot above and below the strike
            mid = self.mid_for(0.0003, ln_sk, 260.0)
            iv, z = M.implied_vol(np.array([mid]), np.array([ln_sk]), np.array([260.0]), TABLE, FIT["s"])
            self.assertAlmostEqual(float(iv[0]) / 0.0003, 1.0, places=3)

    def test_it_is_undefined_when_the_price_disagrees_with_the_spot_side_or_sits_at_the_money(self):
        tau, ln_sk = np.array([260.0] * 3), np.array([0.003] * 3)
        iv, _ = M.implied_vol(np.array([0.40, 0.51, 0.60]), ln_sk, tau, TABLE, FIT["s"])
        self.assertTrue(np.isnan(iv[0]))            # spot above the strike but the market says YES is the underdog
        self.assertTrue(np.isnan(iv[1]))            # |z| ~ 0.03: a binary has no vega there
        self.assertTrue(np.isfinite(iv[2]))

    def test_a_richer_price_for_the_underdog_is_a_higher_implied_vol(self):
        lo = M.implied_vol(np.array([0.60]), np.array([0.003]), np.array([260.0]), TABLE, FIT["s"])[0][0]
        hi = M.implied_vol(np.array([0.55]), np.array([0.003]), np.array([260.0]), TABLE, FIT["s"])[0][0]
        self.assertGreater(hi, lo)                  # mid closer to 0.5 = more vol


class Sampling(unittest.TestCase):
    def test_rows_sit_on_the_five_second_grid_and_the_itm_side_is_the_one_the_spot_favours(self):
        _, pack = pack_of()
        self.assertEqual((float(pack["secs"].min()), float(pack["secs"].max())), (120.0, 720.0))
        self.assertTrue((np.diff(pack["secs"]) == -5.0).all())
        self.assertTrue(pack["itm_yes"].all())
        self.assertEqual(set(pack["ask_itm_m"]), {580.0})                                    # the YES ask
        _, below = pack_of(arrays(spot0=99.7, bid=0.42, ask=0.45))                            # spot under the strike, YES cheap
        self.assertFalse(below["itm_yes"].any())
        self.assertEqual(set(below["ask_itm_m"]), {1000.0 - 420.0})                          # NO ask = 1 - yes_bid
        self.assertTrue(np.array_equal(below["side_won"], 1.0 - below["y"]))

    def test_rows_where_the_price_and_the_spot_disagree_are_dropped_and_counted(self):
        _, pack = pack_of(arrays(spot0=99.7, bid=0.55, ask=0.58))                            # spot below strike, market says YES
        self.assertEqual(len(pack["t"]), 0)
        self.assertGreater(pack["skips"]["iv_undefined"], 100)

    def test_a_row_reads_nothing_after_its_own_second(self):
        d = arrays()
        _, pack = pack_of(d)
        i = int(np.flatnonzero(pack["secs"] == 300.0)[0])
        t = float(pack["t"][i])
        e = {k: v.copy() for k, v in d.items()}
        e["bid"][e["ts"] > t], e["ask"][e["ts"] > t] = 0.60, 0.63
        e["idx"][e["idx_ts"] > t] += 0.3
        _, other = pack_of(e)
        k = int(np.flatnonzero(other["secs"] == 300.0)[0])
        for col in ("lnsk", "bid_m", "ask_m", "z", "iv", "sigma", "ratio"):
            self.assertEqual(pack[col][i], other[col][k], col)
        j, l = int(np.flatnonzero(pack["secs"] == 150.0)[0]), int(np.flatnonzero(other["secs"] == 150.0)[0])
        self.assertNotEqual(pack["lnsk"][j], other["lnsk"][l])                               # the perturbation is real


class CrossLane(unittest.TestCase):
    def test_btc_is_matched_at_the_same_window_and_second_only(self):
        _, alt = pack_of()
        btc = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in alt.items()}
        btc["ratio"] = np.full(len(alt["t"]), 1.2)
        keep = alt["secs"] != 300.0
        btc["key"], btc["ratio"] = btc["key"][keep], btc["ratio"][keep]
        M.attach_btc(alt, btc)
        miss = alt["secs"] == 300.0
        self.assertTrue(np.isnan(alt["btc_ratio"][miss]).all() and (alt["btc_ratio"][~miss] == 1.2).all())

    def test_with_btcs_premium_as_the_fair_iv_the_fair_price_is_the_market_price_when_the_ratios_match(self):
        _, alt = pack_of()
        btc = {"key": alt["key"].copy(), "ratio": alt["ratio"].copy()}
        M.attach_btc(alt, btc)
        mid = (alt["bid_m"] + alt["ask_m"]) / 2000.0
        np.testing.assert_allclose(alt["p_fair"], mid, atol=2e-3)                            # YES is in the money here

    def test_no_btc_rows_at_all_leaves_everything_nan(self):
        _, alt = pack_of()
        M.attach_btc(alt, {"key": np.zeros(0), "ratio": np.zeros(0)})
        self.assertTrue(np.isnan(alt["btc_ratio"]).all() and not M.in_band(alt).any())


class Trigger(unittest.TestCase):
    def pack(self, ratio, btc, ask_m=520.0, n=None):
        n = n or len(ratio)
        return {"tid": np.zeros(n, np.int64), "t": np.arange(n) * 100.0, "ratio": np.asarray(ratio, float),
                "btc_ratio": np.asarray(btc, float), "ask_itm_m": np.full(n, ask_m) if np.isscalar(ask_m) else np.asarray(ask_m, float)}

    def test_the_dispersion_threshold_is_inclusive_at_one_point_three_five(self):
        p = self.pack([1.35, 1.349, 1.40], [1.0, 1.0, 1.0])
        p["tid"] = np.array([0, 1, 2])
        self.assertEqual(M.triggers(p, dispersion=True), [0, 2])
        self.assertEqual(M.triggers(p, dispersion=False), [0, 1, 2])                          # the control ignores the ratio

    def test_the_ask_band_is_35_to_60_cents_and_needs_a_btc_ratio(self):
        p = self.pack([2.0] * 5, [1.0, 1.0, 1.0, 1.0, np.nan], ask_m=[349.0, 350.0, 600.0, 601.0, 500.0])
        p["tid"] = np.arange(5)
        self.assertEqual(M.triggers(p, dispersion=True), [1, 2])

    def test_attempts_are_spaced_and_capped_per_window(self):
        p = self.pack([2.0] * 6, [1.0] * 6)
        p["t"] = np.array([0.0, 10.0, 70.0, 80.0, 140.0, 150.0])
        self.assertEqual(M.triggers(p, dispersion=True), [0, 2])


class Trades(unittest.TestCase):
    def forced(self, d, y, itm_yes):
        lane, pack = pack_of(d, y=y)
        M.attach_btc(pack, {"key": pack["key"].copy(), "ratio": pack["ratio"] / 1.5})         # alt is 1.5x BTC everywhere
        out = {M.CONFIG: {"rows": [], "skips": Counter()}}
        M.run_trades(lane, pack, M.triggers(pack, dispersion=True), M.CONFIG, out)
        return out[M.CONFIG]

    def test_the_in_the_money_yes_is_bought_at_its_ask_and_held_to_settlement(self):
        res = self.forced(arrays(), y=1, itm_yes=True)
        row = res["rows"][0]
        self.assertEqual((row["side"], row["entry"], row["reason"], row["entry_style"]), ("YES", "0.58", "settle", "taker"))
        self.assertEqual(row["order_ts"], row["trigger_ts"] + S.LAG_S)
        self.assertEqual((row["qty"], len(row["feat"])), (S.MAX_QTY, 5))
        self.assertGreater(D(row["fees"]), 0)                                                 # a taker fee, no exit fee at settlement
        self.assertEqual(D(row["pnl"]), (D(1) - D("0.58")) * S.MAX_QTY - D(row["fees"]))

    def test_the_in_the_money_no_pays_one_minus_the_yes_bid(self):
        res = self.forced(arrays(spot0=99.7, bid=0.42, ask=0.45), y=0, itm_yes=False)
        row = res["rows"][0]
        self.assertEqual((row["side"], row["entry"]), ("NO", "0.58"))                         # 1 - 0.42
        self.assertLessEqual(D(row["entry"]), D("0.60"))


class Evaluation(unittest.TestCase):
    def test_brier_rows_scores_the_fair_price_and_the_ask_on_the_same_rows(self):
        pack = {"p_fair": np.array([0.7, 0.2, np.nan]), "ask_itm_m": np.array([600.0, 400.0, 500.0]),
                "side_won": np.array([1.0, 0.0, 1.0])}
        got = M.brier_rows({"KXETH15M": pack}, lambda p: np.ones(3, bool))
        self.assertEqual(got["rows"], 2)
        self.assertAlmostEqual(got["model"], ((0.3 ** 2) + (0.2 ** 2)) / 2)
        self.assertAlmostEqual(got["ask"], ((0.4 ** 2) + (0.4 ** 2)) / 2)

    def test_the_is_bar_is_twenty_trades_pf_and_two_lanes(self):
        ok = {"trades": 20, "pf": 1.25, "profitable_lanes": 2, "crossed_fills": 0}
        self.assertTrue(M.eligible(ok))
        for key, bad in (("trades", 19), ("pf", 1.24), ("profitable_lanes", 1), ("crossed_fills", 1)):
            self.assertFalse(M.eligible({**ok, key: bad}), key)

    def test_an_ineligible_receipt_never_touches_the_ledger_and_a_rescore_override_is_refused(self):
        with tempfile.TemporaryDirectory() as dd:
            d = Path(dd)
            (d / "r.json").write_text(json.dumps({"frozen_input_sha256": {"x": "1"}, "oos_scored": None, "is_eligible": False}))
            with patch.object(M, "OUT", d / "r.json"), patch.object(C, "LEDGER", d / "l.jsonl"), \
                    patch.object(M.T, "pin_tape", return_value={"sha256": {"x": "1"}}):
                with self.assertRaisesRegex(RuntimeError, "stays sealed"):
                    M.score_oos()
                with patch.dict(os.environ, {"RND_OOS_RESCORE": "why"}):
                    with self.assertRaisesRegex(RuntimeError, "never rescore"):
                        M.score_oos()
            self.assertFalse((d / "l.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
