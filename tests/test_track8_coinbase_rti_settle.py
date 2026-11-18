"""Track 8 on hand-built tapes: the repriced settlement average, the divergence trigger and sim_core pricing."""
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
from scripts import track8_coinbase_rti_settle as M
from scripts import rnd_common as RC
from scripts.asian_rti import AsianRTI
from scripts.track6_gbdt_l2_meta import last_vals

CLOSE, STRIKE, CUT = 2000.0, 100.0, 2900.0


def index_arrays(seed=3, level=100.0, close=CLOSE):
    """A 1 Hz RTI tape with the venue's carried accumulator: k pinned at 60 before the window, then k = elapsed seconds."""
    rng = np.random.default_rng(seed)
    ts = np.arange(close - 1200.0, close + 1.0, 1.0)
    value = level + np.cumsum(rng.normal(0, 0.01, len(ts)))
    accum, k = value.copy(), np.full(len(ts), 60.0)
    running = []
    for i, t in enumerate(ts):
        if t > close - 60.0:
            running.append(value[i])
            k[i], accum[i] = len(running), float(np.mean(running))
    return ts, value, accum, k


def arrays(cb_offset_bps=0.0, bid=0.55, ask=0.58, **kw):
    ts, value, accum, k = index_arrays(**kw)
    return {"ts": np.arange(CLOSE - 900.0, CLOSE, 1.0), "bid": bid, "ask": ask, "idx": (ts, value, accum, k),
            "tick_ts": ts.copy(), "tick_spot": value * (1.0 + cb_offset_bps * 1e-4)}


def lane_of(d, y=1):
    name = "KXBTC15M-TEST"
    ts = d["ts"]
    return SimpleNamespace(
        lane="KXBTC15M", tickers=[name], meta={name: {"strike": STRIKE, "close": CLOSE, "y": y}},
        ts=ts, bid=np.full(len(ts), d["bid"]) if np.isscalar(d["bid"]) else d["bid"],
        ask=np.full(len(ts), d["ask"]) if np.isscalar(d["ask"]) else d["ask"], book_bounds=np.array([0, len(ts)]),
        trade_ts=np.zeros(0), trade_px=np.zeros(0), trade_sz=np.zeros(0), trade_ty=np.zeros(0, np.int8), trade_bounds=np.array([0, 0]),
        index={"ts": d["idx"][0], "value": d["idx"][1], "accum": d["idx"][2], "k": d["idx"][3]})


def pack_of(d=None, y=1):
    d = d or arrays()
    lane = lane_of(d, y)
    model = AsianRTI(*d["idx"])
    return lane, model, M.collect(lane, model, (d["tick_ts"], d["tick_spot"]), CUT, "is")


class Formula(unittest.TestCase):
    def test_it_matches_the_repos_asian_model_when_the_spot_is_the_rtis_own(self):
        d = arrays()
        model = AsianRTI(*d["idx"])
        for secs in (85.0, 70.0, 59.0, 45.0, 30.0, 25.0):                                    # both sides of the 60s window opening
            probe = model.fair(CLOSE - secs, CLOSE, STRIKE)
            strike = probe["mean"] + 0.7 * probe["sd"]                                         # a strike inside the distribution
            f = model.fair(CLOSE - secs, CLOSE, strike)
            self.assertTrue(0.1 < f["p"] < 0.5, msg=(secs, f["p"]))                            # not saturated: the variance matters
            mine = M.asian_p(f["spot"], f["locked_sum"], f["elapsed"], secs, f["sigma"], strike)
            self.assertAlmostEqual(mine, f["p"], places=9, msg=secs)

    def test_a_higher_live_spot_raises_the_probability_and_the_locked_part_anchors_it(self):
        lo, hi = (M.asian_p(s, 60 * 99.9 * 0.5, 30.0, 30.0, 0.05, 100.0) for s in (99.8, 100.2))
        self.assertGreater(hi, lo)
        # all 60 fixings locked: the live spot cannot matter, only the locked mean against the strike
        self.assertEqual({M.asian_p(s, 60 * 101.0, 60.0, 0.0, 0.05, 100.0) for s in (50.0, 99.0, 150.0)}, {1.0})
        self.assertEqual({M.asian_p(s, 60 * 99.0, 60.0, 0.0, 0.05, 100.0) for s in (50.0, 150.0)}, {0.0})
        # one fixing left carries a 1/60 weight in the mean
        half = M.asian_p(100.0, 59 * 100.0, 59.0, 1.0, 0.05, 100.0)
        self.assertAlmostEqual(half, 0.5)                                                      # mean exactly at the strike


class Sampling(unittest.TestCase):
    def test_rows_span_the_final_ninety_seconds_and_carry_the_divergence_and_both_probabilities(self):
        _, _, pack = pack_of(arrays(cb_offset_bps=10.0))
        self.assertEqual((float(pack["secs"].min()), float(pack["secs"].max()), len(pack["t"])), (25.0, 90.0, 66))
        np.testing.assert_allclose(pack["div_bps"], 10.0, atol=1e-6)
        self.assertTrue(M.diverged(pack).all())
        self.assertTrue((pack["p_cb"] != pack["p_rti"]).any())
        self.assertEqual((set(pack["ask_y_m"]), set(pack["ask_n_m"])), ({580.0}, {450.0}))   # NO ask = 1 - yes_bid

    def test_a_live_spot_below_the_rti_makes_the_divergence_negative_and_lowers_the_probability(self):
        _, _, below = pack_of(arrays(cb_offset_bps=-10.0))
        _, _, flat = pack_of(arrays(cb_offset_bps=0.0))
        self.assertTrue((below["div_bps"] < 0).all())
        self.assertTrue((below["p_cb"] <= below["p_rti"] + 1e-12).all())
        np.testing.assert_allclose(flat["p_cb"], flat["p_rti"])
        self.assertFalse(M.diverged(flat).any())

    def test_a_row_reads_nothing_after_its_own_second(self):
        d = arrays(cb_offset_bps=10.0)
        _, _, pack = pack_of(d)
        i = int(np.flatnonzero(pack["secs"] == 60.0)[0])
        t = float(pack["t"][i])
        ts, value, accum, k = (a.copy() for a in d["idx"])
        late = ts > t
        value[late] += 3.0
        accum[late] += 3.0
        e = {**d, "idx": (ts, value, accum, k), "tick_spot": np.where(d["tick_ts"] > t, 500.0, d["tick_spot"]),
             "bid": np.where(d["ts"] > t, 0.30, 0.55), "ask": np.where(d["ts"] > t, 0.33, 0.58)}
        _, _, other = pack_of(e)
        for col in ("cb", "rti", "div_bps", "p_cb", "p_rti", "bid_m", "ask_m"):
            self.assertEqual(pack[col][i], other[col][int(np.flatnonzero(other["secs"] == 60.0)[0])], col)

    def test_stale_live_spot_stale_quote_and_missing_rti_state_are_dropped_and_counted(self):
        d = arrays(cb_offset_bps=10.0)
        keep = ~((d["tick_ts"] > CLOSE - 70.0) & (d["tick_ts"] < CLOSE - 40.0))               # the feed goes quiet
        _, _, pack = pack_of({**d, "tick_ts": d["tick_ts"][keep], "tick_spot": d["tick_spot"][keep]})
        self.assertGreater(pack["skips"]["no_live_spot"], 20)
        self.assertFalse(((pack["secs"] < 68.0) & (pack["secs"] > 42.0)).any())
        stale = arrays()
        drop = ~((stale["ts"] > CLOSE - 80.0) & (stale["ts"] < CLOSE - 30.0))
        _, _, pack = pack_of({**stale, "ts": stale["ts"][drop], "bid": np.full(int(drop.sum()), 0.55), "ask": np.full(int(drop.sum()), 0.58)})
        self.assertGreater(pack["skips"]["stale_quote"], 30)

    def test_lane_spot_keeps_one_lane_sorted_without_missing_prices(self):
        ticks = {"series": np.array(["A", "B", "A", "A", "A"]), "ts": np.array([3.0, 1.0, 1.0, 2.0, 4.0]),
                 "spot": np.array([30.0, 99.0, 10.0, 0.0, 40.0])}
        ts, spot = M.lane_spot(ticks, "A")
        np.testing.assert_array_equal(ts, [1.0, 3.0, 4.0])
        np.testing.assert_array_equal(spot, [10.0, 30.0, 40.0])


class Trigger(unittest.TestCase):
    def pack(self, div, p_cb, ask_y=520.0, ask_n=470.0):
        n = len(div)
        return {"tid": np.arange(n), "t": np.arange(n) * 100.0, "div_bps": np.asarray(div, float), "p_cb": np.asarray(p_cb, float),
                "ask_y_m": np.full(n, ask_y) if np.isscalar(ask_y) else np.asarray(ask_y, float),
                "ask_n_m": np.full(n, ask_n) if np.isscalar(ask_n) else np.asarray(ask_n, float)}

    def test_the_divergence_threshold_is_inclusive_at_eight_basis_points_either_way(self):
        p = self.pack([8.0, 7.99, -8.0, -7.99], [0.9, 0.9, 0.1, 0.1])
        self.assertEqual([i for i, _, _ in M.triggers(p, 0.06)], [0, 2])

    def test_the_hurdle_is_p_minus_the_ask_and_the_larger_edge_side_wins(self):
        p = self.pack([10.0, 10.0, 10.0], [0.58, 0.5799, 0.30], ask_y=[520.0, 520.0, 520.0], ask_n=[470.0, 470.0, 470.0])
        got = M.triggers(p, 0.06)
        self.assertEqual([(i, s) for i, s, _ in got], [(0, S.YES), (2, S.NO)])               # 0.58-0.52 = 0.06 yes; 0.70-0.47 = 0.23 no
        self.assertAlmostEqual(got[1][2], 0.70 - 0.47)

    def test_the_ask_band_is_35_to_60_cents_per_side(self):
        p = self.pack([10.0] * 4, [0.99] * 4, ask_y=[349.0, 350.0, 600.0, 601.0])
        p["ask_n_m"] = np.full(4, 950.0)
        self.assertEqual([i for i, _, _ in M.triggers(p, 0.06)], [1, 2])

    def test_attempts_are_spaced_and_capped_per_window(self):
        p = self.pack([10.0] * 6, [0.9] * 6)
        p["tid"] = np.zeros(6, np.int64)
        p["t"] = np.array([0.0, 10.0, 70.0, 80.0, 140.0, 150.0])
        self.assertEqual([i for i, _, _ in M.triggers(p, 0.06)], [0, 2])


class Trades(unittest.TestCase):
    def forced(self, d, y, hurdle=0.06):
        lane, model, pack = pack_of(d, y=y)
        out = {M.config_name(hurdle): {"rows": [], "skips": Counter()}}
        M.run_hurdle(lane, pack, hurdle, out)
        return out[M.config_name(hurdle)], pack

    def test_a_divergent_up_move_buys_yes_at_its_ask_and_holds_to_settlement(self):
        d = arrays(cb_offset_bps=40.0, bid=0.50, ask=0.52)                                    # the live spot is far above the RTI
        res, pack = self.forced(d, y=1)
        row = res["rows"][0]
        self.assertEqual((row["side"], row["entry"], row["reason"], row["entry_style"]), ("YES", "0.52", "settle", "taker"))
        self.assertEqual(row["order_ts"], row["trigger_ts"] + S.LAG_S)
        self.assertEqual((row["qty"], len(row["feat"])), (S.MAX_QTY, 4))
        self.assertEqual(D(row["pnl"]), (D(1) - D("0.52")) * S.MAX_QTY - D(row["fees"]))

    def test_a_divergent_down_move_buys_no_at_one_minus_the_yes_bid(self):
        d = arrays(cb_offset_bps=-40.0, bid=0.50, ask=0.52)
        res, _ = self.forced(d, y=0)
        row = res["rows"][0]
        self.assertEqual((row["side"], D(row["entry"])), ("NO", D("0.50")))                   # 1 - 0.50
        self.assertLessEqual(D(row["entry"]), D("0.60"))

    def test_nothing_trades_without_a_divergence(self):
        res, _ = self.forced(arrays(cb_offset_bps=2.0, bid=0.50, ask=0.52), y=1)
        self.assertEqual(res["rows"], [])


class Controls(unittest.TestCase):
    def test_the_control_drops_only_the_divergence_trigger_and_is_named_apart(self):
        self.assertEqual((M.config_name(0.06), M.config_name(0.06, False)), ("h0.06", "control_h0.06_no_divergence_trigger"))
        p = {"tid": np.arange(3), "t": np.arange(3) * 100.0, "div_bps": np.array([10.0, 1.0, 0.0]), "p_cb": np.full(3, 0.9),
             "ask_y_m": np.full(3, 520.0), "ask_n_m": np.full(3, 950.0)}
        self.assertEqual([i for i, _, _ in M.triggers(p, 0.06)], [0])
        self.assertEqual([i for i, _, _ in M.triggers(p, 0.06, divergence=False)], [0, 1, 2])

    def test_run_all_adds_the_controls_only_when_asked(self):
        lane, _, pack = pack_of(arrays(cb_offset_bps=40.0, bid=0.50, ask=0.52))
        with patch.object(M.AD, "Lane", return_value=lane):
            plain = M.run_all({n: pack for n in M.RC.LANES}, (0.06,))
            both = M.run_all({n: pack for n in M.RC.LANES}, (0.06,), controls=True)
        self.assertEqual(set(plain), {"h0.06"})
        self.assertEqual(set(both), {"h0.06", "control_h0.06_no_divergence_trigger"})

    def test_an_empty_live_spot_series_gives_no_prices_instead_of_a_crash(self):
        self.assertTrue(np.isnan(last_vals(np.zeros(0), np.zeros(0), np.array([1.0, 2.0]))).all())


class Funnel(unittest.TestCase):
    def test_the_funnel_counts_rows_divergent_rows_windows_and_rows_clearing_each_hurdle(self):
        pack = {"windows": 5, "t": np.arange(4.0), "close": np.array([900.0, 900.0, 1800.0, 2700.0]),
                "div_bps": np.array([9.0, -9.0, 1.0, 12.0]), "p_cb": np.array([0.70, 0.30, 0.9, 0.5]),
                "ask_y_m": np.full(4, 520.0), "ask_n_m": np.full(4, 520.0), "skips": Counter({"no_quote": 3})}
        got = M.funnel({"KXBTC15M": pack})["KXBTC15M"]
        self.assertEqual((got["rows"], got["diverged_rows"], got["diverged_windows"]), (4, 3, 2))
        self.assertEqual(got["rows_clearing_hurdle"], {"h0.06": 2, "h0.09": 2, "h0.12": 2})   # 0.70-0.52 = 0.18 and 0.70-0.52 on NO
        self.assertEqual(got["skips"], {"no_quote": 3})


class Trades2(unittest.TestCase):
    def test_the_executed_band_is_passed_to_sim_core(self):
        lane, _, pack = pack_of(arrays(cb_offset_bps=40.0, bid=0.27, ask=0.30))               # YES ask 0.30: below the 0.35 floor
        out = {"h0.06": {"rows": [], "skips": Counter()}}
        with patch.object(M, "triggers", return_value=[(0, S.YES, 0.2)]):
            M.run_hurdle(lane, pack, 0.06, out)
        self.assertEqual((out["h0.06"]["rows"], dict(out["h0.06"]["skips"])), ([], {"cost_band": 1}))


class Evaluation(unittest.TestCase):
    def test_brier_scores_both_probabilities_and_the_ask_on_the_in_band_side_rows(self):
        pack = {"p_cb": np.array([0.8]), "p_rti": np.array([0.6]), "ask_y_m": np.array([500.0]), "ask_n_m": np.array([500.0]),
                "y": np.array([1.0])}
        got = M.brier_rows({"A": pack}, lambda p: np.ones(1, bool))
        self.assertEqual(got["rows"], 2)                                                       # YES side and NO side
        self.assertAlmostEqual(got["cb"], ((0.2 ** 2) + (0.2 ** 2)) / 2)                       # YES 0.8 vs 1; NO 0.2 vs 0
        self.assertAlmostEqual(got["rti"], ((0.4 ** 2) + (0.4 ** 2)) / 2)
        self.assertAlmostEqual(got["ask"], ((0.5 ** 2) + (0.5 ** 2)) / 2)

    def test_brier_leaves_out_side_rows_whose_ask_is_outside_the_band(self):
        pack = {"p_cb": np.array([0.8, 0.8]), "p_rti": np.array([0.6, 0.6]), "ask_y_m": np.array([500.0, 900.0]),
                "ask_n_m": np.array([500.0, 100.0]), "y": np.array([1.0, 1.0])}
        got = M.brier_rows({"A": pack}, lambda p: np.ones(2, bool))
        self.assertEqual(got["rows"], 2)                                                       # only row 0 (YES and NO sides)

    def test_the_is_bar_is_twenty_trades_pf_and_two_lanes(self):
        ok = {"trades": 20, "pf": 1.25, "profitable_lanes": 2, "crossed_fills": 0}
        self.assertTrue(M.eligible(ok))
        for key, bad in (("trades", 19), ("pf", 1.24), ("profitable_lanes", 1), ("crossed_fills", 1)):
            self.assertFalse(M.eligible({**ok, key: bad}), key)

    def test_the_champion_is_the_best_of_those_that_traded_when_none_qualify(self):
        cfg = lambda t, net: {"is": {"trades": t, "pf": 0.5, "profitable_lanes": 0, "crossed_fills": 0, "net": net}}
        got = M.pick_champion({"h0.06": cfg(0, 0.0), "h0.09": cfg(6, -12.0), "h0.12": cfg(3, -5.0)})
        self.assertEqual((got["config"], got["is_eligible"]), ("h0.12", False))

    def receipt(self, d, **over):
        rec = {"frozen_input_sha256": {"x": "1"}, "oos_scored": None, "cut": 1.0, "rule": {"qty": 50},
               "champion": {"config": "h0.06", "is_eligible": True}, "configs": {"h0.06": {"hurdle": 0.06}}}
        rec.update(over)
        (d / "r.json").write_text(json.dumps(rec))
        (d / "is.jsonl").write_text("")

    def test_an_already_scored_receipt_is_refused_even_with_no_ledger_line(self):
        with tempfile.TemporaryDirectory() as dd:
            d = Path(dd)
            self.receipt(d, oos_scored={"oos": {"trades": 1}})
            with patch.object(M, "OUT", d / "r.json"), patch.object(M, "IS_TRADES", d / "is.jsonl"), patch.object(C, "LEDGER", d / "l.jsonl"), \
                    patch.object(M.T, "pin_tape", return_value={"sha256": {"x": "1"}}):
                with self.assertRaisesRegex(RuntimeError, "stays sealed"):
                    M.score_oos()
            self.assertFalse((d / "l.jsonl").exists())

    def test_an_eligible_champion_is_scored_exactly_once_and_the_receipt_and_ledger_record_it(self):
        def row(pnl):
            return {"lane": "KXBTC15M", "close": 5.0, "pnl": str(pnl), "reason": "settle", "entry": "0.5", "fee_entry": "0",
                    "fee_exit": "0", "crossed": 0, "requotes": 0, "feat": [0.5, 0.1, 40.0, 9.0]}
        with tempfile.TemporaryDirectory() as dd:
            d = Path(dd)
            self.receipt(d)
            ledger = d / "l.jsonl"
            fake = {"h0.06": {"rows": [row(5), row(-2), row(3)], "skips": Counter()}}
            with patch.object(M, "OUT", d / "r.json"), patch.object(M, "IS_TRADES", d / "is.jsonl"), patch.object(M, "OOS_TRADES", d / "oos.jsonl"), \
                    patch.object(C, "LEDGER", ledger), patch.object(RC, "LEDGER", ledger), \
                    patch.object(M.T, "pin_tape", return_value={"sha256": {"x": "1"}}), patch.object(M, "gather", return_value=({}, {})), \
                    patch.object(M, "run_all", return_value=fake), patch.object(M, "brier_rows", return_value={"rows": 0}), \
                    patch.object(M, "funnel", return_value={}), patch.dict(os.environ, {"RND_OOS_UNSEAL": "lead"}):
                os.environ.pop("RND_OOS_RESCORE", None)
                obj = M.score_oos()
                self.assertEqual(obj["oos_scored"]["oos"]["trades"], 3)
                lines = [json.loads(line) for line in ledger.read_text().splitlines()]
                self.assertEqual([r["status"] for r in lines], ["started", "scored"])
                self.assertEqual({r["family"] for r in lines}, {M.FAMILY})                      # the once-only rule keys on this name
                self.assertEqual(json.loads((d / "r.json").read_text())["oos_scored"]["oos"]["trades"], 3)
                self.assertEqual(len((d / "oos.jsonl").read_text().splitlines()), 3)
                with self.assertRaises(RuntimeError):                                          # a second look is refused
                    M.score_oos()
                self.assertEqual(len(ledger.read_text().splitlines()), 2)

    def test_oos_refusals_and_the_is_ledger_guard(self):
        with tempfile.TemporaryDirectory() as dd:
            d = Path(dd)
            (d / "r.json").write_text(json.dumps({"frozen_input_sha256": {"x": "1"}, "oos_scored": None,
                                                  "champion": {"config": "h0.06", "is_eligible": False}}))
            with patch.object(M, "OUT", d / "r.json"), patch.object(C, "LEDGER", d / "l.jsonl"), \
                    patch.object(M.T, "pin_tape", return_value={"sha256": {"x": "1"}}):
                with self.assertRaisesRegex(RuntimeError, "stays sealed"):
                    M.score_oos()
                with patch.dict(os.environ, {"RND_OOS_RESCORE": "why"}):
                    with self.assertRaisesRegex(RuntimeError, "never rescore"):
                        M.score_oos()
                (d / "l.jsonl").write_text(json.dumps({"family": M.FAMILY, "status": "started"}) + "\n")
                with self.assertRaisesRegex(RuntimeError, "already has an OOS ledger entry"):
                    M.run_is()


if __name__ == "__main__":
    unittest.main()
