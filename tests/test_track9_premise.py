"""The Coinbase-vs-RTI premise test: registered constants, the lead-lag statistic, the criteria and the single-run gate."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import track8_coinbase_rti_settle as M8
from scripts import track9_coinbase_lead_lag as M

PREREG_SHA256 = "66bc153252603d9cb4d1aeb461c201ac8b6538e3c35fdc8044a6cb21d084354b"   # v2, amended before any code or data
T0 = 1_800_000_000.0


def feeds(n=21_600, lag=0, noise=0.0, seed=1):
    """A random-walk 'Coinbase' price printed every second and an 'RTI' that shows it `lag` seconds later (negative: earlier)."""
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.normal(0.0, 2e-4, n + 20))
    cb = 100.0 * np.exp(walk[10:n + 10])
    rti = 100.0 * np.exp(walk[10 - lag:n + 10 - lag] + rng.normal(0.0, noise, n))
    ts = T0 + np.arange(n)
    return (ts, cb), (ts, rti)


def stat(lag=0, noise=0.0, **kw):
    cb, rti = feeds(lag=lag, noise=noise, **kw)
    return M.lead_lag(cb, rti, T0, T0 + len(cb[0]))


class Registration(unittest.TestCase):
    def test_the_registered_file_is_the_v2_amended_before_any_code_and_has_not_drifted(self):
        self.assertEqual(hashlib.sha256(M.SPEC_PATH.read_bytes()).hexdigest(), PREREG_SHA256)
        self.assertEqual((M.SPEC["version"], M.SPEC_SHA256), (2, PREREG_SHA256))
        self.assertEqual(M.SPEC["amendments"][0]["supersedes_commit"], "81927da6aee5bdbaf0ecdc338f7f335102574285")

    def test_the_registered_numbers_are_the_ones_the_code_uses(self):
        s = M.SPEC
        self.assertEqual((M.LAGS, s["run_gate"]["min_eligible_windows"]), ((1, 2, 5), 288))
        self.assertIn("cut = eligible_closes[int(0.6 * n)]", s["split"]["rule"])
        self.assertEqual((M.FUN["min_abs_divergence_bps"], M.FUN["secs_left"], M.BR["secs_left"]), (8.0, [25, 90], [25, 60]))
        self.assertEqual((M.LL["bootstrap"]["draws"], M.LL["bootstrap"]["seed"], M.LL["bootstrap"]["block_s"]), (2000, 0, 900))
        self.assertEqual(M.LANE_SYMBOL, {"KXBTC15M": "BTC-USD", "KXETH15M": "ETH-USD", "KXSOL15M": "SOL-USD", "KXXRP15M": "XRP-USD"})
        M.check_constants()

    def test_a_changed_track8_constant_stops_the_run_instead_of_quietly_changing_the_test(self):
        with patch.object(M8, "DIVERGENCE_BPS", 4.0):
            with self.assertRaises(AssertionError):
                M.check_constants()


class LeadLag(unittest.TestCase):
    def test_a_one_second_rti_lag_is_found(self):
        r = stat(lag=1)
        self.assertGreater(r["A_1"]["point"], 0.95)
        self.assertLess(abs(r["A_2"]["point"]), 0.05)
        self.assertTrue(r["passes"] and r["S"]["lo"] > 0.9)

    def test_a_two_second_lag_is_found_which_the_first_registration_would_have_missed(self):
        r = stat(lag=2)
        self.assertLess(abs(r["A_1"]["point"]), 0.05)                       # no overlap at lag 1 ...
        self.assertGreater(r["A_2"]["point"], 0.95)                         # ... and the exact-lag statistic still sees it
        self.assertTrue(r["passes"])

    def test_no_lag_a_reversed_lead_and_a_five_second_lag_do_not_pass(self):
        flat, reverse, five = stat(lag=0), stat(lag=-2), stat(lag=5)
        self.assertFalse(flat["passes"])
        self.assertLess(abs(flat["S"]["point"]), 0.05)
        self.assertTrue(reverse["S"]["point"] < -0.9 and not reverse["passes"])      # the RTI leads: negative
        self.assertGreater(five["A_5"]["point"], 0.95)                               # reported
        self.assertFalse(five["passes"])                                             # but not the primary statistic

    def test_noise_shrinks_the_statistic_and_the_interval_brackets_it(self):
        clean, noisy = stat(lag=1), stat(lag=1, noise=2e-4)
        self.assertLess(noisy["S"]["point"], clean["S"]["point"] - 0.1)
        self.assertGreater(noisy["S"]["point"], 0.2)
        self.assertTrue(noisy["S"]["lo"] < noisy["S"]["point"] < noisy["S"]["hi"])

    def test_a_weak_lead_needs_the_effect_size_and_the_interval(self):
        weak = stat(lag=1, noise=2e-3)                                       # a small positive S well under 0.10
        self.assertLess(weak["S"]["point"], 0.10)
        self.assertFalse(weak["passes"])

    def test_nothing_at_or_after_the_cut_is_read(self):
        cb, rti = feeds(lag=1)
        n = len(cb[0])
        cut = T0 + n // 2
        base = M.lead_lag(cb, rti, T0, cut)
        late = (cb[0], np.where(cb[0] >= cut, cb[1] * 3.0, cb[1]))
        late_rti = (rti[0], np.where(rti[0] >= cut, rti[1] * 0.2, rti[1]))
        self.assertEqual(M.lead_lag(late, late_rti, T0, cut)["S"], base["S"])
        self.assertEqual(M.lead_lag(cb, rti, T0, cut)["rows"], base["rows"])
        self.assertLess(base["rows"], n * 0.55)

    def test_stale_prices_make_no_returns_instead_of_stretched_ones(self):
        (ts, cb), rti = feeds(lag=1)
        keep = ~((ts >= T0 + 5000) & (ts < T0 + 5100))                       # Coinbase goes quiet for 100 s
        r = M.lead_lag((ts[keep], cb[keep]), rti, T0, T0 + len(ts))
        full = M.lead_lag((ts, cb), rti, T0, T0 + len(ts))
        self.assertLess(r["rows"], full["rows"] - 90)
        self.assertGreater(r["S"]["point"], 0.9)

    def test_a_lane_passes_only_with_the_effect_size_and_an_interval_above_zero(self):
        self.assertTrue(M.lane_passes({"point": 0.10, "lo": 0.001}))
        self.assertFalse(M.lane_passes({"point": 0.0999, "lo": 0.5}))                     # the interval alone is not enough
        self.assertFalse(M.lane_passes({"point": 0.20, "lo": 0.0}))                        # nor the point estimate
        self.assertFalse(M.lane_passes({"point": 0.20, "lo": -0.01}))

    def test_three_of_four_lanes_are_needed(self):
        lane = lambda ok: {"passes": ok}
        self.assertTrue(M.lead_lag_criterion({"a": lane(True), "b": lane(True), "c": lane(True), "d": lane(False)})["passes"])
        self.assertFalse(M.lead_lag_criterion({"a": lane(True), "b": lane(True), "c": lane(False), "d": lane(False)})["passes"])


class Criteria(unittest.TestCase):
    def packs(self, events: int, closes: int):
        """`events` divergent (lane, window) pairs spread over `closes` windows, plus quiet rows."""
        out = {lane: {"close": [], "div_bps": []} for lane in ("KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M")}
        lanes = list(out)
        for i in range(events):
            lane, close = lanes[(i // closes) % 4], float(900 * (i % closes))      # distinct (lane, window) pairs, `closes` windows
            out[lane]["close"] += [close, close]
            out[lane]["div_bps"] += [9.0, 1.0]
        return {k: {"close": np.array(v["close"]), "div_bps": np.array(v["div_bps"])} for k, v in out.items()}

    def test_the_funnel_needs_100_events_in_40_windows(self):
        self.assertTrue(M.funnel_criterion(self.packs(100, 40))["passes"])
        self.assertFalse(M.funnel_criterion(self.packs(99, 40))["passes"])
        self.assertFalse(M.funnel_criterion(self.packs(100, 39))["passes"])

    def test_an_event_is_a_lane_window_not_a_row(self):
        p = {"KXBTC15M": {"close": np.array([900.0] * 50), "div_bps": np.full(50, 9.0)}}
        got = M.funnel_criterion(p)
        self.assertEqual((got["events"], got["distinct_closes"]), (1, 1))

    def rows(self, better: bool, n_windows=60, per=20, seed=3):
        rng = np.random.default_rng(seed)
        close = np.repeat(np.arange(n_windows) * 900.0, per)
        y = (rng.random(len(close)) < 0.5).astype(float)
        mid = np.where(y == 1, 0.8, 0.2) * 0.5 + 0.25                       # a weakly informative market
        p_cb = np.where(y == 1, 0.95, 0.05) if better else np.clip(mid + rng.normal(0, 0.3, len(close)), 0.01, 0.99)
        return {"close": close, "secs": np.full(len(close), 40.0), "y": y, "p_cb": p_cb, "p_rti": p_cb,
                "bid_m": np.rint((mid - 0.01) * 1000), "ask_m": np.rint((mid + 0.01) * 1000)}

    def test_the_brier_criterion_needs_the_coinbase_probability_to_beat_the_mid_with_the_interval_below_zero(self):
        win = M.brier_criterion({"A": self.rows(True)})
        lose = M.brier_criterion({"A": self.rows(False)})
        self.assertTrue(win["passes"] and win["d_hi"] < 0 and win["mean_d"] < 0)
        self.assertFalse(lose["passes"])
        self.assertAlmostEqual(win["brier_mid"], (win["brier_mid"]), places=12)

    def test_only_the_last_sixty_seconds_count_and_the_interval_matters_not_just_the_mean(self):
        rows = self.rows(True)
        rows["secs"] = np.full(len(rows["secs"]), 80.0)
        self.assertEqual(M.brier_criterion({"A": rows})["rows"], 0)                    # nothing inside 25-60 s
        tie = self.rows(False, n_windows=3, per=4, seed=9)                             # few windows: a wide interval
        got = M.brier_criterion({"A": tie})
        self.assertFalse(got["passes"] and got["d_hi"] >= 0)

    def test_all_three_criteria_are_needed(self):
        yes = {"passes": True}
        no = {"passes": False}
        self.assertTrue(M.decide(yes, yes, yes)["premise_passes"])
        for combo in ((no, yes, yes), (yes, no, yes), (yes, yes, no)):
            self.assertFalse(M.decide(*combo)["premise_passes"])


class Eligibility(unittest.TestCase):
    def series(self, close, secs=None, drop=None):
        ts = np.arange(close - 150.0, close + 0.5, 1.0) if secs is None else secs
        if drop is not None:
            ts = ts[~((ts > close - drop[0]) & (ts < close - drop[1]))]
        return ts, np.full(len(ts), 100.0)

    def test_every_lane_needs_30_prints_and_no_gap_over_three_seconds(self):
        close = 5000.0
        ok = {l: self.series(close) for l in M.LANE_SYMBOL}
        self.assertEqual(M.eligible_closes([close], ok), [close])
        gap = {**ok, "KXSOL15M": self.series(close, drop=(100.0, 95.0))}                # a 6 s hole
        self.assertEqual(M.eligible_closes([close], gap), [])
        three = {**ok, "KXSOL15M": self.series(close, drop=(100.0, 97.5))}              # two prints missing: a gap of exactly 3.0 s
        self.assertEqual(M.eligible_closes([close], three), [close])                    # "at most 3.0" is inclusive
        four = {**ok, "KXSOL15M": self.series(close, drop=(100.0, 96.5))}               # three prints missing: 4.0 s
        self.assertEqual(M.eligible_closes([close], four), [])
        few = {**ok, "KXXRP15M": (np.linspace(close - 150, close, 29), np.ones(29))}
        self.assertEqual(M.eligible_closes([close], few), [])
        no_edge = {**ok, "KXBTC15M": self.series(close, secs=np.arange(close - 140.0, close - 5.0, 1.0))}   # silent at both edges
        self.assertEqual(M.eligible_closes([close], no_edge), [])

    def test_the_cut_is_sixty_percent_of_the_eligible_closes(self):
        closes = [float(i) for i in range(300)]
        self.assertEqual(M.split_cut(closes), 180.0)
        self.assertEqual(M.split_cut([float(i) for i in range(288)]), 172.0)

    def test_only_venue_labelled_coinbase_prints_of_the_right_symbol_are_kept(self):
        spot = {"source": np.array(["coinbase", "kraken", "coinbase", "rest", "coinbase", "coinbase"]),
                "symbol": np.array(["BTC-USD", "BTC-USD", "ETH-USD", "BTC-USD", "BTC-USD", "BTC-USD"]),
                "ts": np.array([3.0, 1.0, 2.0, 4.0, 0.5, 6.0]), "price": np.array([30.0, 99.0, 20.0, 98.0, 10.0, np.nan])}
        ts, px = M.coinbase_series(spot, "BTC-USD")
        np.testing.assert_array_equal(ts, [0.5, 3.0])
        np.testing.assert_array_equal(px, [10.0, 30.0])


class RunGate(unittest.TestCase):
    def inputs(self, n_closes):
        series = {l: (np.arange(10.0), np.ones(10)) for l in M.LANE_SYMBOL}
        return {"pin": {"sha256": {"x": "1"}}, "closes": [float(i * 900) for i in range(n_closes)], "cb": series,
                "rti": {l: (np.arange(10.0), np.ones(10)) for l in M.RC.LANES}}

    def boom(self, *a, **k):
        raise AssertionError("a statistic was computed")

    def test_below_the_gate_nothing_is_computed_and_the_receipt_says_so(self):
        with tempfile.TemporaryDirectory() as d:
            rec = Path(d) / "r.json"
            with patch.object(M, "load_inputs", return_value=self.inputs(50)), patch.object(M, "eligible_closes", return_value=list(range(287))), \
                    patch.object(M, "lead_lag", self.boom), patch.object(M, "gather_packs", self.boom):
                out = M.run(Path(d), rec)
                self.assertEqual((out["status"], out["eligible_windows"], out["run_gate"]), ("insufficient_data", 287, 288))
                self.assertEqual(json.loads(rec.read_text())["spec_sha256"], PREREG_SHA256)
                self.assertFalse(M.run(Path(d), rec, check_only=True)["ready"])
                self.assertEqual(M.run(Path(d), rec)["status"], "insufficient_data")      # an unfinished receipt does not lock the test

    def test_exactly_288_eligible_windows_is_enough_to_run(self):
        eligible = [float(i * 900) for i in range(288)]
        with tempfile.TemporaryDirectory() as d:
            rec = Path(d) / "r.json"
            ll = lambda *a: {"passes": False, "S": {"point": 0.0, "lo": -0.1, "hi": 0.1}}
            with patch.object(M, "load_inputs", return_value=self.inputs(288)), patch.object(M, "eligible_closes", return_value=eligible), \
                    patch.object(M, "lead_lag", ll), patch.object(M, "gather_packs", return_value={}), \
                    patch.object(M, "funnel_criterion", return_value={"passes": False}), patch.object(M, "brier_criterion", return_value={"passes": False}):
                out = M.run(Path(d), rec)
            self.assertEqual((out["status"], out["cut"], out["is_windows"]), ("decided", eligible[172], 172))

    def test_check_only_reports_readiness_without_writing_or_computing(self):
        with tempfile.TemporaryDirectory() as d:
            rec = Path(d) / "r.json"
            with patch.object(M, "load_inputs", return_value=self.inputs(50)), patch.object(M, "eligible_closes", return_value=list(range(288))), \
                    patch.object(M, "lead_lag", self.boom):
                self.assertTrue(M.run(Path(d), rec, check_only=True)["ready"])
            self.assertFalse(rec.exists())

    def test_a_decided_run_reads_only_the_in_sample_and_is_then_never_repeated(self):
        eligible = [float(i * 900) for i in range(300)]
        cut = eligible[180]
        calls = {}

        def ll(cb, rti, t0, t1):
            calls.setdefault("t_end", []).append(t1)
            return {"passes": True, "S": {"point": 0.2, "lo": 0.1, "hi": 0.3}}

        with tempfile.TemporaryDirectory() as d:
            rec = Path(d) / "r.json"
            with patch.object(M, "load_inputs", return_value=self.inputs(300)), patch.object(M, "eligible_closes", return_value=eligible), \
                    patch.object(M, "lead_lag", ll), patch.object(M, "gather_packs", return_value={}), \
                    patch.object(M, "funnel_criterion", return_value={"passes": False}), patch.object(M, "brier_criterion", return_value={"passes": True}):
                out = M.run(Path(d), rec)
                self.assertEqual((out["status"], out["cut"], out["is_windows"], out["oos_windows_unread"]), ("decided", cut, 180, 120))
                self.assertEqual(set(calls["t_end"]), {cut})                                 # the statistic is bounded by the cut
                self.assertFalse(out["decision"]["premise_passes"])                           # the funnel failed
                with self.assertRaisesRegex(RuntimeError, "run once"):
                    M.run(Path(d), rec)


class Tape(unittest.TestCase):
    def test_a_new_tape_is_copied_pinned_and_checked_on_use(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src"
            src.mkdir()
            (src / "spot_ticks.npz").write_bytes(b"ticks")
            (src / "markets.json").write_bytes(b"{}")
            dest = Path(d) / "tape"
            digests = M.pin_new_tape(dest, [src / "spot_ticks.npz", src / "markets.json"])
            self.assertEqual(set(digests), {"spot_ticks.npz", "markets.json"})
            self.assertEqual(M.T.pin_tape(dest)["sha256"], digests)
            with self.assertRaisesRegex(RuntimeError, "already pinned"):
                M.pin_new_tape(dest, [src / "markets.json"])
            (dest / "spot_ticks.npz").write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
                M.T.pin_tape(dest)


if __name__ == "__main__":
    unittest.main()
