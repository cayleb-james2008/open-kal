"""Track 6 on hand-built tapes: causal features, sim_core labels, the grid, and the sealed OOS gate.

Run with the system interpreter, which has scikit-learn:
    /usr/bin/python3 -B -m unittest tests.test_track6_gbdt_l2_meta
"""
from __future__ import annotations

import json
import os
import tempfile
import time
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
from scripts import track4a_empirical_settle as A
from scripts import track6_gbdt_l2_meta as M

CLOSE, STRIKE, CUT = 2000.0, 100.0, 2900.0
FIT = {"nu": 10.0, "s": 1.0}


def arrays(seed=7):
    rng = np.random.default_rng(seed)
    ts = np.arange(CLOSE - 900.0, CLOSE, 1.0)
    idx_ts = np.arange(CLOSE - 1200.0, CLOSE + 1.0, 1.0)
    return {"ts": ts, "bid": np.full(len(ts), 0.55), "ask": np.full(len(ts), 0.58),
            "bid_sz": np.full(len(ts), 300.0), "ask_sz": np.full(len(ts), 100.0), "micro": np.full(len(ts), 0.5675),
            "g_ts": ts.copy(), "g_avail": np.ones(len(ts), bool), "g_conf": np.full(len(ts), 0.3),
            "idx_ts": idx_ts, "idx": 100.0 + np.cumsum(rng.normal(0, 0.02, len(idx_ts))),
            "btc": 50000.0 + np.cumsum(rng.normal(0, 1.0, len(idx_ts))), "eth": 3000.0 + np.cumsum(rng.normal(0, 0.1, len(idx_ts))),
            "tr_ts": np.sort(rng.uniform(CLOSE - 900.0, CLOSE, 50)), "tr_sz": rng.uniform(1, 30, 50)}


def build(d, y=1):
    name = "KXBTC15M-TEST"
    lane = SimpleNamespace(
        lane="KXBTC15M", tickers=[name], meta={name: {"strike": STRIKE, "close": CLOSE, "y": y}},
        ts=d["ts"], bid=d["bid"], ask=d["ask"], book_bounds=np.array([0, len(d["ts"])]),
        trade_ts=d["tr_ts"], trade_px=np.full(len(d["tr_ts"]), 0.57), trade_sz=d["tr_sz"],
        trade_ty=np.zeros(len(d["tr_ts"]), np.int8), trade_bounds=np.array([0, len(d["tr_ts"])]),
        index={"ts": d["idx_ts"], "value": d["idx"]})
    ex = {"bid_sz": d["bid_sz"], "ask_sz": d["ask_sz"], "micro": d["micro"]}
    gates = {0: (d["g_ts"], d["g_avail"], np.zeros(len(d["g_ts"]), bool), d["g_conf"])}
    return lane, ex, gates, (d["idx_ts"], d["btc"]), (d["idx_ts"], d["eth"])


def pack_of(d, y=1):
    lane, ex, gates, btc, eth = build(d, y)
    return lane, M.collect(lane, ex, gates, btc, eth, FIT, {CLOSE}, CUT, "is")


class Sampling(unittest.TestCase):
    def test_every_grid_second_is_a_row_when_book_l2_gate_and_spot_exist(self):
        _, pack = pack_of(arrays())
        self.assertEqual((len(pack["t"]), float(pack["secs"].min()), float(pack["secs"].max())), (316, 90.0, 720.0))
        self.assertEqual((set(pack["ask_y"]), set(pack["ask_n"])), ({0.58}, {0.45}))      # NO ask = 1 - yes_bid
        np.testing.assert_allclose(pack["imb"], 0.5)                                       # (300-100)/(300+100)
        np.testing.assert_allclose(pack["mpm"], 0.5675 - 0.565)

    def test_rows_without_l2_or_gate_are_dropped_and_counted(self):
        d = arrays()
        d["micro"][d["ts"] < CLOSE - 500.0] = np.nan                                      # no sizes before then
        d["g_avail"][d["g_ts"] > CLOSE - 200.0] = False                                   # the gate goes away near the end
        _, pack = pack_of(d)
        self.assertGreater(pack["skips"]["no_l2"], 0)
        self.assertGreater(pack["skips"]["no_gate"], 0)
        self.assertTrue(((pack["secs"] <= 500.0) & (pack["secs"] >= 200.0)).all())

    def test_a_ticker_with_no_book_rows_is_skipped_not_a_crash(self):
        lane, ex, gates, btc, eth = build(arrays())
        lane.ts, lane.bid, lane.ask, lane.book_bounds = np.zeros(0), np.zeros(0), np.zeros(0), np.array([0, 0])
        ex = {k: np.zeros(0) for k in ex}
        pack = M.collect(lane, ex, gates, btc, eth, FIT, {CLOSE}, CUT, "is")
        self.assertEqual((len(pack["t"]), pack["skips"]["no_quote"]), (0, 316))

    def test_a_stale_l2_row_cannot_feed_a_later_decision(self):
        d = arrays()
        keep = ~((d["ts"] > CLOSE - 500.0) & (d["ts"] < CLOSE - 300.0))                     # the book goes quiet for 200s
        for k in ("ts", "bid", "ask", "bid_sz", "ask_sz", "micro"):
            d[k] = d[k][keep]
        _, pack = pack_of(d)
        self.assertGreater(pack["skips"]["no_l2"], 90)
        self.assertFalse(((pack["secs"] < 497.0) & (pack["secs"] > 303.0)).any())            # 2s after the last row before the gap

    def test_a_decision_row_reads_nothing_after_its_own_second(self):
        d = arrays()
        _, pack = pack_of(d)
        i = int(np.flatnonzero(pack["secs"] == 300.0)[0])
        t = float(pack["t"][i])
        e = {k: v.copy() for k, v in d.items()}
        late = e["ts"] > t
        e["bid"][late], e["ask"][late], e["bid_sz"][late], e["micro"][late], e["g_conf"][late] = 0.10, 0.20, 5.0, 0.15, 0.40
        e["idx"][e["idx_ts"] > t] += 5.0
        e["btc"][e["idx_ts"] > t] *= 1.1
        e["eth"][e["idx_ts"] > t] *= 0.9
        _, other = pack_of(e)
        for col in ("ask_y", "ask_n", "imb", "mpm", "gconf", "pty", "z", "b1", "b3", "b10", "e1", "e3", "e10"):
            self.assertEqual(pack[col][i], other[col][i], col)
        j = int(np.flatnonzero(pack["secs"] == 120.0)[0])
        self.assertNotEqual(pack["z"][j], other["z"][j])                                    # the perturbation is real

    def test_impulses_are_trailing_log_returns_and_flip_for_no(self):
        d = arrays()
        d["btc"][:] = 50000.0
        d["btc"][d["idx_ts"] >= CLOSE - 300.0] = 50050.0                                    # a +0.1% step at secs_left 300
        _, pack = pack_of(d)
        i = int(np.flatnonzero(pack["secs"] == 300.0)[0])
        self.assertAlmostEqual(float(pack["b1"][i]), np.log(50050.0 / 50000.0))
        self.assertEqual(float(pack["b3"][i]), float(pack["b1"][i]))
        self.assertEqual(float(pack["b1"][int(np.flatnonzero(pack["secs"] == 330.0)[0])]), 0.0)
        names = M.FEATURES
        yes, no = M.side_x(pack, S.YES)[i], M.side_x(pack, S.NO)[i]
        k = names.index("btc_1s")
        self.assertAlmostEqual(float(yes[k]), -float(no[k]))
        self.assertEqual((float(yes[names.index("entry_ask")]), float(no[names.index("entry_ask")])), (0.58, 0.45))
        self.assertAlmostEqual(float(yes[names.index("student_t_p")]) + float(no[names.index("student_t_p")]), 1.0)

    def test_the_ask_band_is_25_to_62_cents_on_each_side(self):
        d = arrays()
        d["bid"][:], d["ask"][:] = 0.20, 0.23                                               # YES ask 0.23 (out), NO ask 0.80 (out)
        _, pack = pack_of(d)
        self.assertFalse(M.in_domain(pack, S.YES).any() or M.in_domain(pack, S.NO).any())
        d["bid"][:], d["ask"][:] = 0.37, 0.38                                               # YES 0.38 and NO 0.63: only YES
        _, pack = pack_of(d)
        self.assertTrue(M.in_domain(pack, S.YES).all() and not M.in_domain(pack, S.NO).any())


class SignsAndFit(unittest.TestCase):
    def test_no_flips_every_direction_bearing_column_and_nothing_else(self):
        _, pack = pack_of(arrays())
        pack["b1"] = pack["b3"] = pack["b10"] = pack["e1"] = pack["e3"] = pack["e10"] = np.full(len(pack["t"]), 0.001)
        yes, no = M.side_x(pack, S.YES)[0], M.side_x(pack, S.NO)[0]
        names = M.FEATURES
        flipped = ("touch_imbalance", "microprice_minus_mid", "student_t_z", "btc_1s", "btc_3s", "btc_10s", "eth_1s", "eth_3s", "eth_10s")
        for k in flipped:
            self.assertAlmostEqual(float(yes[names.index(k)]), -float(no[names.index(k)]), msg=k)
            self.assertNotEqual(float(yes[names.index(k)]), 0.0, msg=k)
        for k in ("gate_confidence", "spread", "secs_left"):
            self.assertEqual(float(yes[names.index(k)]), float(no[names.index(k)]), msg=k)

    def test_the_no_label_is_the_complement_of_the_outcome(self):
        _, pack = pack_of(arrays(), y=1)
        self.assertTrue((M.side_label(pack, S.YES) == 1).all() and (M.side_label(pack, S.NO) == 0).all())

    def test_the_student_t_fit_never_reads_a_print_from_the_900s_before_the_cut(self):
        rng = np.random.default_rng(1)
        cut = 100000.0
        end = cut - KF.FIT_LEAD
        its = np.arange(cut - 30000.0 + 14.5, cut, 1.0)                                    # grid points land at end - 0.5
        v = 100 * np.exp(np.cumsum(rng.normal(0, 3e-5, len(its))))
        keep = ~((its >= end - 5) & (its < end))
        its, v = its[keep], v[keep]
        j = int(np.searchsorted(its, end))
        its, v = np.insert(its, j, end + 0.2), np.insert(v, j, v[j - 1] * 1.01)             # one print just inside the lead
        reads_it = KF.fit_lane(its[its < cut], v[its < cut], cut)["n"]
        self.assertEqual(reads_it - M.fit_from(its, v, cut)["n"], 5)                        # the unclipped fit used it five times


class Labels(unittest.TestCase):
    def test_net_labels_come_from_sim_core_and_follow_the_outcome_and_the_exit_mode(self):
        lane, pack = pack_of(arrays(), y=1)                                                 # YES wins
        M.label_pass(lane, pack)
        self.assertTrue(np.nanmin(pack["net"][("hold", S.YES)]) == 1.0 and np.nanmax(pack["net"][("hold", S.NO)]) == 0.0)
        i = int(np.flatnonzero(pack["secs"] == 300.0)[0])
        res = M.price(lane, pack, i, S.NO, "hold")
        self.assertEqual((res.entry, res.entry_style, res.qty), (D("0.45"), S.TAKER, S.MAX_QTY))
        self.assertLess(res.pnl, 0)
        self.assertEqual(pack["net"][("hold", S.NO)][i], 0.0)

    def test_a_trade_the_simulator_refuses_has_no_label(self):
        d = arrays()
        d["bid"][:], d["ask"][:] = 0.55, 0.58
        d["ts"] = d["ts"].copy()
        lane, pack = pack_of(d)
        lane.ts = lane.ts + 10_000.0                                                        # the book now lies after every order
        M.label_pass(lane, pack)
        self.assertTrue(np.isnan(pack["net"][("hold", S.YES)]).all())


class Models(unittest.TestCase):
    def test_boosting_is_deterministic_and_inside_the_cpu_budget(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(50000, 14))
        y = (rng.random(50000) < 1 / (1 + np.exp(-x[:, 0]))).astype(float)
        t0 = time.process_time()
        a = M.fit_one(x, y)
        cpu = time.process_time() - t0
        b = M.fit_one(x, y)
        self.assertLess(cpu, 15.0)
        np.testing.assert_array_equal(a.predict_proba(x[:500]), b.predict_proba(x[:500]))
        self.assertGreater(float(np.corrcoef(a.predict_proba(x[:5000])[:, 1], x[:5000, 0])[0, 1]), 0.8)

    def test_one_class_or_tiny_training_sets_fall_back_to_a_constant(self):
        self.assertIsInstance(M.fit_one(np.zeros((500, 3)), np.ones(500)), M.Const)
        self.assertIsInstance(M.fit_one(np.zeros((50, 3)), np.r_[np.ones(25), np.zeros(25)]), M.Const)
        self.assertAlmostEqual(float(M.Const(0.3).predict_proba(np.zeros((2, 3)))[0, 1]), 0.3)

    def test_net_models_train_only_on_rows_the_simulator_priced(self):
        lane, pack = pack_of(arrays(), y=1)
        M.label_pass(lane, pack)
        pack["net"][("hold", S.YES)][:] = np.nan                                            # pretend sim refused every YES
        models = M.fit_models({"KXBTC15M": pack}, {"KXBTC15M": np.ones(len(pack["t"]), bool)})
        self.assertIsInstance(models["net_hold"], M.Const)                                  # only losing NO rows remain: one class
        self.assertEqual(float(models["net_hold"].p), 0.0)

    def test_brier_is_exact(self):
        self.assertAlmostEqual(M.brier(np.array([0.7, 0.2]), np.array([1.0, 0.0])), 0.065)


class Grid(unittest.TestCase):
    def base(self, **cols):
        n = len(next(iter(cols.values())))
        pack = {"tid": np.zeros(n, np.int64), "t": np.arange(n) * 100.0, "ask_y": np.full(n, 0.50), "ask_n": np.full(n, 0.48)}
        pack.update({k: np.asarray(v, float) for k, v in cols.items()})
        return pack

    def preds(self, **kw):
        nan = np.full(len(next(iter(kw.values()))), np.nan)
        out = {(k, s): nan for k in ("win", "net_hold", "net_tp5_stop7") for s in (S.YES, S.NO)}
        out.update({(k, S.YES if k.endswith("y") is False else S.NO): v for k, v in {}.items()})
        return out

    def test_win_model_needs_p_minus_ask_at_least_the_hurdle(self):
        pack = self.base(ask_y=[0.50, 0.50, 0.50])
        pack["tid"] = np.array([0, 1, 2])
        p = self.preds(x=[0, 0, 0])
        p[("win", S.YES)] = np.array([0.53, 0.5299, np.nan])
        got = M.triggers(pack, p, "win", 0.03, "hold")
        self.assertEqual([(i, s) for i, s, _ in got], [(0, S.YES)])

    def test_net_model_needs_the_probability_and_picks_the_better_side(self):
        pack = self.base(ask_y=[0.5])
        p = self.preds(x=[0])
        p[("net_hold", S.YES)], p[("net_hold", S.NO)] = np.array([0.62]), np.array([0.71])
        self.assertEqual([(i, s) for i, s, _ in M.triggers(pack, p, "net", 0.60, "hold")], [(0, S.NO)])
        self.assertEqual(M.triggers(pack, p, "net", 0.75, "hold"), [])
        self.assertEqual(M.triggers(pack, p, "net", 0.60, "tp5_stop7"), [])                 # that exit mode's model is silent

    def test_attempts_are_spaced_and_capped_per_window(self):
        pack = self.base(ask_y=[0.5] * 6)
        pack["t"] = np.array([0.0, 10.0, 70.0, 80.0, 140.0, 150.0])
        p = self.preds(x=[0] * 6)
        p[("win", S.YES)] = np.full(6, 0.9)
        self.assertEqual([i for i, _, _ in M.triggers(pack, p, "win", 0.03, "hold")], [0, 2])

    def test_the_grid_is_twelve_configs_and_names_round_trip(self):
        names = [M.config_name(*g) for g in M.grid()]
        self.assertEqual((len(names), len(set(names))), (12, 12))
        cfg = M.summarize_all({n: {"rows": [], "skips": Counter()} for n in names})
        self.assertEqual({(v["kind"], v["level"], v["exit_mode"]) for v in cfg.values()}, set(M.grid()))

    def test_selection_reads_the_out_of_fold_scorecard_not_the_refit(self):
        def row(pnl):
            return {"lane": "KXBTC15M", "close": 1.0, "pnl": str(pnl), "reason": "settle", "entry": "0.5", "fee_entry": "0",
                    "fee_exit": "0", "crossed": 0, "requotes": 0}
        name = M.config_name(*M.grid()[0])
        empty = {n: {"rows": [], "skips": Counter()} for n in (M.config_name(*g) for g in M.grid())}
        oof = {**empty, name: {"rows": [row(5), row(-1)], "skips": Counter()}}
        cfg = M.build_configs({"oof": oof, "insample": empty, "rolling_scaled": empty})
        self.assertEqual((cfg[name]["is"]["trades"], cfg[name]["is_insample"]["trades"]), (2, 0))

    def test_the_profit_factor_and_crossed_fill_gates_each_exclude(self):
        def cfg(pf, crossed):
            return {"is": {"trades": 20, "pf": pf, "profitable_lanes": 3, "crossed_fills": crossed, "net": 9.0}}
        self.assertTrue(M.pick_champion({"a": cfg(1.25, 0)})["is_eligible"])
        self.assertFalse(M.pick_champion({"a": cfg(1.24, 0)})["is_eligible"])
        self.assertFalse(M.pick_champion({"a": cfg(2.0, 1)})["is_eligible"])

    def test_champion_needs_fifteen_trades_pf_and_two_lanes(self):
        def cfg(trades, pf, lanes, net):
            return {"is": {"trades": trades, "pf": pf, "profitable_lanes": lanes, "crossed_fills": 0, "net": net}}
        got = M.pick_champion({"ok": cfg(15, 1.25, 2, 3.0), "few": cfg(14, 3.0, 3, 99.0), "onelane": cfg(40, 2.0, 1, 50.0)})
        self.assertEqual((got["config"], got["is_eligible"]), ("ok", True))
        none = M.pick_champion({"a": cfg(0, 0.0, 0, 0.0), "b": cfg(10, 0.9, 1, -2.0)})
        self.assertEqual((none["config"], none["is_eligible"]), ("b", False))


class CrossFit(unittest.TestCase):
    def packs(self, windows=10, per=4):
        n = windows * per
        close = np.repeat(np.arange(windows, dtype=float), per)
        cols = {k: np.zeros(n) for k in ("imb", "mpm", "gconf", "pty", "z", "b1", "b3", "b10", "e1", "e3", "e10", "bid", "secs")}
        cols.update(ask=np.full(n, 0.5), ask_y=np.full(n, 0.5), ask_n=np.full(n, 0.5), t=np.arange(n, dtype=float), close=close,
                    rank=close.astype(np.int64), tid=np.zeros(n, np.int64), y=np.ones(n), net={})
        for m in M.EXITS:
            for s in (S.YES, S.NO):
                cols["net"][(m, s)] = np.zeros(n)
        return {"KXBTC15M": cols}

    def test_each_row_is_predicted_once_by_a_model_trained_without_it_or_its_neighbours(self):
        packs, seen = self.packs(), []

        def fake_fit(_, rows):
            seen.append(rows["KXBTC15M"].copy())
            return {k: M.Const(0.5) for k in M.KINDS}

        splits = [{"KXBTC15M": A.fold_masks(packs["KXBTC15M"]["rank"], 10, k)} for k in range(A.FOLDS)]
        out = M.held_out_preds(packs, splits, fit=fake_fit)
        rank = packs["KXBTC15M"]["rank"]
        for (train, held), used in zip((s["KXBTC15M"] for s in splits), seen):
            np.testing.assert_array_equal(train, used)
            self.assertFalse((train & held).any())
            near = (rank >= rank[held].min() - 1) & (rank <= rank[held].max() + 1)         # the held block and one window each side
            self.assertFalse((train & near).any())
        for side in (S.YES, S.NO):
            self.assertTrue(np.isfinite(out["KXBTC15M"][("win", side)]).all())              # every row predicted exactly once
        self.assertEqual(len(seen), A.FOLDS)

    def test_rows_no_split_holds_out_get_no_prediction(self):
        packs = self.packs()
        rank = packs["KXBTC15M"]["rank"]
        splits = [{"KXBTC15M": (rank < 4, (rank >= 5) & (rank < 7))}]
        out = M.held_out_preds(packs, splits, fit=lambda *_: {k: M.Const(0.5) for k in M.KINDS})
        got = np.isfinite(out["KXBTC15M"][("win", S.YES)])
        np.testing.assert_array_equal(got, (rank >= 5) & (rank < 7))


class Evaluation(unittest.TestCase):
    def test_a_rescore_override_is_refused_before_anything_is_read(self):
        with patch.dict(os.environ, {"RND_OOS_RESCORE": "why not"}):
            with self.assertRaisesRegex(RuntimeError, "never rescore"):
                M.score_oos()

    def test_rolling_edge_bar_is_the_four_conditions(self):
        ok = {"trades": 12, "net": 0.01, "pf": 1.25, "profitable_lanes": 2}
        self.assertTrue(M.rolling_edge_bar(ok)["pass"])
        for key, bad in (("trades", 11), ("net", 0.0), ("pf", 1.24), ("profitable_lanes", 1)):
            self.assertFalse(M.rolling_edge_bar({**ok, key: bad})["pass"], key)

    def test_the_report_sets_a_rolling_pass_beside_its_out_of_fold_loss(self):
        def card(trades, net, pf, lanes):
            return {"trades": trades, "net": net, "pf": pf, "profitable_lanes": lanes}
        obj = {"rolling": {"scaled_3h_1p5h": {"folds": 1}}, "configs": {
            "good_roll": {"rolling_scaled": card(23, 126.17, 1.56, 2), "is": card(60, -238.13, 0.71, 0)},
            "bad_roll": {"rolling_scaled": card(30, -10.0, 0.9, 1), "is": card(40, 5.0, 1.1, 1)},
            "idle": {"rolling_scaled": card(0, 0.0, 0.0, 0), "is": card(0, 0.0, 0.0, 0)}}}
        text = M.rolling_text(obj)
        self.assertIn("1 meet the rolling-edge bar: `good_roll` rolling $+126.17, out-of-fold $-238.13 (PF 0.71)", text)
        self.assertIn("$-10.00 to $+126.17 across 2 configs", text)
        self.assertIn("noise-sized", text)

    def test_the_24h_12h_rolling_spec_needs_36_hours_and_the_scaled_one_has_ordered_disjoint_folds(self):
        closes = [20000.0 + 900.0 * i for i in range(30)]                                   # 7.5h of windows
        self.assertEqual(M.rolling_folds(closes, 24.0, 12.0), [])
        folds = M.rolling_folds(closes, 3.0, 1.5)
        self.assertGreaterEqual(len(folds), 2)
        seen = set()
        for tr, te in folds:
            self.assertLess(max(tr), min(te))
            self.assertFalse(seen & set(te))
            seen |= set(te)

    def test_an_ineligible_champion_never_touches_the_ledger_and_a_changed_model_file_is_refused(self):
        with tempfile.TemporaryDirectory() as dd:
            d = Path(dd)
            rec = {"frozen_input_sha256": {"x": "1"}, "oos_scored": None, "cut": 1.0, "l2_closes": [2.0],
                   "model": {"models_sha256": "0" * 64}, "champion": {"config": "win_0.03_hold", "is_eligible": False}}
            (d / "r.json").write_text(json.dumps(rec))
            with patch.object(M, "OUT", d / "r.json"), patch.object(M, "MODELS", d / "m.pkl"), patch.object(C, "LEDGER", d / "l.jsonl"), \
                    patch.object(M.T, "pin_tape", return_value={"sha256": {"x": "1"}}):
                with self.assertRaises(RuntimeError):
                    M.score_oos()
                rec["champion"]["is_eligible"] = True
                (d / "r.json").write_text(json.dumps(rec))
                (d / "m.pkl").write_bytes(b"not the frozen models")
                with self.assertRaisesRegex(RuntimeError, "frozen models changed"):
                    M.score_oos()
            self.assertFalse((d / "l.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
