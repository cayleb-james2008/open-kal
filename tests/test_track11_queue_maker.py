"""Track 11, the queue-aware maker idea: the registration is pinned and its numbers are the ones the simulator and the rule use."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import numpy as np

from openkal import sim_core as S
from scripts import rnd_common as RC
from scripts import track11_queue_maker as M

ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT / "scripts" / "track11_prereg.json"
PREREG_SHA256 = "ed39fd6c19f604c129833374e0e8d60728a19d10833d7740424f2ebd0d6d45d4"
SPEC = json.loads(SPEC_PATH.read_text())


class Registration(unittest.TestCase):
    def test_the_registered_file_has_not_drifted(self):
        self.assertEqual(hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest(), PREREG_SHA256)
        self.assertEqual(SPEC["version"], 2)
        self.assertIsNone(SPEC["oos_scored"])
        self.assertIn("no trade of this track has been simulated", SPEC["status"])
        (v2,) = SPEC["amendments"]
        self.assertEqual((v2["version"], v2["supersedes_commit"], v2["changed"]), (2, "ab5ede3", ["split.oos", "status"]))
        self.assertIn("no rule, number, window, split, bar or control", v2["not_changed"])

    def test_the_split_is_sixty_percent_of_the_46_l2_windows_and_the_rest_is_sealed(self):
        sp = SPEC["split"]
        self.assertEqual((sp["is_windows"], sp["oos_windows"], sp["is_windows"] + sp["oos_windows"]), (27, 19, 46))
        self.assertEqual(int(0.6 * 46), 27)
        self.assertIn("l2_closes[int(0.6 * 46)]", sp["rule"])
        self.assertIn("never indexed, built into a tape, simulated or counted", sp["oos"])
        self.assertEqual(sp["cut_epoch"], 1795001400.0)

    def test_the_rule_numbers_are_in_the_text_and_agree_with_the_canonical_simulator(self):
        r = SPEC["rule"]
        self.assertIn("660, 540, 420 and 300 seconds", r["decision_points"])
        self.assertIn("[0.40, 0.60]", r["price_band"])
        self.assertIn("at most 100 contracts", r["queue_band"])
        self.assertIn("patience 100 s", r["order"])
        self.assertIn("65.58", r["order"])
        self.assertEqual(str(RC.STAKE), "65.58")
        self.assertEqual((S.MAX_QTY, S.LAG_S, str(S.CAP), S.MIN_PRINT), (50, 1.0, "0.62", 50.0))
        self.assertIn("at most 50 contracts", r["order"])
        self.assertIn("queue_model=True", r["all_orders_through"])
        self.assertIn("held to settlement", r["exit"])

    def test_the_patience_is_shorter_than_the_gap_between_decisions_so_orders_never_overlap(self):
        secs = [660, 540, 420, 300]
        self.assertTrue(all(a - b >= 100 for a, b in zip(secs, secs[1:])))

    def test_the_queue_model_text_matches_what_sim_core_implements(self):
        q = SPEC["queue_position_model"]
        for text in ("yes_bid_sz", "yes_ask_sz", "inside the spread has 0 ahead", "below the touch"):
            self.assertIn(text, q["queue_ahead"])
        self.assertIn("queue_ahead + own contracts", q["fill"])
        self.assertIn("strictly through", q["fill"])
        self.assertIn("50 contracts", q["fill"])
        self.assertTrue(hasattr(S.Tape, "queue_ahead") and hasattr(S.Tape, "queue_fill"))
        self.assertIn("queue_model", S.run_trade.__code__.co_varnames)

    def test_the_bar_is_the_earlier_tracks_bar_and_clearing_it_spends_nothing(self):
        bar = " ".join(SPEC["is_bar"]["needs_all_of"])
        for text in ("at least 30 fills", "net > 0", "profit factor >= 1.25", "at least 2 profitable lanes", "p < 0.05", "10000 draws, seed 0"):
            self.assertIn(text, bar)
        self.assertIn("never an OOS look", SPEC["is_bar"]["meaning"])
        self.assertEqual(len(SPEC["controls_descriptive"]), 3)


CLOSE = 10_000.0
ARRIVE = CLOSE - 660.0 + 1.0            # the first decision point's order arrives here


def mk(bid, ask, bsz, asz, prints=(), close=CLOSE, row_ts=None):
    """A tape with one book row long before the first decision point and the given prints (ts, yes_px, size, taker_yes)."""
    pr = list(prints)
    tp = S.Tape(np.array([close - 900.0 if row_ts is None else row_ts]), np.array([bid], dtype=np.float32), np.array([ask], dtype=np.float32),
                np.array([p[0] for p in pr], dtype=np.float64), np.array([p[1] for p in pr], dtype=np.float32),
                np.array([p[2] for p in pr], dtype=np.float32), np.array([p[3] for p in pr], dtype=np.int8), close, 1)
    return dataclasses.replace(tp, yes_bid_sz=np.array([bsz], dtype=np.float32), yes_ask_sz=np.array([asz], dtype=np.float32))


class Decisions(unittest.TestCase):
    def test_the_favourite_is_the_side_with_the_higher_bid_and_ties_are_skipped(self):
        yes = M.decide(mk(0.55, 0.58, 60, 999), 660.0, "primary")
        self.assertEqual(yes[:2], ("order", S.YES))
        no = M.decide(mk(0.40, 0.43, 999, 60), 660.0, "primary")                        # NO bid = 1 - 0.43 = 0.57
        self.assertEqual(no[:2], ("order", S.NO))
        self.assertEqual(M.decide(mk(0.45, 0.55, 60, 60), 660.0, "primary"), ("skip", "tie"))

    def test_the_queue_that_counts_is_the_favourites_own_touch(self):
        self.assertEqual(M.decide(mk(0.55, 0.58, 60, 5000), 660.0, "primary")[0], "order")       # a long ask queue does not matter to a YES bid
        self.assertEqual(M.decide(mk(0.40, 0.43, 5000, 60), 660.0, "primary")[0], "order")       # a NO bid waits behind the YES ask size
        self.assertEqual(M.decide(mk(0.55, 0.58, 5000, 60), 660.0, "primary"), ("skip", "queue_long"))
        self.assertEqual(M.decide(mk(0.40, 0.43, 60, 5000), 660.0, "primary"), ("skip", "queue_long"))

    def test_the_price_band_is_inclusive_and_binds_both_ways(self):
        self.assertEqual(M.decide(mk(0.60, 0.63, 60, 60), 660.0, "primary")[:2], ("order", S.YES))          # 0.60 is in
        self.assertEqual(M.decide(mk(0.61, 0.64, 60, 60), 660.0, "primary"), ("skip", "band"))
        self.assertEqual(M.decide(mk(0.40, 0.65, 60, 60), 660.0, "primary")[:2], ("order", S.YES))          # 0.40 is in (NO bid 0.35 is lower)
        self.assertEqual(M.decide(mk(0.38, 0.65, 60, 60), 660.0, "primary"), ("skip", "band"))              # the favourite's 0.38 is under it
        self.assertEqual(M.decide(mk(0.40, 0.55, 60, 60), 660.0, "primary")[:2], ("order", S.NO))           # NO bid 0.45 beats the YES bid
        self.assertEqual(M.decide(mk(0.30, 0.63, 60, 60), 660.0, "primary"), ("skip", "band"))              # NO bid 0.37, YES bid 0.30: both under 0.40

    def test_the_queue_cap_is_one_hundred_contracts_inclusive_and_an_unknown_queue_is_skipped(self):
        self.assertEqual(M.decide(mk(0.55, 0.58, 100, 1), 660.0, "primary")[0], "order")
        self.assertEqual(M.decide(mk(0.55, 0.58, 101, 1), 660.0, "primary"), ("skip", "queue_long"))
        self.assertEqual(M.decide(mk(0.55, 0.58, float("nan"), 1), 660.0, "primary"), ("skip", "queue_unknown"))
        self.assertEqual(M.decide(mk(0.55, 0.58, 100, 1), 660.0, "long_queue"), ("skip", "queue_short"))
        self.assertEqual(M.decide(mk(0.55, 0.58, 101, 1), 660.0, "long_queue")[0], "order")

    def test_the_stake_is_sized_at_the_bid_and_capped(self):
        self.assertEqual(M.decide(mk(0.55, 0.58, 60, 60), 660.0, "primary")[2], 50)                         # 65.58 / 0.55 = 119, capped at 50
        self.assertEqual(M.decide(mk(0.55, 0.58, 60, 60), 660.0, "primary")[3], CLOSE - 660.0)

    def test_a_crossed_or_missing_quote_is_skipped_not_traded(self):
        self.assertEqual(M.decide(mk(0.58, 0.55, 60, 60), 660.0, "primary"), ("skip", "crossed"))
        self.assertEqual(M.decide(mk(0.55, 0.58, 60, 60, row_ts=CLOSE - 100.0), 660.0, "primary"), ("skip", "no_quote"))    # the first row comes after the order would arrive


class Orders(unittest.TestCase):
    def go(self, tp, arm="primary"):
        counts = Counter()
        res, secs = M.run_window(tp, arm, counts)
        return res, secs, counts

    def test_the_order_waits_its_turn_behind_the_queue_where_the_old_rule_fills_at_once(self):
        prints = [(ARRIVE + 5.0, 0.55, 70.0, 0), (ARRIVE + 9.0, 0.55, 50.0, 0)]                               # queue 60 + 50 own = 110 needed
        tp = mk(0.55, 0.58, 60, 999, prints)
        res, secs, counts = self.go(tp)
        self.assertEqual((res.legs[0].ts, secs, res.entry, res.legs[0].fee), (ARRIVE + 9.0, 660.0, Decimal("0.55"), Decimal("0")))
        old, _, _ = self.go(tp, "legacy_fill")
        self.assertEqual(old.legs[0].ts, ARRIVE + 5.0)

    def test_only_one_position_per_lane_window_and_later_decision_points_are_not_used(self):
        tp = mk(0.55, 0.58, 60, 999, [(ARRIVE + 2.0, 0.55, 500.0, 0)])
        res, secs, counts = self.go(tp)
        self.assertEqual((counts["placed"], counts["filled"], secs), (1, 1, 660.0))

    def test_an_unfilled_order_is_cancelled_and_the_next_decision_point_tries_again(self):
        late = CLOSE - 540.0 + 30.0                                                                           # after the first order's 100 s are over
        tp = mk(0.55, 0.58, 60, 999, [(late, 0.55, 500.0, 0)])
        res, secs, counts = self.go(tp)
        self.assertEqual((counts["placed"], counts["unfilled"], counts["filled"], secs), (2, 1, 1, 540.0))
        self.assertEqual(res.legs[0].ts, late)

    def test_a_window_where_nothing_prints_places_four_orders_and_fills_none(self):
        res, secs, counts = self.go(mk(0.55, 0.58, 60, 999))
        self.assertEqual((res, counts["placed"], counts["unfilled"]), (None, 4, 4))

    def test_the_long_queue_arm_trades_only_long_queues_and_the_primary_never_does(self):
        prints = [(ARRIVE + 3.0, 0.55, 700.0, 0)]
        tp = mk(0.55, 0.58, 500, 999, prints)
        res, _, counts = self.go(tp, "primary")
        self.assertEqual((res, counts["queue_long"]), (None, 4))
        res, _, counts = self.go(tp, "long_queue")
        self.assertEqual(res.legs[0].ts, ARRIVE + 3.0)

    def test_the_position_is_held_to_settlement_with_no_exit_leg(self):
        tp = mk(0.55, 0.58, 60, 999, [(ARRIVE + 2.0, 0.55, 500.0, 0), (ARRIVE + 30.0, 0.62, 500.0, 1)])      # a later rally that would have lifted a +5c exit
        res, _, _ = self.go(tp)
        self.assertEqual((res.reason, [leg.role for leg in res.legs]), ("settle", ["entry", "settle"]))            # no tp, stop or barrier order
        self.assertEqual(res.pnl, Decimal("0.45") * 50)                                                       # YES settled in the money: 1 - 0.55 on 50 contracts


class Bar(unittest.TestCase):
    OK = {"trades": 30, "net": 1.0, "pf": 1.25, "profitable_lanes": 2}

    def test_every_check_must_hold(self):
        self.assertTrue(M.bar(self.OK, 0.049)["eligible"])
        for key, bad, p in (("trades", 29, 0.01), ("net", 0.0, 0.01), ("pf", 1.2499, 0.01), ("profitable_lanes", 1, 0.01), ("trades", 30, 0.05)):
            s = dict(self.OK, **{key: bad})
            self.assertFalse(M.bar(s, p)["eligible"], (key, bad, p))

    def test_the_bar_numbers_are_the_registered_ones(self):
        self.assertEqual((M.MIN_FILLS, M.MIN_PF, M.MIN_LANES, M.ALPHA), (30, 1.25, 2, 0.05))
        self.assertEqual((M.DECISION_SECS, M.Q_MAX, M.PATIENCE_S, str(M.BAND_LO), str(M.BAND_HI)), ((660.0, 540.0, 420.0, 300.0), 100.0, 100.0, "0.40", "0.60"))
        self.assertEqual((M.N_L2, M.CUT_EPOCH), (46, 1795001400.0))
        self.assertEqual(M.SPEC_SHA256, PREREG_SHA256)


class Windows(unittest.TestCase):
    def write_tape(self, d, fractions, settled=True):
        """One window per lane at close 100 with the given fraction of finite touch sizes, and one more at close 200 always finite."""
        meta = {}
        for lane in M.LANES:
            frac = fractions[lane]
            tickers = [f"{lane}-A", f"{lane}-B"]
            n = 10
            tid = np.repeat([0, 1], n)
            bsz = np.where(np.arange(2 * n) % n < int(frac * n), 5.0, np.nan).astype(np.float32)
            bsz[n:] = 5.0
            np.savez(d / f"book_{lane}.npz", tickers=np.array(tickers), tid=tid.astype(np.int32), yes_bid_sz=bsz, yes_ask_sz=bsz.copy())
            meta[tickers[0]] = {"close": 100.0, "y": 1 if settled else None, "strike": 1.0}
            meta[tickers[1]] = {"close": 200.0, "y": 0, "strike": 1.0}
        (d / "markets.json").write_text(json.dumps(meta))

    def test_a_window_needs_sizes_on_more_than_half_its_rows_in_every_lane(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self.write_tape(d, {l: 0.6 for l in M.LANES})
            self.assertEqual(M.l2_closes(d), [100.0, 200.0])
            self.write_tape(d, {**{l: 0.6 for l in M.LANES}, "KXXRP15M": 0.5})                 # exactly half is not more than half
            self.assertEqual(M.l2_closes(d), [200.0])
            self.write_tape(d, {l: 0.6 for l in M.LANES}, settled=False)                       # unsettled windows are not windows
            self.assertEqual(M.l2_closes(d), [200.0])

    def test_the_cut_is_sixty_percent_in_order_and_in_sample_is_everything_before_it(self):
        closes = [float(i) for i in range(46)]
        cut, is_closes = M.split_of(closes)
        self.assertEqual((cut, len(is_closes), max(is_closes)), (27.0, 27, 26.0))


class Run(unittest.TestCase):
    """The whole run with the loaders replaced: only in-sample windows are ever built, nothing touches the ledger, one run only."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.closes = [M.CUT_EPOCH - 900.0 * (27 - i) if i < 27 else M.CUT_EPOCH + 900.0 * (i - 27) for i in range(46)]
        self.built = []
        meta = {f"T{i}": {"close": c, "y": 1, "strike": 1.0} for i, c in enumerate(self.closes + [self.closes[-1] + 900.0])}
        self.lane = type("L", (), {"tickers": list(meta), "meta": meta})()

    def tearDown(self):
        self.tmp.cleanup()

    def go(self, closes=None, ledger=None):
        closes = closes if closes is not None else self.closes
        ledger = ledger or (self.dir / "ledger.jsonl")

        def sized(lane, tid, sizes):
            self.built.append(float(lane.meta[lane.tickers[tid]]["close"]))
            fill = [(float(lane.meta[lane.tickers[tid]]["close"]) - 659.0 + 3.0, 0.55, 700.0, 0)]
            return mk(0.55, 0.58, 60, 999, fill, close=float(lane.meta[lane.tickers[tid]]["close"]))

        with patch.object(M.T, "pin_tape", lambda t: {"sha256": {"a": "b"}}), patch.object(M, "l2_closes", lambda t: closes), \
                patch.object(M.RC, "load_lanes", lambda: {n: self.lane for n in M.LANES}), patch.object(M, "load_sizes", lambda t, n: (None, None)), \
                patch.object(M, "sized_tape", sized), patch.object(M.T.C, "LEDGER", ledger):
            return M.run(self.dir / "tape", self.dir / "out.json", self.dir / "trades.jsonl"), ledger

    def test_only_in_sample_windows_are_ever_built_into_a_tape_or_counted(self):
        result, _ = self.go()
        in_sample = set(self.closes[:27])
        self.assertTrue(set(self.built) <= in_sample)
        self.assertEqual(set(self.built), in_sample)                                         # every in-sample window, once per lane
        self.assertEqual(len(self.built), 27 * len(M.LANES))
        self.assertEqual(result["windows"], {"l2": 46, "in_sample": 27, "oos_sealed": 19})
        rows = [json.loads(x) for x in (self.dir / "trades.jsonl").read_text().splitlines()]
        self.assertTrue(rows and all(r["close"] in in_sample for r in rows))

    def test_the_receipt_says_the_oos_look_is_unspent_and_the_ledger_is_not_touched(self):
        result, ledger = self.go()
        receipt = json.loads((self.dir / "out.json").read_text())
        self.assertIsNone(receipt["oos_scored"])
        self.assertEqual(receipt["spec_sha256"], PREREG_SHA256)
        self.assertEqual(sorted(receipt["arms"]), ["legacy_fill", "long_queue", "primary"])
        self.assertFalse(ledger.exists())

    def test_a_tape_whose_l2_windows_are_not_the_registered_ones_stops_before_any_simulation(self):
        with self.assertRaisesRegex(RuntimeError, "not the registered"):
            self.go(closes=self.closes[:-1])
        self.assertEqual(self.built, [])
        self.assertFalse((self.dir / "out.json").exists())

    def test_the_registered_run_is_made_once(self):
        self.go()
        with self.assertRaisesRegex(RuntimeError, "already made"):
            self.go()

    def test_a_receipt_that_holds_an_oos_look_or_a_ledger_line_blocks_a_rerun(self):
        ledger = self.dir / "ledger.jsonl"
        ledger.write_text(json.dumps({"family": M.FAMILY}) + "\n")
        with self.assertRaisesRegex(RuntimeError, "ledger entry"):
            self.go(ledger=ledger)


if __name__ == "__main__":
    unittest.main()
