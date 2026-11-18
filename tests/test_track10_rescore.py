"""Track 10, the champion re-score: the registration is pinned, and every number in it is the one the old code uses."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import numpy as np

from openkal import sim_core as S
from scripts import track10_champion_rescore as M

ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT / "scripts" / "track10_prereg.json"
PREREG_SHA256 = "0e55a88d67b47252cebe8c6c5ce9fa2fa907dcfe34389e2dc6fa6a64e654cd73"
SPEC = json.loads(SPEC_PATH.read_text())


class Registration(unittest.TestCase):
    def test_the_registered_file_has_not_drifted(self):
        self.assertEqual(hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest(), PREREG_SHA256)
        self.assertEqual(SPEC["version"], 1)
        self.assertNotIn("amendments", SPEC)

    def test_holm_counts_every_family_and_latency_including_the_two_that_cannot_be_scored(self):
        fams = SPEC["families"]
        self.assertEqual(sorted(fams), ["F2_taker_1s", "passive_mm_fade", "zobi_band_084", "zobi_xrp_carry"])
        self.assertEqual(SPEC["latencies_s"], [1, 2, 5])
        self.assertEqual(SPEC["holm"]["m"], len(fams) * len(SPEC["latencies_s"]))
        self.assertEqual([k for k, v in fams.items() if not v["scored"]], ["zobi_xrp_carry", "zobi_band_084"])
        self.assertIn("p = 1", SPEC["holm"]["meaning"])
        self.assertEqual(SPEC["holm"]["alpha"], 0.05)

    def test_the_two_unscored_families_are_blocked_by_the_canonical_ceiling_not_by_choice(self):
        from openkal import sim_core as S
        self.assertEqual(str(S.CAP), "0.62")
        for name in ("zobi_xrp_carry", "zobi_band_084"):
            self.assertIn("0.62", SPEC["families"][name]["reason_not_scored"])
        self.assertIn("0.84", SPEC["families"]["zobi_band_084"]["defined_by"])

    def test_f2_numbers_are_the_frozen_rule_of_f2_structural(self):
        from scripts import f2_structural as F
        self.assertEqual((F.Z_MIN, F.FLOW_MIN, F.REV_MIN, F.DELTA_BPS_MIN), (0.30, 0.10, 0.0, 3.0))
        self.assertEqual((F.COST_LO, F.COST_HI, F.CAP, F.T_LO, F.T_HI), (0.25, 0.60, 0.62, 180.0, 780.0))
        self.assertEqual((F.TP_C, F.STOP_C, F.STOP_CONFIRM_S, F.BARRIER_S, F.COOLDOWN, F.PER_WINDOW), (0.05, 0.07, 2.0, 90.0, 90.0, 3))
        f2 = SPEC["families"]["F2_taker_1s"]
        for text in ("|z| >= 0.30", "flow >= 0.10", "revision >= 0.0", "3 bp", "[0.25, 0.60]", "[180, 780]", "cooldown 90 s", "at most 3 per window"):
            self.assertIn(text, f2["trigger"])
        self.assertIn("[0.25, 0.62]", f2["entry"])
        for text in ("0.05", "0.07 confirmed for 2 s", "90 s"):
            self.assertIn(text, f2["exits"])

    def test_the_fade_is_the_frozen_champion_and_its_numbers_are_the_ones_passive_mm_fade_uses(self):
        from scripts import passive_mm_fade as P
        receipt = ROOT / "data" / "passive-mm-fade-is.json"
        if not receipt.exists():
            self.skipTest("the IS receipt is not on this machine")
        champion = json.loads(receipt.read_text())["champion"]["config"]
        self.assertEqual(champion, "flow_thr2_vq0-1_tp0.06_scratch_hold180")
        self.assertEqual((P.COOLDOWN, P.PER_WINDOW, str(P.COST_HI)), (90.0, 3, "0.55"))
        self.assertEqual(P.EXITS["scratch_hold180"], {"hold_s": 180.0, "barrier_exit": "scratch"})
        self.assertEqual((P.THRS[1], P.VOL_BANDS[0], str(P.TP_CS[1])), (2.0, (0.0, 1.0), "0.06"))
        fade = SPEC["families"]["passive_mm_fade"]
        for text in ("|z_obi| >= 2.0", "(NO: 1 - ask, YES: bid) at most 0.55", "cooldown 90 s, at most 3 per window"):
            self.assertIn(text, fade["trigger"])
        self.assertIn("scratch", fade["exits"])
        self.assertIn("180 s", fade["exits"])
        self.assertIn("0.06", fade["exits"])
        self.assertIn("assumption", fade)                                                  # the family name is an assumption, written down

    def test_the_freeze_is_the_last_window_of_the_frozen_cache(self):
        frozen = ROOT / "data" / "frozen-cache"
        if not (frozen / "markets.json").exists():
            self.skipTest("the frozen cache is not on this machine")
        from scripts import track4_common as T
        closes = T.window_closes(frozen)
        self.assertEqual((len(closes), closes[-1]), (SPEC["freeze"]["frozen_cache_windows"], SPEC["freeze"]["freeze_close_epoch"]))

    def test_the_decision_rules_are_written_before_any_data(self):
        self.assertEqual((SPEC["test"]["null"], SPEC["verdict"]["edge_confirmed"]),
                         ("mean <= 0", "Holm-rejected and at least 30 post-freeze trades"))
        self.assertIn("draws 10000, seed 0", SPEC["test"]["p_value"])
        self.assertIn("fewer than 30", SPEC["verdict"]["underpowered"])
        self.assertIn("5 folds", SPEC["walk_forward"]["folds"])
        self.assertTrue(SPEC["walk_forward"]["nothing_refit"].startswith("all thresholds, the fade's z-score normaliser and every exit are frozen"))


FREEZE = M.FREEZE_CLOSE


def row(close, pnl, family="F2_taker_1s", lane="KXBTC15M", reason="tp"):
    return {"lane": lane, "close": close, "pnl": str(pnl), "reason": reason, "entry": "0.40", "fee_entry": "0.01", "fee_exit": "0",
            "crossed": 0, "requotes": 0, "family": family, "slice": "post" if close > FREEZE else "pre"}


class Constants(unittest.TestCase):
    def test_the_code_uses_the_registered_numbers(self):
        self.assertEqual((M.LATENCIES, M.SCORED, M.M, M.ALPHA), ((1, 2, 5), ("F2_taker_1s", "passive_mm_fade"), 12, 0.05))
        self.assertEqual((M.DRAWS, M.SEED, M.MIN_TRADES, M.FOLDS_PRE), (10_000, 0, 30, 5))
        self.assertEqual(M.FREEZE_CLOSE, SPEC["freeze"]["freeze_close_epoch"])
        self.assertIn("draws 10000, seed 0", SPEC["test"]["p_value"])
        self.assertIn("at least 30", SPEC["verdict"]["edge_confirmed"])
        self.assertIn("5 folds", SPEC["walk_forward"]["folds"])
        self.assertEqual(M.SPEC_SHA256, PREREG_SHA256)


class Statistics(unittest.TestCase):
    def test_holm_steps_down_over_twelve_hypotheses_not_over_the_cells_that_were_scored(self):
        h = M.holm({"a": 0.004, "b": 0.0045, "c": 0.2})
        self.assertAlmostEqual(h["a"]["threshold"], 0.05 / 12)                               # m is 12 even though three cells were scored
        self.assertEqual([h[k]["rejected"] for k in "abc"], [True, True, False])
        self.assertAlmostEqual(h["b"]["threshold"], 0.05 / 11)
        late = M.holm({"a": 0.005, "b": 0.0001})                                            # sorted: b first; a misses its 0.05/11
        self.assertEqual((late["b"]["rejected"], late["a"]["rejected"]), (True, False))

    def test_a_later_p_value_under_its_own_threshold_is_still_not_rejected_after_an_earlier_failure(self):
        h = M.holm({"a": 0.03, "b": 0.04}, m=2)                                             # a misses 0.025, so b may not use its 0.05
        self.assertEqual((h["a"]["rejected"], h["b"]["rejected"]), (False, False))

    def test_once_a_step_fails_no_later_p_value_is_rejected(self):
        h = M.holm({"a": 0.0001, "b": 0.2, "c": 0.2001, "d": 0.0002}, m=4)
        self.assertEqual({k: v["rejected"] for k, v in h.items()}, {"a": True, "d": True, "b": False, "c": False})

    def test_the_bootstrap_resamples_windows_not_trades(self):
        windows = [float(i) for i in range(10)]
        rows = [row(0.0 + 0, 1.0) for _ in range(20)] + [row(float(i), -1.0) for i in range(1, 10)]
        p = M.bootstrap_p(windows, rows)
        self.assertGreater(p["mean"], 0.3)                                                   # 11 net over 29 trades: positive
        self.assertGreater(p["p"], 0.25)                                                     # but every draw that misses window 0 is a loss

    def test_a_draw_with_no_trades_counts_as_a_mean_of_zero(self):
        p = M.bootstrap_p([1.0, 2.0, 3.0], [row(1.0, 1.0), row(1.0, 1.0)])
        self.assertTrue(0.25 < p["p"] < 0.34)                                                # (2/3)^3 = 0.296 of draws miss window 1

    def test_a_clear_edge_is_significant_a_loss_is_not_and_no_trades_is_p_one(self):
        windows = [float(i) for i in range(70)]
        win = M.bootstrap_p(windows, [row(float(i), 2.0) for i in range(70)])
        loss = M.bootstrap_p(windows, [row(float(i), -2.0) for i in range(70)])
        none = M.bootstrap_p(windows, [])
        self.assertEqual(win["p"], 1 / 10_001)
        self.assertEqual(loss["p"], 1.0)
        self.assertEqual((none["p"], none["mean"]), (1.0, None))

    def test_the_bootstrap_is_reproducible_and_reads_only_the_given_rows(self):
        windows = [float(i) for i in range(40)]
        rows = [row(float(i), (-1) ** i * 1.5) for i in range(40)]
        self.assertEqual(M.bootstrap_p(windows, rows), M.bootstrap_p(windows, rows))
        self.assertNotEqual(M.bootstrap_p(windows, rows, seed=1)["p"], M.bootstrap_p(windows, rows, seed=2)["p"])

    def test_the_verdict_needs_thirty_trades_and_a_holm_rejection(self):
        self.assertEqual(M.verdict(29, True), "underpowered")
        self.assertEqual(M.verdict(0, False), "underpowered")
        self.assertEqual(M.verdict(30, True), "edge_confirmed")
        self.assertEqual(M.verdict(30, False), "no_edge_shown")

    def test_folds_are_five_equal_pre_freeze_blocks_and_the_post_freeze_windows_are_the_sixth(self):
        pre = [FREEZE - 900.0 * k for k in range(168)]
        post = [FREEZE + 900.0 * (k + 1) for k in range(70)]
        f = M.folds_of(sorted(pre + post))
        sizes = [sum(1 for v in f.values() if v == k) for k in range(1, 7)]
        self.assertEqual(sizes, [34, 34, 34, 33, 33, 70])
        self.assertEqual((f[min(pre)], f[FREEZE], f[FREEZE + 900.0]), (1, 5, 6))             # the freeze close itself is pre-freeze
        self.assertTrue(all(f[a] <= f[b] for a, b in zip(sorted(f), sorted(f)[1:])))         # folds run forward in time


class FakeTape:
    def __init__(self):
        self.quotes = []

    def quote(self, ts):
        self.quotes.append(ts)
        return 0.40, 0.42


class Trading(unittest.TestCase):
    def call(self, family, lag, side=S.NO):
        tape, seen = FakeTape(), []
        with patch.object(M.S, "run_trade", lambda tp, **kw: seen.append(kw) or S.Skip("unfilled")):
            M.run_one({"family": family, "ts": 100.0, "side": side}, tape, lag)
        return seen[0], tape

    def test_the_scenario_latency_reaches_run_trade_and_the_stake_quote(self):
        for lag in (1, 2, 5):
            kw, tape = self.call("F2_taker_1s", lag)
            self.assertEqual((kw["lag_s"], tape.quotes), (lag, [100.0 + lag]))
            kw, tape = self.call("passive_mm_fade", lag)
            self.assertEqual((kw["lag_s"], tape.quotes), (lag, [100.0 + lag]))

    def test_every_entry_is_a_taker_and_nothing_asks_for_a_cost_above_the_canonical_ceiling(self):
        for fam in M.SCORED:
            kw, _ = self.call(fam, 2)
            self.assertEqual(kw["entry_style"], S.TAKER)
            self.assertNotIn("max_cost", kw)                                                  # run_trade's default is the 0.62 cap
            self.assertEqual(kw["side"], S.NO)

    def test_f2_and_the_fade_use_their_own_frozen_exits(self):
        f2, _ = self.call("F2_taker_1s", 1)
        self.assertEqual((f2["cost_lo"], f2["tp_c"], f2["stop_c"], f2["stop_confirm_s"], f2["hold_s"], f2["barrier_exit"]),
                         (Decimal("0.25"), Decimal("0.05"), Decimal("0.07"), 2.0, 90.0, "taker"))
        fade, _ = self.call("passive_mm_fade", 1)
        self.assertEqual((fade["cost_hi"], fade["tp_c"], fade["hold_s"], fade["barrier_exit"]), (Decimal("0.55"), Decimal("0.06"), 180.0, "scratch"))
        self.assertNotIn("stop_c", fade)

    def test_a_crossed_or_empty_book_leaves_the_stake_for_run_trade_to_skip(self):
        class Crossed:
            def quote(self, ts):
                raise S.CrossedBook("crossed")
        self.assertEqual(M.stake_qty(Crossed(), S.YES, 1.0, 1), S.MAX_QTY)

    def test_rows_are_split_at_the_freeze_and_skips_are_counted_per_slice_and_reason(self):
        trig = [{"family": "F2_taker_1s", "lane": "KXBTC15M", "tid": i, "ticker": "T", "close": c, "ts": 1.0, "side": S.YES}
                for i, c in enumerate([FREEZE - 900, FREEZE, FREEZE + 900, FREEZE + 1800])]
        tapes = {("KXBTC15M", i): FakeTape() for i in range(4)}
        seen = []

        def fake(tape, **kw):
            seen.append(kw["lag_s"])
            return S.Skip("cost_band") if len(seen) in (2, 4) else None

        with patch.object(M, "run_one", lambda t, tape, lag: fake(tape, lag_s=lag) or S.Skip("unfilled")), \
                patch.object(M.RC, "trade_row", lambda tr, **kw: kw):
            rows, skips = M.score_cell(trig, tapes, 5)
        self.assertEqual(rows, [])
        self.assertEqual(skips, {"pre": {"cost_band": 1, "unfilled": 1}, "post": {"cost_band": 1, "unfilled": 1}})
        self.assertEqual(seen, [5, 5, 5, 5])

    def test_the_fade_buys_the_opposite_side_of_the_flow_and_uses_the_frozen_normaliser_without_refitting(self):
        calls = []
        P = {"tid": np.array([0, 0, 0]), "ts": np.array([10.0, 200.0, 400.0]), "close": np.array([900.0] * 3)}
        lane = type("L", (), {"tickers": ["T0"]})()

        def prepare(l, name, cut, which, norm=None):
            calls.append((name, cut, which, norm, M.PM.POOL_DIR))
            return P, norm, 3

        with patch.object(M.PM, "prepare", prepare), \
                patch.object(M.PM, "trigger_mask", lambda P_, cfg: (np.array([True, True, True]), np.array([2.5, -2.5, 3.0]))):
            out = M.fade_triggers("KXBTC15M", lane, Path("/pools"), {"KXBTC15M": {"flow": {"mean": 0.1, "std": 0.5}}})
        self.assertEqual([t["side"] for t in out], [S.NO, S.YES, S.NO])                      # flow buying YES buys NO
        self.assertEqual(calls[0][1:3], (float("inf"), "is"))                                # every row 'in sample': nothing is refit
        self.assertEqual(calls[0][3], {"flow": {"mean": 0.1, "std": 0.5}})
        self.assertEqual(calls[0][4], Path("/pools"))
        self.assertNotEqual(M.PM.POOL_DIR, Path("/pools"))                                   # the module's own pool dir is put back

    def test_f2_triggers_take_the_side_of_the_spot_and_the_frozen_cooldown(self):
        pool = {"tid": np.array([0, 0]), "ts": np.array([1.0, 2.0]), "close": np.array([900.0, 900.0]), "z": np.array([0.5, -0.5]),
                "flow": np.array([0.2, -0.2]), "rev": np.array([0.1, -0.1]), "delta": np.array([0.05, 0.05]), "spot": np.array([100.0, 100.0])}
        lane = type("L", (), {"tickers": ["T0"]})()
        asked = []

        def fake_pick(pool_, mask, *, cooldown, per_window):
            asked.append((cooldown, per_window))
            return [0, 1]

        with patch.object(M.F, "frozen_mask", lambda p, style: (np.array([True, True]), np.array([True, False]))), patch.object(M, "pick", fake_pick):
            out = M.f2_triggers("KXXRP15M", lane, pool)
        self.assertEqual([t["side"] for t in out], [S.YES, S.NO])
        self.assertEqual(asked, [(90.0, 3)])
        self.assertEqual([t["feat"] for t in out], [[0.5, 0.2, 0.1, 5.0], [-0.5, 0.2, 0.1, 5.0]])           # signed by the trade's direction: z, flow, rev, delta in bp


class Run(unittest.TestCase):
    """The whole run with the heavy parts replaced, to pin the ledger order, the gate and the single-run rule."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.pre = [FREEZE - 900.0 * k for k in range(168)]
        self.post = [FREEZE + 900.0 * (k + 1) for k in range(70)]
        self.closes = sorted(self.pre + self.post)
        self.pool_for = {"x": "y"}                                                           # the pool directory was built from the pinned tape

    def tearDown(self):
        self.tmp.cleanup()

    def triggers(self, pre_f2=111, pre_fade=56, post=3):
        def make(fam, pre_closes, n_post):
            rows = [{"family": fam, "lane": "KXBTC15M", "tid": i, "ticker": "T", "close": c, "ts": 1.0, "side": S.YES} for i, c in enumerate(pre_closes)]
            return rows + [{"family": fam, "lane": "KXBTC15M", "tid": 1000 + i, "ticker": "T", "close": self.post[i], "ts": 1.0, "side": S.YES}
                           for i in range(n_post)]
        f2 = [self.pre[i % 168] for i in range(pre_f2)]
        fade = [M.FADE_IS_CUT - 900.0 * (1 + i % 10) for i in range(pre_fade)]              # all before the fade's own cut, hence pre-freeze
        return make("F2_taker_1s", f2, post), make("passive_mm_fade", fade, post)

    def go(self, f2, fade, ledger_during=None, **patches):
        ledger = self.dir / "ledger.jsonl"
        score_calls = []

        def score_cell(triggers, tapes, lag):
            score_calls.append(ledger.read_text().count("\n") if ledger.exists() else 0)
            fam = triggers[0]["family"]
            return [row(self.post[0], 1.5, fam), row(self.pre[0], -1.0, fam)], {"pre": {"cost_band": 1}, "post": {}}

        pools = patch.object(M, "load_pool", lambda name, d: {})
        pool_dir = self.dir / "pool"
        pool_dir.mkdir(exist_ok=True)
        (pool_dir / M.POOL_PIN).write_text(json.dumps({"tape_sha256": self.pool_for}) + "\n")
        with patch.object(M.T, "pin_tape", lambda tape: {"sha256": {"x": "y"}}), patch.object(M.T, "window_closes", lambda tape: self.closes), \
                patch.object(M.RC, "load_lanes", lambda: {n: object() for n in M.LANES}), patch.object(M, "fade_norm", lambda: {}), pools, \
                patch.object(M, "f2_triggers", lambda n, l, p: f2 if n == "KXBTC15M" else []), \
                patch.object(M, "fade_triggers", lambda n, l, d, nm: fade if n == "KXBTC15M" else []), \
                patch.object(M.RC, "tape", lambda lane, tid: FakeTape()), patch.object(M, "score_cell", score_cell), \
                patch.object(M.RC, "LEDGER", ledger), patch.object(M, "ROOT", self.dir):
            out = M.run(self.dir / "tape", self.dir / "pool", self.dir)
        return out, [json.loads(x) for x in ledger.read_text().splitlines()], score_calls

    def test_six_started_lines_are_on_the_ledger_before_any_cell_is_scored_and_six_scored_lines_follow(self):
        f2, fade = self.triggers()
        out, lines, calls = self.go(f2, fade)
        self.assertEqual(calls, [6] * 6)                                                     # every look logged before the first score
        self.assertEqual([x["status"] for x in lines], ["started"] * 6 + ["scored"] * 6)
        names = {x["family"] for x in lines}
        self.assertEqual(names, {f"rescore10:{f}@{l}s" for f in M.SCORED for l in (1, 2, 5)})
        self.assertTrue(all(x["rescore"] and x["rescore_reason"].startswith("Track 10") for x in lines))
        self.assertFalse(names & {"F2_taker_1s", "passive_mm_fade"})                         # never under a family's own name
        self.assertEqual(out["holm"], {"m": 12, "alpha": 0.05, "scored_cells": 6})

    def test_the_receipt_the_trades_and_the_window_ledger_are_written_and_the_pin_cannot_be_scored_twice(self):
        f2, fade = self.triggers()
        out, _, _ = self.go(f2, fade)
        receipt = self.dir / f"track10-rescore-{int(self.closes[-1])}.json"
        stamp = int(self.closes[-1])
        self.assertTrue(receipt.exists() and (self.dir / f"track10-window-ledger-{stamp}.json").exists())
        self.assertEqual(len((self.dir / f"track10-rescore-trades-{stamp}.jsonl").read_text().splitlines()), 12)
        cell = out["cells"]["F2_taker_1s"]["5"]
        self.assertEqual((cell["post"]["trades"], cell["pre"]["trades"], cell["triggers"]), (1, 1, {"pre": 111, "post": 3}))
        self.assertEqual(cell["verdict"], "underpowered")
        self.assertEqual(out["windows"]["post_freeze"], 70)
        ledger = json.loads((self.dir / f"track10-window-ledger-{stamp}.json").read_text())
        self.assertEqual(sorted(ledger), ["F2_taker_1s", "passive_mm_fade", "zobi_band_084", "zobi_xrp_carry"])
        self.assertTrue(ledger["zobi_band_084"]["pre_freeze"]["seen"])
        self.assertTrue(ledger["F2_taker_1s"]["post_freeze"]["seen_before_this_run"])                  # its forward look reached past the freeze
        self.assertFalse(ledger["passive_mm_fade"]["post_freeze"]["seen_before_this_run"])
        self.assertEqual(ledger["F2_taker_1s"]["pre_freeze"]["windows"], 168)
        with self.assertRaisesRegex(RuntimeError, "already scored"):
            self.go(f2, fade)

    def test_pools_that_do_not_rebuild_the_recorded_triggers_stop_the_run_before_anything_is_logged(self):
        f2, fade = self.triggers(pre_f2=110)
        with self.assertRaisesRegex(RuntimeError, "do not reproduce"):
            self.go(f2, fade)
        self.assertFalse((self.dir / "ledger.jsonl").exists())
        f2, fade = self.triggers(pre_fade=57)
        with self.assertRaisesRegex(RuntimeError, "do not reproduce"):
            self.go(f2, fade)

    def test_a_tape_with_no_post_freeze_window_is_refused(self):
        self.closes = self.pre
        f2, fade = self.triggers()
        with self.assertRaisesRegex(RuntimeError, "no post-freeze"):
            self.go(f2, fade)


    def test_a_later_pin_does_not_overwrite_the_first_pins_trades_or_ledger(self):
        f2, fade = self.triggers()
        self.go(f2, fade)
        first = self.dir / f"track10-rescore-trades-{int(self.closes[-1])}.jsonl"
        before = first.read_text()
        self.closes = self.closes + [self.closes[-1] + 900.0]                                 # a later pin: one more window
        self.go(f2, fade)
        self.assertEqual(first.read_text(), before)
        self.assertTrue((self.dir / f"track10-rescore-trades-{int(self.closes[-1])}.jsonl").exists())

    def test_pools_built_from_another_tape_or_with_no_pin_record_are_refused_before_anything_is_logged(self):
        f2, fade = self.triggers()
        self.pool_for = {"x": "other"}
        with self.assertRaisesRegex(RuntimeError, "not built from this pinned tape"):
            self.go(f2, fade)
        self.assertFalse((self.dir / "ledger.jsonl").exists())
        self.pool_for = {"x": "y"}
        (self.dir / "pool" / M.POOL_PIN).unlink()
        with patch.object(M.T, "pin_tape", lambda tape: {"sha256": {"x": "y"}}), patch.object(M.T, "window_closes", lambda tape: self.closes), \
                patch.object(M.RC, "LEDGER", self.dir / "ledger.jsonl"):
            with self.assertRaisesRegex(RuntimeError, "not built from this pinned tape"):
                M.run(self.dir / "tape", self.dir / "pool", self.dir)


class Building(unittest.TestCase):
    def test_the_pool_builder_never_overwrites_the_older_pools_or_a_directory_that_already_holds_pools(self):
        with self.assertRaisesRegex(RuntimeError, "rnd-pool"):
            M.build_pools(M.TAPE, M.RB.POOL)
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "slim_KXBTC15M.npz").write_bytes(b"x")
            with self.assertRaisesRegex(RuntimeError, "already holds pools"):
                M.build_pools(M.TAPE, Path(d))

    def test_a_pool_build_records_which_pinned_tape_it_came_from(self):
        with tempfile.TemporaryDirectory() as d, patch.object(M.T, "pin_tape", lambda t: {"sha256": {"a": "b"}}), \
                patch.object(M.RC, "load_lanes", lambda names: {names[0]: object()}), \
                patch.object(M.F, "build_pools", lambda lanes: {n: {"z": np.array([0.5]), **{k: np.array([1.0]) for k in M.RB.KEEP if k != "z"}} for n in lanes}):
            M.build_pools(M.TAPE, Path(d) / "new", lanes=("KXBTC15M",))
            self.assertEqual(json.loads((Path(d) / "new" / M.POOL_PIN).read_text()), {"tape_sha256": {"a": "b"}})

    def test_a_tape_is_pinned_once_and_only_from_a_consistent_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / "tape"
            with self.assertRaisesRegex(RuntimeError, "changed while it was copied"):
                M.build_tape(dest, snapshot=lambda p: None)
            self.assertFalse((dest / "pin.json").exists())
            M.build_tape(dest, snapshot=lambda p: {"a": "b"})
            self.assertEqual(json.loads((dest / "pin.json").read_text())["sha256"], {"a": "b"})
            with self.assertRaisesRegex(RuntimeError, "already pinned"):
                M.build_tape(dest, snapshot=lambda p: {"a": "b"})


class Features(unittest.TestCase):
    def test_f2_features_travel_with_the_trade_row_and_the_fade_carries_none(self):
        seen = []
        trig = [{"family": "F2_taker_1s", "lane": "KXBTC15M", "tid": 0, "ticker": "T", "close": FREEZE + 900.0, "ts": 1.0, "side": S.YES, "feat": [1.0, 2.0, 3.0, 4.0]},
                {"family": "passive_mm_fade", "lane": "KXBTC15M", "tid": 1, "ticker": "T", "close": FREEZE + 900.0, "ts": 1.0, "side": S.NO}]
        with patch.object(M, "run_one", lambda t, tape, lag: object()), patch.object(M.RC, "trade_row", lambda tr, **kw: seen.append(kw) or kw):
            M.score_cell(trig, {("KXBTC15M", 0): None, ("KXBTC15M", 1): None}, 1)
        self.assertEqual([kw["feat"] for kw in seen], [[1.0, 2.0, 3.0, 4.0], None])


class WindowLedger(unittest.TestCase):
    PRE = [FREEZE - 900.0 * k for k in range(168)]
    POST = [FREEZE + 900.0 * (k + 1) for k in range(70)]

    def test_f2_windows_inside_its_forward_look_are_recorded_as_seen_and_the_others_are_not(self):
        wl = M.window_ledger(sorted(self.PRE + self.POST), {}, {"F2_taker_1s": {FREEZE + 900.0}, "passive_mm_fade": set()})
        f2 = wl["F2_taker_1s"]["post_freeze"]
        end = M.F2_FORWARD_LOOK["inferred_end_close"]
        early = [c for c in self.POST if c <= end]
        self.assertTrue(f2["seen_before_this_run"])
        self.assertEqual(f2["forward_look"]["windows_possibly_scored_by_that_look"], early)
        self.assertTrue(0 < len(early) < len(self.POST))
        self.assertEqual(f2["forward_look"]["fitted_on"], "nothing")
        for fam in ("passive_mm_fade", "zobi_xrp_carry", "zobi_band_084"):
            self.assertFalse(wl[fam]["post_freeze"]["seen_before_this_run"], fam)
            self.assertNotIn("forward_look", wl[fam]["post_freeze"])

    def test_the_forward_look_inference_is_ten_hours_from_five_in_the_morning(self):
        self.assertEqual(M.F2_FORWARD_LOOK["slice_start"], 1794909600.0)                       # 2026-11-17 05:00 EST, forward_oos.FORWARD_OOS_START_TS
        self.assertEqual(M.F2_FORWARD_LOOK["inferred_end_close"] - M.F2_FORWARD_LOOK["slice_start"], 36_000.0)
        self.assertTrue(M.F2_FORWARD_LOOK["slice_start"] < FREEZE < M.F2_FORWARD_LOOK["inferred_end_close"])   # the look straddles the freeze

    def test_every_evidence_file_is_looked_up_where_it_lives_and_a_missing_one_is_marked_missing(self):
        wl = M.window_ledger(sorted(self.PRE + self.POST), {}, {})
        for fam, entry in wl.items():
            for ev in entry["pre_freeze"]["evidence"]:
                path = Path(ev["path"]) if ev["path"].startswith("/") else ROOT / ev["path"]
                self.assertEqual(ev["mtime"] is not None, path.exists(), (fam, ev["path"]))

    def test_the_f2_evidence_that_exists_on_this_machine_is_found_with_its_time(self):
        real = Path("/home/cayleb/open-kal/data/f2-structural.json")
        if not real.exists():
            self.skipTest("the first-session artifacts are not on this machine")
        wl = M.window_ledger(sorted(self.PRE + self.POST), {}, {})
        found = {ev["path"]: ev["mtime"] for ev in wl["F2_taker_1s"]["pre_freeze"]["evidence"]}
        self.assertIsNotNone(found[str(real)])

    def test_the_ledger_can_be_rebuilt_alone_from_the_pin_and_the_trades_file(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rows = [{"family": "F2_taker_1s", "slice": "post", "close": FREEZE + 900.0}, {"family": "F2_taker_1s", "slice": "pre", "close": FREEZE},
                    {"family": "passive_mm_fade", "slice": "post", "close": FREEZE + 1800.0}]
            (d / f"track10-rescore-trades-{int(sorted(self.PRE + self.POST)[-1])}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            with patch.object(M.T, "pin_tape", lambda t: {}), patch.object(M.T, "window_closes", lambda t: sorted(self.PRE + self.POST)), \
                    patch.object(M.RC, "LEDGER", d / "none.jsonl"):
                wl = M.rebuild_window_ledger(d / "tape", d)
            self.assertEqual(wl["F2_taker_1s"]["post_freeze"]["windows_with_a_trade"], [FREEZE + 900.0])
            self.assertEqual(wl["passive_mm_fade"]["post_freeze"]["windows_with_a_trade"], [FREEZE + 1800.0])
            self.assertTrue((d / f"track10-window-ledger-{int(sorted(self.PRE + self.POST)[-1])}.json").exists())
            self.assertFalse((d / "none.jsonl").exists())                                       # no ledger line is written by a rebuild


if __name__ == "__main__":
    unittest.main()
