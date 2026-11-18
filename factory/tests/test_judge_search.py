"""The judge, the search and the epoch runner on synthetic books: the maths, the leak guard, and that every candidate is counted."""
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats as st

from factory import indicators as IND
from factory import judge as J
from factory import leaderboard as LB
from factory import run as R
from factory import screen, search, sim
from factory.book import from_frame

CLOSE0 = 1_800_000_000.0


def full_df(n_windows=48, seed=0, edge=0.0):
    """A synthetic matrix with every registry column. `edge` makes z predict the outcome, so a real signal exists when asked for."""
    rng = np.random.default_rng(seed)
    frames = []
    for w in range(n_windows):
        n = 781
        secs = np.arange(840, 59, -1, dtype=np.float32)
        bid = np.clip(np.round(0.5 + np.cumsum(rng.choice([-0.01, 0.0, 0.01], n)), 2), 0.12, 0.8)
        y = int(rng.integers(0, 2))
        cols = {"lane": np.full(n, w % 4, np.int8), "close": np.full(n, CLOSE0 + 900.0 * (w // 4)), "y": np.full(n, y, np.int8), "secs": secs,
                "bid": bid, "ask": np.round(bid + 0.02, 2), "bsz": np.full(n, 500.0, np.float32), "asz": np.full(n, 500.0, np.float32),
                "valid": np.ones(n, bool)}
        for name in IND.NAMES:
            if name not in cols:
                cols[name] = rng.normal(size=n).astype(np.float32)
        if edge:
            cols["z"] = (rng.normal(size=n) * 0.3 + edge * (2 * y - 1)).astype(np.float32)
        frames.append(pl.DataFrame(cols))
    return pl.concat(frames)


class Maths(unittest.TestCase):
    def test_the_corrected_bound_widens_with_the_number_of_candidates_and_never_below_the_plain_one(self):
        self.assertAlmostEqual(J.corrected_z(1), J.Z_UNCORRECTED)
        self.assertAlmostEqual(J.corrected_z(10_000), st.norm.isf(0.05 / 10_000))
        self.assertGreater(J.corrected_z(10_000), J.corrected_z(100))
        bu, bc = J.bounds(np.array([1.0]), np.array([0.2]), 10_000)
        self.assertAlmostEqual(bu[0], 1.0 - 1.6449 * 0.2, places=3)
        self.assertLess(bc[0], bu[0])

    def test_drawdown_is_the_largest_peak_to_trough_fall_of_the_path_from_zero(self):
        self.assertAlmostEqual(J.max_drawdown(np.array([10.0, -5.0, -10.0, 20.0, -4.0]), 100.0), 0.15)
        self.assertAlmostEqual(J.max_drawdown(np.array([-10.0, 5.0]), 100.0), 0.10)                        # the path starts at zero
        self.assertAlmostEqual(J.max_drawdown(np.array([-10.0, 5.0]), 50.0), 0.20)                         # scaled by the capital given
        self.assertEqual(J.max_drawdown(np.array([]), 100.0), 0.0)

    def test_deflated_sharpe_falls_as_trials_rise_and_matches_the_formula(self):
        x = np.random.default_rng(1).normal(0.2, 1, 240)
        one, many = J.deflated_sharpe(x, 1, 0.01), J.deflated_sharpe(x, 10_000, 0.01)
        self.assertGreater(one, 0.95)
        self.assertLess(many, 0.1)
        sr, sk, ku = x.mean() / x.std(ddof=1), st.skew(x), st.kurtosis(x, fisher=False)
        sr0 = J.expected_max_sharpe(500, 0.01)
        want = st.norm.cdf((sr - sr0) * math.sqrt(239) / math.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2))
        self.assertAlmostEqual(J.deflated_sharpe(x, 500, 0.01), want, places=12)
        self.assertEqual(J.deflated_sharpe(np.zeros(50), 10, 0.01), 0.0)

    def test_pbo_is_about_half_for_no_skill_and_small_when_one_candidate_has_a_real_edge(self):
        null = [J.pbo(np.random.default_rng(s).normal(0, 1, (12, 200)), max_splits=300)["pbo"] for s in range(25)]
        self.assertTrue(0.35 < float(np.mean(null)) < 0.65)
        m = np.random.default_rng(0).normal(0, 1, (12, 200))
        m[:, 5] += 2.0
        self.assertLess(J.pbo(m, max_splits=300)["pbo"], 0.1)
        self.assertIsNone(J.pbo(np.zeros((3, 50)))["pbo"])

    def test_epochs_train_on_everything_before_the_two_test_blocks_and_step_by_one(self):
        e = J.epochs(11)
        self.assertEqual((len(e), e[0]), (6, ([0, 1, 2, 3], [4, 5])))
        self.assertEqual(e[-1][1], [9, 10])
        self.assertTrue(all(max(t) < min(s) for t, s in e))
        self.assertEqual(J.epochs(5), [])                                                                    # too little data for even one epoch

    def test_windows_within_one_window_of_the_test_span_are_purged_from_train(self):
        wclose = CLOSE0 + 900.0 * np.arange(40)
        wblock = J.block_ids(wclose, block_hours=1.0)                                                      # 4 windows a block
        lab = J.labels(wclose, wblock, [0, 1, 2], [3, 4])
        self.assertTrue(np.all(lab[wblock == 3] == 2) and np.all(lab[wblock == 4] == 2))
        self.assertEqual(lab[11], 0)                                                                        # the window right before the first test window is purged
        self.assertEqual(lab[10], 1)                                                                        # two windows before is not
        self.assertFalse(np.any(lab[wblock > 4] == 1))                                                      # nothing after the test blocks is train

    def test_each_registered_condition_alone_rejects_with_its_own_reason_and_all_pass_passes(self):
        ok = dict(oos_n=250, oos_mean=1.0, bound_unc=0.5, bound_corr=0.2, dd=0.1, stress_mean=0.3, dsr=0.97, pbo_epoch=0.3)
        self.assertEqual(J.verdict(**ok)[0], "passed_screen")
        for key, bad, text in (("oos_n", 199, "OOS trades"), ("oos_mean", -0.1, "OOS EV"), ("bound_unc", -0.01, "OOS EV"), ("dd", 0.2, "drawdown"),
                               ("stress_mean", 0.0, "slippage"), ("bound_corr", -0.01, "corrected"), ("dsr", 0.94, "Sharpe"), ("pbo_epoch", 0.51, "PBO"),
                               ("pbo_epoch", None, "PBO")):
            status, reason = J.verdict(**{**ok, key: bad})
            self.assertEqual(status, "rejected", (key, bad))
            self.assertIn(text, reason)


class Screen(unittest.TestCase):
    def test_the_accumulators_match_a_brute_force_calculation_for_train_and_test_labels_separately(self):
        book = from_frame(full_df(40, seed=9), IND.NAMES)
        nw = len(book.wstart)
        sp = search.Space(IND.NAMES, search.quantile_table(book))
        rng = np.random.default_rng(3)
        G = np.stack([sp.random_genome(rng)[0] for _ in range(30)])
        label = np.where(np.arange(nw) % 3 == 0, 2, np.where(np.arange(nw) % 3 == 1, 1, 0))
        acc, blk = screen.screen(book, G, label, np.arange(nw) // 8, 5)
        for p in range(0, 30, 3):
            pnl, win, row, qty = sim.simulate(book, G[p])
            for lab in (1, 2):
                m = label[win] == lab
                x, w = pnl[m], win[m]
                n = m.sum()
                self.assertEqual(acc[p, lab, 0], n)
                self.assertAlmostEqual(acc[p, lab, 1], x.sum(), places=7)
                if n > 1:
                    mu = x.mean()
                    ws = np.unique(w)
                    sw = np.array([x[w == k].sum() for k in ws]); nws = np.array([(w == k).sum() for k in ws])
                    want_se = math.sqrt(((sw - mu * nws) ** 2).sum()) / n
                    got = screen.summarize(acc[p:p + 1, lab])
                    self.assertAlmostEqual(float(got["se"][0]), want_se, places=7)
                    self.assertAlmostEqual(float(got["mean"][0]), mu, places=9)
            self.assertAlmostEqual(blk[p].sum(), pnl.sum(), places=7)


class Search(unittest.TestCase):
    def setUp(self):
        self.book = from_frame(full_df(40, seed=1), IND.NAMES)
        self.space = search.Space(IND.NAMES, search.quantile_table(self.book))

    def test_random_genomes_are_valid_and_every_family_can_be_drawn(self):
        rng = np.random.default_rng(0)
        for fam in search.FAMILIES:
            for _ in range(30):
                g, f = self.space.random_genome(rng, fam)
                self.assertEqual((f, len(g)), (fam, sim.GENOME_LEN))
                used = g[sim.I_CF:sim.I_CF + 9]
                self.assertTrue((used >= 0).any())
                self.assertTrue(set(used[used >= 0].astype(int)) <= set(range(len(IND.NAMES))))
                self.assertIn(int(g[sim.I_SIDE_F]), self.space.side_feats if fam != "cross_coin_lead_lag" else [self.space.idx["btc_mom30"]])
                self.assertLessEqual(g[sim.I_SEC_LO] + 30, g[sim.I_SEC_HI])
        lead = self.space.random_genome(rng, "cross_coin_lead_lag")[0]
        self.assertIn(self.space.lane_idx, lead[sim.I_CF:sim.I_CF + 9])                                     # the lead is conditioned on lane

    def test_mutation_and_crossover_keep_a_condition_and_change_something(self):
        rng = np.random.default_rng(1)
        a, _ = self.space.random_genome(rng, "momentum")
        b, _ = self.space.random_genome(rng, "fade")
        changed = 0
        for _ in range(200):
            m = self.space.mutate(rng, a)
            self.assertTrue((m[sim.I_CF:sim.I_CF + 9] >= 0).any())
            changed += not np.array_equal(m, a)
            c = self.space.crossover(rng, a, b)
            self.assertEqual(len(c), sim.GENOME_LEN)
        self.assertGreater(changed, 150)

    def test_the_code_hash_is_stable_and_sees_a_threshold_and_the_registry_version(self):
        g = self.space.random_genome(np.random.default_rng(2))[0]
        self.assertEqual(search.code_hash(g), search.code_hash(g.copy()))
        h = g.copy(); h[sim.I_CT] += 1e-3
        self.assertNotEqual(search.code_hash(g), search.code_hash(h))

    def test_too_few_train_trades_is_worse_than_any_real_score(self):
        acc = np.zeros((3, 9))
        acc[0, :] = [10, 5.0, 6, 8, 3, 30, 40, 20, 5]                       # 10 trades, a good mean
        acc[1, :] = [100, -50.0, 40, 20, 70, 500, 600, 200, 40]             # 100 trades, a loss
        f = search.fitness(acc)
        self.assertLess(f[0], f[1])
        self.assertLess(f[2], f[1])

    def test_a_lower_standard_error_beats_a_higher_one_at_the_same_mean(self):
        acc = np.zeros((2, 9))
        acc[0, :] = [100, 100.0, 60, 150, 50, 100.0, 100.0, 100.0, 50]       # mean 1.0, se: var_sum = 100 - 200 + 100 = 0 -> 0
        acc[1, :] = [100, 100.0, 60, 150, 50, 900.0, 300.0, 100.0, 50]       # mean 1.0, var_sum = 900 - 600 + 100 = 400 -> se 0.2
        f = search.fitness(acc)
        self.assertGreater(f[0], f[1])

    def test_elites_are_carried_over_unchanged_and_the_rest_are_mutated_children(self):
        book = from_frame(full_df(40, seed=10, edge=0.8), IND.NAMES)
        sp = search.Space(IND.NAMES, search.quantile_table(book))
        pop = 50
        G, fams, acc = search.evolve(book, sp, np.random.default_rng(2), pop=pop, gens=2)
        gen0, gen1 = G[:pop], G[pop:]
        fit0 = search.fitness(acc[:pop])
        k = max(2, int(pop * 0.10))
        top = gen0[np.argsort(-fit0)[:k]]
        self.assertTrue(np.array_equal(gen1[:k], top))                                                       # the elites, in order, unchanged
        elite_set = {g.tobytes() for g in top}
        children = gen1[k:pop - max(1, pop // 20)]
        self.assertGreater(sum(c.tobytes() not in elite_set for c in children), len(children) * 0.8)          # most children differ from every elite

    def test_the_code_hash_changes_with_the_registry_version(self):
        g = np.arange(sim.GENOME_LEN, dtype=float)
        before = search.code_hash(g)
        old = IND.VERSION
        IND.VERSION = "provisional-x"
        try:
            self.assertNotEqual(before, search.code_hash(g))
        finally:
            IND.VERSION = old

    def test_leak_guard_the_evolved_population_does_not_depend_on_any_test_window(self):
        df = full_df(48, seed=3)
        wblock = J.block_ids(df.group_by(["lane", "close"], maintain_order=True).agg(pl.col("close").first().alias("c"))["c"].to_numpy(), 1.0)
        out = []
        for tamper in (False, True):
            d = df
            if tamper:                                                         # flip the outcomes and shuffle the quotes of the last 12 windows
                last = d["close"].to_numpy() >= CLOSE0 + 900.0 * 9
                d = d.with_columns(pl.when(pl.Series(last)).then(1 - pl.col("y")).otherwise(pl.col("y")).alias("y"),
                                   pl.when(pl.Series(last)).then(pl.col("bid") * 0 + 0.3).otherwise(pl.col("bid")).alias("bid"))
            book = from_frame(d, IND.NAMES)
            mask = book.wclose < CLOSE0 + 900.0 * 9
            train = search.subset(book, mask)
            sp = search.Space(IND.NAMES, search.quantile_table(train))
            G, fams, acc = search.evolve(train, sp, np.random.default_rng(5), pop=60, gens=3)
            out.append((G, acc))
        self.assertTrue(np.array_equal(out[0][0], out[1][0]))
        self.assertTrue(np.array_equal(out[0][1], out[1][1]))

    def test_optuna_free_search_improves_train_fitness_over_the_random_start(self):
        df = full_df(60, seed=4, edge=0.8)                                      # z carries the outcome: a real edge to find
        book = from_frame(df, IND.NAMES)
        sp = search.Space(IND.NAMES, search.quantile_table(book))
        G, fams, acc = search.evolve(book, sp, np.random.default_rng(1), pop=300, gens=6)
        f = search.fitness(acc[:, 1] if acc.ndim == 3 else acc)
        first, last = f[:300], f[-300:]
        self.assertGreater(np.sort(last)[-10:].mean(), np.sort(first)[-10:].mean())


class Epoch(unittest.TestCase):
    def test_every_trial_is_counted_every_candidate_is_a_row_and_noise_promotes_nothing(self):
        df = full_df(112, seed=6)
        book = from_frame(df, IND.NAMES)
        wblock = J.block_ids(book.wclose, 1.0)
        with tempfile.TemporaryDirectory() as d:
            db = LB.connect(Path(d) / "lb.db")
            ep = J.epochs(int(wblock.max()) + 1)[0]
            res = R.run_epoch(db, book, 0, ep[0], ep[1], wblock, np.random.default_rng(0), pop=120, gens=3, top_k=40)
            c = LB.counts(db)
            self.assertEqual(res["tried"], 360)
            self.assertEqual(c["trials"], 360)                                   # pop x gens evaluations, duplicates included
            self.assertGreaterEqual(c["candidate_rows"], 40)
            self.assertLessEqual(c["candidate_rows"], 360)
            self.assertEqual(c["by_status"].get("passed_screen", 0), 0)          # random books have no edge to pass the corrected bound
            row = db.execute("SELECT status, failure_reason, n_candidates_in_epoch, registry_version FROM candidates WHERE status != 'train_only' LIMIT 1").fetchone()
            self.assertEqual((row[0], row[2], row[3]), ("rejected", 360, IND.VERSION))
            self.assertIsNotNone(row[1])
            self.assertEqual(db.execute("SELECT COUNT(*) FROM epochs").fetchone()[0], 1)

    def test_the_stress_pass_runs_at_double_latency_and_one_cent_and_every_distinct_candidate_is_a_row(self):
        book = from_frame(full_df(112, seed=11), IND.NAMES)
        wblock = J.block_ids(book.wclose, 1.0)
        ep = J.epochs(int(wblock.max()) + 1)[0]
        calls = []
        orig = screen.screen

        def spy(*a, **k):
            calls.append((k.get("lat", 5), k.get("slip", 0.0)))
            return orig(*a, **k)

        with tempfile.TemporaryDirectory() as d:
            db = LB.connect(Path(d) / "lb.db")
            screen.screen = spy
            try:
                res = R.run_epoch(db, book, 0, ep[0], ep[1], wblock, np.random.default_rng(0), pop=80, gens=2, top_k=15)
            finally:
                screen.screen = orig
            self.assertIn((10, 0.01), calls)
            self.assertIn((5, 0.0), calls)
            self.assertEqual(LB.counts(db)["candidate_rows"], res["distinct"])                                  # train-only rows included: nothing is dropped

    def test_a_second_epoch_adds_to_the_counts_and_nothing_is_ever_removed(self):
        book = from_frame(full_df(112, seed=7), IND.NAMES)
        wblock = J.block_ids(book.wclose, 1.0)
        with tempfile.TemporaryDirectory() as d:
            db = LB.connect(Path(d) / "lb.db")
            eps = J.epochs(int(wblock.max()) + 1)
            R.run_epoch(db, book, 0, *eps[0], wblock, np.random.default_rng(0), pop=60, gens=2, top_k=20)
            before = LB.counts(db)
            R.run_epoch(db, book, 1, *eps[1], wblock, np.random.default_rng(1), pop=60, gens=2, top_k=20)
            after = LB.counts(db)
            self.assertEqual(after["trials"], before["trials"] + 120)
            self.assertGreaterEqual(after["candidate_rows"], before["candidate_rows"] + 20)

    def test_the_test_windows_of_an_epoch_are_never_in_the_search_book(self):
        book = from_frame(full_df(112, seed=8), IND.NAMES)
        wblock = J.block_ids(book.wclose, 1.0)
        tr, te = J.epochs(int(wblock.max()) + 1)[0]
        lab = J.labels(book.wclose, wblock, tr, te)
        seen = {}
        orig = search.evolve

        def spy(train_book, *a, **k):
            seen["closes"] = set(train_book.wclose.tolist())
            return orig(train_book, *a, **k)

        with tempfile.TemporaryDirectory() as d:
            db = LB.connect(Path(d) / "lb.db")
            search.evolve = spy
            try:
                R.run_epoch(db, book, 0, tr, te, wblock, np.random.default_rng(0), pop=40, gens=1, top_k=10)
            finally:
                search.evolve = orig
        test_closes = set(book.wclose[lab == 2].tolist())
        self.assertTrue(seen["closes"].isdisjoint(test_closes))
        self.assertEqual(seen["closes"], set(book.wclose[lab == 1].tolist()))


if __name__ == "__main__":
    unittest.main()
