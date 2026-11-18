"""The numba simulator against the canonical sim_core and against hand-worked trades."""
from __future__ import annotations

import unittest
from decimal import Decimal

import numpy as np
import polars as pl

from factory import sim
from factory.book import from_frame
from openkal import sim_core as S
from openkal.fees import taker_fee

NAMES = ["secs", "sig", "x"]
CLOSE0 = 1_800_000_000.0


def make_df(n_windows=1, seed=0, bid0=None, ask_pad=0.02, size=1e6, flat=None, y=None):
    """Windows of 781 one-second rows (840 .. 60 s left). `flat` fixes bid and ask; otherwise a 0.01-grid random walk."""
    rng = np.random.default_rng(seed)
    frames = []
    for w in range(n_windows):
        secs = np.arange(840, 59, -1, dtype=np.float32)
        if flat is not None:
            bid = np.full(len(secs), flat[0]); ask = np.full(len(secs), flat[1])
        else:
            bid = np.clip(np.round(0.5 + np.cumsum(rng.choice([-0.01, 0.0, 0.01], len(secs))), 2), 0.12, 0.8)
            ask = np.round(bid + ask_pad + 0.01 * rng.integers(0, 2, len(secs)), 2)
        frames.append(pl.DataFrame({"lane": np.zeros(len(secs), np.int8), "close": np.full(len(secs), CLOSE0 + 900.0 * w), "secs": secs,
                                    "y": np.full(len(secs), int(rng.integers(0, 2)) if y is None else y, np.int8), "bid": bid, "ask": ask,
                                    "bsz": np.full(len(secs), size, np.float32), "asz": np.full(len(secs), size, np.float32),
                                    "valid": np.ones(len(secs), bool), "sig": rng.choice([-1.0, 1.0], len(secs)).astype(np.float32),
                                    "x": rng.normal(size=len(secs)).astype(np.float32)}))
    return pl.concat(frames)


def genome(secs_le=600.0, **kw):
    g = sim.new_genome()
    g[sim.I_CF], g[sim.I_CD], g[sim.I_CT] = 0, -1, secs_le          # secs <= secs_le
    g[sim.I_SIDE_F] = 1
    for k, v in kw.items():
        g[getattr(sim, "I_" + k.upper())] = v
    return g


def tape_of(df, w):
    d = df.filter(pl.col("close") == CLOSE0 + 900.0 * w)
    close = float(d["close"][0])
    ts = (close - d["secs"].to_numpy().astype(np.float64))
    return S.Tape(ts, d["bid"].to_numpy().astype(np.float32), d["ask"].to_numpy().astype(np.float32), np.zeros(0), np.zeros(0, np.float32),
                  np.zeros(0, np.float32), np.zeros(0, np.int8), close, int(d["y"][0]))


class Parity(unittest.TestCase):
    def test_hold_to_settlement_taker_trades_equal_sim_core_exactly(self):
        df = make_df(40, seed=3)
        book = from_frame(df, NAMES)
        for stake_frac in (0.05, 0.2, 0.4):
            g = genome(stake=stake_frac)
            pnl, win, row, qty = sim.simulate(book, g, lat=5)
            self.assertGreater(len(pnl), 10)
            checked = 0
            for p, w, r, q in zip(pnl, win, row, qty):
                sig = df["sig"].to_numpy()[r]
                side = S.YES if sig > 0 else S.NO
                tape = tape_of(df, int(w))
                trigger = tape.close_ts - float(df["secs"].to_numpy()[r])
                cost = S.entry_price(side, S.TAKER, *tape.quote(trigger + 5.0))
                if cost > S.CAP:
                    continue
                want_qty = min(50, int(Decimal(str(stake_frac * 164.0)) / cost))
                tr = S.run_trade(tape, side=side, trigger_ts=trigger, entry_style=S.TAKER, qty=want_qty, lag_s=5.0)
                self.assertIsInstance(tr, S.Trade)
                self.assertEqual((tr.qty, float(tr.entry)), (int(q), float(cost)))
                self.assertAlmostEqual(float(tr.pnl), float(p), places=9)
                checked += 1
            self.assertGreater(checked, 10)

    def test_the_fee_matches_the_exact_decimal_formula_everywhere(self):
        bad = [(p, q) for p in np.round(np.arange(0.01, 0.99, 0.001), 3) for q in (1, 2, 5, 9, 17, 33, 50)
               if abs(float(taker_fee(Decimal(str(p)), q)) - sim.fee(float(p), q)) > 1e-9]
        self.assertEqual(bad, [])


class Mechanics(unittest.TestCase):
    def run_flat(self, bid, ask, y=1, **kw):
        df = make_df(1, flat=(bid, ask), y=y)
        df = df.with_columns(pl.lit(1.0).cast(pl.Float32).alias("sig"))
        return sim.simulate(from_frame(df, NAMES), genome(**kw), lat=kw.pop("lat", 5)) if False else sim.simulate(from_frame(df, NAMES), genome(**kw))

    def test_a_yes_buyer_pays_the_ask_a_no_buyer_pays_one_minus_the_bid(self):
        df = make_df(1, flat=(0.40, 0.43), y=1).with_columns(pl.lit(1.0).cast(pl.Float32).alias("sig"))
        yes = sim.simulate(from_frame(df, NAMES), genome(stake=0.1))
        self.assertEqual(int(yes[3][0]), int(16.4 / 0.43))
        self.assertAlmostEqual(yes[0][0], yes[3][0] * (1.0 - 0.43) - sim.fee(0.43, int(yes[3][0])), places=9)
        df_no = df.with_columns(pl.lit(-1.0).cast(pl.Float32).alias("sig"))
        no = sim.simulate(from_frame(df_no, NAMES), genome(stake=0.1))
        cost = 1.0 - 0.40                                                                  # NO ask = 1 - yes bid, and YES won so the NO loses
        self.assertAlmostEqual(no[0][0], -no[3][0] * cost - sim.fee(cost, int(no[3][0])), places=9)

    def test_fading_the_signal_flips_the_side(self):
        df = make_df(1, flat=(0.40, 0.43), y=1).with_columns(pl.lit(1.0).cast(pl.Float32).alias("sig"))
        follow = sim.simulate(from_frame(df, NAMES), genome(side_mode=1))
        fade = sim.simulate(from_frame(df, NAMES), genome(side_mode=-1))
        self.assertGreater(follow[0][0], 0)
        self.assertLess(fade[0][0], 0)

    def test_the_cost_ceiling_blocks_a_dear_entry_and_slippage_counts_against_it(self):
        df = make_df(1, flat=(0.60, 0.62), y=1).with_columns(pl.lit(1.0).cast(pl.Float32).alias("sig"))
        self.assertEqual(len(sim.simulate(from_frame(df, NAMES), genome())[0]), 1)             # exactly 0.62 is allowed
        self.assertEqual(len(sim.simulate(from_frame(df, NAMES), genome(), slip=0.01)[0]), 0)  # 0.63 paid is over the ceiling

    def test_the_stake_is_capped_at_fifty_contracts_and_by_the_displayed_depth(self):
        df = make_df(1, flat=(0.20, 0.22), y=1).with_columns(pl.lit(1.0).cast(pl.Float32).alias("sig"))
        self.assertEqual(int(sim.simulate(from_frame(df, NAMES), genome(stake=0.9))[3][0]), 50)
        thin = df.with_columns(pl.lit(7.0).cast(pl.Float32).alias("asz"))
        self.assertEqual(int(sim.simulate(from_frame(thin, NAMES), genome(stake=0.9))[3][0]), 7)
        unknown = df.with_columns(pl.lit(float("nan")).cast(pl.Float32).alias("asz"))                       # no size recorded: the lane's fallback depth
        self.assertEqual(int(sim.simulate(from_frame(unknown, NAMES, {0: 12.0}), genome(stake=0.9))[3][0]), 12)

    def test_latency_moves_the_arrival_row(self):
        df = make_df(1, flat=(0.40, 0.43)).with_columns(pl.lit(1.0).cast(pl.Float32).alias("sig"))
        bid = df["bid"].to_numpy().copy(); ask = df["ask"].to_numpy().copy()
        ask[:] = 0.43
        ask[5:] = 0.45                                                                     # the ask jumps five seconds into the window
        df = df.with_columns(pl.Series("ask", ask), pl.Series("bid", bid))
        first = sim.simulate(from_frame(df, NAMES), genome(secs_le=840.0, stake=0.1), lat=0)
        late = sim.simulate(from_frame(df, NAMES), genome(secs_le=840.0, stake=0.1), lat=5)
        self.assertEqual(int(first[3][0]), int(16.4 / 0.43))
        self.assertEqual(int(late[3][0]), int(16.4 / 0.45))

    def test_take_profit_exit_arrives_a_latency_later_and_pays_the_fee_on_both_legs(self):
        n = 781
        bid = np.full(n, 0.40); ask = np.full(n, 0.43)
        bid[100:] = 0.50; ask[100:] = 0.53                                               # a 10c rally from row 100
        df = make_df(1, flat=(0.4, 0.43), y=0).with_columns(pl.Series("bid", bid), pl.Series("ask", ask), pl.lit(1.0).cast(pl.Float32).alias("sig"))
        pnl, win, row, qty = sim.simulate(from_frame(df, NAMES), genome(secs_le=840.0, tp=0.05, stake=0.1))
        q = int(qty[0])
        want = q * (0.50 - 0.43) - sim.fee(0.43, q) - sim.fee(0.50, q)                    # sold at the bid after the rally, though YES lost
        self.assertAlmostEqual(pnl[0], want, places=9)

    def test_the_exit_price_is_the_book_a_latency_after_the_decision_not_at_the_decision(self):
        n = 781
        bid = np.full(n, 0.40); ask = np.full(n, 0.43)
        bid[100] = 0.50                                                                   # a one-second spike that triggers the take-profit
        bid[101:] = 0.46                                                                  # the price the exit order actually meets, five seconds on
        ask = bid + 0.03
        df = make_df(1, flat=(0.4, 0.43), y=0).with_columns(pl.Series("bid", bid), pl.Series("ask", ask), pl.lit(1.0).cast(pl.Float32).alias("sig"))
        pnl, win, row, qty = sim.simulate(from_frame(df, NAMES), genome(secs_le=840.0, tp=0.05, stake=0.1))
        q = int(qty[0])
        cost = 0.43                                                                       # bid 0.40 + 0.03 at the arrival row
        self.assertAlmostEqual(pnl[0], q * (0.46 - cost) - sim.fee(cost, q) - sim.fee(0.46, q), places=9)

    def test_stop_needs_two_confirming_seconds_and_a_single_wick_does_not_trigger_it(self):
        n = 781
        bid = np.full(n, 0.40); ask = np.full(n, 0.43)
        bid[50] = 0.30                                                                    # one wick
        df = make_df(1, flat=(0.4, 0.43), y=1).with_columns(pl.Series("bid", bid), pl.Series("ask", ask), pl.lit(1.0).cast(pl.Float32).alias("sig"))
        pnl, *_ = sim.simulate(from_frame(df, NAMES), genome(secs_le=840.0, stop=0.07))
        self.assertGreater(pnl[0], 0)                                                     # held to settlement, YES won
        bid2 = bid.copy(); bid2[50:] = 0.30                                              # a real fall
        df2 = df.with_columns(pl.Series("bid", bid2))
        pnl2, *_ = sim.simulate(from_frame(df2, NAMES), genome(secs_le=840.0, stop=0.07))
        self.assertLess(pnl2[0], 0)
        self.assertGreater(pnl2[0], -0.43 * 38)                                           # stopped out near 0.30, not held to a full loss

    def test_the_time_barrier_exits_at_the_book_a_latency_later(self):
        n = 781
        bid = np.full(n, 0.40); ask = np.full(n, 0.43)
        bid[40:] = 0.44; ask[40:] = 0.47                                                 # the book moves before the exit order arrives (row 5 + 40 + 5 = 50)
        df = make_df(1, flat=(0.4, 0.43), y=0).with_columns(pl.Series("bid", bid), pl.Series("ask", ask), pl.lit(1.0).cast(pl.Float32).alias("sig"))
        pnl, win, row, qty = sim.simulate(from_frame(df, NAMES), genome(secs_le=840.0, hold=40))
        q = int(qty[0])
        self.assertAlmostEqual(pnl[0], q * (0.44 - 0.43) - sim.fee(0.43, q) - sim.fee(0.44, q), places=9)

    def test_one_entry_per_window_unless_more_are_allowed_and_exits_free_the_slot(self):
        df = make_df(1, flat=(0.40, 0.43), y=1).with_columns(pl.lit(1.0).cast(pl.Float32).alias("sig"))
        self.assertEqual(len(sim.simulate(from_frame(df, NAMES), genome(max_ent=3))[0]), 1)          # held to settlement: no second entry
        n = 781
        bid = np.where(np.arange(n) % 40 < 20, 0.40, 0.50); ask = bid + 0.03
        df2 = make_df(1, flat=(0.4, 0.43), y=1).with_columns(pl.Series("bid", bid), pl.Series("ask", ask), pl.lit(1.0).cast(pl.Float32).alias("sig"))
        self.assertEqual(len(sim.simulate(from_frame(df2, NAMES), genome(secs_le=840.0, tp=0.05, max_ent=3, cool=20))[0]), 3)

    def test_nan_features_never_fire_and_an_empty_genome_never_trades(self):
        df = make_df(2).with_columns(pl.lit(float("nan")).cast(pl.Float32).alias("x"))
        g = genome()
        g[sim.I_CF + 1], g[sim.I_CD + 1], g[sim.I_CT + 1] = 2, 1, -9.0                    # x >= -9 would be true for any number
        self.assertEqual(len(sim.simulate(from_frame(df, NAMES), g)[0]), 0)
        self.assertEqual(len(sim.simulate(from_frame(df, NAMES), sim.new_genome())[0]), 0)

    def test_the_clauses_are_or_ed_and_the_conditions_in_a_clause_are_and_ed(self):
        df = make_df(1, flat=(0.40, 0.43)).with_columns(pl.lit(1.0).cast(pl.Float32).alias("sig"))
        never = genome(secs_le=0.0)                                                         # secs <= 0 is never true inside the window
        self.assertEqual(len(sim.simulate(from_frame(df, NAMES), never)[0]), 0)
        g = never.copy()
        g[sim.I_CF + 3], g[sim.I_CD + 3], g[sim.I_CT + 3] = 0, -1, 600.0                  # a second clause that is true
        self.assertEqual(len(sim.simulate(from_frame(df, NAMES), g)[0]), 1)
        h = genome(secs_le=600.0)
        h[sim.I_CF + 1], h[sim.I_CD + 1], h[sim.I_CT + 1] = 0, 1, 700.0                   # AND secs >= 700: impossible together with secs <= 600
        self.assertEqual(len(sim.simulate(from_frame(df, NAMES), h)[0]), 0)

    def test_the_entry_window_and_the_side_deadband_are_respected(self):
        df = make_df(1, flat=(0.40, 0.43)).with_columns(pl.lit(1.0).cast(pl.Float32).alias("sig"))
        r = sim.simulate(from_frame(df, NAMES), genome(secs_le=840.0, sec_hi=500.0, sec_lo=400.0))
        self.assertEqual(float(from_frame(df, NAMES).X[0][int(r[2][0])]), 500.0)           # first row inside the window
        self.assertEqual(len(sim.simulate(from_frame(df, NAMES), genome(side_dead=2.0))[0]), 0)   # |sig| = 1 is inside the deadband


if __name__ == "__main__":
    unittest.main()
