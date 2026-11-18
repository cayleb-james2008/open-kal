"""Track 4B on hand-built tapes: the decision filter, the bid rule, and sim_core pricing."""
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
from scripts import track4b_asian_maker as B

CLOSE = 2000.0


def fake_lane(book, prints=(), *, y=1, close=CLOSE, strike=100.0):
    name = "KXBTC15M-TEST"
    return SimpleNamespace(
        lane="KXBTC15M", tickers=[name], meta={name: {"strike": strike, "close": close, "y": y}},
        ts=np.array([r[0] for r in book], float), bid=np.array([r[1] for r in book], float),
        ask=np.array([r[2] for r in book], float), book_bounds=np.array([0, len(book)]),
        trade_ts=np.array([p[0] for p in prints], float), trade_px=np.array([p[1] for p in prints], float),
        trade_sz=np.array([p[2] for p in prints], float), trade_ty=np.array([p[3] for p in prints], np.int8),
        trade_bounds=np.array([0, len(prints)]))


class Model:
    """Fair value stub: `p` for YES, k locked seconds = 60 - secs_left."""

    def __init__(self, p):
        self.p = p

    def fair(self, t, close, strike):
        return {"p": self.p, "elapsed": 60.0 - (close - t)}


BOOK = [(CLOSE - 200.0, 0.85, 0.87)]
CUT = CLOSE + 900.0   # this window is IS


def run(p, prints, hurdle=0.06, book=BOOK, y=1):
    lane = fake_lane(book, prints, y=y)
    pack = B.collect(lane, Model(p), CUT, "is")
    out = {"c": {"rows": [], "skips": __import__("collections").Counter()}}
    B.run_hurdle(pack, hurdle, "c", out)
    return pack, out["c"]


class Decisions(unittest.TestCase):
    def test_only_locked_fifteen_or_more_seconds_and_the_floor_qualify(self):
        pack = B.collect(fake_lane(BOOK), Model(0.9), CUT, "is")
        self.assertEqual(len(pack["t"]), 21)               # secs_left 45..25: k = 15..35
        self.assertEqual(float(pack["secs"].max()), 45.0)
        self.assertEqual(pack["skips"]["locked_below_15"], 30)
        self.assertEqual(B.collect(fake_lane(BOOK), Model(0.6), CUT, "is")["skips"]["fair_below_70"], 21)

    def test_a_low_yes_probability_is_a_no_decision_on_the_complement(self):
        pack = B.collect(fake_lane(BOOK), Model(0.1), CUT, "is")
        self.assertEqual(set(pack["side"]), {S.NO})
        self.assertAlmostEqual(float(pack["p_side"][0]), 0.9)

    def test_the_other_split_is_not_sampled(self):
        self.assertEqual(len(B.collect(fake_lane(BOOK), Model(0.9), CUT, "oos")["t"]), 0)


class BidRule(unittest.TestCase):
    def test_min_of_cap_and_rounded_fair_minus_hurdle_with_a_floor(self):
        self.assertEqual(B.bid_for(0.90, 0.06), D("0.62"))
        self.assertEqual(B.bid_for(0.75, 0.12), D("0.62"))     # 0.63 is capped
        self.assertEqual(B.bid_for(0.70, 0.12), D("0.58"))
        self.assertEqual(B.bid_for(0.70, 0.25), D("0.45"))
        self.assertIsNone(B.bid_for(0.70, 0.30))               # 0.40 < the $0.45 floor


class Trades(unittest.TestCase):
    def test_a_dump_through_the_bid_fills_at_the_bid_and_settles(self):
        t = CLOSE - 45.0
        pack, res = run(0.9, [(t + 2.0, 0.60, 60, 0)])
        self.assertEqual(len(res["rows"]), 1)
        row = res["rows"][0]
        self.assertEqual((row["entry"], row["side"], row["reason"], D(row["fee_entry"])),
                         ("0.62", "YES", "settle", D(0)))
        self.assertEqual(row["pnl"], str((D("1") - D("0.62")) * S.MAX_QTY))
        self.assertEqual(row["order_ts"], t + S.LAG_S)
        self.assertEqual(len(row["feat"]), 4)

    def test_a_dump_that_stops_above_the_bid_or_under_fifty_contracts_does_not_fill(self):
        t = CLOSE - 45.0
        self.assertEqual(run(0.9, [(t + 2.0, 0.70, 500, 0)])[1]["skips"]["unfilled"], 1)
        self.assertEqual(run(0.9, [(t + 2.0, 0.60, 49, 0)])[1]["skips"]["unfilled"], 1)

    def test_a_no_bid_is_lifted_by_yes_buyers_and_pays_when_no_wins(self):
        t = CLOSE - 45.0
        book = [(CLOSE - 200.0, 0.13, 0.15)]                   # NO ask 0.87, NO bid 0.85
        _, res = run(0.1, [(t + 2.0, 0.38, 60, 1)], book=book, y=0)
        row = res["rows"][0]
        self.assertEqual((row["side"], row["entry"], row["exit_value"]), ("NO", "0.62", "1"))
        self.assertEqual(run(0.1, [(t + 2.0, 0.38, 60, 0)], book=book, y=0)[1]["skips"]["unfilled"], 1)

    def test_a_bid_that_would_cross_the_book_is_refused(self):
        book = [(CLOSE - 200.0, 0.55, 0.60)]                   # the market sits under the 0.62 bid
        self.assertEqual(run(0.9, [], book=book)[1]["skips"]["post_only"], 1)

    def test_one_decision_per_window_and_a_hurdle_that_empties_the_bid(self):
        pack, res = run(0.9, [(CLOSE - 43.0, 0.60, 60, 0)])
        self.assertEqual(len(res["rows"]) + sum(res["skips"].values()), 1)
        self.assertEqual(run(0.7, [], hurdle=0.30)[1]["skips"]["bid_below_floor"], 21)


class Sealed(unittest.TestCase):
    def test_an_ineligible_champion_never_touches_the_ledger(self):
        with tempfile.TemporaryDirectory() as d:
            out, ledger = Path(d) / "r.json", Path(d) / "ledger.jsonl"
            out.write_text(json.dumps({"frozen_input_sha256": {"x": "1"}, "oos_scored": None, "cut": 1.0,
                                       "champion": {"config": "h0.06_maker", "is_eligible": False},
                                       "configs": {"h0.06_maker": {"hurdle": 0.06}}}))
            with patch.object(B, "OUT", out), patch.object(C, "LEDGER", ledger), \
                    patch.object(B.T, "pin_tape", return_value={"sha256": {"x": "1"}}):
                with self.assertRaises(RuntimeError):
                    B.score_oos()
            self.assertFalse(ledger.exists())


if __name__ == "__main__":
    unittest.main()
