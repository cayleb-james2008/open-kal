from __future__ import annotations

import asyncio
import tempfile
import time
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from openkal.kalshi_ws import Book
from openkal.lanes import flip_cheap as F


def _book(yes_bid: str, no_bid: str, yes_sz: str = "100", no_sz: str = "100") -> Book:
    return Book(
        ticker="T",
        yes={Decimal(yes_bid): Decimal(yes_sz)},
        no={Decimal(no_bid): Decimal(no_sz)},
        synced=True,
        updated_ts=time.time(),
    )


def _signal(coin: F.Coin, ts: float, *, filled: bool = True, side: str = "NO",
            mode: str = "paper", ask: str = "0.26") -> F.Signal:
    return F.Signal(
        ts=ts, coin=coin, ticker=f"{coin.series}-26NOV181200-00", close_ts=ts + 60,
        side=side, fell_side="YES", high_ts=ts - 30, high_mid=Decimal("0.96"),
        current_mid=Decimal("0.75"), drop=Decimal("0.21"), volume_60=10.0, p75=5.0,
        yes_mid=Decimal("0.75"), no_mid=Decimal("0.25"), ask=Decimal(ask),
        depth=Decimal("100"), filled=filled, contracts=10 if filled else 0,
        fee=Decimal("0") if filled else Decimal("0"),
        fee_source="kalshi_taker_formula" if filled else "",
        no_fill_reason="" if filled else "depth_below_10", mode=mode,
    )


class FlipCheapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = Path(self.tmp.name) / "flip.db"

    def test_nine_coins(self) -> None:
        self.assertEqual(len(F.COINS), 9)
        self.assertEqual([c.asset for c in F.COINS],
                         ["BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "HYPE", "NEAR", "ZEC"])
        self.assertEqual(F.COIN_BY_SERIES["KXNEAR15M"].asset, "NEAR")
        self.assertEqual(F.COIN_BY_SERIES["KXZEC15M"].asset, "ZEC")

    def test_drop_candidate_yes(self) -> None:
        now = 1000.0
        window = F.MarketWindow(F.COIN_BY_SERIES["KXBTC15M"], "T", now + 100, 100.0)
        window.observe(_book("0.95", "0.03"), now - 50)
        window.observe(_book("0.74", "0.24"), now)
        candidate = window.drop_candidate(now)
        self.assertIsNotNone(candidate)
        assert candidate is not None
        # The cheap side is the one that rose, not the favourite that fell.
        self.assertEqual(candidate.side, "NO")
        self.assertEqual(candidate.fell_side, "YES")
        self.assertEqual(candidate.high_mid, Decimal("0.96"))
        self.assertEqual(candidate.current_mid, Decimal("0.75"))
        self.assertEqual(candidate.drop, Decimal("0.21"))

    def test_drop_candidate_no(self) -> None:
        now = 1000.0
        window = F.MarketWindow(F.COIN_BY_SERIES["KXETH15M"], "T", now + 100, 100.0)
        window.observe(_book("0.04", "0.95"), now - 50)
        window.observe(_book("0.25", "0.74"), now)
        candidate = window.drop_candidate(now)
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.side, "YES")
        self.assertEqual(candidate.fell_side, "NO")
        self.assertEqual(candidate.high_mid, Decimal("0.955"))
        self.assertEqual(candidate.current_mid, Decimal("0.745"))
        self.assertEqual(candidate.drop, Decimal("0.210"))

    def test_drop_requires_last_three_minutes(self) -> None:
        now = 1000.0
        window = F.MarketWindow(F.COIN_BY_SERIES["KXBTC15M"], "T", now + 181, 100.0)
        window.observe(_book("0.95", "0.03"), now - 50)
        window.observe(_book("0.74", "0.24"), now)
        self.assertIsNone(window.drop_candidate(now))
        window.close_ts = now + 180
        self.assertIsNotNone(window.drop_candidate(now))

    def test_drop_requires_twenty_cents(self) -> None:
        now = 1000.0
        high = _book("0.95", "0.03")                 # mid .96
        exactly = _book("0.75", "0.23")              # mid .76, drop .20
        nineteen = _book("0.76", "0.22")             # mid .77, drop .19
        first = F.MarketWindow(F.COIN_BY_SERIES["KXBTC15M"], "T", now + 100, 100.0)
        first.observe(high, now - 50)
        first.observe(exactly, now)
        candidate = first.drop_candidate(now)
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.side, "NO")
        self.assertEqual(candidate.fell_side, "YES")
        self.assertEqual(candidate.drop, Decimal("0.20"))
        second = F.MarketWindow(F.COIN_BY_SERIES["KXBTC15M"], "T2", now + 100, 100.0)
        second.observe(high, now - 50)
        second.observe(nineteen, now)
        self.assertIsNone(second.drop_candidate(now))

    def test_drop_requires_recent_high_and_fresh_book(self) -> None:
        now = 1000.0
        stale_high = F.MarketWindow(F.COIN_BY_SERIES["KXBTC15M"], "T", now + 100, 100.0)
        stale_high.observe(_book("0.95", "0.03"), now - 61)
        stale_high.observe(_book("0.74", "0.24"), now)
        self.assertIsNone(stale_high.drop_candidate(now))

        stale_book = F.MarketWindow(F.COIN_BY_SERIES["KXBTC15M"], "T2", now + 100, 100.0)
        stale_book.observe(_book("0.95", "0.03"), now - 50)
        stale_book.observe(_book("0.74", "0.24"), now - 10)
        self.assertIsNone(stale_book.drop_candidate(now))

    def test_side_ask_and_depth(self) -> None:
        book = _book("0.74", "0.24", yes_sz="100", no_sz="50")
        ask, depth = F.side_ask(book, "YES")
        self.assertEqual((ask, depth), (Decimal("0.76"), Decimal("50")))
        ask, depth = F.side_ask(book, "NO")
        self.assertEqual((ask, depth), (Decimal("0.26"), Decimal("100")))

    def test_entry_reason_is_all_or_nothing(self) -> None:
        self.assertEqual(F.entry_reason(Decimal("0.26"), Decimal("10")), "")
        self.assertEqual(F.entry_reason(Decimal("0.26"), Decimal("9.99")), "depth_below_10")
        self.assertEqual(F.entry_reason(None, Decimal("10")), "no_ask")
        self.assertEqual(F.entry_reason(Decimal("1"), Decimal("10")), "ask_out_of_range")
        self.assertEqual(F.entry_reason(Decimal("0.50"), Decimal("10")),
                         "ask_at_or_above_50")
        self.assertEqual(F.entry_reason(Decimal("0.49"), Decimal("10")), "")
        self.assertEqual(F.entry_reason(Decimal("0.26"), None), "no_depth")
        self.assertEqual(F.entry_reason(Decimal("0.26"), Decimal("1"), 1), "")
        self.assertEqual(F.entry_reason(Decimal("0.26"), Decimal("0.99"), 1), "depth_below_1")

    def test_settlement_pnl(self) -> None:
        self.assertEqual(
            F.settlement_pnl("YES", "yes", 10, Decimal("0.76"), Decimal("0.13")),
            (Decimal("10"), Decimal("2.27"), Decimal("0.227")))
        self.assertEqual(
            F.settlement_pnl("YES", "no", 10, Decimal("0.76"), Decimal("0.13")),
            (Decimal("0"), Decimal("-7.73"), Decimal("-0.773")))
        self.assertEqual(
            F.settlement_pnl("NO", "no", 10, Decimal("0.76"), Decimal("0.13")),
            (Decimal("10"), Decimal("2.27"), Decimal("0.227")))
        self.assertEqual(
            F.settlement_pnl("YES", "yes", 0, Decimal("0.76"), Decimal("0.13")),
            (Decimal("0"), Decimal("0"), Decimal("0")))

    def test_percentile_nearest_rank(self) -> None:
        self.assertEqual(F.percentile([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 0.75), 7.0)
        self.assertEqual(F.percentile([i for i in range(30)], 0.75), 22.0)
        self.assertIsNone(F.percentile([], 0.75))

    def test_volume_tracker_and_p75(self) -> None:
        store = F.FlipStore(self.db_path)
        try:
            tracker = F.VolumeTracker(store)
            tracker.ensure(0.0)
            tracker.add("KXBTC15M", 1.0, 10.0)
            tracker.add("KXBTC15M", 61.0, 20.0)
            tracker.flush(120.0)
            rows = dict(store.db.execute(
                "SELECT bucket_ts,volume FROM volume_buckets WHERE series='KXBTC15M'"
            ).fetchall())
            self.assertEqual(rows, {0.0: 10.0, 60.0: 20.0})
            for i in range(30):
                store.record_bucket("TEST", i * 60.0, float(i))
            self.assertEqual(store.p75("TEST", 30 * 60.0), 22.0)
            self.assertIsNone(store.p75("NEW", 30 * 60.0))
        finally:
            store.close()

    def test_daily_report_and_gate(self) -> None:
        store = F.FlipStore(self.db_path)
        try:
            now = 1_765_000_000.0
            day = datetime.fromtimestamp(now, timezone.utc).date()
            coin = F.COIN_BY_SERIES["KXBTC15M"]
            for i in range(149):
                sid = store.log_signal(_signal(coin, now + i))
                store.settle(sid, "yes", Decimal("10"), Decimal("0.20"),
                             Decimal("0.02"), now + i)
            gate = store.gate()
            self.assertEqual(gate["fills"], 149)
            self.assertFalse(gate["eligible"])
            self.assertIn("149/150 paper fills", gate["reasons"])

            sid = store.log_signal(_signal(coin, now + 149))
            store.settle(sid, "yes", Decimal("10"), Decimal("0.20"), Decimal("0.02"),
                         now + 149)
            gate = store.gate()
            self.assertEqual(gate["fills"], 150)
            self.assertEqual(gate["contracts"], 1500)
            self.assertAlmostEqual(gate["ev_per_contract"], 0.02)
            self.assertAlmostEqual(gate["lower_95"], 0.02)
            self.assertTrue(gate["eligible"])

            report = store.daily(day)
            self.assertEqual(report["overall"]["signals"], 150)
            self.assertEqual(report["overall"]["fills"], 150)
            self.assertEqual(report["overall"]["contracts"], 1500)
            self.assertEqual(report["by_coin"]["BTC"]["signals"], 150)
            self.assertEqual(report["by_coin"]["ETH"]["signals"], 0)
            self.assertEqual(report["focus"]["ETH"]["signals"], 0)
            self.assertEqual(report["focus"]["ZEC"]["signals"], 0)
        finally:
            store.close()

    def test_daily_focus_eth_zec_and_no_fill(self) -> None:
        store = F.FlipStore(self.db_path)
        try:
            now = 1_765_000_000.0
            day = datetime.fromtimestamp(now, timezone.utc).date()
            eth = F.COIN_BY_SERIES["KXETH15M"]
            zec = F.COIN_BY_SERIES["KXZEC15M"]
            sid = store.log_signal(_signal(eth, now))
            store.settle(sid, "yes", Decimal("10"), Decimal("0.20"), Decimal("0.02"), now)
            sid = store.log_signal(_signal(zec, now + 1))
            store.settle(sid, "no", Decimal("0"), Decimal("-0.20"), Decimal("-0.02"), now + 1)
            store.log_signal(_signal(eth, now + 2, filled=False))
            report = store.daily(day)
            self.assertEqual(report["focus"]["ETH"]["signals"], 2)
            self.assertEqual(report["focus"]["ETH"]["fills"], 1)
            self.assertEqual(report["focus"]["ZEC"]["signals"], 1)
            self.assertAlmostEqual(report["focus"]["ETH"]["ev_per_contract"], 0.02)
            self.assertAlmostEqual(report["focus"]["ZEC"]["ev_per_contract"], -0.02)
            self.assertEqual(report["overall"]["no_fills"], 1)
            self.assertEqual(report["overall"]["signals"], 3)
            self.assertEqual(report["overall"]["fills"], 2)
        finally:
            store.close()

    def test_emit_writes_a_paper_fill_and_a_no_fill(self) -> None:
        store = F.FlipStore(self.db_path)
        try:
            now = 1_765_000_000.0
            runner = object.__new__(F.FlipRunner)
            runner.store = store
            runner.mode = "paper"
            runner.canary_done = False
            runner.broker = None

            window = F.MarketWindow(F.COINS[0], "T", now + 60, 100.0)
            window.book = _book("0.74", "0.24", yes_sz="100", no_sz="50")
            candidate = F.DropCandidate("NO", "YES", now - 20, Decimal("0.96"),
                                        Decimal("0.75"), Decimal("0.21"))
            asyncio.run(runner._emit(window, candidate, 10.0, 5.0, now))
            row = store.db.execute("SELECT * FROM signals").fetchone()
            self.assertEqual(row["filled"], 1)
            self.assertEqual(row["contracts"], 10)
            self.assertEqual(row["side"], "NO")
            self.assertEqual(row["fell_side"], "YES")
            self.assertEqual(row["ask"], "0.26")
            self.assertEqual(row["depth"], "100")
            self.assertEqual(row["fee"], "0.14")
            self.assertEqual(row["asset"], "BTC")
            self.assertTrue(window.signaled)
            self.assertEqual(len(store.pending()), 1)

            thin = F.MarketWindow(F.COINS[1], "T2", now + 60, 100.0)
            thin.book = _book("0.74", "0.24", yes_sz="5", no_sz="50")
            candidate = F.DropCandidate("NO", "YES", now - 20, Decimal("0.96"),
                                        Decimal("0.75"), Decimal("0.21"))
            asyncio.run(runner._emit(thin, candidate, 10.0, 5.0, now))
            row = store.db.execute(
                "SELECT * FROM signals WHERE ticker='T2'").fetchone()
            self.assertEqual(row["filled"], 0)
            self.assertEqual(row["contracts"], 0)
            self.assertEqual(row["no_fill_reason"], "depth_below_10")
            self.assertEqual(row["depth"], "5")
            self.assertTrue(thin.signaled)
        finally:
            store.close()

    def test_rejection_ledger_uses_drop_mid(self) -> None:
        store = F.FlipStore(self.db_path)
        try:
            now = 1000.0
            coin = F.COIN_BY_SERIES["KXBTC15M"]
            window = F.MarketWindow(coin, "T", now + 100, 100.0)
            candidate = F.DropCandidate("NO", "YES", now - 20, Decimal("0.96"),
                                        Decimal("0.75"), Decimal("0.21"))
            store.log_rejection(now, window, candidate, 5.0, 10.0, "volume_below_p75")
            row = store.db.execute("SELECT * FROM rejections").fetchone()
            self.assertEqual(row["drop_mid"], "0.21")
            self.assertEqual(row["reason"], "volume_below_p75")
            self.assertEqual(row["asset"], "BTC")
        finally:
            store.close()

    def test_set_fill_updates_a_pending_canary_row(self) -> None:
        store = F.FlipStore(self.db_path)
        try:
            now = 1_765_000_000.0
            sid = store.log_signal(_signal(F.COINS[0], now, filled=False))
            store.set_fill(sid, True, 10, Decimal("0.13"), Decimal("0.76"),
                           Decimal("100"), "", fee_source="kalshi_taker_formula")
            row = store.db.execute("SELECT * FROM signals WHERE id=?", (sid,)).fetchone()
            self.assertEqual(row["filled"], 1)
            self.assertEqual(row["contracts"], 10)
            self.assertEqual(row["fee"], "0.13")
            self.assertEqual(row["no_fill_reason"], "")
            self.assertEqual(len(store.pending()), 1)
        finally:
            store.close()

    def test_cheap_side_entry_is_pinned_below_fifty_cents(self) -> None:
        """The traded side is the one that rose, and its fill is always < 50c."""
        now = 1000.0
        window = F.MarketWindow(F.COIN_BY_SERIES["KXBTC15M"], "T", now + 100, 100.0)
        window.observe(_book("0.95", "0.03"), now - 50)
        window.observe(_book("0.74", "0.24"), now)
        candidate = window.drop_candidate(now)
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.side, "NO")
        self.assertEqual(candidate.fell_side, "YES")
        self.assertEqual(F.MAX_ENTRY, Decimal("0.50"))
        self.assertLess(Decimal(1) - candidate.current_mid, F.MAX_ENTRY)

        store = F.FlipStore(self.db_path)
        try:
            runner = object.__new__(F.FlipRunner)
            runner.store = store
            runner.mode = "paper"
            runner.canary_done = False
            runner.broker = None
            asyncio.run(runner._emit(window, candidate, 10.0, 5.0, now))
            row = store.db.execute("SELECT * FROM signals").fetchone()
            self.assertEqual(row["side"], "NO")
            self.assertEqual(row["fell_side"], "YES")
            self.assertLess(Decimal(row["ask"]), F.MAX_ENTRY)
            # Both mids stay in the ledger, and they are complements.
            self.assertEqual(Decimal(row["yes_mid"]) + Decimal(row["no_mid"]),
                             Decimal("1"))
        finally:
            store.close()

        # A favourite that only falls to 40c leaves the other side at 59c: that
        # is not the cheap side, so there is no candidate at all.
        late = F.MarketWindow(F.COIN_BY_SERIES["KXBTC15M"], "T2", now + 100, 100.0)
        late.observe(_book("0.95", "0.03"), now - 50)
        late.observe(_book("0.40", "0.58"), now)
        self.assertIsNone(late.drop_candidate(now))


if __name__ == "__main__":
    unittest.main()
