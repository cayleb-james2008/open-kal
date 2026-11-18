"""Maker-side tests: order groups, rewards, RFQ recording, demo probe, fav_maker.

Run: .venv/bin/python -m unittest tests/test_maker.py
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
import time
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openkal import arming  # noqa: E402
from openkal.broker import Broker  # noqa: E402
from openkal.config import Settings  # noqa: E402
from openkal.lanes import fav_maker as F  # noqa: E402
from openkal.maker import demo_probe as DP  # noqa: E402
from openkal.maker import order_groups as OG  # noqa: E402
from openkal.maker import rewards as R  # noqa: E402
from openkal.maker import rfq_recorder as RFQ  # noqa: E402

D = Decimal


class FakeGroupClient:
    def __init__(self) -> None:
        self.group_id = "g1"
        self.triggered = False
        self.calls: list[tuple] = []
        self.flip = True

    async def post(self, path, body):
        self.calls.append(("POST", path, body))
        return 201, {"order_group_id": self.group_id}

    async def request(self, method, path, body=None, write=False):
        self.calls.append((method, path, body))
        if path.endswith("/trigger"):
            if self.flip:
                self.triggered = True
            return 200, {}
        if path.endswith("/reset"):
            if self.flip:
                self.triggered = False
            return 200, {}
        return 200, {}

    async def delete(self, path):
        self.calls.append(("DELETE", path))
        return 200, {}


async def group_getter(client, path, params):
    return 200, {
        "contracts_limit_fp": "20.00",
        "exchange_index": 0,
        "is_auto_cancel_enabled": client.triggered,
        "orders": [],
    }


class OrderGroupTests(unittest.TestCase):
    def test_cycle_verifies(self) -> None:
        client = FakeGroupClient()
        switch = OG.OrderGroupKillSwitch(
            client, 20, getter=group_getter, status_timeout=0.2)
        ok, problems, detail = asyncio.run(switch.verify_cycle())
        self.assertTrue(ok, problems)
        self.assertEqual(detail["create_state"], "active")
        self.assertEqual(detail["after_trigger"], "triggered")
        self.assertEqual(detail["after_reset"], "active")
        self.assertTrue(any(call[1].endswith("/trigger") for call in client.calls))
        self.assertTrue(any(call[1].endswith("/reset") for call in client.calls))
        self.assertTrue(any(call[1].startswith("/portfolio/order_groups/")
                            and call[0] == "DELETE" for call in client.calls))

    def test_cycle_fails_if_trigger_never_lands(self) -> None:
        client = FakeGroupClient()
        client.flip = False
        switch = OG.OrderGroupKillSwitch(
            client, 20, getter=group_getter, status_timeout=0.05)
        ok, problems, _ = asyncio.run(switch.verify_cycle())
        self.assertFalse(ok)
        self.assertIn("trigger did not set is_auto_cancel_enabled", problems)

    def test_limit_bounds(self) -> None:
        with self.assertRaises(ValueError):
            OG.OrderGroupKillSwitch(FakeGroupClient(), 0)
        with self.assertRaises(ValueError):
            OG.OrderGroupKillSwitch(FakeGroupClient(), 1_000_001)

    def test_status_state(self) -> None:
        status = OG.GroupStatus("g", "20.00", False)
        self.assertEqual(status.state, "active")
        self.assertEqual(OG.GroupStatus("g", "20.00", True).state, "triggered")


def program(**overrides) -> R.Program:
    values = dict(id="p1", ticker="KXTEST-26NOV20-A", series="KXTEST",
                  reward_usd=D("100"), start_ts=0.0, end_ts=86400.0,
                  target=D("1000"), discount=D("0.5"))
    values.update(overrides)
    return R.Program(**values)


def depth(yes: dict[str, str], no: dict[str, str]) -> R.Depth:
    return R.Depth(
        yes={D(p): D(s) for p, s in yes.items()},
        no={D(p): D(s) for p, s in no.items()},
    )


class RewardModelTests(unittest.TestCase):
    def test_program_parses_centi_cents(self) -> None:
        row = {"market_ticker": "T", "start_date": "2026-11-18T00:00:00Z",
               "end_date": "2026-11-19T00:00:00Z", "period_reward": 950000,
               "target_size_fp": "1000.00", "discount_factor_bps": 5000}
        p = R.Program.from_row(row)
        self.assertIsNotNone(p)
        self.assertEqual(p.reward_usd, D("95"))
        self.assertEqual(p.usd_per_day, D("95"))
        self.assertEqual(p.discount, D("0.5"))

    def test_rank_filters_expired_and_out_of_bounds(self) -> None:
        now = 1000.0
        rows = [
            program(id="a", end_ts=2000, target=D("1000")),
            program(id="b", end_ts=500),
            program(id="c", target=D("50")),
            program(id="d", target=D("30000")),
        ]
        ranked = R.rank(rows, now)
        self.assertEqual([p.id for p in ranked], ["a"])

    def test_reference_price_is_one_fifth(self) -> None:
        levels = {D("0.99"): D("100"), D("0.98"): D("100"),
                  D("0.97"): D("100"), D("0.96"): D("100")}
        self.assertEqual(R.reference_price(levels, D("1000")), D("0.98"))

    def test_score_discounts_each_tick(self) -> None:
        ref = D("0.95")
        self.assertEqual(R.score_at(D("0.95"), D("10"), ref, D("0.5")), D("10"))
        self.assertEqual(R.score_at(D("0.94"), D("10"), ref, D("0.5")), D("5"))
        self.assertEqual(R.score_at(D("0.93"), D("10"), ref, D("0.5")), D("2.5"))

    def test_reward_favourite_allows_zero_yes_bid(self) -> None:
        side, bid, ask = R.favourite_quote(D("0"), D("0.01"), D("0.99"), D("1.00"))
        self.assertEqual((side, bid, ask), ("NO", D("0.99"), D("1.00")))

    def test_estimate_requires_both_sides_at_target(self) -> None:
        book = depth({"0.50": "600", "0.49": "600"}, {"0.50": "500"})
        est = R.estimate(program(), book, "NO", D("0.48"), D("10"))
        self.assertFalse(est.target_ok)
        self.assertEqual(est.accrual_usd, D("0"))
        self.assertEqual(est.reason, "target_size_not_met")

    def test_estimate_scores_our_order_into_the_book(self) -> None:
        book = depth({"0.50": "600", "0.49": "600"},
                     {"0.50": "600", "0.49": "600"})
        est = R.estimate(program(), book, "NO", D("0.48"), D("10"))
        self.assertTrue(est.target_ok)
        self.assertEqual(est.reference, D("0.50"))
        self.assertEqual(est.our_score, D("2.5"))
        self.assertEqual(est.total_score, D("902.5"))
        self.assertGreater(est.share, D("0"))
        self.assertGreater(est.accrual_usd, D("0"))
        self.assertEqual(est.method, R.METHOD)


class FavRuleTests(unittest.TestCase):
    @staticmethod
    def snap(yes_bid: str, yes_ask: str, no_bid: str, no_ask: str) -> F.Snap:
        return F.Snap("T", "S", "binary", "active", D(yes_bid), D(yes_ask),
                      D(no_bid), D(no_ask), time.time() + 3600, D("1"))

    def test_gas_zero_yes_bid_still_has_a_no_favourite(self) -> None:
        s = self.snap("0", "0.01", "0.99", "1.00")
        self.assertEqual(F.favourite(s), ("NO", D("0.99"), D("1.00")))

    def test_price_band_and_one_tick_improvement(self) -> None:
        self.assertIsNone(F.quote_price(D("0.84"), D("0.90")))
        self.assertEqual(F.quote_price(D("0.85"), D("0.90")), D("0.86"))
        self.assertEqual(F.quote_price(D("0.97"), D("0.98")), D("0.97"))
        self.assertEqual(F.quote_price(D("0.99"), D("1.00")), D("0.99"))
        self.assertIsNone(F.quote_price(D("0.99"), D("0.99")))

    def test_terminal_horizon_is_seven_days(self) -> None:
        now = time.time()
        near = F.Snap("T", "S", "binary", "active", D("0.95"), D("0.96"),
                      D("0.04"), D("0.05"), now + 6 * 86400, D("1"))
        far = F.Snap("T", "S", "binary", "active", D("0.95"), D("0.96"),
                     D("0.04"), D("0.05"), now + 8 * 86400, D("1"))
        unknown = F.Snap("T", "S", "binary", "active", D("0.95"), D("0.96"),
                         D("0.04"), D("0.05"), 0.0, D("1"))
        self.assertTrue(F.eligible(near, now))
        self.assertFalse(F.eligible(far, now))
        self.assertFalse(F.eligible(unknown, now))

    def test_queue_cap_steps_inside_or_skips(self) -> None:
        self.assertEqual(F.queue_price(D("0.95"), D("0.97"), D("0.95"), D("500")),
                         D("0.95"))
        self.assertEqual(F.queue_price(D("0.95"), D("0.97"), D("0.95"), D("5000")),
                         D("0.96"))
        # A one-tick spread has nowhere to step inside: no quote at all.
        self.assertIsNone(F.queue_price(D("0.95"), D("0.96"), D("0.95"), D("5000")))
        self.assertIsNone(F.queue_price(D("0.99"), D("1.00"), D("0.99"), D("5000")))

    def test_size_range_and_equity_cap(self) -> None:
        self.assertEqual(F.size_for(D("0.99"), D("100"), D("1000")), 6)
        self.assertEqual(F.size_for(D("0.99"), D("0"), D("1000")), 5)
        self.assertEqual(F.size_for(D("0.99"), D("100"), D("10")), 0)
        self.assertEqual(F.size_for(D("0.99"), D("0"), D("0")), 0)

    def test_paper_equity_falls_back_when_env_is_zero(self) -> None:
        self.assertEqual(F.paper_equity(Settings(paper_equity=D("0"))), F.DEFAULT_PAPER_EQUITY)
        self.assertEqual(F.paper_equity(Settings(paper_equity=D("500"))), D("500"))


class FakeBookClient:
    def __init__(self, result: str = "no") -> None:
        self.result = result
        self.settings = Settings()
        self.requests: list[str] = []

    async def get(self, path, params=None):
        self.requests.append(path)
        if path.endswith("/orderbook"):
            return 200, {"orderbook_fp": {
                "yes_dollars": [["0.9500", "100.00"]],
                "no_dollars": [["0.0400", "100.00"]],
            }}
        if path.startswith("/markets/"):
            return 200, {"market": {"result": self.result}}
        return 404, {}


class PaperFillTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = F.FavStore(Path(self.tmp.name) / "test.db")
        self.client = FakeBookClient()
        self.book = F.PaperBook(self.client, self.store, [], D("1000"))
        self.book.books["T"] = R.Depth(
            yes={D("0.95"): D("100")}, no={D("0.04"): D("100")})

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def quote(self, side: str = "YES", price: str = "0.95",
              queue: str = "100") -> F.Quote:
        snap = F.Snap("T", "S", "binary", "active", D("0.95"), D("0.96"),
                      D("0.04"), D("0.05"), time.time() + 3600, D("1"))
        return F.Quote(snap, side, D(price), 10, D(price), D("0.96"),
                       D(queue), "gas")

    def test_yes_bid_needs_a_seller_and_queue_beyond_the_touch(self) -> None:
        self.book.add(self.quote(), now=time.time())
        self.book.on_trade("T", D("0.95"), D("50"), "no")
        self.assertEqual(len(self.book.fills), 0)
        self.book.on_trade("T", D("0.95"), D("150"), "no")
        self.assertEqual(len(self.book.fills), 1)
        self.assertEqual(self.book.fills[0].contracts, 10)

    def test_wrong_taker_side_does_not_fill(self) -> None:
        self.book.add(self.quote(), now=time.time())
        self.book.on_trade("T", D("0.95"), D("1000"), "yes")
        self.assertEqual(len(self.book.fills), 0)

    def test_sweep_through_the_touch_ignores_the_queue(self) -> None:
        self.book.add(self.quote(queue="10000"), now=time.time())
        self.book.on_trade("T", D("0.94"), D("1"), "no")
        self.assertEqual(len(self.book.fills), 1)
        self.assertEqual(self.book.fills[0].contracts, 1)

    def test_no_bid_fills_on_a_yes_buyer(self) -> None:
        self.book.add(self.quote(side="NO", price="0.97", queue="0"), now=time.time())
        self.book.on_trade("T", D("0.03"), D("5"), "yes")
        self.assertEqual(len(self.book.fills), 1)
        self.assertEqual(self.book.fills[0].quote.side, "NO")

    def test_group_limit_halts_and_cancels(self) -> None:
        self.book.group_limit = 2
        self.book.add(self.quote(queue="0"), now=time.time())
        self.book.add(self.quote(queue="0", price="0.94"), now=time.time())
        self.book.on_trade("T", D("0.95"), D("10"), "no")
        self.assertIsNotNone(self.book.halted)
        self.assertEqual(sum(len(v) for v in self.book.quotes.values()), 0)
        self.assertEqual(self.store.summary()["halts"], 1)

    def test_markout_and_settlement(self) -> None:
        self.book.add(self.quote(side="NO", price="0.97", queue="0"), now=time.time())
        self.book.on_trade("T", D("0.03"), D("10"), "yes")
        fill = self.book.fills[0]
        asyncio.run(self.book.process_marks(time.time() + 3600))
        self.assertEqual(self.book.fills[0].db_id, fill.db_id)
        self.assertGreaterEqual(self.store.summary()["fills"], 1)
        asyncio.run(self.book.settle_positions(time.time()))
        self.assertEqual(self.store.summary()["settlements"], 1)
        self.assertEqual(self.store.summary()["wins"], 1)
        self.assertEqual(D(self.store.summary()["realized_pnl_usd"]), D("0.30"))


class BinaryClient:
    def __init__(self, code: int, body: dict) -> None:
        self.code = code
        self.body = body
        self.requests: list[tuple] = []

    async def post(self, path, body):
        self.requests.append((path, body))
        if path == "/portfolio/events/orders":
            return 201, {"order": {"order_id": "o1", "status": "resting"}}
        return self.code, self.body

    async def get(self, path, params=None):
        return self.code, self.body

    async def close(self):
        return None


class BrokerTests(unittest.TestCase):
    def test_no_quote_is_wired_as_yes_ask_with_group(self) -> None:
        client = BinaryClient(200, {})
        asyncio.run(Broker(client).place_fav_quote("T", "NO", D("0.97"), 5, "g1"))
        body = client.requests[0][1]
        self.assertEqual(body["side"], "ask")
        self.assertEqual(body["price"], "0.0300")
        self.assertEqual(body["order_group_id"], "g1")
        self.assertTrue(body["post_only"])
        self.assertEqual(body["time_in_force"], "good_till_canceled")

    def test_yes_quote_is_a_plain_bid(self) -> None:
        client = BinaryClient(200, {})
        asyncio.run(Broker(client).place_fav_quote("T", "YES", D("0.95"), 5, "g1"))
        body = client.requests[0][1]
        self.assertEqual(body["side"], "bid")
        self.assertEqual(body["price"], "0.9500")

    def test_no_entry_is_wired_as_yes_ask_at_one_minus(self) -> None:
        client = BinaryClient(200, {})
        order = asyncio.run(Broker(client).place_no_entry("T", D("0.24"), 10))
        body = client.requests[0][1]
        self.assertEqual(body["side"], "ask")
        self.assertEqual(body["price"], "0.7600")
        self.assertEqual(body["count"], "10")
        self.assertFalse(body["post_only"])
        self.assertEqual(body["time_in_force"], "immediate_or_cancel")
        self.assertIsNotNone(order)
        assert order is not None
        self.assertEqual(order.side, "NO")
        self.assertEqual(order.price, D("0.24"))

    def test_no_entry_maker_keeps_the_conversion(self) -> None:
        client = BinaryClient(200, {})
        asyncio.run(Broker(client).place_no_entry(
            "T", D("0.97"), 5, tif="good_till_canceled", post_only=True))
        body = client.requests[0][1]
        self.assertEqual(body["side"], "ask")
        self.assertEqual(body["price"], "0.0300")
        self.assertTrue(body["post_only"])
        self.assertEqual(body["time_in_force"], "good_till_canceled")

    def test_group_is_required(self) -> None:
        client = BinaryClient(200, {})
        self.assertIsNone(asyncio.run(
            Broker(client).place_fav_quote("T", "NO", D("0.97"), 5, "")))
        self.assertEqual(client.requests, [])


def rfq_event(kind: str, **body) -> dict:
    return {"type": kind, "seq": 1, "msg": body}


class RFQRecorderTests(unittest.TestCase):
    def test_lifecycle_and_settlement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = RFQ.RFQRecorder(
                Settings(), Path(tmp) / "rfq.db", client=BinaryClient(200, {}))
            rfq_id = "r1"
            recorder.ingest(rfq_event(
                "rfq_created", id=rfq_id, market_ticker="KXMVE-X",
                creator_id="c", target_cost_dollars="100.0000",
                contracts_fp="10.00", created_ts="2026-11-18T00:00:00Z",
                mve_collection_ticker="KXMVE", mve_selected_legs=[
                    {"market_ticker": "A", "side": "yes"},
                    {"market_ticker": "B", "side": "no"}]))
            recorder.ingest(rfq_event(
                "quote_created", quote_id="q1", rfq_id=rfq_id,
                market_ticker="KXMVE-X", yes_bid_dollars="0.0100",
                no_bid_dollars="0.9900", yes_contracts_fp="10.00",
                created_ts="2026-11-18T00:00:01Z"))
            recorder.ingest(rfq_event(
                "quote_accepted", quote_id="q1", rfq_id=rfq_id,
                accepted_side="no", accepted_ts="2026-11-18T00:00:02Z"))
            recorder.ingest(rfq_event(
                "quote_executed", quote_id="q1", rfq_id=rfq_id,
                market_ticker="KXMVE-X", creator_order_id="o1",
                executed_ts="2026-11-18T00:00:03Z"))
            recorder.record_settlement("KXMVE-X", "no")
            recorder.flush()
            summary = recorder.summary()
            self.assertEqual(summary["rfqs"], 1)
            self.assertEqual(summary["combo_rfqs"], 1)
            self.assertEqual(summary["quotes"], 1)
            self.assertEqual(summary["executed_quotes"], 1)
            self.assertEqual(summary["executions"], 1)
            self.assertEqual(summary["settlements"], 1)
            row = recorder.db.execute(
                "SELECT status,batch_id FROM rfqs WHERE id=?", (rfq_id,)).fetchone()
            self.assertEqual(row[0], "open")
            legs = recorder.db.execute(
                "SELECT legs FROM rfqs WHERE id=?", (rfq_id,)).fetchone()[0]
            self.assertEqual(len(RFQ.unpack(legs)), 2)
            quote = recorder.db.execute(
                "SELECT status,accepted_side,creator_order_id FROM quotes WHERE id=?",
                ("q1",)).fetchone()
            self.assertEqual(quote, ("executed", "no", "o1"))
            recorder.ingest(rfq_event("rfq_deleted", id=rfq_id,
                                      deleted_ts="2026-11-18T00:00:04Z"))
            recorder.flush()
            self.assertEqual(recorder.db.execute(
                "SELECT status FROM rfqs WHERE id=?", (rfq_id,)).fetchone()[0],
                "deleted")
            self.assertEqual(recorder.summary()["events"], 5)
            recorder.close()

    def test_market_prefix_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = RFQ.RFQRecorder(
                Settings(), Path(tmp) / "rfq.db", client=BinaryClient(200, {}),
                market_prefix="KXMVECROSS")
            recorder.ingest(rfq_event("rfq_created", id="r1", market_ticker="OTHER"))
            recorder.flush()
            self.assertEqual(recorder.summary()["events"], 0)
            self.assertEqual(recorder.summary()["skipped"], 1)
            recorder.close()


class DemoProbeTests(unittest.TestCase):
    def test_auth_rejection_is_reported_with_the_exact_body(self) -> None:
        client = BinaryClient(401, {"error": {
            "code": "authentication_error",
            "message": "We could not authenticate your request",
            "details": "NOT_FOUND"}})
        result = asyncio.run(DP.probe(Settings(), client=client, save=False))
        self.assertFalse(result["can_post"])
        self.assertEqual(result["reason"], "demo_auth_rejected")
        self.assertEqual(result["responses"][1]["status"], 401)
        self.assertEqual(result["responses"][1]["body"]["error"]["details"], "NOT_FOUND")

    def test_no_open_combo_is_not_a_rejection(self) -> None:
        client = BinaryClient(200, {"rfqs": []})
        result = asyncio.run(DP.probe(Settings(), client=client, save=False))
        self.assertIsNone(result["can_post"])
        self.assertEqual(result["reason"], "no_open_combo_rfq")

    def test_quote_is_posted_and_cancelled(self) -> None:
        class Demo:
            def __init__(self) -> None:
                self.calls = []

            async def get(self, path, params=None):
                if path == "/exchange/status":
                    return 200, {"trading_active": True}
                return 200, {"rfqs": [{
                    "id": "r1", "status": "open", "market_ticker": "KXMVE-X",
                    "mve_collection_ticker": "KXMVE"}]}

            async def post(self, path, body):
                self.calls.append(("POST", path, body))
                return 201, {"id": "q1"}

            async def delete(self, path, params=None):
                self.calls.append(("DELETE", path))
                return 200, {}

            async def close(self):
                return None

        result = asyncio.run(DP.probe(Settings(), client=Demo(), save=False))
        self.assertTrue(result["can_post"])
        self.assertEqual(result["quote_id"], "q1")
        self.assertEqual(result["cancel_status"], 200)


class ArmingTests(unittest.TestCase):
    def test_live_requires_a_fresh_lane_arm(self) -> None:
        settings = Settings(env_name="prod", allow_live=True)
        record = {"ts": time.time(), "by": "tester", "lane": F.LANE}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "live-arm-fav-maker.json"
            with patch.object(F.arming, "arm_path", return_value=path), \
                    patch.object(F.arming, "consume", return_value=record):
                self.assertEqual(F.require_operator_arm(settings)["by"], "tester")
            with patch.object(F.arming, "arm_path", return_value=path), \
                    patch.object(F.arming, "consume", return_value={
                        "ts": time.time(), "by": "tester", "lane": "other"}):
                with self.assertRaises(RuntimeError):
                    F.require_operator_arm(settings)
            with self.assertRaises(RuntimeError):
                F.require_operator_arm(Settings(env_name="prod", allow_live=False))

    def test_lane_record_is_not_crypto_permission(self) -> None:
        record = {"ts": time.time(), "by": "tester", "lane": F.LANE}
        self.assertFalse(arming.armed(record, started=0.0, now=time.time()))
        self.assertTrue(arming.fresh_operator_act(record, now=time.time(), lane=F.LANE))


if __name__ == "__main__":
    unittest.main()
