"""The frozen AAA gas/diesel far-strike NO rule and its operator lock.

Run: .venv/bin/python -m unittest tests/test_gas_no.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openkal import arming  # noqa: E402
from openkal.broker import Broker, VenueOrder  # noqa: E402
from openkal.fees import taker_fee  # noqa: E402
from openkal.lanes import gas_no as G  # noqa: E402

D = Decimal
NOW = datetime(2026, 11, 18, 15, 0, tzinfo=G.ET).timestamp()


def market(no_price: str, depth: str = "100", *, ticker: str = "KXDIESELD-26NOV19-T5.935",
           event: str = "KXDIESELD-26NOV19", series: str = "KXDIESELD",
           strike: str = "5.935", open_ts: float = 1000.0) -> G.Market:
    """A market whose best YES bid implies the requested NO ask."""
    return G.Market(ticker=ticker, event=event, series=series, strike=D(strike),
                    open_ts=open_ts, yes_bid=D(1) - D(no_price),
                    yes_bid_size=D(depth))


def candidate(strike: str, *, price: str = "0.99", depth: str = "100",
              event: str = "KXDIESELD-26NOV19", series: str = "KXDIESELD",
              ticker: str | None = None, reference: str = "5.0") -> G.Candidate:
    row = market(price, depth, ticker=ticker or f"{event}-{strike}", event=event,
                 series=series, strike=strike)
    found = G.candidate_for(row, D(reference))
    assert found is not None, (strike, price, depth)
    return found


class RuleEdges(unittest.TestCase):
    def test_price_edges(self) -> None:
        for price, expected in (("0.94", False), ("0.95", True),
                                ("0.99", True), ("1.00", False)):
            with self.subTest(price=price):
                found = G.candidate_for(market(price))
                self.assertEqual(found is not None, expected)

    def test_depth_edges(self) -> None:
        self.assertIsNone(G.candidate_for(market("0.95", "9.99")))
        self.assertIsNotNone(G.candidate_for(market("0.95", "10")))

    def test_decision_is_six_hours_after_open(self) -> None:
        row = market("0.95", open_ts=1000.0)
        self.assertEqual(row.decision_ts, 1000.0 + 6 * 3600)
        self.assertFalse(G.actionable(row, row.decision_ts - 1))
        self.assertTrue(G.actionable(row, row.decision_ts))
        self.assertTrue(G.actionable(row, row.decision_ts + G.DECISION_GRACE_S))
        self.assertFalse(G.actionable(row, row.decision_ts + G.DECISION_GRACE_S + 1))

    def test_no_price_is_one_minus_yes_bid(self) -> None:
        row = market("0.97")
        self.assertEqual(row.yes_bid, D("0.03"))
        self.assertEqual(row.no_price, D("0.97"))

    def test_event_date_parses(self) -> None:
        self.assertEqual(G.parse_event_date("KXAAAGASD-26NOV18"), date(2026, 11, 18))
        self.assertIsNone(G.parse_event_date("not-a-date"))


class Caps(unittest.TestCase):
    def test_two_farthest_per_event(self) -> None:
        rows = [candidate("5.1"), candidate("5.2"), candidate("5.3")]
        orders = G.plan_orders(rows, [])
        self.assertEqual([o.strike for o in orders], [D("5.3"), D("5.2")])

    def test_existing_position_consumes_event_room(self) -> None:
        rows = [candidate("5.1"), candidate("5.2")]
        open_rows = [{"ticker": "other", "event": "KXDIESELD-26NOV19", "open": True}]
        orders = G.plan_orders(rows, open_rows)
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].strike, D("5.2"))

    def test_two_open_in_event_blocks_it(self) -> None:
        rows = [candidate("5.1")]
        open_rows = [{"ticker": "a", "event": "KXDIESELD-26NOV19", "open": True},
                     {"ticker": "b", "event": "KXDIESELD-26NOV19", "open": True}]
        self.assertEqual(G.plan_orders(rows, open_rows), [])

    def test_lane_wide_cap_of_four(self) -> None:
        rows = [candidate("1.0", event="E1", reference="0"),
                candidate("2.0", event="E2", reference="0"),
                candidate("3.0", event="E3", reference="0"),
                candidate("4.0", event="E4", reference="0")]
        orders = G.plan_orders(rows, [])
        self.assertEqual(len(orders), 4)
        # Two already open leaves room for two, newest events processed in order.
        open_rows = [{"ticker": "x", "event": "E0", "open": True},
                     {"ticker": "y", "event": "E0", "open": True}]
        self.assertEqual(len(G.plan_orders(rows, open_rows)), 2)

    def test_held_ticker_is_not_doubled(self) -> None:
        row = candidate("5.2", ticker="HELD")
        open_rows = [{"ticker": "HELD", "event": "KXDIESELD-26NOV19", "open": True}]
        self.assertEqual(G.plan_orders([row], open_rows), [])

    def test_settled_positions_do_not_count(self) -> None:
        rows = [candidate("5.2")]
        closed = [{"ticker": "old", "event": "KXDIESELD-26NOV19", "open": False}]
        self.assertEqual(len(G.plan_orders(rows, closed)), 1)


class OrderPricing(unittest.TestCase):
    def test_limit_is_the_seen_price_with_ioc(self) -> None:
        order = G.plan_orders([candidate("5.935", price="0.99")], [])[0]
        self.assertEqual(order.side, "NO")
        self.assertEqual(order.price, D("0.99"))
        self.assertEqual(order.contracts, 10)
        self.assertEqual(order.tif, "immediate_or_cancel")
        self.assertEqual(order.depth, D("100"))

    def test_canary_uses_one_contract(self) -> None:
        order = G.plan_orders([candidate("5.935", price="0.99")], [], size=1)[0]
        self.assertEqual(order.contracts, 1)

    def test_premium_and_fee(self) -> None:
        order = G.plan_orders([candidate("5.935", price="0.99")], [])[0]
        self.assertEqual(order.premium, D("9.90"))
        self.assertEqual(order.fee, D("0.01"))

    def test_wire_order_is_yes_denominated(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.body = None

            async def post(self, path, body):
                self.body = body
                return 201, {"order": {"order_id": "o1", "fill_count": "0.00"}}

        client = FakeClient()
        venue = asyncio.run(Broker(client).place_no_entry("T", D("0.95"), 10))
        self.assertEqual(client.body["side"], "ask")
        self.assertEqual(client.body["price"], "0.0500")
        self.assertEqual(client.body["count"], "10")
        self.assertEqual(client.body["time_in_force"], "immediate_or_cancel")
        self.assertFalse(client.body["post_only"])
        self.assertEqual(venue.price, D("0.95"))
        self.assertEqual(venue.side, "NO")

    def test_fill_summary_uses_no_prices(self) -> None:
        fills = [{"count_fp": "6.00", "no_price_dollars": "0.9900", "fee_cost": "0.006",
                  "yes_price_dollars": "0.0100"},
                 {"count_fp": "4.00", "no_price_dollars": "0.9800", "fee_cost": "0.004",
                  "yes_price_dollars": "0.0200"}]
        qty, price, fee = G.summarize_fills(fills, side="NO")
        self.assertEqual(qty, 10)
        self.assertEqual(price, D("0.9860"))
        self.assertEqual(fee, D("0.010"))


class Settlement(unittest.TestCase):
    def test_win_and_loss(self) -> None:
        fee = taker_fee(D("0.95"), 10)
        self.assertEqual(fee, D("0.04"))
        self.assertEqual(G.settlement_pnl(D("0.95"), 10, fee, won=True), D("0.46"))
        self.assertEqual(G.settlement_pnl(D("0.95"), 10, fee, won=False), D("-9.54"))

    def test_far_strike_win_is_small(self) -> None:
        fee = taker_fee(D("0.99"), 10)
        self.assertEqual(fee, D("0.01"))
        self.assertEqual(G.settlement_pnl(D("0.99"), 10, fee, won=True), D("0.09"))

    def test_venue_result_wins_over_the_value(self) -> None:
        self.assertTrue(G.resolve_no(result="no", value="5.99", strike=D("5.935")))
        self.assertFalse(G.resolve_no(result="yes", value="1.00", strike=D("5.935")))
        self.assertTrue(G.resolve_no(result="", value="5.90", strike=D("5.935")))
        self.assertFalse(G.resolve_no(result="", value="6.00", strike=D("5.935")))
        self.assertIsNone(G.resolve_no(result="", value="", strike=D("5.935")))


class Pauses(unittest.TestCase):
    def test_two_losses_today(self) -> None:
        rows = [{"open": False, "pnl": "-1.00", "settled_ts": NOW - 60},
                {"open": False, "pnl": "-2.00", "settled_ts": NOW - 30}]
        self.assertEqual(G.losses_today(rows, NOW), 2)
        state = {"equity_start": "100", "peak": "100", "positions": rows}
        self.assertTrue(any("2 losses" in r for r in G.lane_pause_reasons(state, NOW)))

    def test_one_loss_and_yesterdays_losses_do_not_pause(self) -> None:
        rows = [{"open": False, "pnl": "-1.00", "settled_ts": NOW - 60},
                {"open": False, "pnl": "-5.00", "settled_ts": NOW - 86400}]
        self.assertEqual(G.losses_today(rows, NOW), 1)
        state = {"equity_start": "100", "peak": "100", "positions": rows}
        self.assertEqual(G.lane_pause_reasons(state, NOW), [])

    def test_twenty_percent_drawdown(self) -> None:
        self.assertTrue(G.drawdown_tripped(D("80.00"), D("100.00")))
        self.assertFalse(G.drawdown_tripped(D("80.01"), D("100.00")))
        rows = [{"open": False, "pnl": "-20.01", "settled_ts": NOW - 60}]
        state = {"equity_start": "100", "peak": "100", "positions": rows}
        self.assertTrue(any("drawdown" in r for r in G.lane_pause_reasons(state, NOW)))

    def test_series_thresholds(self) -> None:
        gas = G.DailyChange("KXAAAGASD", D("3.7502"), D("3.7818"),
                            D("-0.0316"), "2026-11-18")
        self.assertIn("2c", G.series_pause_reason(gas, "KXAAAGASD"))
        calm = G.DailyChange("KXAAAGASD", D("3.7500"), D("3.7301"),
                             D("0.0199"), "2026-11-18")
        self.assertIsNone(G.series_pause_reason(calm, "KXAAAGASD"))
        diesel = G.DailyChange("KXDIESELD", D("5.9594"), D("5.9848"),
                               D("-0.0254"), "2026-11-18")
        self.assertIsNone(G.series_pause_reason(diesel, "KXDIESELD"))
        hot = G.DailyChange("KXDIESELD", D("5.9594"), D("5.9294"),
                            D("0.0300"), "2026-11-18")
        self.assertIn("3c", G.series_pause_reason(hot, "KXDIESELD"))
        state = G.DailyChange("KXAAAGASDCA", D("4.50"), D("4.47"),
                              D("0.03"), "2026-11-18")
        self.assertIn("3c", G.series_pause_reason(state, "KXAAAGASDCA"))

    def test_unknown_change_pauses_the_series(self) -> None:
        self.assertIn("unavailable", G.series_pause_reason(None, "KXAAAGASD"))

    def test_pause_latches_and_only_an_operator_clears_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = G.LaneLedger(Path(tmp) / "state.json", Path(tmp) / "log.jsonl")
            first = ledger.latch_pause("2 losses today", NOW)
            again = ledger.latch_pause("a different reason", NOW)
            self.assertEqual(first["reason"], again["reason"])
            self.assertTrue(ledger.pause())
            self.assertTrue(ledger.resume("cayleb", NOW))
            self.assertIsNone(ledger.pause())
            with self.assertRaises(ValueError):
                ledger.resume("   ", NOW)
            kinds = [json.loads(line)["kind"] for line in
                     (Path(tmp) / "log.jsonl").read_text().splitlines()]
            self.assertIn("pause", kinds)
            self.assertIn("resume", kinds)

    def test_daily_series_pause_expires_the_next_day(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = G.LaneLedger(Path(tmp) / "state.json", Path(tmp) / "log.jsonl")
            ledger.pause_series("KXAAAGASD", "daily change 3.16c >= 2c", NOW)
            self.assertIsNotNone(ledger.series_pause("KXAAAGASD", NOW))
            self.assertIsNone(ledger.series_pause("KXAAAGASD", NOW + 86400))


class LaneArming(unittest.TestCase):
    def test_lane_record_has_its_own_path(self) -> None:
        gas = arming.arm_path("gas-no")
        self.assertNotEqual(gas, arming.ARM_PATH)
        self.assertEqual(gas.name, "live-arm-gas-no.json")
        self.assertEqual(arming.arm_path("crypto"), arming.ARM_PATH)
        self.assertEqual(arming.arm_path(None), arming.ARM_PATH)

    def test_lane_record_does_not_arm_crypto(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gas.json"
            record = arming.operator_arm("cayleb", path=path, lane="gas-no",
                                         now=1000.0)
            self.assertEqual(record["lane"], "gas-no")
            self.assertTrue(arming.armed(record, started=1000.0, now=1000.0,
                                         lane="gas-no"))
            self.assertFalse(arming.armed(record, started=1000.0, now=1000.0,
                                          lane="crypto"))
            self.assertFalse(arming.armed(record, started=1000.0, now=1000.0))

    def test_lane_crypto_is_the_legacy_untagged_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            record = arming.operator_arm("cayleb", path=Path(tmp) / "crypto.json",
                                         lane="crypto", now=1000.0)
            self.assertNotIn("lane", record)
            self.assertTrue(arming.armed(record, started=1000.0, now=1000.0))
            self.assertTrue(arming.armed(record, started=1000.0, now=1000.0,
                                         lane="crypto"))

    def test_one_shot_check_is_fresh_named_and_lane_scoped(self) -> None:
        record = {"ts": 1000.0, "by": "cayleb", "lane": "gas-no"}
        self.assertTrue(arming.fresh_operator_act(record, now=1000.0, lane="gas-no"))
        self.assertFalse(arming.fresh_operator_act(record, now=1000.0, lane="crypto"))
        self.assertFalse(arming.fresh_operator_act({"ts": 1000.0, "by": " "},
                                                   now=1000.0, lane="gas-no"))
        self.assertFalse(arming.fresh_operator_act(record, now=1000.0 + arming.MAX_AGE_S + 1,
                                                   lane="gas-no"))


class FakeVenue:
    """Just enough Kalshi read surface for the canary and full-size paths."""

    def __init__(self, now: float, diesel_strikes=(5.935,)) -> None:
        self.now = now
        self.diesel_strikes = diesel_strikes
        self.balance = D("100.00")

    @staticmethod
    def _open_time(now: float) -> str:
        return datetime.utcfromtimestamp(now - 6 * 3600).isoformat() + "Z"

    async def get(self, path, params=None):
        params = params or {}
        if path == "/events":
            series = params.get("series_ticker")
            if params.get("status") == "settled":
                return 200, {"events": [{"event_ticker": f"{series}-26NOV18"},
                                        {"event_ticker": f"{series}-26NOV17"}]}
            return 200, {"events": [{"event_ticker": f"{series}-26NOV19"}]}
        if path == "/markets":
            event = params.get("event_ticker", "")
            series = event.rsplit("-", 1)[0]
            if "26NOV18" in event:
                value = "3.7502" if series == "KXAAAGASD" else "5.9594"
                return 200, {"markets": [{"expiration_value": value}]}
            if "26NOV17" in event:
                value = "3.7818" if series == "KXAAAGASD" else "5.9848"
                return 200, {"markets": [{"expiration_value": value}]}
            rows = []
            if series == "KXAAAGASD":
                rows.append({"ticker": "KXAAAGASD-26NOV19-3.7200", "floor_strike": 3.72,
                             "strike_type": "greater", "status": "active",
                             "open_time": self._open_time(self.now)})
            else:
                for strike in self.diesel_strikes:
                    rows.append({"ticker": f"KXDIESELD-26NOV19-T{strike:.3f}",
                                 "floor_strike": strike, "strike_type": "greater",
                                 "status": "active",
                                 "open_time": self._open_time(self.now)})
            return 200, {"markets": rows}
        if path.endswith("/orderbook"):
            ticker = path.split("/")[-2]
            if ticker.endswith("3.7200"):
                return 200, {"orderbook_fp": {"yes_dollars": [["0.1500", "74.00"]]}}
            if ticker.endswith("T5.930"):
                return 200, {"orderbook_fp": {"yes_dollars": [["0.0200", "500.00"]]}}
            return 200, {"orderbook_fp": {"yes_dollars": [["0.0100", "3001.00"]]}}
        raise AssertionError(f"unexpected GET {path}")

    async def close(self):
        return None


class FakeBroker:
    def __init__(self, venue: FakeVenue) -> None:
        self.venue = venue
        self.placed: list[tuple[str, Decimal, int, str, str]] = []
        self.errors: list[str] = []

    async def balance(self) -> Decimal:
        return self.venue.balance

    async def place_no_entry(self, ticker, no_price, contracts, tif="immediate_or_cancel"):
        order_id = f"o{len(self.placed) + 1}"
        self.placed.append((ticker, D(no_price), int(contracts), tif, order_id))
        self.venue.balance -= D(no_price) * contracts + taker_fee(no_price, contracts)
        return VenueOrder(order_id=order_id, client_order_id="c", ticker=ticker,
                          side="NO", action="entry", price=D(no_price),
                          contracts=contracts, status="executed", filled=contracts)

    async def fills(self, order_id=None, ticker=None, limit=100):
        ticker, price, contracts, _tif, oid = next(
            row for row in self.placed if row[4] == order_id)
        return [{"order_id": oid, "count_fp": f"{contracts}.00",
                 "no_price_dollars": str(price),
                 "fee_cost": str(taker_fee(price, contracts))}]

    async def position(self, ticker: str) -> int:
        return -sum(row[2] for row in self.placed if row[0] == ticker)


class LiveWiring(unittest.TestCase):
    def test_canary_places_one_reconciles_and_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = G.LaneLedger(Path(tmp) / "state.json", Path(tmp) / "log.jsonl")
            venue = FakeVenue(NOW)
            broker = FakeBroker(venue)
            result = asyncio.run(G.canary(None, ledger, live=True, at=NOW,
                                          client=venue, broker=broker))
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["contracts"], 1)
            self.assertEqual(D(result["price"]), D("0.99"))
            self.assertEqual(result["venue_position"], -1)
            self.assertEqual(D(result["balance_delta"]), D("0.00"))
            self.assertEqual(broker.placed[0][3], "immediate_or_cancel")
            self.assertTrue(ledger.canary_ok())
            self.assertEqual(len(ledger.open_positions()), 1)
            self.assertEqual(ledger.open_positions()[0]["contracts"], 1)

    def test_execute_uses_full_size_after_the_canary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = G.LaneLedger(Path(tmp) / "state.json", Path(tmp) / "log.jsonl")
            ledger.data["canary"] = {"ok": True, "ts": NOW}
            venue = FakeVenue(NOW, diesel_strikes=(5.935, 5.930))
            broker = FakeBroker(venue)
            result = asyncio.run(G.execute(None, ledger, live=True, at=NOW,
                                           client=venue, broker=broker))
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["placed"], 2)
            self.assertEqual([row[2] for row in broker.placed], [10, 10])
            self.assertEqual({row[1] for row in broker.placed}, {D("0.99"), D("0.98")})
            self.assertEqual({row[3] for row in broker.placed}, {"immediate_or_cancel"})
            self.assertEqual(len(ledger.open_positions()), 2)
            self.assertEqual(sum(p["contracts"] for p in ledger.open_positions()), 20)

    def test_execute_refuses_without_a_passed_canary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = G.LaneLedger(Path(tmp) / "state.json", Path(tmp) / "log.jsonl")
            result = asyncio.run(G.execute(None, ledger, live=True, at=NOW,
                                           client=FakeVenue(NOW),
                                           broker=FakeBroker(FakeVenue(NOW))))
            self.assertFalse(result["ok"])
            self.assertIn("canary", result["error"])


if __name__ == "__main__":
    unittest.main()
