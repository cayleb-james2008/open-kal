"""Live V2 micro-canary: one contract, reconciled to the cent, flattened clean.

This is the only part of the readiness gate that touches real money, and it exists for
exactly that reason: paper fills prove the strategy, not the order path. The canary
places a single post-only contract on a live 15-minute crypto market, matches the
venue's own `/portfolio/fills` record against the price it asked for, sells it back,
and requires the account balance to move by the amount the fills say it should.

It refuses to run unless all of the following are true, so it can never be an
accidental backdoor around `OPENKAL_FORCE_PAPER`:

  * `--live` was passed explicitly;
  * `.env` has `ALLOW_LIVE=1` and a production key;
  * `OPENKAL_FORCE_PAPER` is not set.

Without `--live` it runs the same selection and reconciliation arithmetic in dry mode
and writes `data/canary-dry.json`, which the readiness gate deliberately rejects: a
dry run is not evidence that a venue order works.

    .venv/bin/python scripts/canary.py            # dry: select, price, report
    .venv/bin/python scripts/canary.py --live     # one real contract, then flat
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openkal.auth import KalshiClient, money  # noqa: E402
from openkal.broker import Broker  # noqa: E402
from openkal.config import SLOTS, load_settings  # noqa: E402

OUT = ROOT / "data" / "canary.json"
DRY = ROOT / "data" / "canary-dry.json"


def _ts(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _dec(value) -> Decimal:
    try:
        return Decimal(str(value or "0"))
    except Exception:  # noqa: BLE001
        return Decimal("0")


async def pick_market(client: KalshiClient) -> dict | None:
    """Nearest open 15-minute contract with a real two-sided book and time to trade."""
    now = time.time()
    best: dict | None = None
    for cfg in SLOTS:
        if not cfg.feed_symbol:
            continue
        # `status=open` only lists the window already trading; the next one lives under
        # `unopened` until the rollover, and a canary wants a fresh book with time on it.
        for status in ("open", "unopened"):
            code, payload = await client.get(
                "/markets", {"series_ticker": cfg.series, "status": status, "limit": 40})
            if code != 200:
                continue
            for market in payload.get("markets") or []:
                close = _ts(market.get("close_time"))
                if close - now < 180 or close - now > 1000:
                    continue
                bid = money(market.get("yes_bid_dollars"))
                ask = money(market.get("yes_ask_dollars"))
                if not (Decimal("0.01") <= bid < ask <= Decimal("0.99")):
                    continue
                if ask - bid > Decimal("0.15"):
                    continue
                if best is None or close < best["close_ts"]:
                    best = {"series": cfg.series, "ticker": market["ticker"],
                            "close_ts": close, "bid": bid, "ask": ask}
    return best


async def live_quote(client: KalshiClient, ticker: str) -> tuple[Decimal, Decimal] | None:
    """The live ladder right now, not the market list's cached top-of-book.

    Pricing a post-only bid off a market snapshot that is even a few seconds old is
    how the venue answers `post only cross`: the book moved under the quote. The
    canary tests the order path, so it must price the way the engine does, off the
    current book.
    """
    from openkal.kalshi_ws import Book

    status, payload = await client.get(f"/markets/{ticker}/orderbook", {"depth": 5})
    if status != 200:
        return None
    ob = payload.get("orderbook_fp") or {}
    book = Book(ticker=ticker)
    book.apply_snapshot({"yes_dollars_fp": ob.get("yes_dollars"),
                         "no_dollars_fp": ob.get("no_dollars")}, 0)
    if book.yes_bid is None or book.yes_ask is None:
        return None
    return book.yes_bid, book.yes_ask


async def order_fill(broker: Broker, order_id: str, timeout: float) -> dict | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        fills = await broker.fills(order_id=order_id, limit=20)
        if fills:
            return fills[0]
        await asyncio.sleep(0.5)
    return None


async def other_fill(broker: Broker, ticker: str, entry_order_id: str,
                     timeout: float, *, after_ts: float = 0.0) -> dict | None:
    """The exit leg after `flatten_all()`, which places its own order id.

    `order_fill` can only match the order it was given, so the fallback flatten is
    invisible to it. Query the ticker's fill tape instead and take the newest fill
    that is neither this run's entry nor an older print on the same ticker: the
    tape keeps every fill the account ever made, and an earlier canary's entry
    looks exactly like an exit if nothing filters it by time. Picking a stale
    fill makes the reconciliation fail on a trade that was in fact flat.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        fills = await broker.fills(ticker=ticker, limit=50)
        # The venue's fill record carries the epoch in `ts`; its `created_time` is an
        # ISO string whose clock does not line up with that epoch, and it has no
        # `created_ts` field at all. Sorting on the missing field silently keeps the
        # API's own order, which is newest first -- so the "newest" fill was being
        # taken from the wrong end of the tape, and the reconciliation was priced off
        # a stale print.
        rest = [f for f in fills
                if f.get("order_id") != entry_order_id
                and float(f.get("ts") or 0) > after_ts]
        if rest:
            rest.sort(key=lambda f: float(f.get("ts") or 0))
            return rest[-1]
        await asyncio.sleep(0.5)
    return None


async def run(live: bool) -> int:
    settings = load_settings()
    payload: dict = {"generated_ts": time.time(), "mode": "live" if live else "dry",
                     "ok": False, "contracts": 1}
    if live and not settings.live:
        payload["error"] = ("live execution is disabled: needs ALLOW_LIVE=1, KALSHI_ENV=prod "
                            "and no OPENKAL_FORCE_PAPER")
        print(json.dumps(payload, indent=2))
        return 2
    client = KalshiClient(settings)
    broker = Broker(client)
    try:
        balance_before = await broker.balance()
        payload["balance_before"] = str(balance_before)
        market = await pick_market(client)
        if market is None:
            payload["error"] = "no open 15-minute market with a two-sided book"
            print(json.dumps(payload, indent=2))
            return 1
        quote = await live_quote(client, market["ticker"])
        if quote is not None and Decimal("0.01") <= quote[0] < quote[1] <= Decimal("0.99"):
            market["bid"], market["ask"] = quote
        payload.update({
            "ticker": market["ticker"], "series": market["series"],
            "close_ts": market["close_ts"],
            "requested_price": str(market["bid"]), "side": "YES",
        })
        print(json.dumps(payload, indent=2), flush=True)
        if not live:
            # Same selection and arithmetic the live path would use, without an order.
            payload["dry_plan"] = {
                "buy": f"1 post_only YES at {market['bid']} on {market['ticker']}",
                "flatten": "1 reduce_only IOC at the bid once filled",
                "reconcile": "fills tape vs balance delta, to the cent",
            }
            DRY.write_text(json.dumps(payload, indent=2, default=str))
            print(f"dry run written to {DRY}")
            return 0

        order = await broker.place(market["ticker"], "YES", "entry", market["bid"], 1,
                                   post_only=True, tif="good_till_canceled")
        if order is None:
            payload["error"] = "venue rejected the bid: " + (
                broker.errors[-1] if broker.errors else "unknown")
            print(json.dumps(payload, indent=2))
            return 1
        payload["order_id"] = order.order_id
        fill = await order_fill(broker, order.order_id, timeout=30.0)
        if fill is None:
            await broker.cancel(order.order_id, market["ticker"])
            payload["error"] = "no fill in 30s; canary cancelled"
            OUT.write_text(json.dumps(payload, indent=2, default=str))
            print(json.dumps(payload, indent=2))
            return 1
        entry_price = _dec(fill.get("yes_price_dollars")) or market["bid"]
        entry_fee = _dec(fill.get("fee_cost"))
        fill_count = int(float(fill.get("count_fp") or 1))
        payload.update({"fill_price": str(entry_price), "entry_fee": str(entry_fee),
                        "fill_count": fill_count, "is_taker": bool(fill.get("is_taker"))})
        if fill_count != 1:
            payload["error"] = f"venue filled {fill_count} contracts, expected 1"
            OUT.write_text(json.dumps(payload, indent=2, default=str))
            return 1

        # Flatten immediately at the touch; a reactive 1c haircut is acceptable for a
        # canary, an open position is not.
        flat = await broker.place(market["ticker"], "YES", "exit", market["bid"], 1,
                                  post_only=False, tif="immediate_or_cancel",
                                  reduce_only=True)
        exit_fill = await order_fill(broker, flat.order_id, timeout=15.0) if flat else None
        if exit_fill is None:
            await broker.flatten_all()
            exit_fill = await order_fill(broker, flat.order_id, timeout=10.0) if flat else None
        if exit_fill is None:
            exit_fill = await other_fill(broker, market["ticker"], order.order_id, 10.0,
                                         after_ts=float(fill.get("ts") or 0))
        if exit_fill is None:
            payload["error"] = "flatten did not fill; flatten_all issued"
            OUT.write_text(json.dumps(payload, indent=2, default=str))
            print(json.dumps(payload, indent=2), flush=True)
            return 1
        exit_price = _dec(exit_fill.get("yes_price_dollars")) or market["bid"]
        exit_fee = _dec(exit_fill.get("fee_cost"))
        position = await broker.position(market["ticker"])
        balance_after = await broker.balance()

        expected = (exit_price - entry_price) - entry_fee - exit_fee
        actual = balance_after - balance_before
        delta = abs(expected - actual)
        payload.update({
            "exit_price": str(exit_price), "exit_fee": str(exit_fee),
            "balance_after": str(balance_after), "expected_move": str(expected),
            "actual_move": str(actual), "reconcile_delta": str(delta),
            "reconciled": delta <= Decimal("0.01"),
            "flattened": position == 0, "position_after": position,
        })
        payload["ok"] = bool(payload["reconciled"] and payload["flattened"]
                             and fill_count == 1)
        payload.pop("error", None)
        OUT.write_text(json.dumps(payload, indent=2, default=str))
        print(json.dumps(payload, indent=2))
        return 0 if payload["ok"] else 1
    finally:
        await client.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="place one real contract (requires ALLOW_LIVE=1, prod, no force-paper)")
    return asyncio.run(run(parser.parse_args().live))


if __name__ == "__main__":
    raise SystemExit(main())
