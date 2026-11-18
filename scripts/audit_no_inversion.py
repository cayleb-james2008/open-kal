#!/usr/bin/env python3
"""Audit every live crypto NO entry in the ledger against the venue's fills.

Kalshi V2 quotes every order on the YES leg.  A NO entry must therefore travel as
`side="ask"` at `1 - no_price`; sending the NO price directly inverts the order
(95c NO becomes 5c NO).  The ledger records the price the venue filled, so the
question is whether that price matches the NO price the strategy asked for.

Without the original wire body, the inversion signature is still visible: a NO
order whose ledger price and venue price sum to about 1.00 is the opposite side
of the same trade.  This script prints, per order, the ledger price, the venue's
NO price, the venue's YES price, and the same-ticker/side/count fill regardless
of timestamp.

Read-only: no orders, no state writes.

    .venv/bin/python scripts/audit_no_inversion.py --db data/openkal-prod.db
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openkal.auth import KalshiClient  # noqa: E402
from openkal.config import load_settings  # noqa: E402

NO_ENTRY_NOTES = ("live fill", "maker", "taker cross")


def dec(value) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception:  # noqa: BLE001 - a malformed venue row is not a price
        return Decimal("0")


def live_no_entries(db: sqlite3.Connection) -> list[dict]:
    """Live NO entries, joined to the trade row that carries the source tag."""
    rows = db.execute(
        """SELECT f.id, f.ts, f.ticker, f.price, f.contracts, f.fee, f.note,
                  t.id AS trade_id, t.source
           FROM fills f
           LEFT JOIN trades t
             ON t.ticker = f.ticker AND t.side = 'NO'
            AND ABS(t.opened_ts - f.ts) < 30.0
           WHERE f.side = 'NO' AND f.kind = 'entry'
             AND (t.source = 'live'
                  OR (t.id IS NULL AND f.note = 'live fill'))
           ORDER BY f.ts""").fetchall()
    return [dict(zip(
        ("fill_id", "ts", "ticker", "price", "contracts", "fee", "note",
         "trade_id", "source"), row)) for row in rows]


async def venue_fills(client: KalshiClient, min_ts: int, max_ts: int) -> list[dict]:
    out: list[dict] = []
    cursor = None
    for _ in range(20):
        params = {"limit": 1000, "min_ts": min_ts, "max_ts": max_ts}
        if cursor:
            params["cursor"] = cursor
        status, payload = await client.get("/portfolio/fills", params)
        if status != 200:
            raise RuntimeError(f"/portfolio/fills HTTP {status}: {str(payload)[:200]}")
        rows = payload.get("fills") or []
        out.extend(rows)
        cursor = payload.get("cursor")
        if not cursor or not rows:
            break
    return out


def venue_orders(fills: list[dict]) -> list[dict]:
    """One row per venue order: every print on it summed to whole contracts."""
    groups: dict[tuple, list[dict]] = {}
    for fill in fills:
        key = (fill.get("order_id"), fill.get("ticker"),
               str(fill.get("side") or "").lower())
        groups.setdefault(key, []).append(fill)
    out = []
    for (order_id, ticker, side), rows in groups.items():
        qty = sum(dec(r.get("count_fp") or r.get("count")) for r in rows)
        if qty <= 0:
            continue
        no = sum(dec(r.get("count_fp") or r.get("count"))
                 * dec(r.get("no_price_dollars")) for r in rows) / qty
        yes = sum(dec(r.get("count_fp") or r.get("count"))
                  * dec(r.get("yes_price_dollars")) for r in rows) / qty
        out.append({
            "order_id": order_id, "ticker": ticker, "side": side,
            "contracts": qty, "no_price": no, "yes_price": yes,
            "fee": sum(dec(r.get("fee_cost")) for r in rows),
        })
    return out


def match(entry: dict, orders: list[dict]) -> dict | None:
    """The venue order for this ledger row: same ticker, NO side, same size."""
    candidates = [
        order for order in orders
        if order["ticker"] == entry["ticker"] and order["side"] == "no"
        and abs(order["contracts"] - Decimal(entry["contracts"])) <= Decimal("1.0")
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda row: abs(row["contracts"]
                                               - Decimal(entry["contracts"])))


async def run(db_path: Path, min_ts: int, max_ts: int) -> dict:
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=20)
    try:
        entries = live_no_entries(db)
    finally:
        db.close()
    client = KalshiClient(load_settings())
    try:
        fills = await venue_fills(client, min_ts, max_ts)
    finally:
        await client.close()
    orders = venue_orders(fills)

    rows = []
    inverted = []
    unmatched = []
    matched = 0
    used: set[str] = set()
    for entry in entries:
        order = match(entry, orders)
        ledger_price = dec(entry["price"])
        row = {
            "fill_id": entry["fill_id"], "ts": entry["ts"],
            "ticker": entry["ticker"], "contracts": entry["contracts"],
            "ledger_no": str(ledger_price), "order_id": None,
            "venue_no": None, "venue_yes": None, "fee": entry["fee"],
            "note": entry["note"],
        }
        if order is None or order["order_id"] in used:
            unmatched.append(row)
            row["verdict"] = "no_venue_match"
            rows.append(row)
            continue
        used.add(order["order_id"])
        venue_no = order["no_price"]
        row["order_id"] = order["order_id"]
        row["venue_no"] = str(venue_no)
        row["venue_yes"] = str(order["yes_price"])
        row["fee"] = str(order["fee"])
        if abs(venue_no - ledger_price) <= Decimal("0.011"):
            row["verdict"] = "ok"
            matched += 1
        elif abs((Decimal(1) - venue_no) - ledger_price) <= Decimal("0.011"):
            row["verdict"] = "inverted"
            inverted.append(row)
        else:
            row["verdict"] = "price_mismatch"
        rows.append(row)

    # The landmark: `place()` forwards the side's own price, so every live NO
    # entry was wired as a YES ask at the NO price.  A cheap NO buy (<50c) is a
    # deep, marketable sell-YES and the venue clamps it to the resting bid, so
    # the fill is right; a NO buy >=50c is a non-marketable YES ask and cannot
    # fill.  Both are wire inversions even when the tape looks correct.
    cheap = [e for e in entries if dec(e["price"]) < Decimal("0.50")]
    expensive = [e for e in entries if dec(e["price"]) >= Decimal("0.50")]
    notional = sum(dec(r["contracts"]) * dec(r["ledger_no"]) for r in inverted)
    delta = sum(dec(r["contracts"]) * abs(dec(r["venue_no"]) - dec(r["ledger_no"]))
                for r in inverted if r["venue_no"])
    return {
        "db": str(db_path),
        "live_no_entries": len(entries),
        "venue_fills_seen": len(fills),
        "venue_orders_seen": len(orders),
        "matched": matched,
        "inverted": len(inverted),
        "unmatched": len(unmatched),
        "inverted_intended_notional": str(notional),
        "inverted_price_delta": str(delta),
        "wire_inverted_entries": len(entries),
        "wire_cheap_marketable": len(cheap),
        "wire_expensive_nonmarketable": len(expensive),
        "wire_cheap_notional": str(sum(dec(e["contracts"]) * dec(e["price"])
                                       for e in cheap)),
        "wire_expensive_notional": str(sum(dec(e["contracts"]) * dec(e["price"])
                                           for e in expensive)),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=str(ROOT / "data" / "openkal-prod.db"))
    parser.add_argument("--min-ts", type=int, default=1794800000)
    parser.add_argument("--max-ts", type=int, default=1795000000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(run(Path(args.db), args.min_ts, args.max_ts))
    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0
    print(f"live NO entries {result['live_no_entries']}  "
          f"venue prints {result['venue_fills_seen']}  "
          f"venue orders {result['venue_orders_seen']}  "
          f"matched {result['matched']}  inverted {result['inverted']}  "
          f"unmatched {result['unmatched']}")
    print(f"wire-inverted entries {result['wire_inverted_entries']}: "
          f"{result['wire_cheap_marketable']} marketable NO<50c "
          f"(${result['wire_cheap_notional']}), "
          f"{result['wire_expensive_nonmarketable']} non-marketable NO>=50c "
          f"(${result['wire_expensive_notional']})")
    for row in result["rows"]:
        print(f"  {row['ts']:.0f} {row['ticker']:34s} {row['contracts']:>4} "
              f"ledger {row['ledger_no']:>6}  venue NO {str(row['venue_no']):>6}  "
              f"YES {str(row['venue_yes']):>6}  {row['verdict']}")
    print(f"inverted intended notional ${result['inverted_intended_notional']}  "
          f"price delta ${result['inverted_price_delta']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
