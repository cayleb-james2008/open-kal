#!/usr/bin/env python3
"""Reconcile the live ledger against the venue's own tapes, to the cent.

Nothing here estimates a fill or a fee. Venue numbers come from
`/portfolio/fills` and `/portfolio/settlements`; local numbers are rows the engine
wrote when the venue reported them. Output is per-ticker cash flow, so a missing
fill, a half-booked split print or a wrong fee shows up as a delta instead of
hiding inside a total.

Cash accounting per ticker:

  * a ticker the venue settled is priced from its settlement row alone --
    payout minus the venue's own cost columns minus the venue fee. The fills for
    that ticker are already inside those cost columns, and the venue repeats the
    trading fee on the settlement row, so summing both legs double counts.
  * a ticker never settled is priced from its fills: buys cost, sells pay, fees
    always cost.

    .venv/bin/python scripts/reconcile_live.py                    # whole live history
    .venv/bin/python scripts/reconcile_live.py --since 1794857851 # current epoch
    .venv/bin/python scripts/reconcile_live.py --json data/reconcile-live.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openkal.auth import KalshiClient  # noqa: E402
from openkal.config import load_settings  # noqa: E402

CENTS = Decimal("0.01")


def dec(value) -> Decimal:
    try:
        return Decimal(str(value or "0"))
    except Exception:  # noqa: BLE001 - a malformed venue number is data, not a crash
        return Decimal("0")


def cents(value: Decimal) -> str:
    return str(value.quantize(CENTS))


def settled_ts(row: dict) -> float:
    raw = row.get("settled_time") or ""
    try:
        return datetime.strptime(raw, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=timezone.utc).timestamp()
    except ValueError:
        return float(row.get("settled_ts") or row.get("ts") or 0)


async def paged(client: KalshiClient, path: str, key: str, limit: int = 200,
                pages: int = 40) -> list[dict]:
    rows: list[dict] = []
    cursor = None
    for _ in range(pages):
        params: dict = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        status, payload = await client.get(path, params)
        if status != 200:
            raise SystemExit(f"{path}: HTTP {status} {str(payload)[:200]}")
        batch = payload.get(key) or []
        rows.extend(batch)
        cursor = payload.get("cursor")
        if not cursor or not batch:
            break
    return rows


def settlement_cash(row: dict) -> Decimal:
    """Payout this settlement delivered minus what the venue says it cost."""
    value = dec(row.get("value")) / Decimal("100")
    payout = (dec(row.get("yes_count_fp")) * value
              + dec(row.get("no_count_fp")) * (Decimal("1") - value))
    cost = dec(row.get("yes_total_cost_dollars")) + dec(row.get("no_total_cost_dollars"))
    return payout - cost - dec(row.get("fee_cost"))


def fill_cash(row: dict) -> Decimal:
    """Signed cash flow of one print: buying costs, selling pays, fee always costs."""
    count = dec(row.get("count_fp"))
    side = row.get("side")
    # An exit print stamped ` side=no ` (e.g. a yes-long position leaving at the offer)
    # pays `yes_price_dollars` per contract, not the thin NO-side price: the position
    # is the same contract both ways. Empirically the venue's own realized_pnl for the
    # 26NOV161700-00 window equals `exit - entry` on the YES side net of fees exactly.
    if side == "yes":
        price = dec(row.get("yes_price_dollars"))
    elif side == "no" and row.get("action") == "sell":
        price = dec(row.get("yes_price_dollars"))
    else:
        price = dec(row.get("no_price_dollars"))
    sign = Decimal("-1") if row.get("action") == "buy" else Decimal("1")
    return sign * count * price - dec(row.get("fee_cost"))


def load_ledger(db: Path, since: float) -> dict[str, list[dict]]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "select * from trades where closed_ts >= ? order by id", (since,))]
    con.close()
    by_ticker: dict[str, list[dict]] = defaultdict(list)
    for t in rows:
        by_ticker[t["ticker"]].append(t)
    return by_ticker


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=float, default=0.0,
                    help="unix ts; default is the first live trade in the db")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    settings = load_settings()
    db = settings.db_path
    if not args.since:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        row = con.execute("select min(opened_ts) from trades where source='live'").fetchone()
        con.close()
        args.since = float(row[0] or 0.0)

    client = KalshiClient(settings)
    try:
        balance = (await client.get("/portfolio/balance"))[1]
        positions = (await client.get("/portfolio/positions", {"limit": 200}))[1]
        orders = (await client.get("/portfolio/orders", {"status": "resting", "limit": 200}))[1]
        fills = await paged(client, "/portfolio/fills", "fills")
        settlements = await paged(client, "/portfolio/settlements", "settlements")
    finally:
        await client.close()

    fills = [f for f in fills if float(f.get("ts") or 0) >= args.since]
    settlements = [s for s in settlements if settled_ts(s) >= args.since]
    # The venue lists one market_position row per ticker the account has ever
    # touched, flat ones included. Counting the rows would report an open
    # position on a flat account.
    pos_rows = [r for r in (positions.get("market_positions") or [])
                if float(r.get("position_fp") or 0) != 0]
    order_rows = orders.get("orders") or []

    venue_fill_cash: dict[str, Decimal] = defaultdict(Decimal)
    venue_fees: dict[str, Decimal] = defaultdict(Decimal)
    venue_prints: dict[str, int] = defaultdict(int)
    for f in fills:
        ticker = f.get("ticker") or f.get("market_ticker") or "?"
        venue_fill_cash[ticker] += fill_cash(f)
        venue_fees[ticker] += dec(f.get("fee_cost"))
        venue_prints[ticker] += 1
    settled: dict[str, Decimal] = {}
    for s in settlements:
        settled[s.get("ticker") or "?"] = settlement_cash(s)

    ledger = load_ledger(db, args.since)
    tickers = sorted(set(venue_fill_cash) | set(settled) | set(ledger)
                     | {p["ticker"] for p in pos_rows})
    rows = []
    total_venue = total_ledger = Decimal("0")
    for ticker in tickers:
        local = sum((dec(t["pnl"]) for t in ledger.get(ticker, [])), Decimal("0"))
        v = settled[ticker] if ticker in settled else venue_fill_cash.get(ticker, Decimal("0"))
        total_venue += v
        total_ledger += local
        rows.append({
            "ticker": ticker,
            "venue_net": cents(v),
            "ledger_net": cents(local),
            "delta": cents(v - local),
            "source": "settlement" if ticker in settled else "fills",
            "venue_prints": venue_prints.get(ticker, 0),
            "venue_fees": cents(venue_fees.get(ticker, Decimal("0"))),
            "trades": [{
                "id": t["id"], "side": t["side"], "entry": t["entry"],
                "exit": t["exit_price"], "contracts": t["contracts"],
                "fee": t["fee"], "pnl": t["pnl"], "reason": t["exit_reason"],
                "closed": time.strftime("%H:%M:%S", time.localtime(t["closed_ts"])),
            } for t in ledger.get(ticker, [])],
        })

    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "env": settings.env_name,
        "since": args.since,
        "since_local": time.strftime("%H:%M:%S", time.localtime(args.since)),
        "balance_dollars": balance.get("balance_dollars"),
        "open_positions": len(pos_rows),
        "resting_orders": len(order_rows),
        "venue_net": cents(total_venue),
        "ledger_net": cents(total_ledger),
        "delta": cents(total_venue - total_ledger),
        "venue_fills": len(fills),
        "venue_settlements": len(settlements),
        "tickers": rows,
        "settlements": [{
            "ticker": s.get("ticker"), "result": s.get("market_result"),
            "value": s.get("value"), "yes": s.get("yes_count_fp"),
            "no": s.get("no_count_fp"), "fee": s.get("fee_cost"),
            "cost": cents(dec(s.get("yes_total_cost_dollars"))
                          + dec(s.get("no_total_cost_dollars"))),
            "cash": cents(settlement_cash(s)),
            "settled": s.get("settled_time"),
        } for s in sorted(settlements, key=settled_ts)],
        "positions": pos_rows,
        "orders": [{k: o.get(k) for k in
                    ("order_id", "ticker", "side", "action", "status",
                     "yes_price_dollars", "no_price_dollars", "remaining_count_fp")}
                   for o in order_rows],
    }
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=str))

    print(f"env {report['env']}  balance ${report['balance_dollars']}  "
          f"positions {report['open_positions']}  resting {report['resting_orders']}")
    print(f"window {report['since_local']} -> now  "
          f"({len(fills)} venue prints, {len(settlements)} settlements)")
    print(f"{'ticker':32s} {'venue':>9s} {'ledger':>9s} {'delta':>8s} {'fees':>7s} src  trades")
    for r in rows:
        print(f"{r['ticker']:32s} {r['venue_net']:>9s} {r['ledger_net']:>9s} "
              f"{r['delta']:>8s} {r['venue_fees']:>7s} {r['source'][:4]:4s} {len(r['trades'])}")
    print(f"{'TOTAL':32s} {report['venue_net']:>9s} {report['ledger_net']:>9s} "
          f"{report['delta']:>8s}")
    for s in report["settlements"]:
        print(f"  settled {s['settled']} {s['ticker']} result={s['result']} "
              f"yes={s['yes']} no={s['no']} cost={s['cost']} fee={s['fee']} cash={s['cash']}")
    if pos_rows:
        print("\nopen positions:", json.dumps(pos_rows)[:600])
    if order_rows:
        print("resting orders:", json.dumps(report["orders"])[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
