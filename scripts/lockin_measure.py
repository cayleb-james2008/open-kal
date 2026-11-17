"""Measure the final-minute lock-in model on recorded production tape.

For every settled 15-minute market we can resolve, this walks the last-minute
accumulator tick by tick, prices YES with `features.final_minute`, and asks the only
question that matters: at the price the book was actually offering, did the model make
money after the exact fee?

Nothing is fitted here. The probability comes from the locked k/60 share and the
Brownian-bridge variance of the rest; the only free input is sigma, which is measured
from the recorded index tape over the same window.

Reports Brier(model) against Brier(book mid) on the identical ticks, and an
in-sample / out-of-sample split by close time.
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import time
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openkal import features  # noqa: E402
from openkal.auth import KalshiClient  # noqa: E402
from openkal.config import load_settings  # noqa: E402
from openkal.kalshi_ws import INDEX_BY_SERIES  # noqa: E402

SERIES = list(INDEX_BY_SERIES)
DB = Path("data/openkal-prod.db")
CACHE = Path("data/settled-markets.json")
HURDLE = Decimal(os.environ.get("HURDLE", "0.008"))
BAND = (Decimal(os.environ.get("BAND_LO", "0.20")), Decimal(os.environ.get("BAND_HI", "0.84")))


async def fetch_settled(pages: int = 6) -> list[dict]:
    client = KalshiClient(load_settings())
    out: list[dict] = []
    for series in SERIES:
        cursor = None
        for _ in range(pages):
            params = {"series_ticker": series, "status": "settled", "limit": 200}
            if cursor:
                params["cursor"] = cursor
            status, payload = await client.get("/markets", params)
            if status != 200:
                break
            out.extend(payload.get("markets") or [])
            cursor = payload.get("cursor")
            if not cursor:
                break
    await client.close()
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(out))
    return out


def load_markets() -> list[dict]:
    if CACHE.exists():
        stamp = time.time() - CACHE.stat().st_mtime
        if stamp < 1800:
            return json.loads(CACHE.read_text())
    return asyncio.run(fetch_settled())


def iso_ts(value: str | None) -> float:
    if not value:
        return 0.0
    from datetime import datetime, timezone
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def measure(db: sqlite3.Connection, market: dict) -> list[dict]:
    series = market.get("ticker", "").rsplit("-", 2)[0]
    index_id = INDEX_BY_SERIES.get(series)
    close = iso_ts(market.get("close_time"))
    strike = market.get("floor_strike")
    if not index_id or not strike or not close:
        return []
    rows = db.execute(
        "SELECT ts, value, final_minute, final_ticks FROM ws_index"
        " WHERE index_id=? AND final_minute IS NOT NULL AND final_minute != 'None'"
        " AND ts BETWEEN ? AND ? ORDER BY ts",
        (index_id, close - 75, close + 20),
    ).fetchall()
    if not rows:
        return []
    tape = [(r[0], float(r[1])) for r in rows if float(r[1]) > 0]
    sigma = features.realized_sigma([(r[0], float(r[1])) for r in rows], 2.0) or 2.5e-5
    result = (market.get("result") or "").lower()
    if result not in {"yes", "no"}:
        return []
    truth = 1.0 if result == "yes" else 0.0
    ticker = market.get("ticker")
    out: list[dict] = []
    entered = False
    for ts, value, accum, ticks in rows:
        if ticks <= 0:
            continue
        seconds_left = close - ts
        if seconds_left <= 0:
            continue
        sig = features.final_minute(spot=float(value), strike=float(strike),
                                    locked_avg=float(accum), locked_ticks=int(ticks),
                                    sigma_per_sqrt_s=sigma, seconds_left=seconds_left)
        if sig is None:
            continue
        book = db.execute(
            "SELECT yes_bid, yes_ask FROM ws_book WHERE ticker=? AND ts BETWEEN ? AND ?"
            " ORDER BY abs(ts-?) LIMIT 1", (ticker, ts - 2, ts + 2, ts)).fetchone()
        # A YES buy needs a YES ask; a NO buy needs a YES bid (NO ask = 1 - yes bid).
        # In the last seconds one side of the book is often empty, and that emptiness is
        # itself the answer: there is nothing to trade against.
        bid = Decimal(book[0]) if book and book[0] not in (None, "None") else None
        ask = Decimal(book[1]) if book and book[1] not in (None, "None") else None
        if ask is None:
            continue
        mid = ask if bid is None else (bid + ask) / 2
        p = sig.p_up
        side, price, p_side, cost_side = None, None, None, None
        if HURDLE and (p - float(ask)) >= float(HURDLE) and BAND[0] <= ask <= BAND[1]:
            side, price, p_side = "YES", ask, p
            payout = truth
        elif bid is not None and ((1 - p) - float(Decimal(1) - bid)) >= float(HURDLE) and BAND[0] <= (Decimal(1) - bid) <= BAND[1]:
            side, price, p_side = "NO", Decimal(1) - bid, 1 - p
            payout = 1 - truth
        else:
            continue
        # One position per window per lane: consecutive ticks are the same trade, not
        # nine independent ones, and counting them as nine would flatter any result.
        if entered:
            continue
        entered = True
        fee = Decimal("0.07") * price * (Decimal(1) - price)
        net = Decimal(str(payout)) - price - fee
        out.append({"ts": ts, "close": close, "series": series, "ticker": ticker,
                    "k": int(ticks), "p": p, "mid": float(mid), "side": side,
                    "price": str(price), "truth": truth, "net": float(net),
                    "detail": sig.detail})
    return out


def main() -> int:
    markets = load_markets()
    db = sqlite3.connect(DB)
    trades: list[dict] = []
    brier_m: list[tuple[float, int]] = []
    brier_b: list[tuple[float, int]] = []
    for market in markets:
        rows = measure(db, market)
        for row in rows:
            trades.append(row)
            brier_m.append((row["p"], int(row["truth"])))
            brier_b.append((row["mid"], int(row["truth"])))
    if not trades:
        print(json.dumps({"trades": 0, "note": "no settled window with accumulator + book overlap"}))
        return 0
    closes = sorted({t["close"] for t in trades})
    split = closes[int(len(closes) * 0.7)] if len(closes) > 1 else closes[0]
    def block(rows: list[dict]) -> dict:
        net = sum(r["net"] for r in rows)
        wins = sum(1 for r in rows if r["net"] > 0)
        return {"trades": len(rows), "wins": wins,
                "win_rate": round(wins / len(rows), 3) if rows else 0,
                "net": round(net, 4),
                "avg_net_c": round(net / len(rows) * 100, 3) if rows else 0,
                "windows": len({r["close"] for r in rows})}
    report = {
        "windows_resolved": len(closes),
        "all": block(trades),
        "in_sample": block([t for t in trades if t["close"] < split]),
        "out_of_sample": block([t for t in trades if t["close"] >= split]),
        "brier_model": round(features.brier(brier_m), 4),
        "brier_book_mid": round(features.brier(brier_b), 4),
        "sample_ticks": len(trades),
    }
    print(json.dumps(report, indent=2))
    print("\nper window:")
    for close in closes:
        rows = [t for t in trades if t["close"] == close]
        b = block(rows)
        print(f"  {time.strftime('%H:%M', time.localtime(close))} {b['trades']:3d} ticks  "
              f"net {b['net']:+.3f}  win {b['win_rate']:.2f}")
    Path("data/lockin-report.json").write_text(json.dumps(
        {"report": report, "trades": trades}, indent=2))
    print("\nwrote data/lockin-report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
