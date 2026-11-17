"""Lock-in measured against the venue's own ladder, not the 1 Hz decision log.

The paper tick quote falls back to a REST snapshot when the socket book loses a
side. In the final minute that snapshot can be 40 seconds old, which is how a book
that was 0.999 with no asks appeared as a 0.31 ask in the tape. This script prices
every candidate at the `ws_book` top of book that existed at that instant, and
refuses to trade when no ask exists.

    .venv/bin/python scripts/lockin_book.py --lane KXBTC15M
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openkal import features  # noqa: E402
from openkal.fees import taker_fee  # noqa: E402
from scripts.micro_study import (INDEX, LANES, chrono_split, connect,  # noqa: E402
                                 load_acc, market_meta)

CACHE = ROOT / "data" / "evolve-cache"

STAKE = Decimal("65")


def lane_study(lane: str, k_min: int, p_min: float, price_max: float,
               edge_min: float) -> dict:
    book = dict(np.load(CACHE / f"book_{lane}.npz"))
    names = [str(x) for x in book["tickers"]]
    markets = market_meta()
    acc = load_acc(connect(), INDEX[lane])
    index = dict(np.load(CACHE / f"index_{INDEX[lane]}.npz"))

    rows = []
    for tid, name in enumerate(names):
        m = markets.get(name)
        if m is None:
            continue
        mask = book["tid"] == tid
        ts = book["ts"][mask]
        bid = book["bid"][mask]
        ask = book["ask"][mask]
        if len(ts) < 10:
            continue
        lo = np.searchsorted(acc["ts"], m["open"])
        hi = np.searchsorted(acc["ts"], m["close"])
        best = None
        for j in range(lo, hi):
            k = float(acc["k"][j])
            secs = m["close"] - float(acc["ts"][j])
            if k < k_min or not (0 < secs <= 60) or k > (60 - secs) + 3:
                continue
            if not math.isfinite(acc["accum"][j]):
                continue
            idx = np.searchsorted(ts, acc["ts"][j], "right") - 1
            if idx < 0 or acc["ts"][j] - ts[idx] > 1.5:
                continue
            b, a = float(bid[idx]), float(ask[idx])
            if not (0 < b <= a < 1):
                continue
            sig = features.final_minute(
                spot=float(acc["value"][j]), strike=m["strike"],
                locked_avg=float(acc["accum"][j]), locked_ticks=int(k),
                sigma_per_sqrt_s=_sigma(index, m["close"]), seconds_left=secs)
            if sig is None or sig.p_up < p_min:
                continue
            mid = (b + a) / 2
            side = "YES" if sig.p_up >= mid else "NO"
            price = Decimal(str(a)) if side == "YES" else Decimal(1) - Decimal(str(b))
            if price < Decimal("0.20") or price > Decimal(str(price_max)):
                continue
            p_side = sig.p_up if side == "YES" else 1 - sig.p_up
            qty = max(1, int(STAKE / price))
            fee = taker_fee(price, qty) / qty
            if Decimal(str(p_side)) - price - fee < Decimal(str(edge_min)):
                continue
            if best is None:
                best = {"lane": lane, "ticker": name, "ts": float(acc["ts"][j]),
                        "close": m["close"], "k": int(k), "secs": secs,
                        "side": side, "price": float(price), "p": p_side,
                        "mid": mid, "y": m["y"], "qty": qty,
                        "won": (side == "YES" and m["y"] == 1) or (side == "NO" and m["y"] == 0)}
        if best is not None:
            fee = taker_fee(Decimal(str(best["price"])), best["qty"]) / best["qty"]
            best["fee"] = float(fee)
            best["pnl"] = ((1.0 if best["won"] else 0.0) - best["price"] - float(fee))
            rows.append(best)
    return split(rows, names, markets)


def _sigma(index: dict, close: float) -> float:
    lo = int(np.searchsorted(index["ts"], close - 150.0))
    hi = int(np.searchsorted(index["ts"], close))
    pts = [(float(a), float(b)) for a, b in zip(index["ts"][lo:hi], index["value"][lo:hi])
           if b > 0]
    return features.realized_sigma(pts)


def split(rows: list[dict], names: list[str], markets: dict[str, dict]) -> dict:
    is_closes, hours = chrono_split(names, markets)
    out = {"rows": rows, "hours": hours}
    for name in ("is", "oos"):
        sel = [r for r in rows if (r["close"] in is_closes) == (name == "is")]
        out[name] = _block(sel, hours[name])
    return out


def _block(rows: list[dict], hours: float = 0.0) -> dict:
    n = len(rows)
    if not n:
        return {"trades": 0, "wins": 0, "win_rate": 0.0, "net": "0", "avg_c": 0.0,
                "trades_per_hour": 0.0, "lanes": {}}
    net = sum(Decimal(str(r["pnl"])) for r in rows)
    wins = sum(1 for r in rows if r["pnl"] > 0)
    lanes = {}
    for r in rows:
        lanes[r["lane"]] = lanes.get(r["lane"], 0) + 1
    return {"trades": n, "wins": wins, "win_rate": round(wins / n, 4),
            "net": str(net.quantize(Decimal("0.01"))),
            "avg_c": round(float(net / n) * 100, 2),
            "trades_per_hour": round(n / hours, 2) if hours else 0.0,
            "avg_price": round(sum(r["price"] for r in rows) / n, 3),
            "lanes": lanes}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", default="all")
    ap.add_argument("--k-min", type=int, default=20)
    ap.add_argument("--p-min", type=float, default=0.985)
    ap.add_argument("--price-max", type=float, default=0.94)
    ap.add_argument("--edge-min", type=float, default=0.02)
    args = ap.parse_args()
    lanes = LANES if args.lane == "all" else (args.lane,)
    out = {}
    for lane in lanes:
        path = CACHE / f"book_{lane}.npz"
        if not path.exists():
            print(f"{lane}: cache not built yet")
            continue
        out[lane] = lane_study(lane, args.k_min, args.p_min, args.price_max, args.edge_min)
    total = {"is": [], "oos": []}
    for lane, res in out.items():
        print(f"\n{lane}: {res['is']['trades']} IS, {res['oos']['trades']} OOS "
              f"({res['hours']['is']:.2f}h / {res['hours']['oos']:.2f}h)")
        for name in ("is", "oos"):
            b = res[name]
            print(f"  {name.upper():3s} n={b['trades']:3d} win={b['win_rate']:.3f} "
                  f"net=${b['net']:>8} {b['trades_per_hour']:5.1f}/hr "
                  f"avg_px={b.get('avg_price')} lanes={b.get('lanes')}")
    print(json.dumps({k: {n: out[k][n] for n in ("is", "oos")} for k in out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
