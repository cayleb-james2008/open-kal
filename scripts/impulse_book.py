"""Spot-impulse snipe measured against the venue's real ladder.

Entry is priced at the `ws_book` ask that existed at the impulse print; the take
profit only fills when the ladder's bid actually reaches it; the reversal stop uses
the settlement index, not the ladder. No 1 Hz tick quote is trusted.

    .venv/bin/python scripts/impulse_book.py --lane KXBTC15M --lookback 5
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
from scripts.micro_study import INDEX, LANES, chrono_split, market_meta  # noqa: E402

CACHE = ROOT / "data" / "evolve-cache"
STAKE = Decimal("65")


def _sigma(index: dict, close: float) -> float:
    lo = int(np.searchsorted(index["ts"], close - 150.0))
    hi = int(np.searchsorted(index["ts"], close))
    return features.realized_sigma([(float(a), float(b)) for a, b in
                                    zip(index["ts"][lo:hi], index["value"][lo:hi]) if b > 0])


def study(lane: str, *, lookback: float, min_bps: float, max_bps: float,
          stale_share: float, edge_min: float, hold_s: float, tp_c: float,
          reversal: float) -> dict:
    book = dict(np.load(CACHE / f"book_{lane}.npz"))
    names = [str(x) for x in book["tickers"]]
    markets = market_meta()
    index = dict(np.load(CACHE / f"index_{INDEX[lane]}.npz"))
    print(f"{lane}: {len(index['ts']):,} index prints, {len(book['ts']):,} book rows")

    rows = []
    opportunities = 0
    for tid, name in enumerate(names):
        m = markets.get(name)
        if m is None:
            continue
        mask = book["tid"] == tid
        bts = book["ts"][mask]
        bbid = book["bid"][mask].astype(np.float64)
        bask = book["ask"][mask].astype(np.float64)
        if len(bts) < 10:
            continue
        sig = _sigma(index, m["close"])
        if sig <= 0:
            continue
        lo = int(np.searchsorted(index["ts"], m["open"]))
        hi = int(np.searchsorted(index["ts"], m["close"] - 80))
        its = index["ts"][lo:hi]
        ival = index["value"][lo:hi]
        for k in range(3, len(its)):
            secs = m["close"] - float(its[k])
            if secs < 90:
                break
            t = float(its[k])
            j0 = int(np.searchsorted(its, t - lookback, "right")) - 1
            if j0 < 0 or (t - lookback) - its[j0] > 1.0:
                continue
            v0, v1 = float(ival[j0]), float(ival[k])
            if v0 <= 0 or v1 <= 0:
                continue
            bps = (v1 / v0 - 1.0) * 10_000.0
            mag = abs(bps)
            if mag < min_bps or mag > max_bps:
                continue
            direction = 1.0 if bps > 0 else -1.0
            bi = int(np.searchsorted(bts, t, "right")) - 1
            bi0 = int(np.searchsorted(bts, t - lookback, "right")) - 1
            if bi < 0 or bi0 < 0 or t - bts[bi] > 1.0 or t - lookback - bts[bi0] > 1.0:
                continue
            bid, ask = float(bbid[bi]), float(bask[bi])
            bid0 = float(bbid[bi0])
            if not (0 < bid <= ask < 1) or not (0 < bid0 < 1):
                continue
            fv_now = features.fair_value(v1, m["strike"], secs, sig)
            fv_then = features.fair_value(v0, m["strike"], secs + (t - float(its[j0])), sig)
            if not (fv_now.enough_data and fv_then.enough_data):
                continue
            implied = (fv_now.p_up - fv_then.p_up) * direction
            moved = (bid - bid0) * direction
            if implied <= 0 or moved > stale_share * implied:
                continue
            side = "YES" if direction > 0 else "NO"
            price = Decimal(str(ask)) if side == "YES" else Decimal(1) - Decimal(str(bid))
            if not (Decimal("0.15") <= price <= Decimal("0.90")):
                continue
            qty = max(1, int(STAKE / price))
            fee_in = taker_fee(price, qty) / qty
            p_side = fv_now.p_up if side == "YES" else 1 - fv_now.p_up
            edge = Decimal(str(p_side)) - price - fee_in
            if edge < Decimal(str(edge_min)):
                continue
            opportunities += 1
            tp = price + Decimal(str(tp_c))
            deadline = t + hold_s
            hi_b = int(np.searchsorted(bts, deadline, "right"))
            exit_px = None
            why = "barrier"
            for q in range(bi + 1, hi_b):
                sbid = float(bbid[q]) if side == "YES" else 1.0 - float(bask[q])
                if sbid >= float(tp):
                    exit_px = float(tp)
                    why = "maker_tp"
                    break
                qs = int(np.searchsorted(its, bts[q], "right")) - 1
                if qs < 0:
                    continue
                if (float(ival[qs]) / v1 - 1.0) * 10_000.0 * direction <= -reversal * mag:
                    exit_px = sbid
                    why = "reversal"
                    break
            if exit_px is None:
                if hi_b - 1 <= bi:
                    continue
                q = hi_b - 1
                exit_px = float(bbid[q]) if side == "YES" else 1.0 - float(bask[q])
            exit_px = max(0.01, min(0.99, exit_px))
            fee_out = Decimal("0") if why == "maker_tp" else taker_fee(Decimal(str(exit_px)), qty) / qty
            pnl = (Decimal(str(exit_px)) - price) - fee_in - fee_out
            rows.append({"lane": lane, "ticker": name, "ts": t, "close": m["close"],
                         "side": side, "price": float(price), "exit": exit_px,
                         "pnl": float(pnl), "why": why, "edge": float(edge),
                         "secs": secs, "bps": bps})
    is_closes, hours = chrono_split(names, markets)
    out = {"opportunities": opportunities, "hours": hours, "rows": rows}
    for name in ("is", "oos"):
        sel = [r for r in rows if (r["close"] in is_closes) == (name == "is")]
        n = len(sel)
        net = sum((Decimal(str(r["pnl"])) for r in sel), Decimal("0"))
        wins = sum(1 for r in sel if r["pnl"] > 0)
        out[name] = {"trades": n, "wins": wins,
                     "win_rate": round(wins / n, 4) if n else 0.0,
                     "net": str(net.quantize(Decimal("0.01"))),
                     "avg_c": round(float(net / n) * 100, 2) if n else 0.0,
                     "trades_per_hour": round(n / hours[name], 2) if hours[name] else 0.0,
                     "tp_share": round(sum(1 for r in sel if r["why"] == "maker_tp") / n, 2) if n else 0.0}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", default="all")
    ap.add_argument("--lookback", type=float, default=5.0)
    ap.add_argument("--min-bps", type=float, default=12.0)
    ap.add_argument("--max-bps", type=float, default=20.0)
    ap.add_argument("--stale", type=float, default=0.5)
    ap.add_argument("--edge", type=float, default=0.025)
    ap.add_argument("--hold", type=float, default=40.0)
    ap.add_argument("--tp", type=float, default=0.05)
    ap.add_argument("--reversal", type=float, default=0.6)
    args = ap.parse_args()
    lanes = LANES if args.lane == "all" else (args.lane,)
    out = {}
    for lane in lanes:
        if not (CACHE / f"book_{lane}.npz").exists():
            print(f"{lane}: cache not built yet")
            continue
        out[lane] = study(lane, lookback=args.lookback, min_bps=args.min_bps,
                          max_bps=args.max_bps, stale_share=args.stale,
                          edge_min=args.edge, hold_s=args.hold, tp_c=args.tp,
                          reversal=args.reversal)
        r = out[lane]
        print(f"  opportunities={r['opportunities']} "
              f"IS n={r['is']['trades']} net=${r['is']['net']} {r['is']['trades_per_hour']}/hr "
              f"| OOS n={r['oos']['trades']} net=${r['oos']['net']} "
              f"{r['oos']['trades_per_hour']}/hr win={r['oos']['win_rate']} "
              f"tp={r['oos']['tp_share']}")
    print(json.dumps({k: {n: out[k][n] for n in ("opportunities", "is", "oos", "hours")}
                      for k in out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
