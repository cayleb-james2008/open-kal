"""Sub-second microstructure study: impulses, book staleness and lock-in outcomes.

Reads only the production tape (openkal-prod.db, read-only) and the settled-market
metadata. Every decision is made from data available at the tick, every fill is priced
with `openkal.fees` in Decimal, and the split is chronological 60/40 by window close.

    .venv/bin/python scripts/micro_study.py            # study, both slices
    .venv/bin/python scripts/micro_study.py --json     # same, machine readable
"""
from __future__ import annotations

import bisect
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openkal import features  # noqa: E402
from openkal.fees import taker_fee  # noqa: E402

DB = ROOT / "data" / "openkal-prod.db"
MARKETS = ROOT / "data" / "settled-markets.json"
LANES = ("KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M")
INDEX = {"KXBTC15M": "BRTI", "KXETH15M": "ETHUSD_RTI",
         "KXSOL15M": "SOLUSD_RTI", "KXXRP15M": "XRPUSD_RTI"}


def iso(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def connect():
    import sqlite3

    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)
    db.row_factory = sqlite3.Row
    return db


def load_index(db, index_id: str) -> dict:
    rows = db.execute(
        "SELECT ts,value,avg60,final_minute,channel FROM ws_index"
        " WHERE index_id=? AND channel='cfbenchmarks_value_5hz' ORDER BY ts", (index_id,)
    ).fetchall()
    return {"ts": np.array([r["ts"] for r in rows], dtype=np.float64),
            "value": np.array([_f(r["value"]) for r in rows], dtype=np.float64)}


def load_acc(db, index_id: str) -> dict:
    rows = db.execute(
        "SELECT ts,value,final_minute,final_ticks FROM ws_index"
        " WHERE index_id=? AND channel='cfbenchmarks_value' AND final_ticks>0 ORDER BY ts",
        (index_id,)).fetchall()
    return {"ts": np.array([r["ts"] for r in rows], dtype=np.float64),
            "value": np.array([_f(r["value"]) for r in rows], dtype=np.float64),
            "accum": np.array([_f(r["final_minute"]) for r in rows], dtype=np.float64),
            "k": np.array([float(r["final_ticks"] or 0) for r in rows], dtype=np.float64)}


def chrono_split(tickers: list[str], markets: dict[str, dict]) -> tuple[set[float], dict[str, float]]:
    """Chronological 60/40 cut over *every* settled window in the lane.

    Cutting on the windows that happened to produce candidates is selection bias:
    late-only candidates would be split against each other and labelled in-sample.
    """
    closes = sorted({markets[t]["close"] for t in tickers if t in markets})
    cut = int(len(closes) * 0.6)
    is_closes = set(closes[:cut])
    oos = [c for c in closes if c not in is_closes]
    return is_closes, {
        "is": (max(is_closes) - min(is_closes)) / 3600.0 if is_closes else 0.0,
        "oos": (max(oos) - min(oos)) / 3600.0 if oos else 0.0,
    }


def _f(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_ticks(db) -> dict[str, list[dict]]:
    rows = db.execute(
        "SELECT ts,series,ticker,yes_bid,yes_ask,last,spot,secs_left FROM ticks"
        " WHERE series IN ('KXBTC15M','KXETH15M','KXSOL15M','KXXRP15M') ORDER BY series,ts"
    ).fetchall()
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        bid, ask = _f(r["yes_bid"]), _f(r["yes_ask"])
        if not (0.0 < bid <= ask < 1.0):
            continue
        out[r["series"]].append({"ts": r["ts"], "ticker": r["ticker"], "bid": bid,
                                 "ask": ask, "last": _f(r["last"]), "spot": _f(r["spot"]),
                                 "secs": _f(r["secs_left"])})
    return out


def market_meta() -> dict[str, dict]:
    out = {}
    for m in json.loads(MARKETS.read_text()):
        if m.get("result") not in ("yes", "no") or not m.get("floor_strike"):
            continue
        try:
            out[m["ticker"]] = {"strike": float(m["floor_strike"]),
                                "close": iso(m["close_time"]),
                                "open": iso(m["open_time"]) if m.get("open_time") else 0.0,
                                "y": 1 if m["result"] == "yes" else 0}
        except (ValueError, TypeError):
            continue
    return out


def window_sigma(index: dict, meta: dict) -> float:
    lo = bisect.bisect_left(index["ts"], meta["open"])
    hi = bisect.bisect_right(index["ts"], meta["close"])
    pts = [(float(a), float(b)) for a, b in
           zip(index["ts"][lo:hi], index["value"][lo:hi]) if b > 0]
    return features.realized_sigma(pts)


def value_at(index: dict, t: float) -> float:
    j = int(np.searchsorted(index["ts"], t, "right")) - 1
    if j < 0:
        return float("nan")
    return float(index["value"][j])


def main(as_json: bool) -> int:
    db = connect()
    markets = market_meta()
    ticks = load_ticks(db)
    index = {lane: load_index(db, INDEX[lane]) for lane in LANES}
    acc = {lane: load_acc(db, INDEX[lane]) for lane in LANES}

    rows: list[dict] = []
    for lane in LANES:
        rs = [r for r in ticks[lane] if r["ticker"] in markets]
        if not rs:
            continue
        sig_by_window = {}
        for r in rs:
            m = markets[r["ticker"]]
            key = r["ticker"]
            if key not in sig_by_window:
                sig_by_window[key] = window_sigma(index[lane], m)
            r["meta"] = m
            r["close"] = m["close"]
            r["y"] = m["y"]
            r["strike"] = m["strike"]
            r["sigma"] = sig_by_window[key]
        rows.extend(rs)

    closes = sorted({r["close"] for r in rows})
    cut = int(len(closes) * 0.6)
    is_closes = set(closes[:cut])
    cut_ts = closes[cut] if cut < len(closes) else 0.0

    def slice_of(r):
        return "is" if r["close"] in is_closes else "oos"

    oos_closes = [c for c in closes if c not in is_closes]
    hours = {"is": (max(is_closes) - min(is_closes)) / 3600.0 if is_closes else 0.0,
             "oos": (max(oos_closes) - min(oos_closes)) / 3600.0 if oos_closes else 0.0}
    HOURS.update(hours)

    # ---- impulse frequency and book response ------------------------------
    events = []
    for lane in LANES:
        idx = index[lane]
        lane_rows = [r for r in rows if r["ticker"] in markets and r["secs"] > 60]
        ts_arr = np.array([r["ts"] for r in lane_rows])
        for horizon in (1.5, 5.0, 10.0, 30.0):
            j = np.searchsorted(idx["ts"], ts_arr - horizon, "right") - 1
            lag = (ts_arr - horizon) - idx["ts"][np.maximum(j, 0)]
            ok = (j >= 0) & (lag < 2.0)
            v1 = np.array([value_at(idx, t) for t in ts_arr])
            v0 = np.where(ok, idx["value"][np.maximum(j, 0)], np.nan)
            bps = (v1 / v0 - 1.0) * 10_000.0
            for i, r in enumerate(lane_rows):
                if not math.isfinite(bps[i]):
                    continue
                events.append({"lane": lane, "ts": r["ts"], "h": horizon, "bps": float(bps[i]),
                               "slice": slice_of(r), "ticker": r["ticker"]})
    freq = {}
    for h in (1.5, 5.0, 10.0, 30.0):
        for lo, hi in ((12, 20), (8, 12), (20, 40)):
            sel = [e for e in events if e["h"] == h and lo <= abs(e["bps"]) < hi]
            # dedupe events within 2s on the same lane
            ded = []
            for e in sorted(sel, key=lambda x: x["ts"]):
                if not ded or e["ts"] - ded[-1]["ts"] > 2.0 or e["lane"] != ded[-1]["lane"]:
                    ded.append(e)
            freq[f"h{h}_{lo}_{hi}"] = {
                "n": len(ded), "per_hour": round(len(ded) / (hours["is"] + hours["oos"]), 2),
                "is": sum(1 for e in ded if e["slice"] == "is"),
                "oos": sum(1 for e in ded if e["slice"] == "oos")}

    # ---- stale-book snipe -------------------------------------------------
    def snipe(*, horizon: float, min_bps: float, max_bps: float, dp_min: float,
              stale_share: float, hold_s: float, band=(0.10, 0.90),
              fill_mode: str = "taker") -> dict:
        out = []
        for lane in LANES:
            rs = [r for r in rows if r["ticker"] in markets and r["strike"] > 0]
            rs.sort(key=lambda r: r["ts"])
            ts_arr = [r["ts"] for r in rs]
            idx = index[lane]
            for i, r in enumerate(rs):
                r = r
                if not (120 < r["secs"] < 780):
                    continue
                j = int(np.searchsorted(idx["ts"], r["ts"] - horizon, "right")) - 1
                if j < 0 or (r["ts"] - horizon) - idx["ts"][j] >= 2.0:
                    continue
                v0, v1 = float(idx["value"][j]), value_at(idx, r["ts"])
                if not (v0 > 0 and v1 > 0):
                    continue
                bps = (v1 / v0 - 1.0) * 10_000.0
                mag = abs(bps)
                if mag < min_bps or mag > max_bps:
                    continue
                d = 1.0 if bps > 0 else -1.0
                k = bisect.bisect_left(ts_arr, r["ts"] - horizon) - 1
                if k < 0:
                    continue
                mid_then = (rs[k]["bid"] + rs[k]["ask"]) / 2
                mid_now = (r["bid"] + r["ask"]) / 2
                fv_now = features.fair_value(v1, r["strike"], r["secs"], r["sigma"])
                fv_then = features.fair_value(v0, r["strike"], r["secs"] + horizon, r["sigma"])
                if not (fv_now.enough_data and fv_then.enough_data):
                    continue
                dp = (fv_now.p_up - fv_then.p_up) * d
                book_move = (mid_now - mid_then) * d
                if dp < dp_min or book_move > stale_share * dp:
                    continue
                side = "YES" if d > 0 else "NO"
                price = r["ask"] if side == "YES" else 1.0 - r["bid"]
                if not (band[0] <= price <= band[1]):
                    continue
                deadline = r["ts"] + hold_s
                ex = None
                for later in rs[i + 1:]:
                    if later["ts"] > deadline:
                        break
                    spot_now = (later["spot"] / r["spot"] - 1.0) * 10_000.0 if r["spot"] > 0 else 0.0
                    if spot_now * d <= -0.6 * mag:
                        ex = later
                        break
                if ex is None:
                    j = bisect.bisect_left(ts_arr, deadline)
                    mark = rs[min(j, len(rs) - 1)]
                else:
                    mark = ex
                exit_px = mark["bid"] if side == "YES" else 1.0 - mark["ask"]
                exit_px = max(0.01, min(0.99, exit_px))
                stake = Decimal("65")
                qty = max(1, int(stake / Decimal(str(price))))
                fee_in = taker_fee(Decimal(str(price)), qty) / qty
                fee_out = taker_fee(Decimal(str(exit_px)), qty) / qty
                pnl = (Decimal(str(exit_px)) - Decimal(str(price))) - fee_in - fee_out
                out.append({"lane": lane, "ts": r["ts"], "close": r["close"],
                            "side": side, "price": price, "exit": exit_px,
                            "pnl": float(pnl), "slice": slice_of(r), "y": r["y"],
                            "why": "thesis" if ex else "barrier", "qty": qty})
        return split_block(out)

    snipe_grid = {}
    for horizon in (1.5, 5.0, 10.0, 30.0):
        for stale in (0.5, 0.9):
            snipe_grid[f"h{horizon}_stale{stale}"] = snipe(
                horizon=horizon, min_bps=12, max_bps=20, dp_min=0.05,
                stale_share=stale, hold_s=40.0)

    # ---- final minute lock-in --------------------------------------------
    def lockin(*, k_min: int = 20, p_min: float = 0.985, price_max: float = 0.94,
               edge_min: float = 0.02) -> dict:
        out = []
        for lane in LANES:
            a = acc[lane]
            rs = [r for r in rows if r["ticker"] in markets and r["strike"] > 0]
            rs.sort(key=lambda r: r["ts"])
            ts_arr = np.array([r["ts"] for r in rs])
            picks = {}
            for j in range(len(a["ts"])):
                if not math.isfinite(a["accum"][j]) or a["k"][j] < k_min:
                    continue
                nxt = int(np.searchsorted(ts_arr, a["ts"][j]))
                best = None
                for cand in (nxt - 1, nxt):
                    if 0 <= cand < len(rs) and abs(rs[cand]["ts"] - a["ts"][j]) <= 0.8:
                        if best is None or abs(rs[cand]["ts"] - a["ts"][j]) < abs(best["ts"] - a["ts"][j]):
                            best = rs[cand]
                if best is None:
                    continue
                r = best
                if not (0 < r["secs"] <= 75):
                    continue
                sig = features.final_minute(spot=float(a["value"][j]), strike=r["strike"],
                                            locked_avg=float(a["accum"][j]),
                                            locked_ticks=int(a["k"][j]),
                                            sigma_per_sqrt_s=r["sigma"], seconds_left=r["secs"])
                if sig is None or sig.p_up < p_min:
                    continue
                mid = (r["bid"] + r["ask"]) / 2
                side = "YES" if sig.p_up >= mid else "NO"
                price = Decimal(str(r["ask"])) if side == "YES" else Decimal(1) - Decimal(str(r["bid"]))
                if price > Decimal(str(price_max)):
                    continue
                p_side = sig.p_up if side == "YES" else 1 - sig.p_up
                stake = Decimal("65")
                qty = max(1, int(stake / price))
                fee = taker_fee(price, qty) / qty
                if Decimal(str(p_side)) - price - fee < Decimal(str(edge_min)):
                    continue
                won = (side == "YES" and r["y"] == 1) or (side == "NO" and r["y"] == 0)
                pnl = (Decimal(1) if won else Decimal(0)) - price - fee
                key = r["ticker"]
                if key not in picks:
                    picks[key] = {"lane": lane, "ts": r["ts"], "close": r["close"],
                                  "side": side, "price": float(price), "p": sig.p_up,
                                  "pnl": float(pnl), "slice": slice_of(r), "won": won,
                                  "qty": qty}
            out.extend(picks.values())
        return split_block(out)

    lock = lockin()
    lock_by_lane = {}
    for lane in LANES:
        pass
    payload = {"windows": len(closes), "is_windows": cut, "oos_windows": len(closes) - cut,
               "cut_ts": cut_ts, "hours": {k: round(v, 2) for k, v in hours.items()},
               "impulse_freq": freq, "snipe": snipe_grid, "lockin": lock}
    if as_json:
        print(json.dumps(payload, indent=2, default=str))
        return 0
    print(json.dumps({k: payload[k] for k in ("windows", "is_windows", "oos_windows", "hours")},
                     indent=2))
    print("\nimpulse frequency (12-20bp):")
    for key, v in freq.items():
        if key.startswith("h") and "_12_20" in key:
            print(f"  {key:16s} n={v['n']:4d} {v['per_hour']:5.1f}/hr  is={v['is']} oos={v['oos']}")
    print("\nsnipe (12-20bp, dp>=0.05, stale book):")
    for key, v in snipe_grid.items():
        b = v
        print(f"  {key:18s} IS n={b['is']['trades']:3d} net={b['is']['net']:>9} "
              f"{b['is']['trades_per_hour']:5.1f}/hr | OOS n={b['oos']['trades']:3d} "
              f"net={b['oos']['net']:>9} {b['oos']['trades_per_hour']:5.1f}/hr")
    print("\nlock-in (k>=20, p>=0.985, <=0.94):")
    for name in ("is", "oos"):
        b = lock[name]
        print(f"  {name.upper():3s} n={b['trades']:3d} win={b['win_rate']:.3f} "
              f"net={b['net']:>9} {b['trades_per_hour']:5.1f}/hr fees={b['fees']} "
              f"lanes={b.get('lanes')}")
    return 0


HOURS: dict[str, float] = {"is": 0.0, "oos": 0.0}


def split_block(rows: list[dict]) -> dict:
    out = {}
    for name in ("is", "oos"):
        sel = [r for r in rows if r["slice"] == name]
        n = len(sel)
        net = sum((Decimal(str(r["pnl"])) for r in sel), Decimal("0"))
        wins = sum(1 for r in sel if r["pnl"] > 0)
        out[name] = {"trades": n, "wins": wins,
                     "win_rate": round(wins / n, 4) if n else 0.0,
                     "net": str(net.quantize(Decimal("0.0001"))),
                     "avg_c": round(float(net / n) * 100, 2) if n else 0.0,
                     "trades_per_hour": round(n / HOURS[name], 2) if HOURS.get(name) else 0.0,
                     "fees": str(sum((Decimal(str(r.get("fee", 0))) for r in sel),
                                     Decimal("0")).quantize(Decimal("0.0001"))),
                     "lanes": dict(sorted({lane: sum(1 for r in sel if r["lane"] == lane)
                                           for lane in {r["lane"] for r in sel}}.items()))}
    return out


if __name__ == "__main__":
    raise SystemExit(main("--json" in sys.argv))
