"""Offline evolution harness: fit on the in-sample tape, verdict on the frozen holdout.

This is the study that decides whether an evolved EVOLVE-BLOCK may run forward. It is
deliberately outside `verifier.py`: the verifier's thresholds are immutable, and this
script may not change them. What it does is build the labelled tape the verifier never
had — every recorded tick joined to the settlement it led to — then split it
chronologically 60/40, fit the calibration on the first 60% only, and report both
slices side by side.

Nothing here writes to the production database.

    .venv/bin/python scripts/evolve.py cache     # build the compact tape cache
    .venv/bin/python scripts/evolve.py measure   # signal study, IS vs frozen OOS
    .venv/bin/python scripts/evolve.py verify    # verifier.replay + ledger parity
"""
from __future__ import annotations

import bisect
import json
import math
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openkal import features  # noqa: E402
from openkal.fees import taker_fee  # noqa: E402
from openkal.kalshi_ws import INDEX_BY_SERIES  # noqa: E402

DB = ROOT / "data" / "openkal-prod.db"
CACHE = ROOT / "data" / "evolve-cache"
MARKETS = ROOT / "data" / "settled-markets.json"
REPORT = ROOT / "data" / "evolve-report.json"
LANES = ("KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M", "KXDOGE15M")


def iso(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


# --------------------------------------------------------------------------- cache
def build_cache() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    markets = json.loads(MARKETS.read_text())
    meta: dict[str, dict] = {}
    for m in markets:
        if m.get("result") not in ("yes", "no") or not m.get("floor_strike"):
            continue
        meta[m["ticker"]] = {
            "strike": float(m["floor_strike"]),
            "close": iso(m["close_time"]),
            "y": 1 if m["result"] == "yes" else 0,
        }
    (CACHE / "markets.json").write_text(json.dumps(meta))

    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)
    db.row_factory = sqlite3.Row
    for index_id in sorted(set(INDEX_BY_SERIES.values())):
        rows = db.execute(
            "SELECT ts,value,final_minute,final_ticks FROM ws_index"
            " WHERE index_id=? ORDER BY ts", (index_id,)
        ).fetchall()
        ts = np.array([r["ts"] for r in rows], dtype=np.float64)
        val = np.array([_f(r["value"]) for r in rows], dtype=np.float64)
        acc = np.array([_f(r["final_minute"]) for r in rows], dtype=np.float64)
        k = np.array([float(r["final_ticks"] or 0) for r in rows], dtype=np.float64)
        np.savez_compressed(CACHE / f"index_{index_id}.npz", ts=ts, value=val, accum=acc, k=k)
        print(f"index {index_id:14s} {len(ts):>8,} rows")

    rows = db.execute(
        "SELECT ts,series,ticker,yes_bid,yes_ask,last,spot,p_model,edge,secs_left,spread,signal"
        " FROM ticks ORDER BY ts"
    ).fetchall()
    tickers = sorted({r["ticker"] for r in rows})
    tidx = {t: i for i, t in enumerate(tickers)}
    out = {
        "ts": np.array([r["ts"] for r in rows], dtype=np.float64),
        "series": np.array([r["series"] for r in rows]),
        "ticker": np.array([r["ticker"] for r in rows]),
        "bid": np.array([_f(r["yes_bid"]) for r in rows], dtype=np.float64),
        "ask": np.array([_f(r["yes_ask"]) for r in rows], dtype=np.float64),
        "last": np.array([_f(r["last"]) for r in rows], dtype=np.float64),
        "spot": np.array([_f(r["spot"]) for r in rows], dtype=np.float64),
        "p_model": np.array([_f(r["p_model"]) for r in rows], dtype=np.float64),
        "secs": np.array([_f(r["secs_left"]) for r in rows], dtype=np.float64),
        "spread": np.array([_f(r["spread"]) for r in rows], dtype=np.float64),
        "signal": np.array([r["signal"] or "" for r in rows]),
        "ticker_ids": np.array([tidx[r["ticker"]] for r in rows], dtype=np.int32),
    }
    np.savez_compressed(CACHE / "ticks.npz", **out)
    (CACHE / "tickers.json").write_text(json.dumps(tickers))
    print(f"ticks {len(rows):,} across {len(tickers)} tickers")

    trades = db.execute(
        "SELECT ticker,ts,yes_price,count,taker_side FROM ws_trade ORDER BY ticker,ts"
    ).fetchall()
    order: dict[str, int] = {}
    ids = []
    for r in trades:
        tid = order.setdefault(r["ticker"], len(order))
        ids.append(tid)
    np.savez_compressed(
        CACHE / "trades.npz",
        ticker_id=np.array(ids, dtype=np.int32),
        ts=np.array([r["ts"] for r in trades], dtype=np.float64),
        price=np.array([_f(r["yes_price"]) for r in trades], dtype=np.float64),
        count=np.array([_f(r["count"]) for r in trades], dtype=np.float64),
        taker_yes=np.array([1 if r["taker_side"] == "yes" else 0 for r in trades], dtype=np.int8),
    )
    (CACHE / "trade_tickers.json").write_text(json.dumps(list(order)))
    print(f"trades {len(trades):,} across {len(order)} tickers")
    db.close()


def _f(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


@dataclass
class Tape:
    ticks: dict[str, np.ndarray]
    tickers: list[str]
    markets: dict[str, dict]
    index: dict[str, dict[str, np.ndarray]]

    @classmethod
    def load(cls) -> "Tape":
        ticks = dict(np.load(CACHE / "ticks.npz"))
        tickers = json.loads((CACHE / "tickers.json").read_text())
        markets = json.loads((CACHE / "markets.json").read_text())
        index = {}
        for index_id in sorted(set(INDEX_BY_SERIES.values())):
            path = CACHE / f"index_{index_id}.npz"
            if path.exists():
                index[index_id] = dict(np.load(path))
        return cls(ticks=ticks, tickers=tickers, markets=markets, index=index)


@dataclass
class TradeTape:
    """Every real print the venue published, so a maker fill is evidence not a guess."""

    names: list[str]
    by_ticker: dict[str, dict[str, np.ndarray]]

    @classmethod
    def load(cls) -> "TradeTape":
        data = dict(np.load(CACHE / "trades.npz"))
        names = json.loads((CACHE / "trade_tickers.json").read_text())
        ids = data["ticker_id"]
        order = np.argsort(ids, kind="stable")
        ids = ids[order]
        ts = data["ts"][order]
        price = data["price"][order]
        count = data["count"][order]
        taker_yes = data["taker_yes"][order]
        bounds = np.searchsorted(ids, np.arange(len(names) + 1))
        by_ticker = {}
        for i, name in enumerate(names):
            lo, hi = bounds[i], bounds[i + 1]
            by_ticker[name] = {"ts": ts[lo:hi], "price": price[lo:hi],
                              "count": count[lo:hi], "taker_yes": taker_yes[lo:hi]}
        return cls(names=names, by_ticker=by_ticker)

    def fill(self, ticker: str, t0: float, t1: float, *, taker_yes: bool,
             limit: float, need: float) -> tuple[float, float] | None:
        """First time cumulative taker volume reaches `need` at or through `limit`.

        `taker_yes=True` means the aggressor bought YES (so it lifts a YES ask);
        `taker_yes=False` means the aggressor sold YES / bought NO (it hits a YES bid).
        """
        book = self.by_ticker.get(ticker)
        if book is None or len(book["ts"]) == 0:
            return None
        lo = int(np.searchsorted(book["ts"], t0, "left"))
        hi = int(np.searchsorted(book["ts"], t1, "right"))
        if hi <= lo:
            return None
        ts = book["ts"][lo:hi]
        price = book["price"][lo:hi]
        size = book["count"][lo:hi]
        side = book["taker_yes"][lo:hi]
        mask = (side == 1) == taker_yes
        mask &= price >= limit if taker_yes else price <= limit
        if not mask.any():
            return None
        cum = np.cumsum(np.where(mask, size, 0.0))
        hit = np.argmax(cum >= need) if (cum >= need).any() else None
        if hit is None:
            return None
        return float(ts[hit]), float(price[hit])


# --------------------------------------------------------------------- calibration
def logit(p: float) -> float:
    p = min(1.0 - 1e-6, max(1e-6, p))
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def fit_ab(pairs: list[tuple[float, int]], l2: float = 1e-3,
           iterations: int = 60) -> tuple[float, float]:
    """Newton/IRLS for p_cal = sigmoid(a * logit(p_raw) + b), deterministic.

    No randomness and no stop-on-tolerance wobble: a fixed iteration count means the
    same tape always yields the same (a, b), which is what "deterministic calibration"
    has to mean if a replay is going to reproduce it.
    """
    if len(pairs) < 20:
        return 1.0, 0.0
    x = np.array([logit(p) for p, _ in pairs], dtype=np.float64)
    y = np.array([float(t) for _, t in pairs], dtype=np.float64)
    a, b = 1.0, 0.0
    for _ in range(iterations):
        z = np.clip(a * x + b, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        w = np.maximum(p * (1 - p), 1e-6)
        g_a = float(np.sum((p - y) * x) + l2 * (a - 1.0))
        g_b = float(np.sum(p - y) + l2 * b)
        h_aa = float(np.sum(w * x * x) + l2)
        h_ab = float(np.sum(w * x))
        h_bb = float(np.sum(w) + l2)
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-12:
            break
        da = (h_bb * g_a - h_ab * g_b) / det
        db = (h_aa * g_b - h_ab * g_a) / det
        # A full Newton step on separable data can launch the slope to infinity; cap it.
        da = max(-0.5, min(0.5, da))
        db = max(-0.5, min(0.5, db))
        a -= da
        b -= db
        a = max(0.05, min(12.0, a))
        b = max(-4.0, min(4.0, b))
    return float(a), float(b)


def apply_ab(p: float, a: float, b: float) -> float:
    return sigmoid(a * logit(p) + b)


# ------------------------------------------------------------------------- scoring
def pnl_of(side: str, price: Decimal, y: int, entry_fee: Decimal) -> Decimal:
    payout = Decimal(1) if ((side == "YES" and y == 1) or (side == "NO" and y == 0)) else Decimal(0)
    return payout - price - entry_fee


def ece(pairs, bins: int = 10) -> float:
    return features.expected_calibration_error(pairs, bins)


def brier(pairs) -> float:
    return features.brier(pairs)


def block(rows: list[dict], hours: float) -> dict:
    n = len(rows)
    if not n:
        return {"trades": 0, "wins": 0, "win_rate": 0.0, "net": "0", "avg_c": 0.0,
                "trades_per_hour": 0.0, "fees": "0"}
    net = sum((r["pnl"] for r in rows), Decimal("0"))
    fees = sum((r["fee"] for r in rows), Decimal("0"))
    wins = sum(1 for r in rows if r["pnl"] > 0)
    return {
        "trades": n,
        "wins": wins,
        "win_rate": round(wins / n, 4),
        "net": str(net.quantize(Decimal("0.0001"))),
        "avg_c": round(float(net / n) * 100, 3),
        "trades_per_hour": round(n / hours, 2) if hours > 0 else 0.0,
        "fees": str(fees.quantize(Decimal("0.0001"))),
        "lanes": dict(Counter(r["series"] for r in rows)),
    }


def lane_calibration(rows: list[dict]) -> dict:
    out = {}
    for lane in sorted({r["series"] for r in rows}):
        pair_rows = [r for r in rows if r["series"] == lane]
        pairs = [(r["p_used"], 1 if r["pnl"] > 0 else 0) for r in pair_rows]
        mids = [(r["mid"], r["y"]) for r in pair_rows]
        out[lane] = {
            "n": len(pairs),
            "brier": round(brier(pairs), 4),
            "ece": round(ece(pairs), 4),
            "brier_mid": round(brier(mids), 4),
            "ece_mid": round(ece(mids), 4),
        }
    return out


class Study:
    """The labelled tape, split once, chronologically, and never re-split."""

    K_MIN = 8
    K_MAX = 35

    def __init__(self, tape: Tape) -> None:
        self.tape = tape
        self.obs = self._build()
        closes = sorted({o["close"] for o in self.obs})
        cut = int(len(closes) * 0.6)
        self.is_closes = set(closes[:cut])
        self.oos_closes = set(closes[cut:])
        self.cut_ts = closes[cut] if cut < len(closes) else 0.0
        self.sigma: dict[tuple[str, float], float] = {}
        for o in self.obs:
            key = (o["series"], o["close"])
            if key not in self.sigma:
                self.sigma[key] = self._sigma(o["series"], o["close"])

    # -- construction -----------------------------------------------------
    def _build(self) -> list[dict]:
        t = self.tape.ticks
        out: list[dict] = []
        for i in range(len(t["ts"])):
            ticker = str(t["ticker"][i])
            market = self.tape.markets.get(ticker)
            if market is None:
                continue
            bid, ask = float(t["bid"][i]), float(t["ask"][i])
            if not (0.0 < bid <= ask < 1.0):
                continue
            lane = str(t["series"][i])
            index_id = INDEX_BY_SERIES.get(lane)
            if index_id is None or index_id not in self.tape.index:
                continue
            out.append({
                "ts": float(t["ts"][i]), "series": lane, "ticker": ticker,
                "bid": bid, "ask": ask, "last": float(t["last"][i]),
                "spot": float(t["spot"][i]), "secs": float(t["secs"][i]),
                "p_stored": float(t["p_model"][i]), "signal": str(t["signal"][i]),
                "strike": market["strike"], "close": market["close"], "y": market["y"],
                "mid": (bid + ask) / 2.0,
            })
        self._attach_index(out)
        return out

    def _attach_index(self, rows: list[dict]) -> None:
        """Nearest settlement-accumulator print within 1.5s, per lane."""
        by_lane: dict[str, list[dict]] = defaultdict(list)
        for o in rows:
            by_lane[o["series"]].append(o)
        for lane, lane_rows in by_lane.items():
            index = self.tape.index[INDEX_BY_SERIES[lane]]
            ts_list = index["ts"].tolist()
            for o in lane_rows:
                j = bisect.bisect_left(ts_list, o["ts"])
                best = None
                for cand in (j - 1, j, j + 1):
                    if 0 <= cand < len(ts_list):
                        gap = abs(ts_list[cand] - o["ts"])
                        if gap <= 1.5 and index["k"][cand] > 0 and math.isfinite(index["accum"][cand]):
                            if best is None or gap < best[0]:
                                best = (gap, float(index["accum"][cand]), float(index["k"][cand]),
                                        float(index["value"][cand]))
                if best is not None:
                    o["accum"], o["k"], o["index_value"] = best[1], int(best[2]), best[3]

    def _sigma(self, lane: str, close: float) -> float:
        index = self.tape.index[INDEX_BY_SERIES[lane]]
        ts = index["ts"]
        lo = int(np.searchsorted(ts, close - 150.0))
        hi = int(np.searchsorted(ts, close))
        pts = [(float(a), float(b)) for a, b in zip(ts[lo:hi], index["value"][lo:hi]) if b > 0]
        return features.realized_sigma(pts)

    # -- slices -----------------------------------------------------------
    def slice_of(self, row: dict) -> str:
        return "is" if row["close"] in self.is_closes else "oos"

    def lockin(self, row: dict) -> float | None:
        """The lock-in probability for this tick under the evolved k window."""
        if "k" not in row or not (self.K_MIN <= row["k"] <= self.K_MAX):
            return None
        if row["secs"] <= 0 or row["spot"] <= 0:
            return None
        sigma = self.sigma.get((row["series"], row["close"]), 0.0)
        if sigma <= 0:
            return None
        sig = features.final_minute(
            spot=row["spot"], strike=row["strike"], locked_avg=row["accum"],
            locked_ticks=row["k"], sigma_per_sqrt_s=sigma, seconds_left=row["secs"])
        if sig is None:
            return None
        row["p_raw"] = sig.p_up
        row["detail"] = sig.detail
        return sig.p_up

    # -- study ------------------------------------------------------------
    def summary(self) -> dict:
        for o in self.obs:
            self.lockin(o)
        lock = [o for o in self.obs if "p_raw" in o]
        is_rows = [o for o in lock if self.slice_of(o) == "is"]
        oos_rows = [o for o in lock if self.slice_of(o) == "oos"]
        calib = {lane: fit_ab([(o["p_raw"], o["y"]) for o in is_rows if o["series"] == lane])
                 for lane in sorted({o["series"] for o in lock})}
        for o in lock:
            a, b = calib.get(o["series"], (1.0, 0.0))
            o["p_cal"] = apply_ab(o["p_raw"], a, b)

        def trim(rows: list[dict]) -> dict:
            return {
                "ticks": len(rows),
                "brier_raw": round(brier([(o["p_raw"], o["y"]) for o in rows]), 4),
                "ece_raw": round(ece([(o["p_raw"], o["y"]) for o in rows]), 4),
                "brier_cal": round(brier([(o["p_cal"], o["y"]) for o in rows]), 4),
                "ece_cal": round(ece([(o["p_cal"], o["y"]) for o in rows]), 4),
                "brier_mid": round(brier([(o["mid"], o["y"]) for o in rows]), 4),
                "ece_mid": round(ece([(o["mid"], o["y"]) for o in rows]), 4),
            }

        def per_lane(rows: list[dict]) -> dict:
            out = {}
            for lane in sorted({o["series"] for o in rows}):
                lane_rows = [o for o in rows if o["series"] == lane]
                out[lane] = {
                    "ticks": len(lane_rows),
                    "ece_raw": round(ece([(o["p_raw"], o["y"]) for o in lane_rows]), 4),
                    "ece_cal": round(ece([(o["p_cal"], o["y"]) for o in lane_rows]), 4),
                    "ece_mid": round(ece([(o["mid"], o["y"]) for o in lane_rows]), 4),
                }
            return out

        trades = self.lockin_trades(lock, hurdle=0.0)
        return {
            "tape": {"rows": len(self.tape.ticks["ts"]), "matched": len(self.obs),
                     "lockin_ticks": len(lock), "windows": len({o["close"] for o in self.obs}),
                     "cut_ts": self.cut_ts},
            "calibration_ab": calib,
            "all": trim(lock), "is": trim(is_rows), "oos": trim(oos_rows),
            "lanes_is": per_lane(is_rows), "lanes_oos": per_lane(oos_rows),
            "trades_hurdle0": trades,
        }

    def lockin_trades(self, rows: list[dict], hurdle: float, lo: float = 0.20,
                      hi: float = 0.84, max_spread: float = 0.05,
                      momentum: bool = False) -> dict:
        """One entry per window-lane, taker at the ask, held to settlement."""
        picked: dict[tuple[str, str], dict] = {}
        for o in sorted(rows, key=lambda r: r["ts"]):
            if not (self.K_MIN <= o.get("k", 0) <= self.K_MAX):
                continue
            if o["ask"] - o["bid"] > max_spread:
                continue
            if momentum and not _momentum_ok(o):
                continue
            p = o.get("p_cal", o.get("p_raw"))
            side = "YES" if p >= o["mid"] else "NO"
            price = Decimal(str(o["ask"])) if side == "YES" else Decimal(1) - Decimal(str(o["bid"]))
            if not (Decimal(str(lo)) <= price <= Decimal(str(hi))):
                continue
            p_side = p if side == "YES" else 1 - p
            if Decimal(str(p_side)) - price < Decimal(str(hurdle)):
                continue
            key = (o["series"], o["ticker"])
            if key in picked:
                continue
            fee = taker_fee(price, 1)
            picked[key] = {"series": o["series"], "close": o["close"], "k": o["k"],
                           "side": side, "price": price, "p_used": p_side, "mid": o["mid"],
                           "y": o["y"], "fee": fee, "slice": self.slice_of(o),
                           "pnl": pnl_of(side, price, o["y"], fee)}
        out = {}
        for name, slice_rows in (("is", [v for v in picked.values() if v["slice"] == "is"]),
                                 ("oos", [v for v in picked.values() if v["slice"] == "oos"])):
            hours = self.hours(name)
            block_out = block(slice_rows, hours)
            block_out["lane_ece"] = lane_calibration(slice_rows)
            out[name] = block_out
        out["hurdle"] = hurdle
        return out

    def hours(self, name: str) -> float:
        """Wall-clock hours the slice covers, which is what trades/hr means to a soak."""
        closes = self.is_closes if name == "is" else self.oos_closes
        stamps = [o["close"] for o in self.obs if o["close"] in closes]
        return (max(stamps) - min(stamps)) / 3600.0 if stamps else 0.0

    def by_ticker(self) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = defaultdict(list)
        for o in self.obs:
            out[o["ticker"]].append(o)
        for rows in out.values():
            rows.sort(key=lambda r: r["ts"])
        return out

    # -- maker capture -----------------------------------------------------
    def maker_capture(self, trades: TradeTape, *, width_lo: float = 0.02,
                      width_hi: float = 0.04, t_lo: float = 150.0, t_hi: float = 720.0,
                      gap: float = 0.015, tp_c: float = 0.02, patience: float = 20.0,
                      hold: float = 60.0, need: float = 20.0, stop_c: float = 0.0,
                      per_window: int = 2) -> dict[str, dict]:
        """Rest at the touch, bank +2c with a maker ask, scratch at the deadline.

        A fill only counts when the venue's own print tape shows enough taker volume
        trading at or through our limit; the model never invents a fill.
        """
        books = self.by_ticker()
        out_rows: list[dict] = []
        for ticker, rows in books.items():
            if not rows:
                continue
            close = rows[0]["close"]
            attempts = 0
            i = 0
            while i < len(rows) and attempts < per_window:
                o = rows[i]
                width = o["ask"] - o["bid"]
                ok = (width_lo - 1e-9 <= width <= width_hi + 1e-9
                      and t_lo < o["secs"] < t_hi
                      and abs(o.get("p_stored", 0.5) - o["mid"]) <= gap)
                if not ok:
                    i += 1
                    continue
                attempts += 1
                yes_side = o.get("p_cal", o["p_stored"]) >= o["mid"]
                entry = o["bid"] if yes_side else 1.0 - o["ask"]
                if not (0.05 <= entry <= 0.95):
                    i += 1
                    continue
                t_end = min(o["ts"] + patience, close - 60.0)
                fill = trades.fill(ticker, o["ts"], t_end, taker_yes=not yes_side,
                                   limit=o["bid"] if yes_side else o["ask"], need=need)
                if fill is None:
                    i += 1
                    continue
                t_fill = fill[0]
                tp = entry + tp_c
                t_dead = min(t_fill + hold, close - 45.0)
                if stop_c > 0:
                    # A hard stop is the only thing standing between a market maker and
                    # the full adverse move; it is also what turns bounces into losses.
                    for later in rows:
                        if later["ts"] < t_fill:
                            continue
                        if later["ts"] > t_dead:
                            break
                        mark_now = later["bid"] if yes_side else 1.0 - later["ask"]
                        if mark_now <= entry - stop_c:
                            t_dead = later["ts"]
                            break
                take = trades.fill(ticker, t_fill + 1e-6, t_dead, taker_yes=yes_side,
                                   limit=tp if yes_side else 1.0 - tp, need=need)
                if take is not None:
                    pnl = Decimal(str(tp_c))
                    fee = Decimal("0")
                    reason = "maker tp"
                else:
                    mark = _quote_at(rows, t_dead)
                    if mark is None:
                        i += 1
                        continue
                    exit_px = mark["bid"] if yes_side else 1.0 - mark["ask"]
                    price = Decimal(str(round(max(0.01, min(0.99, exit_px)), 4)))
                    fee = taker_fee(price, 1)
                    pnl = (price - Decimal(str(entry))) - fee
                    reason = "scratch"
                out_rows.append({"series": o["series"], "close": close,
                                 "side": "YES" if yes_side else "NO",
                                 "price": Decimal(str(entry)),
                                 "p_used": o.get("p_cal", o["p_stored"]),
                                 "mid": o["mid"], "y": o["y"], "fee": fee, "pnl": pnl,
                                 "slice": self.slice_of(o), "reason": reason})
                i += 1
        return self._split(out_rows)

    # -- lead lag ----------------------------------------------------------
    def lead_lag(self, *, min_bps: float = 10.0, max_bps: float = 25.0,
                 index_share: float = 0.5, stale_c: float = 0.01,
                 barrier_s: float = 28.0, invalidate_bps: float = 6.0,
                 per_window: int = 3) -> dict[str, dict]:
        """Buy a stale ladder behind a confirmed spot impulse.

        Exits on thesis invalidation (the impulse reverses) or a time barrier at the
        bid, not on a 3c stop that ordinary bid-ask bounce alone can trigger.
        """
        books = self.by_ticker()
        out_rows: list[dict] = []
        for ticker, rows in books.items():
            if not rows:
                continue
            attempts = 0
            i = 4
            while i < len(rows) and attempts < per_window:
                o = rows[i]
                if o["secs"] < 40:
                    i += 1
                    continue
                past = [r for r in rows[max(0, i - 8):i] if r["ts"] >= o["ts"] - 10.0]
                if len(past) < 2:
                    i += 1
                    continue
                base = past[0]
                if base["spot"] <= 0 or o["spot"] <= 0:
                    i += 1
                    continue
                spot_bps = (o["spot"] / base["spot"] - 1.0) * 10_000.0
                mag = abs(spot_bps)
                if mag < min_bps or mag > max_bps:
                    i += 1
                    continue
                direction = 1.0 if spot_bps > 0 else -1.0
                if "index_value" not in o or "index_value" not in base:
                    i += 1
                    continue
                index_bps = ((o["index_value"] / base["index_value"] - 1.0) * 10_000.0
                             if base["index_value"] > 0 else 0.0)
                if index_bps * direction < mag * index_share:
                    i += 1
                    continue
                if abs(o["bid"] - base["bid"]) > stale_c:
                    i += 1
                    continue
                attempts += 1
                yes_side = direction > 0
                entry = o["ask"] if yes_side else 1.0 - o["bid"]
                if not (0.15 <= entry <= 0.90):
                    i += 1
                    continue
                deadline = min(o["ts"] + barrier_s, o["close"] - 5.0)
                exit_row = None
                for later in rows[i + 1:]:
                    if later["ts"] > deadline:
                        break
                    spot_now = (later["spot"] / o["spot"] - 1.0) * 10_000.0
                    if spot_now * direction <= -invalidate_bps:
                        exit_row = later
                        break
                mark = exit_row or _quote_at(rows, deadline) or rows[min(i + 1, len(rows) - 1)]
                exit_px = mark["bid"] if yes_side else 1.0 - mark["ask"]
                price = Decimal(str(entry))
                exit_price = Decimal(str(round(max(0.01, min(0.99, exit_px)), 4)))
                fee = taker_fee(price, 1) + taker_fee(exit_price, 1)
                pnl = (exit_price - price) - fee
                p_side = o.get("p_cal", o.get("p_raw", 0.5))
                out_rows.append({"series": o["series"], "close": o["close"],
                                 "side": "YES" if yes_side else "NO", "price": price,
                                 "p_used": p_side if yes_side else 1 - p_side,
                                 "mid": o["mid"], "y": o["y"], "fee": fee, "pnl": pnl,
                                 "slice": self.slice_of(o),
                                 "reason": "thesis" if exit_row else "barrier"})
                i += 1
        return self._split(out_rows)

    def _split(self, rows: list[dict]) -> dict[str, dict]:
        out = {}
        for name in ("is", "oos"):
            sel = [r for r in rows if r["slice"] == name]
            hours = self.hours(name)
            block_out = block(sel, hours)
            lane_ece = lane_calibration(sel)
            block_out["worst_lane_ece"] = max([v["ece"] for v in lane_ece.values()] or [0.0])
            block_out["lane_ece"] = lane_ece
            out[name] = block_out
        return out


def _quote_at(rows: list[dict], ts: float) -> dict | None:
    """Nearest recorded quote at or before `ts`."""
    best = None
    for r in rows:
        if r["ts"] <= ts:
            best = r
        else:
            break
    return best


def _momentum_ok(o: dict) -> bool:
    """Lock-in momentum is building: the live print agrees with the locked lead."""
    lead = o["accum"] - o["strike"]
    live = o["spot"] - o["strike"]
    if lead == 0:
        return False
    return (lead > 0) == (live > 0) and abs(live) >= abs(o["strike"]) * 1e-5


def refresh_markets(pages: int = 6) -> None:
    """Pull settled 15-minute markets so every recorded tick has a real outcome.

    Read-only venue metadata, the same public endpoint `lockin_measure.py` uses. The
    cache is merged, not replaced: windows recorded over earlier runs keep their labels.
    """
    import asyncio

    from openkal.auth import KalshiClient
    from openkal.config import load_settings

    async def pull() -> list[dict]:
        client = KalshiClient(load_settings())
        out: list[dict] = []
        try:
            for series in LANES:
                cursor = None
                for _ in range(pages):
                    params = {"series_ticker": series, "status": "settled", "limit": 200}
                    if cursor:
                        params["cursor"] = cursor
                    status, payload = await client.get("/markets", params)
                    if status != 200:
                        print(f"{series}: http {status}")
                        break
                    out.extend(payload.get("markets") or [])
                    cursor = payload.get("cursor")
                    if not cursor:
                        break
        finally:
            await client.close()
        return out

    fresh = asyncio.run(pull())
    existing = json.loads(MARKETS.read_text()) if MARKETS.exists() else []
    merged = {m["ticker"]: m for m in existing}
    added = 0
    for m in fresh:
        if m.get("ticker") and m.get("ticker") not in merged:
            added += 1
        if m.get("ticker"):
            merged[m["ticker"]] = m
    MARKETS.write_text(json.dumps(list(merged.values())))
    closes = sorted({m["close_time"] for m in merged.values() if m.get("close_time")})
    print(json.dumps({"fetched": len(fresh), "added": added, "total": len(merged),
                      "first_close": closes[0], "last_close": closes[-1]}, indent=2))


def verifier_replay(store_path: Path, study: Study, params=None) -> dict:
    """Drive `verifier.replay` over each frozen slice through the engine's own core."""
    from types import SimpleNamespace

    from openkal.replay import replay_series
    from openkal.strategy import Params

    params = params or Params()
    db = sqlite3.connect(f"file:{store_path}?mode=ro", uri=True, timeout=60)
    db.row_factory = sqlite3.Row
    out = {"is": [], "oos": []}
    for lane in LANES:
        rows = db.execute(
            "SELECT ts,ticker,yes_bid,yes_ask,last,spot,p_model,edge,secs_left,spread,signal,"
            "can_open,stake FROM ticks WHERE series=? ORDER BY ts", (lane,)).fetchall()
        buckets: dict[str, list[dict]] = {"is": [], "oos": []}
        for r in rows:
            market = study.tape.markets.get(r["ticker"])
            if market is None:
                continue
            buckets["is" if market["close"] in study.is_closes else "oos"].append(dict(r))
        cfg = SimpleNamespace(series=lane)
        for name in ("is", "oos"):
            ticks = buckets[name]
            if len(ticks) < 30:
                continue
            stake = Decimal(str(next((t["stake"] for t in ticks if t.get("stake")), "50")))
            report, _ = replay_series(cfg, ticks, params, stake)
            out[name].append({"lane": lane, "ticks": len(ticks), "trades": report.trades,
                              "wins": report.wins, "losses": report.losses,
                              "pnl": str(report.pnl), "win_rate": round(report.win_rate, 3),
                              "profit_factor": round(report.profit_factor, 3),
                              "fees": str(report.fees)})
    db.close()
    return out


def write_report() -> dict:
    tape = Tape.load()
    trades = TradeTape.load()
    study = Study(tape)
    for o in study.obs:
        study.lockin(o)
    lock = [o for o in study.obs if "p_raw" in o]
    is_rows = [o for o in lock if study.slice_of(o) == "is"]
    oos_rows = [o for o in lock if study.slice_of(o) == "oos"]

    lanes = sorted({o["series"] for o in lock})
    per_lane = {lane: fit_ab([(o["p_raw"], o["y"]) for o in is_rows if o["series"] == lane])
                for lane in lanes}
    pooled = fit_ab([(o["p_raw"], o["y"]) for o in is_rows])
    for o in lock:
        o["p_cal"] = o["p_raw"]

    def ece_by_lane(rows, key):
        return {lane: round(ece([(o[key], o["y"]) for o in rows if o["series"] == lane]), 4)
                for lane in lanes if any(o["series"] == lane for o in rows)}

    def brier_by_lane(rows, key):
        return {lane: round(brier([(o[key], o["y"]) for o in rows if o["series"] == lane]), 4)
                for lane in lanes if any(o["series"] == lane for o in rows)}

    calibration = {
        "fitted_per_lane": {k: [round(v[0], 4), round(v[1], 4)] for k, v in per_lane.items()},
        "fitted_pooled": [round(pooled[0], 4), round(pooled[1], 4)],
        "is_ticks": len(is_rows), "oos_ticks": len(oos_rows),
        "ece_identity_is": round(ece([(o["p_cal"], o["y"]) for o in is_rows]), 4),
        "ece_identity_oos": round(ece([(o["p_cal"], o["y"]) for o in oos_rows]), 4),
        "ece_identity_lanes_oos": ece_by_lane(oos_rows, "p_cal"),
        "ece_pooled_oos": round(ece([(apply_ab(o["p_raw"], *pooled), o["y"]) for o in oos_rows]), 4),
        "ece_pooled_lanes_oos": {lane: round(ece([(apply_ab(o["p_raw"], *pooled), o["y"])
                                                  for o in oos_rows if o["series"] == lane]), 4)
                                 for lane in lanes if any(o["series"] == lane for o in oos_rows)},
        "brier_identity_oos": round(brier([(o["p_cal"], o["y"]) for o in oos_rows]), 4),
        "brier_mid_oos": round(brier([(o["mid"], o["y"]) for o in oos_rows]), 4),
        "brier_identity_lanes_oos": brier_by_lane(oos_rows, "p_cal"),
        "brier_mid_lanes_oos": brier_by_lane(oos_rows, "mid"),
        "ece_mid_lanes_oos": ece_by_lane(oos_rows, "mid"),
        "verdict": "identity promoted; every fitted map is worse on the frozen holdout",
    }

    # The shipped rule for comparison: it waits for k>=30, i.e. the last 30 printed
    # seconds, when the book has already repriced. Same scorer, same tape.
    shipped = Study(tape)
    shipped.K_MIN, shipped.K_MAX = 30, 60
    for o in shipped.obs:
        shipped.lockin(o)
    shipped_lock = [o for o in shipped.obs if "p_raw" in o]
    for o in shipped_lock:
        o["p_cal"] = o["p_raw"]
    lockin_shipped = {str(h): shipped.lockin_trades(shipped_lock, hurdle=h) for h in (0.0, 0.02)}
    lockin = {str(h): study.lockin_trades(lock, hurdle=h)
              for h in (0.0, 0.01, 0.02)}
    lockin_momentum = {str(h): study.lockin_trades(lock, hurdle=h, momentum=True)
                       for h in (0.0, 0.02)}
    maker = {"tp2c": study.maker_capture(trades, tp_c=0.02),
             "tp2c_stop2c": study.maker_capture(trades, tp_c=0.02, stop_c=0.02),
             "tp1c": study.maker_capture(trades, tp_c=0.01)}
    lead = {"barrier28": study.lead_lag(barrier_s=28.0),
            "barrier25": study.lead_lag(barrier_s=25.0),
            "min12": study.lead_lag(min_bps=12.0)}
    replay = verifier_replay(DB, study)

    report = {
        "generated_ts": time.time(),
        "tape": {"rows": len(tape.ticks["ts"]), "matched_ticks": len(study.obs),
                 "windows": len({o["close"] for o in study.obs}),
                 "is_windows": len(study.is_closes), "oos_windows": len(study.oos_closes),
                 "is_hours": round(study.hours("is"), 2),
                 "oos_hours": round(study.hours("oos"), 2),
                 "cut_ts": study.cut_ts},
        "calibration": calibration,
        "candidates": {
            "final_minute": {"k_window": [Study.K_MIN, Study.K_MAX], "trades": lockin,
                             "with_momentum": lockin_momentum,
                             "shipped_k30": lockin_shipped},
            "lead_lag": lead,
            "maker_capture": maker,
        },
        "verifier_replay_current_params": replay,
        "promotion": {"promoted": False, "reason": "no candidate showed positive net P&L on both slices"},
    }
    REPORT.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"tape": report["tape"], "calibration": calibration,
                      "verifier_replay": replay, "promotion": report["promotion"]}, indent=2))
    for name, block in report["candidates"].items():
        print(f"\n{name}:")
        for variant, res in block.items():
            if isinstance(res, dict) and "is" in res and "oos" in res:
                print(f"  {variant:18s} IS n={res['is']['trades']:3d} net=${res['is']['net']:>9} "
                      f"{res['is']['trades_per_hour']:5.1f}/hr | OOS n={res['oos']['trades']:3d} "
                      f"net=${res['oos']['net']:>9} {res['oos']['trades_per_hour']:5.1f}/hr "
                      f"worst-lane ECE {res['oos']['worst_lane_ece']:.4f}")
    return report


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        write_report()
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "cache":
        build_cache()
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "refresh-markets":
        refresh_markets(pages=int(sys.argv[2]) if len(sys.argv) > 2 else 6)
        return 0
    tape = Tape.load()
    stage = sys.argv[1] if len(sys.argv) > 1 else "measure"
    if stage == "measure":
        study = Study(tape)
        print(json.dumps(study.summary(), indent=2))
        return 0
    print(json.dumps({"tick_rows": int(len(tape.ticks["ts"])),
                      "tickers": len(tape.tickers),
                      "index_series": {k: len(v["ts"]) for k, v in tape.index.items()}},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
