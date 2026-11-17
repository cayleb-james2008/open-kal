"""Bounded-risk Frozen 60/40 OOS discovery sweep.

Four mechanism families, one strict fill model, one frozen out-of-sample slice.

Physics enforced here (not configurable, so a variant cannot cheat):

  * every entry costs <= $0.62 per contract for the side held;
  * a book row with `best_bid >= best_ask` is skipped, never traded;
  * a passive leg (entry, +4c/+5c take-profit, scratch) counts only when >= 50
    contracts print at/through its limit -- a 1-lot touch is not liquidity;
  * a confirmed 2s stop leaves at the book on the tick the confirmation completes
    and pays the exact Kalshi taker fee `ceil(0.07 * C * P * (1 - P))`.

Candidate detection is pooled per family so the grid is cheap: one pass over the
tape builds the feature rows, then every variant is a numpy mask over that pool.
Selection is strictly 60% in-sample; the top IS candidate per family is scored
once on the frozen 40% OOS slice and promoted or rejected.

Usage:
    python -m scripts.bounded_sweep
    python -m scripts.bounded_sweep --lanes KXBTC15M --families F1
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openkal import features  # noqa: E402
from scripts import alpha_search as A  # noqa: E402
from scripts.alpha_discover import (CACHE, EPS, LANES, MIN_PASSIVE_QTY,  # noqa: E402
                                    Lane)

CAP = 0.62                 # hard price cap, per contract, side held
MIN_FILL = 50.0            # contracts through a passive limit
STAKE = Decimal("32.67")   # one live slot
FAMILIES = ("F1", "F2", "F3", "F4")
COLS = ("tid", "ts", "close", "y", "bid", "ask", "spot", "delta", "secs",
        "p_up", "z", "flow", "rev", "shift", "locked")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------------ pool
def new_pool() -> dict:
    return {k: [] for k in COLS} | {"lane": []}


def pool_add(pool: dict, lane_name: str, cols: dict, mask: np.ndarray) -> None:
    n = int(mask.sum())
    if n == 0:
        return
    for k in COLS:
        pool[k].append(np.asarray(cols[k])[mask])
    pool["lane"].append(np.full(n, lane_name, dtype=object))


def pool_finish(pool: dict) -> dict | None:
    if not pool["tid"]:
        return None
    return {k: np.concatenate(v) for k, v in pool.items()}


def pick(pool: dict, mask: np.ndarray, *, cooldown: float,
         per_window: int) -> list[int]:
    """First qualifying row per window, then one per cooldown, capped per window."""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    tid = pool["tid"][idx]
    ts = pool["ts"][idx]
    out: list[int] = []
    lo = 0
    while lo < len(idx):
        hi = int(np.searchsorted(tid, tid[lo], "right"))
        if hi == lo:
            hi = lo + 1
        sel = A._dedupe([int(i) for i in idx[lo:hi]], pool["ts"], cooldown=cooldown,
                      per_ticker=per_window)
        out.extend(sel)
        lo = hi
    return out


def side_cost(side_yes: np.ndarray, bid: np.ndarray, ask: np.ndarray,
              style: str) -> np.ndarray:
    if style == "taker":
        return np.where(side_yes, ask, 1.0 - bid)
    return np.where(side_yes, bid, 1.0 - ask)


def limit_for(side_yes: np.ndarray, bid: np.ndarray, ask: np.ndarray,
              style: str) -> np.ndarray:
    if style == "taker":
        return np.where(side_yes, ask, 1.0 - bid)
    return np.where(side_yes, bid, 1.0 - ask)


# --------------------------------------------------------------- pool builders
def f1_pool(lane: Lane, sig: dict) -> dict:
    """Final-minute lock-in rows: T<=55s, with k seconds of the average printed."""
    pool = new_pool()
    index = lane.index
    if not index:
        return pool
    its = np.asarray(index["ts"], dtype=np.float64)
    ival = np.asarray(index["value"], dtype=np.float64)
    for tid, name, meta, a, b in A._segments(lane):
        close = float(meta["close"])
        strike = float(meta["strike"])
        ts = lane.ts[a:b]
        bid = lane.bid[a:b]
        ask = lane.ask[a:b]
        secs = close - ts
        n = len(ts)
        valid = (bid > 0) & (ask < 1) & (bid < ask)
        lo = int(np.searchsorted(its, close - 60.0, "left"))
        hi = int(np.searchsorted(its, close, "right"))
        t = its[lo:hi]
        v = ival[lo:hi]
        good = np.isfinite(v) & (v > 0)
        t, v = t[good], v[good]
        if len(t) < 100:
            continue
        cs = np.concatenate(([0.0], np.cumsum(v)))
        k = np.searchsorted(t, ts, "right")
        kk = np.clip(k, 1, len(v))
        avg = cs[kk] / kk
        locked = np.clip(60.0 - secs, 0.0, 60.0)
        remaining = np.maximum(0.0, 60.0 - locked)
        spot = A.index_at(lane, ts)
        sigma = sig[tid]
        thr = (60.0 * strike - locked * avg) / np.maximum(remaining, 1e-6)
        # sd of the *average* of a Brownian path over the remaining T seconds is
        # sigma_price * sqrt(T/3). `features.final_minute` uses
        # sqrt(T) * (T/60)**1.5 instead, which is ~10x too tight at T=25s; the
        # sweep tests the mechanism with the correct variance.
        sd = spot * sigma * np.sqrt(np.maximum(remaining, 1e-6) / 3.0)
        ok = (valid & (secs <= 55.0) & (secs >= 6.0) & (k >= 20) & (remaining > 0)
              & np.isfinite(spot) & (spot > 0) & (sigma > 0) & (sd > 0))
        z = np.where(ok, (thr - spot) / np.where(sd > 0, sd, np.inf), 0.0)
        p_up = np.zeros(n)
        if ok.any():
            p_up[ok] = np.array([0.5 * math.erfc(float(v) / math.sqrt(2.0))
                                 for v in z[ok]])
        cols = {
            "tid": np.full(n, tid, dtype=np.int32), "ts": ts, "close": np.full(n, close),
            "y": np.full(n, int(meta["y"]), dtype=np.int8), "bid": bid, "ask": ask,
            "spot": spot, "delta": np.abs(spot - strike), "secs": secs,
            "p_up": p_up, "z": z, "flow": np.zeros(n), "rev": np.zeros(n),
            "shift": np.zeros(n), "locked": locked,
        }
        pool_add(pool, lane.lane, cols, ok)
    return pool


def obi_pool(lane: Lane, sig: dict, *, t_lo: float, t_hi: float,
             delta_bps_floor: float, spot_past: bool = False) -> dict:
    """Touch-book rows with level z, printed flow, revision imbalance, 5s mid shift.

    `spot_past=True` reads the settlement index backward only (no print stamped
    after the row); it is what a forward score must use. Research slices keep the
    nearest-print read they were fitted on.
    """
    pool = new_pool()
    for tid, name, meta, a, b in A._segments(lane):
        ts = lane.ts[a:b]
        bid = lane.bid[a:b]
        ask = lane.ask[a:b]
        close = float(meta["close"])
        strike = float(meta["strike"])
        secs = close - ts
        n = len(ts)
        valid = ((bid > 0) & (ask < 1) & (bid < ask) & ((ask - bid) <= 0.04 + EPS)
                 & (secs >= t_lo) & (secs <= t_hi))
        spot = (A.index_at_past(lane, ts) if spot_past else A.index_at(lane, ts))
        sigma = sig[tid]
        vol = sigma * np.sqrt(np.maximum(secs, 1e-9))
        delta = np.abs(spot - strike)
        ok = (valid & np.isfinite(spot) & (spot > 0) & (vol > 0)
              & (delta >= delta_bps_floor * 1e-4 * np.maximum(spot, 1.0)))
        if not ok.any():
            continue
        z = np.zeros(n)
        z[ok] = np.log(spot[ok] / strike) / vol[ok]
        fair_z = np.zeros(n)
        fair_z[ok] = np.log(spot[ok] / strike) / vol[ok]
        p_up = 0.5 * (1.0 + np.tanh(0.7978845608 * (fair_z + 0.044715 * fair_z ** 3)))
        flow = A.flow_imbalance(lane, tid, ts, 10.0)
        rev = A.revision_imbalance(lane, a, b, 5.0)
        mid = (bid + ask) / 2.0
        j = np.searchsorted(ts, ts - 5.0, "right") - 1
        shift = np.zeros(n)
        good = j >= 0
        shift[good] = mid[good] - mid[j[good]]
        # Pool floor: at least one side must be inside the deepest cost band any
        # variant may use, so the pool never depends on one variant's thresholds.
        c_yes = np.minimum(bid, ask)
        c_no = np.minimum(1.0 - ask, 1.0 - bid)
        ok &= (np.minimum(c_yes, c_no) >= 0.20) & (np.minimum(c_yes, c_no) <= 0.66)
        cols = {
            "tid": np.full(n, tid, dtype=np.int32), "ts": ts, "close": np.full(n, close),
            "y": np.full(n, int(meta["y"]), dtype=np.int8), "bid": bid, "ask": ask,
            "spot": spot, "delta": delta, "secs": secs, "p_up": p_up, "z": z,
            "flow": flow, "rev": rev, "shift": shift, "locked": np.zeros(n),
        }
        pool_add(pool, lane.lane, cols, ok)
    return pool


def lead_pool(lane: Lane, btc: dict, sig: dict, *, lookback: float) -> dict:
    """Rows where BTC's index has moved over `lookback` and the alt book has not."""
    pool = new_pool()
    for tid, name, meta, a, b in A._segments(lane):
        ts = lane.ts[a:b]
        bid = lane.bid[a:b]
        ask = lane.ask[a:b]
        secs = meta["close"] - ts
        n = len(ts)
        spread = ask - bid
        valid = ((bid > 0) & (ask < 1) & (bid < ask) & (spread <= 0.04 + EPS)
                 & (secs >= 120.0) & (secs <= 840.0))
        if valid.sum() < 10:
            continue
        btc_now = A.series_at(btc, ts)
        btc_then = A.series_at(btc, ts - lookback)
        alt_now = A.index_at(lane, ts)
        alt_then = A.index_at(lane, ts - lookback)
        bps = np.full(n, np.nan)
        alt_bps = np.full(n, np.nan)
        ok_b = np.isfinite(btc_now) & np.isfinite(btc_then) & (btc_then > 0)
        ok_a = np.isfinite(alt_now) & np.isfinite(alt_then) & (alt_then > 0)
        bps[ok_b] = (btc_now[ok_b] / btc_then[ok_b] - 1.0) * 10_000.0
        alt_bps[ok_a] = (alt_now[ok_a] / alt_then[ok_a] - 1.0) * 10_000.0
        mid = (bid + ask) / 2.0
        j = np.searchsorted(ts, ts - lookback, "right") - 1
        shift = np.zeros(n)
        good = j >= 0
        shift[good] = mid[good] - mid[j[good]]
        sigma = sig[tid]
        vol = sigma * np.sqrt(np.maximum(secs, 1e-9))
        fair_z = np.zeros(n)
        can = ok_a & (alt_now > 0) & (vol > 0)
        fair_z[can] = np.log(alt_now[can] / meta["strike"]) / vol[can]
        p_up = 0.5 * (1.0 + np.tanh(0.7978845608 * (fair_z + 0.044715 * fair_z ** 3)))
        ok = valid & ok_b & ok_a & (np.abs(bps) >= 4.0) & (np.abs(bps) <= 50.0)
        cols = {
            "tid": np.full(n, tid, dtype=np.int32), "ts": ts,
            "close": np.full(n, float(meta["close"])),
            "y": np.full(n, int(meta["y"]), dtype=np.int8), "bid": bid, "ask": ask,
            "spot": alt_now, "delta": np.abs(alt_now - meta["strike"]), "secs": secs,
            "p_up": p_up, "z": bps, "flow": alt_bps, "rev": np.zeros(n),
            "shift": shift, "locked": np.full(n, lookback),
        }
        pool_add(pool, lane.lane, cols, ok)
    return pool


# ----------------------------------------------------------------- the variants
@dataclass
class Variant:
    family: str
    name: str
    params: A.SimParams
    pool: str                 # "f1" | "f2" | "f4" | "lead"
    lookback: float = 0.0     # lead pool only
    mask: object = None       # callable(pool) -> (mask, side_yes, style)


def f1_variants() -> list[Variant]:
    """The lock-in model's edge over the price actually paid, not a raw threshold.

    A `p_up >= 0.6` rule finds nothing under a 0.62 cap: once 35s of the settlement
    average are printed the book is already at 0.97-0.99, so the only rows left in
    the price band are the genuinely undecided ones. The test is whether the model
    knows more than the book where the book is still cheap.
    """
    out = []
    for min_locked in (35, 45, 52):
        for edge_min in (0.03, 0.06, 0.10):
            for style in ("maker", "taker"):
                def make(ml=min_locked, em=edge_min, st=style):
                    def mask(pool):
                        if st == "taker":
                            cy, cn = pool["ask"], 1.0 - pool["bid"]
                        else:
                            cy, cn = pool["bid"], 1.0 - pool["ask"]
                        ey = pool["p_up"] - cy
                        en = (1.0 - pool["p_up"]) - cn
                        ey = np.where(cy <= CAP + EPS, ey, -9.0)
                        en = np.where(cn <= CAP + EPS, en, -9.0)
                        ly = ey >= en
                        edge = np.where(ly, ey, en)
                        cost = np.where(ly, cy, cn)
                        m = ((pool["locked"] >= ml) & (edge >= em)
                             & (cost >= 0.10) & (cost <= CAP + EPS))
                        return m, ly, st
                    return mask
                out.append(Variant(
                    "F1", f"F1 lock>={min_locked}s edge>={edge_min:.2f} {style}",
                    A.SimParams(tp_c=0.0, barrier_s=0.0,
                                exit_mode="hold_to_settlement",
                                max_entry_price=CAP, min_fill_qty=MIN_FILL),
                    "f1", mask=make()))
    return out


def f2_variants() -> list[Variant]:
    """Spot-vs-strike impulse confirmed by printed flow and touch revisions."""
    out = []
    for z_min in (0.3, 0.5):
        for flow_min in (0.10, 0.25, 0.40):
            for rev_min in (0.0, 0.10):
                for delta_bps in (3.0, 6.0):
                    for tp in (0.04, 0.05):
                        for style in ("maker", "taker"):
                            def make(zm=z_min, fm=flow_min, rm=rev_min, db=delta_bps,
                                     st=style):
                                def mask(pool):
                                    z = pool["z"]
                                    d = np.sign(z)
                                    ly = z > 0
                                    cost = side_cost(ly, pool["bid"], pool["ask"], st)
                                    m = ((np.abs(z) >= zm) & (pool["flow"] * d >= fm)
                                         & (pool["rev"] * d >= rm)
                                         & (pool["delta"] >= db * 1e-4 * pool["spot"])
                                         & (cost >= 0.25 - EPS)
                                         & (cost <= 0.60 + EPS))
                                    return m, ly, st
                                return mask
                            out.append(Variant(
                                "F2", f"F2 z>={z_min} flow>={flow_min} rev>={rev_min} "
                                      f"d>={delta_bps:.0f}bp tp{tp:.2f} {style}",
                                A.SimParams(tp_c=tp, barrier_s=90.0, stop_c=0.07,
                                            stop_confirm_s=2.0, max_entry_price=CAP,
                                            min_fill_qty=MIN_FILL),
                                "f2", mask=make()))
    return out


def f3_variants() -> list[Variant]:
    out = []
    # A >=8bp BTC impulse in 3s leaves ETH in the 0.28-0.62 band in ONE window of
    # 57: the alt book has already repriced. 3-5bp is where the lag is testable.
    for bps_lo in (3.0, 5.0):
        for lookback in (3.0, 5.0):
            for max_reprice in (0.05, 0.10):
                def make(bl=bps_lo, lb=lookback, rp=max_reprice):
                    def mask(pool):
                        bps = pool["z"]
                        alt = pool["flow"]
                        ly = bps > 0
                        m = (((bps >= bl) & (alt > 0) & (alt <= 0.6 * bps)
                              & (pool["shift"] <= rp))
                             | ((bps <= -bl) & (alt < 0) & (alt >= 0.6 * bps)
                                & (pool["shift"] >= -rp)))
                        return m, ly, "maker"
                    return mask
                out.append(Variant(
                    "F3", f"F3 btc>={bps_lo:.0f}bp lb{lookback:.0f}s "
                          f"reprice<={max_reprice:.2f}",
                    A.SimParams(tp_c=0.04, barrier_s=90.0, stop_c=0.07,
                                stop_confirm_s=2.0, max_entry_price=CAP,
                                min_fill_qty=MIN_FILL),
                    "lead", lookback=lookback, mask=make()))
    return out


def f4_variants() -> list[Variant]:
    """Touch absorption: one-sided prints, a touch that did not move, fade it."""
    out = []
    for absorb in (0.5, 0.7):
        for max_shift in (0.02, 0.04):
            for delta_bps in (3.0, 6.0):
                for tp in (0.04, 0.05):
                    for style in ("maker", "taker"):
                        def make(ab=absorb, ms=max_shift, db=delta_bps, st=style):
                            def mask(pool):
                                f = pool["flow"]
                                ly = f < 0
                                cost = side_cost(ly, pool["bid"], pool["ask"], st)
                                m = ((np.abs(f) >= ab)
                                     & (pool["rev"] * np.sign(f) <= 0.6)
                                     & (np.abs(pool["shift"]) <= ms)
                                     & (pool["delta"] >= db * 1e-4 * pool["spot"])
                                     & (cost >= 0.25 - EPS)
                                     & (cost <= 0.60 + EPS))
                                return m, ly, st
                            return mask
                        out.append(Variant(
                            "F4", f"F4 absorb>={absorb} shift<={max_shift:.2f} "
                                  f"d>={delta_bps:.0f}bp tp{tp:.2f} {style}",
                            A.SimParams(tp_c=tp, barrier_s=90.0, stop_c=0.07,
                                        stop_confirm_s=2.0, max_entry_price=CAP,
                                        min_fill_qty=MIN_FILL),
                            "f4", mask=make()))
    return out


def build_variants(families: tuple[str, ...]) -> list[Variant]:
    out: list[Variant] = []
    if "F1" in families:
        out += f1_variants()
    if "F2" in families:
        out += f2_variants()
    if "F3" in families:
        out += f3_variants()
    if "F4" in families:
        out += f4_variants()
    return out


def events_for(pool: dict, variant: Variant, *, slice_name: str,
               lag: float = 1.0, lane: Lane | None = None) -> list[A.Event]:
    """The events a variant would send, priced when the order would arrive.

    The order cannot arrive before the decision. One second later the book is a
    different book, so the fill price is the quote at `trigger + lag`, not the quote
    that triggered the signal: a taker pays the ask that is there when it crosses,
    and a maker rests at the bid that is there when it is posted. The lag used to be
    applied to the timestamp of family F3 only, which left every other family
    filling at the price it had already seen -- the single most optimistic way to
    price an order.
    """
    mask, ly, style = variant.mask(pool)
    mask = mask & (pool["slice"] == slice_name)
    idx = pick(pool, mask, cooldown=F1_COOLDOWN if variant.family == "F1" else
               (120.0 if variant.family == "F4" else 90.0),
               per_window=1 if variant.family == "F1" else
               (2 if variant.family == "F4" else 3))
    out: list[A.Event] = []
    for i in idx:
        t = float(pool["ts"][i])
        close = float(pool["close"][i])
        if variant.family == "F3":
            t += lag
            if close - t < 60.0:
                continue
        b = float(pool["bid"][i])
        a = float(pool["ask"][i])
        if lane is not None and lag > 0.0:
            quote = lane.quote_at(int(pool["tid"][i]), t + lag, valid_only=True,
                                  lookback=2.0)
            if quote is None:
                continue
            b, a = quote
        if not (0.0 < b < a < 1.0):
            continue
        long_yes = bool(ly[i])
        if variant.family == "F3":
            limit = b if long_yes else a
            cost = limit if long_yes else 1.0 - limit
            if not (0.28 - EPS <= cost <= 0.62 + EPS):
                continue
        elif style == "maker":
            limit = b if long_yes else 1.0 - a
        else:
            limit = a if long_yes else 1.0 - b
        p_up = float(pool["p_up"][i])
        out.append(A.Event(
            variant.family, variant.name, tid=int(pool["tid"][i]),
            ticker=str(pool["ticker"][i]), ts=t, close=close,
            y=int(pool["y"][i]), slice=slice_name,
            side="YES" if long_yes else "NO", entry_style=style, limit=limit,
            p_side=(p_up if long_yes else 1.0 - p_up)))
    return out


F1_COOLDOWN = 900.0

# The probability a variant reports is a separate thing from the rule that trades.
# The F2 family ships a tanh approximation of the normal CDF that was never fitted
# to anything, so its calibration error measures the approximation, not the rule.
# The repo's protocol is a Platt fit on the in-sample rows, frozen for the
# out-of-sample slice; `--calibrate` turns that on for the F2 pool.
CALIB: dict = {"a": 1.0, "b": 0.0, "fitted": False}


def _norm_cdf(x):
    import math as _m
    return 0.5 * (1.0 + _m.erf(float(x) / _m.sqrt(2.0)))


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1.0 - p))


def fit_trade_calibration(rows: list[dict]) -> dict:
    """Fit p = norm_cdf(A * logit(p_side) + B) on the candidate's in-sample trades.

    The repo's protocol is "Platt fitted on IS rows only, frozen for OOS", and the
    rows that matter are the candidate's own trades: the calibration error of a
    candidate is measured on the trades it took, and the population the fit sees must
    be that one. Fitting on the whole pool instead describes a population the rule
    never traded and can be much steeper than the trades support.
    """
    pairs = []
    for r in rows:
        p = float(r["p_side"])
        if not (0.0 < p < 1.0):
            continue
        won = 1 if ((int(r["y"]) == 1) == (r["side"] == "YES")) else 0
        pairs.append((_logit(p), won))
    if len(pairs) < 20:
        return {"a": 1.0, "b": 0.0, "fitted": False, "rows": len(pairs)}
    z = np.array([p[0] for p in pairs], dtype=np.float64)
    y = np.array([p[1] for p in pairs], dtype=np.float64)

    def nll(a: float, b: float) -> float:
        p = np.clip(0.5 * (1.0 + np.vectorize(_norm_cdf)(a * z + b)), 1e-6, 1 - 1e-6)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    best = (1.0, 0.0, nll(1.0, 0.0))
    for a in np.arange(0.2, 3.01, 0.1):
        for b in np.arange(-2.0, 2.01, 0.1):
            v = nll(float(a), float(b))
            if v < best[2]:
                best = (float(a), float(b), v)
    a, b = best[0], best[1]
    for step_size in (0.05, 0.02, 0.01):
        improved = True
        while improved:
            improved = False
            for da in np.arange(-0.3, 0.301, step_size):
                for db in np.arange(-0.3, 0.301, step_size):
                    v = nll(a + float(da), b + float(db))
                    if v < best[2] - 1e-12:
                        best = (a + float(da), b + float(db), v)
                        a, b = best[0], best[1]
                        improved = True
    return {"a": a, "b": b, "fitted": True, "rows": len(pairs)}


def apply_trade_calibration(rows: list[dict], cal: dict) -> list[dict]:
    """Rewrite each trade's reported probability with the frozen calibration."""
    if not cal.get("fitted"):
        return rows
    for r in rows:
        p = float(r["p_side"])
        if not (0.0 < p < 1.0):
            continue
        r["p_side_raw"] = p
        r["p_side"] = _norm_cdf(cal["a"] * _logit(p) + cal["b"])
    return rows


def fit_calibration(lanes: tuple[str, ...], variants: list[Variant], btc: dict) -> dict:
    """Fit p = norm_cdf(a*z + b) on the in-sample rows of the F2 pool, and freeze it."""
    z_is: list[np.ndarray] = []
    y_is: list[np.ndarray] = []
    for lane_name in lanes:
        lane = Lane(lane_name)
        sig = A.trailing_sigma(lane)
        parts = pools_for(lane, variants, sig, btc)
        pool = parts.get("f2")
        if pool is None:
            continue
        sel = np.array([lane.slice_of(float(c)) == "is" for c in pool["close"]])
        if sel.any():
            z_is.append(pool["z"][sel])
            y_is.append(pool["y"][sel])
    if not z_is:
        return {"a": 1.0, "b": 0.0, "fitted": False, "rows": 0}
    z = np.concatenate(z_is).astype(np.float64)
    y = np.concatenate(y_is).astype(np.float64)
    if len(z) > 40000:
        step = len(z) // 40000
        z, y = z[::step][:40000], y[::step][:40000]

    def nll(a: float, b: float) -> float:
        p = np.clip(0.5 * (1.0 + np.vectorize(_norm_cdf)(a * z + b)), 1e-6, 1 - 1e-6)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    best = (1.0, 0.0, nll(1.0, 0.0))
    for a in np.arange(0.2, 2.01, 0.1):
        for b in np.arange(-1.5, 1.51, 0.1):
            v = nll(float(a), float(b))
            if v < best[2]:
                best = (float(a), float(b), v)
    a, b = best[0], best[1]
    for step_size in (0.05, 0.02, 0.01):
        improved = True
        while improved:
            improved = False
            for da in np.arange(-0.2, 0.201, step_size):
                for db in np.arange(-0.2, 0.201, step_size):
                    v = nll(a + float(da), b + float(db))
                    if v < best[2] - 1e-12:
                        best = (a + float(da), b + float(db), v)
                        a, b = best[0], best[1]
                        improved = True
    CALIB.update({"a": a, "b": b, "fitted": True, "rows": int(len(z))})
    return dict(CALIB)


def _sim(lane: Lane, events, params: A.SimParams) -> tuple[list[dict], int]:
    rows: list[dict] = []
    crossed = 0
    for e in events:
        try:
            r = A._sim_event(lane, e, params)
        except RuntimeError as exc:
            if "crossed" in str(exc):
                crossed += 1
                continue
            raise
        if r is not None:
            rows.append(r)
    return rows, crossed


# --------------------------------------------------------------------- metrics
def _max_dd(rows: list[dict]) -> float:
    cum = peak = dd = 0.0
    for r in sorted(rows, key=lambda x: float(x["exit_ts"])):
        cum += float(r["pnl"])
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    return round(dd, 2)


def _ece_settle(rows: list[dict]) -> float:
    """Calibration against the settlement outcome, not against trade pnl.

    A take-profit round trip wins ~90% of the time at a 60c entry, so `pnl > 0`
    says nothing about whether `p_side` is the true probability. Settlement does.
    """
    pairs = [(float(r["p_side"]),
              1 if ((int(r["y"]) == 1) == (r["side"] == "YES")) else 0) for r in rows]
    return round(features.expected_calibration_error(pairs), 4) if pairs else 0.0


def summarize(rows: list[dict], hours: float) -> dict:
    n = len(rows)
    if n == 0:
        return {"trades": 0, "win_rate": 0.0, "net_pnl": 0.0, "pf": 0.0, "ece": 0.0,
                "max_dd": 0.0, "avg_entry": 0.0, "taker_fees_paid": 0.0,
                "crossed_fills": 0, "max_side_cost": 0.0, "wins": 0, "losses": 0,
                "by_lane": {}, "by_reason": {}, "trades_per_hour": 0.0}
    pnl = [float(r["pnl"]) for r in rows]
    net = sum(pnl)
    wins = sum(1 for p in pnl if p > 0)
    gross_w = sum(p for p in pnl if p > 0)
    gross_l = -sum(p for p in pnl if p <= 0)
    pf = (gross_w / gross_l) if gross_l > 0 else (99.0 if gross_w > 0 else 0.0)
    costs = [float(r["entry"]) if r["side"] == "YES" else 1.0 - float(r["entry"])
             for r in rows]
    lanes: dict[str, dict] = {}
    for r, p in zip(rows, pnl):
        d = lanes.setdefault(str(r["lane"]), {"trades": 0, "net": 0.0})
        d["trades"] += 1
        d["net"] += p
    for d in lanes.values():
        d["net"] = round(d["net"], 2)
    reasons: dict[str, int] = {}
    for r in rows:
        reasons[str(r["reason"])] = reasons.get(str(r["reason"]), 0) + 1
    return {
        "trades": n,
        "win_rate": round(wins / n, 4),
        "net_pnl": round(net, 2),
        "pf": round(pf, 4),
        "ece": _ece_settle(rows),
        "max_dd": _max_dd(rows),
        "avg_entry": round(float(np.mean(costs)), 4),
        "taker_fees_paid": round(sum(float(r["fee"]) for r in rows), 4),
        "crossed_fills": sum(int(r["crossed"]) for r in rows),
        "max_side_cost": round(max(costs), 4),
        "wins": wins,
        "losses": n - wins,
        "by_lane": dict(sorted(lanes.items())),
        "by_reason": dict(sorted(reasons.items())),
        "trades_per_hour": round(n / hours, 2) if hours > 0 else 0.0,
    }


def gate(oos: dict) -> dict:
    reasons: list[str] = []
    lanes_pos = sum(1 for v in oos["by_lane"].values() if v["net"] > 0)
    if oos["trades"] < 20:
        reasons.append(f"{oos['trades']} OOS trades (<20)")
    if oos["net_pnl"] <= 0:
        reasons.append(f"OOS net ${oos['net_pnl']:.2f} <= 0")
    if oos["losses"] < 3:
        reasons.append(f"PF meaningless: {oos['losses']} OOS losers")
    elif oos["pf"] < 1.15:
        reasons.append(f"OOS PF {oos['pf']:.3f} < 1.15")
    if oos["ece"] > 0.08:
        reasons.append(f"OOS ECE {oos['ece']:.3f} > 0.08")
    if oos["crossed_fills"] != 0:
        reasons.append(f"{oos['crossed_fills']} crossed fills")
    if oos["max_side_cost"] > CAP + EPS:
        reasons.append(f"cost {oos['max_side_cost']:.3f} > {CAP}")
    if lanes_pos < 2:
        reasons.append(f"{lanes_pos} profitable lane(s) (<2)")
    return {"pass": not reasons, "reasons": reasons, "lanes_profitable": lanes_pos}


# ----------------------------------------------------------------------- driver
def variant_key(v: Variant) -> str:
    if v.pool == "lead":
        return f"lead{v.lookback:.0f}"
    return "f2" if v.pool == "f4" else v.pool


def pools_for(lane: Lane, variants: list[Variant], sig: dict, btc: dict) -> dict:
    parts: dict[str, dict] = {}
    raw: dict[str, dict] = {}
    if any(v.pool == "f1" for v in variants):
        raw["f1"] = f1_pool(lane, sig)
    if any(v.pool in ("f2", "f4") for v in variants):
        raw["f2"] = obi_pool(lane, sig, t_lo=180.0, t_hi=780.0, delta_bps_floor=3.0)
    for lb in sorted({v.lookback for v in variants if v.pool == "lead"}):
        if lane.lane == "KXBTC15M":
            continue          # BRTI leading BRTI is not a lead-lag test
        raw[f"lead{lb:.0f}"] = lead_pool(lane, btc, sig, lookback=lb)
    for key, r in raw.items():
        p = pool_finish(r)
        if p is None or not len(p["tid"]):
            continue
        parts[key] = p
    for key, p in parts.items():
        n = len(p["tid"])
        p["ticker"] = np.array([lane.tickers[i] for i in p["tid"]], dtype=object)
        p["slice"] = np.array([lane.slice_of(float(c)) for c in p["close"]], dtype=object)
        if key == "f2" and CALIB["fitted"]:
            # The calibration is a frozen map from the model's z to a probability;
            # it is fitted on in-sample rows only and applied unchanged to both slices.
            p["p_up"] = np.array([_norm_cdf(CALIB["a"] * float(z) + CALIB["b"])
                                  for z in p["z"]])
        _log(f"  pool {key}: {n:,} rows")
    return parts


def run_slice(variants: list[Variant], lanes: tuple[str, ...], slice_name: str,
              btc: dict) -> tuple[dict[str, list[dict]], dict[str, int], dict[str, float]]:
    rows_by: dict[str, list[dict]] = {v.name: [] for v in variants}
    crossed: dict[str, int] = {v.name: 0 for v in variants}
    hours = {"is": 0.0, "oos": 0.0}
    for lane_name in lanes:
        t0 = time.time()
        lane = Lane(lane_name)
        hours["is"] = max(hours["is"], lane.hours["is"])
        hours["oos"] = max(hours["oos"], lane.hours["oos"])
        sig = A.trailing_sigma(lane)
        meta_y = {t: int(m["y"]) for t, m in lane.meta.items()}
        _log(f"{lane_name}: {len(lane.tickers)} windows, {lane.hours['is']:.1f}h/"
             f"{lane.hours['oos']:.1f}h, {len(lane.ts) / 1e6:.1f}M rows "
             f"({time.time() - t0:.0f}s load)")
        parts = pools_for(lane, variants, sig, btc)
        for v in variants:
            p = parts.get(variant_key(v))
            if p is None:
                continue
            ev = events_for(p, v, slice_name=slice_name, lane=lane)
            rows, cr = _sim(lane, ev, v.params)
            for x in rows:
                x["lane"] = lane_name
                x["y"] = meta_y.get(x["ticker"], 0)
            rows_by[v.name].extend(rows)
            crossed[v.name] += cr
            _log(f"  {slice_name.upper()} {v.name}: {len(ev)} events -> "
                 f"{len(rows)} fills ({time.time() - t0:.0f}s)")
        del lane
    return rows_by, crossed, hours


def print_table(result: dict, families: tuple[str, ...]) -> None:
    print()
    print("## Bounded-risk Frozen 60/40 sweep")
    print()
    inv = result["invariants"]
    print(f"Stake ${inv['stake_per_trade']}/trade | cap {CAP:.2f} | passive fill "
          f">= {MIN_FILL:g} contracts | 2s confirmed stops | "
          f"IS {result['hours']['is']:.1f} lane-hours, OOS "
          f"{result['hours']['oos']:.1f} lane-hours")
    print()
    print("| family | candidate | slice | trades | win_rate | net_pnl | PF | ECE | "
          "max_dd | avg_entry | taker_fees | per-lane PnL |")
    print("|" + "---|" * 12)
    for fam in families:
        f_ = result["families"].get(fam) or {}
        if not f_.get("candidate"):
            print(f"| {fam} | - | - | - | - | - | - | - | - | - | - | "
                  f"{f_.get('rejected', 'no candidate')} |")
            continue
        for sl in ("is", "oos"):
            s = f_[sl]
            lanes = " ".join(f"{k.replace('KX', '').replace('15M', '')} "
                             f"{v['net']:+.2f}" for k, v in s["by_lane"].items())
            print(f"| {fam} | {f_['candidate']} | {sl.upper()} | {s['trades']} | "
                  f"{s['win_rate']:.1%} | ${s['net_pnl']:,.2f} | {s['pf']:.2f} | "
                  f"{s['ece']:.3f} | ${s['max_dd']:,.2f} | {s['avg_entry']:.3f} | "
                  f"${s['taker_fees_paid']:,.3f} | {lanes} |")
        g = f_["gate"]
        print(f"| {fam} | GATE | {'PASS' if g['pass'] else 'REJECT'} | "
              f"{'; '.join(g['reasons']) or 'all thresholds met'} | | | | | | | | |")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lanes", default=",".join(LANES))
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--out", default="data/bounded-sweep.json")
    ap.add_argument("--min-is-trades", type=int, default=25)
    ap.add_argument("--dump-trades", action="store_true",
                    help="write the selected candidate's out-of-sample trades to "
                         "data/sweep-trades-<family>-oos.jsonl")
    ap.add_argument("--calibrate-trades", action="store_true",
                    help="fit the candidate's reported probability on its in-sample "
                         "trades (Platt) and freeze it for the out-of-sample slice")
    ap.add_argument("--calibrate", action="store_true",
                    help="fit the F2 probability on the in-sample rows (Platt) and "
                         "freeze it for the out-of-sample slice")
    args = ap.parse_args()

    lanes = tuple(x for x in args.lanes.split(",") if x)
    families = tuple(x for x in args.families.split(",") if x)
    A.STAKE = STAKE
    variants = build_variants(families)
    _log(f"{len(variants)} variants | lanes {lanes} | stake ${STAKE} | cap {CAP} | "
         f"passive fill >= {MIN_FILL:g}")
    btc = dict(np.load(CACHE / "index_BRTI.npz"))

    if args.calibrate:
        cal = fit_calibration(lanes, variants, btc)
        _log(f"calibration: p = norm_cdf({cal['a']:.3f} z {cal['b']:+.3f}) "
             f"fitted on {cal.get('rows', 0):,} in-sample rows")

    rows_is, crossed_is, hours = run_slice(variants, lanes, "is", btc)
    is_sums = {v.name: summarize(rows_is[v.name], hours["is"]) for v in variants}

    result: dict = {
        "generated_ts": time.time(),
        "invariants": {
            "hard_price_cap": CAP,
            "crossed_book": "row skipped (bid >= ask), counted, never traded",
            "passive_fill": f">={MIN_FILL:g} contracts printed at/through the limit",
            "maker_tp": "post-only, $0 fee, same 50-contract requirement",
            "stop": "2s confirmed, exits at the book, exact taker_fee",
            "stake_per_trade": str(STAKE),
            "min_passive_qty": MIN_PASSIVE_QTY,
            "split": "60% in-sample by close time, 40% frozen out-of-sample",
            "ece": "model p_side vs settlement outcome (not trade pnl)",
        },
        "hours": {"is": round(hours["is"], 2), "oos": round(hours["oos"], 2)},
        "calibration": dict(CALIB),
        "families": {},
        "variants": {},
    }
    for v in variants:
        result["variants"][v.name] = {
            "family": v.family, "is": is_sums[v.name],
            "crossed_skips_is": crossed_is[v.name],
        }

    top: dict[str, tuple[str, dict, bool]] = {}
    for fam in families:
        cands = [(v.name, is_sums[v.name]) for v in variants if v.family == fam]
        eligible = [(n, s) for n, s in cands if s["trades"] >= args.min_is_trades]
        if eligible:
            name, s_is = max(eligible, key=lambda kv: kv[1]["net_pnl"])
            top[fam] = (name, s_is, False)
        else:
            live = [(n, s) for n, s in cands if s["trades"] > 0]
            if live:
                name, s_is = max(live, key=lambda kv: (kv[1]["trades"], kv[1]["net_pnl"]))
                top[fam] = (name, s_is, True)

    for fam in families:
        if fam not in top:
            result["families"][fam] = {
                "candidate": None, "is": {}, "max_is_fills": 0,
                "rejected": "no variant produced a single IS fill"}
            _log(f"{fam}: no variant produced a single IS fill")
            continue
        name, s_is, underpowered = top[fam]
        winner = next(v for v in variants if v.name == name)
        _log(f"{fam} top IS: {name} (${s_is['net_pnl']:.2f}, {s_is['trades']} trades)"
             f" -- scoring frozen OOS once")
        rows_oos, crossed_oos, _h = run_slice([winner], lanes, "oos", btc)
        if args.calibrate_trades:
            cal = fit_trade_calibration(rows_is[name])
            _log(f"  trade calibration: p = norm_cdf({cal['a']:.3f} logit(p) "
                 f"{cal['b']:+.3f}) on {cal['rows']} in-sample trades")
            apply_trade_calibration(rows_oos[name], cal)
            result.setdefault("calibrations", {})[fam] = cal
        if args.dump_trades:
            out_rows = ROOT / f"data/sweep-trades-{fam}-oos.jsonl"
            with out_rows.open("w") as fh:
                for r in sorted(rows_oos[name], key=lambda x: float(x["ts"])):
                    fh.write(json.dumps({
                        "lane": r.get("lane"), "ticker": r["ticker"], "ts": r["ts"],
                        "side": r["side"], "entry": str(r["entry"]), "qty": r["qty"],
                        "exit": str(r["exit"]), "pnl": float(r["pnl"]), "fee": float(r["fee"]),
                        "reason": r["reason"], "p_side": round(float(r["p_side"]), 4),
                        "y": r.get("y")}, default=str) + "\n")
            _log(f"  wrote {out_rows}")
        oos = summarize(rows_oos[name], hours["oos"])
        g = gate(oos)
        if underpowered:
            g["pass"] = False
            g["reasons"].insert(0, f"selected on only {s_is['trades']} IS fills "
                                   f"(<{args.min_is_trades}); underpowered")
        result["families"][fam] = {
            "candidate": name, "underpowered": underpowered,
            "max_is_fills": max(s["trades"] for _, s in
                                [(v.name, is_sums[v.name]) for v in variants
                                 if v.family == fam]),
            "params": {k: getattr(winner.params, k) for k in (
                "tp_c", "barrier_s", "stop_c", "stop_confirm_s", "exit_mode",
                "max_entry_price", "min_fill_qty")},
            "is": s_is, "oos": oos,
            "crossed_skips_oos": crossed_oos[name],
            "gate": g,
        }
        _log(f"{fam} OOS: {oos['trades']} trades, ${oos['net_pnl']:.2f}, "
             f"PF {oos['pf']:.2f}, ECE {oos['ece']:.3f} -> "
             f"{'PASS' if g['pass'] else 'REJECT'}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=str))
    _log(f"wrote {out}")
    print_table(result, families)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
