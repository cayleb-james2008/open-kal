#!/usr/bin/env python3
"""Frozen 60/40 sub-second discovery: the two mechanisms one-minute bars cannot see.

The 1-minute tape already killed the band, the favourite buy, the tail buy and the
BTC->alt lead-lag. What it could not resolve is anything that lives inside the
final minute or inside a second. Two mechanisms remain untested on sub-second data,
and this script tests exactly those two, once, against a frozen split:

  A. LOCKIN -- the 5 Hz settlement-average partial-sum lock-in.
     A 15-minute crypto contract resolves YES when the simple average of CF
     Benchmarks' RTI over the *last* minute of the window is at least the average
     over its *first* minute (the published `floor_strike`). The last minute
     accumulates print by print, so inside it a growing part of the settlement
     value is already fixed. With T seconds left and locked average A,
         F = ((60-T) * A + T * B) / 60
     where B is the still-unknown mean of the remaining T seconds. If B is a
     Brownian path with per-sqrt-second volatility sigma, then sd(B) is
     sigma * sqrt(T/3) and sd(F) is (T/60) of that: a T^1.5 law. The model that
     ships in `features.final_minute` uses T^2 and is therefore several times too
     tight, which is how a 52% signal produced a 94%-hit-rate profile.

  B. SUB -- sub-second spot-to-strike delta plus touch imbalance and cross-asset
     lead-lag. The base model is the same settlement model evaluated mid-window
     (nothing locked yet, so the whole 60-second window is still open); the extra
     signal is an impulse of 8-20 bps in the lane's own index or in BTC's that the
     alt's ladder has not repriced yet. The order cannot arrive before the signal:
     every entry pays the book 1.0 s after the trigger, which is the honest cost
     of deciding on a tape.

Physics enforced here (not configurable, so a variant cannot cheat):

  * every entry costs <= $0.62 per contract for the side held;
  * a book row with `best_bid >= best_ask` is never traded (the executor raises);
  * a passive leg (entry, take-profit, scratch) counts only when >= 50 contracts
    print at/through its limit -- a 1-lot touch is not liquidity;
  * a confirmed 2 s stop leaves at the book and pays the exact Kalshi taker fee
    `ceil(0.07 * C * P * (1 - P))`; maker legs pay $0;
  * a trade is capped at 50 contracts: the recording keeps top-of-book only, so a
    larger size cannot be shown to be fillable at the touch.

Selection is strictly 60% in-sample (best IS profit factor among candidates with
>= 20 IS trades and positive IS net). The frozen 40% out-of-sample slice is
scored once per family, and the gate is the desk's own: OOS net > 0, PF >= 1.15,
ECE <= 0.08, >= 20 OOS trades, 0 crossed fills, cost <= 0.62 and at least two
profitable lanes.

The whole study is hard-capped strictly before `FORWARD_OOS_START_TS`: the lane
metadata, settlement-index grid, pools, split, Platt fit, selection and OOS
diagnostics never read a window closing at or after the forward tape start. The
book/index caches are materialised for every window (the noon scorer needs the
forward rows), but no fit or diagnostic here consumes them.

    .venv/bin/python scripts/subsec_discovery.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import alpha_search as A  # noqa: E402
from scripts.alpha_discover import Lane  # noqa: E402
from scripts.bounded_sweep import _ece_settle, _sim, gate, summarize  # noqa: E402
from scripts.forward_oos import FORWARD_OOS_START_TS  # noqa: E402

CAP = 0.62                # hard price cap, per contract, side held
MIN_FILL = 50.0           # contracts through a passive limit
MAX_QTY = 50              # contracts per trade (one verifiable print-through)
LAG_S = 1.0               # mandatory execution delay: trigger -> order arrival
WINDOW_S = 60.0           # the settlement window
SIGMA_LOOKBACK_S = 300.0  # trailing 5 minutes of realized volatility
EPS = 1e-9
OUT = ROOT / "data" / "subsec-discovery.json"

LANE_INDEX = {"KXBTC15M": "BRTI", "KXETH15M": "ETHUSD_RTI",
               "KXSOL15M": "SOLUSD_RTI", "KXXRP15M": "XRPUSD_RTI"}
COLS = ("tid", "ts", "close", "y", "bid", "ask", "spot", "secs", "z", "p_up",
        "delta_bps", "imp_bps", "rev", "locked", "src")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def norm_cdf(x):
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


# --------------------------------------------------------------------- volatility
class Sigma:
    """Trailing realized volatility of the settlement index, per sqrt(second).

    The engine's `realized_sigma` samples log returns about two seconds apart and
    divides by sqrt(dt); this is the same estimator, vectorised so it can be
    evaluated on every book row. Exact duplicate timestamps (the 1 Hz channel
    repeating a 5 Hz print) are dropped first: a zero return followed by a doubled
    one is not a volatility sample.
    """

    def __init__(self, index: dict):
        self.ok = False
        if not index:
            return
        its = np.asarray(index["ts"], dtype=np.float64)
        val = np.asarray(index["value"], dtype=np.float64)
        keep = np.concatenate(([True], np.diff(its) > 0))
        its, val = its[keep], val[keep]
        if len(its) < 50:
            return
        j = np.clip(np.searchsorted(its, its - 2.0, "right") - 1, 0, len(its) - 1)
        dt = its - its[j]
        good = (dt >= 1.9) & (val > 0) & (val[j] > 0)
        r = np.where(good, np.log(np.maximum(val, 1e-9) / np.maximum(val[j], 1e-9))
                     / np.sqrt(np.maximum(dt, 1e-9)), 0.0)
        self.ts = its
        self.cs2 = np.concatenate(([0.0], np.cumsum(r * r * good)))
        self.cn = np.concatenate(([0.0], np.cumsum(good.astype(np.float64))))
        self.ok = True

    def at(self, query: np.ndarray) -> np.ndarray:
        """Per-sqrt-second relative volatility at each query timestamp."""
        if not self.ok:
            return np.zeros(len(query))
        a = np.searchsorted(self.ts, np.asarray(query) - SIGMA_LOOKBACK_S, "left")
        b = np.searchsorted(self.ts, np.asarray(query), "right")
        n = self.cn[b] - self.cn[a]
        s = self.cs2[b] - self.cs2[a]
        return np.where(n >= 10, np.sqrt(s / np.maximum(n, 1.0)), 0.0)


# ----------------------------------------------------------------------- model
def index_before(index: dict,
                 start: float = FORWARD_OOS_START_TS) -> dict:
    """A settlement-index tape with every row at/after `start` removed."""
    if not index:
        return index
    ts = np.asarray(index["ts"], dtype=np.float64)
    keep = ts < start
    return {k: np.asarray(v)[keep] for k, v in index.items()}


def settle_z(*, strike: float, spot: np.ndarray, sigma: np.ndarray,
             secs_left: np.ndarray, locked_avg: np.ndarray | None) -> np.ndarray:
    """z-score of the settlement average against the strike.

    `F = ((60-T)*A + T*B)/60` with A the locked average (zero before the window
    opens), B the remaining mean and E[B] = spot under a martingale. sd(B) over T
    seconds is sigma * sqrt(T/3), so the settlement average carries (T/60) of it.
    """
    T = np.minimum(WINDOW_S, secs_left)
    a = locked_avg if locked_avg is not None else np.zeros_like(T)
    exp_f = ((WINDOW_S - T) * a + T * spot) / WINDOW_S
    sd_f = (T / WINDOW_S) * spot * sigma * np.sqrt(T / 3.0)
    return np.where(sd_f > 0, (exp_f - strike) / np.where(sd_f > 0, sd_f, 1.0), 0.0)


# ----------------------------------------------------------------------- pools
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


def book_at(lane: Lane, tid: int, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The last two-sided book quote at or before each query time.

    The decision is made on the 5 Hz index tape, so the book is looked up on the
    index's grid rather than the other way round: the raw book tape has hundreds
    of rows per second, and a pool of tens of millions of rows would be a memory
    problem rather than a discovery.
    """
    lo, hi = int(lane.book_bounds[tid]), int(lane.book_bounds[tid + 1])
    seg_ts = lane.ts[lo:hi]
    seg_bid = lane.bid[lo:hi]
    seg_ask = lane.ask[lo:hi]
    if len(seg_ts) == 0:
        z = np.zeros(len(query))
        return z, z.copy()
    j = np.searchsorted(seg_ts, query, "right") - 1
    bid = np.zeros(len(query))
    ask = np.zeros(len(query))
    for _ in range(8):                       # walk back to the last two-sided quote
        idx = np.clip(j, 0, len(seg_ts) - 1)
        ok = (j >= 0) & (seg_bid[idx] > 0) & (seg_ask[idx] < 1) & (seg_bid[idx] < seg_ask[idx])
        take = ok & (bid <= 0)
        bid = np.where(take, seg_bid[idx], bid)
        ask = np.where(take, seg_ask[idx], ask)
        j = np.where(ok, j, j - 1)
        if not np.any((j >= 0) & (bid <= 0)):
            break
    return bid, ask


def lockin_pool(lane: Lane, sigma: Sigma) -> dict:
    """Final-minute rows on the index grid: the locked average and the model z."""
    pool = new_pool()
    index = lane.index
    if not index:
        return pool
    its = np.asarray(index["ts"], dtype=np.float64)
    ival = np.asarray(index["value"], dtype=np.float64)
    for tid, name, meta, a, b in A._segments(lane):
        close = float(meta["close"])
        strike = float(meta["strike"])
        lo = int(np.searchsorted(its, close - WINDOW_S, "left"))
        hi = int(np.searchsorted(its, close, "right"))
        t, v = its[lo:hi], ival[lo:hi]
        if len(t) < 100:
            continue
        # The venue's own settlement average is the plain mean of every index print
        # in the window, so the locked average is the same mean, cut short.
        cs = np.concatenate(([0.0], np.cumsum(v)))
        k = np.arange(1, len(v) + 1, dtype=np.float64)
        locked_avg = cs[1:] / k
        secs = close - t
        spot = v
        sig = sigma.at(t)
        sel = ((secs >= 6.0) & (secs <= 55.0) & np.isfinite(spot) & (spot > 0)
               & (sig > 0) & (k >= 40))
        if not sel.any():
            continue
        z = settle_z(strike=strike, spot=spot, sigma=sig, secs_left=secs,
                     locked_avg=locked_avg)
        bid, ask = book_at(lane, tid, t)
        n = len(t)
        # The live path trades only a book with `bid < ask` (openkal/execution.py
        # book_open), and a missing side reads as zero, so a one-sided book is not
        # tradable. The study mirrors the deployment, not the tape.
        sel = ((secs >= 6.0) & (secs <= 55.0) & np.isfinite(spot) & (spot > 0)
               & (sig > 0) & (k >= 40) & (bid > 0) & (ask < 1) & (bid < ask))
        if not sel.any():
            continue
        cols = {
            "tid": np.full(n, tid, dtype=np.int32), "ts": t,
            "close": np.full(n, close), "y": np.full(n, int(meta["y"]), dtype=np.int8),
            "bid": bid, "ask": ask, "spot": spot, "secs": secs, "z": z,
            "p_up": np.zeros(n),
            "delta_bps": np.abs(spot - strike) / np.maximum(spot, 1.0) * 1e4,
            "imp_bps": np.zeros(n), "rev": np.zeros(n),
            "locked": np.clip(WINDOW_S - secs, 0.0, WINDOW_S), "src": np.zeros(n),
        }
        pool_add(pool, lane.lane, cols, sel)
    return pool


def subsec_pool(lane: Lane, btc_index: dict, sigma: Sigma, *,
                secs_lo: float, secs_hi: float) -> dict:
    """Mid-window rows on the index grid: delta, impulse, touch imbalance."""
    pool = new_pool()
    index = lane.index
    if not index or not btc_index:
        return pool
    its = np.asarray(index["ts"], dtype=np.float64)
    ival = np.asarray(index["value"], dtype=np.float64)
    bts = np.asarray(btc_index["ts"], dtype=np.float64)
    bval = np.asarray(btc_index["value"], dtype=np.float64)
    for tid, name, meta, a, b in A._segments(lane):
        close = float(meta["close"])
        strike = float(meta["strike"])
        lo = int(np.searchsorted(its, close - secs_hi, "left"))
        hi = int(np.searchsorted(its, close - secs_lo, "right"))
        t, v = its[lo:hi], ival[lo:hi]
        if len(t) < 100:
            continue
        j5 = np.clip(np.searchsorted(t, t - 5.0, "right") - 1, 0, len(t) - 1)
        own_bps = np.where(v[j5] > 0, (v / v[j5] - 1.0) * 1e4, 0.0)
        jb = np.clip(np.searchsorted(bts, t, "right") - 1, 0, len(bts) - 1)
        jb5 = np.clip(np.searchsorted(bts, t - 5.0, "right") - 1, 0, len(bts) - 1)
        btc_bps = np.where(bval[jb5] > 0, (bval[jb] / bval[jb5] - 1.0) * 1e4, 0.0)
        secs = close - t
        spot = v
        sig = sigma.at(t)
        rev = A.revision_imbalance(lane, a, b, 5.0)
        # the touch imbalance at the index tick, not at a book row
        rj = np.clip(np.searchsorted(lane.ts[a:b], t, "right") - 1, 0, b - a - 1)
        rev_at = rev[rj]
        bid, ask = book_at(lane, tid, t)
        sel = ((bid > 0) & (ask < 1) & (bid < ask) & np.isfinite(spot) & (spot > 0)
               & (sig > 0))
        if not sel.any():
            continue
        z = settle_z(strike=strike, spot=spot, sigma=sig, secs_left=secs,
                     locked_avg=None)
        n = len(t)
        cols = {
            "tid": np.full(n, tid, dtype=np.int32), "ts": t,
            "close": np.full(n, close), "y": np.full(n, int(meta["y"]), dtype=np.int8),
            "bid": bid, "ask": ask, "spot": spot, "secs": secs, "z": z,
            "p_up": np.zeros(n),
            "delta_bps": np.abs(spot - strike) / np.maximum(spot, 1.0) * 1e4,
            "imp_bps": own_bps, "rev": rev_at, "locked": np.zeros(n), "src": btc_bps,
        }
        pool_add(pool, lane.lane, cols, sel)
    return pool


# --------------------------------------------------------------------- variants
@dataclass
class Variant:
    family: str
    name: str
    pool: str
    thr: dict = field(default_factory=dict)
    entry_style: str = "taker"
    tp_c: float = 0.0
    stop_c: float = 0.0
    stop_confirm_s: float = 2.0
    exit_mode: str = "hold_to_settlement"
    min_edge: float = 0.02
    cost_lo: float = 0.05
    cost_hi: float = CAP
    cooldown: float = 900.0
    per_window: int = 1


def a_variants() -> list[Variant]:
    out: list[Variant] = []
    for t_lo in (6.0, 10.0, 15.0):
        for edge in (0.01, 0.02, 0.05, 0.10):
            for style in ("taker", "maker"):
                for cost_lo in (0.05, 0.25):
                    for exit_mode, tp in (("hold_to_settlement", 0.0),
                                          ("maker_tp", 0.04), ("maker_tp", 0.05)):
                        out.append(Variant(
                            family="A",
                            name=f"A_t{t_lo:.0f}_e{edge:.2f}_{style}_c{cost_lo:.2f}"
                                 f"_{exit_mode}{'' if not tp else f'_tp{tp:.2f}'}",
                            pool="lockin", thr={"t_lo": t_lo}, entry_style=style,
                            tp_c=tp, exit_mode=exit_mode, min_edge=edge,
                            cost_lo=cost_lo, cooldown=900.0, per_window=1))
    return out


def b_variants() -> list[Variant]:
    out: list[Variant] = []
    for source in ("own", "btc"):
        for imp in (8.0, 12.0, 20.0):
            for delta in (5.0, 15.0, 40.0):
                for rev_floor in (0.25, 0.50):
                    for style in ("taker", "maker"):
                        out.append(Variant(
                            family="B",
                            name=f"B_{source}_i{imp:.0f}_d{delta:.0f}_r{rev_floor:.2f}"
                                 f"_{style}",
                            pool="sub",
                            thr={"source": source, "imp_bps": imp,
                                 "delta_bps": delta, "rev": rev_floor},
                            entry_style=style, tp_c=0.05, stop_c=0.10,
                            stop_confirm_s=2.0, exit_mode="maker_tp",
                            min_edge=0.01, cost_lo=0.25, cost_hi=CAP,
                            cooldown=300.0, per_window=1))
    return out


# ----------------------------------------------------------------------- events
def select_rows(pool: dict, variant: Variant, slice_name: str) -> np.ndarray:
    """Rows that qualify for this variant inside one slice."""
    mask = np.ones(len(pool["tid"]), dtype=bool)
    if variant.family == "A":
        mask &= pool["secs"] >= variant.thr["t_lo"]
    else:
        src = pool["src"] if variant.thr["source"] == "btc" else pool["imp_bps"]
        mask &= np.abs(src) >= variant.thr["imp_bps"]
        mask &= pool["delta_bps"] >= variant.thr["delta_bps"]
        mask &= np.abs(pool["rev"]) >= variant.thr["rev"]
    mask &= pool["slice"] == slice_name
    idx = np.flatnonzero(mask)
    if not len(idx):
        return idx
    tid = pool["tid"][idx]
    out: list[int] = []
    lo = 0
    while lo < len(idx):
        hi = int(np.searchsorted(tid, tid[lo], "right"))
        if hi == lo:
            hi = lo + 1
        out.extend(A._dedupe([int(i) for i in idx[lo:hi]], pool["ts"],
                             cooldown=variant.cooldown, per_ticker=variant.per_window))
        lo = hi
    return np.array(out, dtype=np.int64)


def _pick_a(variant: Variant, p: float, b: float, a: float,
            have_bid: bool, have_ask: bool, *, direction: int = 0):
    """The side and style this book supports, with the best edge it offers.

    A taker buys the ask on YES and 1-bid on NO; a maker rests at the bid on YES and
    1-ask on NO. Each needs its own side of the book, and the edge the model shows is
    measured against the price that side would actually pay.
    """
    best = None
    for side in ("YES", "NO"):
        if direction and direction * ((p - 0.5) if side == "YES" else (0.5 - p)) <= 0:
            continue
        p_side = p if side == "YES" else 1.0 - p
        for style in (("taker", "maker") if variant.entry_style == "both" else
                      (variant.entry_style,)):
            if style == "taker":
                price = a if side == "YES" else (1.0 - b)
                usable = have_ask if side == "YES" else have_bid
            else:
                price = b if side == "YES" else (1.0 - a)
                usable = have_bid if side == "YES" else have_ask
            if not usable or not (0.0 < price < 1.0):
                continue
            cost = price if side == "YES" else 1.0 - price
            if not (variant.cost_lo - EPS <= cost <= CAP + EPS):
                continue
            edge = p_side - cost
            if edge < variant.min_edge - EPS:
                continue
            if style == "taker":
                # A taker leg pays the venue's fee, so the edge has to clear twice it.
                fee_c = 0.07 * cost * (1.0 - cost)
                if edge < 2.0 * fee_c - EPS:
                    continue
            if best is None or edge > best[3]:
                best = (side, price, p_side, edge)
    if best is None:
        return None, None, None
    return best[0], best[1], best[2]


def events_for(lane: Lane, pool: dict, variant: Variant, slice_name: str) -> list:
    """Turn qualifying rows into events, paying the book one second later."""
    idx = select_rows(pool, variant, slice_name)
    out = []
    for i in idx:
        tid = int(pool["tid"][i])
        arrival = float(pool["ts"][i]) + LAG_S
        close = float(pool["close"][i])
        if close - arrival < 5.0:
            continue
        quote = lane.quote_at(tid, arrival, valid_only=True, lookback=2.0)
        if quote is None:
            continue
        b, a = quote
        if not (0.0 < b < a < 1.0):
            continue
        have_bid = have_ask = True
        p = float(pool["p_up"][i])
        # A taker pays the ask on YES and 1-bid on NO; a maker rests at the bid on YES
        # and 1-ask on NO. Inside the final minute the book is often one-sided, and a
        # resting bid is exactly what a one-sided book is for: requiring two sides
        # everywhere would throw away the maker family's own opportunities.
        c_yes, c_no = (a if have_ask else None), (1.0 - b if have_bid else None)
        if variant.family == "A":
            side, price, p_side = _pick_a(variant, p, b, a, have_bid, have_ask)
            if side is None:
                continue
        else:  # noqa: SIM114 - the two families read differently on purpose
            src = (float(pool["src"][i]) if variant.thr["source"] == "btc"
                   else float(pool["imp_bps"][i]))
            direction = 1 if src >= 0 else -1
            if direction * (p - 0.5) <= 0:
                continue                    # the impulse must point at the model
            side, price, p_side = _pick_a(variant, p, b, a, have_bid, have_ask,
                                          direction=direction)
            if side is None:
                continue
        cost = price if side == "YES" else 1.0 - price
        if not (variant.cost_lo - EPS <= cost <= CAP + EPS):
            continue
        if variant.entry_style == "taker":
            # A taker leg pays the venue's fee, so the edge has to clear twice it or
            # the trade starts underwater. The fee is price-dependent: 0.07*p*(1-p)
            # is 1.75c a contract at the middle of the band and 0.63c at 0.10.
            fee_c = 0.07 * cost * (1.0 - cost)
            edge = (p_side if side == "YES" else 1.0 - p_side) - cost
            if edge < 2.0 * fee_c - EPS:
                continue
        out.append(A.Event(
            variant.family, variant.name, tid=tid, ticker=str(pool["ticker"][i]),
            ts=arrival, close=close, y=int(pool["y"][i]), slice=slice_name,
            side=side, entry_style=variant.entry_style, limit=price, p_side=p_side))
    return out


# ------------------------------------------------------------------ calibration
def fit_platt(z: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, *,
              sample: int = 20000) -> tuple[float, float]:
    """Fit p = norm_cdf(a * z + b) on the in-sample rows; freeze for OOS.

    The settlement model is right in form and wrong in scale: the shipped
    variance is several times too tight, which shows up as a miscalibrated
    probability. Two numbers -- a slope and an offset -- repair the scale and the
    bias without touching the structure.

    The rows inside one window are not independent samples: a window has one
    outcome and hundreds of near-boundary ticks, so an unweighted fit would let a
    few windows speak for the whole tape. `w` is the inverse of the number of rows
    each window contributes.
    """
    if len(z) < 50:
        return 1.0, 0.0
    if len(z) > sample:
        step = len(z) // sample
        z, y = z[::step][:sample], y[::step][:sample]
        w = None if w is None else w[::step][:sample]
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.ones(len(z)) if w is None else np.asarray(w, dtype=np.float64)
    w = w / w.sum()

    def nll(a: float, b: float) -> float:
        p = np.clip(norm_cdf(a * z + b), 1e-6, 1 - 1e-6)
        return float(-np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))))

    best = (1.0, 0.0, nll(1.0, 0.0))
    for a in np.arange(0.2, 2.01, 0.1):
        for b in np.arange(-1.5, 1.51, 0.1):
            v = nll(float(a), float(b))
            if v < best[2]:
                best = (float(a), float(b), v)
    a, b, _ = best
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
    return a, b


# ----------------------------------------------------------------------- driver
def freeze_split(lanes: dict) -> dict:
    """One chronological 60/40 split, built once, shared by every lane."""
    closes = sorted({float(m["close"]) for lane in lanes.values()
                     for m in lane.meta.values()
                     if float(m["close"]) < FORWARD_OOS_START_TS})
    cut = int(len(closes) * 0.6)
    is_c = set(closes[:cut])
    oos_c = set(closes[cut:])
    for lane in lanes.values():
        lane.is_closes = is_c
        lane.oos_closes = oos_c
        lane.cut_ts = closes[cut] if cut < len(closes) else math.inf
    fmt = lambda t: (datetime.fromtimestamp(t, timezone.utc).isoformat()
                     if t else None)
    return {
        "rule": "chronological 60/40 by window close, frozen once for every family",
        "windows": len(closes),
        "is_windows": len(is_c), "oos_windows": len(oos_c),
        "is_start": fmt(closes[0] if closes else None),
        "is_end": fmt(closes[cut - 1] if cut else None),
        "oos_start": fmt(closes[cut] if cut < len(closes) else None),
        "oos_end": fmt(closes[-1] if closes else None),
        "is_lane_hours": round(len(is_c) * 0.25 * len(lanes), 1),
        "oos_lane_hours": round(len(oos_c) * 0.25 * len(lanes), 1),
        "forward_oos_start_ts": FORWARD_OOS_START_TS,
        "boundary": "strictly before the forward tape; no window at/after "
                    "forward_oos_start_ts enters a pool, fit or diagnostic",
    }


def sim_params(v: Variant) -> A.SimParams:
    return A.SimParams(
        tp_c=v.tp_c, barrier_s=45.0, stop_c=v.stop_c, exit_mode=v.exit_mode,
        max_entry_price=CAP, min_fill_qty=MIN_FILL, max_qty=MAX_QTY,
        stop_confirm_s=v.stop_confirm_s)


def run_family(variants: list[Variant], pools: dict, lanes: dict,
               slice_name: str, hours: float) -> tuple[dict, dict]:
    """Simulate every variant on one slice."""
    results: dict[str, dict] = {}
    events_seen: dict[str, int] = {}
    for v in variants:
        rows: list[dict] = []
        crossed = 0
        nevents = 0
        for lane_name, pool in pools.items():
            lane = lanes[lane_name]
            events = events_for(lane, pool, v, slice_name)
            nevents += len(events)
            r, c = _sim(lane, events, sim_params(v))
            meta_y = {t: int(m["y"]) for t, m in lane.meta.items()
                      if float(m["close"]) < FORWARD_OOS_START_TS}
            for row in r:
                row["lane"] = lane_name
                row["y"] = meta_y.get(row["ticker"], 0)
            rows.extend(r)
            crossed += c
        events_seen[v.name] = nevents
        if not rows:
            continue
        s = summarize(rows, hours)
        # What the model's own probabilities say these trades were worth, next to
        # what they realized. A realized number far above the modelled one is a
        # lucky sample, not an edge, and the difference is the thing to look at.
        s["model_ev"] = round(sum(
            ((float(r["p_side"]) - (float(r["entry"]) if r["side"] == "YES"
                                    else 1.0 - float(r["entry"]))) * float(r["qty"])
             - float(r["fee"])) for r in rows), 2)
        s["crossed_fills"] = crossed
        s["max_side_cost"] = max(
            (float(r["entry"]) if r["side"] == "YES" else 1.0 - float(r["entry"])
             for r in rows), default=0.0)
        s["events"] = nevents
        results[v.name] = s
    return results, events_seen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lanes", nargs="*", default=list(LANE_INDEX))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    lanes: dict[str, Lane] = {}
    for name in args.lanes:
        _log(f"loading lane {name}")
        lanes[name] = Lane(name, max_close=FORWARD_OOS_START_TS)
        lanes[name].index = index_before(lanes[name].index)
    split = freeze_split(lanes)
    _log(f"split frozen: {split['is_windows']} IS / {split['oos_windows']} OOS windows "
         f"({split['is_lane_hours']} / {split['oos_lane_hours']} lane-hours)")

    btc_index = lanes["KXBTC15M"].index if "KXBTC15M" in lanes else {}
    pools: dict[str, dict] = {}
    for name, lane in lanes.items():
        sig = Sigma(lane.index)
        p = {"lockin": pool_finish(lockin_pool(lane, sig)),
             "sub": pool_finish(subsec_pool(lane, btc_index, sig,
                                            secs_lo=30.0, secs_hi=780.0))}
        for key, v in p.items():
            if v is None:
                p[key] = None
                continue
            v["ticker"] = np.array([lane.tickers[i] for i in v["tid"]], dtype=object)
            v["slice"] = np.array([lane.slice_of(float(c)) for c in v["close"]],
                                  dtype=object)
            _log(f"  pool {name}/{key}: {len(v['tid']):,} rows")
        pools[name] = p

    report: dict = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "forward_oos_start_ts": FORWARD_OOS_START_TS,
        "boundary": "all pools, fits, selections and OOS diagnostics stop "
                    "strictly before the forward tape start",
        "split": split,
        "lanes": list(lanes),
        "physics": {
            "cap": CAP, "min_fill": MIN_FILL, "max_qty": MAX_QTY,
            "lag_s": LAG_S, "settlement_window_s": WINDOW_S,
            "sigma_lookback_s": SIGMA_LOOKBACK_S,
            "settlement_rule": "YES iff avg60(index, close-60..close) >= "
                               "avg60(index, open-60..open) == floor_strike",
            "variance_law": "sd(F) = (T/60) * spot * sigma * sqrt(T/3)",
        },
        "families": {},
        "pool_rows": {name: {k: (0 if v is None else int(len(v["tid"])))
                             for k, v in p.items()} for name, p in pools.items()},
    }

    a_pools = {n: pools[n]["lockin"] for n in lanes if pools[n]["lockin"] is not None}
    b_pools = {n: pools[n]["sub"] for n in lanes if pools[n]["sub"] is not None}

    for family, pool_map, variants in (("A", a_pools, a_variants()),
                                       ("B", b_pools, b_variants())):
        if not pool_map:
            continue
        z_is, y_is, w_is = [], [], []
        for name, p in pool_map.items():
            sel = p["slice"] == "is"
            z_is.append(p["z"][sel])
            y_is.append(p["y"][sel])
            # one window, one outcome: weight each window equally, not each tick
            tid = p["tid"][sel]
            uniq, counts = np.unique(tid, return_counts=True)
            per = dict(zip(uniq.tolist(), counts.tolist()))
            w_is.append(np.array([1.0 / per[int(t)] for t in tid]))
        z_all = np.concatenate(z_is)
        y_all = np.concatenate(y_is).astype(np.float64)
        w_all = np.concatenate(w_is)
        a_s, b_s = fit_platt(z_all, y_all, w_all)
        for name, p in pool_map.items():
            p["p_up"] = norm_cdf(a_s * p["z"] + b_s)
        raw_ece = _ece_settle(
            [{"p_side": float(pp), "y": int(yy), "side": "YES"}
             for pp, yy in zip(p["p_up"][p["slice"] == "is"][::7], y_all[::7])])
        _log(f"family {family}: p = norm_cdf({a_s:.3f} z {b_s:+.3f}) on "
             f"{len(z_all):,} IS rows, IS ECE {raw_ece:.4f}")
        is_res, events_seen = run_family(variants, pool_map, lanes, "is",
                                         split["is_lane_hours"])
        cands = [(n, s) for n, s in is_res.items()
                 if s["trades"] >= 20 and s["net_pnl"] > 0]
        fam: dict = {"platt": {"a": a_s, "b": b_s}, "is_rows": int(len(z_all)),
                     "is_raw_ece": raw_ece, "is": is_res, "events": events_seen}
        # The best in-sample variant, whether or not it clears the IS bar: a family
        # that is short of trades still has to say what it saw, so the morning report
        # can point at the next tape. The frozen OOS slice is untouched below unless
        # a variant actually qualifies.
        with_trades = [(n, s) for n, s in is_res.items() if s["trades"] > 0]
        if with_trades:
            # The variant with the most in-sample trades is the one that says
            # something about the mechanism rather than about one lucky window.
            mt_name, mt = max(with_trades, key=lambda kv: (kv[1]["trades"],
                                                           kv[1]["net_pnl"]))
            fam["most_traded_is"] = dict(mt, variant=mt_name)
            # The same entries with a different exit: a truncated take-profit on a
            # binary keeps the small win and the full loss, and that geometry, not
            # the signal, is what cost the live account $44.55.
            prefix = "_".join(mt_name.split("_")[:5]) + "_"
            fam["exit_geometry_same_entries"] = [
                {"variant": n, "trades": s["trades"], "net_pnl": s["net_pnl"],
                 "pf": s["pf"], "ece": s["ece"], "model_ev": s.get("model_ev")}
                for n, s in is_res.items() if n.startswith(prefix)]
            best_name, best = max(with_trades,
                             key=lambda kv: (kv[1]["pf"], kv[1]["trades"]))
            v = next(x for x in variants if x.name == best_name)
            winner_rows: list[dict] = []
            for name, pool in pool_map.items():
                lane = lanes[name]
                evs = events_for(lane, pool, v, "is")
                rr, _ = _sim(lane, evs, sim_params(v))
                meta_y = {t: int(m["y"]) for t, m in lane.meta.items()
                      if float(m["close"]) < FORWARD_OOS_START_TS}
                for row in rr:
                    row["lane"] = name
                    row["y"] = meta_y.get(row["ticker"], 0)
                winner_rows.extend(rr)
            fam["best_is_variant"] = best_name
            fam["best_is"] = best
            fam["winner_trades_is"] = [
                {"lane": r["lane"], "ticker": r["ticker"], "side": r["side"],
                 "entry_style": r["entry_style"], "entry": str(r["entry"]),
                 "qty": r["qty"], "exit": str(r["exit"]), "pnl": float(r["pnl"]),
                 "reason": r["reason"], "p_side": round(float(r["p_side"]), 4),
                 "y": r["y"],
                 "secs_left": round(float(r["exit_ts"]) - float(r["ts"]), 1)}
                for r in sorted(winner_rows, key=lambda x: x["ts"])]
        if not cands:
            fam["selected"] = None
            fam["no_candidate"] = ("no variant cleared IS selection "
                                   "(>=20 IS trades, positive IS net)")
            report["families"][family] = fam
            _log(f"family {family}: no IS candidate")
            continue
        best_name, best = max(cands, key=lambda kv: (kv[1]["pf"], kv[1]["trades"]))
        v = next(x for x in variants if x.name == best_name)
        _log(f"family {family}: IS winner {best_name} (PF {best['pf']:.3f}, "
             f"{best['trades']} IS trades) -> scoring the frozen OOS slice once")
        oos_res, oos_events = run_family([v], pool_map, lanes, "oos",
                                         split["oos_lane_hours"])
        oos = oos_res.get(v.name)
        if oos is None:
            oos = {"trades": 0, "net_pnl": 0.0, "pf": 0.0, "ece": 0.0,
                   "crossed_fills": 0, "max_side_cost": 0.0, "by_lane": {},
                   "losses": 0, "wins": 0}
        fam["selected"] = {
            "variant": best_name,
            "params": {"family": family, "thr": v.thr, "entry_style": v.entry_style,
                       "tp_c": v.tp_c, "stop_c": v.stop_c,
                       "stop_confirm_s": v.stop_confirm_s, "exit_mode": v.exit_mode,
                       "min_edge": v.min_edge, "cost_lo": v.cost_lo,
                       "cost_hi": v.cost_hi, "lag_s": LAG_S, "max_qty": MAX_QTY},
            "is": best,
            "oos": oos,
            "gate": gate(oos),
        }
        report["families"][family] = fam
        _log(f"family {family} OOS: {oos.get('trades')} trades net "
             f"{oos.get('net_pnl')} PF {oos.get('pf')} ECE {oos.get('ece')} -> "
             f"{fam['selected']['gate']}")

    Path(args.out).write_text(json.dumps(report, indent=1, default=str))
    _log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
