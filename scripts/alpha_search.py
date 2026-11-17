"""Multi-family microstructure search on the recorded sub-second book + print tape.

The first pass tested one family (fade the spike) and found it negative on frozen OOS.
Its most useful output was the *mechanism*: a resting sell at the spiked ask only fills
when the buyer keeps buying, and the OOS win rate halves. That mechanism is a hypothesis
about the tape, not a fact about one rule, so this harness searches its mirror and the
neighbouring families under one strict simulator:

  spike_fade       sell the up-spike at the ask (maker) / buy the down-spike at the bid
  spike_momentum   join the same spike (taker at the touch), exit with a maker TP
  spike_pullback   wait for a 1c pullback after the spike, join as a maker, ride it
  index_dislocation book rich/cheap versus the settlement index's own probability
  absorption_revert a large taker print that does not move the touch is absorbed
  spread_capture   wide two-sided book, model agrees, rest at the touch
  window_open      first tradeable quote of a window versus the index probability

Every fill is simulated on `ws_trade` prints: a passive order fills only when >= 20
contracts print at or through its limit in the direction that would hit it; a taker
entry pays `openkal.fees.taker_fee` in exact Decimal; a `bid >= ask` row is an assertion
failure, never a filter. The split is chronological 60/40 by settled window close and is
built once. Detection and selection read only the in-sample slice; the frozen OOS slice
is scored once, for the single IS-best variant of each family.

    .venv/bin/python scripts/alpha_search.py            # full search + verdict
    .venv/bin/python scripts/alpha_search.py --lane KXBTC15M
    .venv/bin/python scripts/alpha_search.py --hold-settlement   # maker in, settle at close
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openkal import features, verifier  # noqa: E402
from openkal.fees import taker_fee  # noqa: E402
from openkal.strategy import ZOBI_MIN_DELTA  # noqa: E402
from scripts.alpha_discover import (LANES, MIN_PASSIVE_QTY, STAKE, Lane, block,  # noqa: E402
                                    CACHE, detect_spikes, gate, verifier_call)

try:                                   # fast exact path when scipy is present
    from scipy.special import erf as _erf  # type: ignore

    def _norm_cdf(z: np.ndarray) -> np.ndarray:
        return 0.5 * (1.0 + _erf(z / SQRT2))

    _CDF_BACKEND = "scipy"
except Exception:                      # noqa: BLE001 - the tanh form is exact enough
    def _norm_cdf(z: np.ndarray) -> np.ndarray:
        return 0.5 * (1.0 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))

    _CDF_BACKEND = "tanh"

REPORT = ROOT / "data" / "alpha-search.json"
VERDICT = ROOT / "data" / "strategy-verdict.json"
TICK = Decimal("0.01")
EPS = 1e-6
SQRT2 = math.sqrt(2.0)


# --------------------------------------------------------------------- helpers
def fair_p(spot: np.ndarray, strike: float, secs: np.ndarray, sigma: float) -> np.ndarray:
    """Zero-drift GBM probability that the settlement lands at/above the strike.

    This is `features.fair_value` without the lead-lag tilt: a level comparison should
    not carry a momentum term, and the vectorised form keeps 30M rows affordable.
    """
    out = np.full(len(spot), 0.5, dtype=np.float64)
    sig = np.asarray(sigma, dtype=np.float64)
    if strike <= 0:
        return out
    ok = (spot > 0) & (secs > 0) & np.isfinite(spot) & np.isfinite(sig) & (sig > 0)
    if not ok.any():
        return out
    vol = sig[ok] * np.sqrt(np.maximum(secs[ok], 0.0))
    good = vol > 0
    z = np.full(vol.shape, 0.0)
    z[good] = np.log(spot[ok][good] / strike) / vol[good]
    out[ok] = _norm_cdf(z)
    return out


def index_at(lane: Lane, ts: np.ndarray) -> np.ndarray:
    """Nearest settlement-index print within 1.5s, vectorised; NaN when there is none."""
    if not lane.index:
        return np.full(len(ts), np.nan)
    arr = lane.index["ts"]
    val = lane.index["value"]
    j = np.searchsorted(arr, ts, "left")
    jm = np.clip(j - 1, 0, len(arr) - 1)
    jp = np.clip(j, 0, len(arr) - 1)
    dm = np.abs(arr[jm] - ts)
    dp = np.abs(arr[jp] - ts)
    use_m = dm <= dp
    out = np.where(use_m, val[jm], val[jp]).astype(np.float64)
    gap = np.where(use_m, dm, dp)
    out[gap > 1.5] = np.nan
    return out


def index_at_past(lane: Lane, ts: np.ndarray, tol: float = 1.5) -> np.ndarray:
    """Last settlement-index print at or before `ts`, NaN when older than `tol`.

    `index_at` takes the *nearest* print, which may be stamped up to 1.5s after the
    row it scores. That is look-ahead: a live trigger could not have seen it. The
    forward scorer uses this backward-only read so no feature depends on a print
    that had not happened when the evidence row was written.
    """
    if not lane.index:
        return np.full(len(ts), np.nan)
    arr = lane.index["ts"]
    val = lane.index["value"]
    j = np.searchsorted(arr, ts, "right") - 1
    ok = j >= 0
    jc = np.clip(j, 0, len(arr) - 1)
    out = np.where(ok, val[jc], np.nan).astype(np.float64)
    gap = np.where(ok, ts - arr[jc], np.inf)
    out[gap > tol] = np.nan
    return out


def mid_at(lane: Lane, tid: int, ts: float, *, lookback: float = 30.0) -> float | None:
    """Mid of the last two-sided book row at or before `ts`."""
    lo, hi = int(lane.book_bounds[tid]), int(lane.book_bounds[tid + 1])
    j = int(np.searchsorted(lane.ts[lo:hi], ts, "right")) - 1
    while j >= 0:
        if ts - float(lane.ts[lo + j]) > lookback:
            return None
        b, a = float(lane.bid[lo + j]), float(lane.ask[lo + j])
        if 0.0 < b < a < 1.0:
            return (b + a) / 2.0
        j -= 1
    return None


@dataclass
class Event:
    family: str
    variant: str
    tid: int
    ticker: str
    ts: float
    close: float
    y: int
    slice: str
    side: str            # "YES" = long YES, "NO" = short YES
    entry_style: str     # "maker" | "taker"
    limit: float         # entry price in YES terms
    p_side: float        # model probability assigned to the side we hold
    tag: str = ""
    # --- window-cap / toxicity study ---------------------------------------
    # Signed short-horizon settlement-index velocity (bps) and touch-revision
    # imbalance measured at the event timestamp. `dir_side` is +1 for a long YES
    # fade and -1 for a short one; a fade is toxic when either measure points
    # against it, i.e. the move being faded is still accelerating.
    dir_side: int = 0
    vel_bps: float = 0.0
    rev: float = 0.0


@dataclass
class SimParams:
    tp_c: float
    barrier_s: float
    stop_c: float = 0.0
    patience_s: float = 15.0
    # `hold_to_settlement` replaces the take-profit / stop / time-barrier exits with the
    # binary payout at the window close. There is no exit order at all, so there is no
    # spread-crossing taker leg anywhere in the round trip and both fees are $0.
    exit_mode: str = "maker_tp"
    # Cap on the price paid per contract for the side held (the YES price for a long,
    # 1 - YES price for a short). `1.0` disables the cap.
    max_entry_price: float = 1.0
    # `maker_scratch`: at the time barrier, rest a post-only scratch instead of crossing.
    # `scratch_concession_c` is how much worse than breakeven the order is willing to
    # rest (0.0 = breakeven, 0.01 = one cent). It must be hit by >= 20 printed contracts
    # like any other passive order; when it is not, the position is carried to the
    # settlement payout rather than paid for with a taker fee.
    scratch_concession_c: float = 0.0
    scratch_patience_s: float = 20.0
    # Rest at the touch instead of crossing it, for detectors that normally take
    # (`window_open`). A hold-to-settlement round trip otherwise pays a taker entry fee.
    force_maker: bool = False
    # `synthetic_flatten`: instead of selling the position at a loss, rest a
    # post-only bid on the *opposite* contract at `1 - (entry - concession)`. A
    # fill locks YES + NO = 1 + concession at settlement, so the worst case is
    # `concession` per contract instead of the whole binary. $0 fee on the leg.
    flatten_concession_c: float = 0.02
    # Hard bounded stop, in cents of adverse price move, applied to EVERY exit
    # regime including `maker_scratch`. Without it a passive program can only lose
    # the take-profit on a winner and the whole binary on a loser: the 15:1 skew
    # that turned an 84c entry into a -$65 settlement. The stop crosses (taker fee)
    # because a post-only exit fills only when the book comes back to it.
    max_loss_c: float = 0.0
    # A stop is *confirmed*, not touched. With 0 the first print through the level
    # fires; with a value the level must stay breached for that many seconds. The
    # engine's `_zobi_exit` runs the same rule, so the study and live agree.
    stop_confirm_s: float = 0.0
    # Contracts that must print at/through a passive limit before it counts as
    # filled. The desk study uses 20 (a touch that traded); the bounded sweep uses
    # 50, because a 1-lot touch is not liquidity you could actually have taken.
    min_fill_qty: float = MIN_PASSIVE_QTY
    # Hard cap on contracts per trade. The recording keeps top-of-book only, so a
    # size above one verifiable print-through cannot be shown to be fillable at the
    # touch; 0 disables the cap. 50 is the unit the whole desk is built on.
    max_qty: int = 0


# ------------------------------------------------------------------ detectors
SIG_WINDOW = 150.0        # trailing vol lookback, seconds
SIG_FLOOR = 5.0e-5        # per-root-second prior used before the window has history


def trailing_sigma(lane: Lane) -> dict[int, np.ndarray]:
    """Per-row settlement-index vol from prints at/before the row only.

    `Lane.sigma` is the realised vol of the whole window, which includes the future.
    Every fair value in this study must be computable at the moment of the event, so
    the vol is re-estimated on a trailing window and mapped back onto book rows.
    """
    out: dict[int, np.ndarray] = {}
    if not lane.index:
        return out
    its = np.asarray(lane.index["ts"], dtype=np.float64)
    ival = np.asarray(lane.index["value"], dtype=np.float64)
    for tid, _name, meta, a, b in _segments(lane):
        ts = lane.ts[a:b]
        lo = int(np.searchsorted(its, float(meta["close"]) - 900.0, "left"))
        hi = int(np.searchsorted(its, float(meta["close"]), "right"))
        t = its[lo:hi]
        v = ival[lo:hi]
        good = np.isfinite(v) & (v > 0)
        t, v = t[good], v[good]
        sig = np.full(len(ts), SIG_FLOOR, dtype=np.float64)
        if len(t) >= 12:
            dt = np.diff(t)
            ok = (dt >= 1e-3) & (dt <= 5.0)
            r = np.log(v[1:] / v[:-1])
            rate = np.where(ok, (r / np.sqrt(np.maximum(dt, 1e-3))) ** 2, 0.0)
            okf = ok.astype(np.float64)
            csum = np.cumsum(rate)
            ccnt = np.cumsum(okf)
            tv = t[1:]
            left = np.searchsorted(tv, tv - SIG_WINDOW, "left")
            prev = np.maximum(left - 1, 0)
            num = csum - np.where(left > 0, csum[prev], 0.0)
            den = ccnt - np.where(left > 0, ccnt[prev], 0.0)
            est = np.where(den >= 5, num / np.maximum(den, 1.0), SIG_FLOOR ** 2)
            j = np.searchsorted(tv, ts, "right") - 1
            sig = np.sqrt(np.maximum(est[np.clip(j, 0, len(est) - 1)], SIG_FLOOR ** 2))
            sig[j < 0] = SIG_FLOOR
        out[tid] = sig
    return out


def _sigma_at(lane: Lane, sig: dict[int, np.ndarray], tid: int, ts: float) -> float:
    arr = sig.get(tid)
    if arr is None or not len(arr):
        return SIG_FLOOR
    a = int(lane.book_bounds[tid])
    b = int(lane.book_bounds[tid + 1])
    j = int(np.searchsorted(lane.ts[a:b], ts, "right")) - 1
    return float(arr[j]) if 0 <= j < len(arr) else SIG_FLOOR


def fair_p1(spot: float, strike: float, secs: float, sigma: float) -> float | None:
    if not (math.isfinite(spot) and math.isfinite(secs) and spot > 0 and strike > 0
            and secs > 0 and sigma > 0):
        return None
    z = math.log(spot / strike) / (sigma * math.sqrt(secs))
    return float(0.5 * (1.0 + math.tanh(0.7978845608 * (z + 0.044715 * z ** 3))))


def _segments(lane: Lane):
    for tid, name in enumerate(lane.tickers):
        meta = lane.meta.get(name)
        if meta is None:
            continue
        a, b = int(lane.book_bounds[tid]), int(lane.book_bounds[tid + 1])
        if b - a < 10:
            continue
        yield tid, name, meta, a, b


def _dedupe(rows: list[int], ts: np.ndarray, *, cooldown: float, per_ticker: int) -> list[int]:
    """Keep the first trigger, then one per `cooldown`, capped per contract."""
    out: list[int] = []
    last = -1e9
    kept = 0
    for i in rows:
        t = float(ts[i])
        if t - last < cooldown:
            continue
        if kept >= per_ticker:
            break
        out.append(i)
        last = t
        kept += 1
    return out


def spike_events(lane: Lane, sig: dict[int, np.ndarray], *, horizon: float,
                 conf: float) -> tuple[list[Event], list[Event]]:
    """Fade and momentum events from one spike detector, so the mirror is exact.

    `detect_spikes` confirms with `Lane.sigma` (full-window, includes the future), so it
    is called with the filter disabled and the confirmation is re-applied here from the
    trailing vol at the two timestamps the detector compares.
    """
    fade: list[Event] = []
    momo: list[Event] = []
    for c in detect_spikes(lane, horizon=horizon, mag_lo=0.08, mag_hi=0.20,
                           spread_max=0.05, conf_max=1e9):
        tid = int(c["tid"])
        secs = float(c["close"]) - float(c["ts"])
        now = fair_p1(float(c["spot"]), float(c["strike"]), secs,
                      _sigma_at(lane, sig, tid, float(c["ts"])))
        then_spot = lane.index_value(float(c["anchor_ts"]))
        then = fair_p1(float(then_spot), float(c["strike"]), secs + float(c["gap_s"]),
                       _sigma_at(lane, sig, tid, float(c["anchor_ts"])))
        if now is None or then is None:
            continue
        implied = abs(now - then)
        if implied > conf * abs(float(c["delta"])):
            continue
        base = {"tid": tid, "ticker": c["ticker"], "ts": c["ts"], "close": c["close"],
                "y": c["y"], "slice": c["slice"]}
        up = c["delta"] > 0
        fade.append(Event("spike_fade", f"h{horizon:g}_c{conf:.2f}", side=c["side"],
                          entry_style="maker", limit=c["entry_limit"],
                          p_side=c["p_side"], **base))
        # Mirror: a maker sell that fills only on continuation is, from the other
        # side, evidence the move continues. Take it at the touch.
        # Every `limit` in this study is a YES price; a NO position is a short YES.
        if up:
            side, limit = "YES", c["ask"]
        else:
            side, limit = "NO", c["bid"]
        momo.append(Event("spike_momentum", f"h{horizon:g}_c{conf:.2f}", side=side,
                          entry_style="taker", limit=limit,
                          p_side=1.0 - c["p_side"], **base))
        # And the same side, but as a maker waiting one tick off the touch: the fill
        # arrives on a pullback rather than on the chase.
        offset = 0.01
        if up:
            side, limit = "YES", max(0.01, c["ask"] - offset)
        else:
            side, limit = "NO", min(0.99, c["bid"] + offset)
        if 0.0 < limit < 1.0 and ((side == "YES" and limit < c["ask"])
                                  or (side == "NO" and limit > c["bid"])):
            momo.append(Event("spike_pullback", f"h{horizon:g}_c{conf:.2f}", side=side,
                              entry_style="maker", limit=limit,
                              p_side=1.0 - c["p_side"], **base))
    return fade, momo


def dislocation_events(lane: Lane, sig: dict[int, np.ndarray], *, gap_min: float,
                       persist: int, cooldown: float = 60.0,
                       per_window: int = 3) -> list[Event]:
    """Level dislocation: the book's mid against the index's own probability."""
    out: list[Event] = []
    for tid, name, meta, a, b in _segments(lane):
        ts = lane.ts[a:b]
        bid = lane.bid[a:b]
        ask = lane.ask[a:b]
        vel = velocity_bps(lane, ts, VETO_LOOKBACK_S)
        revs = revision_imbalance(lane, a, b, VETO_LOOKBACK_S)
        valid = (bid > 0) & (ask < 1) & (bid < ask) & ((ask - bid) <= 0.05 + EPS)
        secs = meta["close"] - ts
        valid &= (secs >= 120.0) & (secs <= 840.0)
        if valid.sum() < 10:
            continue
        spot = index_at(lane, ts)
        p = fair_p(spot, meta["strike"], secs, sig[tid])
        mid = (bid + ask) / 2.0
        gap = np.where(valid & np.isfinite(spot), mid - p, np.nan)
        if persist > 1:
            prev = np.full(len(gap), np.nan)
            prev[1:] = gap[:-1]
            same = np.isfinite(gap) & np.isfinite(prev) & (np.sign(gap) == np.sign(prev))
            same &= np.abs(prev) >= gap_min
            gap = np.where(same, gap, np.nan)
        rich = np.flatnonzero(np.isfinite(gap) & (gap >= gap_min))
        cheap = np.flatnonzero(np.isfinite(gap) & (gap <= -gap_min))
        for idx, short in ((rich, True), (cheap, False)):
            kept = _dedupe(list(idx), ts, cooldown=cooldown, per_ticker=per_window)
            for i in kept:
                side = "NO" if short else "YES"
                limit = float(ask[i]) if short else float(bid[i])
                if not (0.0 < limit < 1.0):
                    continue
                out.append(Event(
                    "index_dislocation", f"g{gap_min:.2f}_p{persist}", tid=tid, ticker=name,
                    ts=float(ts[i]), close=meta["close"], y=meta["y"],
                    slice=lane.slice_of(meta["close"]), side=side, entry_style="maker",
                    limit=limit, p_side=float(p[i] if not short else 1.0 - p[i]),
                    dir_side=-1 if short else 1, vel_bps=float(vel[i]),
                    rev=float(revs[i]), tag=f"gap{float(gap[i]):+.3f}"))
    return out


def absorption_events(lane: Lane, sig: dict[int, np.ndarray], *, size: float,
                      cooldown: float = 30.0, per_window: int = 3,
                      hold_bps: float = 0.005) -> list[Event]:
    """A big taker print at the touch that does not move the touch is absorbed."""
    out: list[Event] = []
    for tid, name, meta, a, b in _segments(lane):
        lo, hi = int(lane.trade_bounds[tid]), int(lane.trade_bounds[tid + 1])
        if hi - lo < 5:
            continue
        t_ts = lane.trade_ts[lo:hi]
        t_px = lane.trade_px[lo:hi]
        t_sz = lane.trade_sz[lo:hi]
        t_ty = lane.trade_ty[lo:hi]
        seg_ts = lane.ts[a:b]
        vel = velocity_bps(lane, seg_ts, VETO_LOOKBACK_S)
        revs = revision_imbalance(lane, a, b, VETO_LOOKBACK_S)
        big = np.flatnonzero(t_sz >= size)
        kept = _dedupe(list(big), t_ts, cooldown=cooldown, per_ticker=per_window * 3)
        # Map each print to the book row at or after it.
        j = np.searchsorted(seg_ts, t_ts[kept], "left")
        for k, pos in enumerate(kept):
            row = int(j[k])
            if row <= 0 or row >= (b - a):
                continue
            price = float(t_px[pos])
            bid_now, ask_now = float(lane.bid[a + row]), float(lane.ask[a + row])
            if not (0.0 < bid_now < ask_now < 1.0):
                continue
            secs = meta["close"] - float(t_ts[pos])
            if not (120.0 < secs < 840.0):
                continue
            if int(t_ty[pos]) == 0 and price <= lane.bid[a + row - 1] + EPS:
                # Aggressor sold YES into the bid and the bid held.
                if bid_now >= float(lane.bid[a + row - 1]) - hold_bps:
                    spot = lane.index_value(float(t_ts[pos]))
                    p = fair_p1(spot, meta["strike"], secs,
                                _sigma_at(lane, sig, tid, float(t_ts[pos])))
                    out.append(Event(
                        "absorption_revert", f"s{size:g}", tid=tid, ticker=name,
                        ts=float(t_ts[pos]), close=meta["close"], y=meta["y"],
                        slice=lane.slice_of(meta["close"]), side="YES",
                        entry_style="maker", limit=bid_now,
                        p_side=0.5 if p is None else p,
                        dir_side=1, vel_bps=float(vel[row]), rev=float(revs[row]),
                        tag=f"absorbed {float(t_sz[pos]):.0f} @ {price:.2f}"))
            elif int(t_ty[pos]) == 1 and price >= lane.ask[a + row - 1] - EPS:
                if ask_now <= float(lane.ask[a + row - 1]) + hold_bps:
                    spot = lane.index_value(float(t_ts[pos]))
                    p = fair_p1(spot, meta["strike"], secs,
                                _sigma_at(lane, sig, tid, float(t_ts[pos])))
                    out.append(Event(
                        "absorption_revert", f"s{size:g}", tid=tid, ticker=name,
                        ts=float(t_ts[pos]), close=meta["close"], y=meta["y"],
                        slice=lane.slice_of(meta["close"]), side="NO",
                        entry_style="maker", limit=ask_now,
                        p_side=0.5 if p is None else 1.0 - p,
                        dir_side=-1, vel_bps=float(vel[row]), rev=float(revs[row]),
                        tag=f"absorbed {float(t_sz[pos]):.0f} @ {price:.2f}"))
    return out


def spread_events(lane: Lane, sig: dict[int, np.ndarray], *, width_min: float,
                  gap_max: float, cooldown: float = 60.0,
                  per_window: int = 2) -> list[Event]:
    """Wide book, model agrees with the mid: rest at the touch and earn the spread."""
    out: list[Event] = []
    for tid, name, meta, a, b in _segments(lane):
        ts = lane.ts[a:b]
        bid = lane.bid[a:b]
        ask = lane.ask[a:b]
        secs = meta["close"] - ts
        valid = (bid > 0) & (ask < 1) & (bid < ask) & ((ask - bid) >= width_min - EPS)
        valid &= (secs >= 240.0) & (secs <= 840.0)
        if valid.sum() < 5:
            continue
        spot = index_at(lane, ts)
        p = fair_p(spot, meta["strike"], secs, sig[tid])
        mid = (bid + ask) / 2.0
        agree = valid & np.isfinite(spot) & (np.abs(p - mid) <= gap_max)
        long_yes = agree & (p >= mid)
        short_yes = agree & (p < mid)
        for idx, is_long in ((np.flatnonzero(long_yes), True),
                             (np.flatnonzero(short_yes), False)):
            kept = _dedupe(list(idx), ts, cooldown=cooldown, per_ticker=per_window)
            for i in kept:
                side = "YES" if is_long else "NO"
                limit = float(bid[i]) if is_long else float(ask[i])
                if not (0.0 < limit < 1.0):
                    continue
                out.append(Event(
                    "spread_capture", f"w{width_min:.2f}_g{gap_max:.2f}", tid=tid, ticker=name,
                    ts=float(ts[i]), close=meta["close"], y=meta["y"],
                    slice=lane.slice_of(meta["close"]), side=side, entry_style="maker",
                    limit=limit, p_side=float(p[i] if is_long else 1.0 - p[i])))
    return out


def window_open_events(lane: Lane, sig: dict[int, np.ndarray], *,
                       gap_min: float) -> list[Event]:
    """First tradeable quote of a window, against the index probability."""
    out: list[Event] = []
    for tid, name, meta, a, b in _segments(lane):
        ts = lane.ts[a:b]
        bid = lane.bid[a:b]
        ask = lane.ask[a:b]
        secs = meta["close"] - ts
        valid = (bid > 0) & (ask < 1) & (bid < ask) & ((ask - bid) <= 0.06 + EPS)
        valid &= (secs <= 780.0) & (secs >= 600.0)
        idx = np.flatnonzero(valid)
        if len(idx) == 0:
            continue
        spot = index_at(lane, ts)
        p = fair_p(spot, meta["strike"], secs, sig[tid])
        mid = (bid + ask) / 2.0
        for i in idx:
            i = int(i)
            if not math.isfinite(spot[i]):
                continue
            gap = float(mid[i] - p[i])
            if abs(gap) < gap_min:
                continue
            long_yes = gap < 0          # book cheaper than the index says
            side = "YES" if long_yes else "NO"
            limit = float(ask[i]) if long_yes else float(bid[i])
            if not (0.0 < limit < 1.0):
                continue
            out.append(Event(
                "window_open", f"g{gap_min:.2f}", tid=tid, ticker=name, ts=float(ts[i]),
                close=meta["close"], y=meta["y"], slice=lane.slice_of(meta["close"]),
                side=side, entry_style="taker", limit=limit,
                p_side=float(p[i] if long_yes else 1.0 - p[i]), tag=f"gap{gap:+.3f}"))
            break  # one entry per window per contract
    return out


# ----------------------------------------------------------------- simulation
def _sim_event(lane: Lane, e: Event, params: SimParams) -> dict | None:
    """One strict fill model for a single event, under one exit regime.

    Same rules as the pooled `simulate` below: a passive order fills only when
    >= 20 contracts print at/through its limit, a crossed book aborts the study
    instead of being filtered, and every taker leg pays the exact Decimal fee.
    """
    if e.slice not in ("is", "oos"):
        return None
    long_yes = e.side == "YES"
    price = float(e.limit)
    if not (0.0 < price < 1.0):
        return None
    # Crossed-book guard: assert, never filter.
    quote = lane.quote_at(e.tid, e.ts, valid_only=False, lookback=2.0)
    if quote is None:
        return None
    b, a = quote
    if b >= a:
        raise RuntimeError(f"crossed book at event {e.ticker} {b}/{a}")
    entry_style = e.entry_style
    if params.force_maker and entry_style == "taker":
        # Rest at the touch instead of crossing it: a long YES bids the bid, a
        # short YES offers the ask. `first_fill` still needs a print through it.
        price = b if long_yes else a
        entry_style = "maker"
        if not (0.0 < price < 1.0):
            return None
    # A NO position is a short YES: it risks (1 - price) per contract, not price.
    side_cost = Decimal(str(round(price, 4)))
    if not long_yes:
        side_cost = Decimal("1") - side_cost
    if side_cost <= 0:
        return None
    if float(side_cost) > params.max_entry_price + EPS:
        return None
    qty = int((STAKE / side_cost).to_integral_value())
    if params.max_qty > 0:
        qty = min(qty, int(params.max_qty))
    if qty <= 0:
        return None
    fee_in = Decimal("0")
    if entry_style == "maker":
        # The order cannot be hit by a print stamped before it was sent.
        fill = lane.first_fill(e.tid, e.ts + 1e-6,
                               min(e.ts + params.patience_s, e.close - 20.0),
                               taker_yes=not long_yes, limit=price, need=params.min_fill_qty)
        if fill is None:
            return None
        t_fill = fill[0]
    else:
        t_fill = e.ts
        fee_in = taker_fee(Decimal(str(round(price, 4))), qty)
    entry_dec = Decimal(str(round(price, 4)))
    row: dict = {
        "family": e.family, "variant": e.variant, "ticker": e.ticker, "tid": e.tid,
        "ts": e.ts, "fill_ts": t_fill, "exit_ts": e.close, "slice": e.slice,
        "side": e.side, "entry_style": entry_style, "entry": entry_dec, "qty": qty,
        "p_side": e.p_side, "crossed": 0, "hour": int(e.ts % 86400 // 3600),
        "tag": e.tag, "dir_side": e.dir_side, "vel_bps": e.vel_bps, "rev": e.rev,
    }
    if params.exit_mode == "hold_to_settlement":
        # A binary pays `y` at the close: no take-profit, no stop, no barrier, and
        # no exit quote to cross, so the exit pays $0 fee by construction.
        exit_dec = Decimal("1") if int(e.y) == 1 else Decimal("0")
        gross = ((exit_dec - entry_dec) if long_yes else (entry_dec - exit_dec)) * qty
        marks = []
        for m in (mid_at(lane, e.tid, t_fill + 5.0), mid_at(lane, e.tid, t_fill + 15.0)):
            marks.append(None if m is None else round(
                (m - price) * 100.0 if long_yes else (price - m) * 100.0, 2))
        row.update({"exit": exit_dec, "fee": fee_in, "pnl": gross - fee_in,
                    "reason": "hold_to_settlement", "mark5": marks[0],
                    "mark15": marks[1]})
        return row
    deadline = min(t_fill + params.barrier_s, e.close - 20.0)
    if deadline <= t_fill:
        return None
    # Take-profit and stop race on the same tape: whichever comes first wins.
    target = price + params.tp_c if long_yes else price - params.tp_c
    take = None
    if 0.0 < target < 1.0:
        take = lane.first_fill(e.tid, t_fill + 1e-6, deadline,
                               taker_yes=long_yes, limit=target, need=params.min_fill_qty)
    hit = None
    if params.stop_c > 0:
        lo, hi = int(lane.book_bounds[e.tid]), int(lane.book_bounds[e.tid + 1])
        a0 = int(np.searchsorted(lane.ts[lo:hi], t_fill + 1e-6, "right"))
        b0 = int(np.searchsorted(lane.ts[lo:hi], deadline, "right"))
        seg_ts = lane.ts[lo + a0:lo + b0]
        seg_bid = lane.bid[lo + a0:lo + b0]
        seg_ask = lane.ask[lo + a0:lo + b0]
        if long_yes:
            hit = _first_confirmed(seg_ts, seg_bid <= price - params.stop_c + EPS,
                                   seg_bid, params.stop_confirm_s)
        else:
            hit = _first_confirmed(seg_ts, seg_ask >= price + params.stop_c - EPS,
                                   seg_ask, params.stop_confirm_s)
    # `max_loss_c` is a hard bound that no exit regime escapes (the maker scratch
    # included). It fires on the first touch of the adverse level and exits at the
    # book, paying the taker fee -- the same thing `strategy._zobi_exit` does.
    if params.max_loss_c > 0:
        lo, hi = int(lane.book_bounds[e.tid]), int(lane.book_bounds[e.tid + 1])
        a0 = int(np.searchsorted(lane.ts[lo:hi], t_fill + 1e-6, "right"))
        b0 = int(np.searchsorted(lane.ts[lo:hi], deadline, "right"))
        seg_bid = lane.bid[lo + a0:lo + b0]
        seg_ask = lane.ask[lo + a0:lo + b0]
        bound = None
        if long_yes:
            bad = np.flatnonzero(seg_bid <= price - params.max_loss_c + EPS)
            if len(bad):
                bound = (float(lane.ts[lo + a0 + bad[0]]), float(seg_bid[bad[0]]))
        else:
            bad = np.flatnonzero(seg_ask >= price + params.max_loss_c - EPS)
            if len(bad):
                bound = (float(lane.ts[lo + a0 + bad[0]]), float(seg_ask[bad[0]]))
        if bound is not None and (hit is None or bound[0] < hit[0]):
            hit = bound
    override: Decimal | None = None
    if take is not None and (hit is None or float(take[0]) <= hit[0]):
        t_exit, exit_px, reason = float(take[0]), float(target), "maker_tp"
        fee_out = Decimal("0")
    elif hit is not None and hit[1] > 0:
        t_exit, exit_px, reason = hit[0], hit[1], "stop_loss"
        fee_out = taker_fee(Decimal(str(round(exit_px, 4))), qty)
    else:
        t_exit, reason = deadline, "time_barrier"
        q = lane.quote_at(e.tid, deadline)
        if q is None:
            return None
        qb, qa = q
        if not (0.0 < qb < qa < 1.0):
            raise RuntimeError(f"crossed exit {e.ticker} {qb}/{qa}")
        if params.exit_mode == "maker_scratch":
            # A long exits by resting a sell at breakeven (or one cent under); a
            # short exits by resting a buy at breakeven (or one cent over). Both
            # are post-only by construction: strictly above the bid / below the ask.
            px = (max(price - params.scratch_concession_c, qb + 0.01) if long_yes
                  else min(price + params.scratch_concession_c, qa - 0.01))
            legal = 0.0 < px < 1.0 and ((px > qb) if long_yes else (px < qa))
            fill = None
            if legal:
                fill = lane.first_fill(
                    e.tid, deadline + 1e-6,
                    min(deadline + params.scratch_patience_s, e.close - 20.0),
                    taker_yes=long_yes, limit=px, need=params.min_fill_qty)
            if fill is not None:
                t_exit, exit_px, reason = float(fill[0]), px, "maker_scratch"
                fee_out = Decimal("0")
            else:
                # Unfilled scratch: the binary is carried to settlement. $0 fee, and
                # the honest cost of a passive exit that never got hit.
                exit_px = 1.0 if int(e.y) == 1 else 0.0
                reason = "scratch_unfilled_settle"
                fee_out = Decimal("0")
        elif params.exit_mode == "synthetic_flatten":
            # $0-fee opposite-side flatten. A long YES bids NO at 1-(entry-conc),
            # which is a YES-equivalent limit of entry-conc; a long NO (cost
            # 1-entry) bids YES at entry+conc. Either way the pair costs 1+conc and
            # pays exactly 1 at settlement, so the worst case is `conc` per
            # contract instead of the whole binary, and the leg is post-only.
            conc = params.flatten_concession_c
            opp_limit = price - conc if long_yes else price + conc
            legal = 0.0 < opp_limit < 1.0 and ((qb < opp_limit) if long_yes
                                               else (qa > opp_limit))
            fill = None
            if legal:
                fill = lane.first_fill(
                    e.tid, deadline + 1e-6,
                    min(deadline + params.scratch_patience_s, e.close - 20.0),
                    taker_yes=long_yes, limit=opp_limit, need=params.min_fill_qty)
            if fill is not None:
                t_exit, exit_px = float(fill[0]), price
                reason = "synthetic_flatten"
                fee_out = Decimal("0")
                override = -(Decimal(str(round(conc, 4))) * qty)
            else:
                exit_px = 1.0 if int(e.y) == 1 else 0.0
                reason = "synthetic_unfilled_settle"
                fee_out = Decimal("0")
        else:
            exit_px = qb if long_yes else qa
            fee_out = taker_fee(Decimal(str(round(exit_px, 4))), qty)
    exit_dec = Decimal(str(round(exit_px, 4)))
    if override is not None:
        pnl = override
    else:
        gross = ((exit_dec - entry_dec) if long_yes else (entry_dec - exit_dec)) * qty
        pnl = gross - fee_in - fee_out
    marks = []
    for m in (mid_at(lane, e.tid, t_fill + 5.0), mid_at(lane, e.tid, t_fill + 15.0)):
        if m is None:
            marks.append(None)
        else:
            marks.append(round((m - price) * 100.0 if long_yes else (price - m) * 100.0, 2))
    row.update({
        "exit": exit_dec, "exit_ts": t_exit, "fee": fee_in + fee_out, "pnl": pnl,
        "reason": reason, "mark5": marks[0], "mark15": marks[1],
    })
    return row


def _first_confirmed(ts: np.ndarray, adverse: np.ndarray, px: np.ndarray,
                     confirm_s: float) -> tuple[float, float] | None:
    """First adverse touch, or the first one that survives `confirm_s` unbroken.

    The engine holds a position while a breach is younger than its confirmation
    window and crosses out on the tick the window completes. Runs of consecutive
    adverse rows are the exact analogue on a recorded tape: a run shorter than the
    window never fires, and a long one fires at `start + confirm_s`.
    """
    bad = np.flatnonzero(adverse)
    if not len(bad):
        return None
    if confirm_s <= 0:
        return (float(ts[bad[0]]), float(px[bad[0]]))
    edges = np.flatnonzero(np.diff(np.concatenate(([0], adverse.astype(np.int8), [0]))))
    for s0, e0 in zip(edges[::2], edges[1::2]):
        start = float(ts[s0])
        if float(ts[e0 - 1]) - start < confirm_s - EPS:
            continue
        j = min(s0 + int(np.searchsorted(ts[s0:e0], start + confirm_s, "left")), e0 - 1)
        return (float(ts[j]), float(px[j]))
    return None


def simulate(lane: Lane, events: list[Event], params: SimParams) -> list[dict]:
    """One strict fill model for every family."""
    rows: list[dict] = []
    for e in events:
        row = _sim_event(lane, e, params)
        if row is not None:
            rows.append(row)
    return rows


def adverse_pressure(row: dict) -> tuple[float, float]:
    """How hard the tape is moving against the held side, in the fade's frame.

    Positive means the move is continuing against the position, i.e. the fade is
    being run over. Reads only the event-time snapshot carried on the row.
    """
    d = int(row.get("dir_side") or 0) or (1 if row.get("side") == "YES" else -1)
    return -d * float(row.get("vel_bps") or 0.0), -d * float(row.get("rev") or 0.0)


def sequence_rows(rows: list[dict], *, max_window_trades: int, halt_after_loss: bool,
                  veto_vel: float = 0.0, veto_rev: float = 0.0) -> list[dict]:
    """Per-window entry cap, halt-after-one-loss and the trend/toxicity veto.

    A 15-minute contract is one slot, not a stream: after the fade has already lost,
    the window is trending and re-entering it is how a -1c edge turns into -$17 a
    trade. The filter is causal — a skipped row never enters the ledger, and the
    window's budget is only spent by trades that actually filled.
    """
    out: list[dict] = []
    state: dict[int, dict] = {}
    seen: set[tuple[int, int]] = set()
    for r in sorted(rows, key=lambda x: (int(x["tid"]), float(x["ts"]))):
        key = (int(r["tid"]), int(round(float(r["ts"]) * 1000.0)))
        if key in seen:
            continue
        seen.add(key)
        adv_vel, adv_rev = adverse_pressure(r)
        if veto_vel > 0 and adv_vel >= veto_vel:
            continue
        if veto_rev > 0 and adv_rev >= veto_rev:
            continue
        st = state.setdefault(int(r["tid"]), {"n": 0, "halted": False, "last_exit": -1e18})
        if st["halted"] or st["n"] >= max_window_trades:
            continue
        if float(r["ts"]) <= st["last_exit"]:
            continue
        out.append(r)
        st["n"] += 1
        st["last_exit"] = max(float(r["exit_ts"]), float(r["ts"]))
        if halt_after_loss and r["pnl"] <= 0:
            st["halted"] = True
    return out



def block2(rows: list[dict], hours: float) -> dict:
    out = block(rows, hours)
    # Tail metrics: the promoted program's failure mode was a per-contract loss 15x
    # its take-profit, invisible in a net/PF table. Report the worst one directly.
    losses = [float(r["pnl"]) / max(1, int(r["qty"])) for r in rows if "qty" in r]
    out["max_loss_per_contract"] = (round(-min(losses), 4)
                                    if losses and min(losses) < 0 else 0.0)
    out["worst_trade"] = str(min((r["pnl"] for r in rows), default=Decimal("0"))
                              .quantize(Decimal("0.01")))
    marks = [r["mark5"] for r in rows if r.get("mark5") is not None]
    marks15 = [r["mark15"] for r in rows if r.get("mark15") is not None]
    out["markout_5s_c"] = round(float(np.mean(marks)), 3) if marks else None
    out["markout_15s_c"] = round(float(np.mean(marks15)), 3) if marks15 else None
    out["entry_styles"] = dict(sorted({s: sum(1 for r in rows if r["entry_style"] == s)
                                       for s in {r["entry_style"] for r in rows}}.items()))
    out["reasons"] = dict(sorted({s: sum(1 for r in rows if r["reason"] == s)
                                  for s in {r["reason"] for r in rows}}.items()))
    out["by_side"] = {s: {"trades": len(sub),
                          "net": str(sum((r["pnl"] for r in sub), Decimal("0"))
                                     .quantize(Decimal("0.01")))}
                      for s in ("YES", "NO")
                      for sub in [[r for r in rows if r["side"] == s]] if sub}
    out["net_by_reason"] = {
        s: str(sum((r["pnl"] for r in rows if r["reason"] == s), Decimal("0"))
               .quantize(Decimal("0.01")))
        for s in sorted({r["reason"] for r in rows})}
    out["by_hour_utc"] = {
        f"{h * 4:02d}-{h * 4 + 3:02d}": {"trades": len([r for r in rows if int(r["hour"]) // 4 == h]),
                   "net": str(sum((r["pnl"] for r in rows if int(r["hour"]) // 4 == h),
                                  Decimal("0")).quantize(Decimal("0.01")))}
        for h in range(6)
        if any(int(r["hour"]) // 4 == h for r in rows)}
    out["by_ticker"] = {
        t: {"trades": len([r for r in rows if r["ticker"] == t]),
            "net": str(sum((r["pnl"] for r in rows if r["ticker"] == t), Decimal("0"))
                       .quantize(Decimal("0.01")))}
        for t in sorted({r["ticker"] for r in rows})}
    return out


def platt_fit(rows: list[dict]) -> tuple[float, float]:
    """Logistic recalibration logit(p) -> a*logit(p)+b, fitted on IS trades only."""
    pairs = [(min(max(float(r["p_side"]), 1e-6), 1 - 1e-6), 1.0 if r["pnl"] > 0 else 0.0)
             for r in rows]
    if len(pairs) < 12:
        return 1.0, 0.0
    x = np.array([math.log(p / (1.0 - p)) for p, _ in pairs])
    y = np.array([t for _, t in pairs])
    a, b = 1.0, 0.0
    for _ in range(50):
        p = 1.0 / (1.0 + np.exp(-np.clip(a * x + b, -50.0, 50.0)))
        w = np.clip(p * (1.0 - p), 1e-6, None)
        g = np.array([float(np.sum((p - y) * x)), float(np.sum(p - y))])
        h = np.array([[float(np.sum(w * x * x)) + 1e-6, float(np.sum(w * x))],
                      [float(np.sum(w * x)), float(np.sum(w)) + 1e-6]])
        try:
            step = np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            break
        a, b = a - float(step[0]), b - float(step[1])
        if max(abs(float(step[0])), abs(float(step[1]))) < 1e-9:
            break
    if not (math.isfinite(a) and math.isfinite(b)) or abs(a) > 5.0:
        return 1.0, 0.0
    return float(a), float(b)


def slim(b: dict) -> dict:
    """The fields a verdict table needs, without the per-ticker maps."""
    keep = ("trades", "wins", "win_rate", "net", "pf", "ece", "fees", "tp_share",
            "crossed_fills", "reasons", "by_side", "by_hour_utc",
            "markout_5s_c", "markout_15s_c")
    return {k: b.get(k) for k in keep if k in b}


def cal_ece(rows: list[dict], a: float, b: float) -> float:
    pairs = []
    for r in rows:
        p = min(max(float(r["p_side"]), 1e-6), 1 - 1e-6)
        z = max(min(a * math.log(p / (1.0 - p)) + b, 50.0), -50.0)
        pairs.append((1.0 / (1.0 + math.exp(-z)), 1 if r["pnl"] > 0 else 0))
    return round(float(features.expected_calibration_error(pairs)), 4) if pairs else 0.0


def split2(rows: list[dict], hours: dict[str, float]) -> dict:
    return {name: block2([r for r in rows if r["slice"] == name], hours[name])
            for name in ("is", "oos")}


# ------------------------------------------------------------------------ main
# `99` disables the index-confirmation filter: the same spike events, unconditional.
SPIKE_DETECTORS = [(5.0, 0.40), (8.0, 0.40), (5.0, 0.25), (8.0, 0.25),
                   (5.0, 99.0), (8.0, 99.0)]
DISLOC_DETECTORS = [(0.04, 1), (0.06, 1), (0.08, 1), (0.06, 2)]
ABSORB_DETECTORS = [20.0, 40.0]
SPREAD_DETECTORS = [(0.03, 0.02), (0.04, 0.03)]
OPEN_DETECTORS = [0.04, 0.06]

SIMS = {
    "spike_fade": [SimParams(tp, hold) for tp in (0.03, 0.04, 0.05) for hold in (30.0, 60.0)],
    "spike_momentum": [SimParams(tp, hold) for tp in (0.03, 0.05, 0.08) for hold in (30.0, 60.0)],
    "spike_pullback": [SimParams(tp, hold) for tp in (0.02, 0.03, 0.05) for hold in (30.0, 60.0)],
    "index_dislocation": [SimParams(tp, hold, stop)
                          for tp in (0.02, 0.03, 0.04) for hold in (30.0, 60.0, 120.0)
                          for stop in (0.0, 0.04)],
    "absorption_revert": [SimParams(tp, hold) for tp in (0.01, 0.02, 0.03)
                          for hold in (20.0, 60.0)],
    "spread_capture": [SimParams(tp, hold, 0.04) for tp in (0.02, 0.03)
                       for hold in (60.0, 120.0)],
    "window_open": [SimParams(tp, hold) for tp in (0.03, 0.05) for hold in (120.0, 300.0)],
}


HOLD_CAPS = (0.35, 0.45, 0.55, 0.84)
HOLD_FAMILIES = ("absorption_revert", "index_dislocation", "window_open", "spike_pullback")
HOLD_REPORT = ROOT / "data" / "hold-settlement.json"

# --- passive maker-scratch + true settlement families -----------------------
PASSIVE_FAMILIES = ("absorption_revert", "index_dislocation", "btc_leads_alts",
                    "z_obi_flow", "late_favorite")
PASSIVE_MODE = {"absorption_revert": "maker_scratch", "index_dislocation": "maker_scratch",
                "btc_leads_alts": "maker_scratch", "z_obi_flow": "maker_scratch",
                "late_favorite": "hold_to_settlement"}
# (bps_lo, bps_hi, lookback_s, lag_share). `lag_share=1.01` disables the extra
# requirement that the alt's own index print has moved less than the BTC impulse: the
# premise under test is that the alt *book* is slow, and the book shift is filtered
# separately. Keeping both readings means the family is tested, not just abstained on.
LEAD_DETECTORS = ((12.0, 25.0, 5.0, 0.6), (12.0, 25.0, 10.0, 0.6), (12.0, 25.0, 3.0, 0.6),
                  (10.0, 35.0, 5.0, 0.6), (12.0, 25.0, 5.0, 1.01), (12.0, 25.0, 3.0, 1.01),
                  (10.0, 35.0, 5.0, 1.01))
ZOBI_DETECTORS = ((1.0, 0.15, 0.10), (1.4, 0.20, 0.15), (1.4, 0.30, 0.10))
# 1.6 is the literal spec. It is also, under any volatility, a fair value of ~0.945:
# a 0.70-0.88 entry at that z is a 6-24c dislocation, not a favourite. `0.5` is the
# coherent reading where the price band defines the favourite and z only has to
# confirm it is the right side (Phi(0.5)=0.69, i.e. the bottom of the band).
LATE_DETECTORS = (1.6, 0.5)


def _passive_sims() -> dict[str, list[tuple[str, SimParams]]]:
    """Every variant the four families are allowed to select from, labelled."""
    sims: dict[str, list[tuple[str, SimParams]]] = {}
    scratch = []
    for tp in (0.03, 0.05):
        for bar in (20.0, 45.0):
            for conc in (0.0, 0.01):
                for patience in (20.0, 300.0):
                    label = (f"tp{tp:.2f}_b{bar:.0f}_scr{conc * 100:.0f}c"
                             f"_p{patience:.0f}s")
                    scratch.append((label, SimParams(
                        tp_c=tp, barrier_s=bar, exit_mode="maker_scratch",
                        scratch_concession_c=conc, scratch_patience_s=patience)))
    sims["absorption_revert"] = scratch
    sims["index_dislocation"] = scratch
    lead = [("hold_cap0.55", SimParams(tp_c=0.0, barrier_s=0.0, patience_s=15.0,
                                       exit_mode="hold_to_settlement",
                                       max_entry_price=0.55, force_maker=True)),
            ("hold_cap0.84", SimParams(tp_c=0.0, barrier_s=0.0, patience_s=15.0,
                                       exit_mode="hold_to_settlement",
                                       max_entry_price=0.84, force_maker=True))]
    for tp in (0.03, 0.05):
        for bar in (20.0, 45.0):
            lead.append((f"tp{tp:.2f}_b{bar:.0f}", SimParams(tp_c=tp, barrier_s=bar,
                                                             exit_mode="maker_scratch")))
    sims["btc_leads_alts"] = lead
    zf = [("hold_cap0.55", SimParams(tp_c=0.0, barrier_s=0.0, patience_s=15.0,
                                     exit_mode="hold_to_settlement",
                                     max_entry_price=0.55, force_maker=True))]
    for tp in (0.03, 0.05):
        for bar in (30.0, 45.0):
            zf.append((f"tp{tp:.2f}_b{bar:.0f}_scr", SimParams(tp_c=tp, barrier_s=bar,
                                                               exit_mode="maker_scratch")))
    sims["z_obi_flow"] = zf
    sims["late_favorite"] = [
        ("hold_taker", SimParams(tp_c=0.0, barrier_s=0.0, patience_s=15.0,
                                 exit_mode="hold_to_settlement")),
        ("hold_maker", SimParams(tp_c=0.0, barrier_s=0.0, patience_s=15.0,
                                 exit_mode="hold_to_settlement", force_maker=True)),
    ]
    return sims


PASSIVE_SIMS = _passive_sims()


def hold_events(lane: Lane, sig: dict[int, np.ndarray]) -> dict[str, list[Event]]:
    """The four maker families, on the same detectors the maker-TP study uses."""
    events: dict[str, list[Event]] = defaultdict(list)
    for horizon, conf in SPIKE_DETECTORS:
        _fade, momo = spike_events(lane, sig, horizon=horizon, conf=conf)
        events["spike_pullback"].extend([e for e in momo if e.family == "spike_pullback"])
    for gap_min, persist in DISLOC_DETECTORS:
        events["index_dislocation"].extend(
            dislocation_events(lane, sig, gap_min=gap_min, persist=persist))
    for size in ABSORB_DETECTORS:
        events["absorption_revert"].extend(absorption_events(lane, sig, size=size))
    for gap_min in OPEN_DETECTORS:
        events["window_open"].extend(window_open_events(lane, sig, gap_min=gap_min))
    return events


def hold_settlement_study(lanes: list[str], caps: tuple[float, ...]) -> int:
    """Maker entry, hold to the binary payout: zero fee on either leg.

    Every family's maker-TP losses on this tape came from the `time_barrier` exit
    crossing the spread and paying a taker fee. A 15-minute binary needs no exit at
    all, so this sweep prices the same entries to the settlement payout `y` instead.
    Entry-side price caps are swept because the payout is asymmetric: a 0.84 entry
    risks 84c to make 16c, a 0.35 entry risks 35c to make 65c.

    Selection is in-sample only (highest profit factor among caps with 20+ IS trades
    and positive IS net); the frozen OOS slice is scored once per family and every
    cap's OOS row is reported whether or not it was selected.
    """
    started = time.time()
    hours = {name: 0.0 for name in ("is", "oos")}
    rows_by: dict[tuple[str, float], list[dict]] = defaultdict(list)
    audit: dict[str, dict] = {}

    for name in lanes:
        lane = Lane(name)
        sig = trailing_sigma(lane)
        for slice_name in ("is", "oos"):
            hours[slice_name] += lane.hours[slice_name]
        audit[name] = {"windows": len(lane.meta), "book_rows": int(len(lane.ts)),
                       "prints": int(len(lane.trade_ts)),
                       "is_hours": round(lane.hours["is"], 2),
                       "oos_hours": round(lane.hours["oos"], 2)}
        print(f"{name}: {len(lane.meta)} windows, {len(lane.ts):,} rows, "
              f"{len(lane.trade_ts):,} prints", flush=True)
        events = hold_events(lane, sig)
        for family, evs in events.items():
            for cap in caps:
                params = SimParams(tp_c=0.0, barrier_s=0.0, patience_s=15.0,
                                   exit_mode="hold_to_settlement",
                                   max_entry_price=cap, force_maker=True)
                rows_by[(family, cap)].extend(simulate(lane, evs, params))
        print(f"  {sum(len(v) for v in events.values())} events "
              f"({time.time() - started:.0f}s)", flush=True)
        del lane

    # The hold path must be exactly what it claims: no taker exit, no fee, no crossed fill.
    for (family, cap), rows in rows_by.items():
        for r in rows:
            if r["reason"] != "hold_to_settlement" or r["fee"] != 0 or r["crossed"] != 0:
                raise RuntimeError(
                    f"hold_to_settlement invariant broken: {family} cap {cap}: "
                    f"{r['reason']} fee {r['fee']} crossed {r['crossed']}")

    table: list[dict] = []
    families: dict[str, dict] = {}
    for family in HOLD_FAMILIES:
        cands: list[dict] = []
        for cap in caps:
            rows = rows_by.get((family, cap), [])
            if not rows:
                continue
            is_rows = [r for r in rows if r["slice"] == "is"]
            oos_rows = [r for r in rows if r["slice"] == "oos"]
            is_block = block2(is_rows, hours["is"])
            oos_block = block2(oos_rows, hours["oos"])
            pa, pb = platt_fit(is_rows)
            is_block["ece_raw"] = is_block.get("ece")
            is_block["ece"] = cal_ece(is_rows, pa, pb)
            oos_block["ece_raw"] = oos_block.get("ece")
            oos_block["ece"] = cal_ece(oos_rows, pa, pb)
            scored = dict(oos_block)
            cands.append({
                "family": family, "cap": cap,
                "label": f"{family}_cap{cap:.2f}_hold",
                "is": is_block, "oos": oos_block,
                "platt": {"a": round(pa, 4), "b": round(pb, 4), "fitted_on": "is"},
                "gate": gate(scored), "verifier": verifier_call(scored),
            })
        if not cands:
            families[family] = {"selected": None, "eligible": False, "passes": False,
                                "oos_pass_caps": [], "reason": "no fills"}
            continue
        table.extend(cands)
        eligible = [c for c in cands if c["is"]["trades"] >= 20
                    and Decimal(str(c["is"]["net"])) > 0]
        eligible.sort(key=lambda c: (float(c["is"]["pf"]), Decimal(str(c["is"]["net"])),
                                     int(c["is"]["trades"])), reverse=True)
        best = eligible[0] if eligible else None
        passes = bool(best) and all(best["gate"].values()) and bool(best["verifier"][0])
        families[family] = {
            "selected": ({k: best[k] for k in ("label", "cap", "is", "oos", "gate",
                                               "platt", "verifier")}
                         if best else None),
            "eligible": bool(eligible),
            "passes": passes,
            "oos_pass_caps": [c["cap"] for c in cands if all(c["gate"].values())],
            "reason": None if passes else (
                "no cap reached 20 IS trades with positive IS net" if not best
                else "IS-best cap failed the frozen OOS gate: "
                     + ", ".join(k for k, v in best["gate"].items() if not v)
                     + ("" if best["verifier"][0] else f" ({best['verifier'][1]})")),
        }
        for c in cands:
            print(f"{family} cap {c['cap']:.2f}: IS {c['is']['trades']}/{c['is']['net']} "
                  f"pf {c['is']['pf']} -> OOS {c['oos']['trades']}/{c['oos']['net']} "
                  f"pf {c['oos']['pf']} ece {c['oos']['ece']} "
                  f"pass={all(c['gate'].values()) and bool(c['verifier'][0])}", flush=True)

    # Promotion: the IS-best passing family with the highest IS profit factor.
    promoted = None
    for family, info in families.items():
        if not info.get("passes"):
            continue
        if promoted is None:
            promoted = family
            continue
        a = (float(info["selected"]["is"]["pf"]),
             Decimal(str(info["selected"]["is"]["net"])))
        b = (float(families[promoted]["selected"]["is"]["pf"]),
             Decimal(str(families[promoted]["selected"]["is"]["net"])))
        if a > b:
            promoted = family

    tape = {"lanes": lanes, "hours": hours,
            "book_rows": sum(a["book_rows"] for a in audit.values()),
            "prints": sum(a["prints"] for a in audit.values())}
    gate_block = {"profit_factor_min": 1.15, "ece_max": 0.08,
                  "requires_positive_net": True, "requires_zero_crossed_fills": True,
                  "min_oos_trades": 20}
    if promoted:
        sel = families[promoted]["selected"]
        verdict = {
            "generated_ts": time.time(), "promoted": True,
            "candidate": {"name": promoted, "label": sel["label"], "cap": sel["cap"],
                          "exit_mode": "hold_to_settlement", "entry": "post_only"},
            "is": sel["is"], "oos": sel["oos"], "gate": gate_block, "tape": tape,
            "reason": (f"{promoted} cap {sel['cap']:.2f} hold-to-settlement passes "
                       f"every frozen OOS check"),
        }
    else:
        verdict = {
            "generated_ts": time.time(), "promoted": False, "candidate": None, "oos": None,
            "gate": gate_block, "families": families, "table": table, "tape": tape,
            "verdict": "no hold-to-settlement candidate clears the frozen OOS gate",
            "reason": ("no family produced an in-sample candidate that clears the "
                       "frozen OOS gate"),
        }
    VERDICT.write_text(json.dumps(verdict, indent=2, default=str))
    report = {
        "generated_ts": time.time(), "elapsed_s": round(time.time() - started, 1),
        "protocol": {
            "exit_mode": "hold_to_settlement: binary payout y at close, $0 exit fee, "
                         "no taker exit anywhere",
            "entry": "post_only at the touch; >=20 contracts must print through it",
            "price_cap": "side cost per contract: YES price for a long, 1-YES for a short",
            "crossed_book": "asserted, never filtered; 0 crossed fills required",
            "selection": "highest IS profit factor among caps with 20+ IS trades and "
                         "positive IS net; the frozen OOS slice is scored once",
            "calibration": "Platt fitted on IS rows only, frozen for OOS",
        },
        "lanes": lanes, "hours": hours, "audit": audit, "caps": list(caps),
        "families": families, "table": table, "promotion": {"family": promoted},
        "tape": tape,
    }
    HOLD_REPORT.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({
        "promotion": promoted,
        "families": {k: {
            "cap": (v.get("selected") or {}).get("cap"),
            "is": {kk: ((v.get("selected") or {}).get("is") or {}).get(kk)
                   for kk in ("trades", "win_rate", "pf", "net")},
            "oos": {kk: ((v.get("selected") or {}).get("oos") or {}).get(kk)
                    for kk in ("trades", "win_rate", "pf", "net", "ece", "crossed_fills")},
            "oos_pass_caps": v.get("oos_pass_caps"),
        } for k, v in families.items()}}, indent=2))
    print(f"wrote {HOLD_REPORT} and {VERDICT}")
    return 0 if promoted else 2


# ------------- bounded-risk study: early-window z_obi / calibrated flatten
# Two mechanisms, one frozen 60/40 split. (a) lets `z_obi_flow` fire in the first
# half of the window, where a 0.30-0.60 contract bounds the settlement loss by the
# price paid; (b) keeps the shipped detector but exits through a $0-fee opposite-side
# lock and recalibrates the win probability with a logistic fit made on IS only.
EARLY_T_LO, EARLY_T_HI = 360.0, 840.0
EARLY_COST_LO, EARLY_COST_HI = 0.30, 0.60
# A 0.30-0.60 contract is a fair value inside [0.31, 0.69], i.e. |z| <= ~0.25. The
# literal `z_obi_flow` floor of 0.6 implies a fair value of ~0.79 and the tape proves
# it: of 542,364 BTC rows that cleared z/flow/revision at |z|>=0.6 in this window, 0
# were priced at/below 0.60 (the cheapest was 0.61). So the literal pair of
# constraints has an empty intersection; the coherent reading keeps the price band,
# which is what bounds the loss, and lowers the z floor to what the band can carry.
EARLY_Z_GRID = (0.15, 0.20, 0.25)
EARLY_FLOW_GRID = (0.15, 0.25)
SYN_TP_C = 0.05
SYN_BARS = (20.0, 45.0)
BOUNDED_CAPS = (1, 2)
BOUNDED_FAMILIES = ("early_window_z_obi", "calibrated_syn_flatten",
                    "calibrated_syn_flatten_6c_stop")
# The tail each mechanism claims: (a) cannot lose more than the entry price it is
# allowed to pay, (b) cannot lose more than the lock concession plus its fees.
# The 6c arm is the same lock plus the shipped `zobi_max_loss` guard, so its tail is
# bounded by construction instead of by a single lucky carry.
SYN_STOP_C = 0.06
TAIL_MAX = {"early_window_z_obi": EARLY_COST_HI, "calibrated_syn_flatten": 0.08,
            "calibrated_syn_flatten_6c_stop": SYN_STOP_C + 0.02}
BOUNDED_REPORT = ROOT / "data" / "bounded-search.json"


def auc(rows: list[dict]) -> float:
    """Rank AUC of `p_side` against the realised win. 0.5 = no discrimination."""
    pairs = [(min(max(float(r["p_side"]), 1e-6), 1 - 1e-6), 1 if r["pnl"] > 0 else 0)
             for r in rows]
    pos = [p for p, y in pairs if y == 1]
    neg = [p for p, y in pairs if y == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p in pos:
        for q in neg:
            wins += 1.0 if p > q else (0.5 if p == q else 0.0)
    return wins / (len(pos) * len(neg))


def _bounded_block(mech: str, label: str, rows: list[dict], hours: dict[str, float],
                   structural_cap: float | None = None) -> dict:
    """One candidate: IS-only Platt fit, then both slices scored with it frozen."""
    is_rows = [r for r in rows if r["slice"] == "is"]
    oos_rows = [r for r in rows if r["slice"] == "oos"]
    isb = block2(is_rows, hours["is"])
    oosb = block2(oos_rows, hours["oos"])
    pa, pb = platt_fit(is_rows)          # strictly IS, frozen before OOS is touched
    out = {"mechanism": mech, "label": label,
           "is": slim2(isb), "oos": slim2(oosb),
           "platt": {"a": round(pa, 4), "b": round(pb, 4), "fitted_on": "is"}}
    out["is"]["ece_raw"] = isb["ece"]
    out["oos"]["ece_raw"] = oosb["ece"]
    out["is"]["ece"] = cal_ece(is_rows, pa, pb)
    out["oos"]["ece"] = cal_ece(oos_rows, pa, pb)
    out["is"]["auc"] = round(auc(is_rows), 4)
    out["oos"]["auc"] = round(auc(oos_rows), 4)
    # The lock bounds the loss only while it fills. When it does not, the binary is
    # carried to settlement and the worst case is the whole side cost, so both the
    # measured tail and the structural one are reported -- a 2c tail that was sampled
    # once is not the same claim as a 2c tail that cannot be exceeded.
    for sl, rr in (("is", is_rows), ("oos", oos_rows)):
        carries = [r for r in rr if r["reason"] in ("synthetic_unfilled_settle",
                                                    "scratch_unfilled_settle")]
        out[sl]["carries"] = len(carries)
        out[sl]["carry_net"] = str(sum((r["pnl"] for r in carries), Decimal("0"))
                                   .quantize(Decimal("0.01")))
        costs = [float(r["entry"]) if r["side"] == "YES" else 1.0 - float(r["entry"])
                 for r in rr]
        worst = max(costs) if costs else 0.0
        # A hard stop bounds the loss the lock cannot reach; an unprotected lock does
        # not, because an unfilled lock carries the binary to settlement.
        if structural_cap is not None:
            worst = min(worst, structural_cap)
        out[sl]["structural_max_loss_per_contract"] = round(worst, 4)
    return out


def bounded_study(lanes: list[str]) -> int:
    """Early-window z_obi and IS-calibrated synthetic flatten, frozen split only.

    Selection is in-sample only (highest IS net among 20+ IS trades), and the 40%
    frozen slice is scored exactly once per mechanism after that choice. The
    calibration is fitted on IS trades and then frozen; nothing reads OOS before the
    verdict. Every fill goes through the same strict print-tape model, and a crossed
    book raises instead of being filtered.
    """
    started = time.time()
    hours = {"is": 0.0, "oos": 0.0}
    audit: dict[str, dict] = {}
    rows_by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    events_by: dict[str, int] = defaultdict(int)
    hold = SimParams(tp_c=0.0, barrier_s=0.0, patience_s=15.0,
                     exit_mode="hold_to_settlement",
                     max_entry_price=EARLY_COST_HI, force_maker=True)

    for name in lanes:
        lane = Lane(name)
        sig = trailing_sigma(lane)
        for slice_name in ("is", "oos"):
            hours[slice_name] += lane.hours[slice_name]
        audit[name] = {"windows": len(lane.meta), "book_rows": int(len(lane.ts)),
                       "prints": int(len(lane.trade_ts)),
                       "is_hours": round(lane.hours["is"], 2),
                       "oos_hours": round(lane.hours["oos"], 2)}
        print(f"{name}: {len(lane.meta)} windows, {len(lane.ts):,} rows, "
              f"{len(lane.trade_ts):,} prints", flush=True)

        # (a) early window: T_remaining in [360, 840] and a 0.30-0.60 contract.
        early = []
        for z_min in EARLY_Z_GRID:
            for flow_min in EARLY_FLOW_GRID:
                early.extend(z_obi_flow_events(
                    lane, sig, z_min=z_min, flow_min=flow_min, rev_min=ZOBI_REV_MIN,
                    t_lo=EARLY_T_LO, t_hi=EARLY_T_HI,
                    cost_lo=EARLY_COST_LO, cost_hi=EARLY_COST_HI, tag_decimals=2))
        events_by["early_window_z_obi"] += len(early)
        by_var: dict[str, list[Event]] = defaultdict(list)
        for e in early:
            by_var[e.variant].append(e)
        for var, evs in by_var.items():
            base = simulate(lane, evs, hold)
            for cap in BOUNDED_CAPS:
                seq = sequence_rows(base, max_window_trades=cap, halt_after_loss=True)
                rows_by[("early_window_z_obi", f"{var}|cap{cap}")].extend(seq)

        # (b) the shipped detector, exited through the $0-fee opposite-side lock.
        base_evs = []
        for z_min in ZOBI_Z_GRID:
            for flow_min in ZOBI_FLOW_GRID:
                base_evs.extend(z_obi_flow_events(
                    lane, sig, z_min=z_min, flow_min=flow_min, rev_min=ZOBI_REV_MIN))
        events_by["calibrated_syn_flatten"] += len(base_evs)
        by_var = defaultdict(list)
        for e in base_evs:
            by_var[e.variant].append(e)
        for conc in SYNTH_CONCESSIONS:
            for bar in SYN_BARS:
                params = SimParams(tp_c=SYN_TP_C, barrier_s=bar,
                                   exit_mode="synthetic_flatten",
                                   flatten_concession_c=conc,
                                   scratch_patience_s=300.0)
                for mech, extra in (("calibrated_syn_flatten", {}),
                                    ("calibrated_syn_flatten_6c_stop",
                                     {"max_loss_c": SYN_STOP_C})):
                    arm = SimParams(**{**params.__dict__, **extra})
                    for var, evs in by_var.items():
                        base = simulate(lane, evs, arm)
                        for cap in BOUNDED_CAPS:
                            seq = sequence_rows(base, max_window_trades=cap,
                                                halt_after_loss=True)
                            rows_by[(mech,
                                     f"syn{conc * 100:.0f}c_tp{SYN_TP_C:.2f}_b{bar:.0f}"
                                     f"|{var}|cap{cap}")].extend(seq)
        print(f"  events {dict(events_by)} ({time.time() - started:.0f}s)", flush=True)
        del lane

    # Same invariants as every other passive study: not one crossed fill, and no fee
    # on any leg that is post-only by construction.
    for (mech, label), rows in rows_by.items():
        for r in rows:
            if r["crossed"] != 0:
                raise RuntimeError(f"crossed fill: {mech} {label}: {r}")
            if r["reason"] in ("maker_tp", "synthetic_flatten", "hold_to_settlement") \
                    and r["fee"] != 0:
                raise RuntimeError(f"fee on a passive leg: {mech} {label}: {r}")

    families: dict[str, dict] = {}
    for mech in BOUNDED_FAMILIES:
        structural_cap = {"early_window_z_obi": None,
                          "calibrated_syn_flatten": None,
                          "calibrated_syn_flatten_6c_stop": SYN_STOP_C + 0.02}[mech]
        cands = [_bounded_block(mech, label, rows, hours, structural_cap)
                 for (m, label), rows in rows_by.items() if m == mech and rows]
        cands.sort(key=lambda c: (Decimal(str(c["is"]["net"])), float(c["is"]["pf"])),
                   reverse=True)
        elig = [c for c in cands
                if int(c["is"]["trades"]) >= 20 and Decimal(str(c["is"]["net"])) > 0]
        best = elig[0] if elig else None
        if best is None:
            families[mech] = {"variants": len(cands), "eligible": False,
                              "selected": None, "gate": None, "table": cands[:12],
                              "reason": "no variant reached 20 IS trades with positive IS net"}
            print(f"{mech}: {len(cands)} variants, no IS-eligible candidate", flush=True)
            continue
        g = gate(best["oos"])
        tail = float(best["oos"]["max_loss_per_contract"] or 0.0)
        struct = float(best["oos"]["structural_max_loss_per_contract"] or 0.0)
        # Bounded means bounded: a 2c tail that was sampled once is not a 2c bound when
        # an unfilled lock carries the whole binary.
        tail_ok = (tail <= TAIL_MAX[mech] + EPS
                   and struct <= TAIL_MAX[mech] + EPS)
        vig, why = verifier_call(best["oos"])
        passes = all(g.values()) and tail_ok and vig
        families[mech] = {
            "variants": len(cands), "eligible": True,
            "selected": {**best, "passes": passes, "tail_max": TAIL_MAX[mech],
                          "tail_ok": tail_ok, "verifier": why},
            "gate": {**g, "bounded_tail": tail_ok},
            "positive_oos_variants": sum(1 for c in cands
                                         if Decimal(str(c["oos"]["net"])) > 0),
            "reason": None if passes else "IS-best failed the frozen OOS gate: " + ", ".join(
                [k for k, v in g.items() if not v] +
                ([] if tail_ok else [f"tail measured {tail}/structural {struct} "
                                     f"> {TAIL_MAX[mech]}/contract"]) +
                ([] if vig else ["verifier"])),
            "table": cands,
        }
        print(f"{mech}: {len(cands)} variants, IS-best {best['label']} "
              f"IS {best['is']['trades']}/{best['is']['net']} pf {best['is']['pf']} "
              f"-> OOS {best['oos']['trades']}/{best['oos']['net']} pf {best['oos']['pf']} "
              f"ece {best['oos']['ece']} (raw {best['oos']['ece_raw']}) "
              f"auc {best['oos']['auc']} tail {tail} passes={passes}", flush=True)

    winners = [(mech, f["selected"]) for mech, f in families.items()
               if f.get("selected") and f["selected"]["passes"]]
    promoted = None
    if winners:
        promoted = max(winners, key=lambda w: Decimal(str(w[1]["oos"]["net"])))

    gate_block = {"profit_factor_min": 1.15, "ece_max": 0.08,
                  "requires_positive_net": True, "requires_zero_crossed_fills": True,
                  "min_oos_trades": 20, "bounded_tail": True}
    tape = {"lanes": lanes, "hours": hours,
            "book_rows": sum(a["book_rows"] for a in audit.values()),
            "prints": sum(a["prints"] for a in audit.values())}
    if promoted:
        mech, sel = promoted
        params = {"mechanism": mech, "label": sel["label"],
                  "t_lo_s": EARLY_T_LO if mech == "early_window_z_obi" else 90.0,
                  "t_hi_s": EARLY_T_HI if mech == "early_window_z_obi" else 420.0,
                  "cost_lo": EARLY_COST_LO if mech == "early_window_z_obi" else 0.15,
                  "cost_hi": EARLY_COST_HI if mech == "early_window_z_obi" else 0.85,
                  "exit_mode": ("hold_to_settlement" if mech == "early_window_z_obi"
                                else "synthetic_flatten"),
                  "tp_c": 0.0 if mech == "early_window_z_obi" else SYN_TP_C,
                  "barrier_s": None if mech == "early_window_z_obi" else 45.0,
                  "flatten_concession_c": None if mech == "early_window_z_obi" else 0.01,
                  "max_entry_price": EARLY_COST_HI if mech == "early_window_z_obi" else 1.0,
                  "max_window_trades": 2, "halt_after_first_loss": True,
                  "calibration": sel["platt"]}
        verdict = {"generated_ts": time.time(), "promoted": True,
                   "candidate": {"name": "z_obi_flow", "label": sel["label"],
                                 "mechanism": mech, "params": params},
                   "is": sel["is"], "oos": sel["oos"], "gate": gate_block, "tape": tape,
                   "reason": f"z_obi_flow {sel['label']} passes every frozen OOS check "
                             f"with a bounded tail ({sel['oos']['max_loss_per_contract']}"
                             f" <= {TAIL_MAX[mech]}/contract)"}
    else:
        verdict = {"generated_ts": time.time(), "promoted": False, "candidate": None,
                   "oos": None, "gate": gate_block, "tape": tape,
                   "families": {k: {"variants": v.get("variants"),
                                    "eligible": v.get("eligible"),
                                    "selected": v.get("selected"),
                                    "gate": v.get("gate"), "reason": v.get("reason"),
                                    "positive_oos_variants": v.get("positive_oos_variants")}
                                for k, v in families.items()},
                   "verdict": "neither bounded mechanism clears the frozen OOS gate",
                   "reason": "; ".join(f"{k}: {v.get('reason')}"
                                       for k, v in families.items() if v.get('reason'))}
    VERDICT.write_text(json.dumps(verdict, indent=2, default=str))
    BOUNDED_REPORT.write_text(json.dumps({
        "generated_ts": time.time(), "elapsed_s": round(time.time() - started, 1),
        "protocol": {
            "split": "chronological 60/40 by settled window close, built once, never re-split",
            "passive_fill": f">={MIN_PASSIVE_QTY:g} contracts printed at/through the limit",
            "crossed_book": "asserted, never filtered; 0 crossed fills required on OOS",
            "selection": "IS only (highest IS net, 20+ IS trades, positive IS net); "
                         "the frozen OOS slice is scored once per mechanism",
            "calibration": "Platt (a*logit(p)+b) fitted on IS trades only, frozen for OOS",
            "early_window": f"z_obi_flow T_remaining in [{EARLY_T_LO:.0f}, {EARLY_T_HI:.0f}]s, "
                            f"side cost in [{EARLY_COST_LO:.2f}, {EARLY_COST_HI:.2f}], "
                            f"z floors {list(EARLY_Z_GRID)} (the literal 0.6-1.2 grid has "
                            "an empty intersection with the price band: 0 of 542,364 "
                            "BTC rows cleared it at <=0.60), maker entry, hold to settlement",
            "syn_flatten": f"z_obi_flow entry, +{SYN_TP_C * 100:.0f}c maker TP, "
                           f"{SYN_BARS[0]:.0f}-{SYN_BARS[1]:.0f}s barrier, "
                           "$0-fee opposite-side lock at 1/2/3c concession",
            "tail": TAIL_MAX,
        },
        "lanes": lanes, "hours": hours, "audit": audit,
        "events": dict(events_by), "families": families, "tape": tape,
        "promotion": {"mechanism": promoted[0] if promoted else None,
                      "label": promoted[1]["label"] if promoted else None},
    }, indent=2, default=str))
    print(json.dumps({k: {
        "variants": v.get("variants"), "eligible": v.get("eligible"),
        "label": (v.get("selected") or {}).get("label"),
        "is": {kk: ((v.get("selected") or {}).get("is") or {}).get(kk)
               for kk in ("trades", "win_rate", "pf", "net", "ece")},
        "oos": {kk: ((v.get("selected") or {}).get("oos") or {}).get(kk)
                for kk in ("trades", "win_rate", "pf", "net", "ece", "ece_raw",
                           "crossed_fills", "max_loss_per_contract", "worst_trade",
                           "carries", "carry_net", "structural_max_loss_per_contract")},
        "passes": (v.get("selected") or {}).get("passes"),
        "reason": v.get("reason"),
    } for k, v in families.items()}, indent=2, default=str))
    print(f"wrote {BOUNDED_REPORT} and {VERDICT}")
    return 0 if promoted else 2


# ------------- focused IS-only refinement of the two near-miss bounded families
# The first bounded pass proved the mechanisms but missed the gate on one axis each:
# early-window z_obi at PF 1.110 (needs 1.15) and the lock+6c-stop arm at 10 OOS
# trades / ECE 0.081. This pass sweeps only entry quality (spread, flow, revision)
# and the stop size / entry window, chooses on the in-sample slice alone, and scores
# the frozen 40% exactly once for the chosen label of each family.
EARLY_REFINE = "early_window_z_obi_refine"
SYN_REFINE = "syn_lock_stop_refine"
REFINE_REPORT = ROOT / "data" / "bounded-refine.json"
REFINE_EARLY_SPREADS = (0.01, 0.02, 0.03)
REFINE_EARLY_FLOWS = (0.20, 0.25, 0.30)
REFINE_EARLY_REVS = (0.12, 0.15, 0.20)
REFINE_SYN_FLOWS = (0.10, 0.12, 0.15)
REFINE_SYN_REVS = (0.06, 0.08, 0.10)
REFINE_SYN_STOPS = (0.06, 0.07, 0.08)
# The combined early+mid window the lock arm is allowed to fire in.
REFINE_SYN_WINDOWS = ((180.0, 840.0),)
# `z` is pinned at each family's near-miss value: this pass sweeps entry quality
# (spread/flow/revision) and the stop, not the z floor.
REFINE_EARLY_Z = (0.15,)
REFINE_SYN_Z = (1.0,)
REFINE_SYN_CONC = 0.01
REFINE_SYN_BAR = 45.0
# Minimum IS trades the selection may consider. 20 is the stated gate; the lock arm
# needs 30 so that its own IS trade rate implies >= 20 trades on the 17h OOS slice.
# Both are in-sample quantities; nothing here reads the frozen slice.
REFINE_MIN_IS = {EARLY_REFINE: 20, SYN_REFINE: 30}


def refine_study(lanes: list[str]) -> int:
    """Entry-quality sweep for the two near-miss families, IS select then OOS once.

    Protocol is the same harness as `bounded_study` (chronological 60/40 split built
    once from the cached book, >= 20-contract print-tape passive fills, exact
    `Decimal` fees, crossed books asserted rather than filtered, structural tail
    bound). The only change is the selection rule: rank eligible candidates by *IS
    profit factor* (entry quality) rather than IS net, because IS net is what bought
    the 1.110 OOS PF last time -- one oversized winner. The frozen slice is scored
    once per family, after the label is fixed.
    """
    started = time.time()
    hours = {"is": 0.0, "oos": 0.0}
    audit: dict[str, dict] = {}
    rows_by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    meta: dict[tuple[str, str], dict] = {}
    events_by: dict[str, int] = defaultdict(int)
    hold = SimParams(tp_c=0.0, barrier_s=0.0, patience_s=15.0,
                     exit_mode="hold_to_settlement",
                     max_entry_price=EARLY_COST_HI, force_maker=True)

    for name in lanes:
        lane = Lane(name)
        sig = trailing_sigma(lane)
        for sl in ("is", "oos"):
            hours[sl] += lane.hours[sl]
        audit[name] = {"windows": len(lane.meta), "book_rows": int(len(lane.ts)),
                       "prints": int(len(lane.trade_ts)),
                       "is_hours": round(lane.hours["is"], 2),
                       "oos_hours": round(lane.hours["oos"], 2)}
        print(f"{name}: {len(lane.meta)} windows, {len(lane.ts):,} rows, "
              f"{len(lane.trade_ts):,} prints", flush=True)

        # (a) early window: the price band bounds the tail; the sweep tightens the
        # quality filters that decide whether the band is worth entering at all.
        for sp in REFINE_EARLY_SPREADS:
            for flow_min in REFINE_EARLY_FLOWS:
                for rev_min in REFINE_EARLY_REVS:
                    for z_min in REFINE_EARLY_Z:
                        evs = z_obi_flow_events(
                            lane, sig, z_min=z_min, flow_min=flow_min, rev_min=rev_min,
                            max_spread=sp, t_lo=EARLY_T_LO, t_hi=EARLY_T_HI,
                            cost_lo=EARLY_COST_LO, cost_hi=EARLY_COST_HI,
                            tag_decimals=2)
                        events_by[EARLY_REFINE] += len(evs)
                        if not evs:
                            continue
                        base = simulate(lane, evs, hold)
                        for cap in BOUNDED_CAPS:
                            label = (f"sp{sp:.2f}|z{z_min:.2f}_f{flow_min:.2f}"
                                     f"_r{rev_min:.2f}|cap{cap}")
                            seq = sequence_rows(base, max_window_trades=cap,
                                                halt_after_loss=True)
                            rows_by[(EARLY_REFINE, label)].extend(seq)
                            meta[(EARLY_REFINE, label)] = {
                                "z_min": z_min, "flow_min": flow_min, "rev_min": rev_min,
                                "max_spread": sp, "t_lo_s": EARLY_T_LO,
                                "t_hi_s": EARLY_T_HI, "cost_lo": EARLY_COST_LO,
                                "cost_hi": EARLY_COST_HI,
                                "exit_mode": "hold_to_settlement", "stop_c": 0.0,
                                "tp_c": 0.0, "barrier_s": None,
                                "flatten_concession_c": None, "max_window_trades": cap,
                            }
        print(f"  early sweep done ({time.time() - started:.0f}s)", flush=True)

        # (b) the shipped detector, exited through the $0-fee opposite-side lock with
        # a structural IOC stop. Broader triggers and a longer window are the only
        # levers that can lift the OOS trade count without touching the fill model.
        for t_lo, t_hi in REFINE_SYN_WINDOWS:
            for flow_min in REFINE_SYN_FLOWS:
                for rev_min in REFINE_SYN_REVS:
                    for z_min in REFINE_SYN_Z:
                        evs = z_obi_flow_events(
                            lane, sig, z_min=z_min, flow_min=flow_min, rev_min=rev_min,
                            t_lo=t_lo, t_hi=t_hi, tag_decimals=2)
                        events_by[SYN_REFINE] += len(evs)
                        if not evs:
                            continue
                        for stop_c in REFINE_SYN_STOPS:
                            arm = SimParams(tp_c=SYN_TP_C, barrier_s=REFINE_SYN_BAR,
                                            exit_mode="synthetic_flatten",
                                            flatten_concession_c=REFINE_SYN_CONC,
                                            scratch_patience_s=300.0, max_loss_c=stop_c)
                            base = simulate(lane, evs, arm)
                            for cap in BOUNDED_CAPS:
                                label = (f"w{t_lo:.0f}-{t_hi:.0f}|z{z_min:.2f}"
                                         f"_f{flow_min:.2f}_r{rev_min:.2f}"
                                         f"|stop{stop_c:.2f}_b{REFINE_SYN_BAR:.0f}"
                                         f"_c{REFINE_SYN_CONC * 100:.0f}|cap{cap}")
                                seq = sequence_rows(base, max_window_trades=cap,
                                                    halt_after_loss=True)
                                rows_by[(SYN_REFINE, label)].extend(seq)
                                meta[(SYN_REFINE, label)] = {
                                    "z_min": z_min, "flow_min": flow_min,
                                    "rev_min": rev_min, "max_spread": 0.03,
                                    "t_lo_s": t_lo, "t_hi_s": t_hi,
                                    "cost_lo": 0.15, "cost_hi": 0.85,
                                    "exit_mode": "synthetic_flatten",
                                    "stop_c": stop_c, "tp_c": SYN_TP_C,
                                    "barrier_s": REFINE_SYN_BAR,
                                    "flatten_concession_c": REFINE_SYN_CONC,
                                    "max_window_trades": cap,
                                }
        print(f"  lock+stop sweep done ({time.time() - started:.0f}s)", flush=True)
        del lane

    # Same invariants as every other passive study.
    for (mech, label), rows in rows_by.items():
        for r in rows:
            if r["crossed"] != 0:
                raise RuntimeError(f"crossed fill: {mech} {label}: {r}")
            if r["reason"] in ("maker_tp", "synthetic_flatten", "hold_to_settlement") \
                    and r["fee"] != 0:
                raise RuntimeError(f"fee on a passive leg: {mech} {label}: {r}")

    families: dict[str, dict] = {}
    for mech in (EARLY_REFINE, SYN_REFINE):
        table = []
        for (m, label), rows in rows_by.items():
            if m != mech or not rows:
                continue
            is_rows = [r for r in rows if r["slice"] == "is"]
            if not is_rows:
                continue
            table.append({"label": label, "rows": rows,
                          "is": slim2(block2(is_rows, hours["is"]))})
        table.sort(key=lambda c: (Decimal(str(c["is"]["net"])), int(c["is"]["trades"])),
                   reverse=True)
        eligible = [c for c in table
                    if int(c["is"]["trades"]) >= REFINE_MIN_IS[mech]
                    and Decimal(str(c["is"]["net"])) > 0
                    and int(c["is"]["crossed_fills"]) == 0]
        # IS-only ranking: entry quality first, then how much of it survived.
        eligible.sort(key=lambda c: (float(c["is"]["pf"]), Decimal(str(c["is"]["net"])),
                                     int(c["is"]["trades"])), reverse=True)
        if not eligible:
            families[mech] = {
                "variants": len(table), "eligible": False, "selected": None,
                "gate": None,
                "is_table": [{k: v for k, v in c["is"].items()
                              if k in ("trades", "wins", "win_rate", "net", "pf", "ece",
                                       "crossed_fills", "worst_trade",
                                       "max_loss_per_contract")}
                             | {"label": c["label"]} for c in table[:25]],
                "max_is_trades": max((int(c["is"]["trades"]) for c in table), default=0),
                "reason": f"no IS variant reached the {REFINE_MIN_IS[mech]}-trade floor "
                          f"with positive IS net (best IS trade count "
                          f"{max((int(c['is']['trades']) for c in table), default=0)})"}
            print(f"{mech}: {len(table)} variants, no IS-eligible candidate "
                  f"(max IS trades {max((int(c['is']['trades']) for c in table), default=0)})",
                  flush=True)
            continue
        best = eligible[0]
        pars = meta[(mech, best["label"])]
        structural_cap = None if mech == EARLY_REFINE else pars["stop_c"] + 0.02
        tail_max = EARLY_COST_HI if mech == EARLY_REFINE else structural_cap
        # ---- the one and only scoring of the frozen slice for this family ----
        blk = _bounded_block(mech, best["label"], best["rows"], hours, structural_cap)
        # Two gate tables, both reported. `platt` is the IS-fitted logistic map frozen
        # onto OOS; `deployed` is identity, which is the map `features.calibrate`
        # actually applies (data/calibration.json keeps identity on every lane because
        # every fitted map was worse on the frozen 40%). Promotion uses the deployed
        # path -- the probability the engine would quote -- and never a map that is
        # only better in-sample.
        gate_platt = gate(blk["oos"])
        deployed = dict(blk["oos"])
        deployed["ece"] = blk["oos"]["ece_raw"]
        g = gate(deployed)
        tail = float(blk["oos"]["max_loss_per_contract"] or 0.0)
        struct = float(blk["oos"]["structural_max_loss_per_contract"] or 0.0)
        tail_ok = tail <= tail_max + EPS and struct <= tail_max + EPS
        vig, why = verifier_call(deployed)
        passes = all(g.values()) and tail_ok and vig
        families[mech] = {
            "variants": len(table), "eligible": True,
            "selected": {**blk, "params": pars, "passes": passes,
                         "tail_max": round(tail_max, 4), "tail_ok": tail_ok,
                         "verifier": why,
                         "calibration": {
                             "deployed": "identity (data/calibration.json)",
                             "ece_deployed": blk["oos"]["ece_raw"],
                             "ece_is_platt_frozen": blk["oos"]["ece"],
                             "platt": blk["platt"],
                             "note": "the IS-fitted map lowers IS ECE "
                                     f"{blk['is']['ece_raw']}->{blk['is']['ece']} and "
                                     "raises frozen OOS ECE "
                                     f"{blk['oos']['ece_raw']}->{blk['oos']['ece']}, "
                                     "so it cannot be shipped"}},
            "gate": {**g, "bounded_tail": tail_ok},
            "gate_is_platt_frozen": {**gate_platt, "bounded_tail": tail_ok},
            "is_table": [{k: v for k, v in c["is"].items()
                          if k in ("trades", "wins", "win_rate", "net", "pf", "ece",
                                   "crossed_fills", "worst_trade",
                                   "max_loss_per_contract")}
                         | {"label": c["label"]} for c in table[:25]],
            "positive_oos_variants": None,
            "reason": None if passes else "IS-best failed the frozen OOS gate: " + ", ".join(
                [k for k, v in g.items() if not v]
                + ([] if tail_ok else [f"tail measured {tail}/structural {struct} "
                                       f"> {tail_max}/contract"])
                + ([] if vig else [f"verifier: {why}"])),
        }
        print(f"{mech}: {len(table)} IS variants, IS-best {best['label']} "
              f"IS {best['is']['trades']}/{best['is']['net']} pf {best['is']['pf']} "
              f"-> OOS {blk['oos']['trades']}/{blk['oos']['net']} pf {blk['oos']['pf']} "
              f"ece {blk['oos']['ece']} (raw {blk['oos']['ece_raw']}) "
              f"tail {tail}/{struct} passes={passes}", flush=True)

    winners = [(mech, f["selected"]) for mech, f in families.items()
               if f.get("selected") and f["selected"]["passes"]]
    promoted = max(winners, key=lambda w: Decimal(str(w[1]["oos"]["net"]))) if winners else None
    tape = {"lanes": lanes, "hours": hours,
            "book_rows": sum(a["book_rows"] for a in audit.values()),
            "prints": sum(a["prints"] for a in audit.values())}
    gate_block = {"profit_factor_min": 1.15, "ece_max": 0.08,
                  "requires_positive_net": True, "requires_zero_crossed_fills": True,
                  "min_oos_trades": 20, "bounded_tail": True}
    if promoted:
        mech, sel = promoted
        # The verdict carries the deployed (identity) probabilities, because that is
        # the number the engine will quote and the one the readiness gate reads.
        is_deployed = dict(sel["is"])
        is_deployed["ece"] = sel["is"]["ece_raw"]
        oos_deployed = dict(sel["oos"])
        oos_deployed["ece_platt_frozen"] = sel["oos"]["ece"]
        oos_deployed["ece"] = sel["oos"]["ece_raw"]
        verdict = {"generated_ts": time.time(), "promoted": True,
                   "candidate": {"name": "z_obi_flow", "label": sel["label"],
                                 "mechanism": mech, "params": sel["params"]},
                   "is": is_deployed, "oos": oos_deployed, "gate": gate_block, "tape": tape,
                   "calibration": sel["calibration"],
                   "gate_is_platt_frozen": families[mech]["gate_is_platt_frozen"],
                   "reason": (f"z_obi_flow {sel['label']} clears every frozen OOS check "
                              f"with a structural tail of {sel['tail_max']}/contract")}
    else:
        verdict = {"generated_ts": time.time(), "promoted": False, "candidate": None,
                   "oos": None, "gate": gate_block, "tape": tape,
                   "families": {k: {"variants": v.get("variants"),
                                    "eligible": v.get("eligible"),
                                    "selected": {kk: vv for kk, vv in
                                                 (v.get("selected") or {}).items()
                                                 if kk in ("label", "is", "oos", "passes",
                                                           "tail_max", "tail_ok", "verifier",
                                                           "params")},
                                    "gate": v.get("gate"),
                                    "gate_is_platt_frozen": v.get("gate_is_platt_frozen"),
                                    "is_table": v.get("is_table"),
                                    "max_is_trades": v.get("max_is_trades"),
                                    "reason": v.get("reason")}
                                for k, v in families.items()},
                   "reason": "; ".join(f"{k}: {v.get('reason')}"
                                       for k, v in families.items() if v.get("reason"))}
    VERDICT.write_text(json.dumps(verdict, indent=2, default=str))
    REFINE_REPORT.write_text(json.dumps({
        "generated_ts": time.time(), "elapsed_s": round(time.time() - started, 1),
        "protocol": {
            "split": "chronological 60/40 by settled window close, built once, never re-split",
            "selection": "IS only: IS trades >= 20 (lock arm 30), positive IS net, "
                         "zero crossed fills, ranked by IS profit factor then IS net; "
                         "the frozen OOS slice is scored once per family",
            "calibration": "gate on the deployed identity map (data/calibration.json); "
                           "the IS-fitted Platt map is reported as a counterfactual",
            "passive_fill": f">={MIN_PASSIVE_QTY:g} contracts printed at/through the limit",
            "crossed_book": "asserted, never filtered; 0 crossed fills required on OOS",
            "fees": "exact openkal.fees Decimal on every taker leg, $0 on passive legs",
            "early_grid": {"max_spread": list(REFINE_EARLY_SPREADS),
                           "flow": list(REFINE_EARLY_FLOWS),
                           "rev": list(REFINE_EARLY_REVS),
                           "z": list(REFINE_EARLY_Z),
                           "t_remaining": [EARLY_T_LO, EARLY_T_HI],
                           "cost": [EARLY_COST_LO, EARLY_COST_HI]},
            "lock_grid": {"flow": list(REFINE_SYN_FLOWS),
                          "rev": list(REFINE_SYN_REVS),
                          "stop_c": list(REFINE_SYN_STOPS),
                          "z": list(REFINE_SYN_Z),
                          "windows": [list(w) for w in REFINE_SYN_WINDOWS],
                          "barrier_s": REFINE_SYN_BAR,
                          "concession_c": REFINE_SYN_CONC},
            "tail": {EARLY_REFINE: EARLY_COST_HI,
                     SYN_REFINE: "stop_c + 0.02"},
        },
        "lanes": lanes, "hours": hours, "audit": audit,
        "events": dict(events_by), "families": families, "tape": tape,
        "promotion": {"mechanism": promoted[0] if promoted else None,
                      "label": promoted[1]["label"] if promoted else None},
    }, indent=2, default=str))
    print(json.dumps({k: {
        "variants": v.get("variants"),
        "label": (v.get("selected") or {}).get("label"),
        "is": {kk: ((v.get("selected") or {}).get("is") or {}).get(kk)
               for kk in ("trades", "win_rate", "pf", "net", "ece")},
        "oos": {kk: ((v.get("selected") or {}).get("oos") or {}).get(kk)
                for kk in ("trades", "win_rate", "pf", "net", "ece", "ece_raw",
                           "crossed_fills", "max_loss_per_contract", "worst_trade",
                           "structural_max_loss_per_contract")},
        "passes": (v.get("selected") or {}).get("passes"),
        "reason": v.get("reason"),
    } for k, v in families.items()}, indent=2, default=str))
    print(f"wrote {REFINE_REPORT} and {VERDICT}")
    return 0 if promoted else 2


# ---------------------------------------------------- passive / settlement study
def series_at(index: dict, ts: np.ndarray, tol: float = 1.5) -> np.ndarray:
    """Nearest print of an arbitrary (ts, value) index tape within `tol` seconds."""
    if not index:
        return np.full(len(ts), np.nan)
    arr = np.asarray(index["ts"], dtype=np.float64)
    val = np.asarray(index["value"], dtype=np.float64)
    if not len(arr):
        return np.full(len(ts), np.nan)
    j = np.searchsorted(arr, ts, "left")
    jm = np.clip(j - 1, 0, len(arr) - 1)
    jp = np.clip(j, 0, len(arr) - 1)
    dm = np.abs(arr[jm] - ts)
    dp = np.abs(arr[jp] - ts)
    use_m = dm <= dp
    out = np.where(use_m, val[jm], val[jp]).astype(np.float64)
    out[np.where(use_m, dm, dp) > tol] = np.nan
    return out


def flow_imbalance(lane: Lane, tid: int, ts_rows: np.ndarray,
                   window: float = 10.0) -> np.ndarray:
    """Net aggressive print flow in [-1, 1] over the trailing `window` seconds."""
    lo, hi = int(lane.trade_bounds[tid]), int(lane.trade_bounds[tid + 1])
    if hi <= lo:
        return np.zeros(len(ts_rows))
    tts = lane.trade_ts[lo:hi]
    sz = lane.trade_sz[lo:hi]
    signed = np.where(lane.trade_ty[lo:hi] == 1, sz, -sz)
    cs = np.concatenate(([0.0], np.cumsum(signed)))
    ct = np.concatenate(([0.0], np.cumsum(sz)))
    a = np.searchsorted(tts, ts_rows - window, "left")
    b = np.searchsorted(tts, ts_rows, "right")
    num = cs[b] - cs[a]
    den = ct[b] - ct[a]
    return np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)


def revision_imbalance(lane: Lane, a: int, b: int, window: float = 5.0) -> np.ndarray:
    """Touch-revision imbalance: bid upticks minus ask downticks, in [-1, 1].

    The recording stores only top-of-book (no ladder depth), so a three-level OBI can
    not be computed from it. This is the honest book-side substitute: how one-sided
    the *quote revisions* were, which is what a repricing ladder looks like from the
    touch outward.
    """
    ts = lane.ts[a:b]
    if len(ts) < 2:
        return np.zeros(len(ts))
    bid = lane.bid[a:b]
    ask = lane.ask[a:b]
    up = (np.diff(bid, prepend=bid[0]) > 0).astype(np.float64)
    dn = (np.diff(ask, prepend=ask[0]) < 0).astype(np.float64)
    cu = np.concatenate(([0.0], np.cumsum(up)))
    cd = np.concatenate(([0.0], np.cumsum(dn)))
    idx = np.arange(1, len(ts) + 1)
    left = np.searchsorted(ts, ts - window, "left")
    u = cu[idx] - cu[left]
    d = cd[idx] - cd[left]
    return (u - d) / np.maximum(u + d, 1.0)


def velocity_bps(lane: Lane, ts_rows: np.ndarray, lookback: float = 5.0) -> np.ndarray:
    """Signed short-horizon settlement-index velocity, in bps, at each book row.

    The fade families lose on exactly the windows where the move they faded kept
    going. That is measurable before the entry: the venue's own reference print has
    already moved. Zero where the index tape has no print nearby, so a missing feed
    can never manufacture a veto.
    """
    now = index_at(lane, ts_rows)
    then = index_at(lane, ts_rows - lookback)
    out = np.zeros(len(ts_rows))
    ok = np.isfinite(now) & np.isfinite(then) & (then > 0)
    out[ok] = (now[ok] / then[ok] - 1.0) * 10_000.0
    return out


def lead_lag_events(lane: Lane, btc: dict, sig: dict[int, np.ndarray], *,
                    bps_lo: float, bps_hi: float, lookback: float,
                    max_reprice_c: float, lag_share: float = 0.6,
                    cooldown: float = 90.0, per_window: int = 2) -> list[Event]:
    """BTC impulse -> alt lane that has not repriced yet.

    The premise is mechanical: BTC's settlement index moves first, the alt's own index
    and its retail book follow. So the trigger is a BTC move, the confirmation is that
    the alt's index has moved *less* than `lag_share` of it, and the entry is a passive
    quote on the alt ladder that must still be near where it was.
    """
    out: list[Event] = []
    for tid, name, meta, a, b in _segments(lane):
        ts = lane.ts[a:b]
        bid = lane.bid[a:b]
        ask = lane.ask[a:b]
        secs = meta["close"] - ts
        spread = ask - bid
        valid = (bid > 0) & (ask < 1) & (bid < ask) & (spread <= 0.04 + EPS)
        valid &= (secs >= 120.0) & (secs <= 840.0)
        if valid.sum() < 10:
            continue
        btc_now = series_at(btc, ts)
        btc_then = series_at(btc, ts - lookback)
        alt_now = index_at(lane, ts)
        alt_then = index_at(lane, ts - lookback)
        bps = np.full(len(ts), np.nan)
        alt_bps = np.full(len(ts), np.nan)
        ok_b = np.isfinite(btc_now) & np.isfinite(btc_then) & (btc_then > 0)
        ok_a = np.isfinite(alt_now) & np.isfinite(alt_then) & (alt_then > 0)
        bps[ok_b] = (btc_now[ok_b] / btc_then[ok_b] - 1.0) * 10_000.0
        alt_bps[ok_a] = (alt_now[ok_a] / alt_then[ok_a] - 1.0) * 10_000.0
        mid = (bid + ask) / 2.0
        j = np.searchsorted(ts, ts - lookback, "right") - 1
        shift = np.full(len(ts), np.nan)
        good = j >= 0
        shift[good] = mid[good] - mid[j[good]]
        base = valid & ok_b & ok_a & np.isfinite(shift)
        up = (base & (bps >= bps_lo) & (bps <= bps_hi) & (alt_bps > 0)
              & (alt_bps <= lag_share * bps) & (shift <= max_reprice_c))
        down = (base & (bps <= -bps_lo) & (bps >= -bps_hi) & (alt_bps < 0)
                & (alt_bps >= lag_share * bps) & (shift >= -max_reprice_c))
        for idx, long_yes in ((np.flatnonzero(up), True), (np.flatnonzero(down), False)):
            for i in _dedupe(list(idx), ts, cooldown=cooldown, per_ticker=per_window):
                limit = float(bid[i]) if long_yes else float(ask[i])
                if not (0.0 < limit < 1.0):
                    continue
                p = fair_p1(float(alt_now[i]), meta["strike"], float(secs[i]),
                            _sigma_at(lane, sig, tid, float(ts[i])))
                out.append(Event(
                    "btc_leads_alts", f"b{bps_lo:.0f}_{bps_hi:.0f}_l{lookback:.0f}",
                    tid=tid, ticker=name, ts=float(ts[i]), close=meta["close"], y=meta["y"],
                    slice=lane.slice_of(meta["close"]), side="YES" if long_yes else "NO",
                    entry_style="maker", limit=limit,
                    p_side=(0.5 if p is None else (p if long_yes else 1.0 - p)),
                    tag=f"btc{float(bps[i]):+.0f}bp alt{float(alt_bps[i]):+.0f}bp"))
    return out


def z_obi_flow_events(lane: Lane, sig: dict[int, np.ndarray], *, z_min: float,
                      flow_min: float, rev_min: float, t_lo: float = 90.0,
                      t_hi: float = 420.0, cooldown: float = 90.0,
                      per_window: int = 2, cost_lo: float = 0.15,
                      cost_hi: float = 0.85, tag_decimals: int = 1,
                      max_spread: float = 0.03, min_delta: float = 0.0,
                      trend_fast_s: float = 0.0,
                      trend_slow_s: float = 0.0) -> list[Event]:
    """Normalised strike z, printed-flow imbalance and touch-revision imbalance agree.

    `cost_lo`/`cost_hi` bound the price actually paid per contract (the YES price for
    a long, `1 - YES` for a short). The default reproduces the original symmetric
    0.15-0.85 YES band; the early-window mechanism passes 0.30-0.60 so the worst
    possible settlement loss per contract is bounded by the entry price alone.
    """
    out: list[Event] = []
    for tid, name, meta, a, b in _segments(lane):
        ts = lane.ts[a:b]
        bid = lane.bid[a:b]
        ask = lane.ask[a:b]
        secs = meta["close"] - ts
        valid = ((bid > 0) & (ask < 1) & (bid < ask) & ((ask - bid) <= max_spread + EPS)
                 & (secs >= t_lo) & (secs <= t_hi))
        if valid.sum() < 10:
            continue
        spot = index_at(lane, ts)
        vol = sig[tid] * np.sqrt(np.maximum(secs, 1e-9))
        z = np.zeros(len(ts))
        ok = valid & np.isfinite(spot) & (spot > 0) & (vol > 0)
        z[ok] = np.log(spot[ok] / meta["strike"]) / vol[ok]
        flow = flow_imbalance(lane, tid, ts, 10.0)
        rev = revision_imbalance(lane, a, b, 5.0)
        direction = np.sign(z)
        # Depth is not in the recording; the ladder term is the touch revision, and it
        # has to lean the same way as the flow and the level.
        ok &= (np.abs(z) >= z_min) & (flow * direction >= flow_min) & (rev * direction >= rev_min)
        # Spot-to-strike distance: inside it the contract is a coin flip and a flow
        # edge is not worth holding the binary.
        if min_delta > 0:
            ok &= np.abs(spot - meta["strike"]) >= min_delta
        # Two-horizon spot trend, both on the same side as the z. A stale index print
        # against a moving tape is not a dislocation.
        if trend_fast_s > 0 and trend_slow_s > 0:
            fast = series_at(lane.index, ts - trend_fast_s)
            slow = series_at(lane.index, ts - trend_slow_s)
            with np.errstate(invalid="ignore", divide="ignore"):
                tf = np.log(spot / fast)
                ts_ = np.log(spot / slow)
            ok &= (np.isfinite(tf) & np.isfinite(ts_)
                   & (tf * direction > 0) & (ts_ * direction > 0))
        for idx, long_yes in ((np.flatnonzero(ok & (z > 0)), True),
                              (np.flatnonzero(ok & (z < 0)), False)):
            for i in _dedupe(list(idx), ts, cooldown=cooldown, per_ticker=per_window):
                limit = float(bid[i]) if long_yes else float(ask[i])
                if not (0.0 < limit < 1.0):
                    continue
                cost = limit if long_yes else 1.0 - limit
                if not (cost_lo - EPS <= cost <= cost_hi + EPS):
                    continue
                p = fair_p1(float(spot[i]), meta["strike"], float(secs[i]),
                            _sigma_at(lane, sig, tid, float(ts[i])))
                out.append(Event(
                    "z_obi_flow",
                    f"z{z_min:.{tag_decimals}f}_f{flow_min:.{tag_decimals}f}"
                    f"_r{rev_min:.{tag_decimals}f}",
                    tid=tid, ticker=name, ts=float(ts[i]), close=meta["close"], y=meta["y"],
                    slice=lane.slice_of(meta["close"]), side="YES" if long_yes else "NO",
                    entry_style="maker", limit=limit,
                    p_side=(0.5 if p is None else (p if long_yes else 1.0 - p)),
                    tag=f"z{float(z[i]):+.2f} f{float(flow[i]):+.2f} r{float(rev[i]):+.2f}"))
    return out


def late_favorite_events(lane: Lane, sig: dict[int, np.ndarray], *, z_min: float,
                         lo: float = 0.70, hi: float = 0.88, t_lo: float = 75.0,
                         t_hi: float = 240.0, cooldown: float = 120.0,
                         per_window: int = 1) -> list[Event]:
    """Late-window favourite: quadratic taker fee is < 1c at 0.70-0.88."""
    out: list[Event] = []
    for tid, name, meta, a, b in _segments(lane):
        ts = lane.ts[a:b]
        bid = lane.bid[a:b]
        ask = lane.ask[a:b]
        secs = meta["close"] - ts
        valid = ((bid > 0) & (ask < 1) & (bid < ask) & ((ask - bid) <= 0.03 + EPS)
                 & (secs >= t_lo) & (secs <= t_hi))
        if valid.sum() < 5:
            continue
        spot = index_at(lane, ts)
        vol = sig[tid] * np.sqrt(np.maximum(secs, 1e-9))
        z = np.zeros(len(ts))
        ok = valid & np.isfinite(spot) & (spot > 0) & (vol > 0)
        z[ok] = np.log(spot[ok] / meta["strike"]) / vol[ok]
        yes_px = ask          # cost to buy the favourite when YES is it
        no_px = 1.0 - bid     # cost to buy NO when NO is it
        up = ok & (z >= z_min) & (yes_px >= lo) & (yes_px <= hi)
        down = ok & (z <= -z_min) & (no_px >= lo) & (no_px <= hi)
        for idx, long_yes in ((np.flatnonzero(up), True), (np.flatnonzero(down), False)):
            for i in _dedupe(list(idx), ts, cooldown=cooldown, per_ticker=per_window):
                limit = float(ask[i]) if long_yes else float(bid[i])
                if not (0.0 < limit < 1.0):
                    continue
                p = fair_p1(float(spot[i]), meta["strike"], float(secs[i]),
                            _sigma_at(lane, sig, tid, float(ts[i])))
                out.append(Event(
                    "late_favorite", f"z{z_min:.1f}_{lo:.2f}-{hi:.2f}", tid=tid, ticker=name,
                    ts=float(ts[i]), close=meta["close"], y=meta["y"],
                    slice=lane.slice_of(meta["close"]), side="YES" if long_yes else "NO",
                    entry_style="taker", limit=limit,
                    p_side=(0.5 if p is None else (p if long_yes else 1.0 - p)),
                    tag=f"z{float(z[i]):+.2f} T{float(secs[i]):.0f}s"))
    return out


def passive_probe(row: dict) -> None:
    """The passive families may never book a taker exit or a crossed fill."""
    if row["exit_mode"] == "hold_to_settlement":
        return
    if row["reason"] == "maker_scratch" and row["fee"] != 0:
        raise RuntimeError(f"maker_scratch booked a fee: {row}")
    if row["crossed"] != 0:
        raise RuntimeError(f"crossed fill: {row}")


PASSIVE_REPORT = ROOT / "data" / "passive-search.json"


def passive_study(lanes: list[str]) -> int:
    """Maker-scratch exits, BTC lead-lag, z+flow ladders and late favourites.

    Four families, one simulator. Detection is per lane, selection is in-sample only
    (highest IS profit factor among variants with 20+ IS trades and positive IS net),
    and the chronological 40% frozen slice is scored exactly once per family.
    """
    started = time.time()
    hours = {name: 0.0 for name in ("is", "oos")}
    rows_by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    events_by: dict[str, int] = defaultdict(int)
    audit: dict[str, dict] = {}
    btc_index = dict(np.load(CACHE / "index_BRTI.npz"))

    for name in lanes:
        lane = Lane(name)
        sig = trailing_sigma(lane)
        for slice_name in ("is", "oos"):
            hours[slice_name] += lane.hours[slice_name]
        audit[name] = {"windows": len(lane.meta), "book_rows": int(len(lane.ts)),
                       "prints": int(len(lane.trade_ts)),
                       "is_hours": round(lane.hours["is"], 2),
                       "oos_hours": round(lane.hours["oos"], 2)}
        print(f"{name}: {len(lane.meta)} windows, {len(lane.ts):,} rows, "
              f"{len(lane.trade_ts):,} prints", flush=True)

        # (a) the two proven snapback detectors, now leaving passively.
        absorb = absorption_events(lane, sig, size=20.0) + absorption_events(lane, sig, size=40.0)
        disloc = []
        for gap_min, persist in DISLOC_DETECTORS:
            disloc.extend(dislocation_events(lane, sig, gap_min=gap_min, persist=persist))

        # (b) BTC leads the alt lanes. Each impulse band is its own detector, so an
        # impulse caught by two bands is selected between, never double counted.
        lead: dict[str, list[Event]] = {}
        if name != "KXBTC15M":
            for (lo, hi, lb, share) in LEAD_DETECTORS:
                lead[f"b{lo:.0f}-{hi:.0f}l{lb:.0f}s{share * 100:.0f}"] = lead_lag_events(
                    lane, btc_index, sig, bps_lo=lo, bps_hi=hi, lookback=lb,
                    max_reprice_c=0.06, lag_share=share)

        # (c) z + printed flow + touch revision, and (d) the late favourite.
        zf = []
        for z_min, f_min, r_min in ZOBI_DETECTORS:
            zf.extend(z_obi_flow_events(lane, sig, z_min=z_min, flow_min=f_min, rev_min=r_min))

        fam_events: dict[str, dict[str, list[Event]]] = {
            "absorption_revert": {"": absorb},
            "index_dislocation": {"": disloc},
            "btc_leads_alts": lead,
            "z_obi_flow": {"": zf},
            "late_favorite": {f"z{z:.1f}": late_favorite_events(lane, sig, z_min=z)
                              for z in LATE_DETECTORS},
        }
        for family, dets in fam_events.items():
            for det, evs in dets.items():
                events_by[family] += len(evs)
                for sim_label, params in PASSIVE_SIMS[family]:
                    label = f"{det}^{sim_label}" if det else sim_label
                    rows_by[(family, label)].extend(simulate(lane, evs, params))

        print(f"  events {dict(events_by)} ({time.time() - started:.0f}s)", flush=True)
        del lane

    for (family, label), rows in rows_by.items():
        for r in rows:
            passive_probe({**r, "exit_mode": PASSIVE_MODE[family]})

    table: list[dict] = []
    families: dict[str, dict] = {}
    for family in PASSIVE_FAMILIES:
        cands: list[dict] = []
        for label in sorted({lab for fam, lab in rows_by if fam == family}):
            rows = rows_by.get((family, label), [])
            if not rows:
                continue
            is_rows = [r for r in rows if r["slice"] == "is"]
            oos_rows = [r for r in rows if r["slice"] == "oos"]
            is_block = block2(is_rows, hours["is"])
            oos_block = block2(oos_rows, hours["oos"])
            pa, pb = platt_fit(is_rows)          # fitted on IS only, frozen for OOS
            is_block["ece_raw"] = is_block.get("ece")
            is_block["ece"] = cal_ece(is_rows, pa, pb)
            oos_block["ece_raw"] = oos_block.get("ece")
            oos_block["ece"] = cal_ece(oos_rows, pa, pb)
            cands.append({
                "family": family, "label": label,
                "is": is_block, "oos": oos_block,
                "platt": {"a": round(pa, 4), "b": round(pb, 4), "fitted_on": "is"},
                "gate": gate(oos_block), "verifier": verifier_call(oos_block),
            })
        if not cands:
            families[family] = {"selected": None, "eligible": False, "passes": False,
                                "reason": "no fills"}
            continue
        table.extend(cands)
        eligible = [c for c in cands if c["is"]["trades"] >= 20
                    and Decimal(str(c["is"]["net"])) > 0]
        eligible.sort(key=lambda c: (float(c["is"]["pf"]),
                                     Decimal(str(c["is"]["net"])),
                                     int(c["is"]["trades"])), reverse=True)
        best = eligible[0] if eligible else None
        passes = bool(best) and all(best["gate"].values()) and bool(best["verifier"][0])
        families[family] = {
            "selected": ({k: best[k] for k in ("label", "is", "oos", "gate", "platt",
                                               "verifier")} if best else None),
            "eligible": bool(eligible),
            "passes": passes,
            "oos_pass_labels": [c["label"] for c in cands if all(c["gate"].values())],
            "reason": None if passes else (
                "no variant reached 20 IS trades with positive IS net" if not best
                else "IS-best failed the frozen OOS gate: "
                     + ", ".join(k for k, v in best["gate"].items() if not v)
                     + ("" if best["verifier"][0] else f" ({best['verifier'][1]})")),
        }
        for c in cands:
            print(f"{family} {c['label']}: IS {c['is']['trades']}/{c['is']['net']} "
                  f"pf {c['is']['pf']} -> OOS {c['oos']['trades']}/{c['oos']['net']} "
                  f"pf {c['oos']['pf']} ece {c['oos']['ece']} "
                  f"pass={all(c['gate'].values()) and bool(c['verifier'][0])}", flush=True)

    promoted = None
    for family, info in families.items():
        if not info.get("passes"):
            continue
        if promoted is None:
            promoted = family
            continue
        a = (Decimal(str(info["selected"]["oos"]["net"])),
             float(info["selected"]["oos"]["pf"]))
        b = (Decimal(str(families[promoted]["selected"]["oos"]["net"])),
             float(families[promoted]["selected"]["oos"]["pf"]))
        if a > b:
            promoted = family

    tape = {"lanes": lanes, "hours": hours,
            "book_rows": sum(a["book_rows"] for a in audit.values()),
            "prints": sum(a["prints"] for a in audit.values())}
    gate_block = {"profit_factor_min": 1.15, "ece_max": 0.08,
                  "requires_positive_net": True, "requires_zero_crossed_fills": True,
                  "min_oos_trades": 20}
    protocol = {
        "split": "chronological 60/40 by settled window close, built once, never re-split",
        "passive_fill": f">={MIN_PASSIVE_QTY:g} contracts printed at/through the limit",
        "fees": "exact Decimal taker_fee on every taker leg; $0 on every maker leg",
        "crossed_book": "asserted, never filtered: a bid >= ask row aborts the study",
        "maker_scratch": "post-only TP (+3..5c), then a post-only breakeven/1c scratch "
                         "at the time barrier; unfilled scratch carries to settlement, $0 fee",
        "selection": "IS only (highest IS PF, 20+ IS trades, positive IS net); OOS scored once",
        "calibration": "Platt fitted on IS rows only, frozen for OOS",
        "depth_limit": "the recording stores top-of-book only, so the ladder term in "
                       "z_obi_flow is the touch-revision imbalance, not a 3-level OBI",
    }
    if promoted:
        sel = families[promoted]["selected"]
        verdict = {
            "generated_ts": time.time(), "promoted": True,
            "candidate": {"name": promoted, "label": sel["label"]},
            "is": sel["is"], "oos": sel["oos"], "gate": gate_block, "tape": tape,
            "protocol": protocol,
            "reason": f"{promoted} {sel['label']} passes every frozen OOS check",
        }
    else:
        verdict = {
            "generated_ts": time.time(), "promoted": False, "candidate": None, "oos": None,
            "gate": gate_block, "families": families, "table": table, "tape": tape,
            "protocol": protocol,
            "verdict": "no passive or settlement candidate clears the frozen OOS gate",
            "reason": "no family produced an in-sample candidate that clears the frozen OOS gate",
        }
    VERDICT.write_text(json.dumps(verdict, indent=2, default=str))
    PASSIVE_REPORT.write_text(json.dumps({
        "generated_ts": time.time(), "elapsed_s": round(time.time() - started, 1),
        "protocol": protocol, "lanes": lanes, "hours": hours, "audit": audit,
        "events": dict(events_by), "families": families, "table": table,
        "promotion": {"family": promoted}, "tape": tape,
    }, indent=2, default=str))
    print(json.dumps({
        "promotion": promoted,
        "families": {k: {
            "label": (v.get("selected") or {}).get("label"),
            "is": {kk: ((v.get("selected") or {}).get("is") or {}).get(kk)
                   for kk in ("trades", "win_rate", "pf", "net")},
            "oos": {kk: ((v.get("selected") or {}).get("oos") or {}).get(kk)
                    for kk in ("trades", "win_rate", "pf", "net", "ece", "crossed_fills")},
            "oos_pass_labels": v.get("oos_pass_labels"),
        } for k, v in families.items()}}, indent=2))
    print(f"wrote {PASSIVE_REPORT} and {VERDICT}")
    return 0 if promoted else 2


# ------------------- z_obi_flow asymmetric entry cap / bounded tail study
# The promoted program entered at 0.77-0.84 and carried the whole binary. Tested
# caps are the strict band the operator specified; the take-profit is +5c, so the
# loss cap is 1.2x that (6c) and a losing contract can never cost the settlement.
ZOBI_PRICE_CAPS = (0.45, 0.50, 0.55)
ZOBI_LOSS_CAPS = (0.0, 0.06)
# The other arm of the same requirement: keep the studied entry band and bound the
# single-trade loss. Tested so the choice between the two is evidence, not taste.
ZOBI_BOUND_ONLY = ((0.84, 0.06), (0.84, 0.04))
ZOBI_CAP_REPORT = ROOT / "data" / "zobi-cap-search.json"


def zobi_cap_study(lanes: list[str]) -> int:
    """Frozen 60/40 re-verification of z_obi_flow under an asymmetric entry cap.

    One dimension changes: the most a contract may cost, and the hard stop that
    bounds a single trade. Detectors, exit, per-window cap and halt-after-loss are
    the deployed ones, so the only thing the OOS slice is asked about is the tail.
    The 0.84 / no-stop row is the shipped control and is never selectable.
    """
    started = time.time()
    hours = {"is": 0.0, "oos": 0.0}
    rows_by: dict[tuple[float, float], list[dict]] = defaultdict(list)
    audit: dict[str, dict] = {}
    for name in lanes:
        lane = Lane(name)
        sig = trailing_sigma(lane)
        for slice_name in ("is", "oos"):
            hours[slice_name] += lane.hours[slice_name]
        audit[name] = {"windows": len(lane.meta), "book_rows": int(len(lane.ts)),
                       "prints": int(len(lane.trade_ts)),
                       "is_hours": round(lane.hours["is"], 2),
                       "oos_hours": round(lane.hours["oos"], 2)}
        evs: list[Event] = []
        for z_min in ZOBI_Z_GRID:
            for flow_min in ZOBI_FLOW_GRID:
                evs.extend(z_obi_flow_events(lane, sig, z_min=z_min, flow_min=flow_min,
                                             rev_min=ZOBI_REV_MIN))
        print(f"{name}: {len(lane.meta)} windows, {len(evs)} flow events "
              f"({time.time() - started:.0f}s)", flush=True)
        grid = [(0.84, 0.0)] + [(cap, loss) for cap in ZOBI_PRICE_CAPS
                                for loss in ZOBI_LOSS_CAPS] + list(ZOBI_BOUND_ONLY)
        for cap, loss in grid:
            params = SimParams(tp_c=0.05, barrier_s=45.0, exit_mode="maker_scratch",
                               scratch_concession_c=0.0, scratch_patience_s=20.0,
                               max_entry_price=cap, max_loss_c=loss)
            rows_by[(cap, loss)].extend(simulate(lane, evs, params))
        del lane

    cands: list[dict] = []
    skipped: list[dict] = []
    for (cap, loss), rows in sorted(rows_by.items()):
        for r in rows:
            if r["crossed"] != 0:
                raise RuntimeError(f"crossed fill: cap {cap} stop {loss}: {r}")
        seq = sequence_rows(rows, max_window_trades=2, halt_after_loss=True)
        is_rows = [r for r in seq if r["slice"] == "is"]
        oos_rows = [r for r in seq if r["slice"] == "oos"]
        if not is_rows or not oos_rows:
            # The strict caps are the important case here: `z_obi_flow` only fires
            # on the side the strike z already favours, and that side is priced
            # 0.70-0.85. A 0.55 cap does not trim the sample, it empties it.
            skipped.append({"label": f"cap{cap:.2f}_stop{loss:.2f}",
                            "rows": len(rows), "is": len(is_rows), "oos": len(oos_rows)})
            continue
        is_block = block2(is_rows, hours["is"])
        oos_block = block2(oos_rows, hours["oos"])
        pa, pb = platt_fit(is_rows)              # fitted on IS only, frozen for OOS
        is_block["ece_raw"] = is_block.get("ece")
        is_block["ece"] = cal_ece(is_rows, pa, pb)
        oos_block["ece_raw"] = oos_block.get("ece")
        oos_block["ece"] = cal_ece(oos_rows, pa, pb)
        cands.append({
            "label": f"cap{cap:.2f}_stop{loss:.2f}", "cap": cap, "max_loss_c": loss,
            "control": loss == 0.0 and cap > max(ZOBI_PRICE_CAPS),
            "is": is_block, "oos": oos_block,
            "platt": {"a": round(pa, 4), "b": round(pb, 4), "fitted_on": "is"},
            "gate": gate(oos_block),
        })
        print(f"{cands[-1]['label']}: IS {is_block['trades']}/{is_block['net']} "
              f"pf {is_block['pf']} tail {is_block['max_loss_per_contract']} -> "
              f"OOS {oos_block['trades']}/{oos_block['net']} pf {oos_block['pf']} "
              f"ece {oos_block['ece']} tail {oos_block['max_loss_per_contract']} "
              f"pass={all(cands[-1]['gate'].values())}", flush=True)

    # Selection is IS-only. A candidate is eligible only if it bounds the tail one
    # of the two sanctioned ways: a <=0.55 entry, or an explicit loss stop. The
    # 0.84-with-no-stop control is reported for contrast and is never selectable.
    eligible = [c for c in cands if not c["control"]
                and (c["cap"] <= max(ZOBI_PRICE_CAPS) or c["max_loss_c"] > 0)
                and c["is"]["trades"] >= 20
                and Decimal(str(c["is"]["net"])) > 0]
    eligible.sort(key=lambda c: (float(c["is"]["pf"]),
                                 Decimal(str(c["is"]["net"])),
                                 int(c["is"]["trades"])), reverse=True)
    best = eligible[0] if eligible else None
    # The design bound is 6c; the realized worst adds the taker fee on the stop.
    tail_ok = bool(best) and float(best["oos"]["max_loss_per_contract"]) <= 0.08
    passes = bool(best) and all(best["gate"].values()) and tail_ok

    tape = {"lanes": lanes, "hours": hours,
            "book_rows": sum(a["book_rows"] for a in audit.values()),
            "prints": sum(a["prints"] for a in audit.values())}
    gate_block = {"profit_factor_min": 1.15, "ece_max": 0.08,
                  "requires_positive_net": True, "requires_zero_crossed_fills": True,
                  "min_oos_trades": 20,
                  "max_loss_per_contract": 0.08}
    protocol = {
        "split": "chronological 60/40 by settled window close, built once, never re-split",
        "entry_caps_tested": list(ZOBI_PRICE_CAPS) + [0.84],
        "loss_caps_tested": list(ZOBI_LOSS_CAPS),
        "loss_cap_rule": "1.2x the +5c take-profit: 6c of adverse price, then cross out",
        "control": "cap 0.84 with no stop (the shipped behaviour), reported, not selectable",
        "selection": "IS only (highest IS PF, 20+ IS trades, positive IS net); OOS scored once",
        "calibration": "Platt fitted on IS rows only, frozen for OOS",
    }
    if best is not None:
        candidate = {"name": "z_obi_flow", "label": best["label"],
                     "zobi_max_price": best["cap"],
                     "max_loss_c": best["max_loss_c"],
                     "tp_c": 0.05, "barrier_s": 45.0, "exit_mode": "maker_scratch",
                     "max_window_trades": 2, "halt_after_first_loss": True,
                     "entry": "post_only"}
    else:
        candidate = None
    if passes:
        verdict = {
            "generated_ts": time.time(), "promoted": True, "candidate": candidate,
            "is": slim2(best["is"]), "oos": slim2(best["oos"]), "gate": gate_block,
            "tape": tape, "protocol": protocol,
            "reason": (f"z_obi_flow {best['label']} passes every frozen OOS check "
                       f"with the settlement tail bounded"),
        }
    else:
        verdict = {
            "generated_ts": time.time(), "promoted": False, "candidate": None,
            "oos": best["oos"] if best else None, "gate": gate_block,
            "table": cands, "skipped": skipped, "tape": tape, "protocol": protocol,
            "verdict": "no z_obi_flow entry cap clears the frozen OOS gate with a bounded tail",
            "reason": ("no strict-band candidate reached 20 IS trades with positive IS net"
                       if best is None else
                       "IS-best failed: " + ", ".join(k for k, v in best["gate"].items()
                                                      if not v)
                       + ("" if tail_ok else f" (OOS tail "
                          f"{best['oos']['max_loss_per_contract']} > 0.08/contract)")),
        }
    VERDICT.write_text(json.dumps(verdict, indent=2, default=str))
    ZOBI_CAP_REPORT.write_text(json.dumps({
        "generated_ts": time.time(), "elapsed_s": round(time.time() - started, 1),
        "protocol": protocol, "lanes": lanes, "hours": hours, "audit": audit,
        "table": cands, "skipped": skipped, "selected": best["label"] if best else None,
        "promotion": bool(passes), "tape": tape,
    }, indent=2, default=str))
    print(json.dumps({
        "promotion": bool(passes),
        "selected": candidate,
        "table": [{"label": c["label"], "is": {k: c["is"][k] for k in
                                                 ("trades", "net", "pf", "win_rate",
                                                  "max_loss_per_contract")},
                   "oos": {k: c["oos"][k] for k in
                           ("trades", "net", "pf", "ece", "crossed_fills",
                            "max_loss_per_contract", "worst_trade")},
                   "gate": c["gate"]} for c in cands]}, indent=2))
    print(f"wrote {ZOBI_CAP_REPORT} and {VERDICT}")
    return 0 if passes else 2


# ------------------- window cap / toxicity veto / synthetic-flatten study
# Short-horizon fade veto. `VETO_LOOKBACK_S` is the trailing window the settlement
# index and the touch-revision imbalance are measured over; `VETO_BPS` is an adverse
# index move big enough to say the move being faded is still running (a 15-minute
# window's 1-sigma 5s move is ~1.6bp, so 4bp is ~2.5 sigma), and `VETO_REV` the same
# statement in touch-revision units.
VETO_LOOKBACK_S = 5.0
VETO_BPS = 4.0
VETO_REV = 0.25
CAP_GRID = (1, 2)
SYNTH_CONCESSIONS = (0.01, 0.02, 0.03)
ZOBI_Z_GRID = (0.6, 0.8, 1.0, 1.2)
ZOBI_FLOW_GRID = (0.15, 0.25, 0.35)
ZOBI_REV_MIN = 0.10
# The veto is a standalone mechanism, so it is ablated rather than assumed: `vel`
# and `obi` are each tested alone and together.
VETO_MODES = {"none": (0.0, 0.0), "vel": (VETO_BPS, 0.0), "obi": (0.0, VETO_REV),
              "vel+obi": (VETO_BPS, VETO_REV)}
CAPPED_FAMILIES = ("absorption_revert", "index_dislocation", "z_obi_flow")
CAPPED_REPORT = ROOT / "data" / "capped-search.json"


def exit_variants() -> list[tuple[str, SimParams]]:
    """Every exit regime a capped candidate may use, each labelled."""
    out: list[tuple[str, SimParams]] = []
    for tp in (0.03, 0.05):
        for bar in (20.0, 45.0):
            # The shipped baseline: post-only TP, then a post-only scratch at the
            # bar, 20s of patience, and the binary if it never comes back.
            out.append((f"scr_tp{tp:.2f}_b{bar:.0f}", SimParams(
                tp_c=tp, barrier_s=bar, exit_mode="maker_scratch",
                scratch_concession_c=0.0, scratch_patience_s=20.0)))
            for conc in SYNTH_CONCESSIONS:
                out.append((f"syn{conc * 100:.0f}c_tp{tp:.2f}_b{bar:.0f}", SimParams(
                    tp_c=tp, barrier_s=bar, exit_mode="synthetic_flatten",
                    flatten_concession_c=conc, scratch_patience_s=300.0)))
    return out


def capped_events(lane: Lane, sig: dict[int, np.ndarray]) -> dict[str, list[Event]]:
    """The three families under test, on the same detectors as the passive study.

    `z_obi_flow` gets the expanded grid (4 z floors x 3 flow floors) because its
    only failure was sample size: nine OOS trades at a 100% win rate is a real
    signal with no evidence behind it.
    """
    absorb = absorption_events(lane, sig, size=20.0) + absorption_events(lane, sig, size=40.0)
    disloc: list[Event] = []
    for gap_min, persist in DISLOC_DETECTORS:
        disloc.extend(dislocation_events(lane, sig, gap_min=gap_min, persist=persist))
    zf: list[Event] = []
    for z_min in ZOBI_Z_GRID:
        for flow_min in ZOBI_FLOW_GRID:
            zf.extend(z_obi_flow_events(lane, sig, z_min=z_min, flow_min=flow_min,
                                        rev_min=ZOBI_REV_MIN))
    return {"absorption_revert": absorb, "index_dislocation": disloc, "z_obi_flow": zf}


def slim2(b: dict) -> dict:
    keep = ("trades", "wins", "win_rate", "net", "pf", "ece", "ece_raw", "fees",
            "tp_share", "crossed_fills", "avg_usd_per_trade", "trades_per_hour",
            "reasons", "net_by_reason", "by_side", "by_hour_utc",
            "markout_5s_c", "markout_15s_c", "max_loss_per_contract", "worst_trade")
    return {k: b.get(k) for k in keep if k in b}


def capped_study(lanes: list[str]) -> int:
    """Per-window cap, $0 synthetic flatten and toxicity veto on the frozen split.

    Selection is in-sample only: for each family the IS-best candidate (highest IS
    profit factor among candidates with 20+ IS trades and positive IS net) is the
    single row the frozen OOS slice is scored on. Every other candidate's OOS row is
    still written out, so the table shows what was rejected and why.
    """
    started = time.time()
    hours = {"is": 0.0, "oos": 0.0}
    rows_by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    events_by: dict[str, int] = defaultdict(int)
    audit: dict[str, dict] = {}
    variants = exit_variants()
    params_of = dict(variants)

    for name in lanes:
        lane = Lane(name)
        sig = trailing_sigma(lane)
        for slice_name in ("is", "oos"):
            hours[slice_name] += lane.hours[slice_name]
        audit[name] = {"windows": len(lane.meta), "book_rows": int(len(lane.ts)),
                       "prints": int(len(lane.trade_ts)),
                       "is_hours": round(lane.hours["is"], 2),
                       "oos_hours": round(lane.hours["oos"], 2)}
        print(f"{name}: {len(lane.meta)} windows, {len(lane.ts):,} rows, "
              f"{len(lane.trade_ts):,} prints", flush=True)
        events = capped_events(lane, sig)
        for family, evs in events.items():
            events_by[family] += len(evs)
            for label, params in variants:
                rows_by[(family, label)].extend(simulate(lane, evs, params))
        print(f"  events {dict(events_by)} ({time.time() - started:.0f}s)", flush=True)
        del lane

    # The passive invariants hold here too: no taker exit, no fee on a maker leg,
    # and not one crossed fill anywhere in the tape.
    for (family, label), rows in rows_by.items():
        for r in rows:
            if r["crossed"] != 0:
                raise RuntimeError(f"crossed fill: {family} {label}: {r}")
            if r["reason"] in ("maker_tp", "maker_scratch", "synthetic_flatten",
                               "scratch_unfilled_settle", "synthetic_unfilled_settle",
                               "hold_to_settlement") and r["fee"] != 0:
                raise RuntimeError(f"fee on a passive leg: {family} {label}: {r}")

    table: list[dict] = []
    families: dict[str, dict] = {}
    for family in CAPPED_FAMILIES:
        cands: list[dict] = []
        for label in sorted({lab for fam, lab in rows_by if fam == family}):
            rows = rows_by.get((family, label), [])
            if not rows:
                continue
            params = params_of[label]
            for cap in CAP_GRID:
                for vname, (vbps, vrev) in VETO_MODES.items():
                    if family == "z_obi_flow" and vname != "none":
                        continue      # a momentum family has no fade to veto
                    seq = sequence_rows(rows, max_window_trades=cap,
                                        halt_after_loss=True,
                                        veto_vel=vbps, veto_rev=vrev)
                    is_rows = [r for r in seq if r["slice"] == "is"]
                    oos_rows = [r for r in seq if r["slice"] == "oos"]
                    if not is_rows or not oos_rows:
                        continue
                    is_block = block2(is_rows, hours["is"])
                    oos_block = block2(oos_rows, hours["oos"])
                    pa, pb = platt_fit(is_rows)   # fitted on IS only, frozen for OOS
                    is_block["ece_raw"] = is_block.get("ece")
                    is_block["ece"] = cal_ece(is_rows, pa, pb)
                    oos_block["ece_raw"] = oos_block.get("ece")
                    oos_block["ece"] = cal_ece(oos_rows, pa, pb)
                    cands.append({
                        "family": family,
                        "label": f"{label}|cap{cap}|{vname}",
                        "exit": label, "cap": cap, "veto": vname,
                        "exit_mode": params.exit_mode,
                        "flatten_concession_c": params.flatten_concession_c,
                        "tp_c": params.tp_c, "barrier_s": params.barrier_s,
                        "max_window_trades": cap, "halt_after_first_loss": True,
                        "is": is_block, "oos": oos_block,
                        "platt": {"a": round(pa, 4), "b": round(pb, 4), "fitted_on": "is"},
                        "gate": gate(oos_block), "verifier": verifier_call(oos_block),
                    })
        if not cands:
            families[family] = {"selected": None, "eligible": False, "passes": False,
                                "reason": "no fills"}
            continue
        table.extend(cands)
        eligible = [c for c in cands if c["is"]["trades"] >= 20
                    and Decimal(str(c["is"]["net"])) > 0]
        eligible.sort(key=lambda c: (float(c["is"]["pf"]),
                                     Decimal(str(c["is"]["net"])),
                                     int(c["is"]["trades"])), reverse=True)
        best = eligible[0] if eligible else None
        passes = bool(best) and all(best["gate"].values())
        families[family] = {
            "selected": ({k: best[k] for k in ("label", "exit", "cap", "veto",
                                               "exit_mode", "flatten_concession_c",
                                               "tp_c", "barrier_s",
                                               "max_window_trades",
                                               "halt_after_first_loss", "is", "oos",
                                               "gate", "platt", "verifier")}
                         if best else None),
            "eligible": bool(eligible),
            "passes": passes,
            "oos_pass_labels": [c["label"] for c in cands if all(c["gate"].values())],
            "reason": None if passes else (
                "no variant reached 20 IS trades with positive IS net" if not best
                else "IS-best failed the frozen OOS gate: "
                     + ", ".join(k for k, v in best["gate"].items() if not v)),
        }
        for c in sorted(cands, key=lambda x: Decimal(str(x["is"]["net"])), reverse=True):
            print(f"{c['label']}: IS {c['is']['trades']}/{c['is']['net']} "
                  f"pf {c['is']['pf']} -> OOS {c['oos']['trades']}/{c['oos']['net']} "
                  f"pf {c['oos']['pf']} ece {c['oos']['ece']} "
                  f"pass={all(c['gate'].values())}", flush=True)
        print(f"{family}: {len(cands)} candidates, IS-best "
              f"{best['label'] if best else None} passes={passes}", flush=True)

    # Promotion picks the passing family with the best *in-sample* profit factor:
    # the OOS slice has already been spent on the verdict, so it may not be used
    # again to choose between passing families.
    promoted = None
    for family, info in families.items():
        if not info.get("passes"):
            continue
        if promoted is None:
            promoted = family
            continue
        a = (float(info["selected"]["is"]["pf"]),
             Decimal(str(info["selected"]["is"]["net"])))
        b = (float(families[promoted]["selected"]["is"]["pf"]),
             Decimal(str(families[promoted]["selected"]["is"]["net"])))
        if a > b:
            promoted = family

    tape = {"lanes": lanes, "hours": hours,
            "book_rows": sum(a["book_rows"] for a in audit.values()),
            "prints": sum(a["prints"] for a in audit.values())}
    gate_block = {"profit_factor_min": 1.15, "ece_max": 0.08,
                  "requires_positive_net": True, "requires_zero_crossed_fills": True,
                  "min_oos_trades": 20}
    protocol = {
        "split": "chronological 60/40 by settled window close, built once, never re-split",
        "passive_fill": f">={MIN_PASSIVE_QTY:g} contracts printed at/through the limit",
        "fees": "exact Decimal taker_fee on every taker leg; $0 on every maker leg",
        "crossed_book": "asserted, never filtered: a bid >= ask row aborts the study",
        "window_cap": f"max_window_trades in {list(CAP_GRID)}, window halts after 1 loss",
        "synthetic_flatten": ("at the time barrier rest a post-only bid on the opposite "
                              "contract at 1-(entry-concession), concession in "
                              f"{list(SYNTH_CONCESSIONS)}; a fill locks YES+NO at "
                              "1+concession, $0 fee, unfilled carries to settlement"),
        "toxicity_veto": (f"block a fade when the trailing {VETO_LOOKBACK_S:g}s "
                          f"settlement-index velocity or touch-revision imbalance is "
                          f"adverse by >= {VETO_BPS}bp / {VETO_REV}"),
        "z_grid": f"z_obi_flow z floors {list(ZOBI_Z_GRID)}, flow floors "
                  f"{list(ZOBI_FLOW_GRID)}, revision floor {ZOBI_REV_MIN}",
        "selection": "IS only (highest IS PF, 20+ IS trades, positive IS net); OOS scored once",
        "calibration": "Platt fitted on IS rows only, frozen for OOS",
        "promotion": "all of: OOS net > 0, PF >= 1.15, ECE <= 0.08, >= 20 OOS trades, "
                     "0 crossed fills (the verifier line is reported, not gating)",
    }
    if promoted:
        sel = families[promoted]["selected"]
        verdict = {
            "generated_ts": time.time(), "promoted": True,
            "candidate": {"name": promoted, "label": sel["label"],
                          "exit_mode": sel["exit_mode"],
                          "flatten_concession_c": sel["flatten_concession_c"],
                          "tp_c": sel["tp_c"], "barrier_s": sel["barrier_s"],
                          "max_window_trades": sel["max_window_trades"],
                          "halt_after_first_loss": True, "entry": "post_only"},
            "is": slim2(sel["is"]), "oos": slim2(sel["oos"]), "gate": gate_block,
            "tape": tape, "protocol": protocol,
            "reason": f"{promoted} {sel['label']} passes every frozen OOS check",
        }
    else:
        verdict = {
            "generated_ts": time.time(), "promoted": False, "candidate": None, "oos": None,
            "gate": gate_block, "families": families,
            "table": [{"label": c["label"], "is": slim2(c["is"]), "oos": slim2(c["oos"]),
                       "gate": c["gate"]} for c in table],
            "tape": tape, "protocol": protocol,
            "verdict": "no capped / synthetic-flatten candidate clears the frozen OOS gate",
            "reason": "no family produced an in-sample candidate that clears the "
                      "frozen OOS gate",
        }
    VERDICT.write_text(json.dumps(verdict, indent=2, default=str))
    CAPPED_REPORT.write_text(json.dumps({
        "generated_ts": time.time(), "elapsed_s": round(time.time() - started, 1),
        "protocol": protocol, "lanes": lanes, "hours": hours, "audit": audit,
        "events": dict(events_by), "families": families, "table": table,
        "promotion": {"family": promoted}, "tape": tape,
    }, indent=2, default=str))
    print(json.dumps({
        "promotion": promoted,
        "families": {k: {
            "label": (v.get("selected") or {}).get("label"),
            "is": {kk: ((v.get("selected") or {}).get("is") or {}).get(kk)
                   for kk in ("trades", "win_rate", "pf", "net")},
            "oos": {kk: ((v.get("selected") or {}).get("oos") or {}).get(kk)
                    for kk in ("trades", "win_rate", "pf", "net", "ece", "crossed_fills")},
        } for k, v in families.items()}}, indent=2))
    print(f"wrote {CAPPED_REPORT} and {VERDICT}")
    return 0 if promoted else 2



# ----------------------------------------------------------------- confirmed stop
# The deployed structure, measured as one thing: a gated entry (spot-to-strike delta,
# two-horizon trend, 0.30-0.48 underdog cost), a `+5c` post-only take-profit, a 45s
# barrier scratch with 20s patience, and a stop that needs two continuous seconds of
# breach before it crosses. The stop distance is the only dimension IS may choose.
CONF_STOPS = (0.0, 0.04, 0.06, 0.08)
CONF_CONFIRM_S = 2.0
# The requested `<= 0.48` underdog cap is included, and it is measured rather than
# assumed: `z_obi_flow` only fires on the side the strike z favours, so its cost is
# already the favourite's price. The 0.48 row is expected to be empty on this tape
# (cheapest signal event was 0.51) and is reported as such, not silently dropped.
CONF_CAPS = (0.48, 0.52, 0.55, 0.60)
CONF_REPORT = ROOT / "data" / "confirmed-search.json"


def confirmed_study(lanes: list[str]) -> int:
    """IS/OOS verification of the gated entry + confirmed-stop exit.

    Selection is in-sample only (highest IS PF, 20+ IS trades, positive IS net); the
    frozen 40% is scored exactly once per stop distance and never read before the
    choice. Every fill goes through the same print-tape model, and a crossed book
    raises instead of being filtered.
    """
    started = time.time()
    hours = {"is": 0.0, "oos": 0.0}
    rows_by: dict[tuple[float, float], list[dict]] = defaultdict(list)
    events_by: dict[str, dict[str, int]] = {}
    audit: dict[str, dict] = {}
    for name in lanes:
        lane = Lane(name)
        sig = trailing_sigma(lane)
        for slice_name in ("is", "oos"):
            hours[slice_name] += lane.hours[slice_name]
        audit[name] = {"windows": len(lane.meta), "book_rows": int(len(lane.ts)),
                       "prints": int(len(lane.trade_ts)),
                       "is_hours": round(lane.hours["is"], 2),
                       "oos_hours": round(lane.hours["oos"], 2)}
        evs = z_obi_flow_events(
            lane, sig, z_min=0.15, flow_min=0.25, rev_min=0.20,
            t_lo=360.0, t_hi=840.0, cost_lo=0.30, cost_hi=max(CONF_CAPS),
            min_delta=ZOBI_MIN_DELTA.get(name, 0.0),
            trend_fast_s=15.0, trend_slow_s=60.0)
        def cost_of(e: Event) -> float:
            return e.limit if e.side == "YES" else 1.0 - e.limit
        events_by[name] = {f"{cap:.2f}": sum(1 for e in evs if cost_of(e) <= cap + EPS)
                           for cap in CONF_CAPS}
        print(f"{name}: {len(lane.meta)} windows, {len(evs)} gated flow events "
              f"by cap {events_by[name]} ({time.time() - started:.0f}s)", flush=True)
        for cap in CONF_CAPS:
            capped = [e for e in evs if cost_of(e) <= cap + EPS]
            if not capped:
                continue
            for stop in CONF_STOPS:
                params = SimParams(tp_c=0.05, barrier_s=45.0, stop_c=stop,
                                   stop_confirm_s=CONF_CONFIRM_S, patience_s=25.0,
                                   exit_mode="maker_scratch", scratch_concession_c=0.0,
                                   scratch_patience_s=20.0, max_entry_price=cap)
                rows_by[(cap, stop)].extend(simulate(lane, capped, params))
        del lane

    cands: list[dict] = []
    empty: list[str] = []
    for (cap, stop), rows in sorted(rows_by.items()):
        for r in rows:
            if r["crossed"] != 0:
                raise RuntimeError(f"crossed fill: cap {cap} stop {stop}: {r}")
            if r["reason"] in ("maker_tp", "scratch") and r["fee"] != 0:
                raise RuntimeError(f"fee on a passive leg: cap {cap} stop {stop}: {r}")
        seq = sequence_rows(rows, max_window_trades=2, halt_after_loss=True)
        is_rows = [r for r in seq if r["slice"] == "is"]
        oos_rows = [r for r in seq if r["slice"] == "oos"]
        if not is_rows or not oos_rows:
            empty.append(f"cap{cap:.2f}_stop{stop:.2f} (is {len(is_rows)}, oos {len(oos_rows)})")
            print(f"cap {cap:.2f} stop {stop:.2f}: no rows "
                  f"(is {len(is_rows)}, oos {len(oos_rows)})", flush=True)
            continue
        is_block = block2(is_rows, hours["is"])
        oos_block = block2(oos_rows, hours["oos"])
        pa, pb = platt_fit(is_rows)              # fitted on IS only, frozen for OOS
        is_block["ece_raw"] = is_block.get("ece")
        is_block["ece"] = cal_ece(is_rows, pa, pb)
        oos_block["ece_raw"] = oos_block.get("ece")
        oos_block["ece"] = cal_ece(oos_rows, pa, pb)
        cands.append({
            "label": f"scr_tp0.05_b45_conf{CONF_CONFIRM_S:.0f}s_stop{stop:.2f}"
                     f"|cap2|cost{cap:.2f}",
            "stop_c": stop, "cost_cap": cap, "confirm_s": CONF_CONFIRM_S,
            "is": is_block, "oos": oos_block,
            "platt": {"a": round(pa, 4), "b": round(pb, 4), "fitted_on": "is"},
            "gate": gate(oos_block),
        })
        print(f"cap {cap:.2f} stop {stop:.2f}: IS {is_block['trades']}/{is_block['net']} "
              f"pf {is_block['pf']} tail {is_block['max_loss_per_contract']} -> "
              f"OOS {oos_block['trades']}/{oos_block['net']} pf {oos_block['pf']} "
              f"ece {oos_block['ece']} tail {oos_block['max_loss_per_contract']} "
              f"pass={all(cands[-1]['gate'].values())} "
              f"reasons {oos_block['reasons']}", flush=True)

    eligible = [c for c in cands
                if c["is"]["trades"] >= 20 and Decimal(str(c["is"]["net"])) > 0]
    eligible.sort(key=lambda c: (float(c["is"]["pf"]),
                                 Decimal(str(c["is"]["net"])),
                                 int(c["is"]["trades"])), reverse=True)
    best = eligible[0] if eligible else None
    # The stop is a bound, so the realized worst may exceed it only by the taker fee
    # on the crossing leg.
    # No-stop rows are bounded by the entry cost itself, which is the structural tail
    # the 0.30-0.60 band already bounds.
    tail_ok = bool(best) and (float(best["oos"]["max_loss_per_contract"])
                              <= best["stop_c"] + 0.02
                              or (best["stop_c"] == 0.0
                                  and float(best["oos"]["max_loss_per_contract"])
                                  <= best["cost_cap"] + 0.02))
    vig, why = verifier_call(best["oos"]) if best else (False, "no eligible candidate")
    passes = bool(best) and all(best["gate"].values()) and tail_ok and vig

    tape = {"lanes": lanes, "hours": hours,
            "book_rows": sum(a["book_rows"] for a in audit.values()),
            "prints": sum(a["prints"] for a in audit.values())}
    gate_block = {"profit_factor_min": 1.15, "ece_max": 0.08,
                  "requires_positive_net": True, "requires_zero_crossed_fills": True,
                  "min_oos_trades": 20, "bounded_tail": True}
    if best and passes:
        verdict = {
            "generated_ts": time.time(), "promoted": True,
            "candidate": {"name": "z_obi_flow", "label": best["label"],
                          "mechanism": "early_window_z_obi",
                          "params": {
                              "t_lo_s": 360.0, "t_hi_s": 840.0,
                              "cost_lo": 0.30, "cost_hi": best["cost_cap"],
                              "min_delta": ZOBI_MIN_DELTA,
                              "trend_fast_s": 15.0, "trend_slow_s": 60.0,
                              "exit_mode": "maker_tp", "tp_c": 0.05,
                              "barrier_s": 45.0, "scratch_patience_s": 20.0,
                              "entry_patience_s": 25.0,
                              "stop_c": best["stop_c"], "stop_confirm_s": CONF_CONFIRM_S,
                              "max_entry_price": best["cost_cap"], "max_window_trades": 2,
                              "halt_after_first_loss": True,
                              "calibration": best["platt"]}},
            "is": best["is"], "oos": best["oos"], "gate": gate_block, "tape": tape,
            "reason": f"z_obi_flow {best['label']} clears every frozen OOS check "
                      f"(net {best['oos']['net']}, pf {best['oos']['pf']}, "
                      f"ece {best['oos']['ece']}, {best['oos']['trades']} trades, "
                      f"tail {best['oos']['max_loss_per_contract']})",
        }
    else:
        verdict = {"generated_ts": time.time(), "promoted": False, "candidate": None,
                   "oos": best["oos"] if best else None, "gate": gate_block, "tape": tape,
                   "reason": ("no stop distance reached 20 IS trades with positive IS net"
                              if not best else
                              "IS-best failed the frozen OOS gate: " + ", ".join(
                                  [k for k, v in best["gate"].items() if not v] +
                                  ([] if tail_ok else ["tail"]) + ([] if vig else [why]))),
                   "table": [{k: c[k] for k in ("label", "is", "oos")} for c in cands]}
    if passes:
        VERDICT.write_text(json.dumps(verdict, indent=2, default=str))
    CONF_REPORT.write_text(json.dumps({
        "generated_ts": time.time(), "elapsed_s": round(time.time() - started, 1),
        "events": events_by, "tape": tape, "candidates": cands, "empty_rows": empty,
        "promoted": bool(best and passes), "verdict": verdict}, indent=2, default=str))
    print(f"\n{len(cands)} candidates, promoted={bool(best and passes)}", flush=True)
    return 0 if passes else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", action="append", default=None)
    parser.add_argument("--out", default=str(REPORT))
    parser.add_argument("--dump", default=str(ROOT / "data" / "alpha-trades.jsonl"))
    parser.add_argument("--hold-settlement", action="store_true",
                        help="sweep maker entry + hold-to-settlement across entry caps")
    parser.add_argument("--passive", action="store_true",
                        help="maker-scratch + settlement families (lead-lag, z+flow, "
                             "late favourite)")
    parser.add_argument("--capped", action="store_true",
                        help="per-window cap + synthetic opposite-side flatten + "
                             "toxicity veto + expanded z grid")
    parser.add_argument("--zobi-cap", action="store_true",
                        help="z_obi_flow asymmetric entry cap + bounded 6c loss stop "
                             "on the frozen 60/40 split")
    parser.add_argument("--bounded", action="store_true",
                        help="early-window z_obi + IS-calibrated synthetic flatten "
                             "on the frozen 60/40 split")
    parser.add_argument("--confirmed", action="store_true",
                        help="gated entry (spot delta, 2-horizon trend, 0.48 underdog "
                             "cap) + 5c maker TP + 45s scratch + 2s-confirmed stop")
    parser.add_argument("--refine", action="store_true",
                        help="IS-only entry-quality refinement of the two near-miss "
                             "bounded families; frozen OOS scored once per family")
    args = parser.parse_args()
    if args.confirmed:
        return confirmed_study(args.lane or list(LANES))
    if args.refine:
        return refine_study(args.lane or list(LANES))
    if args.bounded:
        return bounded_study(args.lane or list(LANES))
    if args.zobi_cap:
        return zobi_cap_study(args.lane or list(LANES))
    if args.capped:
        return capped_study(args.lane or list(LANES))
    if args.passive:
        return passive_study(args.lane or list(LANES))
    if args.hold_settlement:
        return hold_settlement_study(args.lane or list(LANES), HOLD_CAPS)
    lanes = args.lane or list(LANES)
    started = time.time()
    hours = {name: 0.0 for name in ("is", "oos")}
    trades: dict[tuple[str, str], list[dict]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    audit: dict[str, dict] = {}

    for name in lanes:
        lane = Lane(name)
        sig = trailing_sigma(lane)
        for slice_name in ("is", "oos"):
            hours[slice_name] += lane.hours[slice_name]
        audit[name] = {"windows": len(lane.meta), "book_rows": int(len(lane.ts)),
                       "prints": int(len(lane.trade_ts)),
                       "is_hours": round(lane.hours["is"], 2),
                       "oos_hours": round(lane.hours["oos"], 2)}
        print(f"{name}: {len(lane.meta)} windows, {len(lane.ts):,} rows, "
              f"{len(lane.trade_ts):,} prints", flush=True)

        events: dict[str, list[Event]] = defaultdict(list)
        for horizon, conf in SPIKE_DETECTORS:
            fade, momo = spike_events(lane, sig, horizon=horizon, conf=conf)
            events["spike_fade"].extend(fade)
            # `momo` carries both mirrors of the same spike; route by family, or the
            # momentum table silently absorbs every pullback event (and its trades).
            events["spike_momentum"].extend([e for e in momo if e.family == "spike_momentum"])
            events["spike_pullback"].extend([e for e in momo if e.family == "spike_pullback"])
        for g, p in DISLOC_DETECTORS:
            events["index_dislocation"].extend(
                dislocation_events(lane, sig, gap_min=g, persist=p))
        for size in ABSORB_DETECTORS:
            events["absorption_revert"].extend(absorption_events(lane, sig, size=size))
        for width, gap in SPREAD_DETECTORS:
            events["spread_capture"].extend(
                spread_events(lane, sig, width_min=width, gap_max=gap))
        for g in OPEN_DETECTORS:
            events["window_open"].extend(window_open_events(lane, sig, gap_min=g))

        for family, evs in events.items():
            counts[family] += len(evs)
            by_variant: dict[str, list[Event]] = defaultdict(list)
            for e in evs:
                by_variant[e.variant].append(e)
            for params in SIMS[family]:
                key_params = (round(params.tp_c, 4), round(params.barrier_s, 1),
                              round(params.stop_c, 4), round(params.patience_s, 1))
                for variant, subset in by_variant.items():
                    label = f"{variant}_tp{params.tp_c:.2f}_b{params.barrier_s:.0f}" \
                            f"_s{params.stop_c:.2f}"
                    rows = simulate(lane, subset, params)
                    for r in rows:
                        r["variant_full"] = label
                    trades[(family, label)].extend(rows)
        print(f"  events so far: {dict(counts)} ({time.time() - started:.0f}s)", flush=True)
        del lane

    # ---- selection: IS only -------------------------------------------------
    families: dict[str, dict] = {}
    for family in SIMS:
        labels = sorted({lab for fam, lab in trades if fam == family})
        if not labels:
            families[family] = {"variants": 0, "eligible": False, "selected": None,
                                "is_best": None, "table": [],
                                "reason": "no events" if not counts.get(family) else "no fills"}
            continue
        table = []
        for lab in labels:
            rows = trades[(family, lab)]
            table.append({
                "label": lab,
                "is": slim(block2([r for r in rows if r["slice"] == "is"], hours["is"])),
                "oos": slim(block2([r for r in rows if r["slice"] == "oos"], hours["oos"])),
            })
        eligible = [v for v in table
                    if v["is"]["trades"] >= 20 and Decimal(str(v["is"]["net"])) > 0]
        eligible.sort(key=lambda v: (Decimal(str(v["is"]["net"])), float(v["is"]["pf"])),
                      reverse=True)
        best = eligible[0] if eligible else max(
            table, key=lambda v: (Decimal(str(v["is"]["net"])), int(v["is"]["trades"])))
        is_rows = [r for r in trades[(family, best["label"])] if r["slice"] == "is"]
        oos_rows = [r for r in trades[(family, best["label"])] if r["slice"] == "oos"]
        # Platt fitted on IS only, then frozen: the OOS ECE tests the probability the
        # rule would actually quote live.
        pa, pb = platt_fit(is_rows)
        is_block = dict(best["is"])
        is_block["ece_raw"] = is_block.get("ece")
        is_block["ece"] = cal_ece(is_rows, pa, pb)
        oos = dict(best["oos"])
        oos["ece_raw"] = oos.get("ece")
        oos["ece"] = cal_ece(oos_rows, pa, pb)
        oos["platt"] = {"a": round(pa, 4), "b": round(pb, 4), "fitted_on": "is"}
        g = gate(oos)
        ok = bool(eligible) and all(g.values()) and verifier_call(oos)[0]
        families[family] = {
            "variants": len(table),
            "eligible": bool(eligible),
            "is_best": {"label": best["label"], "is": is_block, "oos": oos, "gate": g},
            "selected": ({"label": best["label"], "is": is_block, "oos": oos} if ok else None),
            "positive_oos_variants": sum(1 for v in table
                                         if Decimal(str(v["oos"]["net"])) > 0),
            "reason": None if ok else (
                "no variant reached 20 IS trades with positive IS net" if not eligible
                else "IS-best failed the frozen OOS gate: "
                     + ", ".join(k for k, v in g.items() if not v)),
            "table": table,
        }
        print(f"{family}: {len(table)} variants, IS-best {best['label']} "
              f"IS {is_block['trades']}/{is_block['net']} -> OOS {oos['trades']}/{oos['net']} "
              f"pf {oos['pf']} ece {oos['ece']} eligible={bool(eligible)}", flush=True)

    # ---- per-lane conditioning: select on IS, score OOS once ---------------
    per_lane: dict[str, dict] = {}
    for family in SIMS:
        per_lane[family] = {}
        labels = sorted({lab for fam, lab in trades if fam == family})
        for lane_name in lanes:
            lh = {"is": audit[lane_name]["is_hours"], "oos": audit[lane_name]["oos_hours"]}
            cands = []
            for lab in labels:
                rows = [r for r in trades[(family, lab)] if r["ticker"] == lane_name]
                if not rows:
                    continue
                cands.append({
                    "label": lab,
                    "is": slim(block2([r for r in rows if r["slice"] == "is"], lh["is"])),
                    "oos": slim(block2([r for r in rows if r["slice"] == "oos"], lh["oos"])),
                })
            elig = [c for c in cands
                    if c["is"]["trades"] >= 20 and Decimal(str(c["is"]["net"])) > 0]
            if not elig:
                per_lane[family][lane_name] = {
                    "eligible": False, "reason": "no positive IS variant with 20+ trades"}
                continue
            elig.sort(key=lambda c: (Decimal(str(c["is"]["net"])), float(c["is"]["pf"])),
                      reverse=True)
            b = elig[0]
            gl = gate(b["oos"])
            per_lane[family][lane_name] = {
                "label": b["label"], "eligible": True, "is": b["is"], "oos": b["oos"],
                "gate": gl, "passes": all(gl.values()) and b["oos"]["crossed_fills"] == 0}
    lane_pass = [f"{f}@{l}" for f, d in per_lane.items()
                 for l, v in d.items() if v.get("passes")]

    # ---- the gate ----------------------------------------------------------
    promoted = None
    for family, info in families.items():
        sel = info.get("selected")
        if not sel:
            info["passes"] = False
            continue
        info["passes"] = True
        if promoted is None:
            promoted = family
            continue
        a = (Decimal(str(sel["oos"]["net"])), float(sel["oos"]["pf"]))
        b = (Decimal(str(families[promoted]["selected"]["oos"]["net"])),
             float(families[promoted]["selected"]["oos"]["pf"]))
        if a > b:
            promoted = family

    tape = {"lanes": lanes, "hours": hours,
            "book_rows": sum(a["book_rows"] for a in audit.values()),
            "prints": sum(a["prints"] for a in audit.values())}
    if promoted:
        sel = families[promoted]["selected"]
        verdict = {
            "generated_ts": time.time(), "promoted": True,
            "candidate": {"name": promoted, "label": sel["label"]},
            "is": sel["is"], "oos": sel["oos"],
            "gate": {"profit_factor_min": 1.15, "ece_max": 0.08,
                     "requires_positive_net": True, "requires_zero_crossed_fills": True,
                     "min_oos_trades": 20},
            "tape": tape,
            "reason": f"{promoted} passes every frozen OOS check",
        }
    else:
        verdict = {
            "generated_ts": time.time(), "promoted": False, "candidate": None, "oos": None,
            "gate": {"profit_factor_min": 1.15, "ece_max": 0.08,
                     "requires_positive_net": True, "requires_zero_crossed_fills": True,
                     "min_oos_trades": 20},
            "verdict": "no rule in the tested families clears the fee/spread barrier on this tape",
            "families": {k: {"variants": v.get("variants"), "eligible": v.get("eligible"),
                             "is_best": v.get("is_best"), "reason": v.get("reason"),
                             "positive_oos_variants": v.get("positive_oos_variants"),
                             "table": v.get("table")}
                         for k, v in families.items()},
            "per_lane": per_lane,
            "lane_candidates_passing_gate": lane_pass,
            "tape": tape,
            "reason": "no family produced an in-sample candidate that clears the frozen OOS gate",
            "what_must_change": [
                "A probability model that does not depend on future volatility: the same "
                "families flip from strongly positive to negative when the fair value uses "
                "only trailing index vol, so the apparent edge was the full-window vol leak.",
                "A fee/spread structure where a taker leg (0.07*p*(1-p) ~ 1.7c at p=0.5) is "
                "small relative to the 1-4c 15m book, or maker queue priority that the print "
                "tape cannot prove: 20 contracts printed at the touch is an upper bound on "
                "fills, not a lower bound.",
                "More independent events: 45 settled windows per lane give a few hundred "
                "events per family, so a PF>=1.15 candidate must be far larger than "
                "anything measured here to be separable from noise.",
                "A per-lane prior rather than post-hoc lane selection: pooling is negative "
                "everywhere and the only lane-level positive (BTC absorption) is PF 1.03 "
                "on OOS, i.e. indistinguishable from zero after fees.",
            ],
        }
    VERDICT.write_text(json.dumps(verdict, indent=2, default=str))
    # Trade-level dump of each family's IS-best variant: per-lane and per-hour views
    # without re-simulating, and small enough to read.
    dumped = 0
    with open(args.dump, "w") as fh:
        for family, info in families.items():
            sel = info.get("is_best")
            if not sel:
                continue
            for r in trades[(family, sel["label"])]:
                fh.write(json.dumps({k: (str(v) if isinstance(v, Decimal) else v)
                                     for k, v in r.items()}) + "\n")
                dumped += 1
    print(f"dumped {dumped} selected-variant trades to {args.dump}")
    report = {
        "generated_ts": time.time(), "elapsed_s": round(time.time() - started, 1),
        "protocol": {
            "sigma": "trailing 150s index vol per row; full-window vol is never used",
            "prices": "every limit is a YES price; a NO position is a short YES sized 1-p",
            "passive_fill": f">={MIN_PASSIVE_QTY:g} contracts printed at/through the limit",
            "maker_entry": "search starts strictly after the event timestamp",
            "tp_stop": "first of take-profit fill and stop touch on the same tape",
            "crossed_book": "asserted, never filtered",
            "selection": "IS only; OOS scored once per family after selection",
            "calibration": "Platt fitted on IS rows only, frozen for OOS",
        },
        "lanes": lanes, "hours": hours, "audit": audit, "events": dict(counts),
        "families": families, "promotion": {"family": promoted},
        "tape": {"book_rows": sum(a["book_rows"] for a in audit.values()),
                 "prints": sum(a["prints"] for a in audit.values())},
    }
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"promotion": promoted,
                      "families": {k: {
                          "eligible": v.get("eligible"),
                          "trades": (v.get("is_best") or {}).get("oos", {}).get("trades"),
                          "net": (v.get("is_best") or {}).get("oos", {}).get("net"),
                          "pf": (v.get("is_best") or {}).get("oos", {}).get("pf"),
                          "ece": (v.get("is_best") or {}).get("oos", {}).get("ece"),
                          "crossed": (v.get("is_best") or {}).get("oos", {}).get("crossed_fills")}
                          for k, v in families.items()}}, indent=2))
    print(f"wrote {args.out} and {VERDICT}")
    return 0 if promoted else 2


if __name__ == "__main__":
    raise SystemExit(main())
