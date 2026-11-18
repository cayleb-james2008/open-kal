"""Deterministic quantitative kernel: no model ever computes these.

The plan's hard rule is that a System-One decision model must never do binary
pricing arithmetic from raw ticks. So the fair value, the volatility estimate,
the lead-lag drift, the spread, the quote age and the exact fee are all computed
here in plain Python with Decimal where money is involved.
"""
from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from .fees import taker_fee

SQRT2 = math.sqrt(2.0)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT2))


def log_returns(points: list[tuple[float, float]], window_s: float) -> list[float]:
    """Log returns per sqrt(second), sampled between points ~`window_s` apart."""
    out: list[float] = []
    for i in range(len(points) - 1, 0, -1):
        t_now, p_now = points[i]
        for j in range(i - 1, -1, -1):
            t_prev, p_prev = points[j]
            if t_now - t_prev >= window_s:
                dt = t_now - t_prev
                if p_prev > 0 and p_now > 0 and dt > 0:
                    out.append(math.log(p_now / p_prev) / math.sqrt(dt))
                break
    return out


def realized_sigma(points: list[tuple[float, float]], window_s: float = 2.0) -> float:
    """Per-sqrt-second volatility from the spot tape, EWMA-weighted.

    Returns 0.0 when the tape is too short to be trusted; callers treat 0 as "no
    view" and abstain rather than pretending a flat line is an edge.
    """
    rets = log_returns(points, window_s)
    if len(rets) < 10:
        return 0.0
    # Plain RMS of per-sqrt-second returns. A recency-weighted estimate that starts
    # from zero reads far too calm for the first few minutes, which is exactly when
    # the window is most tradeable.
    var = sum(r * r for r in rets) / len(rets)
    if var <= 0:
        return 0.0
    # Floor: a tape that looks perfectly quiet is still not a certain outcome.
    return max(math.sqrt(var), 2.5e-5)


@dataclass(frozen=True)
class FairValue:
    p_up: float
    sigma: float
    drift: float
    z: float
    seconds_left: float
    enough_data: bool


def fair_value(spot: float, strike: float, seconds_left: float, sigma_per_sqrt_s: float,
               momentum: float = 0.0) -> FairValue:
    """Zero-drift GBM probability that the settlement average lands at or above strike.

    The contract settles on the 60-second BRTI average before close against the
    average before open, which for a 15-minute window is close enough to a plain
    binary on spot at close. The model is deliberately simple and auditable: one
    normal CDF, no fitted parameters on the hot path.
    """
    if spot <= 0 or strike <= 0 or seconds_left <= 0:
        return FairValue(0.5, sigma_per_sqrt_s, 0.0, 0.0, seconds_left, False)
    vol = sigma_per_sqrt_s * math.sqrt(seconds_left)
    if vol <= 0:
        return FairValue(0.5, sigma_per_sqrt_s, 0.0, 0.0, seconds_left, False)
    # Lead-lag tilt: a fraction of the recent constituent move, capped at 40% of
    # one window's sigma so a single print can never flip the view.
    drift = max(-0.4 * vol, min(0.4 * vol, momentum * 0.35))
    z = (math.log(spot / strike) - drift) / vol
    return FairValue(normal_cdf(z), sigma_per_sqrt_s, drift, z, seconds_left, True)


def momentum(points: list[tuple[float, float]], horizon_s: float = 30.0) -> float:
    """Log return over the last `horizon_s` seconds of spot ticks."""
    if len(points) < 2:
        return 0.0
    t_end, p_end = points[-1]
    for t, p in reversed(points):
        if t_end - t >= horizon_s:
            if p <= 0 or p_end <= 0:
                return 0.0
            return math.log(p_end / p)
    return 0.0


def spread(bid: Decimal | None, ask: Decimal | None) -> Decimal:
    if bid is None or ask is None:
        return Decimal("1")
    return max(Decimal("0"), ask - bid)


def quote_age_s(iso_ts: str | None) -> float:
    if not iso_ts:
        return 1e9
    try:
        ts = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    except ValueError:
        return 1e9
    return max(0.0, (datetime.now(timezone.utc) - ts).total_seconds())


def net_edge(side_price: Decimal, contracts: int, p_model: float, maker: bool) -> Decimal:
    """Modeled profit per contract after the exact fee, for a fill at `side_price`."""
    if contracts <= 0:
        return Decimal("0")
    p = Decimal(str(p_model))
    gross = (p - Decimal(side_price)) if side_price < Decimal("0.5") else (Decimal(1) - p - (Decimal(1) - Decimal(side_price)))
    # `side_price` is the price paid for the side the model favours, so the
    # expected payout is the model probability of that side winning.
    fee = taker_fee(side_price, contracts) / Decimal(contracts) if not maker else Decimal("0")
    return gross - fee


def brier(pairs: list[tuple[float, int]]) -> float:
    if not pairs:
        return 0.0
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


# ---------------------------------------------------------------------------
# Probability calibration
#
# `p_cal = sigmoid(a * logit(p_raw) + b)` is a two-parameter map, fitted by IRLS on
# in-sample (predicted, realised) pairs only. It is deterministic: a fixed iteration
# count means the same tape always yields the same (a, b), which is what a replay
# needs to reproduce a live probability.
#
# The fit is allowed to return identity. On this tape it does: fitted per lane on 12h
# of in-sample ticks, every (a, b) made the frozen out-of-sample error *worse* (0.053
# pooled, 0.086-0.116 per lane) than the raw probability (0.031 total, 0.0625 worst
# lane). A calibration that only looks better in the window it was fitted to is a
# regression, so the validated default here is identity — the machinery is live, the
# fitted values are not promoted until they beat identity on the holdout.
# ---------------------------------------------------------------------------


def _logit(p: float) -> float:
    q = min(1.0 - 1e-6, max(1e-6, float(p)))
    return math.log(q / (1.0 - q))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass(frozen=True)
class Calibrator:
    """A per-lane monotone recalibration of the kernel's probability."""

    a: float = 1.0
    b: float = 0.0
    samples: int = 0
    note: str = "identity (no out-of-sample improvement from a fitted map)"

    def apply(self, p: float) -> float:
        if self.a == 1.0 and self.b == 0.0:
            return p
        return _sigmoid(self.a * _logit(p) + self.b)

    def to_json(self) -> dict[str, Any]:
        return {"a": round(self.a, 6), "b": round(self.b, 6),
                "samples": self.samples, "note": self.note}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Calibrator":
        return cls(a=float(data.get("a", 1.0)), b=float(data.get("b", 0.0)),
                   samples=int(data.get("samples", 0)), note=str(data.get("note", "")))


def fit_calibrator(pairs: list[tuple[float, int]], *, iterations: int = 60,
                   l2: float = 1e-3) -> Calibrator:
    """Newton/IRLS for (a, b) over (p_raw, y). Pure Python, fixed iteration count."""
    if len(pairs) < 100:
        return Calibrator(samples=len(pairs), note="too few pairs; identity")
    xs = [_logit(p) for p, _ in pairs]
    ys = [float(y) for _, y in pairs]
    a, b = 1.0, 0.0
    for _ in range(iterations):
        g_a = g_b = h_aa = h_ab = h_bb = 0.0
        for x, y in zip(xs, ys):
            z = max(-30.0, min(30.0, a * x + b))
            p = _sigmoid(z)
            w = max(p * (1.0 - p), 1e-6)
            g_a += (p - y) * x
            g_b += p - y
            h_aa += w * x * x
            h_ab += w * x
            h_bb += w
        g_a += l2 * (a - 1.0)
        g_b += l2 * b
        h_aa += l2
        h_bb += l2
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-12:
            break
        da = (h_bb * g_a - h_ab * g_b) / det
        db = (h_aa * g_b - h_ab * g_a) / det
        a = max(0.05, min(12.0, a - max(-0.5, min(0.5, da))))
        b = max(-4.0, min(4.0, b - max(-0.5, min(0.5, db))))
    return Calibrator(a=a, b=b, samples=len(pairs), note="fitted by IRLS")


def load_calibrators(path: Any) -> dict[str, Calibrator]:
    """Lane -> calibrator from a JSON file; identity when the file is absent."""
    from pathlib import Path as _Path

    target = _Path(path)
    if not target.exists():
        return {}
    try:
        raw = json.loads(target.read_text())
    except (ValueError, OSError):
        return {}
    lanes = raw.get("lanes") if isinstance(raw, dict) else None
    if not isinstance(lanes, dict):
        return {}
    return {str(k): Calibrator.from_json(v) for k, v in lanes.items() if isinstance(v, dict)}


def calibrate(p: float, calibrator: Calibrator | None) -> float:
    if calibrator is None:
        return p
    return min(1.0, max(0.0, calibrator.apply(p)))


def expected_calibration_error(pairs: list[tuple[float, int]], bins: int = 10) -> float:
    """ECE over (predicted probability, realised outcome) pairs, equal-width bins."""
    if not pairs:
        return 0.0
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for p, y in pairs:
        idx = min(bins - 1, max(0, int(p * bins)))
        buckets[idx].append((p, y))
    total = len(pairs)
    ece = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        avg_p = sum(p for p, _ in bucket) / len(bucket)
        avg_y = sum(y for _, y in bucket) / len(bucket)
        ece += (len(bucket) / total) * abs(avg_p - avg_y)
    return ece


class SpotTape:
    """Rolling spot prices per asset, with a max age so a stale feed cannot price a trade."""

    def __init__(self, max_age_s: float = 120.0, max_points: int = 2400) -> None:
        self.max_age_s = max_age_s
        self.max_points = max_points
        self._points: dict[str, deque[tuple[float, float]]] = {}

    def add(self, asset: str, ts: float, price: float) -> None:
        dq = self._points.setdefault(asset, deque(maxlen=self.max_points))
        if dq and dq[-1][0] == ts:
            dq[-1] = (ts, price)
        else:
            dq.append((ts, price))

    def points(self, asset: str) -> list[tuple[float, float]]:
        dq = self._points.get(asset)
        if not dq:
            return []
        cutoff = dq[-1][0] - self.max_age_s
        return [(t, p) for t, p in dq if t >= cutoff]

    def last(self, asset: str) -> tuple[float, float] | None:
        pts = self.points(asset)
        return pts[-1] if pts else None

    def age(self, asset: str, now: float) -> float:
        last = self.last(asset)
        return 1e9 if last is None else now - last[0]


# ---------------------------------------------------------------------------
# Microstructure alpha kernel
#
# The zero-drift GBM above is a description of a coin flip with a strike, and the
# tape confirmed it: Brier 0.2271 against the book's 0.2187. These three signals are
# the replacements. Each one is a mechanical statement about a specific market
# microstructure, and each one is falsifiable on the recorded tape.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Signal:
    """One tradeable thesis, priced by the kernel, never by a model."""

    name: str                  # impulse_snipe | final_minute | cross_book
    p_up: float                # probability YES settles 1
    detail: str
    regime: str = "micro"
    arb_profit: Decimal = Decimal("0")
    impulse_bps: float = 0.0   # signed size of the spot move that triggered it
    # The settlement program's own evidence, carried with the thesis so the
    # strategy can apply its own filters without re-reading the tape.
    l2_obi: float = 0.0        # true-L2 book imbalance in [-1, 1]
    flow_fast: float = 0.0     # signed aggressor print flow, fast window
    flow_slow: float = 0.0     # signed aggressor print flow, slow window


def spot_impulse_bps(points: list[tuple[float, float]], lookback_s: float = 10.0) -> float:
    """Signed spot move over the last `lookback_s`, in basis points."""
    if len(points) < 3:
        return 0.0
    t_end, p_end = points[-1]
    for t, p in reversed(points):
        if t_end - t >= lookback_s:
            if p <= 0 or p_end <= 0:
                return 0.0
            return (p_end / p - 1.0) * 10_000.0
    return 0.0


def index_impulse_bps(prints: list[tuple[float, float]], lookback_s: float = 5.0) -> float:
    """The same impulse measured on the CF Benchmarks prints the contract settles on."""
    return spot_impulse_bps(prints, lookback_s)


def lead_lag(*, spot_bps: float, index_bps: float, book_move_c: float, min_bps: float,
             max_bps: float, max_reprice_c: float) -> Signal | None:
    """Spot -> BRTI -> Kalshi, in that order.

    Only fires when the settlement index has already confirmed the spot impulse and
    the Kalshi ladder has not repriced yet. That gap is the entire edge: we are buying
    a stale quote on a move the venue's own reference feed has already printed.
    """
    magnitude = abs(spot_bps)
    if magnitude < min_bps or magnitude > max_bps:
        return None
    direction = 1.0 if spot_bps > 0 else -1.0
    # The index must confirm in the same direction, at least half the move.
    if index_bps * direction < magnitude * 0.5:
        return None
    # And the book must still be where it was: a repriced ladder is no longer stale.
    if abs(book_move_c) > max_reprice_c:
        return None
    # Map an 8-15bp impulse onto a modest probability tilt. 15bp on a 15-minute
    # window is roughly a third of a sigma, so ~0.60-0.65 is honest, not 0.9.
    tilt = min(0.16, magnitude / 10_000.0 * 110.0)
    p_up = 0.5 + direction * tilt
    return Signal("lead_lag", p_up,
                  f"spot {spot_bps:+.1f}bp confirmed by index {index_bps:+.1f}bp, "
                  f"book still {book_move_c:+.2f}c", regime="lead_lag")


def final_minute(*, spot: float, strike: float, locked_avg: float, locked_ticks: int,
                 sigma_per_sqrt_s: float, seconds_left: float,
                 window_s: float = 60.0) -> Signal | None:
    """The Asian lock-in: how much of the settlement average is already printed.

    The final minute's settlement average is `(k * avg_sofar + (60-k) * future) / 60`.
    With k prints already fixed, only the remaining (60-k) seconds are unknown, and
    their variance collapses as the cube of that share: a Brownian bridge over a
    shrinking slice of the window.
    """
    if strike <= 0 or spot <= 0 or locked_ticks <= 0 or seconds_left <= 0:
        return None
    k = float(min(locked_ticks, window_s))
    remaining = max(0.0, window_s - k)
    if remaining <= 0:
        # Every second is printed: settlement is arithmetic, not a probability.
        settle = locked_avg
        return Signal("final_minute", 1.0 if settle >= strike else 0.0,
                      f"locked {settle:.2f} vs strike {strike:.2f}")
    # Future average must clear this for settlement to land at/above the strike.
    threshold = (window_s * strike - k * locked_avg) / remaining
    # The standard deviation of the *mean* of a Brownian path over `remaining`
    # seconds is sigma * sqrt(remaining / 3): the variance of a time average
    # collapses like the cube of the slice it averages. The form this used to
    # ship, sqrt(T) * (T/60) ** 1.5, is a T^2 law, and it is far too tight inside
    # the final minute -- 8x too tight with 10 seconds left, 1.6x too tight with
    # 30 seconds left. Measured on the frozen tape (5,991 last-minute index rows,
    # 38 windows per lane), the T^2 law collapses fourteen genuinely uncertain
    # rows into "confident" ones in the 15-to-30-seconds band and puts the
    # calibration gap there at 0.438 against the T^1.5 law's 0.112. A model that
    # is certain when the outcome is a coin flip is how a 52% signal produced a
    # 94%-hit-rate profile and cost the account $44.55.
    # sigma is a log-return scale, so the dollar sigma scales with spot.
    sd = spot * sigma_per_sqrt_s * math.sqrt(remaining / 3.0)
    if sd <= 0:
        return None
    z = (threshold - spot) / sd
    p_up = 1.0 - normal_cdf(z)
    share = k / window_s
    return Signal("final_minute", min(1.0, max(0.0, p_up)),
                  f"{k:.0f}/60 locked at {locked_avg:.2f} vs strike {strike:.2f}, "
                  f"needs {threshold:.2f}, spot {spot:.2f}, sd {sd:.2f}",
                  regime="lock_in" if share > 0.5 else "final_minute")


def impulse_snipe(*, prints: list[tuple[float, float]], spot: float, strike: float,
                  seconds_left: float, sigma: float,
                  book_hist: list[tuple[float, Decimal]] | None,
                  min_seconds_left: float, lookback_s: float = 5.0,
                  min_bps: float = 12.0, max_bps: float = 20.0,
                  stale_share: float = 0.5, max_book_age_s: float = 2.0) -> Signal | None:
    """Buy a ladder that has not repriced a settlement-index impulse yet.

    The move is measured on the CF Benchmarks prints the contract settles on, not on
    a Coinbase tick: that is the venue's own reference, so an impulse in it is
    information the ladder can be behind. The ladder's bid must still be within
    `stale_share` of where it was; a repriced book is no longer an opportunity.
    """
    if len(prints) < 3 or strike <= 0 or seconds_left < min_seconds_left or sigma <= 0:
        return None
    t_end, v_end = prints[-1]
    if v_end <= 0:
        return None
    base = None
    for ts, value in reversed(prints):
        if t_end - ts >= lookback_s:
            base = (ts, value)
            break
    if base is None or (t_end - lookback_s) - base[0] > 1.0 or base[1] <= 0:
        return None
    t0, v0 = base
    bps = (v_end / v0 - 1.0) * 10_000.0
    magnitude = abs(bps)
    if magnitude < min_bps or magnitude > max_bps:
        return None
    direction = 1.0 if bps > 0 else -1.0
    fv_now = fair_value(v_end, strike, seconds_left, sigma)
    fv_then = fair_value(v0, strike, seconds_left + (t_end - t0), sigma)
    if not (fv_now.enough_data and fv_then.enough_data):
        return None
    implied = (fv_now.p_up - fv_then.p_up) * direction
    if implied <= 0:
        return None
    # Without a ladder history there is no staleness measurement, and a blind snipe
    # is just a taker fee. Abstain rather than assume the book is behind.
    if not book_hist:
        return None
    now_pt = None
    then_pt = None
    for ts, bid in reversed(book_hist):
        if now_pt is None and ts <= t_end:
            now_pt = (ts, bid)
        if ts <= t0:
            then_pt = (ts, bid)
            break
    if now_pt is None or then_pt is None or t_end - now_pt[0] > max_book_age_s:
        return None
    moved = float(now_pt[1] - then_pt[1]) * direction
    if moved > stale_share * implied:
        return None
    return Signal("impulse_snipe", fv_now.p_up,
                  f"{bps:+.1f}bp in {t_end - t0:.1f}s, implied {implied * 100:+.1f}c, "
                  f"ladder moved {moved * 100:+.1f}c",
                  regime="impulse", impulse_bps=bps)


def cross_book(*, yes_ask: Decimal | None, no_ask: Decimal | None,
               min_profit: Decimal = Decimal("0.01")) -> Signal | None:
    """YES + NO under a dollar is a riskless round trip if both sides can fill."""
    if yes_ask is None or no_ask is None:
        return None
    total = yes_ask + no_ask
    profit = Decimal(1) - total
    if profit < min_profit:
        return None
    return Signal("cross_book", 0.5, f"YES {yes_ask:.2f} + NO {no_ask:.2f} = {total:.2f}",
                  regime="arb", arb_profit=profit)


def maker_capture(*, yes_bid: Decimal | None, yes_ask: Decimal | None, p_up: float,
                  edge_hurdle: float) -> Signal | None:
    """Rest inside a wide book and sell the spread, paying no fee on either leg."""
    if yes_bid is None or yes_ask is None:
        return None
    width = yes_ask - yes_bid
    if width < Decimal("0.02"):
        return None
    mid = (yes_bid + yes_ask) / 2
    # Only quote when the model agrees with the mid and the spread pays the hurdle.
    if abs(Decimal(str(p_up)) - mid) > Decimal("0.04"):
        return None
    if width / 2 < Decimal(str(edge_hurdle)):
        return None
    return Signal("maker_capture", p_up,
                  f"{width * 100:.0f}c book, model {p_up:.2f} at mid {mid:.2f}",
                  regime="spread_capture")


# ---------------------------------------------------------------------------
# Printed flow + touch revisions (the promoted program)
#
# The recording keeps top-of-book only, so the ladder term is the balance of quote
# *revisions*: a repricing book shows one-sided bid upticks and ask downticks before
# it shows a new level. `z_obi_flow` fires when the normalised strike z and both of
# those agree, and it is the only program with a frozen out-of-sample verdict.
# ---------------------------------------------------------------------------


def print_flow(prints: list[tuple[float, float, int]], now: float,
               window_s: float = 10.0) -> float:
    """Net aggressive print flow in [-1, 1] over the trailing `window_s`.

    `prints` is (ts, contracts, side) with side=1 when the taker bought YES. This is
    the same definition the alpha study scored: signed printed size over total
    printed size.
    """
    lo = now - window_s
    num = den = 0.0
    for ts, size, side in reversed(prints):
        if ts < lo:
            break
        if ts > now or size <= 0:
            continue
        num += size if int(side) == 1 else -size
        den += size
    return num / den if den > 0 else 0.0


def touch_revisions(touch: list[tuple[float, Decimal, Decimal]], now: float,
                    window_s: float = 5.0) -> float:
    """Bid upticks minus ask downticks over the trailing `window_s`, in [-1, 1]."""
    lo = now - window_s
    up = dn = 0
    prev: tuple[Decimal, Decimal] | None = None
    for ts, bid, ask in touch:
        if ts < lo:
            prev = (bid, ask)
            continue
        if ts > now:
            break
        if prev is not None:
            if bid > prev[0]:
                up += 1
            if ask < prev[1]:
                dn += 1
        prev = (bid, ask)
    return (up - dn) / max(up + dn, 1)


def z_obi_flow(*, spot: float, strike: float, seconds_left: float, sigma: float,
               flow: float, revision: float, min_z: float = 0.15,
               min_flow: float = 0.25, min_revision: float = 0.20,
               min_delta: float = 0.0, trend_fast: float = 0.0,
               trend_slow: float = 0.0) -> Signal | None:
    """Strike z, printed-flow imbalance and touch revisions all lean one way.

    Three gates sit in front of the flow read, and each one exists because of a live
    loss rather than a backtest preference:

    * `min_delta` is the underlying spot-to-strike distance. Inside it the contract is
      a coin flip with a flow signal on top, and a half-window flow edge is not enough
      to hold a binary through settlement -- KXBTC at 0.58 with the index a few dollars
      from the strike settled `no` for -$49.88.
    * `trend_fast`/`trend_slow` are log returns on two horizons. A z against a strike
      that the tape is already unwinding is a stale print, not a dislocation: both
      horizons must point the way the z does.
    * The flow and revision floors below are the promoted program's own.
    """
    if strike <= 0 or spot <= 0 or sigma <= 0 or seconds_left <= 0:
        return None
    vol = sigma * math.sqrt(seconds_left)
    if vol <= 0:
        return None
    z = math.log(spot / strike) / vol
    direction = 1.0 if z > 0 else -1.0
    if abs(z) < min_z:
        return None
    if min_delta > 0 and abs(spot - strike) < min_delta:
        return None
    if trend_fast and trend_slow:
        if trend_fast * direction <= 0 or trend_slow * direction <= 0:
            return None
    if flow * direction < min_flow or revision * direction < min_revision:
        return None
    fv = fair_value(spot, strike, seconds_left, sigma)
    if not fv.enough_data:
        return None
    return Signal("z_obi_flow", fv.p_up,
                  f"z {z:+.2f}, flow {flow:+.2f}, rev {revision:+.2f} agree, "
                  f"delta {abs(spot - strike):.4g}, trend {trend_fast:+.2e}/{trend_slow:+.2e}",
                  regime="flow")


# ---------------------------------------------------------------------------
# F2: short-horizon taker momentum, priced by its own payoff event
#
# The F2 family is the first rule that cleared every economic gate on a frozen
# out-of-sample slice, and the gate that stopped it was telling the truth: its
# stated probability described the market's settlement (ECE 0.2543) while the
# thing it actually trades is a +5c take-profit round trip (payoff ECE 0.0764).
# A momentum rule's probability should be the chance its own trade finishes
# green, so the frozen rule carries a payoff calibrator fitted on the in-sample
# trades only and frozen before the out-of-sample slice was scored.
#
# The calibrator is a ridge logistic on the four features the rule already
# reads, signed by the trade's direction, fit by scripts/f2_structural.py on 46
# in-sample trades to the target `1 if net > 0 else 0`. The coefficients are
# copied here rather than recomputed at runtime: refitting against a slice that
# has been read is how a calibration becomes a backtest.
# ---------------------------------------------------------------------------

F2_Z_MIN = 0.30
F2_FLOW_MIN = 0.10
F2_REV_MIN = 0.0
F2_DELTA_BPS_MIN = 3.0

# Frozen in-sample fit (46 IS trades, `scripts/f2_structural.py`, target
# `P(trade net > 0)`). Feature order: [z, flow*d, rev*d, delta_bps].
F2_PAYOFF_WEIGHTS: tuple[float, ...] = (
    -1.556138392458678,
    -0.09587882607865836,
    -0.3481172288662512,
    -0.006108663828638203,
    3.0964590345928897,
)
F2_PAYOFF_NORMALISER: tuple[float, ...] = (
    -0.1862172909773686, 0.2842503527894131, 0.1019263136452415, 5.729300712016454,
    0.3241936308135562, 0.16672569376883345, 0.1705345529095152, 1.8158927899888653,
)


@dataclass(frozen=True)
class PayoffCalibrator:
    """A frozen logistic map from F2's own features to `P(trade net > 0)`.

    Deliberately not a `Calibrator`: that one maps a settlement probability to a
    settlement probability (monotone in logit space, two parameters). This one
    is a different target on a different feature space, and collapsing the two
    into one type is how the settlement/payoff confusion started.
    """

    weights: tuple[float, ...] = F2_PAYOFF_WEIGHTS
    normaliser: tuple[float, ...] = F2_PAYOFF_NORMALISER
    samples: int = 46
    target: str = "P(trade net > 0)"

    @property
    def fitted(self) -> bool:
        return bool(self.weights) and len(self.weights) == len(self.normaliser) // 2 + 1

    def apply(self, row: list[float] | tuple[float, ...]) -> float:
        n = len(self.normaliser) // 2
        if not self.fitted or len(row) != n:
            return 0.5
        mu, sd = self.normaliser[:n], self.normaliser[n:]
        z = self.weights[-1]
        for i in range(n):
            z += self.weights[i] * ((float(row[i]) - mu[i]) / (sd[i] or 1.0))
        return _sigmoid(z)

    def to_json(self) -> dict[str, Any]:
        return {"weights": list(self.weights), "normaliser": list(self.normaliser),
                "samples": self.samples, "target": self.target, "fitted": self.fitted}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "PayoffCalibrator":
        weights = tuple(float(v) for v in (data.get("weights") or []))
        norm = tuple(float(v) for v in (data.get("normaliser") or []))
        return cls(weights=weights, normaliser=norm,
                   samples=int(data.get("samples") or data.get("rows") or 0),
                   target=str(data.get("target") or "P(trade net > 0)"))


def load_payoff_calibrator(path: Any, variant: str = "F2_taker_1s") -> PayoffCalibrator:
    """The frozen payoff calibrator from `data/f2-structural.json`, or the constants.

    Reading the artifact lets an operator audit which coefficients were fitted
    without trusting a code comment; falling back to the constants keeps a
    replay deterministic when the artifact is missing or has been rewritten.
    """
    from pathlib import Path as _Path

    target = _Path(path)
    try:
        raw = json.loads(target.read_text())
        cal = ((raw.get("variants") or {}).get(variant) or {}).get("calibration") or {}
        if cal.get("fitted"):
            return PayoffCalibrator.from_json(cal)
    except (OSError, ValueError, AttributeError):
        pass
    return PayoffCalibrator()


def f2_features(*, z: float, flow: float, revision: float, delta_bps: float) -> list[float]:
    """The rule's own four features, signed by the direction z points."""
    d = 1.0 if z > 0 else -1.0
    return [float(z), float(flow) * d, float(revision) * d, float(delta_bps)]


def f2_taker_1s(*, spot: float, strike: float, seconds_left: float, sigma: float,
                flow: float, revision: float,
                calibrator: PayoffCalibrator | None = None,
                min_z: float = F2_Z_MIN, min_flow: float = F2_FLOW_MIN,
                min_revision: float = F2_REV_MIN,
                min_delta_bps: float = F2_DELTA_BPS_MIN) -> Signal | None:
    """The F2 rule: strike z, printed flow and touch revisions agree, trade it.

    `p_up` is NOT the settlement probability. It is the calibrated chance that
    this trade's own round trip finishes positive (`net > 0`), which is the
    quantity the entry hurdle in `strategy.decide` is written against for this
    program. Every threshold is the frozen one; nothing here is tuned on the
    slice it is scored on.
    """
    if strike <= 0 or spot <= 0 or sigma <= 0 or seconds_left <= 0:
        return None
    vol = sigma * math.sqrt(seconds_left)
    if vol <= 0:
        return None
    z = math.log(spot / strike) / vol
    direction = 1.0 if z > 0 else -1.0
    if abs(z) < min_z:
        return None
    delta_bps = abs(spot - strike) / max(spot, 1e-9) * 10_000.0
    if delta_bps < min_delta_bps:
        return None
    if flow * direction < min_flow or revision * direction < min_revision:
        return None
    p_payoff = (calibrator or PayoffCalibrator()).apply(
        f2_features(z=z, flow=flow, revision=revision, delta_bps=delta_bps))
    return Signal("F2_taker_1s", min(1.0, max(0.0, p_payoff)),
                  f"z {z:+.2f}, flow {flow:+.2f}, rev {revision:+.2f} agree, "
                  f"{delta_bps:.1f}bp from strike, payoff p {p_payoff:.3f}",
                  regime="momentum")


# ---------------------------------------------------------------------------
# student_t_l2_settle: Student-t settlement value + true-L2 imbalance
#
# A 15-minute crypto binary settles on a 60-second average of the index, so the
# quantity to price is the average, not the last print. Two things follow. The
# variance of the time left to average is smaller than the variance of the whole
# window -- hence `tau_eff = seconds_left - 40` -- and the index's own
# standardised returns are fat-tailed, so the settlement value is read off a
# Student-t rather than a normal.
#
# The per-lane degrees of freedom and scale are the frozen Track 2B in-sample
# fit, copied here rather than refitted at runtime: refitting against a slice
# that has been read is how a calibration becomes a backtest. The fit is of the
# *index*'s standardised returns, so the sigma fed in must be the index's own.
# ---------------------------------------------------------------------------

# (nu, s) per lane: the frozen Student-t fit of the index's standardised returns.
STUDENT_T_FIT: dict[str, tuple[float, float]] = {
    "KXBTC15M": (8.0, 1.13),
    "KXETH15M": (6.0, 0.78),
    "KXSOL15M": (10.0, 0.68),
    "KXXRP15M": (10.0, 1.06),
}
# The settlement reference is a 60s average, so the variance of the time that is
# left to average is the window minus its own fixings.
STUDENT_T_TAU_OFFSET_S = 40.0


def _betacf(a: float, b: float, x: float) -> float:
    """The continued fraction of the incomplete beta function (Lentz's method)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3.0e-16:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """`I_x(a, b)`: the regularized incomplete beta function."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    back = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                    + b * math.log1p(-x) + a * math.log(x))
    return 1.0 - back * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(x: float, nu: float) -> float:
    """`P(T <= x)` for a standard Student-t with `nu` degrees of freedom.

    Exact, not a table: the live path prices a binary on this number, and a
    lookup table with a step would put a bias between the live fair value and the
    scored one.
    """
    if nu <= 0:
        raise ValueError("nu must be positive")
    if x == 0.0:
        return 0.5
    tail = regularized_incomplete_beta(nu / 2.0, 0.5, nu / (nu + x * x))
    return 1.0 - 0.5 * tail if x > 0 else 0.5 * tail


def student_t_settlement_p_up(*, spot: float, strike: float, seconds_left: float,
                              sigma: float, nu: float, s: float) -> float:
    """`P(the contract settles YES)` under the lane's Student-t settlement model.

    `x = log(strike / spot) / (sigma * sqrt(tau_eff))` is the standardised distance
    to the strike over the time that is left to average, and the settlement
    probability is `1 - T_nu(x / s)`.
    """
    if strike <= 0 or spot <= 0 or sigma <= 0:
        return 0.5
    tau_eff = max(seconds_left - STUDENT_T_TAU_OFFSET_S, 1.0)
    vol = sigma * math.sqrt(tau_eff)
    if vol <= 0:
        return 0.5
    x = math.log(strike / spot) / vol
    return min(1.0, max(0.0, 1.0 - student_t_cdf(x / s, nu)))


def l2_imbalance(yes_bid_sz: float | None, yes_ask_sz: float | None) -> float | None:
    """True L2 order-book imbalance in [-1, 1].

    `(yes_bid_sz - yes_ask_sz) / (yes_bid_sz + yes_ask_sz)`: the resting size on
    the bid against the resting size on the ask, at the touch. `None` when either
    side's size is unknown or empty -- an unknown depth is not a neutral book, it
    is no book, and a filter that reads it as neutral would fire on a ladder the
    bot cannot actually see.
    """
    if yes_bid_sz is None or yes_ask_sz is None:
        return None
    bid, ask = float(yes_bid_sz), float(yes_ask_sz)
    if bid <= 0 or ask <= 0:
        return None
    return (bid - ask) / (bid + ask)


def student_t_l2_settle(*, spot: float, strike: float, seconds_left: float, sigma: float,
                         nu: float, s: float, l2_obi: float | None,
                         flow_fast: float, flow_slow: float,
                         min_seconds_left: float = 60.0,
                         max_seconds_left: float = 780.0) -> Signal | None:
    """The settlement thesis: a Student-t fair value, a true-L2 imbalance, signed flow.

    This prices the contract and reports the book; it does not choose the side.
    Which side is even tradeable inside the cost band depends on the ladder, so
    that is the strategy's business (`strategy._student_t_l2_entry`).

    `None` means "no thesis": the window is closed, the index is unusable, or the
    tape carries no resting size to read an imbalance from.
    """
    if strike <= 0 or spot <= 0 or sigma <= 0:
        return None
    if not (min_seconds_left <= seconds_left <= max_seconds_left):
        return None
    if l2_obi is None:
        return None
    p_up = student_t_settlement_p_up(spot=spot, strike=strike, seconds_left=seconds_left,
                                     sigma=sigma, nu=nu, s=s)
    tau_eff = max(seconds_left - STUDENT_T_TAU_OFFSET_S, 1.0)
    x = math.log(strike / spot) / (sigma * math.sqrt(tau_eff))
    return Signal("student_t_l2_settle", p_up,
                  f"t{nu:.0f} x {x:+.2f} p {p_up:.3f}, obi {l2_obi:+.2f}, "
                  f"flow {flow_fast:+.2f}/{flow_slow:+.2f}",
                  regime="settlement", l2_obi=l2_obi,
                  flow_fast=flow_fast, flow_slow=flow_slow)
