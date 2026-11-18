"""Exact settlement probability for the 15-minute crypto binaries' final-120s pin.

`KXBTC15M`/`KXETH15M`/`KXSOL15M`/`KXXRP15M` settle YES when the simple average of
CF Benchmarks' RTI over the last 60 seconds of the window is at or above the published
floor strike. Kalshi locks one fixing per second and streams the locked part as it
fills: measured on the frozen tape (`scripts/verify_settlement.py`, and rows re-read
around real closes in charter 3), `k` is how many seconds of the 60 are locked and
`accum` is their running mean; before the averaging window opens the venue keeps
streaming the *open* average with `k` pinned at 60, which every guard below refuses.

So at a decision time with `secs_left` seconds to close:

  * `k` fixings are already fixed, with sum `accum * k` -- known, not random;
  * the remaining `n = 60 - k` fixings start from the current index print, so the
    settlement average has mean `(locked_sum + spot * n) / 60`;
  * with `secs_left <= 60` the remaining fixings are spread over the seconds that are
    left and the conditional variance is `sigma^2 * secs_left^3 / 10800` (the T^1.5
    law `scripts/verify_settlement.py` measures); with more than 60s left nothing is
    locked and the whole fixing window lies `secs_left - 60 .. secs_left` seconds
    ahead, giving variance `sigma^2 * (secs_left - 40)`;
  * `sigma` is the trailing 150s realised volatility of the index tape in price units
    per sqrt-second: squared one-second price moves only (a gap in the tape is never
    bridged into a return), unfloored, so a dead-flat tape prices degenerately at
    variance zero and a too-sparse one is refused.

The quote is refused, never approximated, when the locked accumulator is stale (the
elapsed fixings are unknown), runs ahead of the clock, repeats a count with two
different averages, first reports a fixing outside its clock slot, when no index
print is fresh, when too few one-second pairs feed sigma, or when the window is
outside the averaging period. Without `accum`/`k` columns the model falls back to
reading the locked fixings off the raw print tape, refusing any second that is
missing or printed with two different values. `scripts/asian_rti.py` never reads a
settlement outcome; `y` is used by the sweep only after a trade is booked.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WIN_S = 60.0            # settlement average window, seconds
SPOT_TOL_S = 1.5        # an index print older than this cannot price a quote
K_TOL_S = 1.0           # venue locked count vs. elapsed window seconds
KSEQ_TOL_S = 0.5        # when a fixing may first be reported, vs. its clock slot
SIG_WINDOW_S = 150.0    # trailing window of the realised-volatility estimate
SIG_MIN_PAIRS = 5       # fewer one-second pairs than this: sigma is unavailable
SECS_MAX = 840.0        # a 15-minute window; nothing is priced outside it


def settlement_parts(*, spot: float, locked_sum: float, locked_s: float,
                     secs_left: float, sigma: float, strike: float,
                     window_s: float = WIN_S) -> dict:
    """Conditional settlement mean, variance and P(average >= strike).

    `sigma` is price units per sqrt-second. `locked_s` is the number of fixings the
    venue has already locked, `locked_sum` their sum (accum * k).
    """
    if not (strike > 0.0 and spot > 0.0 and math.isfinite(sigma) and sigma >= 0.0):
        raise ValueError("settlement_parts needs a positive strike, spot and sigma")
    locked = min(max(float(locked_s), 0.0), window_s)
    n_rem = max(0.0, window_s - locked)
    mean = (float(locked_sum) + spot * n_rem) / window_s
    if secs_left > window_s:      # nothing locked: every fixing is still in the future
        variance = sigma * sigma * (secs_left - 40.0)
    else:
        variance = sigma * sigma * secs_left ** 3 / 10800.0
    if variance <= 0.0:
        p = 1.0 if mean >= strike else 0.0
    else:
        p = 0.5 * (1.0 + math.erf((mean - strike) / (math.sqrt(variance) * math.sqrt(2.0))))
    return {"p": p, "mean": mean, "variance": variance, "sd": math.sqrt(variance),
            "locked_s": locked, "remaining_s": n_rem}


def _sigma_prefix(ts: np.ndarray, value: np.ndarray):
    """Cumulative sums of squared one-second price moves for the trailing sigma.

    Pair j (prints j and j+1, 1e-3 <= dt <= 1s) contributes (dv)^2 / dt stamped at the
    later point. Longer gaps are not bridged: a missing print is missing information,
    not a return. A query at t only reads pairs stamped in (t - 150, t].
    """
    dt = np.diff(ts)
    ok = (dt >= 1e-3) & (dt <= 1.0 + 1e-9)
    dv = np.diff(value)
    rate = np.where(ok, dv * dv / np.maximum(dt, 1e-3), 0.0)
    C = np.concatenate(([0.0], np.cumsum(rate)))
    N = np.concatenate(([0.0], np.cumsum(ok.astype(np.float64))))
    return ts[1:], C, N


class AsianRTI:
    """Causal fair value for one lane's index tape (ts, value[, accum, k]).

    With `accum`/`k` the locked fixings come from the venue's carried aggregate;
    without them they are read off the raw print tape (one fixing per second, every
    locked second present with a single value). `sigma_of(t)` may override sigma.
    """

    def __init__(self, ts, value, accum=None, k=None, *, sigma_of=None) -> None:
        ts = np.asarray(ts, dtype=np.float64)
        value = np.asarray(value, dtype=np.float64)
        if (accum is None) != (k is None):
            raise ValueError("accum and k must be provided together")
        self.has_venue = accum is not None
        if self.has_venue:
            accum = np.asarray(accum, dtype=np.float64)
            k = np.asarray(k, dtype=np.float64)
            if not (len(ts) == len(value) == len(accum) == len(k)):
                raise ValueError("index arrays must have one row each")
        good = np.isfinite(ts) & np.isfinite(value) & (value > 0)
        self.ts, self.value = ts[good], value[good]
        self.accum = accum[good] if self.has_venue else None
        self.k = k[good] if self.has_venue else None
        if len(self.ts) > 1 and np.any(np.diff(self.ts) < 0):
            raise ValueError("index tape is not time-sorted")
        self.tv, self.C, self.N = _sigma_prefix(self.ts, self.value)
        self._sigma_of = sigma_of

    def sigma_at(self, t: float) -> float | None:
        """Trailing price-unit sigma at t, or None when too few one-second pairs."""
        if self._sigma_of is not None:
            s = float(self._sigma_of(t))
            return s if math.isfinite(s) and s >= 0.0 else None
        hi = int(np.searchsorted(self.tv, t, "right"))
        lo = int(np.searchsorted(self.tv, t - SIG_WINDOW_S, "right"))
        cnt = float(self.N[hi] - self.N[lo])
        if cnt < SIG_MIN_PAIRS:
            return None
        return math.sqrt((self.C[hi] - self.C[lo]) / cnt)

    def _last_row(self, t: float) -> int | None:
        j = int(np.searchsorted(self.ts, t, "right")) - 1
        if j < 0 or t - float(self.ts[j]) > SPOT_TOL_S:
            return None
        return j

    def _venue_locked(self, t: float, close_ts: float) -> tuple[float, int] | None:
        """(accum * k, k) of the venue's carried aggregate; None when unusable."""
        lo = int(np.searchsorted(self.ts, close_ts - WIN_S, "left"))
        hi = int(np.searchsorted(self.ts, t, "right"))
        if hi <= lo:
            return None
        acc, kk = self.accum[lo:hi], self.k[lo:hi]
        lab = np.isfinite(acc) & (kk > 0)
        if not lab.any():
            return None
        kkl, acl = kk[lab], acc[lab]
        tsl = self.ts[lo:hi][lab]
        # The venue restarts its accumulator each window: rows before a count decrease
        # belong to the previous window (the open average it streams with k pinned 60).
        dec = np.flatnonzero(np.diff(kkl) < 0.0)
        start = int(dec[-1]) + 1 if len(dec) else 0
        kkl, acl, tsl = kkl[start:], acl[start:], tsl[start:]
        if np.any(np.diff(kkl) < 0.0):
            return None                       # a second reset inside one window
        starts = np.flatnonzero(np.r_[True, np.diff(kkl) != 0.0])
        u_k, first_ts = kkl[starts], tsl[starts]
        if np.any(np.minimum.reduceat(acl, starts)
                  != np.maximum.reduceat(acl, starts)):
            return None                       # two averages for one fixing count
        if np.any(np.abs(first_ts - (close_ts - WIN_S + u_k)) > KSEQ_TOL_S):
            return None                       # a fixing reported outside its clock slot
        k_last = int(kkl[-1])
        elapsed = WIN_S - (close_ts - t)
        if k_last < elapsed - K_TOL_S or k_last > elapsed + K_TOL_S:
            return None
        return float(acl[-1]) * k_last, k_last

    def _raw_locked(self, t: float, close_ts: float) -> tuple[float, int] | None:
        """(sum, count) of locked per-second fixings read off the raw print tape."""
        last = min(math.floor(t), close_ts)
        if last <= close_ts - WIN_S:
            return 0.0, 0
        lo = int(np.searchsorted(self.ts, close_ts - WIN_S - 0.5, "left"))
        hi = int(np.searchsorted(self.ts, t, "right"))
        sec = np.rint(self.ts[lo:hi])
        vals = self.value[lo:hi]
        keep = (sec > close_ts - WIN_S) & (sec <= last)
        sec, vals = sec[keep], vals[keep]
        req = np.arange(close_ts - WIN_S + 1.0, last + 1.0)
        if not len(sec):
            return None
        starts = np.flatnonzero(np.r_[True, np.diff(sec) != 0.0])
        u_sec, first_vals = sec[starts], vals[starts]
        if len(u_sec) != len(req) or np.any(u_sec != req):
            return None                       # a locked second is missing entirely
        ends = np.r_[starts[1:], len(sec)]
        mins = np.minimum.reduceat(vals, starts)
        maxs = np.maximum.reduceat(vals, starts)
        if np.any(mins != maxs):
            return None                       # a second printed with two values
        return math.fsum(first_vals), len(req)

    def fair(self, t: float, close_ts: float, strike: float) -> dict | None:
        """Fair value at `t` for a window closing at `close_ts`; None when unpriced."""
        if not (math.isfinite(t) and math.isfinite(close_ts) and strike > 0.0):
            return None
        secs_left = float(close_ts) - float(t)
        if secs_left < 0.0 or secs_left > SECS_MAX:
            return None
        j = self._last_row(t)
        if j is None:
            return None
        spot = float(self.value[j])
        sigma = self.sigma_at(t)
        if sigma is None:
            return None
        if secs_left >= WIN_S:
            locked_sum, locked_s = 0.0, 0
        else:
            got = (self._venue_locked(t, close_ts) if self.has_venue
                   else self._raw_locked(t, close_ts))
            if got is None:
                return None
            locked_sum, locked_s = got
        d = settlement_parts(spot=spot, locked_sum=locked_sum, locked_s=locked_s,
                             secs_left=secs_left, sigma=sigma, strike=float(strike))
        d.update({"spot": spot, "sigma": sigma, "secs_left": secs_left,
                  "elapsed": locked_s, "locked_sum": locked_sum,
                  "last_rti_ts": float(self.ts[j])})
        return d
