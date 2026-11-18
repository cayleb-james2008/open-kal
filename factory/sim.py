"""One numba pass per candidate over the precomputed one-second feature matrix.

Taker-only in version 1 (see prereg_judge.json): entry at the ask of the side held, IOC at the seen touch size, the venue's fee
formula, exits by take-profit, confirmed stop, time barrier or settlement. A genome is a flat float64 vector (layout below) so a
candidate is one array and one call. Rows of a lane-window are contiguous, time ascending (seconds left descending), one per second.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numba import njit

N_CLAUSES, N_CONDS = 3, 3
# genome layout
I_CF, I_CD, I_CT = 0, 9, 18              # clause feature idx (-1 unused), direction (+1 x >= thr, -1 x <= thr), threshold
I_SIDE_F, I_SIDE_MODE, I_SIDE_DEAD = 27, 28, 29
I_SEC_LO, I_SEC_HI, I_STAKE, I_MAX_ENT, I_COOL, I_TP, I_STOP, I_HOLD = 30, 31, 32, 33, 34, 35, 36, 37
GENOME_LEN = 38
CAP, MAX_QTY, STOP_CONFIRM = 0.62, 50, 2


@njit(cache=True)
def fee(price: float, qty: int) -> float:
    """ceil(0.07 * qty * P * (1 - P)) to the cent, as openkal.fees.taker_fee; price on the 0.001 tick."""
    p = np.rint(price * 1000.0) / 1000.0
    raw = 0.07 * qty * p * (1.0 - p) * 100.0
    return math.ceil(raw - 1e-7) / 100.0


@njit(cache=True)
def _signal(g, X, row):
    any_clause = False
    for c in range(N_CLAUSES):
        used, ok = False, True
        for k in range(N_CONDS):
            f = int(g[I_CF + c * N_CONDS + k])
            if f < 0:
                continue
            used = True
            x = X[f, row]
            if not (x == x):
                ok = False
                break
            if g[I_CD + c * N_CONDS + k] > 0:
                if not x >= g[I_CT + c * N_CONDS + k]:
                    ok = False
                    break
            else:
                if not x <= g[I_CT + c * N_CONDS + k]:
                    ok = False
                    break
        if used and ok:
            any_clause = True
            break
    return any_clause


@njit(cache=True)
def run(g, X, secs_idx, bid, ask, bsz, asz, valid, wstart, wend, wy, wdepth, lat, slip, capital, out_pnl, out_win, out_row, out_qty):
    """Simulate one genome over every window; returns the number of trades written to the out arrays."""
    n = 0
    side_f = int(g[I_SIDE_F])
    mode = g[I_SIDE_MODE]
    dead = g[I_SIDE_DEAD]
    sec_lo, sec_hi = g[I_SEC_LO], g[I_SEC_HI]
    max_ent = int(g[I_MAX_ENT])
    cool = int(g[I_COOL])
    tp, stop, hold = g[I_TP], g[I_STOP], int(g[I_HOLD])
    stake = g[I_STAKE] * capital
    for w in range(len(wstart)):
        s, e = wstart[w], wend[w]
        i, entries = s, 0
        while i < e and entries < max_ent:
            secs = X[secs_idx, i]
            if secs > sec_hi or secs < sec_lo or not _signal(g, X, i):
                i += 1
                continue
            sv = X[side_f, i]
            if not (sv == sv) or abs(sv) <= dead:
                i += 1
                continue
            yes = (sv > 0) == (mode > 0)
            arr = i + lat
            if arr >= e or not valid[arr]:
                i += 1
                continue
            cost = ask[arr] if yes else 1.0 - bid[arr]
            cost = np.rint(cost * 1000.0) / 1000.0
            paid = cost + slip
            if cost <= 0.0 or paid > CAP:
                i += 1
                continue
            depth = asz[arr] if yes else bsz[arr]
            if not (depth == depth):
                depth = wdepth[w]
            qty = min(MAX_QTY, int(stake / paid), int(depth))
            if qty < 1:
                i += 1
                continue
            fee_in = fee(paid, qty)
            # exit scan: decide at row j, the order arrives lat rows later
            exit_row, exit_val, bad = -1, 0.0, 0
            for j in range(arr, e):
                if not valid[j]:
                    continue
                mark = bid[j] if yes else 1.0 - ask[j]
                hit = False
                if tp > 0.0 and mark >= cost + tp:
                    hit = True
                if stop > 0.0:
                    bad = bad + 1 if mark <= cost - stop else 0
                    if bad >= STOP_CONFIRM:
                        hit = True
                if hold > 0 and j - arr >= hold:
                    hit = True
                if hit:
                    xr = j + lat
                    if xr < e and valid[xr]:
                        exit_row = xr
                        exit_val = (bid[xr] if yes else 1.0 - ask[xr]) - slip
                    break
            if exit_row >= 0:
                pnl = qty * (exit_val - paid) - fee_in - fee(max(exit_val, 0.0), qty)
                next_i = exit_row + cool
            else:
                won = (wy[w] == 1) == yes
                pnl = qty * ((1.0 if won else 0.0) - paid) - fee_in
                next_i = e
            out_pnl[n], out_win[n], out_row[n], out_qty[n] = pnl, w, i, qty
            n += 1
            entries += 1
            i = max(next_i, i + 1)
    return n


@dataclass
class Book:
    """The matrix the simulator runs on: feature-major float32, per-row quotes, per-window bounds."""
    X: np.ndarray                      # (n_feat, n_rows) float32
    names: list[str]
    bid: np.ndarray
    ask: np.ndarray
    bsz: np.ndarray
    asz: np.ndarray
    valid: np.ndarray
    wstart: np.ndarray
    wend: np.ndarray
    wy: np.ndarray
    wclose: np.ndarray
    wlane: np.ndarray
    wdepth: np.ndarray
    capital: float = 164.0
    out: dict = field(default_factory=dict)

    @property
    def secs_idx(self) -> int:
        return self.names.index("secs")

    @property
    def max_trades(self) -> int:
        return int(len(self.wstart) * 3)


def simulate(book: Book, genome: np.ndarray, *, lat: int = 5, slip: float = 0.0):
    """Trades of one genome: (pnl, window index, entry row, qty) arrays, in window order."""
    n = book.max_trades
    pnl, win, row, qty = np.empty(n), np.empty(n, np.int64), np.empty(n, np.int64), np.empty(n, np.int64)
    k = run(genome, book.X, book.secs_idx, book.bid, book.ask, book.bsz, book.asz, book.valid, book.wstart, book.wend, book.wy,
            book.wdepth, int(lat), float(slip), float(book.capital), pnl, win, row, qty)
    return pnl[:k], win[:k], row[:k], qty[:k]


def new_genome() -> np.ndarray:
    g = np.zeros(GENOME_LEN)
    g[I_CF:I_CF + 9] = -1
    g[I_CD:I_CD + 9] = 1
    g[I_SIDE_MODE] = 1
    g[I_SEC_LO], g[I_SEC_HI], g[I_STAKE], g[I_MAX_ENT] = 60, 840, 0.2, 1
    return g
