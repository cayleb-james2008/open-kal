"""Tier 1: screen a whole population in one parallel numba call.

For every genome, one pass over every window gives, per window label (1 train, 2 test), the sufficient statistics of the per-trade
PnL with the window-clustered variance, plus a per-block PnL row for the overfitting tests. Trade lists, drawdown and the deflated
Sharpe ratio are rebuilt for the few candidates that survive (`sim.simulate`).
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from factory import sim

# acc[g, label, :]: 0 n trades, 1 sum pnl, 2 wins, 3 gross win, 4 gross loss (positive), 5 sum S_w^2, 6 sum S_w*n_w, 7 sum n_w^2, 8 windows traded
N_ACC = 9


@njit(parallel=True)   # no cache: it inlines sim.run from another file, which numba does not track
def screen_batch(G, X, secs_idx, bid, ask, bsz, asz, valid, wstart, wend, wy, wdepth, wlabel, wblock, n_blocks, lat, slip, capital):
    P = G.shape[0]
    nw = len(wstart)
    acc = np.zeros((P, 3, N_ACC))
    blk = np.zeros((P, n_blocks))
    maxn = nw * 3
    for p in prange(P):
        pnl = np.empty(maxn)
        win = np.empty(maxn, np.int64)
        row = np.empty(maxn, np.int64)
        qty = np.empty(maxn, np.int64)
        k = sim.run(G[p], X, secs_idx, bid, ask, bsz, asz, valid, wstart, wend, wy, wdepth, lat, slip, capital, pnl, win, row, qty)
        # trades come grouped by window; accumulate per window
        t = 0
        while t < k:
            w = win[t]
            s_w, n_w = 0.0, 0
            lab = wlabel[w]
            while t < k and win[t] == w:
                x = pnl[t]
                s_w += x
                n_w += 1
                if lab > 0:
                    a = acc[p, lab]
                    a[0] += 1
                    a[1] += x
                    if x > 0:
                        a[2] += 1
                        a[3] += x
                    else:
                        a[4] -= x
                t += 1
            if lab > 0:
                a = acc[p, lab]
                a[5] += s_w * s_w
                a[6] += s_w * n_w
                a[7] += n_w * n_w
                a[8] += 1
            if wblock[w] >= 0:
                blk[p, wblock[w]] += s_w
    return acc, blk


def screen(book, G: np.ndarray, wlabel: np.ndarray, wblock: np.ndarray, n_blocks: int, *, lat: int = 5, slip: float = 0.0):
    """(acc, block_pnl) for a (P, GENOME_LEN) population."""
    return screen_batch(np.ascontiguousarray(G), book.X, book.secs_idx, book.bid, book.ask, book.bsz, book.asz, book.valid, book.wstart,
                        book.wend, book.wy, book.wdepth, wlabel.astype(np.int64), wblock.astype(np.int64), int(n_blocks), int(lat), float(slip),
                        float(book.capital))


def summarize(acc: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorised per-candidate stats from an (P, N_ACC) accumulator slice."""
    n = acc[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = acc[:, 1] / n
        # cluster-robust variance of the sum: sum_w (S_w - mean n_w)^2 = SS - 2 mean SSn + mean^2 Sn2
        var_sum = acc[:, 5] - 2 * mean * acc[:, 6] + mean ** 2 * acc[:, 7]
        se = np.sqrt(np.maximum(var_sum, 0.0)) / n
        pf = np.where(acc[:, 4] > 0, acc[:, 3] / acc[:, 4], np.where(acc[:, 3] > 0, np.inf, 0.0))
        win_rate = acc[:, 2] / n
    return {"n": n, "mean": mean, "se": se, "sum": acc[:, 1], "win_rate": win_rate, "pf": pf, "windows": acc[:, 8]}
