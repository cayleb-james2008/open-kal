"""The judge: walk-forward epochs with purging, clustered and multiple-testing-corrected bounds, deflated Sharpe, PBO.

Everything here is registered in prereg_judge.json (v2). Nothing in this module reads a candidate's out-of-sample result to choose
anything: it only measures.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import stats as st

ALPHA = 0.05
MIN_OOS_TRADES, MAX_DD, CAPITAL = 200, 0.20, 164.0
MIN_TRAIN_BLOCKS, TEST_BLOCKS, STEP_BLOCKS = 4, 2, 1
EMBARGO_S = 900.0                       # one window
Z_UNCORRECTED = float(st.norm.ppf(1 - ALPHA))
EULER = 0.5772156649015329


def block_ids(wclose: np.ndarray, block_hours: float = 6.0) -> np.ndarray:
    return np.floor((wclose - wclose.min()) / (block_hours * 3600.0)).astype(np.int64)


def epochs(n_blocks: int) -> list[tuple[list[int], list[int]]]:
    """(train blocks, test blocks) in order: train on every block before the test blocks, test on the next two, step one."""
    out = []
    for start in range(MIN_TRAIN_BLOCKS, n_blocks - TEST_BLOCKS + 1, STEP_BLOCKS):
        out.append((list(range(start)), list(range(start, start + TEST_BLOCKS))))
    return out


def labels(wclose: np.ndarray, wblock: np.ndarray, train: list[int], test: list[int]) -> np.ndarray:
    """1 train, 2 test, 0 unused; every window within one window of a test block's span is purged from train."""
    lab = np.zeros(len(wclose), np.int64)
    lab[np.isin(wblock, train)] = 1
    lab[np.isin(wblock, test)] = 2
    t0, t1 = wclose[lab == 2].min() - EMBARGO_S, wclose[lab == 2].max() + EMBARGO_S
    lab[(lab == 1) & (wclose >= t0) & (wclose <= t1)] = 0
    return lab


def corrected_z(n_candidates: int) -> float:
    """Bonferroni: the standard normal quantile of 1 - alpha / N, never below the uncorrected 1.645."""
    return max(Z_UNCORRECTED, float(st.norm.isf(ALPHA / max(1, n_candidates))))


def bounds(mean: np.ndarray, se: np.ndarray, n_candidates: int) -> tuple[np.ndarray, np.ndarray]:
    return mean - Z_UNCORRECTED * se, mean - corrected_z(n_candidates) * se


def max_drawdown(pnl_in_time_order: np.ndarray, capital: float = CAPITAL) -> float:
    """Largest peak-to-trough fall of the cumulative PnL path, as a fraction of capital; the path starts at zero."""
    if len(pnl_in_time_order) == 0:
        return 0.0
    path = np.r_[0.0, np.cumsum(pnl_in_time_order)]
    return float((np.maximum.accumulate(path) - path).max() / capital)


def expected_max_sharpe(n_trials: int, var_sharpe: float) -> float:
    """E[max SR] over n independent trials with null mean zero (Bailey and Lopez de Prado)."""
    if n_trials <= 1:
        return 0.0
    return math.sqrt(var_sharpe) * ((1 - EULER) * st.norm.ppf(1 - 1.0 / n_trials) + EULER * st.norm.ppf(1 - 1.0 / (n_trials * math.e)))


def deflated_sharpe(window_pnl: np.ndarray, n_trials: int, var_sharpe: float) -> float:
    """Probability that the true Sharpe exceeds the best of n_trials null strategies, from per-window PnL (zeros included)."""
    x = np.asarray(window_pnl, float)
    T = len(x)
    if T < 3 or x.std(ddof=1) == 0:
        return 0.0
    sr = x.mean() / x.std(ddof=1)
    skew, kurt = float(st.skew(x)), float(st.kurtosis(x, fisher=False))
    sr0 = expected_max_sharpe(n_trials, var_sharpe)
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4.0 * sr ** 2))
    return float(st.norm.cdf((sr - sr0) * math.sqrt(T - 1) / denom))


def pbo(block_pnl: np.ndarray, max_splits: int = 3000, seed: int = 0) -> dict:
    """Probability of backtest overfitting by CSCV on a (blocks, candidates) PnL matrix.

    Half the blocks pick the best candidate by mean PnL, the other half rank it; PBO is the share of splits where the in-sample
    winner lands at or below the out-of-sample median. Splits are all combinations (or a random sample past `max_splits`).
    """
    n_blocks, n_cand = block_pnl.shape
    s = min(16, n_blocks - n_blocks % 2)
    if s < 4 or n_cand < 2:
        return {"pbo": None, "splits": 0, "blocks": n_blocks}
    M = block_pnl[:s]
    combos = list(itertools.combinations(range(s), s // 2))
    if len(combos) > max_splits:
        rng = np.random.default_rng(seed)
        combos = [combos[i] for i in rng.choice(len(combos), max_splits, replace=False)]
    below = 0
    for c in combos:
        ins = np.zeros(s, bool)
        ins[list(c)] = True
        is_mean, oos_mean = M[ins].mean(axis=0), M[~ins].mean(axis=0)
        best = int(np.argmax(is_mean))
        rank = (oos_mean < oos_mean[best]).sum() + 0.5 * ((oos_mean == oos_mean[best]).sum() - 1)
        w = (rank + 1) / (n_cand + 1)
        below += math.log(w / (1 - w)) <= 0
    return {"pbo": below / len(combos), "splits": len(combos), "blocks": s}


def verdict(*, oos_n: float, oos_mean: float, bound_unc: float, bound_corr: float, dd: float, stress_mean: float, dsr: float,
            pbo_epoch: float | None) -> tuple[str, str]:
    """(status, reason): every registered condition; the first failing one is the reason."""
    checks = [("trades", oos_n >= MIN_OOS_TRADES, f"{oos_n:.0f} OOS trades < {MIN_OOS_TRADES}"),
              ("ev", oos_mean > 0 and bound_unc > 0, "OOS EV or its 95% lower bound <= 0"),
              ("dd", dd < MAX_DD, f"drawdown {dd:.1%} >= {MAX_DD:.0%}"),
              ("stress", stress_mean > 0, "EV <= 0 under 2x latency and 1 cent slippage"),
              ("corrected", bound_corr > 0, "multiple-testing-corrected lower bound <= 0"),
              ("dsr", dsr >= 0.95, f"deflated Sharpe {dsr:.2f} < 0.95"),
              ("pbo", pbo_epoch is not None and pbo_epoch <= 0.5, f"PBO {pbo_epoch} > 0.5 or not measurable")]
    for _, ok, reason in checks:
        if not ok:
            return "rejected", reason
    return "passed_screen", "passed every registered screen condition; awaits the tier-2 replay and the fresh-range re-test"
