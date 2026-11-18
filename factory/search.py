"""Pure-code strategy search: random seeds by family, a numpy genetic algorithm, Optuna for thresholds. No LLM anywhere.

The search only ever sees a train-only `Book` (windows before the epoch's test blocks, minus the purge): out-of-sample windows are
not in the arrays it runs on, which is the leak guard of the registration.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np

from factory import indicators as IND
from factory import screen, sim

FAMILIES = ("momentum", "fade", "late_window_convergence", "cross_coin_lead_lag", "random_tree")
SIDE_FEATURES = ["flow10", "flow30", "mom10", "mom30", "rev5", "shift5", "delta_bps", "z", "l2imb"]
COND_EXCLUDE = {"secs", "lane"}
MIN_TRAIN_TRADES = 30


def subset(book, mask: np.ndarray):
    """A Book over only the windows in `mask` (rows stay shared); the search never touches the others."""
    return replace(book, wstart=book.wstart[mask], wend=book.wend[mask], wy=book.wy[mask], wclose=book.wclose[mask],
                   wlane=book.wlane[mask], wdepth=book.wdepth[mask])


def quantile_table(book, n: int = 99, sample: int = 400_000, seed: int = 0) -> np.ndarray:
    """(n_feat, n) quantiles of every feature over the rows of the book's own windows."""
    rows = np.concatenate([np.arange(s, e) for s, e in zip(book.wstart, book.wend)])
    rng = np.random.default_rng(seed)
    if len(rows) > sample:
        rows = rng.choice(rows, sample, replace=False)
    out = np.zeros((len(book.names), n))
    for f in range(len(book.names)):
        v = book.X[f, rows]
        v = v[np.isfinite(v)]
        out[f] = np.quantile(v, np.linspace(0.02, 0.98, n)) if len(v) else 0.0
    return out


class Space:
    def __init__(self, names: list[str], qt: np.ndarray):
        self.names, self.qt = names, qt
        self.idx = {n: i for i, n in enumerate(names)}
        self.cond_feats = [i for i, n in enumerate(names) if n not in COND_EXCLUDE]
        self.side_feats = [self.idx[n] for n in SIDE_FEATURES]
        self.lane_idx = self.idx["lane"]

    def thr(self, rng, f: int) -> float:
        return float(self.qt[f, rng.integers(0, self.qt.shape[1])])

    def random_cond(self, rng, g: np.ndarray, c: int, k: int, feat: int | None = None):
        f = int(rng.choice(self.cond_feats)) if feat is None else feat
        g[sim.I_CF + c * 3 + k], g[sim.I_CD + c * 3 + k], g[sim.I_CT + c * 3 + k] = f, rng.choice([-1, 1]), self.thr(rng, f)

    def exits(self, rng, g: np.ndarray):
        g[sim.I_STAKE] = rng.choice([0.05, 0.1, 0.2, 0.3])
        g[sim.I_MAX_ENT] = rng.choice([1, 1, 2, 3])
        g[sim.I_COOL] = rng.integers(0, 60)
        g[sim.I_TP] = rng.choice([0.0, 0.0, 0.03, 0.05, 0.08, 0.12])
        g[sim.I_STOP] = rng.choice([0.0, 0.0, 0.05, 0.08, 0.12])
        g[sim.I_HOLD] = rng.choice([0, 0, 30, 60, 120])

    def random_genome(self, rng, family: str | None = None) -> tuple[np.ndarray, str]:
        family = family or str(rng.choice(FAMILIES))
        g = sim.new_genome()
        lo = int(rng.integers(65, 500))
        g[sim.I_SEC_LO], g[sim.I_SEC_HI] = lo, int(rng.integers(lo + 60, 841))
        sf = int(rng.choice(self.side_feats))
        mode = 1
        if family == "fade":
            mode = -1
        elif family == "cross_coin_lead_lag":
            sf = self.idx["btc_mom30"]
        elif family == "late_window_convergence":
            sf = self.idx[str(rng.choice(["z", "delta_bps"]))]
            g[sim.I_SEC_LO], g[sim.I_SEC_HI] = 65, int(rng.integers(150, 420))
        g[sim.I_SIDE_F], g[sim.I_SIDE_MODE] = sf, mode
        g[sim.I_SIDE_DEAD] = abs(self.thr(rng, sf)) * rng.choice([0.0, 0.5, 1.0])
        for c in range(int(rng.integers(1, 4))):
            for k in range(int(rng.integers(1, 3 if family != "random_tree" else 4))):
                self.random_cond(rng, g, c, k)
        if family == "cross_coin_lead_lag":                         # the lead only means something for the other coins
            self.random_cond(rng, g, 0, 2, feat=self.lane_idx)
            g[sim.I_CD + 2], g[sim.I_CT + 2] = 1, 0.5
        self.exits(rng, g)
        return g, family

    def mutate(self, rng, parent: np.ndarray, strength: float = 1.0) -> np.ndarray:
        g = parent.copy()
        for _ in range(1 + int(rng.random() < 0.5 * strength) + int(rng.random() < 0.25 * strength)):
            r = rng.random()
            c, k = int(rng.integers(0, 3)), int(rng.integers(0, 3))
            ci = sim.I_CF + c * 3 + k
            if r < 0.35 and g[ci] >= 0:                                   # nudge a threshold within the feature's own quantiles
                f = int(g[ci])
                q = np.searchsorted(self.qt[f], g[sim.I_CT + c * 3 + k])
                g[sim.I_CT + c * 3 + k] = self.qt[f, int(np.clip(q + rng.integers(-8, 9), 0, self.qt.shape[1] - 1))]
            elif r < 0.50:
                self.random_cond(rng, g, c, k)
            elif r < 0.58:
                g[ci] = -1                                                   # drop a condition
            elif r < 0.64 and g[ci] >= 0:
                g[sim.I_CD + c * 3 + k] *= -1
            elif r < 0.72:
                g[sim.I_SIDE_F] = int(rng.choice(self.side_feats))
            elif r < 0.78:
                g[sim.I_SIDE_MODE] *= -1
            elif r < 0.86:
                lo = int(np.clip(g[sim.I_SEC_LO] + rng.integers(-60, 61), 65, 700))
                g[sim.I_SEC_LO], g[sim.I_SEC_HI] = lo, int(np.clip(g[sim.I_SEC_HI] + rng.integers(-60, 61), lo + 30, 840))
            else:
                self.exits(rng, g)
        if not (g[sim.I_CF:sim.I_CF + 9] >= 0).any():
            self.random_cond(rng, g, 0, 0)
        return g

    def crossover(self, rng, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        g = a.copy()
        for c in range(3):                                                   # whole clauses swap
            if rng.random() < 0.5:
                for base in (sim.I_CF, sim.I_CD, sim.I_CT):
                    g[base + c * 3:base + c * 3 + 3] = b[base + c * 3:base + c * 3 + 3]
        if rng.random() < 0.5:
            g[sim.I_STAKE:sim.I_HOLD + 1] = b[sim.I_STAKE:sim.I_HOLD + 1]
        if rng.random() < 0.5:
            g[sim.I_SIDE_F:sim.I_SIDE_DEAD + 1] = b[sim.I_SIDE_F:sim.I_SIDE_DEAD + 1]
        return g


def code_hash(g: np.ndarray) -> str:
    key = np.round(g.astype(np.float64), 6).tobytes() + IND.VERSION.encode()
    return hashlib.sha1(key).hexdigest()[:16]


def fitness(acc: np.ndarray) -> np.ndarray:
    """Train fitness: the noise-penalised mean net per trade (mean - se); too few trades is worse than any real score."""
    s = screen.summarize(acc)
    f = np.nan_to_num(s["mean"] - s["se"], nan=-1e6, posinf=-1e6, neginf=-1e6)
    return np.where(s["n"] >= MIN_TRAIN_TRADES, f, -1e3 + s["n"] / 1e3)


def evolve(train_book, space: Space, rng, *, pop: int = 2000, gens: int = 8, elite: float = 0.10, wblock=None, n_blocks: int = 1):
    """Run the GA on a train-only book. Returns (all genomes tried, their families, their train accumulators) with duplicates kept:
    every evaluation is a trial."""
    label = np.ones(len(train_book.wstart), np.int64)
    wb = np.zeros(len(train_book.wstart), np.int64) if wblock is None else wblock
    P = []
    fams = []
    for _ in range(pop):
        g, f = space.random_genome(rng)
        P.append(g)
        fams.append(f)
    P = np.stack(P)
    all_g, all_fam, all_acc = [], [], []
    for gen in range(gens):
        acc, _ = screen.screen(train_book, P, label, wb, max(n_blocks, 1))
        fit = fitness(acc[:, 1])
        all_g.append(P.copy())
        all_fam.append(list(fams))
        all_acc.append(acc[:, 1].copy())
        order = np.argsort(-fit)
        k = max(2, int(pop * elite))
        elites = P[order[:k]]
        elite_fams = [fams[i] for i in order[:k]]
        new, new_f = [elites[i] for i in range(k)], list(elite_fams)
        while len(new) < pop:
            ia, ib = rng.integers(0, k, 2) if rng.random() < 0.7 else (rng.integers(0, k), None)
            child = space.crossover(rng, elites[ia], elites[ib]) if ib is not None else elites[ia]
            new.append(space.mutate(rng, child))
            new_f.append(elite_fams[ia])
        # fresh blood keeps the search from collapsing onto one lineage
        for j in range(pop - max(1, pop // 20), pop):
            new[j], new_f[j] = space.random_genome(rng)
        P, fams = np.stack(new), new_f
    return np.concatenate(all_g), [f for gen in all_fam for f in gen], np.concatenate(all_acc)
