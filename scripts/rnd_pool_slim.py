#!/usr/bin/env python3
"""Shrink the cached trigger pools to what the tracks read.

data/rnd-pool/pool_<LANE>.npz (from sim_core_parity.py) is ~12 GB uncompressed, most
of it string columns. This keeps the numeric columns in compact types and the rows
with |z| >= 0.10 (below every z threshold any track uses), and writes
data/rnd-pool/slim_<LANE>.npz. The ticker of a row is `lane.tickers[tid]`. Rows are
not thinned in time: a 0.25s bucket thinning was tried and lost 4 of 9 ETH F2
triggers, because a qualifying row is often a single book update.

Columns are read one at a time (NpzFile loads lazily), so peak memory stays near one
column of the largest lane.

    .venv/bin/python scripts/rnd_pool_slim.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / "data" / "rnd-pool"
LANES = ("KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M")
KEEP = {"tid": np.int32, "ts": np.float64, "close": np.float64, "y": np.int8,
        # Every column a threshold reads stays float64, as `Lane` holds it. NumPy
        # compares a float32 array with a Python float in float32, so a stored ask of
        # 0.60 passes `<= 0.60 + 1e-9` in float32 and fails it in float64: float32
        # here added 28 F2 triggers that the float64 pool does not have.
        "bid": np.float64, "ask": np.float64, "spot": np.float64, "delta": np.float64,
        "secs": np.float32, "p_up": np.float32, "z": np.float64, "flow": np.float64,
        "rev": np.float64, "shift": np.float64}
Z_FLOOR = 0.10


def main() -> int:
    for lane in (sys.argv[1:] or LANES):
        src = np.load(POOL / f"pool_{lane}.npz", allow_pickle=False)
        z = src["z"]
        tid = src["tid"]
        ts = src["ts"]
        idx = np.flatnonzero(np.abs(z) >= Z_FLOOR)
        assert np.all(np.diff(tid[idx]) >= 0), "pool rows must stay grouped by tid"
        out = {}
        for k, dt in KEEP.items():
            out[k] = np.asarray(src[k][idx], dtype=dt)
        del z, tid, ts
        np.savez(POOL / f"slim_{lane}.npz", **out)
        print(f"{lane}: {len(idx):,} rows kept", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
