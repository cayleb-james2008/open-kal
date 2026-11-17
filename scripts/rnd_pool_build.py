#!/usr/bin/env python3
"""Build the tracks' trigger pools with a causal spot, one lane at a time.

`bounded_sweep.obi_pool` reads spot through `alpha_search.index_at`, which returns
the NEAREST settlement-index print within 1.5s on either side of a book row. A trigger
at T can therefore see a print from up to T + 1.5s, half a second past the order's
own T + 1s arrival. Here `index_at` is replaced, for this build only, by the last
print at or before the row (still within 1.5s), so z, delta and p_up use only what was
known at the trigger. Flow, revision imbalance and the trailing sigma were already
causal and are unchanged.

Output: data/rnd-pool/slim_<LANE>.npz, numeric columns, rows with |z| >= 0.10. Every
column a threshold reads is float64, as `Lane` holds it: NumPy compares a float32
array with a Python float in float32, so a float32 ask of 0.60 passes
`<= 0.60 + 1e-9` and flips the F2 cost band (seen: 137 triggers instead of 109).
One lane at a time keeps the peak near one lane's pool build.

    flock data/.heavy.lock timeout 1800 nice -n 10 .venv/bin/python scripts/rnd_pool_build.py
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import alpha_search as A  # noqa: E402
from scripts import f2_structural as F  # noqa: E402
from scripts import rnd_common as R  # noqa: E402

POOL = ROOT / "data" / "rnd-pool"
Z_FLOOR = 0.10
PREFIX = "slim_"
KEEP = {"tid": np.int32, "ts": np.float64, "close": np.float64, "y": np.int8,
        "bid": np.float64, "ask": np.float64, "spot": np.float64, "delta": np.float64,
        "secs": np.float32, "p_up": np.float32, "z": np.float64, "flow": np.float64,
        "rev": np.float64, "shift": np.float64}


def causal_index_at(lane, ts: np.ndarray) -> np.ndarray:
    """Last settlement-index print at or before each `ts`, within 1.5s; else NaN."""
    if not lane.index:
        return np.full(len(ts), np.nan)
    arr = lane.index["ts"]
    j = np.searchsorted(arr, ts, "right") - 1
    jj = np.clip(j, 0, len(arr) - 1)
    out = lane.index["value"][jj].astype(np.float64)
    out[(j < 0) | (ts - arr[jj] > 1.5)] = np.nan
    return out


def main() -> int:
    A.index_at = causal_index_at   # obi_pool resolves A.index_at at call time
    POOL.mkdir(parents=True, exist_ok=True)
    for name in (sys.argv[1:] or R.LANES):
        t0 = time.time()
        lanes = R.load_lanes((name,))
        pool = F.build_pools(lanes)[name]
        idx = np.flatnonzero(np.abs(pool["z"]) >= Z_FLOOR)
        assert np.all(np.diff(pool["tid"][idx]) >= 0), "pool rows must stay grouped by tid"
        out = {k: np.asarray(pool[k][idx], dtype=dt) for k, dt in KEEP.items()}
        np.savez(POOL / f"{PREFIX}{name}.npz", **out)
        print(f"{name}: {len(idx):,} rows ({time.time() - t0:.0f}s)", flush=True)
        del lanes, pool, out
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
