"""Load the per-lane parquet feature matrices into the simulator's `Book`."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from factory import indicators as IND
from factory.features import LANES
from factory.sim import Book

DEPTH_FALLBACK = 25.0


def from_frame(df: pl.DataFrame, names: list[str], median_touch: dict[int, float] | None = None, capital: float = 164.0) -> Book:
    """A Book from one frame sorted by (lane, close, secs descending); windows are the runs of equal (lane, close)."""
    lane = df["lane"].to_numpy()
    close = df["close"].to_numpy()
    key_change = np.r_[True, (lane[1:] != lane[:-1]) | (close[1:] != close[:-1])]
    starts = np.flatnonzero(key_change)
    ends = np.r_[starts[1:], len(df)]
    X = np.ascontiguousarray(df.select(names).to_numpy().astype(np.float32).T)
    mt = median_touch or {}
    wlane = lane[starts].astype(np.int64)
    return Book(X=X, names=list(names), bid=df["bid"].to_numpy().astype(np.float64), ask=df["ask"].to_numpy().astype(np.float64),
                bsz=df["bsz"].to_numpy().astype(np.float64), asz=df["asz"].to_numpy().astype(np.float64), valid=df["valid"].to_numpy(),
                wstart=starts.astype(np.int64), wend=ends.astype(np.int64), wy=df["y"].to_numpy()[starts].astype(np.int64),
                wclose=close[starts], wlane=wlane, wdepth=np.array([mt.get(int(l), DEPTH_FALLBACK) or DEPTH_FALLBACK for l in wlane]),
                capital=capital)


def load(feature_dir: Path, names: list[str] | None = None) -> Book:
    meta = json.loads((feature_dir / "meta.json").read_text())
    if meta["registry_version"] != IND.VERSION:
        raise RuntimeError(f"feature matrix was built with registry {meta['registry_version']}, code has {IND.VERSION}")
    names = names or IND.NAMES
    df = pl.concat([pl.read_parquet(feature_dir / f"{k}.parquet") for k in LANES]).sort(["lane", "close", "secs"], descending=[False, False, True])
    mt = {i: meta["median_touch"][k] for i, k in enumerate(LANES)}
    return from_frame(df, names, mt)
