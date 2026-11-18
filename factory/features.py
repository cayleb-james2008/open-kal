"""Per-lane feature matrix on a one-second grid, written once to parquet; every candidate is a pass over it.

Reuses the audited helpers (alpha_search flow and revision imbalance, trailing sigma, the causal index lookup of rnd_pool_build) so
a feature here is the feature the earlier tracks used. Everything at grid second t uses data at or before t.
"""
from __future__ import annotations

import gc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from factory import indicators as IND  # noqa: E402

GRID_FROM_S, GRID_TO_S = 840, 60                # seconds left, descending, one row per second
LANES = ("KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M")
OUT_ROOT = ROOT / "data" / "factory"


def last_at_or_before(ts: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Index of the last element of sorted `ts` at or before each grid second; -1 where there is none."""
    return np.searchsorted(ts, grid, "right") - 1


def p_up_from_z(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))


def window_frame(*, lane_id: int, close: float, strike: float, y: int, book_ts, bid, ask, bsz, asz, rev, sigma_row,
                 spot_at, flow_at, hod0: float) -> dict[str, np.ndarray]:
    """Features of one window on its grid. `spot_at(ts)` and `flow_at(ts, window)` are causal lookups for arbitrary seconds."""
    secs = np.arange(GRID_FROM_S, GRID_TO_S - 1, -1, dtype=np.float64)
    g = close - secs
    j = last_at_or_before(book_ts, g)
    ok = j >= 0
    jj = np.clip(j, 0, len(book_ts) - 1)
    f32 = np.float32
    b, a = np.where(ok, bid[jj], np.nan).astype(f32), np.where(ok, ask[jj], np.nan).astype(f32)
    bs, as_ = np.where(ok, bsz[jj], np.nan).astype(f32), np.where(ok, asz[jj], np.nan).astype(f32)
    mid = (b + a) / 2.0
    j5 = np.clip(last_at_or_before(book_ts, g - 5.0), 0, len(book_ts) - 1)
    mid5 = (bid[j5] + ask[j5]) / 2.0
    spot = spot_at(g)
    sigma = np.where(ok, sigma_row[jj], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.log(spot / strike) / (sigma * np.sqrt(secs))
        ret = lambda s0: (spot / spot_at(g - s0) - 1.0) * 1e4
        l2 = (bs - as_) / (bs + as_)
        touch = np.log(np.minimum(bs, as_))
    cols = {
        "lane": np.full(len(g), lane_id, dtype=np.int8), "close": np.full(len(g), close), "secs": secs.astype(f32), "y": np.full(len(g), y, dtype=np.int8),
        "strike": np.full(len(g), strike), "bid": b, "ask": a, "bsz": bs, "asz": as_, "valid": (ok & (b > 0) & (a < 1) & (b < a)),
        "z": z.astype(f32), "p_up": p_up_from_z(z).astype(f32), "delta_bps": ((spot - strike) / spot * 1e4).astype(f32),
        "flow10": flow_at(g, 10.0).astype(f32), "flow30": flow_at(g, 30.0).astype(f32),
        "rev5": np.where(ok, rev[jj], np.nan).astype(f32), "shift5": (mid - mid5).astype(f32),
        "mom10": ret(10.0).astype(f32), "mom30": ret(30.0).astype(f32),
        "spread": (a - b).astype(f32), "mid": mid.astype(f32), "l2imb": l2.astype(f32), "log_touch": touch.astype(f32),
        "sigma_bps": (sigma * 1e4).astype(f32), "quote_age": np.where(ok, g - book_ts[jj], np.nan).astype(f32),
        "hod": (hod0 + (g - g[0]) / 3600.0).astype(f32), "spot": spot,
    }
    return cols


def build_lane(lane, lane_id: int, sizes: tuple[np.ndarray, np.ndarray], spot_at_factory) -> pl.DataFrame:
    """The grid frame of every settled window of one `alpha_discover.Lane`."""
    from scripts import alpha_search as A
    sig = A.trailing_sigma(lane)
    frames = []
    bsz_all, asz_all = sizes
    for tid, name, meta, a, b in A._segments(lane):
        if tid not in sig:
            continue
        ts = lane.ts[a:b]
        rev = A.revision_imbalance(lane, a, b, 5.0)
        close = float(meta["close"])
        hod0 = (time.localtime(close - GRID_FROM_S).tm_hour * 3600 + time.localtime(close - GRID_FROM_S).tm_min * 60) / 3600.0
        cols = window_frame(lane_id=lane_id, close=close, strike=float(meta["strike"]), y=int(meta["y"]), book_ts=ts,
                            bid=lane.bid[a:b], ask=lane.ask[a:b], bsz=bsz_all[a:b], asz=asz_all[a:b], rev=rev, sigma_row=sig[tid],
                            spot_at=spot_at_factory(lane), flow_at=lambda g, w, _t=tid: A.flow_imbalance(lane, _t, g, w), hod0=hod0)
        frames.append(pl.DataFrame(cols))
    return pl.concat(frames) if frames else pl.DataFrame()


def add_cross_lane(frames: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """btc_mom30 and btc_z: BTC's value at the same close and second, joined onto every lane."""
    btc = frames["KXBTC15M"].select(["close", "secs", pl.col("mom30").alias("btc_mom30"), pl.col("z").alias("btc_z")])
    return {k: v.join(btc, on=["close", "secs"], how="left") for k, v in frames.items()}


def median_touch(frames: dict[str, pl.DataFrame]) -> dict[str, float]:
    out = {}
    for k, v in frames.items():
        m = v.filter(pl.col("bsz").is_finite() & pl.col("asz").is_finite()).select(pl.concat_list(["bsz", "asz"]).list.min().median()).item()
        out[k] = float(m) if m is not None else 0.0
    return out


def write(frames: dict[str, pl.DataFrame], out_dir: Path, meta: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, v in frames.items():
        v.write_parquet(out_dir / f"{k}.parquet", compression="zstd")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return out_dir


def build(tape: Path, out_dir: Path | None = None) -> Path:
    """Pin-check the tape, build the four lanes one at a time, write parquet and meta.json."""
    from scripts import alpha_search as A
    from scripts import rnd_common as RC
    from scripts import rnd_pool_build as RB
    from scripts import track4_common as T

    pin = T.pin_tape(tape)
    closes = T.window_closes(tape)
    out_dir = out_dir or OUT_ROOT / f"features-{int(closes[-1])}"
    if (out_dir / "meta.json").exists():
        raise RuntimeError(f"{out_dir} already holds a feature matrix")
    A.index_at = RB.causal_index_at
    frames = {}
    for i, name in enumerate(LANES):
        t0 = time.time()
        lane = RC.load_lanes((name,))[name]
        with np.load(tape / f"book_{name}.npz", allow_pickle=False) as z:
            sizes = (z["yes_bid_sz"].astype(np.float32), z["yes_ask_sz"].astype(np.float32))
        frames[name] = build_lane(lane, i, sizes, lambda ln: (lambda ts: RB.causal_index_at(ln, ts)))
        print(f"{name}: {frames[name].height:,} rows ({time.time() - t0:.0f}s)", flush=True)
        del lane, sizes
        gc.collect()
    frames = add_cross_lane(frames)
    meta = {"registry_version": IND.VERSION, "features": IND.NAMES, "tape": str(tape), "pin_sha256": pin["sha256"],
            "grid": {"from_secs": GRID_FROM_S, "to_secs": GRID_TO_S, "step_s": 1}, "median_touch": median_touch(frames),
            "windows": len(closes), "first_close": closes[0], "last_close": closes[-1]}
    return write(frames, out_dir, meta)


if __name__ == "__main__":
    print(build(ROOT / "data" / "track10-tape"))
