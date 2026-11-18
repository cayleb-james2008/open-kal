#!/usr/bin/env python3
"""Read-only export of the venue-labelled spot prints (`spot_ticks`) from the production database.

The production DB is live: the engine writes it in WAL mode. This script only ever reads it:
  * it connects with `file:...?mode=ro` and `PRAGMA query_only=ON`, so it cannot write or create the DB;
  * the snapshot is every row with rowid <= the max rowid seen once at the start, so rows written during the
    export are left for the next one;
  * rows are read in short rowid-ranged chunks, each its own read transaction, because a long read would hold
    back the WAL checkpoint of the live writer;
  * the copy is one new file, `spot_ticks.npz`, written to a temporary name in the destination directory and
    renamed into place, so a failure never leaves a partial file and never touches any other file there.
The default destination is the evolve cache, beside the other exports, per the owner's instruction; pass --dest to
write elsewhere. Nothing is read or written until you run it; the premise test needs about three days of ticks.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_DB = Path("/home/cayleb/open-kal/data/openkal-prod.db")
DEFAULT_DEST = Path("/home/cayleb/open-kal/data/evolve-cache")
OUT_NAME = "spot_ticks.npz"
CHUNK = 200_000
COLUMNS = "rowid, ts, symbol, source, price, bid, ask"


def open_ro(db: Path) -> sqlite3.Connection:
    """A connection that cannot write: sqlite refuses INSERT and will not create a missing file."""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30.0)
    conn.execute("PRAGMA query_only=ON")
    return conn


def check_dest(db: Path, dest: Path) -> Path:
    """The output path, after refusing a missing directory or anything that is the database or its WAL files."""
    if not dest.is_dir():
        raise RuntimeError(f"destination {dest} is not a directory")
    out = (dest / OUT_NAME).resolve()
    guarded = {db.resolve(), Path(f"{db.resolve()}-wal"), Path(f"{db.resolve()}-shm")}
    if out in guarded:
        raise RuntimeError("refusing to write over the production database")
    return out


def read_snapshot(db: Path, *, chunk: int = CHUNK, on_snapshot=None) -> tuple[dict, int]:
    """(arrays, max_rowid): every spot_ticks row up to the rowid bound taken at the start, sorted by time."""
    conn = open_ro(db)
    try:
        bound = conn.execute("SELECT max(rowid) FROM spot_ticks").fetchone()[0] or 0
        if on_snapshot is not None:
            on_snapshot(bound)                              # test hook: rows added after this point must stay out
        cols = {k: [] for k in ("ts", "symbol", "source", "price", "bid", "ask")}
        last = 0
        while last < bound:
            rows = conn.execute(f"SELECT {COLUMNS} FROM spot_ticks WHERE rowid > ? AND rowid <= ? ORDER BY rowid LIMIT ?",
                                (last, bound, chunk)).fetchall()
            if not rows:
                break
            last = rows[-1][0]
            for k, i in (("ts", 1), ("symbol", 2), ("source", 3), ("price", 4), ("bid", 5), ("ask", 6)):
                cols[k].extend(r[i] for r in rows)
    finally:
        conn.close()
    num = lambda xs: np.array([np.nan if x is None else x for x in xs], dtype=np.float64)
    ts = num(cols["ts"])
    order = np.argsort(ts, kind="stable")
    arrays = {"ts": ts[order], "symbol": np.array(cols["symbol"], dtype="U16")[order] if cols["symbol"] else np.zeros(0, "U16"),
              "source": np.array(cols["source"], dtype="U16")[order] if cols["source"] else np.zeros(0, "U16"),
              "price": num(cols["price"])[order], "bid": num(cols["bid"])[order], "ask": num(cols["ask"])[order]}
    return arrays, int(bound)


def export(db: Path = DEFAULT_DB, dest: Path = DEFAULT_DEST, *, chunk: int = CHUNK, on_snapshot=None) -> dict:
    """Write `spot_ticks.npz` into `dest` atomically and return a summary of what was copied."""
    out = check_dest(db, dest)
    arrays, bound = read_snapshot(db, chunk=chunk, on_snapshot=on_snapshot)
    arrays["meta_max_rowid"] = np.array([bound], dtype=np.int64)
    arrays["meta_exported_at"] = np.array([time.time()], dtype=np.float64)
    arrays["meta_source_db"] = np.array([str(db)])
    tmp = out.with_name(f"{OUT_NAME[:-4]}.tmp-{os.getpid()}.npz")
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    sources, counts = np.unique(arrays["source"], return_counts=True)
    ts = arrays["ts"]
    return {"path": str(out), "rows": int(len(ts)), "max_rowid": bound, "ts_min": float(ts.min()) if len(ts) else None,
            "ts_max": float(ts.max()) if len(ts) else None, "rows_by_source": {str(s): int(c) for s, c in zip(sources, counts)}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = ap.parse_args()
    print(export(args.db, args.dest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
