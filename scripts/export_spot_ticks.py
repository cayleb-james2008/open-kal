#!/usr/bin/env python3
"""Read-only, accumulating export of the venue-labelled spot prints (`spot_ticks`) from the production database.

The production DB is live: the engine writes it in WAL mode, and `Store.prune()` deletes `spot_ticks` rows older than
three days whenever the engine process shuts down. This script therefore keeps an ARCHIVE (`spot_ticks.npz`): each run
appends only the rows newer than the archive's rowid watermark, so rows survive a later prune. Run it daily, and before any
engine restart; a single export on the day the premise test is due can hold at most three days.

It only ever reads the database:
  * it connects with `file:...?mode=ro` and `PRAGMA query_only=ON`, so it cannot write the DB;
  * it refuses when the `-wal`/`-shm` files do not exist (the engine is stopped): a reader would create them beside the DB.
    With the engine running, a WAL reader writes reader-coordination state into `-shm`; that is inherent to SQLite and holds no
    data. The main file and the WAL are never modified (tests compare their hashes);
  * the batch is every row with watermark < rowid <= the max rowid seen once at the start, read in short rowid-ranged chunks,
    each its own read transaction, so a long read never holds back the live writer's WAL checkpoint; afterwards the row count
    is checked against the count taken when the snapshot started, so a row pruned before it was read fails the run instead of dropping silently;
  * the archive is written to a temporary name beside it, fsynced, and renamed into place, so a failure never leaves a partial
    file or touches any other file in the directory.
The default destination is the evolve cache, beside the other exports, per the owner's instruction; pass --dest to write elsewhere.
Nothing is read or written until you run it.
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
FIELDS = ("ts", "symbol", "source", "price", "bid", "ask")


def open_ro(db: Path) -> sqlite3.Connection:
    """A connection that cannot write: sqlite refuses INSERT and will not create a missing file."""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=30.0)
    conn.execute("PRAGMA query_only=ON")
    return conn


def check_paths(db: Path, dest: Path) -> Path:
    """The archive path, after refusing a bad destination, the database itself, or a database with no WAL files."""
    if not dest.is_dir():
        raise RuntimeError(f"destination {dest} is not a directory")
    out = (dest / OUT_NAME).resolve()
    if out == db.resolve():
        raise RuntimeError("refusing to write over the production database")
    missing = [s for s in ("-wal", "-shm") if not Path(f"{db.resolve()}{s}").exists()]
    if missing:
        raise RuntimeError(f"{db} has no {', '.join(missing)}: the engine is stopped, and reading would create files beside the "
                           "production database. Export while the engine runs, or point --db at a copy.")
    return out


def to_strings(values) -> np.ndarray:
    return np.array([("" if v is None else str(v)) for v in values]) if len(values) else np.zeros(0, "U1")


def to_floats(values) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in values], dtype=np.float64)


def read_new_rows(db: Path, watermark: int, *, chunk: int = CHUNK, on_snapshot=None, on_chunk=None) -> tuple[dict, int]:
    """(arrays, bound): the rows with watermark < rowid <= the bound taken at the start, in rowid order."""
    conn = open_ro(db)
    try:
        bound = conn.execute("SELECT max(rowid) FROM spot_ticks").fetchone()[0] or 0
        if bound < watermark:
            raise RuntimeError(f"the table's max rowid {bound} is below the archive watermark {watermark}: it was rebuilt; refusing to merge")
        expected = conn.execute("SELECT count(*) FROM spot_ticks WHERE rowid > ? AND rowid <= ?", (watermark, bound)).fetchone()[0]
        if on_snapshot is not None:
            on_snapshot(bound)                              # test hook: rows added after this point must stay out
        cols = {k: [] for k in FIELDS}
        last = watermark
        while last < bound:
            rows = conn.execute(f"SELECT {COLUMNS} FROM spot_ticks WHERE rowid > ? AND rowid <= ? ORDER BY rowid LIMIT ?",
                                (last, bound, chunk)).fetchall()
            if not rows:
                break
            last = rows[-1][0]
            for i, k in enumerate(FIELDS, start=1):
                cols[k].extend(r[i] for r in rows)
            if on_chunk is not None:
                on_chunk(last)                              # test hook: a prune between chunks
    finally:
        conn.close()
    if expected != len(cols["ts"]):
        raise RuntimeError(f"{expected} rows were in the database for this range when the snapshot started but {len(cols['ts'])} were "
                           "read: rows vanished during the export (a prune?). Nothing was written; run it again.")
    return {"ts": to_floats(cols["ts"]), "symbol": to_strings(cols["symbol"]), "source": to_strings(cols["source"]),
            "price": to_floats(cols["price"]), "bid": to_floats(cols["bid"]), "ask": to_floats(cols["ask"])}, int(bound)


def load_archive(out: Path) -> tuple[dict, int]:
    """(arrays, watermark) of the existing archive, or empty arrays and 0 when there is none."""
    if not out.exists():
        return {"ts": np.zeros(0), "symbol": np.zeros(0, "U1"), "source": np.zeros(0, "U1"), "price": np.zeros(0), "bid": np.zeros(0),
                "ask": np.zeros(0)}, 0
    z = np.load(out)
    return {k: z[k] for k in FIELDS}, int(z["meta_max_rowid"][0])


def write_archive(out: Path, arrays: dict) -> None:
    tmp = out.with_name(f"{OUT_NAME[:-4]}.tmp-{os.getpid()}.npz")
    try:
        np.savez_compressed(tmp, **arrays)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)


def export(db: Path = DEFAULT_DB, dest: Path = DEFAULT_DEST, *, chunk: int = CHUNK, on_snapshot=None, on_chunk=None) -> dict:
    """Append the new rows to `spot_ticks.npz` in `dest` (created on the first run) and return a summary."""
    out = check_paths(db, dest)
    old, watermark = load_archive(out)
    new, bound = read_new_rows(db, watermark, chunk=chunk, on_snapshot=on_snapshot, on_chunk=on_chunk)
    merged = {k: np.concatenate([old[k], new[k]]) for k in FIELDS}
    order = np.argsort(merged["ts"], kind="stable")
    arrays = {k: v[order] for k, v in merged.items()}
    arrays["meta_max_rowid"] = np.array([bound], dtype=np.int64)
    arrays["meta_exported_at"] = np.array([time.time()], dtype=np.float64)
    arrays["meta_source_db"] = np.array([str(db)])
    write_archive(out, arrays)
    sources, counts = np.unique(arrays["source"], return_counts=True)
    ts = arrays["ts"]
    return {"path": str(out), "rows": int(len(ts)), "new_rows": int(len(new["ts"])), "max_rowid": bound,
            "ts_min": float(ts.min()) if len(ts) else None, "ts_max": float(ts.max()) if len(ts) else None,
            "rows_by_source": {str(s): int(c) for s, c in zip(sources, counts)}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = ap.parse_args()
    print(export(args.db, args.dest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
