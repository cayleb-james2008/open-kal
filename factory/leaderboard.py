"""The leaderboard: every candidate any tier scored is a row, and the trial counter is kept beside it (never deleted)."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
  id INTEGER PRIMARY KEY AUTOINCREMENT, code_hash TEXT NOT NULL, genome_json TEXT NOT NULL, family TEXT NOT NULL, epoch INTEGER NOT NULL,
  train_end REAL, registry_version TEXT, created_ts REAL, train_metrics TEXT, oos_metrics TEXT, stress_metrics TEXT,
  bound_uncorrected REAL, bound_corrected REAL, n_candidates_in_epoch INTEGER, status TEXT, failure_reason TEXT, tier TEXT DEFAULT 'screen',
  UNIQUE(code_hash, epoch));
CREATE TABLE IF NOT EXISTS trials (epoch INTEGER NOT NULL, family TEXT NOT NULL, tried INTEGER NOT NULL, PRIMARY KEY (epoch, family));
CREATE TABLE IF NOT EXISTS epochs (epoch INTEGER PRIMARY KEY, started REAL, finished REAL, train_blocks TEXT, test_blocks TEXT, tried INTEGER,
  distinct_candidates INTEGER, pbo REAL, screen_seconds REAL, notes TEXT);
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    return db


def add_trials(db, epoch: int, families: list[str]) -> None:
    counts: dict[str, int] = {}
    for f in families:
        counts[f] = counts.get(f, 0) + 1
    for f, n in counts.items():
        db.execute("INSERT INTO trials(epoch, family, tried) VALUES (?,?,?) ON CONFLICT(epoch, family) DO UPDATE SET tried = tried + excluded.tried", (epoch, f, n))


def trials_in_epoch(db, epoch: int) -> int:
    return int(db.execute("SELECT COALESCE(SUM(tried), 0) FROM trials WHERE epoch = ?", (epoch,)).fetchone()[0])


def log_candidates(db, rows: list[dict]) -> None:
    """Insert or update by (code_hash, epoch); a repeated candidate keeps its first row (the trial counter holds the repeats)."""
    now = time.time()
    db.executemany(
        "INSERT INTO candidates(code_hash, genome_json, family, epoch, train_end, registry_version, created_ts, train_metrics, oos_metrics, stress_metrics,"
        " bound_uncorrected, bound_corrected, n_candidates_in_epoch, status, failure_reason, tier) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(code_hash, epoch) DO UPDATE SET oos_metrics=excluded.oos_metrics, stress_metrics=excluded.stress_metrics, "
        "bound_uncorrected=excluded.bound_uncorrected, bound_corrected=excluded.bound_corrected, n_candidates_in_epoch=excluded.n_candidates_in_epoch, "
        "status=excluded.status, failure_reason=excluded.failure_reason, tier=excluded.tier",
        [(r["code_hash"], json.dumps(r["genome"]), r["family"], r["epoch"], r.get("train_end"), r.get("registry_version"), now,
          json.dumps(r.get("train_metrics")), json.dumps(r.get("oos_metrics")), json.dumps(r.get("stress_metrics")), r.get("bound_uncorrected"),
          r.get("bound_corrected"), r.get("n_candidates_in_epoch"), r.get("status", "train_only"), r.get("failure_reason"), r.get("tier", "screen"))
         for r in rows])


def counts(db) -> dict:
    return {"candidate_rows": db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0],
            "trials": int(db.execute("SELECT COALESCE(SUM(tried), 0) FROM trials").fetchone()[0]),
            "by_status": dict(db.execute("SELECT status, COUNT(*) FROM candidates GROUP BY status").fetchall())}
