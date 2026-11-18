"""Operator arming: a person's act, not a file the process can find for itself.

The old gate read `OPENKAL_ARM_LIVE=1` and armed as soon as readiness passed. That
made every restart, every wrapper script and every inherited shell a potential arm:
nothing in the process could tell an operator's decision from a stale line in `.env`.

The engine now requires `data/live-arm.json`, written only by
`scripts/arm_live.py`, stamped with the name of the person who ran it and with a
timestamp *after this process started*. The record is discarded at boot, so a
restart is never an arm, and it is consumed when the arm succeeds, so one act arms
one process. A verdict, a promotion, a meta row and an environment variable are all
things this module refuses to read as permission.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
ARM_PATH = ROOT / "data" / "live-arm.json"

# An arm is a decision about *now*. Fifteen minutes is long enough for an operator
# to run the command and watch the next readiness pass, and short enough that a
# forgotten record is not a standing licence.
MAX_AGE_S = 15 * 60.0
# Process start times and the file's write time come from the same clock, but a
# rounding step between `time.time()` calls must not invalidate a fresh record.
CLOCK_SLACK_S = 2.0


def read(path: Path | None = None) -> dict[str, Any] | None:
    """The arm record on disk, or None. A malformed file is None, never a pass."""
    target = path or ARM_PATH
    try:
        data = json.loads(target.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def operator_arm(by: str, *, note: str = "", path: Path | None = None,
                 now: float | None = None) -> dict[str, Any]:
    """Write the arm record. The caller is a person naming themselves.

    `by` is required and cannot be a role or a hostname placeholder: an arm with no
    name attached is exactly the kind of anonymous state this lock exists to reject.
    """
    name = (by or "").strip()
    if not name:
        raise ValueError("arming requires the name of the person doing it (--by)")
    record = {
        "ts": time.time() if now is None else float(now),
        "by": name,
        "note": note,
        "nonce": uuid.uuid4().hex,
    }
    target = path or ARM_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(tmp, target)
    return record


def consume(path: Path | None = None) -> dict[str, Any] | None:
    """Read and remove the record: one operator act arms exactly one process."""
    target = path or ARM_PATH
    record = read(target)
    try:
        target.unlink()
    except OSError:
        pass
    return record


def clear(path: Path | None = None) -> dict[str, Any] | None:
    """Discard any record left over from an earlier boot. Called before the loop."""
    return consume(path)


def armed(record: dict[str, Any] | None, *, started: float, now: float,
          max_age_s: float = MAX_AGE_S) -> bool:
    """True only for a named record written after this process started.

    `record["ts"] >= started` is the restart rule: a record that predates the boot
    belonged to a different process and may not arm this one, even if the file is
    still on disk because the previous process died before it could arm.
    """
    if not record:
        return False
    if not str(record.get("by") or "").strip():
        return False
    try:
        ts = float(record.get("ts") or 0.0)
    except (TypeError, ValueError):
        return False
    if ts < started - CLOCK_SLACK_S:
        return False
    return 0.0 <= now - ts <= max_age_s
