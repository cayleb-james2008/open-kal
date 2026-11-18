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


def arm_path(lane: str | None = None) -> Path:
    """Where one lane's operator record lives.

    The crypto engine owns `data/live-arm.json`. Every separate lane gets its own
    file, so arming a new lane can never arm the crypto lanes (or each other). The
    name is a path-safe slug, never a caller-supplied path.
    """
    if not lane or lane.strip().lower() == "crypto":
        return ARM_PATH
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-"
                   for ch in lane.strip().lower()).strip("-")
    if not safe:
        raise ValueError("lane name must contain a letter or digit")
    return ROOT / "data" / f"live-arm-{safe}.json"

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
                 now: float | None = None, lane: str | None = None) -> dict[str, Any]:
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
    if lane and lane.strip().lower() != "crypto":
        # `--lane crypto` is the legacy engine record, not a tagged lane record.
        record["lane"] = lane.strip().lower()
    target = path or arm_path(lane)
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
          max_age_s: float = MAX_AGE_S, lane: str | None = None) -> bool:
    """True only for a named record written after this process started.

    `record["ts"] >= started` is the restart rule: a record that predates the boot
    belonged to a different process and may not arm this one, even if the file is
    still on disk because the previous process died before it could arm.
    """
    if not record:
        return False
    if not str(record.get("by") or "").strip():
        return False
    # A lane record is only ever permission for that lane. With no lane asked for
    # (the crypto engine's call), any record carrying a lane is refused, so a
    # gas-no file copied onto `live-arm.json` still cannot arm crypto.
    tagged = str(record.get("lane") or "").strip().lower()
    wanted = (lane.strip().lower()
              if lane is not None and lane.strip().lower() != "crypto" else None)
    if wanted is None:
        if tagged:
            return False
    elif tagged != wanted:
        return False
    try:
        ts = float(record.get("ts") or 0.0)
    except (TypeError, ValueError):
        return False
    if ts < started - CLOCK_SLACK_S:
        return False
    return 0.0 <= now - ts <= max_age_s


def fresh_operator_act(record: dict[str, Any] | None, *, now: float,
                       lane: str | None = None,
                       max_age_s: float = MAX_AGE_S) -> bool:
    """A one-shot consumer's arm check: a named, fresh, lane-specific act.

    `armed()` requires the record to be written *after* the process started, which
    is right for a long-running engine: a restart must not inherit the previous
    boot's act. A one-shot command is armed *before* it starts, so that rule would
    reject every legitimate use. The protections that still apply are the ones
    that matter here: a named person, the lane's own record, a 15-minute window,
    and `consume()` so one act runs exactly one process.
    """
    if not record:
        return False
    if not str(record.get("by") or "").strip():
        return False
    tagged = str(record.get("lane") or "").strip().lower()
    wanted = (lane.strip().lower()
              if lane is not None and lane.strip().lower() != "crypto" else None)
    if wanted is None:
        if tagged:
            return False
    elif tagged != wanted:
        return False
    try:
        ts = float(record.get("ts") or 0.0)
    except (TypeError, ValueError):
        return False
    return 0.0 <= now - ts <= max_age_s
