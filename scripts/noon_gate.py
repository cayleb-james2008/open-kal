#!/usr/bin/env python3
"""Sleep until the noon forward-OOS gate, then hand off to `day_supervisor.py`.

This wrapper exists so the 12:00:05 single shot survives the shell that starts it.
It deliberately does not score, tune, promote, or supervise anything itself:
`scripts/day_supervisor.py` already owns the rebuild, the one-shot forward score,
the 13:00/14:00 trade-count retries, promotion, the 15-minute checks, the hard
breakers, and the 18:00 report. A fresh state file is used so the old supervisor's
pre-noon supervision rows cannot consume the 12:00/13:00/14:00 schedule.

The wrapper also strips operation-only paper/live overrides from its own
environment before it sleeps, so the supervisor it execs, the canary it starts,
and the `restart_flat.sh --new-epoch` engine all read the normal `.env` live gate.
"""
from __future__ import annotations

import argparse
import os
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET_TS = 1794934805.0  # 2026-11-17 12:00:05 America/New_York
LOG = ROOT / "data" / "noon-gate.log"
STATE = ROOT / "data" / "noon-gate-state.json"
SUPERVISOR = ROOT / "scripts" / "day_supervisor.py"

# The engine was deliberately started with the hard paper lock for the morning
# audit. None of these may leak into the noon supervisor's process tree, or the
# post-gate restart would stay paper and never reach live_armed: true.
SANITIZE_KEYS = (
    "OPENKAL_FORCE_PAPER",
    "OPENKAL_ARM_LIVE",
    "ALLOW_LIVE",
    "OPENKAL_ENV",
)

CHECKS = ",".join((
    "2026-11-17T12:00:00-05:00",
    "2026-11-17T13:00:00-05:00",
    "2026-11-17T14:00:00-05:00",
))


def log(message: str) -> None:
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"{stamp} noon-gate: {message}", flush=True)


def sanitize_env() -> list[str]:
    removed: list[str] = []
    for key in SANITIZE_KEYS:
        if key in os.environ:
            os.environ.pop(key, None)
            removed.append(key)
    return removed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-ts", type=float, default=TARGET_TS)
    ap.add_argument("--supervisor", default=str(SUPERVISOR))
    ap.add_argument("--log-file", default=str(LOG))
    ap.add_argument("--state-file", default=str(STATE))
    args = ap.parse_args()

    removed = sanitize_env()
    log(f"env sanitized: removed {removed or 'none'}; "
        f"keys={','.join(SANITIZE_KEYS)}")

    target_ts = float(args.target_ts)
    target = datetime.fromtimestamp(target_ts).astimezone()
    supervisor = Path(args.supervisor)
    log_file = Path(args.log_file)
    state_file = Path(args.state_file)
    log(f"waiting for {target.isoformat()} (ts {target_ts:.0f}); "
        f"supervisor={supervisor} log={log_file} state={state_file}")

    last = 0.0
    while True:
        now = time.time()
        left = target_ts - now
        if left <= 0:
            break
        if now - last >= 300.0:
            log(f"still waiting: {left / 3600.0:.2f}h left (pid {os.getpid()})")
            last = now
        time.sleep(min(30.0, max(0.2, left)))

    command = [
        sys.executable, str(supervisor),
        "--since", "1794909600",
        "--until", "2026-11-17T18:00:00-05:00",
        "--checks", CHECKS,
        "--interval", "900",
        "--min-trades", "20",
        "--balance-floor", "105.00",
        "--delta-cap", "0.05",
        "--log-file", str(log_file),
        "--state-file", str(state_file),
    ]
    log(f"target reached at {datetime.now().astimezone().isoformat(timespec='seconds')} "
        f"(wake_ts={time.time():.3f}, target_ts={target_ts:.3f}); exec {shlex.join(command)}")
    os.chdir(ROOT)
    os.execv(sys.executable, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
