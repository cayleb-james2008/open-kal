#!/usr/bin/env python3
"""Overnight Recurring 5-Minute True-L2 Ingestion & Evolution Loop.

Schedule:
  - Every 5-min block (:00, :05, :10, ...):
    Check /home/cayleb/open-kal/data/evolve-cache/book_KXBTC15M.npz (mode=ro)
    for non-NaN yes_bid_sz and yes_ask_sz rows exported by Session 1's 2-hourly cache rebuilds.
  - Once >= 24 settled windows (>= 6 hours) of true L2 depth (yes_bid_sz, yes_ask_sz, microprice)
    are present:
    Run scripts/track2c_true_l2_settle.py (chronological 60% IS / 40% OOS),
    score the single sealed OOS look to data/oos-ledger.jsonl,
    update RND_REPORT.md + NEXT.json, and push all commits to origin/rnd/alpha-evolve.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.track2c_true_l2_settle import EVOLVE_CACHE, MIN_WINDOWS_REQUIRED, audit_l2_depth

LOG_FILE = ROOT / "data" / "overnight-rnd-loop.log"
INTERVAL_S = 300


def log(msg: str) -> None:
    now_str = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    line = f"[{now_str}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a") as f:
        f.write(line + "\n")


def check_once() -> dict:
    audit = audit_l2_depth(EVOLVE_CACHE)
    log(f"Cache check: status={audit['status']}, valid_l2_rows={audit.get('valid_l2_rows', 0)}, windows_with_l2={audit['windows_with_l2']}/{MIN_WINDOWS_REQUIRED}")
    return audit


def run_loop(*, single_run=False, check_only=False, publish=False, until=float("inf")):
    now = time.time()
    if now > until:
        log("Shift deadline already passed; did not read the cache.")
        return
    log(f"Starting 300s true-L2 checks (check_only={check_only}, publish={publish}).")
    slot = int(now // INTERVAL_S) * INTERVAL_S
    child = latest = None
    finished = False

    def read_child(rc):
        log(f"Track 2C exited {rc}" + ("; deferring to next slot" if rc == 75 else ""))
        return rc != 75

    while slot <= until:
        delay = slot - time.time()
        if delay > 0:
            time.sleep(delay)
        current = int(time.time() // INTERVAL_S) * INTERVAL_S
        if current > slot:
            log(f"Missed {(current-slot)//INTERVAL_S} wall-clock slots; no backfill.")
            slot = current
            if slot > until:
                break
        if child is not None and (rc := child.poll()) is not None:
            finished, child = read_child(rc), None
        try:
            audit = check_once()
            if audit["status"] in ("accumulating", "ready", "zero_valid_l2_rows"):
                latest = f"{audit['windows_with_l2']}/{MIN_WINDOWS_REQUIRED} at " + \
                    datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
            if not check_only and not finished and child is None and audit["status"] == "ready":
                command = [sys.executable, "-B", str(ROOT / "scripts" / "track2c_true_l2_settle.py")]
                if publish:
                    command.append("--publish")
                child = subprocess.Popen(command, cwd=ROOT)
                log(f"Prerequisite met ({audit['windows_with_l2']} windows); started Track 2C PID {child.pid}.")
        except Exception as e:
            log(f"Error during check cycle: {e}")

        if single_run:
            break
        slot += INTERVAL_S
    if child is not None:
        read_child(child.wait(timeout=900))
    log(f"Check span ended. Last successful coverage: {latest or 'none'}.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="Run single check and exit")
    ap.add_argument("--check-only", action="store_true", help="Check depth without starting a sweep")
    ap.add_argument("--publish", action="store_true", help="Commit and push returned Track 2C results")
    ap.add_argument("--until", help="Inclusive final 5-minute slot, ISO 8601 with timezone")
    args = ap.parse_args()
    until = float("inf")
    if args.until:
        deadline = datetime.datetime.fromisoformat(args.until)
        if deadline.tzinfo is None or deadline.timestamp() % INTERVAL_S:
            ap.error("--until must include a timezone and fall on a five-minute boundary")
        until = deadline.timestamp()
    with (ROOT / "data/overnight-rnd-loop.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_loop(single_run=args.once, check_only=args.check_only, publish=args.publish, until=until)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
