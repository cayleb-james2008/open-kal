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
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.track2c_true_l2_settle import MIN_WINDOWS_REQUIRED, audit_l2_depth

LOG_FILE = ROOT / "data" / "overnight-rnd-loop.log"
EVOLVE_CACHE = Path("/home/cayleb/open-kal/data/evolve-cache")


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


def run_loop(single_run: bool = False) -> None:
    log("Starting Overnight 5-Minute True-L2 Ingestion & Evolution Loop")
    while True:
        try:
            audit = check_once()
            if audit["windows_with_l2"] >= MIN_WINDOWS_REQUIRED:
                log(f"PREREQUISITE MET: {audit['windows_with_l2']} windows >= {MIN_WINDOWS_REQUIRED}. Launching Track 2C True-L2 Evolution...")
                res = subprocess.run([sys.executable, str(ROOT / "scripts" / "track2c_true_l2_settle.py")],
                                     capture_output=True, text=True)
                log(f"Track 2C completed with exit code {res.returncode}:\n{res.stdout}\n{res.stderr}")
                break
        except Exception as e:
            log(f"Error during check cycle: {e}")

        if single_run:
            break

        # Sleep until the next 5-minute mark (or exactly 300s)
        time.sleep(300)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="Run single check and exit")
    args = ap.parse_args()
    run_loop(single_run=args.once)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
