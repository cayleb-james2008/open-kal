#!/usr/bin/env python3
"""Refresh settled markets and refuse to continue without the latest close.

The noon rebuild is a single shot. `evolve.py refresh-markets` is therefore not
enough on its own: at 12:00:05 Kalshi may not yet have returned the
11:45-12:00 EST contract as settled. This script performs the refresh and then
polls, within a hard wall-clock bound, until every requested lane has a settled
market whose close_time is the most recent 15-minute boundary that has passed.

It exits 0 only when that window is present. On timeout it prints the missing
lanes and exits 1, so rebuild_cache.sh stops before it writes a stale cache.
It never touches the production database or the forward OOS scorer.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openkal.config import SLOTS  # noqa: E402

DEFAULT_FILE = ROOT / "data" / "settled-markets.json"
# The recorder's lane set, not the wider evolve.py metadata universe: KXDOGE15M
# is explicitly not a lane in config.SLOTS, even though refresh-markets fetches it.
DEFAULT_LANES = tuple(slot.series for slot in SLOTS)
PY = ROOT / ".venv" / "bin" / "python"


def observed_window_seconds(path: Path, default: int = 900) -> int:
    """Infer the settlement cadence from actual close_time intervals.

    The settled file carries five series at the same close times; dedupe before
    taking the modal gap. A missing/empty file falls back to the observed
    15-minute cadence only so a first bootstrap can still run.
    """
    closes = sorted({value for value in
                     (_close_epoch(m.get("close_time")) for m in read_markets(path))
                     if value is not None})
    gaps = [b - a for a, b in zip(closes, closes[1:]) if b > a]
    return Counter(gaps).most_common(1)[0][0] if gaps else default


def latest_closed_epoch(now: float | None = None, window: int = 900) -> int:
    """Most recent settlement boundary that has already passed."""
    return int((time.time() if now is None else now) // window * window)


def _close_epoch(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp())
    except (TypeError, ValueError):
        return None


def missing_lanes(markets: list[dict], lanes: tuple[str, ...],
                  target_epoch: int) -> list[str]:
    have = set()
    for market in markets:
        ticker = str(market.get("ticker") or "")
        # Real shape: KXBTC15M-26NOV171200-15. Match the series segment exactly;
        # do not let a longer sibling series satisfy a shorter prefix.
        series = ticker.split("-", 1)[0]
        lane = next((name for name in lanes if series == name), None)
        if lane and _close_epoch(market.get("close_time")) == target_epoch:
            have.add(lane)
    return [lane for lane in lanes if lane not in have]


def read_markets(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def refresh_once(pages: int, timeout: float) -> tuple[int, str]:
    """Refresh the real settled file through the existing evolve CLI."""
    cmd = [str(PY), str(ROOT / "scripts" / "evolve.py"),
           "refresh-markets", str(pages)]
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                              timeout=timeout)
        return proc.returncode, ((proc.stdout or "") + (proc.stderr or "")).strip()
    except subprocess.TimeoutExpired as exc:
        return 124, f"refresh subprocess timeout after {timeout:.1f}s: {exc}"
    except OSError as exc:
        return 1, f"refresh subprocess failed: {type(exc).__name__}: {exc}"


def wait_for_latest(path: Path, lanes: tuple[str, ...], target_epoch: int,
                    timeout: float, poll: float,
                    refresh) -> tuple[bool, list[str]]:
    """Poll until all lanes have `target_epoch`, or fail at the deadline.

    `refresh` takes the remaining seconds and returns (rc, output). It is
    injectable so the success and shortfall paths can be exercised with a temp
    settled file and a stub, without a venue call.
    """
    deadline = time.monotonic() + timeout
    target_iso = datetime.fromtimestamp(target_epoch, timezone.utc).isoformat()
    attempt = 0
    while True:
        missing = missing_lanes(read_markets(path), lanes, target_epoch)
        if not missing:
            print(f"market refresh: {target_iso} present on all lanes: "
                  f"{', '.join(lanes)}")
            return True, []
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(f"MARKET REFRESH SHORTFALL: latest closed {target_iso} missing on "
                  f"{', '.join(missing)} after {timeout:.0f}s; refusing cache build",
                  file=sys.stderr)
            return False, missing
        attempt += 1
        print(f"market refresh attempt {attempt}: missing {', '.join(missing)}; "
              f"fetching (remaining {remaining:.1f}s)")
        rc, out = refresh(remaining)
        last = out.strip().splitlines()[-1] if out.strip() else ""
        print(f"market refresh fetch rc={rc}: {last}")
        missing = missing_lanes(read_markets(path), lanes, target_epoch)
        if not missing:
            print(f"market refresh: {target_iso} present on all lanes: "
                  f"{', '.join(lanes)}")
            return True, []
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(f"MARKET REFRESH SHORTFALL: latest closed {target_iso} missing on "
                  f"{', '.join(missing)} after {timeout:.0f}s; refusing cache build",
                  file=sys.stderr)
            return False, missing
        time.sleep(min(poll, remaining))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lanes", default=",".join(DEFAULT_LANES))
    ap.add_argument("--pages", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=180.0,
                    help="hard bound for the complete refresh-and-wait stage")
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--as-of", type=float, default=None,
                    help="epoch to treat as now; test only")
    args = ap.parse_args()
    lanes = tuple(x.strip() for x in args.lanes.split(",") if x.strip())
    window = observed_window_seconds(DEFAULT_FILE)
    target = latest_closed_epoch(args.as_of, window)
    target_iso = datetime.fromtimestamp(target, timezone.utc).isoformat()
    print(f"market refresh window {window}s; target latest closed {target_iso}; "
          f"lanes {', '.join(lanes)}; bound {args.timeout:.0f}s")
    ok, missing = wait_for_latest(
        DEFAULT_FILE, lanes, target, args.timeout, args.poll,
        lambda remaining: refresh_once(args.pages, remaining),
    )
    if not ok:
        print(f"MARKET REFRESH SHORTFALL: missing {', '.join(missing)}; "
              f"refusing to rebuild data/evolve-cache", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
