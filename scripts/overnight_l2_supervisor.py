#!/usr/bin/env python3
"""Five-minute overnight supervision for the paper L2 build.

Three cadences, nothing else:

  * **every 5 minutes** -- is `run.py` alive, is `daemon_247.py` alive, does
    `laya-serve` answer `/health`, is the engine still in paper mode, is the venue
    balance still $121.46 (a $0.00 delta), and how many new `ws_book` rows carry a
    real `yes_bid_sz`/`yes_ask_sz`/`depth` and how many `ticks` carry a Laya gate
    verdict. Written to `data/overnight-l2-progress.json`.
  * **every 30 minutes** -- `git fetch origin`, then read `origin/rnd/alpha-evolve`
    for R&D commits this night has not seen yet and append them to
    `data/evening-report.md`, updating `NEXT.json`. `daemon_247.py` owns the commit
    and push, so this process never writes to git.
  * **every 2 hours** -- export the L2 cache (`scripts/book_cache.py --force`).

The row counts are incremental: `ws_book` is a 26 GB tape and `ts` has no index,
so every query is `rowid > cursor`, which is a primary-key range scan. This process
never arms live, never promotes a strategy, never restarts the engine and never
touches `.env` or `keys/`.

    nohup setsid .venv/bin/python scripts/overnight_l2_supervisor.py \
        --until 2026-11-18T06:00:00-05:00 > data/overnight-l2-supervisor.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "openkal-prod.db"
PROGRESS = ROOT / "data" / "overnight-l2-progress.json"
REPORT = ROOT / "data" / "evening-report.md"
NEXT = ROOT / "NEXT.json"
LOG = ROOT / "data" / "overnight-l2-supervisor.log"
STATE_URL = "http://127.0.0.1:8787/api/state"
LAYA_HEALTH = "http://127.0.0.1:8000/health"
RND_BRANCH = "origin/rnd/alpha-evolve"
TARGET_BALANCE = Decimal("121.46")
HISTORY = 288                      # 24 hours of 5-minute checks


def now() -> datetime:
    return datetime.now(timezone.utc).astimezone()


def log(msg: str) -> None:
    line = f"{now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)


def http_json(url: str, timeout: float = 5.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as fh:
            return json.loads(fh.read().decode())
    except (urllib.error.URLError, OSError, ValueError):
        return None


def alive(pattern: str) -> bool:
    """True when a process whose command line contains `pattern` is running."""
    try:
        out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True,
                             timeout=15).stdout
    except (subprocess.SubprocessError, OSError):
        return False
    return bool(out.strip())


def max_rowid(table: str) -> int:
    """Current high-water rowid, so a fresh start counts from launch, not from zero."""
    try:
        con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=30)
        try:
            return int(con.execute(f"SELECT max(rowid) FROM {table}").fetchone()[0] or 0)
        finally:
            con.close()
    except (sqlite3.Error, OSError):
        return 0


def read_db(cursor: int, tick_cursor: int = 0) -> dict:
    """New L2 rows and gate verdicts since `cursor` (a ws_book rowid)."""
    out: dict = {"cursor": cursor, "book_rows": 0, "sized_rows": 0, "depth_rows": 0,
                 "ticks": 0, "gate_verdicts": 0, "gate_allow": 0, "error": None}
    try:
        con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=30)
        con.execute("PRAGMA busy_timeout=20000")
        try:
            row = con.execute(
                "SELECT count(*), sum(yes_bid_sz IS NOT NULL), sum(yes_ask_sz IS NOT NULL),"
                " sum(depth IS NOT NULL) FROM ws_book WHERE rowid > ?", (cursor,)).fetchone()
            out["book_rows"], out["sized_rows"], out["depth_rows"] = (
                int(row[0] or 0), int(row[1] or 0), int(row[2] or 0))
            out["cursor"] = int(con.execute("SELECT max(rowid) FROM ws_book").fetchone()[0] or 0)
            tick = con.execute(
                "SELECT count(*), sum(gate IS NOT NULL) FROM ticks WHERE rowid > ?",
                (tick_cursor,)).fetchone()
            out["ticks"], out["gate_verdicts"] = int(tick[0] or 0), int(tick[1] or 0)
            out["tick_cursor"] = int(con.execute("SELECT max(rowid) FROM ticks").fetchone()[0] or 0)
        finally:
            con.close()
    except (sqlite3.Error, OSError) as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def check(state: dict) -> dict:
    st = state.get("state") or {}
    balance = st.get("venue_balance")
    try:
        delta = None if balance is None else str(Decimal(str(balance)) - TARGET_BALANCE)
    except Exception:  # noqa: BLE001 - a malformed number is data, not a crash
        delta = None
    return {
        "ts": now().isoformat(timespec="seconds"),
        "engine": alive("run.py --port"),
        "daemon": alive("daemon_247.py"),
        "laya": http_json(LAYA_HEALTH) is not None,
        "mode": st.get("mode"),
        "live_armed": st.get("live_armed"),
        "balance": balance,
        "delta": delta,
    }


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=str))
    os.replace(tmp, path)


def rnd_discoveries(seen: str | None) -> list[str]:
    """Commits on origin/rnd/alpha-evolve that this night has not recorded yet."""
    try:
        subprocess.run(["git", "fetch", "origin", "--quiet"], cwd=ROOT,
                       capture_output=True, text=True, timeout=120)
        rev = subprocess.run(["git", "rev-parse", RND_BRANCH], cwd=ROOT,
                             capture_output=True, text=True, timeout=30)
        if rev.returncode != 0:
            return []
        head = rev.stdout.strip()
        if seen == head:
            return []
        rng = f"{seen}..{head}" if seen else f"-5..{head}"
        logr = subprocess.run(["git", "log", "--oneline", rng], cwd=ROOT,
                              capture_output=True, text=True, timeout=30)
        return [line for line in logr.stdout.splitlines() if line.strip()]
    except (subprocess.SubprocessError, OSError):
        return []


def append_report(progress: dict, commits: list[str], sha: str) -> None:
    stamp = now().strftime("%Y-%m-%d %H:%M %Z")
    lines = ["", f"## Overnight R&D sync — {stamp}", ""]
    lines += [f"- `{c}`" for c in commits] or ["- no new commits on `origin/rnd/alpha-evolve`"]
    lines += ["",
              f"- L2 tape so far: {progress['new_book_rows']:,} new `ws_book` rows, "
              f"{progress['new_sized_rows']:,} with both touch sizes, "
              f"{progress['new_depth_rows']:,} with a ladder; "
              f"{progress['gate_verdicts']:,} `ticks.gate` verdicts.",
              f"- Account: balance {progress['last_check']['balance']} "
              f"(delta {progress['last_check']['delta']}), mode "
              f"{progress['last_check']['mode']}, armed {progress['last_check']['live_armed']}.",
              ""]
    try:
        with REPORT.open("a") as fh:
            fh.write("\n".join(lines) + "\n")
    except OSError as exc:
        log(f"evening report append failed: {exc}")
        return
    try:
        doc = json.loads(NEXT.read_text())
    except (OSError, ValueError):
        doc = {"next": []}
    doc.setdefault("overnight_l2", {})
    doc["overnight_l2"].update({"updated_at": stamp, "rnd_head": sha,
                                "new_commits": len(commits)})
    write_json(NEXT, doc)
    log(f"evening report + NEXT.json updated: {len(commits)} R&D commit(s) at {sha[:8]}")


def export_cache() -> None:
    log("2-hour block: exporting the L2 cache")
    try:
        out = subprocess.run([str(ROOT / ".venv/bin/python"),
                              str(ROOT / "scripts" / "book_cache.py"), "--force"],
                             cwd=ROOT, capture_output=True, text=True, timeout=7200)
        log(f"cache export rc={out.returncode}: {(out.stdout or out.stderr).strip()[-300:]}")
    except (subprocess.SubprocessError, OSError) as exc:
        log(f"cache export failed: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", required=True, help="ISO local time to stop (e.g. 06:00)")
    ap.add_argument("--interval", type=float, default=300.0)
    ap.add_argument("--cache-every", type=float, default=7200.0)
    args = ap.parse_args()

    until = datetime.fromisoformat(args.until)
    if until < now():
        until = until + timedelta(days=1)

    prior = {}
    try:
        prior = json.loads(PROGRESS.read_text())
    except (OSError, ValueError):
        prior = {}
    # A resumable run continues from its cursor. A fresh one starts at the current
    # high-water mark: counting the 199M rows already on the tape is a full scan of
    # 26 GB, and "new rows this night" is the number that matters.
    cursor = int(prior.get("cursor", 0) or 0) or max_rowid("ws_book")
    tick_cursor = int(prior.get("tick_cursor", 0) or 0) or max_rowid("ticks")
    progress = {
        "started_at": prior.get("started_at") or now().isoformat(timespec="seconds"),
        "until": until.isoformat(),
        "cursor": cursor, "tick_cursor": tick_cursor,
        "new_book_rows": 0, "new_sized_rows": 0, "new_depth_rows": 0,
        "gate_verdicts": 0, "cycles": 0, "last_check": {}, "checks": [],
        "rnd_head": prior.get("rnd_head"),
        "last_cache_export": None, "errors": [],
    }
    log(f"overnight supervisor up: until {until.isoformat()}, every {args.interval:.0f}s, "
        f"cursor {cursor}")

    # First export two hours in, not at launch: a nightly start is not a cache hour.
    next_cache = time.time() + args.cache_every
    while now() < until:
        progress["cycles"] += 1
        state = {"state": http_json(STATE_URL)}
        last = check(state)
        db = read_db(cursor, tick_cursor)
        if db.get("error"):
            progress["errors"] = (progress["errors"] + [db["error"]])[-20:]
        cursor = db.get("cursor", cursor)
        tick_cursor = db.get("tick_cursor", tick_cursor)
        progress.update({
            "updated_at": last["ts"], "last_check": last,
            "cursor": cursor, "tick_cursor": tick_cursor,
            "new_book_rows": progress["new_book_rows"] + db.get("book_rows", 0),
            "new_sized_rows": progress["new_sized_rows"] + db.get("sized_rows", 0),
            "new_depth_rows": progress["new_depth_rows"] + db.get("depth_rows", 0),
            "gate_verdicts": progress["gate_verdicts"] + db.get("gate_verdicts", 0),
        })
        progress["checks"] = (progress["checks"] + [last])[-HISTORY:]
        write_json(PROGRESS, progress)
        log(f"check {last['ts']} engine={last['engine']} daemon={last['daemon']} "
            f"laya={last['laya']} mode={last['mode']} armed={last['live_armed']} "
            f"balance={last['balance']} delta={last['delta']} "
            f"+book={db.get('book_rows', 0)} +sized={db.get('sized_rows', 0)} "
            f"+gates={db.get('gate_verdicts', 0)}")

        # :00 and :30 -- R&D discoveries, appended for daemon_247.py to sync.
        if now().minute % 30 < args.interval / 60.0:
            rev = subprocess.run(["git", "rev-parse", RND_BRANCH], cwd=ROOT,
                                 capture_output=True, text=True, timeout=30).stdout.strip()
            commits = rnd_discoveries(progress.get("rnd_head"))
            if commits or (rev and rev != progress.get("rnd_head")):
                progress["rnd_head"] = rev or progress.get("rnd_head")
                append_report(progress, commits, rev or "")
                write_json(PROGRESS, progress)

        # every 2 hours -- the L2 cache export.
        if time.time() >= next_cache:
            export_cache()
            progress["last_cache_export"] = now().isoformat(timespec="seconds")
            next_cache = time.time() + args.cache_every
            write_json(PROGRESS, progress)

        # Sleep to the next 5-minute boundary, never past the stop time.
        target = min(until.timestamp(), time.time() + args.interval)
        time.sleep(max(1.0, target - time.time()))
    log("overnight supervisor finished its window")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
