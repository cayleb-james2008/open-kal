#!/usr/bin/env python3
"""Five-minute overnight supervision for the paper L2 build.

Three cadences, nothing else:

  * **every 5 minutes** -- is `run.py` alive, is `daemon_247.py` alive, does
    `laya-serve` answer `/health`, is the engine still in paper mode, is the venue
    balance still $121.46 (a $0.00 delta), and how many new `ws_book` rows carry a
    real `yes_bid_sz`/`yes_ask_sz`/`depth` and how many `ticks` carry a Laya gate
    verdict. Every cycle is appended in full to `data/overnight-l2-progress.json`
    (`cycle_log`), with the cumulative counters next to it.
  * **every 30 minutes** -- `git fetch origin`, then read `origin/rnd/alpha-evolve`
    for R&D commits this night has not seen yet and append them to
    `data/evening-report.md`, updating `NEXT.json`. `daemon_247.py` owns the commit
    and push, so this process never writes to git.
  * **every 2 hours** -- export the L2 cache (`scripts/book_cache.py --force`).

The row counts are incremental: `ws_book` is a 26 GB tape and `ts` has no index,
so every query is `rowid > cursor`, which is a primary-key range scan. A fresh
start begins at the current high-water mark; "new rows" means rows this run saw.

A cycle never raises out of the loop. A failed check is recorded in the cycle
record and in `errors`, one line is logged, and the next cycle proceeds: an
unattended night cannot die because one `curl` timed out.

This process never arms live, never promotes a strategy, never restarts the engine
and never touches `.env` or `keys/`.

    .venv/bin/python scripts/overnight_l2_supervisor.py \
        --until 2026-11-18T06:00:00-05:00

Verification flags (`--cycles`, `--rnd-now`, `--rnd-every`, `--cache-now`,
`--cache-only`) exist so the branches can be exercised on demand without waiting
for the clock.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
import traceback
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
RND_LEDGER = ROOT / "data" / "overnight-rnd.json"
STATE_URL = "http://127.0.0.1:8787/api/state"
LAYA_HEALTH = "http://127.0.0.1:8000/health"
RND_BRANCH = "origin/rnd/alpha-evolve"
TARGET_BALANCE = Decimal("121.46")
HISTORY = 288                      # 24 hours of 5-minute cycles


def now() -> datetime:
    return datetime.now(timezone.utc).astimezone()


def log(msg: str) -> None:
    print(f"{now().isoformat(timespec='seconds')} {msg}", flush=True)


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


def read_db(cursor: int, tick_cursor: int) -> dict:
    """New L2 rows and gate verdicts since the cursors."""
    out: dict = {"cursor": cursor, "tick_cursor": tick_cursor, "book_rows": 0,
                 "sized_rows": 0, "depth_rows": 0, "ticks": 0, "gate_verdicts": 0,
                 "error": None}
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


def check_state() -> dict:
    st = (http_json(STATE_URL) or {})
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


def git(*args: str, timeout: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          timeout=timeout)


def rnd_discoveries(seen: str | None) -> dict:
    """Fetch origin, then report what is new on `origin/rnd/alpha-evolve`.

    Never raises: a missing remote branch, an offline network or a failed fetch all
    resolve to "nothing new" with the reason attached, which is what an unattended
    night needs. Returns {head, commits, note}.
    """
    try:
        fetch = git("fetch", "origin", "--quiet")
        rev = git("rev-parse", RND_BRANCH, timeout=30)
        if rev.returncode != 0:
            return {"head": None, "commits": [],
                    "note": f"branch {RND_BRANCH} not found (fetch rc={fetch.returncode})"}
        head = rev.stdout.strip()
        if seen == head:
            return {"head": head, "commits": [], "note": "no new commits since last check"}
        rng = f"{seen}..{head}" if seen else "-5..HEAD"
        logr = git("log", "--oneline", rng, timeout=30)
        commits = [line for line in logr.stdout.splitlines() if line.strip()]
        return {"head": head, "commits": commits,
                "note": f"{len(commits)} new commit(s)" if commits else "head moved, no new rows"}
    except (subprocess.SubprocessError, OSError) as exc:
        return {"head": None, "commits": [], "note": f"fetch failed: {type(exc).__name__}"}


def append_report(progress: dict, commits: list[str], sha: str) -> None:
    stamp = now().strftime("%Y-%m-%d %H:%M %Z")
    lines = ["", f"## Overnight R&D sync — {stamp}", ""]
    lines += [f"- `{c}`" for c in commits] or ["- no new commits on `origin/rnd/alpha-evolve`"]
    last = progress.get("last_check") or {}
    lines += ["",
              f"- L2 tape so far: {progress['new_book_rows']:,} new `ws_book` rows, "
              f"{progress['new_sized_rows']:,} with both touch sizes, "
              f"{progress['new_depth_rows']:,} with a ladder; "
              f"{progress['gate_verdicts']:,} `ticks.gate` verdicts.",
              f"- Account: balance {last.get('balance')} (delta {last.get('delta')}), "
              f"mode {last.get('mode')}, armed {last.get('live_armed')}.",
              ""]
    try:
        with REPORT.open("a") as fh:
            fh.write("\n".join(lines) + "\n")
    except OSError as exc:
        progress["errors"] = (progress.get("errors") or [])[-19:] + [f"report append: {exc}"]
        log(f"evening report append failed: {exc}")
        return
    try:
        doc = json.loads(NEXT.read_text())
    except (OSError, ValueError):
        doc = {}
    doc.setdefault("overnight_l2", {})
    doc["overnight_l2"].update({"updated_at": stamp, "rnd_head": sha,
                                "new_commits": len(commits)})
    write_json(NEXT, doc)
    # `daemon_247.py` regenerates the report and NEXT.json from its own state every
    # fifteen minutes, so an append there is fleeting. The ledger is the durable
    # record of what this loop actually saw on origin/rnd/alpha-evolve.
    try:
        ledger = json.loads(RND_LEDGER.read_text())
    except (OSError, ValueError):
        ledger = {"checks": []}
    ledger["updated_at"] = stamp
    ledger["head"] = sha
    ledger["checks"] = (ledger.get("checks") or [])[-40:] + [{
        "at": stamp, "head": sha, "new_commits": len(commits), "commits": commits,
    }]
    write_json(RND_LEDGER, ledger)
    log(f"evening report + NEXT.json updated: {len(commits)} R&D commit(s) at {sha[:8]} "
        f"(ledger {RND_LEDGER.name})")


CACHE_LOG = ROOT / "data" / "book-cache.log"


def start_cache_export(lanes: list[str], limit: int = 0,
                       out: str = "") -> subprocess.Popen | None:
    """Start the cache export in its own process.

    A full export of the 26 GB tape takes minutes; running it inline stopped the
    supervision loop for the whole export, so a night's monitoring would have a
    six-minute hole in it -- and a hung export would end the loop outright. It runs
    detached instead, and each cycle polls it.
    """
    cmd = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts" / "book_cache.py"),
           "--force", *(["--out", out] if out else []),
           *(["--limit", str(limit)] if limit else []), *lanes]
    log(f"2-hour block: starting the L2 cache export ({', '.join(lanes) or 'all lanes'})")
    try:
        with CACHE_LOG.open("a") as fh:
            fh.write(f"\n===== {now().isoformat(timespec='seconds')} export "
                     f"{' '.join(lanes) or 'all'} =====\n")
            return subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                    start_new_session=True)
    except OSError as exc:
        log(f"cache export failed to start: {type(exc).__name__}: {exc}")
        return None


def finish_cache_export(proc: subprocess.Popen, progress: dict, started: float,
                        cache_dir: Path) -> bool:
    rc = proc.poll()
    if rc is None:
        return False
    files = {path.name: digest(path)
             for path in sorted(cache_dir.glob("book_*.npz"))[:4]}
    progress["cache"] = {
        "started_at": progress.get("cache", {}).get("started_at"),
        "finished_at": now().isoformat(timespec="seconds"),
        "duration_s": round(time.time() - started, 1),
        "rc": rc,
        "dir": str(cache_dir),
        "files": files,
    }
    progress["last_cache_export"] = progress["cache"]["finished_at"]
    progress["cache_ok"] = rc == 0
    # A completed export is evidence, so it is appended to a history instead of only
    # living in the key the next export overwrites.
    progress["cache_history"] = (progress.get("cache_history") or [])[-8:] + [
        {k: progress["cache"].get(k) for k in
         ("started_at", "finished_at", "duration_s", "rc", "dir", "files")}]
    log(f"cache export finished rc={rc} in {progress['cache']['duration_s']}s "
        f"({len(progress['cache'].get('files') or {})} npz file(s))")
    if rc != 0:
        detail = CACHE_LOG.read_text(errors="ignore").strip().splitlines()[-3:] if CACHE_LOG.exists() else []
        for line in detail:
            log(f"  cache log: {line}")
    return True


def digest(path: Path) -> dict:
    try:
        import hashlib
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return {"bytes": path.stat().st_size, "sha256": h.hexdigest()[:16]}
    except OSError:
        return {"ok": False, "error": "unreadable"}


def run_cycle(progress: dict, state: dict) -> dict:
    """One 5-minute cycle. Never raises; failures land in the record."""
    started = now()
    errors: list[str] = []
    db = read_db(int(state.get("cursor") or 0), int(state.get("tick_cursor") or 0))
    if db.get("error"):
        errors.append(db["error"])
    try:
        last = check_state()
    except Exception as exc:  # noqa: BLE001 - a cycle must not die on one read
        errors.append(f"state: {type(exc).__name__}: {exc}")
        last = {"ts": now().isoformat(timespec="seconds"), "engine": False,
                "daemon": False, "laya": False, "mode": None, "live_armed": None,
                "balance": None, "delta": None}
    state["cursor"] = db.get("cursor", state.get("cursor"))
    state["tick_cursor"] = db.get("tick_cursor", state.get("tick_cursor"))
    cumulative = {
        "new_book_rows": progress["new_book_rows"] + db.get("book_rows", 0),
        "new_sized_rows": progress["new_sized_rows"] + db.get("sized_rows", 0),
        "new_depth_rows": progress["new_depth_rows"] + db.get("depth_rows", 0),
        "gate_verdicts": progress["gate_verdicts"] + db.get("gate_verdicts", 0),
    }
    progress.update(cumulative)
    record = {
        "cycle": progress["cycles"] + 1,
        "started_at": started.isoformat(timespec="seconds"),
        "ended_at": now().isoformat(timespec="seconds"),
        "duration_s": round((now() - started).total_seconds(), 2),
        **last,
        "book_rows": db.get("book_rows", 0),
        "sized_rows": db.get("sized_rows", 0),
        "depth_rows": db.get("depth_rows", 0),
        "ticks": db.get("ticks", 0),
        "gate_verdicts_this_cycle": db.get("gate_verdicts", 0),
        "cumulative": cumulative,
        "cursor": state.get("cursor"),
        "errors": errors,
    }
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", default=None, help="ISO local time to stop (e.g. 06:00)")
    ap.add_argument("--interval", type=float, default=300.0)
    ap.add_argument("--cycles", type=int, default=0, help="stop after N cycles (0 = until)")
    ap.add_argument("--rnd-every", type=float, default=1800.0)
    ap.add_argument("--rnd-now", action="store_true", help="run the R&D ingest immediately")
    ap.add_argument("--cache-every", type=float, default=7200.0)
    ap.add_argument("--cache-now", action="store_true", help="export the cache this cycle")
    ap.add_argument("--cache-lanes", default="",
                    help="space/comma separated lanes to export (default: all)")
    ap.add_argument("--cache-limit", type=int, default=0,
                    help="tickers per lane for a bounded proof export (0 = all)")
    ap.add_argument("--cache-out", default="",
                    help="cache directory for a proof export (default: the live cache)")
    ap.add_argument("--progress", default=str(PROGRESS),
                    help="progress file (default: data/overnight-l2-progress.json)")
    ap.add_argument("--cache-timeout", type=float, default=7200.0)
    ap.add_argument("--cache-only", action="store_true",
                    help="run one cache export, wait for it, log it, and exit")
    args = ap.parse_args()
    cache_lanes = [tok for tok in args.cache_lanes.replace(",", " ").split() if tok]
    cache_dir = Path(args.cache_out) if args.cache_out else ROOT / "data" / "evolve-cache"

    if args.cache_only:
        proc = start_cache_export(cache_lanes, args.cache_limit, args.cache_out)
        started = time.time()
        while proc is not None and proc.poll() is None and time.time() - started < args.cache_timeout:
            time.sleep(2)
        if proc is None:
            return 1
        if proc.poll() is None:
            proc.terminate()
            log("cache-only run timed out")
            return 1
        log(f"cache-only run finished rc={proc.returncode} "
            f"in {time.time() - started:.0f}s")
        return 0 if proc.returncode == 0 else 1

    until = datetime.fromisoformat(args.until) if args.until else None
    if until is not None and until < now():
        until = until + timedelta(days=1)

    progress_path = Path(args.progress)
    prior: dict = {}
    try:
        prior = json.loads(progress_path.read_text())
    except (OSError, ValueError):
        prior = {}
    # A resumable run continues from its cursors. A fresh one starts at the current
    # high-water mark: counting the 199M rows already on the tape is a full scan of
    # 26 GB, and "new rows this night" is the number that matters.
    cursor = int(prior.get("cursor", 0) or 0) or max_rowid("ws_book")
    tick_cursor = int(prior.get("tick_cursor", 0) or 0) or max_rowid("ticks")
    state = {"cursor": cursor, "tick_cursor": tick_cursor}
    progress = {
        "started_at": prior.get("started_at") or now().isoformat(timespec="seconds"),
        "until": until.isoformat() if until else None,
        "interval_s": args.interval,
        "cursor": cursor, "tick_cursor": tick_cursor,
        "new_book_rows": int(prior.get("new_book_rows", 0) or 0),
        "new_sized_rows": int(prior.get("new_sized_rows", 0) or 0),
        "new_depth_rows": int(prior.get("new_depth_rows", 0) or 0),
        "gate_verdicts": int(prior.get("gate_verdicts", 0) or 0),
        "cycles": int(prior.get("cycles", 0) or 0),
        "last_check": prior.get("last_check") or {},
        "checks": (prior.get("checks") or [])[-HISTORY:],
        "cycle_log": (prior.get("cycle_log") or [])[-HISTORY:],
        "rnd_head": prior.get("rnd_head"),
        "last_rnd": prior.get("last_rnd"),
        "last_cache_export": prior.get("last_cache_export"),
        "cache_ok": prior.get("cache_ok"),
        # Carried across a restart: a resumed run used to drop the export record it
        # had already earned, leaving rc=None in the night's evidence.
        "cache": prior.get("cache") or {},
        "cache_history": prior.get("cache_history") or [],
        "errors": (prior.get("errors") or [])[-20:],
    }
    log(f"overnight supervisor up: until {progress['until']}, every {args.interval:.0f}s, "
        f"cursor {cursor}, resumed={bool(prior)}")

    next_rnd = time.time() + (0 if args.rnd_now else min(
        args.rnd_every, 1800 - (now().minute % 30) * 60))
    next_cache = time.time() + (0 if args.cache_now else args.cache_every)
    cache_proc: subprocess.Popen | None = None
    cache_started = 0.0
    # `--cycles` counts this run's cycles, not the lifetime counter: a resumed run
    # that inherited cycles=1 used to stop after its first cycle.
    run_cycles = 0

    while True:
        started = time.time()
        try:
            record = run_cycle(progress, state)
            progress["cycles"] = record["cycle"]
            progress["last_check"] = {k: record[k] for k in
                                      ("ts", "engine", "daemon", "laya", "mode",
                                       "live_armed", "balance", "delta")}
            progress["cursor"], progress["tick_cursor"] = record["cursor"], state["tick_cursor"]
            progress["cycle_log"] = (progress["cycle_log"] + [record])[-HISTORY:]
            progress["checks"] = (progress["checks"]
                                  + [{k: record[k] for k in
                                      ("ts", "engine", "daemon", "laya", "mode",
                                       "live_armed", "balance", "delta")}])[-HISTORY:]
            progress["errors"] = (progress["errors"] + record["errors"])[-20:]
            progress["updated_at"] = record["ended_at"]
            log(f"cycle {record['cycle']} {record['started_at']} "
                f"engine={record['engine']} daemon={record['daemon']} "
                f"laya={record['laya']} mode={record['mode']} "
                f"armed={record['live_armed']} balance={record['balance']} "
                f"delta={record['delta']} +book={record['book_rows']} "
                f"+sized={record['sized_rows']} +gates={record['gate_verdicts_this_cycle']} "
                f"cum_sized={record['cumulative']['new_sized_rows']} "
                f"cum_gates={record['cumulative']['gate_verdicts']}"
                + (f" errors={record['errors']}" if record["errors"] else ""))
            write_json(progress_path, progress)
        except Exception as exc:  # noqa: BLE001 - containment is the point
            detail = f"cycle crashed: {type(exc).__name__}: {exc}"
            progress["errors"] = (progress["errors"] + [detail])[-20:]
            log(detail)
            log(traceback.format_exc(limit=1).strip().replace("\n", " | "))
            write_json(progress_path, progress)

        if time.time() >= next_rnd:
            info = rnd_discoveries(progress.get("rnd_head"))
            log(f"rnd: head={info['head']} {info['note']}")
            if info["head"] and (info["head"] != progress.get("rnd_head")
                                 or info["commits"]):
                progress["rnd_head"] = info["head"]
                append_report(progress, info["commits"], info["head"])
            progress["last_rnd"] = {"ts": now().isoformat(timespec="seconds"),
                                    "head": info["head"], "new_commits": len(info["commits"]),
                                    "note": info["note"]}
            write_json(progress_path, progress)
            next_rnd = time.time() + args.rnd_every

        if cache_proc is not None:
            # Detached: polled every cycle, so supervision keeps its cadence while a
            # six-minute export runs beside it.
            if finish_cache_export(cache_proc, progress, cache_started, cache_dir):
                write_json(progress_path, progress)
                cache_proc = None
            elif time.time() - cache_started > args.cache_timeout:
                cache_proc.terminate()
                log(f"cache export exceeded {args.cache_timeout:.0f}s; terminated")
                progress["cache"] = {"finished_at": now().isoformat(timespec="seconds"),
                                    "rc": None, "note": "timeout"}
                progress["cache_ok"] = False
                write_json(progress_path, progress)
                cache_proc = None

        if time.time() >= next_cache:
            if cache_proc is not None:
                log("cache export still running; not starting a second one")
            else:
                progress["cache"] = {"started_at": now().isoformat(timespec="seconds"),
                                     "lanes": cache_lanes or "all"}
                cache_proc = start_cache_export(cache_lanes, args.cache_limit,
                                                args.cache_out)
                cache_started = time.time()
                write_json(progress_path, progress)
            next_cache = time.time() + args.cache_every

        run_cycles += 1
        if args.cycles and run_cycles >= args.cycles:
            log(f"stopping after {args.cycles} cycle(s) as requested")
            break
        if until is not None and now() >= until:
            log("overnight supervisor finished its window")
            break

        # Sleep to the next interval boundary, never past the stop time -- while
        # polling the detached export, so its completion (and true duration) is
        # recorded when it happens rather than up to a cycle later.
        target = started + args.interval
        if until is not None:
            target = min(target, until.timestamp())
        while True:
            remaining = target - time.time()
            if remaining <= 0:
                break
            time.sleep(min(5.0, remaining))
            if cache_proc is not None and cache_proc.poll() is not None:
                finish_cache_export(cache_proc, progress, cache_started, cache_dir)
                write_json(progress_path, progress)
                cache_proc = None

    # Never leave an export orphaned: its result is part of the night's evidence, so
    # the last thing this process does is record how it finished.
    if cache_proc is not None and cache_proc.poll() is None:
        log("waiting for the running cache export to finish before exit")
        while cache_proc.poll() is None and time.time() - cache_started < args.cache_timeout:
            time.sleep(5)
        if cache_proc.poll() is None:
            cache_proc.terminate()
            log(f"cache export exceeded {args.cache_timeout:.0f}s; terminated")
            cache_proc = None
        else:
            finish_cache_export(cache_proc, progress, cache_started, cache_dir)
        write_json(progress_path, progress)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
