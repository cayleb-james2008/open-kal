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
    `data/evening-report.md`, updating `NEXT.json`. Because `daemon_247.py`
    regenerates both of those wholesale every fifteen minutes, the durable record of
    the same check goes to `docs/overnight-rnd.json`, which the daemon owns neither
    of and which is inside a path it commits. The daemon keeps the git contract.
  * **every 2 hours** -- export the L2 cache (`scripts/book_cache.py --force`).

The export runs detached, and it is tracked durably: its pid, start time and rc file
are written into the progress file before it starts, so a supervisor that is
restarted mid-export either **adopts** the running child and records its completion,
or **abandons** it with a recorded reason. Two exports never run at once: the
supervisor refuses to start one while any `book_cache.py --force` is alive (the
daemon's own rebuild runs the same script, and stacking on it would race the same
`.tmp.npz` paths).

The row counts are incremental: `ws_book` is a 26 GB tape and `ts` has no index, so
every query is `rowid > cursor`, which is a primary-key range scan. A fresh start
begins at the current high-water mark.

A cycle never raises out of the loop. A failed check is recorded in the cycle record
and in `errors`, one line is logged, and the next cycle proceeds.

This process never arms live, never promotes a strategy, never restarts the engine
and never touches `.env` or `keys/`.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
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
RND_LEDGER = ROOT / "docs" / "overnight-rnd.json"
RND_LEDGER_LEGACY = ROOT / "data" / "overnight-rnd.json"
CACHE_RC = ROOT / "data" / "cache-export.rc"
CACHE_LOG = ROOT / "data" / "book-cache.log"
LIVE_CACHE = ROOT / "data" / "evolve-cache"
STATE_URL = "http://127.0.0.1:8787/api/state"
LAYA_HEALTH = "http://127.0.0.1:8000/health"
RND_BRANCH = "origin/rnd/alpha-evolve"
TARGET_BALANCE = Decimal("121.46")
HISTORY = 288                      # 24 hours of 5-minute cycles


def now() -> datetime:
    return datetime.now(timezone.utc).astimezone()


def parse_until(raw: str | None) -> datetime | None:
    """`--until` as an aware local deadline; a bare `08:00` means this box's 08:00.

    `datetime.fromisoformat` returns a naive datetime when the operator leaves the
    UTC offset off, and comparing that with `now()` raised `TypeError: can't
    compare offset-naive and offset-aware datetimes` before the loop ever started.
    `naive.astimezone()` reads a naive datetime as local time, which is what a bare
    wall-clock argument means here.
    """
    if not raw:
        return None
    until = datetime.fromisoformat(raw)
    if until.tzinfo is None:
        until = until.astimezone()
    if until < now():
        until = until + timedelta(days=1)
    return until


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
    # Keys starting with `_` are process-local plumbing (the progress path itself).
    clean = {k: v for k, v in payload.items() if not k.startswith("_")}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean, indent=1, default=str))
    os.replace(tmp, path)


def git(*args: str, timeout: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          timeout=timeout)


# ------------------------------------------------------------------ R&D ingest
def rnd_discoveries(seen: str | None) -> dict:
    """Fetch origin, then report what is new on `origin/rnd/alpha-evolve`.

    Never raises: a missing remote branch, an offline network or a failed fetch all
    resolve to "nothing new" with the reason attached.
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
    try:
        doc = json.loads(NEXT.read_text())
    except (OSError, ValueError):
        doc = {}
    doc.setdefault("overnight_l2", {})
    doc["overnight_l2"].update({"updated_at": stamp, "rnd_head": sha,
                                "new_commits": len(commits)})
    try:
        write_json(NEXT, doc)
    except OSError as exc:
        log(f"NEXT.json update failed: {exc}")
    log(f"evening report + NEXT.json updated: {len(commits)} R&D commit(s) at {sha[:8]}")


def record_rnd(progress: dict, info: dict) -> None:
    """Append one R&D check to the durable ledger, no-op included.

    `daemon_247.py` rewrites the report and NEXT.json from its own state every
    fifteen minutes, so the only snapshot that survives a regeneration is one it
    does not own. docs/ is inside its commit allowlist, so this file also reaches
    origin/master without the daemon being touched. A clean no-op is recorded too:
    "the fetch ran and found nothing" is evidence, not an absence of it.
    """
    stamp = now().strftime("%Y-%m-%d %H:%M %Z")
    try:
        ledger = json.loads(RND_LEDGER.read_text())
    except (OSError, ValueError):
        ledger = {"checks": []}
    ledger["updated_at"] = stamp
    ledger["head"] = info.get("head")
    ledger["branch"] = RND_BRANCH
    ledger["checks"] = (ledger.get("checks") or [])[-40:] + [{
        "at": stamp,
        "head": info.get("head"),
        "new_commits": len(info.get("commits") or []),
        "commits": info.get("commits") or [],
        "note": info.get("note"),
    }]
    try:
        RND_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        write_json(RND_LEDGER, ledger)
    except OSError as exc:
        log(f"R&D ledger write failed: {exc}")


# --------------------------------------------------------------- cache export
def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (TypeError, ValueError, OSError):
        return False
    return True


def export_pids() -> list[int]:
    """Every running `book_cache.py`, ours or the daemon's rebuild.

    Read from `/proc/<pid>/cmdline` rather than `pgrep -f`: a shell whose command
    line merely *mentions* the script (an operator wrapping the supervisor, a log
    tail) is not an export, and treating it as one refused a legitimate start.
    """
    pids: list[int] = []
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            raw = Path(entry.path, "cmdline").read_bytes()
        except OSError:
            continue
        argv = [part.decode("utf-8", "ignore") for part in raw.split(b"\0") if part]
        if not argv:
            continue
        if any(os.path.basename(a) == "book_cache.py" for a in argv) and "--force" in argv:
            pids.append(int(entry.name))
    return sorted(pids)


def read_rc(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


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


def cache_files(cache_dir: Path) -> dict:
    return {p.name: digest(p) for p in sorted(cache_dir.glob("book_*.npz"))[:4]}


def start_cache_export(lanes: list[str], limit: int = 0, out: str = "") -> dict | None:
    """Launch the export detached and return its durable job record.

    The job is identified by the wrapper's pid and by a file it writes its exit
    status into, so a restarted supervisor can still learn how it finished.
    """
    target = Path(out) if out else LIVE_CACHE
    running = export_pids()
    if running and target.resolve() == LIVE_CACHE.resolve():
        # The live cache is the one everybody writes `book_<LANE>.tmp.npz` into, so a
        # second export there races the first. A proof export to another directory
        # touches different files and may proceed.
        log(f"cache export not started: {len(running)} export process(es) already running "
            f"against the live cache (pids {running[:4]})")
        return None
    if running:
        log(f"note: {len(running)} other export(s) running to a different directory; "
            f"this proof export targets {target.name}/")
    args = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts" / "book_cache.py"), "--force"]
    if out:
        args += ["--out", out]
    if limit:
        args += ["--limit", str(limit)]
    args += lanes
    if CACHE_RC.exists():
        try:
            CACHE_RC.unlink()
        except OSError:
            pass
    inner = " ".join(shlex.quote(a) for a in args)
    cmd = ["bash", "-c", f"{inner}; rc=$?; echo $rc > {shlex.quote(str(CACHE_RC))}; exit $rc"]
    log(f"2-hour block: starting the L2 cache export ({', '.join(lanes) or 'all lanes'})")
    try:
        with CACHE_LOG.open("a") as fh:
            fh.write(f"\n===== {now().isoformat(timespec='seconds')} export "
                     f"{' '.join(lanes) or 'all'} =====\n")
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                    start_new_session=True)
    except OSError as exc:
        log(f"cache export failed to start: {type(exc).__name__}: {exc}")
        return None
    return {"pid": proc.pid, "started_at": now().isoformat(timespec="seconds"),
            "started_ts": time.time(), "rcfile": str(CACHE_RC),
            "lanes": lanes or "all", "dir": out or str(LIVE_CACHE), "adopted": False}


def finish_cache_export(job: dict, progress: dict, rc: int | None,
                        cache_dir: Path) -> None:
    elapsed = round(time.time() - float(job.get("started_ts") or time.time()), 1)
    record = {
        "started_at": job.get("started_at"),
        "finished_at": now().isoformat(timespec="seconds"),
        "duration_s": elapsed,
        "rc": rc,
        "lanes": job.get("lanes"),
        "dir": str(cache_dir),
        "adopted": bool(job.get("adopted")),
        "files": cache_files(cache_dir),
    }
    progress["cache"] = record
    progress["last_cache_export"] = record["finished_at"]
    progress["cache_ok"] = rc == 0
    progress["cache_history"] = (progress.get("cache_history") or [])[-8:] + [record]
    log(f"cache export finished rc={rc} in {elapsed}s "
        f"({len(record['files'])} npz file(s), adopted={record['adopted']})")
    if rc != 0 and CACHE_LOG.exists():
        try:
            for line in CACHE_LOG.read_text(errors="ignore").strip().splitlines()[-3:]:
                log(f"  cache log: {line}")
        except OSError:
            pass
    write_json(Path(progress.get("_path") or PROGRESS), progress)


def poll_cache(job: dict, progress: dict, cache_dir: Path,
               timeout: float) -> str:
    """Advance one detached export. Returns running | done | timeout."""
    started_ts = float(job.get("started_ts") or time.time())
    if _pid_alive(job["pid"]):
        if time.time() - started_ts > timeout:
            try:
                os.kill(job["pid"], 15)
            except OSError:
                pass
            log(f"cache export {job['pid']} exceeded {timeout:.0f}s; sent SIGTERM")
            job["timed_out"] = True
            return "timeout"
        return "running"
    rc = read_rc(Path(job.get("rcfile") or CACHE_RC))
    if rc is None and time.time() - started_ts < 10:
        # The wrapper may still be writing its rc after its python child exited.
        return "running"
    finish_cache_export(job, progress, rc, cache_dir)
    return "done"


def adopt_export(progress: dict, cache_dir: Path) -> dict | None:
    """Re-attach to an export that outlived the supervisor that started it."""
    rec = progress.get("cache") or {}
    if rec.get("finished_at"):
        return None
    pid = rec.get("pid")
    if not pid:
        # Written by a supervisor that kept the job in memory only: no pid, no rc
        # file, nothing to adopt. Close it out so it is never read as still in
        # flight, and so the night's record says plainly that it was unaccounted.
        progress["cache"] = {**rec,
                             "finished_at": now().isoformat(timespec="seconds"),
                             "rc": None,
                             "abandoned": True,
                             "reason": "legacy cache record carried no pid or rc file; "
                                       "completion unknown"}
        progress["cache_ok"] = False
        progress["cache_history"] = (progress.get("cache_history") or [])[-8:] + [
            {k: progress["cache"].get(k) for k in
             ("started_at", "finished_at", "rc", "abandoned", "reason")}]
        log("abandoned legacy cache record (no pid, no rc file); recorded as unknown")
        return None
    try:
        started = datetime.fromisoformat(str(rec.get("started_at")))
        started_ts = started.timestamp()
    except (TypeError, ValueError):
        started_ts = time.time()
    job = {"pid": int(pid), "started_at": rec.get("started_at"),
           "started_ts": started_ts, "rcfile": rec.get("rcfile") or str(CACHE_RC),
           "lanes": rec.get("lanes"), "adopted": True,
           "dir": rec.get("dir") or str(cache_dir)}
    if _pid_alive(job["pid"]):
        log(f"adopted in-flight cache export pid={job['pid']} started {job['started_at']}")
        return job
    rc = read_rc(Path(job["rcfile"]))
    if rc is not None:
        finish_cache_export(job, progress, rc, cache_dir)
        log("recorded the export that finished while no supervisor was watching")
        return None
    progress["cache"] = {**rec,
                         "finished_at": now().isoformat(timespec="seconds"),
                         "rc": None,
                         "abandoned": True,
                         "reason": "export process was gone at restart and left no rc file; "
                                   "completion unknown"}
    progress["cache_ok"] = False
    progress["cache_history"] = (progress.get("cache_history") or [])[-8:] + [
        {k: progress["cache"].get(k) for k in
         ("started_at", "finished_at", "rc", "abandoned", "reason")}]
    log("abandoned in-flight cache export: pid gone and no rc file; recorded as unknown")
    return None


# ---------------------------------------------------------------------- cycle
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
    return {
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
    ap.add_argument("--cache-timeout", type=float, default=7200.0)
    ap.add_argument("--cache-only", action="store_true",
                    help="run one cache export, wait for it, log it, and exit")
    ap.add_argument("--progress", default=str(PROGRESS),
                    help="progress file (default: data/overnight-l2-progress.json)")
    args = ap.parse_args()

    cache_lanes = [tok for tok in args.cache_lanes.replace(",", " ").split() if tok]
    cache_dir = Path(args.cache_out) if args.cache_out else LIVE_CACHE
    progress_path = Path(args.progress)

    if args.cache_only:
        job = start_cache_export(cache_lanes, args.cache_limit, args.cache_out)
        if job is None:
            return 1
        scratch = {"_path": str(progress_path), "cache_history": []}
        while poll_cache(job, scratch, cache_dir, args.cache_timeout) == "running":
            time.sleep(2)
        log(f"cache-only run finished rc={scratch.get('cache', {}).get('rc')}")
        return 0 if scratch.get("cache", {}).get("rc") == 0 else 1

    until = parse_until(args.until)

    prior: dict = {}
    try:
        prior = json.loads(progress_path.read_text())
    except (OSError, ValueError):
        prior = {}
    # One-time move of the ledger out of the ignored data/ tree.
    if not RND_LEDGER.exists() and RND_LEDGER_LEGACY.exists():
        try:
            RND_LEDGER.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(RND_LEDGER_LEGACY, RND_LEDGER)
            log(f"R&D ledger moved to {RND_LEDGER.relative_to(ROOT)}")
        except OSError as exc:
            log(f"ledger move failed: {exc}")
    # A resumable run continues from its cursors. A fresh one starts at the current
    # high-water mark: counting the 199M rows already on the tape is a full scan of
    # 26 GB, and "new rows this night" is the number that matters.
    cursor = int(prior.get("cursor", 0) or 0) or max_rowid("ws_book")
    tick_cursor = int(prior.get("tick_cursor", 0) or 0) or max_rowid("ticks")
    state = {"cursor": cursor, "tick_cursor": tick_cursor}
    progress = {
        "_path": str(progress_path),
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
        "cache": prior.get("cache") or {},
        "cache_history": prior.get("cache_history") or [],
        "errors": (prior.get("errors") or [])[-20:],
    }
    log(f"overnight supervisor up: until {progress['until']}, every {args.interval:.0f}s, "
        f"cursor {cursor}, resumed={bool(prior)}")

    # An export can outlive the supervisor that started it. Adopt it, or record why
    # it cannot be accounted for -- never leave it silently orphaned.
    cache_job = adopt_export(progress, cache_dir)
    if cache_job is not None and cache_job.get("dir"):
        # An adopted job may have been writing a proof directory, not the live cache.
        cache_dir = Path(cache_job["dir"])

    next_rnd = time.time() + (0 if args.rnd_now else min(
        args.rnd_every, 1800 - (now().minute % 30) * 60))
    next_cache = time.time() + (0 if args.cache_now else args.cache_every)
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
            record_rnd(progress, info)
            log(f"rnd ledger: {RND_LEDGER.relative_to(ROOT)} updated "
                f"({len(info['commits'])} commit(s), note \"{info['note']}\")")
            progress["last_rnd"] = {"ts": now().isoformat(timespec="seconds"),
                                    "head": info["head"], "new_commits": len(info["commits"]),
                                    "note": info["note"]}
            write_json(progress_path, progress)
            next_rnd = time.time() + args.rnd_every

        if cache_job is not None:
            verdict = poll_cache(cache_job, progress, cache_dir, args.cache_timeout)
            if verdict in ("done", "timeout"):
                write_json(progress_path, progress)
                cache_job = None

        if time.time() >= next_cache:
            if cache_job is not None:
                log("cache export still running; not starting a second one")
                next_cache = time.time() + args.interval
            else:
                cache_job = start_cache_export(cache_lanes, args.cache_limit, args.cache_out)
                if cache_job is None:
                    # Another process (the daemon's rebuild) is exporting the same
                    # cache. Retry on the next cycle instead of stacking a second one.
                    next_cache = time.time() + args.interval
                else:
                    progress["cache"] = {k: cache_job[k] for k in
                                         ("started_at", "pid", "rcfile", "lanes", "dir")}
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
            if cache_job is not None:
                verdict = poll_cache(cache_job, progress, cache_dir, args.cache_timeout)
                if verdict in ("done", "timeout"):
                    write_json(progress_path, progress)
                    cache_job = None

    # Never leave an export unaccounted for: its pid stays in the progress file, so a
    # successor adopts it; here we wait for it and record how it finished.
    if cache_job is not None and _pid_alive(cache_job["pid"]):
        log("waiting for the running cache export to finish before exit")
        while poll_cache(cache_job, progress, cache_dir, args.cache_timeout) == "running":
            time.sleep(5)
        write_json(progress_path, progress)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
