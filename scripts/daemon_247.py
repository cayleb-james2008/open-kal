#!/usr/bin/env python3
"""24/7 gate + compounding supervisor for F2_taker_1s.

Two modes, one process, no 18:00 stop:

  * Not armed: at every hourly boundary it rebuilds the sub-second cache and
    scores the frozen F2 rule once on `ts >= --since`. A gate pass with at least
    20 forward trades runs the same audited promotion chain as the day
    supervisor (verdict, tests, dashboard, flat check, live canary,
    `restart_flat.sh --new-epoch`, wait for `live_armed`). A gate failure is a
    result, not a failed slot: `forward_oos.py` exits 1 on a refused gate and
    the daemon still reads that artifact, so the sample keeps accumulating.

  * Armed: every 15-minute window it records balance, equity and the reconcile
    delta, confirms the engine re-sliced `venue_balance / 4` into each lane,
    and enforces a trailing drawdown floor of
    `max(--balance-floor, (1 - --drawdown) * peak_balance)` plus the $0.05
    reconciliation guard. A breach demotes the verdict, waits for a flat venue
    and restarts flat once. After a breaker the daemon will not re-arm on the
    artifact that was already on disk: only a freshly rebuilt and re-scored
    artifact can promote again.

    nohup setsid .venv/bin/python scripts/daemon_247.py \
        --since 1794909600 >> data/daemon-247.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
import traceback
import urllib.request
from datetime import datetime
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openkal.config import rolling_decay_breaker  # noqa: E402
from scripts import day_supervisor as DS  # noqa: E402

SINCE = 1794909600.0
DEFERRED = object()   # evaluate_slot: rebuild held back, the slot is not spent
LOG = ROOT / "data" / "daemon-247.log"
STATE = ROOT / "data" / "daemon-247-state.json"
FORWARD = ROOT / "data" / "forward-oos.json"
REPORT = ROOT / "data" / "evening-report.md"
NEXT = ROOT / "NEXT.json"

# The day supervisor's audited helpers read these module globals.
DS.LOG = LOG
DS.STATE = STATE
DS.VERDICT = ROOT / "data" / "strategy-verdict.json"
DS.FORWARD_PATH = FORWARD
DS.REPORT = REPORT
DS.NEXT = NEXT

STOP = {"flag": False}

# The only files the report sync may commit. This is a report committer, not a
# working-tree committer: on 2026-11-18 at 12:00 the directory allowlist swept 14
# source files into a "chore: sync report" commit while a feature was in progress.
# Source changes are the operator's to review and commit; anything else stays
# local, and databases, caches, keys and build output never enter the repository.
GIT_PATHS = ["NEXT.json", "data/evening-report.md"]
_GIT_LOCK = threading.Lock()


def _forbidden(path: str) -> bool:
    p = path.replace("\\", "/")
    # Reject on any path *component*, not just a whole-path prefix: a secrets file
    # nested under an allowlisted directory (scripts/.env) and a root-level virtualenv
    # (".venv/bin/python") both slipped through the prefix checks. The allowlist keeps
    # them out of staging today, so this is the defence-in-depth layer, not the only one.
    parts = {part for part in p.split("/") if part}
    if parts & {"keys", ".env", ".venv", ".git", ".freebuff", ".remember"}:
        return True
    if p.startswith("keys/") or "/keys/" in p:
        return True
    if p.endswith((".pem", ".key", ".db", ".sqlite", ".sqlite3", ".npz", ".pyc")):
        return True
    if ".db-" in p or "/node_modules/" in p or p.startswith("node_modules/"):
        return True
    if "/.shots/" in p or "/dist/" in p or "/__pycache__/" in p:
        return True
    if p.startswith("data/") and p != "data/evening-report.md":
        return True
    return False


def _git_sync_work(reason: str) -> None:
    """Commit and push only the report files this daemon writes.

    Both the staged diff and the commit are path-limited. A shared working tree
    means anything the operator has staged (or any source edit in flight) would
    ride along under a plain `git commit`; a `git commit -- <paths>` leaves the
    rest of the index and the tree exactly where the operator left them.

    Any refusal resets only the report paths; the supervisor's trading loop must
    never depend on GitHub being reachable.
    """
    if not (ROOT / ".git").exists():
        return
    paths = [p for p in GIT_PATHS if (ROOT / p).exists()]
    if not paths:
        return
    code, _ = DS.run(["git", "-C", str(ROOT), "add", "--", *paths], timeout=60)
    if code != 0:
        log(f"git sync skipped: add rc={code}")
        return
    code, out = DS.run(["git", "-C", str(ROOT), "diff", "--cached", "--name-only",
                        "--", *paths], timeout=30)
    staged = [line.strip() for line in out.splitlines() if line.strip()]
    if code != 0 or not staged:
        return
    bad = [p for p in staged if _forbidden(p)]
    big = []
    for p in staged:
        try:
            if (ROOT / p).stat().st_size > 5_000_000:
                big.append(p)
        except OSError:
            bad.append(p)
    if bad or big:
        DS.run(["git", "-C", str(ROOT), "reset", "--", *staged], timeout=30)
        log(f"git sync REFUSED: forbidden={bad[:5]} oversized={big[:5]}")
        return
    msg = f"chore: sync {reason} · {now().isoformat(timespec='seconds')}"
    code, out = DS.run(["git", "-C", str(ROOT), "commit", "-m", msg, "--", *paths],
                       timeout=90, env={"GIT_TERMINAL_PROMPT": "0"})
    if code != 0:
        log(f"git sync commit warning: {out[-200:]}")
        return
    code, out = DS.run(["git", "-C", str(ROOT), "push", "origin", "master"], timeout=120,
                       env={"GIT_TERMINAL_PROMPT": "0"})
    if code != 0:
        log(f"git sync push warning (commit kept locally): {out[-200:]}")
    else:
        log(f"git sync pushed: {msg} ({len(staged)} file(s))")


def live_position_open(fallback: dict | None = None) -> bool:
    """Ask the running engine, not a cached read, before loading SQLite.

    The API is loopback and the engine is the authority on open orders and
    positions. If it is down, fall back to the state already read this loop;
    an unreachable engine cannot be trading.
    """
    try:
        with urllib.request.urlopen("http://127.0.0.1:8787/api/state", timeout=3) as resp:
            payload = json.load(resp)
        if int((payload.get("account") or {}).get("open_positions") or 0) > 0:
            return True
        return any(s.get("position") is not None or s.get("order") is not None
                   for s in payload.get("slots") or [])
    except Exception as exc:  # noqa: BLE001 - never let the probe stop the supervisor
        log(f"live position probe unavailable ({type(exc).__name__}); using local state")
        return any(s.get("position") is not None or s.get("order") is not None
                   for s in (fallback or {}).get("slots") or [])


def maybe_rebuild_cache(state: dict, args, st: dict, ts: float) -> None:
    """Run the 6-hour cache rebuild only when the venue is flat.

    A live position defers the job for 60 seconds via a persisted timestamp,
    so the next loop can retry without waiting for the next quarter-hour. The
    build runs at the lowest CPU and I/O priority: heavy SQLite scans must not
    starve the engine's sub-second tick loop.
    """
    last_rebuild = float(state.get("last_rebuild") or 0)
    if ts - last_rebuild < args.cache_hours * 3600:
        return
    if ts < float(state.get("cache_defer_until") or 0):
        return
    if live_position_open(st):
        state["cache_defer_until"] = ts + 60
        save(state)
        log("6-hour cache rebuild deferred 60s: live position open")
        return
    log("6-hour cache rebuild starting (nice 19, ionice idle)")
    code, out = DS.run(
        ["nice", "-n", "19", "ionice", "-c", "3", "bash",
         str(ROOT / "scripts" / "rebuild_cache.sh")],
        timeout=7200)
    last = out.strip().splitlines()[-1] if out.strip() else ""
    state["last_rebuild"] = ts
    state["cache_defer_until"] = 0
    state["events"].append({"ts": now().isoformat(timespec="seconds"),
                            "event": "cache rebuild", "reason": f"rc={code} {last[:160]}"})
    state["events"] = state["events"][-200:]
    save(state)
    log(f"6-hour cache rebuild rc={code} ({last[:160]})")


def git_sync(reason: str) -> None:
    """Fire-and-forget sync; a slow network never delays a settlement check."""
    if not _GIT_LOCK.acquire(blocking=False):
        return

    def work() -> None:
        try:
            _git_sync_work(reason)
        except Exception as exc:  # noqa: BLE001 - GitHub must never stop the daemon
            log(f"git sync error: {type(exc).__name__}: {exc}")
        finally:
            _GIT_LOCK.release()

    threading.Thread(target=work, daemon=True, name="git-sync").start()


def now() -> datetime:
    return datetime.now().astimezone()


def log(msg: str) -> None:
    DS.log(msg)


def load() -> dict:
    state = DS.read_json(STATE, backup=True)
    if state is None:
        return {"checks": [], "events": [], "decision": "waiting"}
    state.setdefault("checks", [])
    state.setdefault("events", [])
    return state


def save(state: dict) -> None:
    DS.save_state(state, STATE)


def dec(value) -> Decimal | None:
    try:
        return Decimal(str(value))
    except Exception:  # noqa: BLE001 - venue output is data
        return None


# --------------------------------------------------------------- forward gate
def evaluate_slot(slot: datetime, state: dict, args=None) -> dict | None:
    """One rebuild and one frozen score for this hourly slot.

    `forward_oos.py` returns 1 when the gate refuses the numbers; that is a
    scored artifact, not a failed run. Only a rebuild failure or a missing /
    stale artifact spends the slot without a result.
    """
    iso = slot.isoformat()
    if live_position_open(state):
        # A heavy SQLite scan while a live binary is open is the one thing this
        # check must never do. Defer without spending the slot or deleting the
        # last artifact; the next loop retries once the lane is flat.
        log(f"forward check {iso}: deferred, live position open")
        return DEFERRED
    if (state.get("slot_attempt") or {}).get("slot") == iso:
        # A restart in the middle of a check leaves slot_attempt set but
        # last_eval_slot behind. Without this the hourly branch looks due on every
        # loop and republishes a "no artifact" report (and a report commit) until
        # the hour rolls. The slot is spent; record that and let the loop sleep.
        log(f"slot {iso} already attempted; not repeating")
        state["last_eval_slot"] = iso
        save(state)
        return None
    started = time.time()
    state["slot_attempt"] = {"slot": iso, "started": started, "since": int(SINCE)}
    save(state)
    try:
        FORWARD.unlink()
    except FileNotFoundError:
        pass
    log(f"forward check {iso}: rebuilding the sub-second cache")
    code, out = DS.run(["nice", "-n", "19", "ionice", "-c", "3", "bash",
                        str(ROOT / "scripts" / "rebuild_cache.sh")], timeout=7200)
    last = out.strip().splitlines()[-1] if out.strip() else ""
    log(f"rebuild rc={code} (last line: {last[:200]})")
    if code != 0:
        state["last_eval_slot"] = iso
        save(state)
        return None
    log(f"forward check {iso}: scoring the frozen F2 rule")
    cmd = [str(DS.PY), str(ROOT / "scripts" / "forward_oos.py"),
           "--since", str(int(SINCE)), "--out", str(FORWARD)]
    if getattr(args, "regime_mode", False):
        cmd.append("--regime-mode")
    code, out = DS.run(cmd, timeout=3600)
    if code not in (0, 1):
        log(f"forward scorer failed rc={code}: {out[-300:]}")
        state["last_eval_slot"] = iso
        save(state)
        return None
    report = DS.read_json(FORWARD)
    if report is None:
        log(f"forward scorer rc={code} produced no artifact: {out[-300:]}")
    elif float(report.get("since_ts") or -1) != float(int(SINCE)):
        log(f"artifact is for since={report.get('since_ts')}; ignored")
        report = None
    elif not FORWARD.exists() or FORWARD.stat().st_mtime < started - 5:
        log("artifact was not written by this run; ignored")
        report = None
    state["last_eval_slot"] = iso
    save(state)
    return report


def operator_approved() -> bool:
    """The second key in the arming lock: the verdict must say so in writing."""
    return (DS.read_json(DS.VERDICT) or {}).get("operator_arm_approved") is True


def promote(fwd: dict, state: dict, allow_auto_arm: bool = False) -> bool:
    gate = DS.artifact_gate(fwd)
    trades = int((fwd.get("trades") or {}).get("trades") or 0)
    log(f"forward: {trades} trades, gate pass={gate['pass']}, reasons={gate['reasons']}")
    if not gate["pass"]:
        state["decision"] = "waiting"
        state["next_action"] = (f"next hourly check; {trades} trades, "
                                f"{'; '.join(gate['reasons']) or 'gate refused'}")
        state["events"].append({"ts": now().isoformat(timespec="seconds"),
                                "event": "gate refused", "trades": trades,
                                "reasons": gate["reasons"]})
        state["events"] = state["events"][-200:]
        save(state)
        write_report(state, "gate refused at the hourly check")
        return False
    # Operator arming lock. Scoring is always allowed; putting real money at risk
    # is not. A passing gate only becomes a candidate the operator can approve.
    if not (allow_auto_arm and operator_approved()):
        why = ("--allow-auto-arm is not set" if not allow_auto_arm
               else "operator_arm_approved is not true in data/strategy-verdict.json")
        state["decision"] = "candidate_ready_for_operator_approval"
        state["next_action"] = (f"gate passed on {trades} trades; arming blocked: {why}")
        state["events"].append({"ts": now().isoformat(timespec="seconds"),
                                "event": "candidate_ready_for_operator_approval",
                                "trades": trades, "blocked_by": why})
        state["events"] = state["events"][-200:]
        save(state)
        log(f"candidate_ready_for_operator_approval: {trades} trades, gate pass; {why}")
        write_report(state, "candidate ready; no arm (operator lock)")
        return False
    state["decision"] = "promoting"
    save(state)
    log("gate passed; starting the audited promotion chain")
    result = DS.attempt_promotion(fwd, state, allow_arm=True)
    state["promotion"] = result
    if not result.get("ok"):
        state["decision"] = "waiting"
        state["next_action"] = f"promotion stopped at {result.get('reason')}; next hourly check"
        state["events"].append({"ts": now().isoformat(timespec="seconds"),
                                "event": "promotion stopped",
                                "reason": result.get("reason")})
        save(state)
        write_report(state, "promotion stopped before arming")
        return False
    epoch = DS.epoch_start()
    state["epoch_start"] = epoch
    # A fresh arm starts the 6-hour rebuild clock; rebuilding during the first
    # live window would block supervision and load the box while it trades.
    state["last_rebuild"] = time.time()
    state["decision"] = "promoted"
    state["next_action"] = "24/7 supervision: every 15-minute window"
    bal = DS.venue_balance()
    if bal is not None:
        state["peak_balance"] = str(bal)
    state["events"].append({"ts": now().isoformat(timespec="seconds"),
                            "event": "live armed", "epoch_start": epoch,
                            "balance": str(bal) if bal is not None else None})
    save(state)
    log(f"live armed: epoch_start={epoch} balance={bal}")
    write_report(state, "promotion chain completed; live compounding under supervision")
    return True


# ---------------------------------------------------------------- supervision
def floor_for(state: dict, drawdown: Decimal) -> Decimal:
    peak = dec(state.get("peak_balance")) or Decimal("0")
    return max(Decimal("105.00"), (peak * (Decimal("1") - drawdown)).quantize(Decimal("0.01")))


def unarmed_window_check(state: dict, st: dict, floor_base: Decimal,
                         drawdown: Decimal, delta_cap: Decimal,
                         epoch: float | None, force: bool = False) -> bool:
    """Refresh the venue balance and the reconcile delta while nothing is armed.

    The unarmed loop only evaluated the hourly gate, so it never re-read the
    venue: `data/evening-report.md` and `NEXT.json` kept republishing the 127.03
    delta from the last armed window at 16:01 for the rest of the day. This runs
    the same reconcile as the armed path on the same quarter-hour cadence, after
    the settlement print (60-120s past the close), and is forced once right before
    an hourly rebuild, which blocks this loop for minutes.

    It is deliberately not the breaker: nothing is armed to break, no verdict is
    demoted and no promotion chain is touched. A delta above the cap is recorded
    as `reconcile_alarm` for the operator, because a book that does not reconcile
    must never be the reason an artifact looks promotable.
    """
    ts = time.time()
    boundary = int(ts // 900 * 900)
    if state.get("last_unarmed_window") == boundary:
        return False
    if not (force or 60 <= ts % 900 < 120):
        return False
    bal = DS.venue_balance()
    if bal is not None:
        state["last_balance"] = str(bal)
        peak = dec(state.get("peak_balance")) or bal
        state["peak_balance"] = str(max(peak, bal))
    floor = max(floor_base, floor_for(state, drawdown))
    state["floor"] = str(floor)
    rec = DS.reconcile(epoch)
    delta = dec((rec or {}).get("delta")) if rec else None
    if delta is not None:
        state["last_delta"] = str(delta)
    state["last_unarmed_window"] = boundary
    acc = st.get("account") or {}
    slots = st.get("slots") or []
    state["checks"].append({
        "ts": now().isoformat(timespec="seconds"),
        "window": boundary,
        "armed": False,
        "equity": acc.get("equity"),
        "balance": str(bal) if bal is not None else None,
        "delta": str(delta) if delta is not None else None,
        "floor": str(floor),
        "peak": state.get("peak_balance"),
        "open_lanes": sum(1 for s in slots
                          if s.get("position") is not None or s.get("order") is not None),
    })
    state["checks"] = state["checks"][-2000:]
    if delta is not None and abs(delta) > delta_cap:
        state["reconcile_alarm"] = f"delta {delta} > cap {delta_cap} while unarmed"
        log(f"reconcile alarm: {state['reconcile_alarm']}")
    else:
        state.pop("reconcile_alarm", None)
    save(state)
    log(f"unarmed window {boundary}: balance={bal} delta={delta} floor={floor}")
    write_report(state, "unarmed 15-minute reconcile")
    return True


def trip(state: dict, reason: str) -> None:
    log(f"BREAKER: {reason}")
    DS.demote(reason)
    state["decision"] = "disarmed"
    state["breaker_open"] = True
    state["breaker_attempted"] = False
    state["breaker_ts"] = time.time()
    state["events"].append({"ts": now().isoformat(timespec="seconds"),
                            "event": "breaker", "reason": reason})
    state["events"] = state["events"][-200:]
    save(state)
    while not STOP["flag"]:
        flat, detail = DS.venue_flat()
        if flat:
            code, out = DS.restart_flat(new_epoch=False)
            state["breaker_attempted"] = True
            state["breaker_restarts"] = int(state.get("breaker_restarts") or 0) + 1
            state["breaker_last_restart"] = {"rc": code, "detail": out[-400:]}
            save(state)
            log(f"breaker restart_flat rc={code} (one attempt) {out[-200:]}")
            break
        log(f"breaker restart deferred: venue not flat ({detail})")
        time.sleep(20)
    state["decision"] = "waiting"
    state["next_action"] = "breaker open: next hourly check needs a fresh score before arming"
    save(state)


AFTERNOON_DATE = "2026-11-17"
AFTERNOON_BASELINE = 164.00   # last pre-audit check, 15:01 EST
AFTERNOON_START_TS = 1794946080  # 15:08:00 EST
EVENING_FINAL = ROOT / "data" / "evening-final.json"


def afternoon_snapshot(state: dict, st: dict, acc: dict, open_slots: list) -> dict:
    """Live 15:08-18:00 P&L, frozen at the first report after 18:00.

    Freezing matters: every later window would otherwise replace the 6 PM
    number with a 6:15 or 9 PM number. `data/evening-final.json` is local and
    ignored; the public artifacts are the report and NEXT.json.
    """
    frozen = DS.read_json(EVENING_FINAL) or {}
    if frozen:
        return frozen
    if now().strftime("%Y-%m-%d") != AFTERNOON_DATE:
        return {}
    fills = [f for f in (st.get("fills") or [])
             if float(f.get("ts") or 0) >= AFTERNOON_START_TS]
    entries = [f for f in fills if f.get("kind") == "entry"]
    exits = [f for f in fills if f.get("kind") == "exit"]
    equity = float(acc.get("equity") or 0)
    snap = {
        "window": "2026-11-17 15:08-18:00 EST",
        "final": now().strftime("%H:%M") >= "18:00",
        "as_of": now().isoformat(timespec="seconds"),
        "baseline_equity": AFTERNOON_BASELINE,
        "equity": round(equity, 4),
        "pnl": round(equity - AFTERNOON_BASELINE, 4),
        "slot_equity": round(equity / 4, 4),
        "realized": acc.get("realized"),
        "venue_balance": state.get("last_balance"),
        "floor": state.get("floor"),
        "reconcile_delta": state.get("last_delta"),
        "open_lanes": len(open_slots),
        "fills": len(fills),
        "entries": len(entries),
        "exits": len(exits),
        "exit_pnl": round(sum(float(f.get("pnl") or 0) for f in exits), 4),
    }
    if snap["final"]:
        DS.atomic_write(EVENING_FINAL, json.dumps(snap, indent=1))
    return snap


def window_report(state: dict, st: dict) -> None:
    acc = st.get("account") or {}
    slots = st.get("slots") or []
    open_slots = [s for s in slots if s.get("position") is not None or s.get("order") is not None]
    snap = afternoon_snapshot(state, st, acc, open_slots)
    mc = DS.read_json(ROOT / "data" / "mc-drawdown.json") or {}
    lines = [
        "# Open-Kal 24/7 report",
        "",
        f"Updated {now().isoformat(timespec='seconds')} by `scripts/daemon_247.py`.",
        "",
        "## State",
        "",
        f"- Decision: **{state.get('decision')}**",
        f"- Engine: mode={st.get('mode')} live_armed={st.get('live_armed')} "
        f"strategy={st.get('strategy')}",
        f"- Equity: {acc.get('equity')} (start {acc.get('start_equity')}, "
        f"peak {acc.get('peak_equity')}, realized {acc.get('realized')})",
        f"- Venue balance: {state.get('last_balance')}",
        f"- Trailing floor: {state.get('floor')} (peak {state.get('peak_balance')})",
        f"- Reconcile delta: {state.get('last_delta')} (cap $0.05)",
        f"- Open lanes: {len(open_slots)}"
        + (f" ({', '.join(str(s.get('series')) for s in open_slots)})" if open_slots else ""),
        "",
        "## Forward OOS",
        "",
    ]
    fwd = DS.read_json(FORWARD) or {}
    if fwd:
        t = fwd.get("trades") or {}
        lines += [
            f"- Windows scored: {fwd.get('windows_scored')} (events {fwd.get('events')})",
            f"- Post-decide trades: {t.get('trades')} ({t.get('wins')}W/{t.get('losses')}L)",
            f"- Net: ${t.get('net_pnl')}  PF: {t.get('pf')}  payoff ECE: {fwd.get('payoff_ece')}",
            f"- Gate: {'PASS' if (fwd.get('gate') or {}).get('pass') else 'FAIL'} "
            f"{' '.join((fwd.get('gate') or {}).get('reasons') or [])}",
        ]
    else:
        lines.append("- No forward artifact on disk.")
    lines += ["", "## Recent 15-minute checks", "",
              "| Time | Armed | Equity | Balance | Delta | Floor |", "|---|---|---|---|---|---|"]
    for c in (state.get("checks") or [])[-12:]:
        lines.append(f"| {c.get('ts')} | {c.get('armed')} | {c.get('equity')} | "
                     f"{c.get('balance')} | {c.get('delta')} | {c.get('floor')} |")
    if snap:
        label = "final 18:00 snapshot" if snap.get("final") else "live, updates each window"
        lines += ["", f"## Afternoon audit · 15:08–18:00 EST ({label})", "",
                  f"- Baseline equity: ${snap.get('baseline_equity'):.2f} "
                  f"(15:01 check; audit starts 15:08)",
                  f"- Equity at {snap.get('as_of')}: ${snap.get('equity'):.2f} "
                  f"→ P&L ${snap.get('pnl'):+.2f}",
                  f"- Account realized: ${snap.get('realized')} · "
                  f"equity per lane (equity/4): ${snap.get('slot_equity'):.2f}",
                  f"- Fills since 15:08: {snap.get('fills')} "
                  f"({snap.get('entries')} entries, {snap.get('exits')} exits, "
                  f"exit P&L ${snap.get('exit_pnl'):+.2f})",
                  f"- Venue balance: ${snap.get('venue_balance')} · "
                  f"floor: ${snap.get('floor')} · "
                  f"reconcile: ${snap.get('reconcile_delta')} · "
                  f"open lanes: {snap.get('open_lanes')}"]
    if mc:
        sens = {(round(float(r.get("p_win", 0)), 2), r.get("slots")): r
                for r in (mc.get("sensitivity") or [])}
        obs = {r.get("slots"): r for r in (mc.get("results") or [])
               if r.get("arrival") == "observed"}
        lines += ["", "## Capacity decision · 2 vs 4 crypto slots", "",
                  "- OOS: 36 trades, 34W/2L (94.4%), +$42.65. Forward: 20/20, +$33.13.",
                  "- Peak concurrent: 2 in-sample / 1 out-of-sample; "
                  "skipped fills at cap 2: 0 ($0.00).",
                  f"- MC observed concurrency: 20k paths, 250 windows; "
                  f"K=2 breach {100*obs.get(2, {}).get('breach_prob', 0):.3f}% vs "
                  f"K=4 {100*obs.get(4, {}).get('breach_prob', 0):.3f}%; "
                  f"median P&L ${obs.get(2, {}).get('median_final', 0):,.0f} vs "
                  f"${obs.get(4, {}).get('median_final', 0):,.0f}.",
                  "- Edge erosion (simultaneous, breach K=2 vs K=4): "
                  + "; ".join(
                      f"{100*p:.0f}% {100*sens[(p,2)]['breach_prob']:.3f}% vs "
                      f"{100*sens[(p,4)]['breach_prob']:.3f}%"
                      for p in (0.90, 0.80, 0.70) if (p, 2) in sens and (p, 4) in sens)
                  + ".",
                  f"- **Recommendation: keep `max_concurrent_crypto=2`.** "
                  f"A fourth slot adds no filled trades on the observed tape; "
                  f"revisit only if forward peak concurrency exceeds 2 with "
                  f"skipped fills, or the 94.4% OOS win rate holds through a "
                  f"full 30-day forward sample. The MC does not justify 4: at "
                  f"94.4% both caps never touch the $131.20 floor, and below "
                  f"80% both breach; 4 is marginally safer only because more "
                  f"positive-EV trades compound the buffer faster, which is a "
                  f"counterfactual the tape does not support."]
    if state.get("events"):
        lines += ["", "## Events", ""]
        for e in state["events"][-12:]:
            lines.append(f"- {e.get('ts')} **{e.get('event')}**: "
                         f"{e.get('reason') or e.get('reasons') or ''}")
    DS.atomic_write(REPORT, "\n".join(lines) + "\n")

    verdict = DS.read_json(DS.VERDICT) or {}
    payload = {
        "updated": now().isoformat(timespec="seconds"),
        "goal": "24/7 forward-OOS gate, live compounding, trailing drawdown breaker.",
        "state": {
            "decision": state.get("decision"),
            "engine_pid": (st.get("pid") or DS.read_pid()),
            "mode": st.get("mode"),
            "live_armed": st.get("live_armed"),
            "strategy": st.get("strategy"),
            "verdict_promoted": verdict.get("promoted"),
            "epoch_start": state.get("epoch_start"),
            "equity": acc.get("equity"),
            "start_equity": acc.get("start_equity"),
            "realized": acc.get("realized"),
            "venue_balance": state.get("last_balance"),
            "peak_balance": state.get("peak_balance"),
            "floor": state.get("floor"),
            "reconcile_delta": state.get("last_delta"),
            "open_lanes": len(open_slots),
        },
        "forward_oos": {
            "start_ts": int(SINCE),
            "trades": (fwd.get("trades") or {}).get("trades"),
            "net": (fwd.get("trades") or {}).get("net_pnl"),
            "pf": (fwd.get("trades") or {}).get("pf"),
            "payoff_ece": fwd.get("payoff_ece"),
            "gate": fwd.get("gate"),
            "artifact": "data/forward-oos.json",
        },
        "afternoon_audit": snap or None,
        "capacity_decision": {
            "recommendation": (mc.get("recommendation") or
                               "Keep max_concurrent_crypto=2; no observed skipped fills."),
            "observed_peak_concurrency": {"is": 2, "oos": 1},
            "skipped_fills": 0,
            "mc_artifact": "data/mc-drawdown.json",
            "mc_observed_breach": {
                str(r.get("slots")): r.get("breach_prob")
                for r in (mc.get("results") or [])
                if r.get("arrival") == "observed"
            },
        },
        "supervision": {
            "checks": len(state.get("checks") or []),
            "breaker_trips": sum(1 for e in state.get("events") or []
                                 if e.get("event") == "breaker"),
            "breaker_restarts": state.get("breaker_restarts") or 0,
            "log": "data/daemon-247.log",
            "report": "data/evening-report.md",
            "cadence": "hourly gate / 15-minute live checks / 6-hour cache rebuild",
        },
    }
    DS.atomic_write(NEXT, json.dumps(payload, indent=1, default=str))
    log("report + NEXT.json updated")
    prune_ws_now()
    git_sync(f"report {state.get('last_window') or 'gate'}")


def prune_ws_now() -> None:
    """Keep the websocket tapes inside their 3-day window.

    The engine's own maintenance thread prunes from the next build onward. This
    bridge means a long-running engine that predates that build still stops
    growing `ws_book`/`ws_trade`, without an engine restart: the work is the same
    chunked operator path, bounded so it can never hold the write lock for long.
    """
    code, out = DS.run([str(ROOT / ".venv/bin/python"),
                        str(ROOT / "scripts" / "prune_ws.py"),
                        "--budget-s", "15", "--passes", "2"], timeout=120)
    if code != 0:
        log(f"ws prune rc={code}: {out[-160:]}")
        return
    last = out.strip().splitlines()[-1] if out.strip() else ""
    log(f"ws prune: {last[:200]}")


def write_report(state: dict, note: str) -> None:
    st = DS.engine_state() or {}
    if note:
        state["events"].append({"ts": now().isoformat(timespec="seconds"),
                                "event": "report", "reason": note})
    window_report(state, st)


# --------------------------------------------------------------------- main
def run(args) -> int:
    state = load()
    drawdown = Decimal(str(args.drawdown))
    delta_cap = Decimal(str(args.delta_cap))
    floor_base = Decimal(str(args.balance_floor))
    state["started"] = state.get("started") or now().isoformat(timespec="seconds")
    state.setdefault("last_rebuild", time.time())
    if not state.get("last_eval_slot"):
        # Taking over mid-hour means the day supervisor already spent this
        # slot. Only a boot in the first five minutes may run it.
        slot = now().replace(minute=0, second=0, microsecond=0)
        if (now() - slot).total_seconds() >= 300:
            state["last_eval_slot"] = slot.isoformat()
    save(state)
    log(f"daemon up: since {int(SINCE)} floor ${floor_base} drawdown {drawdown} "
        f"delta cap ${delta_cap}")

    if args.promote_now and not args.allow_auto_arm:
        log("promote-now ignored: the operator arming lock is active "
            "(--allow-auto-arm is not set)")
    elif args.promote_now and not (DS.engine_state() or {}).get("live_armed"):
        fwd = DS.read_json(FORWARD)
        age = (time.time() - FORWARD.stat().st_mtime) if FORWARD.exists() else 1e9
        gate = DS.artifact_gate(fwd) if fwd else {"pass": False, "reasons": ["no artifact"]}
        if fwd and age <= 3600 and gate["pass"]:
            log(f"promote-now: fresh gate-passing artifact (age {age:.0f}s); promoting")
            state["last_eval_slot"] = now().replace(
                minute=0, second=0, microsecond=0).isoformat()
            save(state)
            promote(fwd, state, args.allow_auto_arm)
        else:
            log(f"promote-now: no fresh passing artifact (age {age:.0f}s, "
                f"reasons={gate.get('reasons')})")

    if state.get("decision") == "promoting":
        # A crash inside the chain: adopt only a verdict + armed engine that both
        # agree, otherwise demote and require the next fresh score.
        verdict = DS.read_json(DS.VERDICT) or {}
        st = DS.engine_state() or {}
        if verdict.get("promoted") and st.get("live_armed"):
            state["decision"] = "promoted"
            state["epoch_start"] = DS.epoch_start()
            log("recovery: verdict promoted and engine armed; supervising")
        else:
            DS.demote("promotion interrupted before the engine reported armed")
            state["decision"] = "waiting"
            log("recovery: promotion interrupted; demoted")
        save(state)

    while not STOP["flag"]:
        try:
            st = DS.engine_state() or {}
            armed = bool(st.get("live_armed"))
            if state.get("heartbeat_bucket") != int(time.time()) // 300:
                state["heartbeat_bucket"] = int(time.time()) // 300
                state["heartbeat"] = now().isoformat(timespec="seconds")
                save(state)
                log(f"tick armed={armed} last_window={state.get('last_window')}")
            epoch = state.get("epoch_start") or DS.epoch_start()
            if epoch:
                state["epoch_start"] = epoch

            if not armed:
                if state.get("breaker_open"):
                    # Wait for the flat-venue restart to settle, then let the
                    # hourly gate decide again on a fresh artifact.
                    state["breaker_open"] = False
                    state["breaker_attempted"] = False
                    save(state)
                slot = now().replace(minute=0, second=0, microsecond=0)
                due = (now() - slot).total_seconds() >= 5
                hourly = due and state.get("last_eval_slot") != slot.isoformat()
                # The rebuild below holds this loop for minutes, so the report's
                # reconcile is refreshed first rather than after it.
                unarmed_window_check(state, st, floor_base, drawdown, delta_cap,
                                     epoch, force=hourly)
                if hourly:
                    fwd = evaluate_slot(slot, state, args)
                    if fwd is DEFERRED:
                        time.sleep(min(args.interval, 30))
                        continue
                    if fwd is None:
                        state["decision"] = "waiting"
                        state["next_action"] = f"hourly check {slot.isoformat()} produced no artifact"
                        save(state)
                        write_report(state, "hourly slot produced no artifact")
                    else:
                        fresh_after_breaker = (
                            not state.get("breaker_ts")
                            or FORWARD.stat().st_mtime > float(state["breaker_ts"]))
                        if not fresh_after_breaker:
                            log("artifact predates the breaker; refusing to re-arm on it")
                            state["decision"] = "waiting"
                            save(state)
                        elif promote(fwd, state, args.allow_auto_arm):
                            pass
                time.sleep(min(args.interval, 30))
                continue

            # ---- armed ----
            # Venue and reconcile reads belong to the 15-minute settlement, not
            # to every 30-second loop; a 20% trailing stop has no use for
            # sub-minute venue traffic.
            ts = time.time()
            # Outside the settlement window, retry a deferred rebuild every
            # loop. 120s after the boundary leaves the 15:31 reconcile clear.
            if 120 <= ts % 900:
                maybe_rebuild_cache(state, args, st, ts)
            boundary = int(ts // 900 * 900)
            # Check one minute after each quarter-hour close, when the venue's
            # settlement print is in. A reconcile 20s before the close sees an
            # unsettled exit as a delta and would trip the breaker on a trade
            # that is already booked.
            mod = ts % 900
            if state.get("last_window") == boundary or not (60 <= mod < 120):
                time.sleep(min(args.interval, 30))
                continue
            bal = DS.venue_balance()
            if bal is not None:
                state["last_balance"] = str(bal)
                peak = dec(state.get("peak_balance")) or bal
                if epoch and state.get("epoch_start") != epoch:
                    peak = bal
                peak = max(peak, bal)
                state["peak_balance"] = str(peak)
            floor = max(floor_base, floor_for(state, drawdown))
            state["floor"] = str(floor)
            rec = DS.reconcile(epoch)
            delta = dec((rec or {}).get("delta"))
            state["last_delta"] = str(delta) if delta is not None else None
            save(state)

            breaks = []
            if bal is not None and bal < floor:
                breaks.append(f"venue balance {bal} < trailing floor {floor}")
            if delta is not None and abs(delta) > delta_cap:
                breaks.append(f"reconcile delta {delta} > cap {delta_cap}")
            if breaks:
                trip(state, "; ".join(breaks))
                continue

            # Rolling-decay guard. The forward gate says the rule was worth
            # arming; only this epoch's own closed trades say it still is. The
            # trades table is newest-first and holds older epochs, so the window
            # is sorted by close time and cut at this epoch's start.
            epoch_trades = sorted(
                (t for t in (st.get("trades") or [])
                 if float(t.get("closed_ts") or 0) >= float(epoch or 0)),
                key=lambda t: float(t.get("closed_ts") or 0))
            decay = rolling_decay_breaker([t.get("pnl") for t in epoch_trades],
                                         (st.get("account") or {}).get("equity") or 0)
            if decay:
                trip(state, f"rolling decay: {decay}")
                continue

            state["last_window"] = boundary
            acc = st.get("account") or {}
            slots = st.get("slots") or []
            state["checks"].append({
                "ts": now().isoformat(timespec="seconds"),
                "window": boundary,
                "armed": armed,
                "equity": acc.get("equity"),
                "balance": str(bal) if bal is not None else None,
                "delta": str(delta) if delta is not None else None,
                "floor": str(floor),
                "peak": state.get("peak_balance"),
                "open_lanes": sum(1 for s in slots
                                  if s.get("position") is not None or s.get("order") is not None),
            })
            state["checks"] = state["checks"][-2000:]
            save(state)
            log(f"window {boundary}: equity={acc.get('equity')} balance={bal} "
                f"delta={delta} floor={floor}")
            write_report(state, "15-minute window check")

            # 6-hour cache rebuild, only while every lane is flat.
            maybe_rebuild_cache(state, args, st, ts)

            time.sleep(min(args.interval, 30))
        except Exception as exc:  # noqa: BLE001 - the loop must survive a bad read
            log(f"loop error: {type(exc).__name__}: {exc}\n{traceback.format_exc()[-1200:]}")
            time.sleep(30)
    save(state)
    log("daemon stopped")
    return 0


def main() -> int:
    global SINCE
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=float, default=SINCE)
    ap.add_argument("--balance-floor", default="105.00")
    ap.add_argument("--delta-cap", default="0.05")
    ap.add_argument("--drawdown", default="0.20")
    ap.add_argument("--cache-hours", type=float, default=6.0)
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--allow-auto-arm", action="store_true", default=False,
                    help="permit the hourly gate to arm live; still requires "
                         "\"operator_arm_approved\": true in the verdict")
    ap.add_argument("--promote-now", action="store_true",
                    help="handover: run the audited chain now on a fresh passing artifact")
    ap.add_argument("--regime-mode", action="store_true",
                    help="score the rolling 48h slice with the decay gate "
                         "(forward_oos.py --regime-mode)")
    args = ap.parse_args()
    SINCE = float(args.since)

    def handle(signum, frame):  # noqa: ANN001 - signal signature
        STOP["flag"] = True
        log(f"signal {signum}: stopping after this step")

    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
