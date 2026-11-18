#!/usr/bin/env python3
"""Daytime supervisor: wait for the forward OOS slice, then arm only if it passes.

This runs from the morning until 18:00 EST and does four things, in this order:

  1. Leaves the recording engine alone. From now until the forward slice's end
     nothing is restarted, nothing is armed, and `scripts/canary.py --live` is not
     run: the tape is the evidence and an interruption is a hole in it.
  2. At 12:00, 13:00 and 14:00 EST it rebuilds the sub-second cache from the
     production tape and scores the frozen F2 rule ONCE per slot on
     `ts >= --since`. If fewer than 20 forward trades exist the rule has not been
     tested, and the script waits an hour for more tape. A slot whose rebuild or
     score fails is spent, not retried, and the next slot tries again; an
     artifact already written for a slot is adopted instead of re-scored. It
     never lowers the floor and never moves a threshold.
  3. If the forward slice clears every gate, it writes the verdict, runs the
     unit tests and the dashboard checks, proves the venue flat, runs the live
     one-contract canary, and arms through `scripts/restart_flat.sh --new-epoch`.
     Any failure in that sequence demotes the verdict again and leaves the
     engine in paper mode.
  4. Once armed, it owns the hard circuit breaker: venue balance below the floor
     or a reconcile delta above the cap demotes the verdict and restarts flat.
     Every fifteen minutes it records the engine state, the balance and the
     reconciliation. At 18:00 it writes `data/evening-report.md` and `NEXT.json`.

    bash scripts/day_watchdog.sh --until "2026-11-17T18:00:00" --since 1794909600
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOG = ROOT / "data" / "day-supervisor.log"
STATE = ROOT / "data" / "day-state.json"
VERDICT = ROOT / "data" / "strategy-verdict.json"
FORWARD = ROOT / "data" / "forward-oos.json"
# The path the scorer writes and the supervisor reads. Overridable so a drill can
# point at a sandbox artifact without overwriting the real forward result.
FORWARD_PATH = FORWARD
CANARY_LIVE = ROOT / "data" / "canary.json"
CANARY_DRY = ROOT / "data" / "canary-dry.json"
# Set only by `--dry-run`. In a dry run the two steps that cannot be rehearsed
# safely -- the live one-contract canary and the engine restart that arms -- are
# replaced by their non-destructive counterparts (a dry canary and a recorded
# no-op), and nothing else changes: the tests, the dashboard checks, the venue
# read and the demote-on-failure path all run exactly as they will at 12:00.
DRY_RUN = False
RECONCILE = ROOT / "data" / "day-reconcile.json"
REPORT = ROOT / "data" / "evening-report.md"
NEXT = ROOT / "NEXT.json"
PY = ROOT / ".venv" / "bin" / "python"
FORWARD_SINCE = 1794909600


def now() -> datetime:
    return datetime.now().astimezone()


def log(msg: str) -> None:
    line = f"{now().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


def read_json(path: Path, *, backup: bool = False) -> dict | None:
    """Parse a JSON file, optionally falling back to its last valid copy.

    With `backup`, a truncated primary file is not the same thing as an absent
    one: the `.bak` beside it holds the previous complete contents, so a reader
    recovers the real verdict/state instead of an empty dict that would look
    like "not promoted" or "no schedule". Without a usable backup the caller
    gets None and must say so, not guess.
    """
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        if backup:
            alt = path.with_name(path.name + ".bak")
            try:
                data = json.loads(alt.read_text())
            except (OSError, ValueError):
                pass
            else:
                log(f"{path.name} unreadable ({type(exc).__name__}); "
                    f"recovered the previous contents from {alt.name}")
                return data
        return None


def load_state(path: Path | None = None) -> dict:
    state = read_json(path or STATE, backup=True)
    if state is None:
        return {"checks": [], "decision": "pending", "steps": []}
    return state


def atomic_write(path: Path, text: str) -> None:
    """Write-then-rename, so a kill mid-write cannot leave a truncated file.

    The watchdog restarts this process with SIGKILL when the window closes, and
    the engine reads the verdict on its next window roll. A half-written
    `day-state.json` would restart the schedule from scratch; a half-written
    verdict would disarm an armed engine at random. Rename is atomic on the same
    filesystem, so readers see either the old file or the new one.
    """
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    # Keep the last *valid* contents beside the file, so a truncated primary can
    # be recognised and recovered rather than silently read as an empty object.
    if path.exists():
        try:
            json.loads(path.read_text())
        except (OSError, ValueError):
            pass
        else:
            try:
                shutil.copyfile(path, path.with_name(path.name + ".bak"))
            except OSError:
                pass
    os.replace(tmp, path)


def save_state(state: dict, path: Path | None = None) -> None:
    atomic_write(path or STATE, json.dumps(state, indent=1, default=str))


def run(cmd: list[str], *, timeout: float = 600, cwd: Path | None = None,
        env: dict | None = None) -> tuple[int, str]:
    merged = dict(os.environ)
    if env:
        merged.update(env)
    try:
        proc = subprocess.run(cmd, cwd=str(cwd or ROOT), capture_output=True, text=True,
                              timeout=timeout, env=merged)
        out = ((proc.stdout or "") + (proc.stderr or "")).strip()
        return proc.returncode, out
    except (subprocess.SubprocessError, OSError) as exc:
        return 1, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------- venue reads
def venue_balance() -> Decimal | None:
    code, out = run([str(PY), "-c", (
        "import asyncio,sys;sys.path.insert(0,%r)\n"
        "from openkal.auth import KalshiClient\n"
        "from openkal.config import load_settings\n"
        "async def m():\n"
        "    c=KalshiClient(load_settings())\n"
        "    try:\n"
        "        s,d=await c.get('/portfolio/balance')\n"
        "    finally:\n"
        "        await c.close()\n"
        "    print(s, (d or {}).get('balance_dollars',''))\n"
        "asyncio.run(m())\n" % str(ROOT))], timeout=90)
    parts = (out or "").split()
    if len(parts) < 2 or parts[0] != "200":
        log(f"balance read failed: rc={code} {out[-200:]!r}")
        return None
    try:
        return Decimal(parts[1])
    except Exception:  # noqa: BLE001 - a malformed venue number is data, not a crash
        return None


def venue_flat() -> tuple[bool, str]:
    """The same proof `scripts/restart_flat.sh` requires before it will restart."""
    code, out = run([str(PY), "-c", (
        "import asyncio,sys;sys.path.insert(0,%r)\n"
        "from openkal.auth import KalshiClient\n"
        "from openkal.config import load_settings\n"
        "async def m():\n"
        "    c=KalshiClient(load_settings())\n"
        "    try:\n"
        "        s1,p=await c.get('/portfolio/positions')\n"
        "        s2,o=await c.get('/portfolio/orders',{'status':'resting'})\n"
        "    finally:\n"
        "        await c.close()\n"
        "    held=[x['ticker'] for x in p.get('market_positions',[]) "
        "if float(x.get('position_fp') or 0)!=0]\n"
        "    print(s1, s2, len(held), len(o.get('orders',[])), ','.join(held))\n"
        "asyncio.run(m())\n" % str(ROOT))], timeout=90)
    parts = (out or "").split()
    if len(parts) < 4 or parts[0] != "200" or parts[1] != "200":
        return False, f"venue read failed: {out[-160:]!r}"
    positions, orders = int(parts[2]), int(parts[3])
    return positions == 0 and orders == 0, (
        f"{positions} position(s), {orders} resting order(s)")


def engine_state() -> dict | None:
    code, out = run(["curl", "-s", "--max-time", "10", "http://127.0.0.1:8787/api/state"],
                    timeout=25)
    if not out.strip():
        return None
    try:
        return json.loads(out)
    except ValueError:
        return None


def wait_for_live_arm(strategy: str = "F2_taker_1s", timeout: float = 1200.0,
                      interval: float = 5.0) -> tuple[bool, str]:
    """Poll /api/state until the restarted engine is genuinely live.

    A fixed sleep is not evidence: the engine boots into ARM_WAIT and can need
    one 15-minute window roll before readiness flips. Only the exact strategy
    with live_armed=True is accepted; paper or stale state is not.
    """
    deadline = time.time() + timeout
    last: dict | None = None
    while time.time() < deadline:
        st = engine_state()
        if st is not None:
            last = st
            if bool(st.get("live_armed")) and st.get("strategy") == strategy:
                return True, (f"live_armed=True strategy={st.get('strategy')} "
                              f"mode={st.get('mode')} pid={st.get('pid')}")
        time.sleep(interval)
    return False, f"timeout after {timeout:.0f}s; last={last}"


def read_pid() -> int | None:
    try:
        return int((ROOT / "data" / "engine.pid").read_text().strip())
    except (OSError, ValueError):
        return None


def epoch_start() -> float | None:
    code, out = run([str(PY), "-c", (
        "import sys;sys.path.insert(0,%r)\n"
        "from openkal.config import load_settings\n"
        "from openkal.store import Store\n"
        "s=Store(load_settings().db_path)\n"
        "print(s.get_meta('epoch_start') or '')\n" % str(ROOT))], timeout=60)
    try:
        return float(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def reconcile(since: float | None) -> dict | None:
    cmd = [str(PY), str(ROOT / "scripts" / "reconcile_live.py"), "--json", str(RECONCILE)]
    if since:
        cmd += ["--since", str(int(since))]
    run(cmd, timeout=600)
    return read_json(RECONCILE)


# --------------------------------------------------------------- the decision
def evaluate_forward(since: float, injected: str | None = None, *,
                     slot: datetime | None = None,
                     state: dict | None = None) -> dict | None:
    """One rebuild, one frozen score per scheduled slot. The evidence base.

    Two crash windows are closed here, because the watchdog can restart this
    process at any moment:

      * a result already written for this slot is adopted, not scored again: a
        kill between the scorer's write and the state save must not turn the
        single shot into a second look at the forward slice;
      * a slot already attempted but left without a usable artifact is spent,
        not retried: the noon evaluation and each hourly re-check happen once
        per slot even across restarts.

    A stale artifact (one written before this slot's window) is still deleted,
    so yesterday's pass cannot be read as today's. `--forward-report` skips the
    rebuild and score for a drill, and says so in the log and in the state.
    """
    if injected:
        report = read_json(Path(injected))
        if report is None:
            log(f"INJECTED forward report is unreadable: {injected}")
            return None
        report["injected"] = str(injected)
        # Mirror it to the artifact path the report and NEXT.json read, so a
        # drill's handoff describes the numbers it actually decided on.
        atomic_write(FORWARD_PATH, json.dumps(report, indent=1, default=str))
        log(f"INJECTED forward report from {injected}: "
            f"{(report.get('trades') or {}).get('trades')} trades, "
            f"gate pass={(report.get('gate') or {}).get('pass')}")
        return report
    st = state if state is not None else load_state()
    slot_iso = slot.isoformat() if slot else None
    slot_ts = slot.timestamp() if slot else 0.0
    existing = read_json(FORWARD_PATH)
    if (existing is not None and slot_ts
            and float(existing.get("since_ts") or -1) == float(int(since))
            and slot_ts - 120 <= FORWARD_PATH.stat().st_mtime <= slot_ts + 1800):
        log(f"forward check: adopting the result already on disk for slot "
            f"{slot_iso}; the slice is not scored again")
        return existing
    prior = st.get("slot_attempt") or {}
    if slot_iso and prior.get("slot") == slot_iso:
        log(f"forward check: slot {slot_iso} was already attempted; not repeating "
            f"it (exactly one attempt per slot)")
        return None
    started = time.time()
    # Persist the attempt before the first child runs: a kill during the rebuild
    # must spend this slot, not hand the restarted process a second rebuild.
    st["slot_attempt"] = {"slot": slot_iso, "started": started, "since": int(since)}
    if state is not None:
        state["slot_attempt"] = st["slot_attempt"]
    save_state(st)
    try:
        FORWARD_PATH.unlink()
    except FileNotFoundError:
        pass
    log("forward check: rebuilding the sub-second cache from the production tape")
    code, out = run(["bash", str(ROOT / "scripts" / "rebuild_cache.sh")], timeout=7200)
    log(f"rebuild rc={code} (last line: {out.strip().splitlines()[-1] if out.strip() else ''}")
    if code != 0:
        log("rebuild failed; this slot is spent without a score")
        return None
    log("forward check: scoring the frozen F2 rule once on the forward slice")
    code, out = run([str(PY), str(ROOT / "scripts" / "forward_oos.py"),
                     "--since", str(int(since)), "--out", str(FORWARD_PATH)], timeout=3600)
    if code != 0:
        # A nonzero scorer is a failed slot, not an artifact to read. Without this
        # check a partial or stale file written by a crashed scorer could be
        # adopted as the forward result and arm the engine.
        log(f"forward score failed (rc={code}); this slot is spent without a result")
        return None
    report = read_json(FORWARD_PATH)
    if report is None:
        log(f"forward score produced no artifact (rc={code}): {out[-400:]}")
        return None
    if float(report.get("since_ts") or -1) != float(int(since)):
        log(f"forward artifact is for since={report.get('since_ts')}, not {int(since)}: ignored")
        return None
    if not FORWARD_PATH.exists() or FORWARD_PATH.stat().st_mtime < started - 5:
        log("forward artifact was not written by this run: ignored")
        return None
    return report


def write_verdict(fwd: dict) -> None:
    trades = fwd["trades"]
    candidate = {
        "name": "F2_taker_1s",
        "label": "F2_taker_1s|tp0.05|stop0.07c2s|bar90",
        "exit_mode": "maker_tp",
        "tp_c": 0.05,
        "stop_c": 0.07,
        "probability_target": "payoff",
        "detector": {
            "z_min": 0.30, "flow_min": 0.10, "revision_min": 0.0,
            "delta_bps_min": 3.0, "cost_band": [0.25, 0.62],
            "seconds_left": [180, 780], "max_spread": 0.04,
            "entry_lag_s": 1.0, "stop_confirm_s": 2.0, "barrier_s": 90.0,
            "grid": "single configuration, not a union",
        },
    }
    oos = {
        "trades": trades["trades"],
        "wins": trades["wins"], "losses": trades["losses"],
        "win_rate": trades["win_rate"],
        "net": str(trades["net_pnl"]),
        "pf": trades["pf"],
        "ece": fwd["payoff_ece"],
        "payoff_ece": fwd["payoff_ece"],
        "ece_target": "payoff",
        "fees": str(trades.get("taker_fees_paid", 0)),
        "crossed_fills": trades["crossed_fills"],
        "max_entry_cost": str(trades["max_side_cost"]),
        "avg_usd_per_trade": round(trades["net_pnl"] / max(1, trades["trades"]), 4),
        "by_lane": trades.get("by_lane") or {},
        "by_reason": trades.get("by_reason") or {},
        "slice": "forward OOS, every settled window closing at/after the tape start",
        "slice_start_ts": fwd["since_ts"],
        "slice_start_local": fwd["since_local"],
        "windows_scored": fwd.get("windows_scored"),
        "generated": fwd.get("generated"),
    }
    payload = {
        "generated_ts": time.time(),
        "generated": now().isoformat(),
        "promoted": True,
        "candidate": candidate,
        "oos": oos,
        "is": {"note": "the in-sample fit that froze the rule and its payoff calibrator",
               "trades": 46, "net": "38.54", "pf": 2.324, "payoff_ece": 0.0764,
               "source": "scripts/f2_structural.py on the 60/40 cache split"},
        "forward": {"gate": fwd["gate"], "all_trades": fwd.get("all_trades"),
                    "calibration": fwd.get("calibration")},
        "notes": [
            "ECE is measured against the trade's own payoff event (1 if net > 0 else 0), "
            "which is the quantity this program's probability describes.",
            "Thresholds, cost band, calibrator and execution lag were frozen before the "
            "forward slice was scored; nothing was tuned on it.",
        ],
    }
    atomic_write(VERDICT, json.dumps(payload, indent=1, default=str))


def demote(reason: str) -> bool:
    verdict = read_json(VERDICT, backup=True)
    if verdict is None:
        # Unreadable with no valid copy behind it. Write a valid, fail-closed
        # record rather than leaving a file every later reader has to guess at;
        # never report this as "already demoted", because nothing was read.
        atomic_write(VERDICT, json.dumps({
            "generated": now().isoformat(),
            "promoted": False,
            "recovery": "verdict file was unreadable; reset to not promoted",
            "disarmed_at": now().isoformat(),
            "disarm_reason": reason,
        }, indent=1))
        log(f"verdict unreadable and no backup; reset to not promoted: {reason}")
        return True
    if verdict.get("promoted"):
        verdict["promoted"] = False
        verdict["disarmed_at"] = now().isoformat()
        verdict["disarm_reason"] = reason
        atomic_write(VERDICT, json.dumps(verdict, indent=1, default=str))
        log(f"verdict demoted: {reason}")
        return True
    log(f"verdict already demoted; disarm reason: {reason}")
    return False


def restart_flat(new_epoch: bool = False) -> tuple[int, str]:
    # Hard guard: a dry run must never restart the recording engine. This sits in
    # the function rather than at its call sites so a future caller cannot get it
    # wrong; normal operation never sets DRY_RUN.
    if DRY_RUN:
        cmd = ("OPENKAL_ARM_LIVE=1 bash scripts/restart_flat.sh"
               + (" --new-epoch" if new_epoch else ""))
        log(f"DRY RUN: would run {cmd} (not executed)")
        return 0, f"DRY RUN: {cmd} not executed"
    cmd = ["bash", str(ROOT / "scripts" / "restart_flat.sh")]
    if new_epoch:
        cmd.append("--new-epoch")
    return run(cmd, timeout=300, env={"OPENKAL_ARM_LIVE": "1"})


def artifact_gate(fwd: dict) -> dict:
    """Re-score a forward artifact with the frozen gate before trusting it.

    The artifact carries its own `gate.pass`, but a verdict is written from that
    file. Recomputing from the numbers means a hand-edited, stale or partially
    written result cannot arm an engine on a claim nobody checked.
    """
    from scripts import forward_oos as F

    # A regime artifact is re-scored by the regime gate. Reading it with the
    # frozen thresholds would refuse every artifact the rolling 48h run passed,
    # and reading a frozen artifact with the regime thresholds would arm on half
    # the evidence -- the artifact has to say which gate scored it.
    return F.gate_from_summary(fwd.get("trades") or {},
                               float(fwd.get("payoff_ece") or 1.0),
                               fwd.get("lanes_profitable"),
                               regime=bool(fwd.get("regime_mode")))


def attempt_promotion(fwd: dict, state: dict, allow_arm: bool = True) -> dict:
    """Verdict -> gate re-check -> tests -> dashboard -> flat -> canary -> arm.

    Any failing step demotes the verdict again and returns before the next step:
    a chain that continues past a failure is how an unverified engine gets armed.
    With `DRY_RUN` the live canary becomes `scripts/canary.py` in dry mode and the
    restart is recorded instead of executed; the ordering, the checks and the
    failure handling are the same code the real run uses.
    """
    if not allow_arm:
        # The operator arming lock, enforced at the only function that writes
        # `promoted: true`: no flag, no approval, no promotion chain.
        log("promotion refused: operator arming lock is active")
        state["decision"] = "candidate_ready_for_operator_approval"
        save_state(state)
        return {"ok": False, "steps": [], "reason": "operator approval required"}

    steps: list[dict] = []

    def step(name: str, ok: bool, detail: str) -> bool:
        steps.append({"step": name, "ok": ok, "detail": detail[-1500:]})
        state["steps"] = steps
        save_state(state)
        log(f"{'PASS' if ok else 'FAIL'} {name}: {detail.strip().splitlines()[-1] if detail.strip() else ''}")
        return ok

    gate = artifact_gate(fwd)
    if not step("gate re-check on the artifact's numbers", gate["pass"],
                "all seven conditions hold" if gate["pass"]
                else "; ".join(gate["reasons"])):
        # Fail closed even if a previous run left the verdict promoted: the
        # re-check exists so no chain can proceed on numbers the frozen gate
        # refuses, and that includes leaving a stale `promoted: true` behind.
        demote("arming stopped: gate re-check failed")
        state["decision"] = "blocked"
        save_state(state)
        return {"ok": False, "steps": steps, "reason": "gate re-check"}

    write_verdict(fwd)
    log("verdict written: promoted true (F2_taker_1s, payoff ECE)")

    checks: list[tuple[str, list[str], float]] = [
        ("unit tests: live exit", [str(PY), str(ROOT / "tests" / "test_live_exit.py")], 300),
        ("unit tests: controls (parity)",
         [str(PY), str(ROOT / "tests" / "test_gate_controls.py"),
          str(ROOT / "data" / "openkal-parity.db")], 600),
        ("unit tests: controls (prod)",
         [str(PY), str(ROOT / "tests" / "test_gate_controls.py"),
          str(ROOT / "data" / "openkal-prod.db")], 900),
        ("unit tests: controls bind",
         [str(PY), str(ROOT / "tests" / "test_controls_bind.py"),
          str(ROOT / "data" / "openkal-parity.db")], 300),
        ("unit tests: day pipeline",
         [str(PY), str(ROOT / "tests" / "test_day_pipeline.py")], 900),
        ("dashboard build", ["npm", "run", "build"], 900),
        ("dashboard role-coverage", ["npm", "run", "role-coverage"], 300),
    ]
    for name, cmd, timeout in checks:
        cwd = ROOT / "dashboard" if cmd[0] == "npm" else ROOT
        code, out = run(cmd, timeout=timeout, cwd=cwd)
        if not step(name, code == 0, out):
            demote(f"arming stopped: {name} failed")
            state["decision"] = "blocked"
            return {"ok": False, "steps": steps, "reason": name}

    code, out = run(["npm", "run", "render-check"], timeout=1200, cwd=ROOT / "dashboard",
                    env={"BASE_URL": "http://127.0.0.1:8787"})
    if not step("dashboard render-check", code == 0 and "PASSED" in out, out):
        demote("arming stopped: render-check failed")
        state["decision"] = "blocked"
        return {"ok": False, "steps": steps, "reason": "render-check"}

    flat, detail = venue_flat()
    if not step("venue flat", flat, detail):
        demote("arming stopped: venue not flat")
        state["decision"] = "blocked"
        return {"ok": False, "steps": steps, "reason": "venue not flat"}

    if DRY_RUN:
        dry_started = time.time()
        code, out = run([str(PY), str(ROOT / "scripts" / "canary.py")], timeout=300)
        dry = read_json(CANARY_DRY) or {}
        fresh = float(dry.get("generated_ts") or 0) >= dry_started - 5
        ok = (code == 0 and bool(dry.get("dry_plan")) and fresh
              and int(dry.get("contracts") or 0) == 1)
        if not step("live canary (1 contract) [DRY RUN: canary.py without --live]", ok,
                    f"rc={code} ticker={dry.get('ticker')} fresh={fresh} "
                    f"contracts={dry.get('contracts')} "
                    f"would_run={PY} scripts/canary.py --live"):
            demote("arming stopped: canary dry run did not produce a plan")
            state["decision"] = "blocked"
            return {"ok": False, "steps": steps, "reason": "canary"}
        canary = dry
    else:
        canary_started = time.time()
        code, out = run([str(PY), str(ROOT / "scripts" / "canary.py"), "--live"], timeout=300)
        canary = read_json(CANARY_LIVE) or {}
        # A stale canary.json is not evidence for this arm: a --live run that
        # exits 0 without writing a fresh file must not pass on last night's.
        fresh = float(canary.get("generated_ts") or 0) >= canary_started - 5
        ok = (code == 0 and bool(canary.get("ok")) and fresh
              and int(canary.get("contracts") or 0) == 1)
        if not step("live canary (1 contract)", ok,
                    f"rc={code} ok={canary.get('ok')} fresh={fresh} "
                    f"contracts={canary.get('contracts')} "
                    f"delta={canary.get('reconcile_delta')} "
                    f"flat={canary.get('flattened')}"):
            demote("arming stopped: canary did not pass")
            state["decision"] = "blocked"
            return {"ok": False, "steps": steps, "reason": "canary"}

    flat, detail = venue_flat()
    if not step("venue flat after the canary", flat, detail):
        demote("arming stopped: venue not flat after the canary")
        state["decision"] = "blocked"
        return {"ok": False, "steps": steps, "reason": "venue not flat after canary"}

    if DRY_RUN:
        would = ("OPENKAL_ARM_LIVE=1 bash scripts/restart_flat.sh --new-epoch")
        step("armed via restart_flat.sh --new-epoch [DRY RUN]", True,
             f"would run: {would} (not executed; engine untouched)")
        state["decision"] = "verified (dry run)"
        state["next_action"] = "none: dry run, engine left in paper mode"
        save_state(state)
        return {"ok": True, "steps": steps, "epoch_start": state.get("epoch_start"),
                "dry_run": True, "would_run": would}

    code, out = restart_flat(new_epoch=True)
    ok = step("armed via restart_flat.sh --new-epoch", code == 0, out)
    if not ok:
        demote("arming stopped: restart_flat failed")
        state["decision"] = "blocked"
        return {"ok": False, "steps": steps, "reason": "restart_flat"}

    # The new process boots into ARM_WAIT and flips `live_armed` on the first
    # window roll once readiness passes. Do not write "promoted" until the
    # engine itself reports the exact strategy and live_armed=True.
    live_ok, live_detail = wait_for_live_arm("F2_taker_1s", timeout=1200.0)
    if not step("live engine reports live_armed=True and strategy=F2_taker_1s",
                live_ok, live_detail):
        demote("arming stopped: engine did not report live_armed/F2_taker_1s")
        state["decision"] = "blocked"
        return {"ok": False, "steps": steps, "reason": "live state"}
    start = epoch_start()
    state["epoch_start"] = start
    state["decision"] = "promoted"
    save_state(state)
    return {"ok": True, "steps": steps, "epoch_start": start}


def recover_promoting(state: dict, check_times: list[datetime], until: datetime) -> dict:
    """Resolve a boot that finds the promotion marker. It never stays set.

    Three cases, decided from the verdict and the engine's own state rather than
    from the marker alone:

      * verdict promoted and the engine reports armed -- the chain finished just
        before the crash: adopt the run and supervise, with no second restart;
      * verdict promoted but the engine is not armed -- the crash landed inside
        the arming window: demote, because arming unattended on a half-executed
        chain is the one thing this supervisor must not do;
      * verdict not promoted -- the crash landed before the verdict existed.

    In the last two cases the hour's evaluation is spent (the gate already ran and
    passed), so `checks_done` advances: the next boot must not rebuild the same
    slice again, and the day must not end sitting on the marker. The next
    scheduled check is a real one, so the process is never left idle or blocked
    unless the schedule itself is exhausted.
    """
    verdict = read_json(VERDICT, backup=True)
    if verdict is None:
        # The marker plus an unreadable verdict and no backup: rewrite a valid
        # fail-closed record now, so nothing downstream has to interpret a
        # truncated file as "not promoted".
        demote("promotion interrupted with an unreadable verdict file")
        verdict = {}
    st = engine_state() or {}
    armed = bool(st.get("live_armed"))
    did = int(state.get("checks_done") or 0)
    log(f"recovery: promotion marker present (verdict promoted={verdict.get('promoted')}, "
        f"engine armed={armed}, checks_done={did})")
    if verdict.get("promoted") and armed:
        # The crash may have landed between restart_flat's new epoch and the
        # state save: supervise from the epoch the engine actually booted with.
        start = epoch_start()
        if start:
            state["epoch_start"] = start
        state["decision"] = "promoted"
        state["next_check"] = None
        state["next_action"] = f"15-minute supervision until {until.isoformat()}"
        log("recovery: verdict promoted and engine armed; adopting the run, no restart")
        return state
    if verdict.get("promoted"):
        demote("promotion interrupted: recovered on boot before the engine reported armed")
    state["checks_done"] = did + 1
    if did + 1 < len(check_times):
        when = check_times[did + 1].isoformat()
        state["decision"] = "pending"
        state["next_check"] = when
        state["next_action"] = f"forward OOS evaluation (#{did + 2}) at {when}"
        log(f"recovery: demoted and rescheduled to {when} (check #{did + 1} is spent)")
    else:
        state["decision"] = "blocked"
        state["next_check"] = None
        state["next_action"] = "none: promotion interrupted on the last scheduled check"
        log("recovery: demoted; no scheduled checks remain")
    return state


# ------------------------------------------------------------------ reporting
def write_report(state: dict, args, extra: dict) -> None:
    checks = state.get("checks") or []
    fwd = read_json(FORWARD_PATH) or {}
    canary = read_json(CANARY_LIVE) or {}
    verdict = read_json(VERDICT) or {}
    lines = [
        "# Evening report — 18:00 EST",
        "",
        f"Written {now().isoformat(timespec='seconds')} by `scripts/day_supervisor.py`.",
        "",
        "## Outcome",
        "",
        f"- Decision: **{state.get('decision')}**",
        f"- Forward OOS start: {args.since} ({datetime.fromtimestamp(args.since).astimezone().isoformat()})",
        f"- Verdict: promoted={verdict.get('promoted')} "
        f"program={(verdict.get('candidate') or {}).get('name')}",
        f"- Engine: {'armed' if (engine_state() or {}).get('live_armed') else 'paper'}",
        "",
        "## Forward OOS (single shot, frozen rule)",
        "",
    ]
    if fwd:
        t = fwd.get("trades") or {}
        lines += [
            f"- Windows scored: {fwd.get('windows_scored')}, events {fwd.get('events')}",
            f"- Trades after the desk hurdle: {t.get('trades')} "
            f"({t.get('wins')}W/{t.get('losses')}L)",
            f"- Net: ${t.get('net_pnl')}   PF: {t.get('pf')}",
            f"- Payoff ECE: {fwd.get('payoff_ece')}   crossed fills: {t.get('crossed_fills')}",
            f"- Max side cost: {t.get('max_side_cost')}   lanes profitable: "
            f"{fwd.get('lanes_profitable')}",
            f"- Gate: {'PASS' if (fwd.get('gate') or {}).get('pass') else 'FAIL'} "
            f"{' '.join((fwd.get('gate') or {}).get('reasons') or [])}",
            f"- By lane: `{json.dumps(t.get('by_lane') or {})}`",
            "",
        ]
    else:
        lines += ["- No forward artifact was produced.", ""]

    steps = state.get("steps") or []
    if steps:
        lines += ["## Arming sequence", "", "| Step | Result | Detail |", "|---|---|---|"]
        for s in steps:
            detail = str(s.get("detail", "")).strip().splitlines()
            lines.append(f"| {s['step']} | {'PASS' if s['ok'] else 'FAIL'} | "
                         f"`{(detail[-1] if detail else '')[:160]}` |")
        lines.append("")
    if canary:
        # Exact name on purpose: the dry-run step also starts with "live canary"
        # in its title and must not make the report claim a live canary ran.
        ran_today = any(s.get("step") == "live canary (1 contract)" for s in steps)
        stamp = canary.get("generated_ts")
        when = (datetime.fromtimestamp(float(stamp)).astimezone().isoformat()
                if stamp else "unknown time")
        lines += ["## Canary", "",
                  f"- {'run this session' if ran_today else 'NOT run this session; last recorded evidence'}"
                  f" ({when})",
                  f"- ok={canary.get('ok')} ticker={canary.get('ticker')} "
                  f"fill={canary.get('fill_price')} delta={canary.get('reconcile_delta')} "
                  f"flat={canary.get('flattened')}", ""]

    lines += ["## 15-minute supervision", "",
              "| Time | Mode | Armed | Balance | Reconcile delta | Breaker |",
              "|---|---|---|---|---|---|"]
    for c in checks[-60:]:
        lines.append(f"| {c.get('ts')} | {c.get('mode')} | {c.get('armed')} | "
                     f"{c.get('balance')} | {c.get('delta')} | {c.get('breaker') or '-'} |")
    lines += ["", "## Breakers", ""]
    trips = [c for c in checks if c.get("breaker")]
    lines.append(f"- Trips: {len(trips)}" + (
        f" ({'; '.join(str(c['breaker']) for c in trips)})" if trips else ""))
    lines += ["", "## Notes", ""]
    for line in extra.get("notes") or []:
        lines.append(f"- {line}")
    lines.append("")
    atomic_write(REPORT, "\n".join(lines))
    log(f"evening report written to {REPORT}")

    payload = read_json(NEXT) or {}
    payload.update({
        "updated": now().isoformat(timespec="seconds"),
        "goal": "Forward-OOS-verified live arming, or an honest refusal, on the 05:00-12:00 EST tape.",
        "state": {
            "decision": state.get("decision"),
            "engine_pid": ((engine_state() or {}).get("pid")
                           or read_pid()),
            "mode": (engine_state() or {}).get("mode"),
            "live_armed": (engine_state() or {}).get("live_armed"),
            "verdict_promoted": verdict.get("promoted"),
            "program": (verdict.get("candidate") or {}).get("name"),
            "venue_balance": state.get("last_balance"),
            "epoch_start": state.get("epoch_start"),
        },
        "forward_oos": {
            "start_ts": args.since,
            "trades": (fwd.get("trades") or {}).get("trades"),
            "net": (fwd.get("trades") or {}).get("net_pnl"),
            "pf": (fwd.get("trades") or {}).get("pf"),
            "payoff_ece": fwd.get("payoff_ece"),
            "gate": fwd.get("gate"),
            "artifact": "data/forward-oos.json",
        },
        "arming": {"attempted": bool(steps), "steps": steps,
                   "canary": canary or None},
        "supervision": {"checks": len(checks), "breaker_trips": len(trips),
                        "log": "data/day-supervisor.log",
                        "report": "data/evening-report.md"},
        "notes": extra.get("notes") or [],
    })
    atomic_write(NEXT, json.dumps(payload, indent=1, default=str))
    log("NEXT.json updated")


def main() -> int:
    global STATE, REPORT, NEXT, LOG, VERDICT, FORWARD_PATH, DRY_RUN
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=float, default=float(FORWARD_SINCE))
    ap.add_argument("--until", default="2026-11-17T18:00:00-05:00")
    ap.add_argument("--checks", default="2026-11-17T12:00:00-05:00,"
                                       "2026-11-17T13:00:00-05:00,"
                                       "2026-11-17T14:00:00-05:00")
    ap.add_argument("--interval", type=float, default=900.0)
    ap.add_argument("--balance-floor", default="105.00")
    ap.add_argument("--delta-cap", default="0.05")
    ap.add_argument("--min-trades", type=int, default=20)
    ap.add_argument("--no-arm", action="store_true")
    # Overridable so a smoke test can exercise the loop without rewriting the
    # handoff files the next session reads.
    ap.add_argument("--state-file", default=str(STATE))
    ap.add_argument("--report-file", default=str(REPORT))
    ap.add_argument("--next-file", default=str(NEXT))
    ap.add_argument("--log-file", default=str(LOG))
    ap.add_argument("--verdict-file", default=str(VERDICT))
    ap.add_argument("--forward-out", default=str(FORWARD_PATH))
    ap.add_argument("--forward-report", default=None,
                    help="drill only: read this artifact instead of rebuilding and scoring")
    ap.add_argument("--dry-run", action="store_true",
                    help="rehearse the promotion chain: dry canary, no restart, no arming")
    args = ap.parse_args()

    STATE, REPORT, NEXT = Path(args.state_file), Path(args.report_file), Path(args.next_file)
    LOG = Path(args.log_file)
    VERDICT = Path(args.verdict_file)
    FORWARD_PATH = Path(args.forward_out)
    DRY_RUN = bool(args.dry_run)
    floor, cap = Decimal(args.balance_floor), Decimal(args.delta_cap)
    check_times = [datetime.fromisoformat(t.strip()) for t in args.checks.split(",") if t.strip()]
    until = datetime.fromisoformat(args.until)
    state = load_state()
    if state.get("decision") == "promoting":
        state = recover_promoting(state, check_times, until)
        save_state(state)
    stop = {"flag": False}

    def handle(signum, frame):  # noqa: ANN001 - signal handler signature
        stop["flag"] = True
        log("signal received; stopping after this check")

    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)
    log(f"day supervisor up: since {args.since}, checks {[t.isoformat() for t in check_times]}, "
        f"until {until.isoformat()}, floor ${floor}, delta cap ${cap}, arm={not args.no_arm}")

    extra: dict = {"notes": []}
    exit_code = 0
    while True:
        t = now()
        st = engine_state()
        armed = bool((st or {}).get("live_armed"))
        bal = venue_balance()
        since = state.get("epoch_start") or epoch_start()
        state["epoch_start"] = since
        rec = reconcile(since) if armed else None
        delta = None
        if rec:
            try:
                delta = Decimal(str(rec.get("delta", "0")))
            except Exception:  # noqa: BLE001
                delta = None
        breaks: list[str] = []
        if armed and bal is not None and bal < floor:
            breaks.append(f"venue balance {bal} < floor {floor}")
        if armed and delta is not None and abs(delta) > cap:
            breaks.append(f"reconcile delta {delta} > cap {cap}")
        check = {"ts": t.isoformat(timespec="seconds"), "mode": (st or {}).get("mode"),
                 "armed": armed, "balance": str(bal) if bal is not None else None,
                 "delta": str(delta) if delta is not None else None,
                 "breaker": "; ".join(breaks) or None}
        state.setdefault("checks", []).append(check)
        state["checks"] = state["checks"][-200:]
        state["last_balance"] = str(bal) if bal is not None else state.get("last_balance")
        save_state(state)
        log(f"check balance={bal} mode={check['mode']} armed={armed} delta={delta}")
        if breaks:
            demote("; ".join(breaks))
            if not state.get("breaker_open"):
                # Open the trip and demote at once; the restart waits for a flat
                # venue. A breaker often trips while a position is still open,
                # and restart_flat.sh refuses to boot over one, so calling it
                # immediately would spend the trip's single attempt on a refusal.
                state["breaker_open"] = True
                state["breaker_restart_attempted"] = False
                state["decision"] = "disarmed"
                save_state(state)
                log("breaker open; verdict demoted, restart pending a flat venue")
            else:
                log("breaker already open for this trip; verdict remains demoted")
            extra["notes"].append("breaker: " + "; ".join(breaks))

        # One trip, one restart, and only once the venue is flat. Until then the
        # pending trip is re-checked; after the attempt the flag keeps every later
        # 15-minute check from restarting the same trip.
        if state.get("breaker_open") and not state.get("breaker_restart_attempted"):
            flat, flat_detail = venue_flat()
            if flat:
                code, out = restart_flat()
                state["breaker_restart_attempted"] = True
                state["breaker_restarts"] = int(state.get("breaker_restarts") or 0) + 1
                log(f"breaker restart_flat rc={code} (one attempt) {out[-160:]}")
                state["decision"] = "disarmed"
                save_state(state)
            else:
                log(f"breaker restart deferred: venue not flat ({flat_detail})")
                save_state(state)
        elif not armed and state.get("breaker_open"):
            # The engine came back flat and unarmed: the trip is closed and a
            # future breach is a new trip with its own single restart.
            log("engine is disarmed; breaker trip closed")
            state["breaker_open"] = False
            state["breaker_restart_attempted"] = False
            save_state(state)

        # The forward check is a scheduled one-shot, not part of the poll loop.
        # `checks_done` lives in the state file, so a watchdog restart resumes at
        # the next scheduled hour instead of re-scoring the slice on every bounce.
        done = int(state.get("checks_done") or 0)
        due = check_times[done] if done < len(check_times) else None
        if state.get("decision") in ("pending", "waiting") and due is not None \
                and t >= due:
            log(f"forward check due ({due.isoformat()}, #{done + 1}); starting")
            fwd = evaluate_forward(args.since, args.forward_report,
                                   slot=due, state=state)
            if fwd is None:
                # A failed rebuild or scorer spends this slot, not the day: the
                # next scheduled slot tries again, exactly once, and the 18:00
                # report is still written. No floor or threshold moves.
                extra["notes"].append(
                    f"forward evaluation at {due.isoformat()} produced no artifact")
                if done + 1 < len(check_times):
                    nxt = check_times[done + 1].isoformat()
                    state["decision"] = "waiting"
                    state["next_check"] = nxt
                    state["next_action"] = (f"forward OOS retry at {nxt} "
                                            f"(slot {due.isoformat()} produced no artifact)")
                    log(f"forward evaluation produced no artifact; next slot {nxt}")
                else:
                    state["decision"] = "blocked"
                    state["next_check"] = None
                    state["next_action"] = "none: forward evaluation produced no artifact"
            else:
                gate = artifact_gate(fwd)
                trades = int((fwd.get("trades") or {}).get("trades") or 0)
                claimed = bool((fwd.get("gate") or {}).get("pass"))
                if claimed != gate["pass"]:
                    log(f"artifact gate field says {claimed}, the numbers say "
                        f"{gate['pass']}; the numbers win")
                passed = gate["pass"]
                log(f"forward: {trades} trades, gate pass={passed}, "
                    f"reasons={gate['reasons']}")
                if passed:
                    if args.no_arm:
                        state["decision"] = "verified (arming disabled)"
                        extra["notes"].append("forward gate passed; --no-arm kept paper mode")
                    else:
                        # Mark the chain in flight before the first irreversible step.
                        # The watchdog restarts this process automatically; without
                        # this marker a crash mid-chain would re-run the canary and
                        # the arming sequence on the next boot.
                        state["decision"] = "promoting"
                        save_state(state)
                        result = attempt_promotion(fwd, state)
                        if result["ok"]:
                            extra["notes"].append(
                                "forward gate passed; promotion chain completed"
                                + (" (dry run: nothing armed)" if DRY_RUN else "; engine armed"))
                        else:
                            extra["notes"].append(
                                f"forward gate passed but arming stopped at {result['reason']}")
                elif trades < args.min_trades and done + 1 < len(check_times):
                    state["decision"] = "waiting"
                    nxt = check_times[done + 1].isoformat()
                    state["next_check"] = nxt
                    state["next_action"] = (f"forward OOS re-check at {nxt} "
                                            f"({trades} trades so far, floor {args.min_trades})")
                    log(f"only {trades} forward trades; rescheduling to {nxt} "
                        f"(never lowering the {args.min_trades}-trade floor)")
                elif trades < args.min_trades:
                    state["decision"] = "insufficient sample"
                    state["next_check"] = None
                    state["next_action"] = (f"none: {trades} forward trades by the last check; "
                                            f"floor {args.min_trades} not lowered")
                    extra["notes"].append(
                        f"only {trades} forward trades by the last check; the frozen rule "
                        f"was never testable and the {args.min_trades}-trade floor was not "
                        f"lowered")
                else:
                    state["decision"] = "rejected"
                    state["next_check"] = None
                    state["next_action"] = "none: forward gate failed on a full sample"
                    extra["notes"].append(
                        f"forward gate failed at {trades} trades: "
                        f"{'; '.join(gate['reasons'] or [])}")
            state["checks_done"] = done + 1
            save_state(state)

        if t >= until.astimezone(t.tzinfo):
            log("window closed; writing the evening handoff")
            write_report(state, args, extra)
            # Surface a day that never armed: the watchdog has already stopped
            # (the window is closed), so a non-zero code is what a shell or a
            # parent job sees instead of a silent zero after a gate/shortfall
            # refusal. A verified evaluation or a live arm is the only success.
            exit_code = 0 if state.get("decision") in (
                "promoted", "verified (dry run)", "verified (arming disabled)") else 1
            log(f"day supervisor exiting rc={exit_code} "
                f"(decision={state.get('decision')})")
            break
        if stop["flag"]:
            # A manual stop is not the 18:00 handoff: writing one now would replace
            # the session's real report with a half-finished one. The watchdog
            # restarts this script, which resumes from `data/day-state.json`.
            log("stopped before the window closed; state saved for resume")
            save_state(state)
            break
        nxt = until.astimezone(t.tzinfo)
        idx = int(state.get("checks_done") or 0)
        if check_times and state.get("decision") in ("pending", "waiting") \
                and idx < len(check_times):
            upcoming = check_times[idx].astimezone(t.tzinfo)
            state["next_check"] = upcoming.isoformat()
            state["next_action"] = f"forward OOS evaluation (#{idx + 1}) at {upcoming.isoformat()}"
            if upcoming > t:
                nxt = min(nxt, upcoming)
        elif state.get("decision") == "promoted":
            state["next_action"] = f"15-minute supervision until {until.isoformat()}"
            state["next_check"] = None
        elif state.get("decision") in ("pending", "waiting"):
            # Reachable only with a state file that outlived its schedule. Say so
            # instead of looking like a normal wait that will never fire.
            log(f"warning: decision {state.get('decision')} with no scheduled checks left; "
                f"nothing further will fire in this window")
            state["next_action"] = f"none: {state.get('decision')}, schedule exhausted"
            state["next_check"] = None
        else:
            state["next_action"] = (f"none: {state.get('decision')}; "
                                    f"paper recording continues to {until.isoformat()}")
            state["next_check"] = None
        save_state(state)
        nap = min(args.interval, max(5.0, (nxt - t).total_seconds()))
        time.sleep(nap)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
