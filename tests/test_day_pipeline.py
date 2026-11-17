#!/usr/bin/env python
"""Drills for the scheduled day pipeline, stages 2-10.

Stage 1 (the real 12:00 evaluation) is run through the CLI against the live tape;
this file proves the parts that must not be rehearsed on live money:

  2. the gate blocks on each of the seven conditions alone, and refuses an
     artifact whose `gate.pass` field disagrees with its own numbers;
  3. the promotion chain runs in order in a dry run -- dry canary, recorded
     restart -- and a single failing step demotes the verdict and stops;
  4. the balance/delta breaker fires at 104.99 or 0.06, not at 105.00 or 0.05,
     and one trip cannot restart the engine twice.

Everything imports the real `scripts/day_supervisor.py` and calls its real
functions. Only the outside world is replaced: subprocess.run, the venue balance,
/api/state, reconcile, and restart_flat. Nothing here can arm, restart or spend.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from openkal import features  # noqa: E402
from scripts import day_supervisor as D  # noqa: E402
from scripts import forward_oos as F  # noqa: E402
from scripts.forward_oos import gate_from_summary  # noqa: E402

SANDBOX = ROOT / "data" / "drill-sandbox"
SANDBOX.mkdir(parents=True, exist_ok=True)
QUARANTINE = ROOT / "data" / "drill-quarantine-2026-11-17"
CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"{'ok  ' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not ok else ""))
    return bool(ok)


def read(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


@contextmanager
def sandbox(**overrides):
    """Point the supervisor's globals at a sandbox and put them back afterwards."""
    saved = {k: getattr(D, k) for k in overrides}
    try:
        for key, value in overrides.items():
            setattr(D, key, value)
        yield
    finally:
        for key, value in saved.items():
            setattr(D, key, value)


def seed_verdict(promoted: bool, path: Path) -> Path:
    path.write_text(json.dumps({
        "generated_ts": time.time(),
        "generated": "2026-11-17T05:21:00-05:00",
        "promoted": promoted,
        "candidate": {"name": "F2_taker_1s", "probability_target": "payoff"},
        "oos": {"trades": 46, "net": "38.54", "pf": 2.324, "payoff_ece": 0.0764},
    }, indent=1))
    return path


def passing_forward() -> dict:
    """A gate-passing artifact built from the frozen read-slice study.

    The numbers are the ones `data/f2-repro.json` actually measured (72 trades,
    +$93.18, PF 5.50, payoff ECE 0.0115, 0 crossed, max cost 0.59, three lanes
    up). Only the slice label is rewritten, and the file says it is injected:
    this is the promotion drill's input, never forward evidence.
    """
    repro = read(ROOT / "data" / "f2-repro.json")
    trades = dict(repro.get("trades") or {})
    if not trades:  # a fresh checkout should still be able to run the drill
        trades = {"trades": 72, "wins": 69, "losses": 3, "win_rate": 0.9583,
                  "net_pnl": 93.18, "pf": 5.5014, "crossed_fills": 0,
                  "max_side_cost": 0.59, "taker_fees_paid": 63.82,
                  "by_lane": {"KXBTC15M": {"net": 41.0}, "KXETH15M": {"net": 33.0},
                              "KXXRP15M": {"net": 19.0}}}
    return {
        "since_ts": float(D.FORWARD_SINCE),
        "since_local": "2026-11-17T05:00:00-05:00",
        "payoff_ece": 0.0115,
        "lanes_profitable": sum(1 for v in (trades.get("by_lane") or {}).values()
                                if float(v.get("net") or 0) > 0),
        "trades": trades,
        "gate": {"pass": True, "reasons": []},
        "windows_scored": 130,
        "generated": datetime.now(timezone.utc).isoformat(),
        "drill": "INJECTED from data/f2-repro.json (read slice), not forward evidence",
    }


def quiet_forward(trades: int = 5) -> dict:
    """A gate-failing artifact for the quiet-flow drills (never forward evidence)."""
    lanes = {"KXBTC15M": {"trades": max(1, trades - 2), "net": 1.0},
             "KXETH15M": {"trades": 2, "net": 0.5}}
    return {
        "since_ts": float(D.FORWARD_SINCE),
        "since_local": "2026-11-17T05:00:00-05:00",
        "generated": datetime.now(timezone.utc).isoformat(),
        "payoff_ece": 0.02,
        "lanes_profitable": 2,
        "trades": {
            "trades": trades, "wins": trades, "losses": 0, "win_rate": 1.0,
            "net_pnl": 1.0, "pf": 2.0, "crossed_fills": 0,
            "max_side_cost": 0.5, "by_lane": lanes,
        },
        "gate": {"pass": False, "reasons": [f"{trades} forward trades < 20"]},
        "windows_scored": 3,
        "drill": "INJECTED quiet-flow artifact, not forward evidence",
    }


def forward_artifact(trades: int = 22, *, net: float = 1.0, pf: float = 2.0,
                     ece: float = 0.02, crossed: int = 0,
                     max_cost: float = 0.5,
                     lanes: dict | None = None) -> dict:
    """A controllable artifact for the retry/gate drills (never forward evidence)."""
    lanes = lanes or {"KXBTC15M": {"trades": max(1, trades - 2), "net": 1.0},
                      "KXETH15M": {"trades": 2, "net": 0.5}}
    summary = {
        "trades": trades, "wins": trades, "losses": 0, "win_rate": 1.0,
        "net_pnl": net, "pf": pf, "crossed_fills": crossed,
        "max_side_cost": max_cost, "by_lane": lanes,
    }
    return {
        "since_ts": float(D.FORWARD_SINCE),
        "since_local": "2026-11-17T05:00:00-05:00",
        "generated": datetime.now(timezone.utc).isoformat(),
        "payoff_ece": ece,
        "lanes_profitable": sum(1 for v in lanes.values()
                                if float(v.get("net") or 0) > 0),
        "trades": summary,
        "gate": gate_from_summary(summary, ece),
        "windows_scored": 130,
        "drill": "INJECTED retry artifact, not forward evidence",
    }


class Recorder:
    """Stands in for subprocess.run and records every command it is given.

    `forward_trades` makes the stub scorer write a quiet-flow artifact to the
    `--out` path the supervisor passed, exactly where the real scorer would;
    `rebuild_fail` makes the cache rebuild exit non-zero. Neither touches the
    real tape or the real forward artifact.
    """

    def __init__(self, fail_on: str | None = None, *,
                 forward_trades: int | None = None,
                 forward_artifacts: list[dict] | None = None,
                 score_fail_at: int | None = None,
                 rebuild_fail: bool = False) -> None:
        self.calls: list[dict] = []
        self.fail_on = fail_on
        self.forward_trades = forward_trades
        self.forward_artifacts = (list(forward_artifacts)
                                  if forward_artifacts is not None else None)
        self.score_fail_at = score_fail_at
        self.rebuild_fail = rebuild_fail
        self.rebuilds = 0
        self.scores = 0
        self.score_rcs: list[int] = []
        self.score_argv: list[list[str]] = []
        self.score_trades: list[int] = []

    def __call__(self, cmd, timeout=None, cwd=None, env=None):
        joined = " ".join(str(c) for c in cmd)
        self.calls.append({"cmd": [str(c) for c in cmd], "cwd": str(cwd or ""),
                           "env": dict(env or {})})
        if "canary.py" in joined and "--live" not in joined:
            D.CANARY_DRY.write_text(json.dumps(
                {"generated_ts": time.time(), "contracts": 1,
                 "dry_plan": {"ticker": "KXBTC15M", "side": "yes", "qty": 1},
                 "ticker": "KXBTC15M", "note": "stub"}, indent=1))
        if "rebuild_cache.sh" in joined:
            self.rebuilds += 1
            if self.rebuild_fail:
                return 1, "INJECTED rebuild failure"
        if "forward_oos.py" in joined:
            self.scores += 1
            idx = self.scores - 1
            rc = 1 if self.score_fail_at == idx else 0
            if self.forward_artifacts is not None:
                art = self.forward_artifacts[min(idx, len(self.forward_artifacts) - 1)]
                out = Path(cmd[cmd.index("--out") + 1])
                out.write_text(json.dumps(art, indent=1, default=str))
                n = int((art.get("trades") or {}).get("trades") or 0)
            elif self.forward_trades is not None:
                out = Path(cmd[cmd.index("--out") + 1])
                out.write_text(json.dumps(quiet_forward(self.forward_trades),
                                          indent=1, default=str))
                n = self.forward_trades
            else:
                n = 0
            self.score_rcs.append(rc)
            self.score_argv.append([str(c) for c in cmd])
            self.score_trades.append(n)
            return rc, f"stub score rc={rc} ({n} trades)"
        if self.fail_on and self.fail_on in joined:
            return 1, f"INJECTED FAILURE: {joined}"
        if "render-check" in joined:
            return 0, "6/6 PASSED"
        return 0, f"stub ok: {joined}"


EXPECTED_STEPS = [
    "gate re-check on the artifact's numbers",
    "unit tests: live exit",
    "unit tests: controls (parity)",
    "unit tests: controls (prod)",
    "unit tests: controls bind",
    "unit tests: day pipeline",
    "dashboard build",
    "dashboard role-coverage",
    "dashboard render-check",
    "venue flat",
    "live canary (1 contract) [DRY RUN: canary.py without --live]",
    "venue flat after the canary",
    "armed via restart_flat.sh --new-epoch [DRY RUN]",
]


# --------------------------------------------------------------------- stage 2
def stage_two() -> None:
    base = passing_forward()["trades"]
    ok = gate_from_summary(base, 0.0115)
    check("2.0 all seven conditions -> pass", ok["pass"], str(ok))

    cases = {
        "net": (dict(base, net_pnl=-0.01), 0.0115),
        "pf": (dict(base, pf=1.1499), 0.0115),
        "payoff-ECE": (base, 0.0801),
        "trade count": (dict(base, trades=19), 0.0115),
        "crossed fills": (dict(base, crossed_fills=1), 0.0115),
        "cost cap": (dict(base, max_side_cost=0.6201), 0.0115),
        "lanes": (dict(base, by_lane={"KXBTC15M": {"net": 41.0}}), 0.0115),
    }
    for name, (summary, ece) in cases.items():
        gate = gate_from_summary(summary, ece)
        alone = len(gate["reasons"]) == 1
        check(f"2.1 {name} failing alone blocks", not gate["pass"] and alone,
              str(gate["reasons"]))

    # The real stage-1 artifact, if the 12:00 drill has already run, must block.
    live = read(QUARANTINE / "forward-oos.json")
    if live:
        gate = D.artifact_gate(live)
        check("2.2 real forward slice (below the trade floor) blocks",
              not gate["pass"], str(gate["reasons"]))

    # A lying artifact: `gate.pass` true, one number out of bounds.
    liar = passing_forward()
    liar["trades"] = dict(liar["trades"], max_side_cost=0.63)
    liar["gate"] = {"pass": True, "reasons": []}
    check("2.3 artifact claiming pass with a failing number is refused",
          not D.artifact_gate(liar)["pass"], str(D.artifact_gate(liar)["reasons"]))

    d = SANDBOX / "t2"
    d.mkdir(exist_ok=True)
    rec = Recorder()
    with sandbox(STATE=d / "state.json", REPORT=d / "report.md", NEXT=d / "next.json",
                 LOG=d / "log.txt", VERDICT=d / "verdict.json",
                 FORWARD_PATH=d / "forward.json", CANARY_DRY=d / "canary-dry.json",
                 DRY_RUN=True, run=rec,
                 venue_flat=lambda: (True, "flat"),
                 restart_flat=lambda new_epoch=False: (0, "SHOULD NOT RUN")):
        seed_verdict(False, D.VERDICT)
        state = {"decision": "pending", "checks_done": 0, "checks": []}
        D.save_state(state)
        result = D.attempt_promotion(liar, state)
        verdict = read(D.VERDICT)
        check("2.4 promotion refuses before writing a verdict",
              not result["ok"] and result["reason"] == "gate re-check"
              and verdict.get("promoted") is False and not rec.calls,
              f"ok={result['ok']} reason={result['reason']} promoted={verdict.get('promoted')} "
              f"calls={len(rec.calls)}")


# --------------------------------------------------------------------- stage 3
def stage_three() -> None:
    fwd = passing_forward()
    d = SANDBOX / "t3"
    d.mkdir(exist_ok=True)
    rec = Recorder()
    with sandbox(STATE=d / "state.json", REPORT=d / "report.md", NEXT=d / "next.json",
                 LOG=d / "log.txt", VERDICT=d / "verdict.json",
                 FORWARD_PATH=d / "forward.json", CANARY_DRY=d / "canary-dry.json",
                 DRY_RUN=True, run=rec,
                 venue_flat=lambda: (True, "flat, no resting orders"),
                 restart_flat=lambda new_epoch=False: (0, "SHOULD NOT RUN")):
        seed_verdict(False, D.VERDICT)
        state = {"decision": "pending", "checks_done": 0, "checks": [],
                 "epoch_start": float(D.FORWARD_SINCE)}
        D.save_state(state)
        result = D.attempt_promotion(fwd, state)
        verdict = read(D.VERDICT)
        names = [s["step"] for s in result["steps"]]
        check("3.1 promotion chain completes in dry run", result["ok"], str(result.get("reason")))
        check("3.2 every step ran, in the documented order", names == EXPECTED_STEPS, str(names))
        check("3.3 every step passed", all(s["ok"] for s in result["steps"]),
              str([s["step"] for s in result["steps"] if not s["ok"]]))
        check("3.4 verdict promoted in the sandbox copy", verdict.get("promoted") is True)
        check("3.5 no command was a live canary and none restarted the engine",
              all("--live" not in " ".join(c["cmd"]) for c in rec.calls)
              and all("restart_flat" not in " ".join(c["cmd"]) for c in rec.calls),
              str([c["cmd"] for c in rec.calls]))
        check("3.6 nine commands were issued (5 tests, 3 dashboard, 1 dry canary)",
              len(rec.calls) == 9, str([c["cmd"][:3] for c in rec.calls]))
        check("3.7 decision is a dry-run verdict, not 'promoted'",
              state.get("decision") == "verified (dry run)", str(state.get("decision")))
        check("3.8 the arm step recorded the exact command it would run",
              "OPENKAL_ARM_LIVE=1 bash scripts/restart_flat.sh --new-epoch"
              in (result.get("would_run") or ""), str(result.get("would_run")))

    # A single failing intermediate step must demote and stop.
    d = SANDBOX / "t3-fail"
    d.mkdir(exist_ok=True)
    rec = Recorder(fail_on="role-coverage")
    with sandbox(STATE=d / "state.json", REPORT=d / "report.md", NEXT=d / "next.json",
                 LOG=d / "log.txt", VERDICT=d / "verdict.json",
                 FORWARD_PATH=d / "forward.json", CANARY_DRY=d / "canary-dry.json",
                 DRY_RUN=True, run=rec,
                 venue_flat=lambda: (True, "flat"),
                 restart_flat=lambda new_epoch=False: (0, "SHOULD NOT RUN")):
        seed_verdict(False, D.VERDICT)
        state = {"decision": "pending", "checks_done": 0, "checks": []}
        D.save_state(state)
        result = D.attempt_promotion(fwd, state)
        verdict = read(D.VERDICT)
        names = [s["step"] for s in result["steps"]]
        check("3.9 a failing step stops the chain",
              not result["ok"] and result["reason"] == "dashboard role-coverage"
              and names[-1] == "dashboard role-coverage", str(names))
        check("3.10 nothing runs after the failing step",
              not any("canary" in n or "restart" in n for n in names), str(names))
        check("3.11 the failure demotes the verdict again",
              verdict.get("promoted") is False
              and "role-coverage" in (verdict.get("disarm_reason") or ""),
              str(verdict.get("disarm_reason")))
        check("3.12 the engine was never restarted",
              not any("restart_flat" in " ".join(c["cmd"]) for c in rec.calls)
              and state.get("decision") == "blocked", str(state.get("decision")))


# --------------------------------------------------------------------- stage 4
def breaker_case(name: str, balance: str, delta: str, expect: int,
                 seconds: float = 5.0) -> None:
    d = SANDBOX / f"t4-{name}"
    d.mkdir(exist_ok=True)
    seed_verdict(True, d / "verdict.json")
    (d / "state.json").write_text(json.dumps(
        {"decision": "pending", "checks_done": 0, "checks": [],
         "epoch_start": float(D.FORWARD_SINCE)}, indent=1))
    restarts = {"n": 0}

    def fake_restart(new_epoch: bool = False):
        restarts["n"] += 1
        return 0, "stub restart (engine untouched)"

    until = datetime.now(timezone.utc) + timedelta(seconds=seconds)
    argv = [
        "day_supervisor.py", "--since", str(int(D.FORWARD_SINCE)),
        "--until", until.isoformat(), "--checks", "2026-11-17T23:00:00-05:00",
        "--interval", "1",
        "--state-file", str(d / "state.json"), "--report-file", str(d / "report.md"),
        "--next-file", str(d / "next.json"), "--log-file", str(d / "log.txt"),
        "--verdict-file", str(d / "verdict.json"), "--forward-out", str(d / "forward.json"),
        "--dry-run",
    ]
    with sandbox(STATE=d / "state.json", REPORT=d / "report.md", NEXT=d / "next.json",
                 LOG=d / "log.txt", VERDICT=d / "verdict.json",
                 FORWARD_PATH=d / "forward.json", DRY_RUN=True,
                 venue_balance=lambda: Decimal(balance),
                 engine_state=lambda: {"mode": "live", "live_armed": True, "pid": 775298},
                 reconcile=lambda since=None: {"delta": str(delta)},
                 restart_flat=fake_restart,
                 epoch_start=lambda: float(D.FORWARD_SINCE),
                 venue_flat=lambda: (True, "flat")):
        sys.argv = argv
        D.main()
    state = read(d / "state.json")
    verdict = read(d / "verdict.json")
    checks = state.get("checks") or []
    tripped = [c for c in checks if c.get("breaker")]
    label = f"balance {balance} delta {delta}"
    check(f"4.x {name}: {label} -> {expect} restart(s)",
          restarts["n"] == expect, f"got {restarts['n']} over {len(checks)} checks")
    if expect:
        check(f"4.y {name}: breaker demotes the verdict", verdict.get("promoted") is False,
              str(verdict.get("promoted")))
        check(f"4.z {name}: one trip records one restart",
              int(state.get("breaker_restarts") or 0) == 1 and bool(tripped)
              and all("balance 104.99" in (c.get("breaker") or "")
                      or "delta 0.06" in (c.get("breaker") or "") for c in tripped),
              f"restarts={state.get('breaker_restarts')} trips={len(tripped)}")
        check(f"4.w {name}: every later check suppressed a second restart",
              len(checks) >= 3 and all(not c.get("armed") or c.get("breaker") is not None
                                       for c in checks),
              f"{len(checks)} checks")
    else:
        check(f"4.y {name}: untouched verdict stays promoted", verdict.get("promoted") is True,
              str(verdict.get("promoted")))
        check(f"4.z {name}: no breaker recorded", not tripped and not state.get("breaker_open"),
              f"trips={len(tripped)}")


def stage_three_marker() -> None:
    """The real decision branch marks the chain in flight, then completes it."""
    d = SANDBOX / "t3-marker"
    d.mkdir(exist_ok=True)
    seed_verdict(False, d / "verdict.json")
    (d / "forward-injected-pass.json").write_text(json.dumps(passing_forward(), indent=1, default=str))
    (d / "state.json").write_text(json.dumps(
        {"decision": "pending", "checks_done": 0, "checks": [],
         "epoch_start": float(D.FORWARD_SINCE)}, indent=1))
    until = datetime.now(timezone.utc) + timedelta(seconds=2)
    argv = [
        "day_supervisor.py", "--since", str(int(D.FORWARD_SINCE)),
        "--until", until.isoformat(), "--checks", "2026-11-17T00:00:00-05:00",
        "--interval", "1", "--forward-report", str(d / "forward-injected-pass.json"),
        "--state-file", str(d / "state.json"), "--report-file", str(d / "report.md"),
        "--next-file", str(d / "next.json"), "--log-file", str(d / "log.txt"),
        "--verdict-file", str(d / "verdict.json"), "--forward-out", str(d / "forward.json"),
        "--dry-run",
    ]
    with sandbox(STATE=d / "state.json", REPORT=d / "report.md", NEXT=d / "next.json",
                 LOG=d / "log.txt", VERDICT=d / "verdict.json",
                 FORWARD_PATH=d / "forward.json", DRY_RUN=True,
                 venue_balance=lambda: Decimal("130.49"),
                 engine_state=lambda: {"mode": "paper", "live_armed": False, "pid": 775298},
                 reconcile=lambda since=None: None,
                 restart_flat=lambda new_epoch=False: (0, "SHOULD NOT RUN"),
                 epoch_start=lambda: float(D.FORWARD_SINCE),
                 venue_flat=lambda: (True, "flat"),
                 run=Recorder()):
        sys.argv = argv
        D.main()
    state = read(d / "state.json")
    verdict = read(d / "verdict.json")
    steps = state.get("steps") or []
    check("3.14 the real decision branch runs the chain and ends in a dry-run verdict",
          state.get("decision") == "verified (dry run)"
          and len(steps) == len(EXPECTED_STEPS)
          and all(s["ok"] for s in steps) and verdict.get("promoted") is True,
          f"decision={state.get('decision')} steps={len(steps)} promoted={verdict.get('promoted')}")


def stage_three_resume() -> None:
    """A watchdog restart in the middle of the chain must fail closed."""
    d = SANDBOX / "t3-resume"
    d.mkdir(exist_ok=True)
    seed_verdict(True, d / "verdict.json")
    (d / "state.json").write_text(json.dumps(
        {"decision": "promoting", "checks_done": 0, "checks": [],
         "epoch_start": float(D.FORWARD_SINCE),
         "steps": [{"step": "unit tests: live exit", "ok": True}]}, indent=1))
    until = datetime.now(timezone.utc) + timedelta(seconds=2)
    argv = [
        "day_supervisor.py", "--since", str(int(D.FORWARD_SINCE)),
        "--until", until.isoformat(), "--checks", "2026-11-17T23:00:00-05:00",
        "--interval", "1",
        "--state-file", str(d / "state.json"), "--report-file", str(d / "report.md"),
        "--next-file", str(d / "next.json"), "--log-file", str(d / "log.txt"),
        "--verdict-file", str(d / "verdict.json"), "--forward-out", str(d / "forward.json"),
        "--dry-run",
    ]
    with sandbox(STATE=d / "state.json", REPORT=d / "report.md", NEXT=d / "next.json",
                 LOG=d / "log.txt", VERDICT=d / "verdict.json",
                 FORWARD_PATH=d / "forward.json", DRY_RUN=True,
                 venue_balance=lambda: Decimal("130.49"),
                 engine_state=lambda: {"mode": "paper", "live_armed": False, "pid": 775298},
                 reconcile=lambda since=None: None,
                 restart_flat=lambda new_epoch=False: (0, "SHOULD NOT RUN"),
                 epoch_start=lambda: float(D.FORWARD_SINCE),
                 venue_flat=lambda: (True, "flat"),
                 run=Recorder()):
        sys.argv = argv
        D.main()
    state = read(d / "state.json")
    verdict = read(d / "verdict.json")
    check("3.13 a restart mid-promotion fails closed and demotes",
          state.get("decision") == "blocked" and verdict.get("promoted") is False
          and "promotion" in (verdict.get("disarm_reason") or ""),
          f"decision={state.get('decision')} promoted={verdict.get('promoted')} "
          f"reason={verdict.get('disarm_reason')}")


def stage_four() -> None:
    breaker_case("floor-trip", "104.99", "0.00", 1)
    breaker_case("floor-edge", "105.00", "0.00", 0)
    breaker_case("delta-trip", "130.49", "0.06", 1)
    breaker_case("delta-edge", "130.49", "0.05", 0)


def boot(d: Path, *, armed: bool, checks: str = "2026-11-17T23:00:00-05:00",
         seconds: float = 2.0, forward: str | None = None):
    """Run the real CLI's main() against a sandbox. Returns (state, verdict, restarts)."""
    restarts = {"n": 0}

    def fake_restart(new_epoch: bool = False):
        restarts["n"] += 1
        return 0, "stub restart (engine untouched)"

    # A fixed morning clock: the recovery drills schedule 12:00/13:00 checks and
    # must not fire them just because the suite is run in the afternoon.
    clock = [datetime.fromisoformat("2026-11-17T11:00:00-05:00")]

    def fake_now() -> datetime:
        return clock[0]

    def fake_sleep(secs: float) -> None:
        clock[0] += timedelta(seconds=float(secs))

    until = clock[0] + timedelta(seconds=seconds)
    argv = [
        "day_supervisor.py", "--since", str(int(D.FORWARD_SINCE)),
        "--until", until.isoformat(), "--checks", checks, "--interval", "1",
        "--state-file", str(d / "state.json"), "--report-file", str(d / "report.md"),
        "--next-file", str(d / "next.json"), "--log-file", str(d / "log.txt"),
        "--verdict-file", str(d / "verdict.json"), "--forward-out", str(d / "forward.json"),
        "--dry-run",
    ]
    if forward:
        argv += ["--forward-report", forward]
    with sandbox(STATE=d / "state.json", REPORT=d / "report.md", NEXT=d / "next.json",
                 LOG=d / "log.txt", VERDICT=d / "verdict.json",
                 FORWARD_PATH=d / "forward.json", DRY_RUN=True,
                 venue_balance=lambda: Decimal("130.49"),
                 engine_state=lambda: {"mode": "live" if armed else "paper",
                                       "live_armed": armed, "pid": 775298},
                 reconcile=lambda since=None: {"delta": "0.00"},
                 restart_flat=fake_restart,
                 epoch_start=lambda: float(D.FORWARD_SINCE),
                 venue_flat=lambda: (True, "flat"), now=fake_now,
                 run=Recorder()):
        with mock.patch.object(D.time, "sleep", fake_sleep):
            sys.argv = argv
            D.main()
    return read(d / "state.json"), read(d / "verdict.json"), restarts["n"]


def seeded_marker(d: Path, verdict_promoted: bool, checks_done: int = 0) -> None:
    d.mkdir(exist_ok=True)
    seed_verdict(verdict_promoted, d / "verdict.json")
    (d / "state.json").write_text(json.dumps(
        {"decision": "promoting", "checks_done": checks_done, "checks": [],
         "epoch_start": float(D.FORWARD_SINCE),
         "steps": [{"step": "gate re-check on the artifact's numbers", "ok": True}]}, indent=1))


def stage_five() -> None:
    """Crash recovery: the promoting marker, and truncated state/verdict files."""
    # 5.1 A boot that finds the marker with the engine already armed adopts the run.
    d = SANDBOX / "t5-armed"
    seeded_marker(d, True)
    state, verdict, restarts = boot(d, armed=True)
    check("5.1 marker + promoted verdict + armed engine -> adopt, no restart",
          state.get("decision") == "promoted" and verdict.get("promoted") is True
          and restarts == 0 and "supervision" in (state.get("next_action") or ""),
          f"decision={state.get('decision')} restarts={restarts} action={state.get('next_action')}")

    # 5.2 marker + promoted verdict + unarmed engine: demote, spend the hour, reschedule.
    d = SANDBOX / "t5-unarmed"
    seeded_marker(d, True)
    state, verdict, restarts = boot(
        d, armed=False,
        checks="2026-11-17T12:00:00-05:00,2026-11-17T13:00:00-05:00")
    check("5.2 marker + promoted verdict + unarmed engine -> demote and reschedule",
          state.get("decision") == "pending" and verdict.get("promoted") is False
          and int(state.get("checks_done") or 0) == 1
          and state.get("next_check") == "2026-11-17T13:00:00-05:00"
          and restarts == 0,
          f"decision={state.get('decision')} promoted={verdict.get('promoted')} "
          f"checks_done={state.get('checks_done')} next={state.get('next_check')}")

    # 5.3 Truncated verdict with a valid .bak: demote from the recovered contents.
    d = SANDBOX / "t5-truncated-verdict"
    d.mkdir(exist_ok=True)
    good = json.dumps({"generated": "x", "promoted": True,
                       "candidate": {"name": "F2_taker_1s"}}, indent=1)
    D.atomic_write(d / "verdict.json", good)          # first write: no backup yet
    D.atomic_write(d / "verdict.json", good)          # second write: .bak holds `good`
    (d / "verdict.json").write_text('{"promoted": tr')
    with sandbox(VERDICT=d / "verdict.json", LOG=d / "log.txt"):
        ok = D.demote("drill: truncated verdict")
    recovered = read(d / "verdict.json")
    check("5.3 truncated verdict is read back from .bak and demoted, not silently empty",
          ok and recovered.get("promoted") is False
          and "truncated verdict" in (recovered.get("disarm_reason") or "")
          and recovered.get("candidate", {}).get("name") == "F2_taker_1s",
          f"ok={ok} parsed={bool(recovered)} promoted={recovered.get('promoted')} "
          f"reason={recovered.get('disarm_reason')}")

    # 5.4 Truncated state with a valid .bak: the schedule survives the crash.
    d = SANDBOX / "t5-truncated-state"
    d.mkdir(exist_ok=True)
    body = json.dumps({"decision": "pending", "checks_done": 2,
                       "next_check": "2026-11-17T14:00:00-05:00"}, indent=1)
    D.atomic_write(d / "state.json", body)
    D.atomic_write(d / "state.json", body)
    (d / "state.json").write_text('{"decision": "pen')
    with sandbox(STATE=d / "state.json", LOG=d / "log.txt"):
        recovered = D.load_state()
    check("5.4 truncated state is read back from .bak, schedule intact",
          int(recovered.get("checks_done") or 0) == 2
          and recovered.get("next_check") == "2026-11-17T14:00:00-05:00",
          str({k: recovered.get(k) for k in ("checks_done", "next_check")}))

    # 5.5 Truncated verdict with no backup: reset to a valid fail-closed file.
    d = SANDBOX / "t5-no-backup"
    seeded_marker(d, True)
    (d / "verdict.json").write_text('{"promo')
    state, verdict, restarts = boot(
        d, armed=False,
        checks="2026-11-17T12:00:00-05:00,2026-11-17T13:00:00-05:00")
    check("5.5 unreadable verdict without a backup becomes a valid not-promoted record",
          verdict.get("promoted") is False
          and "unreadable" in (verdict.get("recovery") or "")
          and state.get("decision") == "pending"
          and state.get("next_check") == "2026-11-17T13:00:00-05:00",
          f"verdict={verdict} decision={state.get('decision')} next={state.get('next_check')}")

    # 5.6 Truncated state with no backup: defaults to pending, never an exception.
    d = SANDBOX / "t5-state-no-backup"
    d.mkdir(exist_ok=True)
    (d / "state.json").write_text('{"decision": "wai')
    with sandbox(STATE=d / "state.json", LOG=d / "log.txt"):
        recovered = D.load_state()
    check("5.6 unreadable state without a backup falls back to a pending schedule",
          recovered.get("decision") == "pending" and recovered.get("checks") == [], str(recovered))

    # 5.7 atomic_write keeps the previous valid contents beside the file.
    d = SANDBOX / "t5-bak"
    d.mkdir(exist_ok=True)
    D.atomic_write(d / "v.json", json.dumps({"n": 1}))
    D.atomic_write(d / "v.json", json.dumps({"n": 2}))
    bak = read(d / "v.json.bak")
    check("5.7 atomic writes leave the previous contents in .bak",
          bak.get("n") == 1 and read(d / "v.json").get("n") == 2, str(bak))

    # 5.8 A crash between the verdict write and the arm step: no double arm, no strand.
    d = SANDBOX / "t5-arm-window"
    seeded_marker(d, True)
    st = read(d / "state.json")
    st["steps"] = [
        {"step": s, "ok": True, "detail": "drill"} for s in EXPECTED_STEPS[:-1]
    ]
    (d / "state.json").write_text(json.dumps(st, indent=1))
    state, verdict, restarts = boot(
        d, armed=False,
        checks="2026-11-17T12:00:00-05:00,2026-11-17T13:00:00-05:00")
    check("5.8 kill between verdict and arm -> one clean demote, zero restarts, next check",
          verdict.get("promoted") is False and restarts == 0
          and state.get("decision") == "pending"
          and state.get("next_check") == "2026-11-17T13:00:00-05:00",
          f"promoted={verdict.get('promoted')} restarts={restarts} "
          f"decision={state.get('decision')} next={state.get('next_check')}")


def drive(d: Path, *, checks: str, rec: Recorder, seconds: float = 3.0,
          state: dict | None = None, clean_forward: bool = True) -> None:
    """Run the real CLI's main() against a fresh sandbox for `seconds`."""
    d.mkdir(exist_ok=True)
    for name in ("report.md", "next.json", "log.txt", "state.json"):
        (d / name).unlink(missing_ok=True)
    if clean_forward:
        (d / "forward.json").unlink(missing_ok=True)
    if state is None:
        state = {"decision": "pending", "checks_done": 0, "checks": []}
    (d / "state.json").write_text(json.dumps(state, indent=1))
    until = datetime.now(timezone.utc) + timedelta(seconds=seconds)
    argv = [
        "day_supervisor.py", "--since", str(int(D.FORWARD_SINCE)),
        "--until", until.isoformat(), "--checks", checks, "--interval", "0.1",
        "--state-file", str(d / "state.json"), "--report-file", str(d / "report.md"),
        "--next-file", str(d / "next.json"), "--log-file", str(d / "log.txt"),
        "--verdict-file", str(d / "verdict.json"), "--forward-out", str(d / "forward.json"),
        "--dry-run",
    ]
    with sandbox(STATE=d / "state.json", REPORT=d / "report.md", NEXT=d / "next.json",
                 LOG=d / "log.txt", VERDICT=d / "verdict.json",
                 FORWARD_PATH=d / "forward.json", CANARY_DRY=d / "canary-dry.json",
                 DRY_RUN=True, run=rec,
                 venue_balance=lambda: Decimal("130.49"),
                 engine_state=lambda: {"mode": "paper", "live_armed": False,
                                       "pid": 775298},
                 reconcile=lambda since=None: None,
                 restart_flat=lambda new_epoch=False: (0, "SHOULD NOT RUN"),
                 epoch_start=lambda: float(D.FORWARD_SINCE),
                 venue_flat=lambda: (True, "flat")):
        sys.argv = argv
        D.main()


def drive_clock(d: Path, *, checks: str, rec: Recorder, until: str,
                start: str = "2026-11-17T08:30:00-05:00",
                state: dict | None = None) -> None:
    """Run the real CLI against a sandbox with a deterministic clock.

    The scheduler's own `now()` and `time.sleep()` are replaced so the drill can
    walk 12:00 -> 13:00 -> 14:00 -> 18:00 in milliseconds without waiting or
    touching the live supervisor. Everything else (gate, verdict, report) is real.
    """
    d.mkdir(exist_ok=True)
    for name in ("report.md", "next.json", "log.txt", "state.json", "forward.json"):
        (d / name).unlink(missing_ok=True)
    if state is None:
        state = {"decision": "pending", "checks_done": 0, "checks": []}
    (d / "state.json").write_text(json.dumps(state, indent=1))
    seed_verdict(False, d / "verdict.json")
    clock = [datetime.fromisoformat(start)]

    def fake_now() -> datetime:
        return clock[0]

    def fake_sleep(seconds: float) -> None:
        # The sandbox artifact is written by the stub with a real mtime; stamp it
        # with the fake clock at the moment it was scored, not at the end of the
        # sleep, so the next fake slot cannot "adopt" it as fresh.
        stamp = clock[0].timestamp()
        clock[0] += timedelta(seconds=float(seconds))
        fwd = d / "forward.json"
        if fwd.exists() and abs(time.time() - fwd.stat().st_mtime) < 5:
            # Only a just-written artifact (real mtime) is re-stamped; later
            # sleeps must not walk its mtime forward onto the next slot.
            os.utime(fwd, (stamp, stamp))

    argv = [
        "day_supervisor.py", "--since", str(int(D.FORWARD_SINCE)),
        "--until", until, "--checks", checks, "--interval", "900",
        "--state-file", str(d / "state.json"), "--report-file", str(d / "report.md"),
        "--next-file", str(d / "next.json"), "--log-file", str(d / "log.txt"),
        "--verdict-file", str(d / "verdict.json"), "--forward-out", str(d / "forward.json"),
        "--dry-run",
    ]
    with sandbox(STATE=d / "state.json", REPORT=d / "report.md", NEXT=d / "next.json",
                 LOG=d / "log.txt", VERDICT=d / "verdict.json",
                 FORWARD_PATH=d / "forward.json", CANARY_DRY=d / "canary-dry.json",
                 DRY_RUN=True, run=rec, now=fake_now,
                 venue_balance=lambda: Decimal("130.49"),
                 engine_state=lambda: {"mode": "paper", "live_armed": False,
                                       "pid": 775298},
                 reconcile=lambda since=None: None,
                 restart_flat=lambda new_epoch=False: (0, "SHOULD NOT RUN"),
                 epoch_start=lambda: float(D.FORWARD_SINCE),
                 venue_flat=lambda: (True, "flat")):
        with mock.patch.object(D.time, "sleep", fake_sleep):
            sys.argv = argv
            D.main()


def stage_six() -> None:
    """The 12:00/13:00/14:00 control flow, driven with no forward data at all."""
    past = ("2026-11-17T00:00:00-05:00,2026-11-17T01:00:00-05:00,"
            "2026-11-17T02:00:00-05:00")

    # 6.1 Quiet flow: each slot rebuilds and scores once, then waits.
    d = SANDBOX / "t6-quiet"
    rec = Recorder(forward_trades=5)
    drive(d, checks=past, rec=rec, seconds=3.0)
    state = read(d / "state.json")
    report = (d / "report.md").read_text() if (d / "report.md").exists() else ""
    nxt = read(d / "next.json")
    check("6.1 three quiet-flow slots -> three rebuilds and three scores",
          rec.rebuilds == 3 and rec.scores == 3,
          f"rebuilds={rec.rebuilds} scores={rec.scores}")
    log = (d / "log.txt").read_text() if (d / "log.txt").exists() else ""
    check("6.2 the 20-trade floor is never lowered",
          int(state.get("checks_done") or 0) == 3
          and state.get("decision") == "insufficient sample"
          and "floor was not lowered" in report
          and log.count("never lowering the 20-trade floor") == 2,
          f"decision={state.get('decision')} "
          f"reschedules={log.count('never lowering the 20-trade floor')}")
    check("6.3 the frozen thresholds were not touched",
          features.F2_Z_MIN == 0.30 and features.F2_FLOW_MIN == 0.10
          and features.F2_REV_MIN == 0.0 and features.F2_DELTA_BPS_MIN == 3.0
          and F.MIN_TRADES == 20 and F.REQUIRED_PF == 1.15
          and F.REQUIRED_ECE == 0.08 and F.CAP == 0.62,
          "a threshold changed")
    check("6.4 the 18:00 report and NEXT.json are written",
          "## Outcome" in report and "## Forward OOS" in report
          and "## 15-minute supervision" in report
          and nxt.get("state", {}).get("decision") == "insufficient sample"
          and int(nxt.get("supervision", {}).get("checks") or 0) >= 3,
          f"report={len(report)} next={bool(nxt)}")
    score_calls = [c["cmd"] for c in rec.calls
                   if "forward_oos.py" in " ".join(c["cmd"])]
    rebuild_calls = [c["cmd"] for c in rec.calls
                     if "rebuild_cache.sh" in " ".join(c["cmd"])]
    check("6.10 the supervisor's score command is the production one",
          bool(score_calls) and all("--since" in c
                                    and str(int(D.FORWARD_SINCE)) in c
                                    and "--until" not in c for c in score_calls),
          str(score_calls[:1]))
    check("6.11 the supervisor's rebuild command is the production one",
          bool(rebuild_calls) and all(c[0] == "bash"
                                      and c[1].endswith("scripts/rebuild_cache.sh")
                                      for c in rebuild_calls),
          str(rebuild_calls[:1]))

    # 6.5 A failed rebuild spends the slot, not the day; the report still lands.
    d = SANDBOX / "t6-rebuild-fail"
    rec = Recorder(rebuild_fail=True)
    drive(d, checks=past, rec=rec, seconds=3.0)
    state = read(d / "state.json")
    check("6.5 a failed rebuild is attempted once per slot and never scores",
          rec.rebuilds == 3 and rec.scores == 0
          and state.get("decision") == "blocked",
          f"rebuilds={rec.rebuilds} scores={rec.scores} "
          f"decision={state.get('decision')}")
    check("6.6 a failing child did not stop the 18:00 report",
          (d / "report.md").exists() and (d / "next.json").exists(),
          "report missing")

    # 6.7 A restart after the score wrote adopts the artifact: no re-score.
    d = SANDBOX / "t6-adopt"
    slot = datetime.now(timezone.utc) - timedelta(seconds=30)
    d.mkdir(exist_ok=True)
    D.atomic_write(d / "forward.json",
                   json.dumps(quiet_forward(5), indent=1, default=str))
    rec = Recorder(forward_trades=5)
    drive(d, checks=slot.isoformat(), rec=rec, seconds=2.0, clean_forward=False,
          state={"decision": "pending", "checks_done": 0, "checks": []})
    state = read(d / "state.json")
    check("6.7 an artifact already on disk for this slot is adopted, not re-scored",
          rec.rebuilds == 0 and rec.scores == 0
          and int(state.get("checks_done") or 0) == 1,
          f"rebuilds={rec.rebuilds} scores={rec.scores} "
          f"checks_done={state.get('checks_done')}")

    # 6.8 A slot attempted before a crash but left without an artifact is spent.
    d = SANDBOX / "t6-spent"
    slot = datetime.now(timezone.utc) - timedelta(seconds=30)
    rec = Recorder(forward_trades=5)
    drive(d, checks=slot.isoformat(), rec=rec, seconds=2.0,
          state={"decision": "pending", "checks_done": 0, "checks": [],
                 "slot_attempt": {"slot": slot.isoformat(),
                                  "started": time.time(),
                                  "since": int(D.FORWARD_SINCE)}})
    state = read(d / "state.json")
    check("6.8 a slot attempted before a crash is spent, never rebuilt twice",
          rec.rebuilds == 0 and rec.scores == 0
          and state.get("decision") == "blocked",
          f"rebuilds={rec.rebuilds} scores={rec.scores} "
          f"decision={state.get('decision')}")

    # 6.9 A failing promotion step blocks arming but not the 18:00 report.
    d = SANDBOX / "t6-promo-fail"
    rec = Recorder(forward_trades=72, fail_on="role-coverage")
    drive(d, checks=past.split(",")[0], rec=rec, seconds=3.0)
    state = read(d / "state.json")
    verdict = read(d / "verdict.json")
    report = (d / "report.md").read_text() if (d / "report.md").exists() else ""
    check("6.9 a failed promotion step demotes and still writes the 18:00 report",
          state.get("decision") == "blocked"
          and verdict.get("promoted") is False
          and "dashboard role-coverage" in report and "## Outcome" in report,
          f"decision={state.get('decision')} promoted={verdict.get('promoted')} "
          f"report={len(report)}")


def stage_seven() -> None:
    """Arming safety (a stale canary) and the second circuit-breaker trip."""
    # 7.1 A stale canary.json cannot satisfy the live-canary step.
    d = SANDBOX / "t7-stale-canary"
    d.mkdir(exist_ok=True)
    seed_verdict(False, d / "verdict.json")
    (d / "canary.json").write_text(json.dumps({
        "generated_ts": time.time() - 86400, "ok": True, "contracts": 1,
        "flattened": True, "reconcile_delta": "0.00"}, indent=1))
    rec = Recorder()
    restarts = {"n": 0}

    def no_restart(new_epoch: bool = False):
        restarts["n"] += 1
        return 0, "SHOULD NOT RUN"

    with sandbox(STATE=d / "state.json", REPORT=d / "report.md", NEXT=d / "next.json",
                 LOG=d / "log.txt", VERDICT=d / "verdict.json",
                 FORWARD_PATH=d / "forward.json", CANARY_LIVE=d / "canary.json",
                 DRY_RUN=False, run=rec,
                 venue_flat=lambda: (True, "flat"), restart_flat=no_restart):
        state = {"decision": "pending", "checks_done": 0, "checks": []}
        D.save_state(state)
        result = D.attempt_promotion(passing_forward(), state)
        verdict = read(D.VERDICT)
        canary_step = [s for s in result["steps"]
                       if s["step"] == "live canary (1 contract)"]
        check("7.1 a stale canary.json is refused even when it says ok",
              not result["ok"] and result["reason"] == "canary"
              and canary_step and not canary_step[0]["ok"]
              and "fresh=False" in canary_step[0]["detail"]
              and verdict.get("promoted") is False and restarts["n"] == 0,
              f"reason={result['reason']} restarts={restarts['n']} "
              f"detail={canary_step[0]['detail'] if canary_step else 'missing'}")

    # 7.2 A second trip after an observed disarm gets its own single restart.
    d = SANDBOX / "t7-second-trip"
    d.mkdir(exist_ok=True)
    seed_verdict(True, d / "verdict.json")
    (d / "state.json").write_text(json.dumps(
        {"decision": "pending", "checks_done": 0, "checks": [],
         "epoch_start": float(D.FORWARD_SINCE)}, indent=1))
    start = time.time()
    restarts = {"n": 0}

    def armed_now() -> bool:
        return not (2.5 <= time.time() - start < 3.5)

    def fake_restart(new_epoch: bool = False):
        restarts["n"] += 1
        return 0, "stub restart (engine untouched)"

    until = datetime.now(timezone.utc) + timedelta(seconds=6.5)
    argv = [
        "day_supervisor.py", "--since", str(int(D.FORWARD_SINCE)),
        "--until", until.isoformat(), "--checks", "2026-11-17T23:00:00-05:00",
        "--interval", "1",
        "--state-file", str(d / "state.json"), "--report-file", str(d / "report.md"),
        "--next-file", str(d / "next.json"), "--log-file", str(d / "log.txt"),
        "--verdict-file", str(d / "verdict.json"), "--forward-out", str(d / "forward.json"),
        "--dry-run",
    ]
    with sandbox(STATE=d / "state.json", REPORT=d / "report.md", NEXT=d / "next.json",
                 LOG=d / "log.txt", VERDICT=d / "verdict.json",
                 FORWARD_PATH=d / "forward.json", DRY_RUN=True,
                 venue_balance=lambda: Decimal("104.99"),
                 engine_state=lambda: {"mode": "live" if armed_now() else "paper",
                                       "live_armed": armed_now(), "pid": 775298},
                 reconcile=lambda since=None: {"delta": "0.00"},
                 restart_flat=fake_restart,
                 epoch_start=lambda: float(D.FORWARD_SINCE),
                 venue_flat=lambda: (True, "flat")):
        sys.argv = argv
        D.main()
    state = read(d / "state.json")
    checks = state.get("checks") or []
    tripped = [c for c in checks if c.get("breaker")]
    check("7.2 a second trip after an observed disarm gets its own single restart",
          restarts["n"] == 2 and int(state.get("breaker_restarts") or 0) == 2
          and len(tripped) >= 2,
          f"restarts={restarts['n']} "
          f"breaker_restarts={state.get('breaker_restarts')} trips={len(tripped)} "
          f"armed={[c.get('armed') for c in checks]}")


def stage_eight() -> None:
    """The watchdog restarts a dead supervisor and closes a missing 18:00 report."""
    root = Path(tempfile.mkdtemp(prefix="wd-drill-"))
    try:
        (root / "scripts").mkdir()
        (root / "data").mkdir()
        (root / ".venv" / "bin").mkdir(parents=True)
        shutil.copy(ROOT / "scripts" / "day_watchdog.sh",
                    root / "scripts" / "day_watchdog.sh")
        fake = root / ".venv" / "bin" / "python"
        fake.write_text(
            "#!/bin/bash\n"
            "root=\"$(cd \"$(dirname \"$0\")/../..\" && pwd)\"\n"
            "echo \"$(date +%s) $*\" >> \"$root/invocations.log\"\n"
            "until_iso=\"\"; prev=\"\"\n"
            "for a in \"$@\"; do [ \"$prev\" = \"--until\" ] && until_iso=\"$a\" && break; prev=\"$a\"; done\n"
            "until_ts=$(date -d \"$until_iso\" +%s)\n"
            "if [ \"$(date +%s)\" -ge \"$until_ts\" ]; then\n"
            "  mkdir -p \"$root/data\"; echo report > \"$root/data/evening-report.md\"\n"
            "fi\n"
            "exit 1\n")
        fake.chmod(0o755)
        env = dict(os.environ, OPENKAL_WATCHDOG_SLEEP="0.1",
                   OPENKAL_SUPERVISOR_CMD=f"{fake} {root}/scripts/day_supervisor.py",
                   OPENKAL_SUPERVISOR_PATTERN=f"^{root}/")

        def run_wd(until: str) -> None:
            subprocess.run(["bash", str(root / "scripts" / "day_watchdog.sh"),
                            "--until", until, "--since", "1"],
                           env=env, timeout=30, capture_output=True, text=True)

        report = root / "data" / "evening-report.md"
        run_wd((datetime.now() - timedelta(seconds=1)).isoformat())
        calls = (root / "invocations.log").read_text().splitlines()
        log = (root / "data" / "day-watchdog.log").read_text()
        check("8.1 a deadline crossed with no supervisor -> one closing run writes the report",
              report.exists() and len(calls) == 1 and "one closing run" in log,
              f"calls={len(calls)} report={report.exists()}")

        report.unlink()
        (root / "invocations.log").unlink()
        (root / "data" / "day-watchdog.log").unlink()
        run_wd((datetime.now() + timedelta(seconds=2)).isoformat())
        calls = (root / "invocations.log").read_text().splitlines()
        log = (root / "data" / "day-watchdog.log").read_text()
        check("8.2 a dead supervisor is restarted before the deadline and the report lands",
              report.exists() and len(calls) >= 2 and "restarting" in log,
              f"calls={len(calls)} report={report.exists()}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def stage_discovery_boundary() -> None:
    """The rebuild's discovery never reads a window at/after the forward start."""
    from scripts import subsec_discovery as SD

    class FakeLane:
        def __init__(self, closes: list[float]) -> None:
            self.meta = {f"T{i}": {"close": c, "strike": 1.0, "y": 0}
                         for i, c in enumerate(closes)}

    start = SD.FORWARD_OOS_START_TS
    lanes = {"X": FakeLane([start - 900.0, start - 1.0, start, start + 900.0])}
    split = SD.freeze_split(lanes)
    check("9.1 the discovery split is hard-capped before the forward start",
          split["windows"] == 2
          and split.get("forward_oos_start_ts") == start
          and str(split.get("oos_end", "")).startswith("2026-11-17T09:59:59")
          and start not in lanes["X"].is_closes
          and start not in lanes["X"].oos_closes,
          f"windows={split.get('windows')} oos_end={split.get('oos_end')}")
    art = ROOT / "data" / "subsec-discovery.json"
    if art.exists():
        data = json.loads(art.read_text())
        s = data.get("split", {})
        end = s.get("oos_end")
        end_ts = datetime.fromisoformat(end).timestamp() if end else 0
        check("9.2 the discovery artifact's own boundary is pre-forward",
              s.get("forward_oos_start_ts") == start
              and data.get("forward_oos_start_ts") == start
              and end_ts < start,
              f"oos_end={end}")


def stage_ten() -> None:
    """The short-sample retry path, the gate matrix and the 18:00 stop."""
    noon = "2026-11-17T12:00:00-05:00"
    one = "2026-11-17T13:00:00-05:00"
    two = "2026-11-17T14:00:00-05:00"
    checks = ",".join((noon, one, two))
    until = "2026-11-17T18:00:00-05:00"

    # 10.1 The noon short sample schedules exactly 13:00 and promotes nothing.
    d = SANDBOX / "t10-noon-short"
    rec = Recorder(forward_artifacts=[quiet_forward(5)])
    drive_clock(d, checks=checks, rec=rec,
                until="2026-11-17T12:30:00-05:00",
                start="2026-11-17T11:59:00-05:00")
    state = read(d / "state.json")
    verdict = read(d / "verdict.json")
    check("10.1 12:00 short sample -> waiting with exactly one 13:00 next check",
          state.get("decision") == "waiting"
          and state.get("next_check") == one
          and int(state.get("checks_done") or 0) == 1
          and rec.scores == 1 and rec.score_trades == [5]
          and verdict.get("promoted") is False,
          f"decision={state.get('decision')} next={state.get('next_check')} "
          f"scores={rec.scores} promoted={verdict.get('promoted')}")

    # 10.2 Fresh scorer per hourly slot, then promotion on the 22-trade pass.
    d = SANDBOX / "t10-retry-pass"
    rec = Recorder(forward_artifacts=[quiet_forward(5), quiet_forward(12),
                                      forward_artifact(22)])
    drive_clock(d, checks=checks, rec=rec, until=until)
    state = read(d / "state.json")
    verdict = read(d / "verdict.json")
    report = (d / "report.md").read_text() if (d / "report.md").exists() else ""
    log = (d / "log.txt").read_text() if (d / "log.txt").exists() else ""
    check("10.2 each later hour re-invokes the scorer fresh with the frozen argv",
          rec.scores == 3 and rec.score_trades == [5, 12, 22]
          and all("--since" in c and str(int(D.FORWARD_SINCE)) in c
                  and "--until" not in c for c in rec.score_argv)
          and "adopting the result already on disk" not in log
          and log.count("never lowering the 20-trade floor") == 2,
          f"scores={rec.scores} trades={rec.score_trades}")
    check("10.3 promotion only after the 14:00 >=20-trade passing artifact",
          state.get("decision") == "verified (dry run)"
          and verdict.get("promoted") is True
          and "Trades after the desk hurdle: 22" in report
          and "Gate: PASS" in report
          and all(s.get("ok") for s in (state.get("steps") or [])),
          f"decision={state.get('decision')} promoted={verdict.get('promoted')} "
          f"steps={len(state.get('steps') or [])}")
    check("10.4 the loop stops at 18:00 and never scores after 14:00",
          rec.scores == 3 and state.get("next_check") is None
          and bool(state.get("checks"))
          and str(state["checks"][-1].get("ts", "")).startswith("2026-11-17T18:00")
          and "window closed" in log,
          f"scores={rec.scores} "
          f"last={state['checks'][-1].get('ts') if state.get('checks') else None}")

    # 10.5 All three slots short: insufficient sample, no 15:00 retry, stop at 18:00.
    d = SANDBOX / "t10-retry-short"
    rec = Recorder(forward_artifacts=[quiet_forward(5), quiet_forward(12),
                                      quiet_forward(5)])
    drive_clock(d, checks=checks, rec=rec, until=until)
    state = read(d / "state.json")
    verdict = read(d / "verdict.json")
    report = (d / "report.md").read_text() if (d / "report.md").exists() else ""
    check("10.5 the last short check ends insufficient sample, not a 15:00 retry",
          state.get("decision") == "insufficient sample"
          and state.get("next_check") is None
          and int(state.get("checks_done") or 0) == 3
          and rec.scores == 3
          and "only 5 forward trades by the last check" in report
          and verdict.get("promoted") is False,
          f"decision={state.get('decision')} next={state.get('next_check')} "
          f"scores={rec.scores}")

    # 10.6 A >=20-trade artifact with exactly one failing gate never promotes.
    cases = {
        "net": {"net": -0.01},
        "PF": {"pf": 1.10},
        "payoff-ECE": {"ece": 0.09},
        "crossed fills": {"crossed": 1},
        "max cost": {"max_cost": 0.63},
        "two-lane": {"lanes": {"KXBTC15M": {"trades": 22, "net": 1.0}}},
    }
    for label, kw in cases.items():
        d = SANDBOX / ("t10-gate-" + label.replace(" ", "-").replace("/", "-"))
        artifact = forward_artifact(22, **kw)
        rec = Recorder(forward_artifacts=[quiet_forward(5), quiet_forward(12),
                                          artifact])
        drive_clock(d, checks=checks, rec=rec, until=until)
        state = read(d / "state.json")
        verdict = read(d / "verdict.json")
        report = (d / "report.md").read_text() if (d / "report.md").exists() else ""
        reason = (artifact.get("gate") or {}).get("reasons") or []
        check(f"10.6 {label}: one failing gate at 22 trades -> rejected, no promotion",
              state.get("decision") == "rejected"
              and verdict.get("promoted") is False
              and rec.scores == 3
              and "Gate: FAIL" in report
              and all(r in report for r in reason),
              f"decision={state.get('decision')} promoted={verdict.get('promoted')} "
              f"reasons={reason}")

    # 10.7 A nonzero scorer exit is a failed slot, not an artifact to adopt.
    d = SANDBOX / "t10-score-fail"
    rec = Recorder(forward_artifacts=[forward_artifact(22), quiet_forward(5),
                                      quiet_forward(5)], score_fail_at=0)
    drive_clock(d, checks=checks, rec=rec, until=until)
    state = read(d / "state.json")
    verdict = read(d / "verdict.json")
    log = (d / "log.txt").read_text() if (d / "log.txt").exists() else ""
    check("10.7 a nonzero scorer exit spends the slot even if its artifact would pass",
          rec.score_rcs == [1, 0, 0] and rec.scores == 3
          and state.get("decision") == "insufficient sample"
          and state.get("next_check") is None
          and verdict.get("promoted") is False
          and "forward score failed" in log,
          f"rcs={rec.score_rcs} decision={state.get('decision')} "
          f"promoted={verdict.get('promoted')}")


def main() -> int:
    stage_two()
    stage_three()
    stage_three_marker()
    stage_three_resume()
    stage_four()
    stage_five()
    stage_six()
    stage_seven()
    stage_eight()
    stage_discovery_boundary()
    stage_ten()
    failed = [c for c in CHECKS if not c[1]]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    for name, _, detail in failed:
        print(f"  FAILED {name}  {detail}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
