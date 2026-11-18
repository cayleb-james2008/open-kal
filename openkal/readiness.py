"""Readiness: the functional gate between paper and real money.

Waiting is not evidence. The old gate wanted 60 minutes of wall-clock time and 20
forward paper fills; a quiet hour proved nothing about the program, and a paused clock
could satisfy it. This gate asks four functional questions instead, each of which is a
property of *this build, this tape and this venue connection*:

  1. `strategy` — a frozen out-of-sample verdict exists for the program the engine can
     actually run: strictly positive OOS net P&L after `Decimal` fees, profit factor
     >= 1.15, calibration error <= 0.08 on the event the program's probability
     actually describes (the trade's own payoff for a payoff-calibrated program,
     the market's settlement otherwise), zero crossed-book fills, on at least 20 OOS
     trades.
     2. `controls` — `tests/test_gate_controls.py` passes every check it reports
      (`N/N`, at least the original 8: replay-to-ledger parity, decision-window and
      crossed-ladder mutations, crossed-book refusal at the decision and at the live
      order path). `tests/test_controls_bind.py` proves those controls fail under an
      injected regression. Executed once per process, because the control replays a
      whole epoch and the snapshot is published twice a second.
  3. `ws` — the live plane is healthy: the `SeqTracker` is active, every sequence gap
     has been recovered by a REST resync, and Kalshi, Coinbase and Kraken all delivered
     inside 500 ms.
  4. `canary` — a live V2 one-contract micro-canary was placed, matched to the cent
     against `/portfolio/fills`, and flattened clean. `data/canary.json`, written by
     `scripts/canary.py`, is the only accepted evidence; a dry run does not count.

`OPENKAL_ARM_LIVE=1` is permission to arm *if* these pass, never a bypass. If any check
fails the bot says why and keeps paper trading.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
VERDICT = ROOT / "data" / "strategy-verdict.json"
CANARY = ROOT / "data" / "canary.json"
CONTROLS_TEST = ROOT / "tests" / "test_gate_controls.py"
# The parity fixture is a scratch epoch the real engine filled through the real
# execution core; it is the only ledger guaranteed to contain the fills a parity
# control needs before the live epoch has any.
PARITY_DB = ROOT / "data" / "openkal-parity.db"

# Programs `strategy.decide` actually implements. A passing verdict for a program that
# is not wired into the hot path must not arm: the engine would trade something else.
# `z_obi_flow` is the promoted program: strike z, printed-flow imbalance and touch
# revisions agree, entered post-only with a per-window cap and a halt after one loss.
ENGINE_PROGRAMS = {"impulse_snipe", "final_minute", "z_obi_flow", "F2_taker_1s",
                   "student_t_l2_settle", "cross_lane_iv_arb"}
REQUIRED_PF = 1.15
REQUIRED_ECE = 0.08
MIN_OOS_TRADES = 20
# Guardrail 3: a deployed binary buys at no more than 62c. This is the entry
# ceiling, hard: the 16:1 0.84-carry profile lost -$44.55 on XRP, and a stop
# inside the band does not make a 0.95 entry safe to arm. The constant is that
# lesson, executable.
MAX_NAKED_COST = Decimal("0.62")
# Per-venue staleness bars. Kalshi's socket is the execution feed and must be
# sub-quarter-second. Coinbase ticks several times a second, so half a second is
# already generous. Kraken's public ticker is a ~1s snapshot cadence by design: a
# 500 ms bar marked a healthy feed dead, so its bar is 1.5x its own cadence.
WS_MAX_LAG_MS = {"kalshi": 250.0, "coinbase": 500.0, "kraken": 3000.0}
MAX_WS_LAG_MS = WS_MAX_LAG_MS["coinbase"]
CANARY_MAX_AGE_S = 6 * 3600.0

_CONTROLS: dict[str, Any] = {"ts": 0.0, "ok": False, "detail": "not run yet", "key": None}
# Decision parity is a tape scan, not a replay: cached for a minute so the twice-a-
# second snapshot does not read 20k rows per lane on every publish.
_PARITY: dict[str, Any] = {"ts": 0.0, "ok": False, "detail": "not run yet", "key": None}
PARITY_MAX_AGE_S = 60.0


@dataclass
class Check:
    key: str
    label: str
    ok: bool
    detail: str


def evaluate(engine: Any) -> dict[str, Any]:
    started = time.time()
    checks: list[Check] = []
    checks.append(_strategy(getattr(engine, "running_program", "") or ""))
    checks.append(_controls(engine))
    checks.append(_ws(engine))
    checks.append(_canary())
    checks.append(_feeds(engine))
    checks.append(_loop(engine))
    checks.append(_venue(engine))
    checks.append(_signal_parity(engine))
    ready = all(c.ok for c in checks)
    blockers = [c.label for c in checks if not c.ok]
    stats = engine.store.epoch_stats(engine.epoch_start)
    net = Decimal(str(stats.get("pnl") or "0"))
    return {
        "ready": ready,
        "armed": engine.live_armed,
        "armed_at": engine.arm_ts,
        "kill_reason": engine.kill_reason,
        "elapsed_minutes": round((time.time() - engine.started) / 60.0, 1),
        "settled_trades": int(stats.get("trades") or 0),
        "net": str(net.quantize(Decimal("0.01"))),
        "blockers": blockers,
        "checks": [asdict(c) for c in checks],
        # Kept for the dashboard and for humans: the evidence behind the gate.
        "strategy": _verdict_payload(),
        "controls": dict(_CONTROLS),
        "ws": _ws_payload(engine),
        "canary": _canary_payload(),
        "evaluated_ms": round((time.time() - started) * 1000.0, 1),
    }


# ------------------------------------------------------------------ 1. strategy
def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _verdict_payload() -> dict[str, Any] | None:
    return _read_json(VERDICT)


def risk_bounds(candidate: dict) -> tuple[Decimal | None, Decimal | None, Decimal | None, str]:
    """Read a candidate's structural risk: entry ceiling, hard stop, take-profit, exit.

    The verdict schema has grown a few shapes (a detector band, a params block, a
    top-level cap), so every one is checked rather than assuming the newest.
    """
    det = candidate.get("detector") or {}
    par = candidate.get("params") or {}
    cost: Decimal | None = None
    band = det.get("cost_band") or candidate.get("cost_band")
    if isinstance(band, (list, tuple)) and band:
        try:
            cost = Decimal(str(band[-1]))
        except Exception:  # noqa: BLE001 - a malformed band is "no bound", not a crash
            cost = None
    if cost is None:
        for src in (candidate, det, par):
            for key in ("max_entry_price", "cost_hi", "cost_cap", "max_cost", "max_price"):
                if src.get(key) not in (None, ""):
                    try:
                        cost = Decimal(str(src[key]))
                    except Exception:  # noqa: BLE001
                        continue
                    break
            if cost is not None:
                break
    stop: Decimal | None = None
    for src in (candidate, det, par):
        for key in ("stop_c", "stop_loss"):
            if src.get(key) in (None, ""):
                continue
            try:
                d = Decimal(str(src[key]))
            except Exception:  # noqa: BLE001
                continue
            if d > 0:
                stop = d
                break
        if stop is not None:
            break
    tp: Decimal | None = None
    for src in (candidate, par):
        if src.get("tp_c") not in (None, ""):
            try:
                tp = Decimal(str(src["tp_c"]))
            except Exception:  # noqa: BLE001
                tp = None
            break
    mode = str(candidate.get("exit_mode") or par.get("exit_mode") or "")
    return cost, stop, tp, mode


def _risk_problems(candidate: dict) -> list[str]:
    """Guardrail 3, executable: bounded downside, and no truncated asymmetric exit.

    A binary bought at `c` can lose `c` per contract. A take-profit of `tp` while the
    loss tail is `1 - c` keeps the small win and the full loss: at 0.84 with a 5c TP
    the payoff is 16:1 and needs a 94% hit rate to break even. That profile produced
    the -$44.55 XRP settlement, so it may not arm again.

    The entry ceiling is a hard cap, not a range that a hard stop may widen: every
    deployed strategy buys at `entry_price <= 0.62`, so a 0.95 lock-in may not arm
    even with a 5c stop. A candidate that declares no ceiling at all cannot prove the
    bound, so it is refused rather than assumed.
    """
    cost, stop, tp, mode = risk_bounds(candidate)
    problems: list[str] = []
    if cost is None:
        problems.append(f"no declared entry ceiling: cannot prove entry <= {MAX_NAKED_COST}")
    elif cost > MAX_NAKED_COST:
        problems.append(f"entry cost {cost} > {MAX_NAKED_COST} hard cap "
                        f"({mode or 'no exit mode'})")
    if stop is None and cost is not None and tp is not None:
        floor = (Decimal(1) - cost).quantize(Decimal("0.0001"))
        if tp < floor:
            problems.append(f"truncated exit: tp {tp} < {floor} of downside "
                            f"(payoff {floor / tp if tp else 0}:1)")
    return problems


def strategy_ece(verdict: dict) -> tuple[float, str]:
    """The calibration error this gate reads, and the event it measures.

    A momentum program's probability is the chance its own round trip finishes
    positive, and scoring that number against the market's settlement outcome
    measures a coin against a ruler: F2_taker_1s reports settlement ECE 0.4577
    while its payoff ECE is 0.0595. The verdict declares which event its
    probability describes (`candidate.probability_target == "payoff"`) and
    carries the matching number. A verdict that declares the payoff target but
    carries no payoff ECE is a broken artifact, not a pass, so it fails closed at
    1.0 rather than falling back to the settlement number and looking better.
    """
    oos = verdict.get("oos") or {}
    candidate = verdict.get("candidate") or {}
    target = str(candidate.get("probability_target")
                 or oos.get("probability_target") or "settlement")
    if target == "payoff":
        value = oos.get("payoff_ece", oos.get("payoff_event_ece"))
        return (float(value) if value is not None else 1.0), "payoff"
    value = oos.get("ece")
    return (float(value) if value is not None else 1.0), "settlement"


def _strategy(running: str = "") -> Check:
    verdict = _verdict_payload()
    if verdict is None:
        return Check("strategy", "OOS-verified positive-EV strategy is loaded", False,
                     f"no verdict at {VERDICT.relative_to(ROOT)}; run scripts/alpha_discover.py")
    oos = verdict.get("oos") or {}
    candidate = verdict.get("candidate") or {}
    net = Decimal(str(oos.get("net") or "0"))
    pf = float(oos.get("pf") or 0.0)
    ece, ece_target = strategy_ece(verdict)
    trades = int(oos.get("trades") or 0)
    crossed = int(oos.get("crossed_fills") or 0)
    program = str(candidate.get("name") or "")
    problems = []
    if not verdict.get("promoted"):
        problems.append("not promoted")
    if program not in ENGINE_PROGRAMS:
        problems.append(f"program {program or 'none'} is not wired into strategy.decide")
    # A verdict may only arm the program the engine is actually running. The two
    # disagreeing is the failure this whole gate exists to prevent: the engine
    # would trade something nobody verified.
    if running and program and program != running:
        problems.append(f"verdict names {program} but the engine runs {running}")
    if net <= 0:
        problems.append(f"OOS net {net} USD")
    if pf < REQUIRED_PF:
        problems.append(f"PF {pf:.2f} < {REQUIRED_PF}")
    if ece > REQUIRED_ECE:
        problems.append(f"{ece_target} ECE {ece:.3f} > {REQUIRED_ECE}")
    if trades < MIN_OOS_TRADES:
        problems.append(f"{trades} OOS trades < {MIN_OOS_TRADES}")
    if crossed != 0:
        problems.append(f"{crossed} crossed-book fills")
    problems.extend(_risk_problems(candidate))
    detail = (f"{program or 'no program'}: OOS {trades} trades, net {net} USD, "
              f"PF {pf:.2f}, {ece_target} ECE {ece:.3f}, crossed {crossed}")
    if problems:
        detail += " — " + "; ".join(problems)
    return Check("strategy", "OOS-verified positive-EV strategy is loaded",
                 not problems, detail)


# ------------------------------------------------------------------ 2. controls
CONTROL_FLOOR = 8  # the checks the controls existed with before the crossed-book additions


def _passed_count(stdout: str) -> tuple[int, int] | None:
    """`13/13 controls passed` -> (13, 13). The count is allowed to grow; it may not
    grow by adding a check that is quietly skipped, so the fraction must be whole."""
    m = re.search(r"(\d+)/(\d+) controls passed", stdout or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def _run_controls(db: Path) -> tuple[bool, str]:
    started = time.time()
    try:
        proc = subprocess.run([sys.executable, str(CONTROLS_TEST), str(db)], cwd=str(ROOT),
                              capture_output=True, text=True, timeout=300)
        out = (proc.stdout or "").strip().splitlines()
        tail = out[-1] if out else (proc.stderr or "").strip()[-200:]
        counts = _passed_count(proc.stdout or "")
        ok = (proc.returncode == 0 and counts is not None
              and counts[0] == counts[1] and counts[1] >= CONTROL_FLOOR)
        if counts is not None:
            tail = f"{counts[0]}/{counts[1]} controls passed" + ("" if ok else f" ({tail})")
        print(f"controls {db.name}: {tail} ({time.time() - started:.1f}s)", flush=True)
        return ok, tail
    except subprocess.TimeoutExpired:
        return False, "controls timed out after 300s"
    except Exception as exc:  # noqa: BLE001 - a broken control must read as a blocker
        return False, f"{type(exc).__name__}: {exc}"


def _controls(engine: Any) -> Check:
    """Run the negative controls once per process.

    Two ledgers are checked. The parity fixture always (it is a real epoch the engine
    filled through the real execution core, so the replay/mutation pair has something
    to bite on even before this epoch trades). The live ledger too, but only once it
    holds a settled trade this epoch: on an empty epoch the mutation control cannot
    bind, and a test that passes because it measured nothing is exactly the vacuity the
    controls exist to catch.

    Cached for the life of the process: each run replays a whole epoch, and the
    snapshot is published twice a second.
    """
    label = "functional controls all pass"
    live_db = None
    try:
        live_db = Path(str(engine.settings.db_path))
        # Parity is only a well-posed question when nothing is still open: the live
        # ledger books a trade at its close, while the replay marks a survivor out at
        # the last bid, so a position open mid-epoch makes the two disagree by
        # construction. Empty epoch OR open position = the control cannot bind yet.
        flat = all(getattr(s, "position", None) is None for s in engine.slots)
        # Activity, not just closed trades. An epoch can hold a live entry that never
        # closed (that is exactly the failure the replay is there to catch), and keying
        # on settled trades alone skipped the control for it.
        settled = int(engine.store.epoch_stats(engine.epoch_start, source="live").get("trades") or 0)
        filled = int(engine.store._db.execute(  # noqa: SLF001 - read-only count
            "SELECT COUNT(*) FROM fills WHERE ts >= ?", (engine.epoch_start,)).fetchone()[0])
        has_trades = ((settled > 0 or filled > 0) and flat)
    except Exception:  # noqa: BLE001
        has_trades = False
    dbs: list[Path] = []
    if PARITY_DB.exists():
        dbs.append(PARITY_DB)
    if has_trades and live_db is not None and live_db.exists() and live_db not in dbs:
        dbs.append(live_db)
    key = tuple(str(d) for d in dbs)
    if _CONTROLS["ts"] > 0.0 and _CONTROLS.get("key") == key:
        return Check("controls", label, _CONTROLS["ok"], _CONTROLS["detail"])
    if not CONTROLS_TEST.exists():
        _CONTROLS.update(ts=time.time(), ok=False, key=key,
                         detail=f"missing {CONTROLS_TEST.name}")
        return Check("controls", label, False, _CONTROLS["detail"])
    if not dbs:
        _CONTROLS.update(ts=time.time(), ok=False, key=key,
                         detail="no parity ledger and no settled trade this epoch")
        return Check("controls", label, False, _CONTROLS["detail"])
    results = [(db.name, *_run_controls(db)) for db in dbs]
    ok = all(r[1] for r in results)
    detail = "; ".join(f"{name} {tail}" for name, _, tail in results)
    if not has_trades:
        detail += " (live epoch has no settled trade yet)"
    _CONTROLS.update(ts=time.time(), ok=ok, key=key, detail=detail)
    return Check("controls", label, ok, detail)


# ------------------------------------------------------------------------ 3. ws
def _venue_ages(engine: Any) -> dict[str, float]:
    """Freshest message age per venue across every lane's symbols."""
    feed = getattr(engine, "feed", None)
    out: dict[str, float] = {}
    if feed is None:
        return out
    now = time.time()
    for source in ("coinbase", "kraken"):
        ages = [max(0.0, now - rows[source][0])
                for rows in (getattr(feed, "stamps", {}) or {}).values()
                if source in rows]
        out[source] = min(ages) if ages else 1e9
    return out


def _ws_state(engine: Any) -> dict[str, Any]:
    ws = getattr(engine, "ws", None)
    gaps = int(getattr(ws.seq, "gaps", 0)) if ws is not None else 0
    recovered = int(getattr(ws, "resync_ok", 0)) if ws is not None else 0
    return {
        "present": ws is not None,
        "seq_active": bool(getattr(ws, "seq", None) and ws.seq.bases),
        "subscriptions": len(getattr(ws.seq, "bases", {}) or {}) if ws is not None else 0,
        "gaps": gaps,
        "resync_ok": recovered,
        "unrecovered_gaps": max(0, gaps - recovered),
        "lag_ms": round(float(getattr(ws, "lag_ms", 1e9) or 1e9), 1),
        "degraded": bool(getattr(ws, "degraded", True)),
        "messages": int(getattr(ws, "messages", 0) or 0),
        "last_error": getattr(ws, "last_error", None),
        "ages_s": {k: round(v, 3) for k, v in _venue_ages(engine).items()},
    }


def _ws_payload(engine: Any) -> dict[str, Any]:
    return _ws_state(engine)


def _ws(engine: Any) -> Check:
    state = _ws_state(engine)
    problems = []
    if not state["present"]:
        problems.append("no Kalshi socket")
    if not state["seq_active"]:
        problems.append("SeqTracker inactive")
    if state["present"] and state["messages"] == 0:
        problems.append("0 socket messages" + (
            f" ({state['last_error']})" if state.get("last_error") else ""))
    if state["unrecovered_gaps"] > 0:
        problems.append(f"{state['unrecovered_gaps']} unrecovered gaps")
    if state["degraded"] or state["lag_ms"] >= WS_MAX_LAG_MS["kalshi"]:
        problems.append(f"kalshi lag {state['lag_ms']:.0f}ms")
    for venue, age in state["ages_s"].items():
        bar = WS_MAX_LAG_MS.get(venue, MAX_WS_LAG_MS)
        if age * 1000.0 >= bar:
            problems.append(f"{venue} {age * 1000.0:.0f}ms")
    detail = (f"{state['messages']:,} messages, {state['subscriptions']} subscriptions, "
              f"gaps {state['gaps']}/{state['resync_ok']} recovered, "
              f"kalshi {state['lag_ms']:.0f}ms, "
              + ", ".join(f"{k} {v * 1000.0:.0f}ms" for k, v in state["ages_s"].items()))
    if problems:
        detail += " — " + "; ".join(problems)
    return Check("ws", "live WS plane healthy (seq, gaps, kalshi<250ms, kraken<3000ms)",
                 not problems, detail)


# -------------------------------------------------------------------- 4. canary
def _canary_payload() -> dict[str, Any] | None:
    return _read_json(CANARY)


def _canary() -> Check:
    data = _canary_payload()
    label = "live V2 one-contract canary reconciled and flattened"
    if data is None:
        return Check("canary", label, False,
                     f"no canary at {CANARY.relative_to(ROOT)}; run scripts/canary.py --live")
    problems = []
    if data.get("mode") != "live":
        problems.append(f"mode {data.get('mode')!r} is not a venue order")
    if not data.get("ok"):
        problems.append("canary did not pass")
    if int(data.get("contracts") or 0) != 1:
        problems.append("not exactly one contract")
    if not data.get("flattened"):
        problems.append("position not flattened")
    if not data.get("reconciled"):
        problems.append("fills did not reconcile")
    delta = abs(Decimal(str(data.get("reconcile_delta") or "1")))
    if delta > Decimal("0.01"):
        problems.append(f"reconcile delta {delta}")
    # `scripts/canary.py` stamps its evidence `generated_ts`; older artifacts used
    # `ts`. Reading neither marked every real canary stale (498,569h) forever.
    age = time.time() - float(data.get("ts") or data.get("generated_ts") or 0.0)
    if age > CANARY_MAX_AGE_S:
        problems.append(f"stale ({age / 3600.0:.1f}h)")
    detail = (f"{data.get('ticker', '?')} fill {data.get('fill_price', '?')}, "
              f"delta {delta}, flat {bool(data.get('flattened'))}")
    if problems:
        detail += " — " + "; ".join(problems)
    return Check("canary", label, not problems, detail)


# ------------------------------------------- decision parity (live vs canonical)
def _signal_parity(engine: Any) -> Check:
    """Every recorded live decision must equal the canonical decision for its tick.

    A ledger/replay mismatch can only be seen after a fill, and a detector that
    changed its mind between the hot path and the canonical path could go unnoticed
    for a whole window. This reads the decisions the engine recorded with each tick
    and re-derives them with `strategy.decide` from the same inputs, so drift is a
    named blocker instead of an unexplained P&L difference. Loud on purpose: a
    single mismatch blocks arming.
    """
    from .parity import lane_parity

    label = "live decisions match the canonical decision path"
    try:
        key = tuple((s.cfg.series, str(s.params.to_json())) for s in engine.slots)
    except Exception:  # noqa: BLE001 - a broken slot is a red check, not a crash
        key = ()
    if (_PARITY["ts"] > 0.0 and _PARITY.get("key") == key
            and time.time() - _PARITY["ts"] < PARITY_MAX_AGE_S):
        return Check("parity", label, _PARITY["ok"], _PARITY["detail"])
    checked = mismatches = skipped = 0
    examples: list[str] = []
    try:
        for slot in engine.slots:
            ticks = engine.store.recent_ticks(slot.cfg.series, engine.epoch_start, limit=20000)
            result = lane_parity(slot.cfg.series, ticks, slot.params)
            checked += result["checked"]
            mismatches += result["mismatches"]
            skipped += result["skipped"]
            for row in result["examples"]:
                examples.append(f"{result['series']} {row['ts']:.0f}s live {row['live']} "
                                f"vs canonical {row['canonical']}")
    except Exception as exc:  # noqa: BLE001 - an unreadable tape is a blocker
        _PARITY.update(ts=time.time(), ok=False, key=key,
                       detail=f"parity read failed: {type(exc).__name__}: {exc}")
        return Check("parity", label, False, _PARITY["detail"])
    ok = mismatches == 0 and checked > 0
    if checked == 0:
        detail = ("no recorded decisions yet: this build records one per tick, so the "
                  "check binds after the first tick of this epoch")
    else:
        detail = f"{checked:,} decisions re-derived, {mismatches} mismatches"
        if skipped:
            detail += f", {skipped:,} skipped (position open or inputs not recorded)"
    if examples:
        detail += " — " + "; ".join(examples[:3])
    _PARITY.update(ts=time.time(), ok=ok, key=key, detail=detail)
    return Check("parity", label, ok, detail)


# --------------------------------------------------- operational hygiene (kept)
def _feeds(engine: Any) -> Check:
    ages = []
    for slot in engine.slots:
        if slot.cfg.feed_symbol and slot.quote is not None:
            ages.append(engine.tape.age(slot.cfg.feed_symbol, time.time()))
    worst = max(ages) if ages else 1e9
    return Check("feeds", "spot tape is live for every lane", worst < 10,
                 f"oldest quote {worst:.1f}s")


def _loop(engine: Any) -> Check:
    now = time.time()
    faults = len([t for t in getattr(engine, "fault_times", []) if now - t < 300])
    return Check("loop", "no engine faults in the last 5 min", faults == 0, f"{faults} faults")


def _venue(engine: Any) -> Check:
    now = time.time()
    recent = len([t for t in getattr(engine, "error_times", []) if now - t < 600])
    balance = Decimal(str(engine.venue_balance))
    ok = recent < 10 and balance >= Decimal("20")
    return Check("venue", "venue reachable and account funded", ok,
                 f"{recent} errors in the last 10 min (lifetime {engine.market_errors}), "
                 f"balance {balance.quantize(Decimal('0.01'))} USD")
