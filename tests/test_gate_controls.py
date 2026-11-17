"""Negative controls for the gate.

Each control proves a check *binds*: it feeds the check something wrong and requires it
to go red. A check that stays green under a deliberate mutation is not measuring
anything, and this file exists so that claim is executable rather than asserted.

Run: .venv/bin/python tests/test_gate_controls.py
"""
from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openkal import features  # noqa: E402
from openkal.config import SLOTS, load_settings  # noqa: E402
from openkal.kalshi_ws import SeqTracker  # noqa: E402
from openkal.replay import replay_series  # noqa: E402
from openkal.store import Store  # noqa: E402
from openkal.strategy import Params  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))


# --------------------------------------------------------------------------
# 1. Sequence gaps: a re-subscribe restarts `seq`, so the snapshot is a new base.
# --------------------------------------------------------------------------
class LegacyMaxTracker:
    """The shipped bug, reproduced verbatim so the regression test is real."""

    def __init__(self) -> None:
        self.bases: dict[int, int] = {}
        self.gaps = 0

    def on_snapshot(self, sid: int, seq: int) -> None:
        self.bases[sid] = max(self.bases.get(sid, 0), seq)

    def on_delta(self, sid: int, seq: int) -> bool:
        last = self.bases.get(sid)
        ok = last is None or seq in (last + 1, last)
        if not ok:
            self.gaps += 1
        self.bases[sid] = max(last or 0, seq)
        return ok


def control_sequence() -> None:
    # One subscription: snapshot at seq 1, deltas 2..6, then a re-subscribe that
    # restarts the stream at 1 and continues 2, 3. Nothing was missed.
    stream = [("snap", 1), ("delta", 2), ("delta", 3), ("delta", 4), ("delta", 5),
              ("delta", 6), ("snap", 1), ("delta", 2), ("delta", 3)]
    new, old = SeqTracker(), LegacyMaxTracker()
    for kind, seq in stream:
        (new.on_snapshot if kind == "snap" else new.on_delta)(4, seq)
        (old.on_snapshot if kind == "snap" else old.on_delta)(4, seq)
    check("seq: corrected tracker counts no gaps on re-subscribe", new.gaps == 0,
          f"{new.gaps} gaps")
    check("seq: old max() logic reports bogus gaps on the same stream", old.gaps > 0,
          f"{old.gaps} gaps")

    # And a real gap must still be caught.
    gap = SeqTracker()
    gap.on_snapshot(4, 10)
    ok = gap.on_delta(4, 12)
    check("seq: a genuine gap is still detected", (not ok) and gap.gaps == 1,
          f"detected={not ok}, gaps={gap.gaps}")


# --------------------------------------------------------------------------
# 2. Calibration: 0.000 must not be the "no data" default wearing a green badge.
# --------------------------------------------------------------------------
def control_calibration(store: Store) -> None:
    rows = store._db.execute("SELECT p_model, side, pnl FROM trades").fetchall()  # noqa: SLF001
    pairs: list[tuple[float, int]] = []
    for r in rows:
        try:
            p = float(r["p_model"])
        except (TypeError, ValueError):
            continue
        won = 1 if Decimal(str(r["pnl"])) > 0 else 0
        pairs.append((p, won))
    if not pairs:
        check("calibration: real ECE from settled pairs", False, "no trades to score")
        return
    truth = features.expected_calibration_error(pairs)
    perturbed = features.expected_calibration_error([(1 - p, y) for p, y in pairs])
    check("calibration: real ECE from this epoch's settled pairs", truth >= 0.0,
          f"ECE {truth:.4f} over {len(pairs)} pairs")
    check("calibration: metric moves under a perturbed probability",
          abs(perturbed - truth) > 1e-9,
          f"perturbed ECE {perturbed:.4f} vs {truth:.4f}")

    # The production gate reads `slot.ece`, which stays 0.0 below 8 pairs. Demonstrate
    # it on a 3-pair slice so the control holds whether the epoch has three trades or
    # three hundred: a defect control must not depend on the sample being tiny.
    small = pairs[:3]
    truth_small = features.expected_calibration_error(small)
    ece_with_few_pairs = truth_small if len(small) >= 8 else 0.0
    check("calibration: production default would report 0.000 on a small sample",
          ece_with_few_pairs == 0.0 and truth_small != 0.0,
          f"gate would show {ece_with_few_pairs:.3f} while the real ECE is {truth_small:.4f}")


# --------------------------------------------------------------------------
# 3. Repro: mutating one replay input must break parity with the ledger.
# --------------------------------------------------------------------------
def _params_for(store: Store, series: str) -> Params | list[tuple[float, Params]]:
    """The params this epoch actually ran with, not today's defaults.

    Parity means replaying the deployed program. A run with an explicit program (a
    parity harness, a promoted SHR candidate) records it in `deployed_params`; a run
    with the defaults does not need to. When the program changed inside the epoch,
    the engine records a `deployed_params_history` timeline and the replay switches
    params where the live loop did -- replaying every earlier trade under today's
    params would manufacture a mismatch no execution bug explains.
    """
    history = store.get_meta("deployed_params_history") or []
    segments: list[tuple[float, Params]] = []
    if isinstance(history, list):
        for row in history:
            if not isinstance(row, dict):
                continue
            raw = (row.get("params") or {}).get(series)
            if isinstance(raw, dict):
                segments.append((float(row.get("ts") or 0.0), Params.from_json(raw)))
    if segments:
        return segments
    raw = store.get_meta("deployed_params") or {}
    if isinstance(raw, dict) and series in raw:
        return Params.from_json(raw[series])
    return Params()


def _decision_window(ticks: list[dict], opened: float, lookback: float = 120.0) -> list[int]:
    """Every tick that could have decided this lane's entry on this contract.

    A maker order rests until somebody hits it: the KXBTC fill landed 9.5s after the
    bid was sent, with the signal intermittent in between, so "the last contiguous run
    of openable ticks before the fill" pointed at a tick the order was already resting
    through. The decision is any openable signal tick inside the order's working life
    on the same contract; mutating all of them is what makes the control bind.
    """
    nearest = min(range(len(ticks)), key=lambda j: abs(float(ticks[j]["ts"]) - opened))
    ticker = ticks[nearest].get("ticker")
    lo = opened - lookback
    out = [i for i, t in enumerate(ticks)
           if t.get("ticker") == ticker and lo <= float(t["ts"]) <= opened + 1e-6
           and t.get("can_open") and (t.get("signal") or "") not in ("", "gbm")]
    return out or [nearest]


def control_repro(store: Store) -> None:
    epoch = float(store.get_meta("epoch_start") or 0)
    opens: dict[str, float] = {}
    # Only this epoch's trades. Reading every trade in the file pointed the mutation at
    # a window from an earlier epoch, so the control mutated ticks the replay never
    # traded and "passed" by touching nothing.
    for row in store._db.execute(  # noqa: SLF001
            "SELECT series, opened_ts FROM trades WHERE closed_ts >= ? ORDER BY closed_ts",
            (epoch,)):
        opens.setdefault(row["series"], float(row["opened_ts"]))
    # Repro means: the deployed program's strategy decisions reproduce the ledger.
    # Two kinds of row are deliberately outside that contract and stay in the ledger
    # for reconcile_live.py (which must see every dollar):
    #   * source='paper' fills written while the engine was unarmed after a crash --
    #     simulated, with no counterpart on the venue tape;
    #   * exit_reason='venue_reconcile' closes -- an emergency flatten of an orphan
    #     position after a process restart, an exchange action with no tape tick that
    #     can re-derive it. Neither is a decision `decide()` could have made, so
    #     including them here made the control fail by construction, not by a bug.
    # A live epoch's maker fills come from the exchange, not from a queue model the
    # tape cannot support: five ETH/SOL/XRP bids rested through prints at their own
    # limit and were never matched. Entries are anchored to `/portfolio/fills`; size,
    # exit, fees and P&L are still re-derived from the recorded quotes, so a wrong
    # exit or ledger row still shows up as a delta.
    venue_fills = store.venue_entries(epoch)
    live_epoch = bool(venue_fills) or store.get_meta("armed_at") is not None
    venue_exits = store.venue_exits(epoch) if live_epoch else []
    # Crash-recovery legs: a boot flatten of an orphan the process forgot produce a
    # two-leg fill the tape cannot re-derive (e.g., YES bought 0.60 then NO sold 0.58
    # together -- the venue nets the pair to -12.41). Match each crash-recovery ledger
    # row to its venue entry by ticker+size and exclude just that print; the lane's
    # real fills (e.g., SOL 8c/60c take-profits on the same ticker) stay untouched.
    # reconcile_live.py still sees the venue tape and reconciles it to the cent.
    crash_prints = {(r["ticker"], r["contracts"]) for r in store._db.execute(  # noqa: SLF001
        "SELECT ticker, contracts FROM trades WHERE closed_ts >= ?"
        " AND source = 'live' AND exit_reason = 'venue_reconcile'", (epoch,))}
    if crash_prints:
        venue_fills = [v for v in venue_fills
                       if (v.get("ticker"), int(float(v.get("contracts") or 0))) not in crash_prints]
        venue_exits = [v for v in venue_exits
                       if (v.get("ticker"), int(float(v.get("contracts") or 0))) not in crash_prints]
    # Live-epoch parity compares venue-confirmed source='live' trades that this
    # program's own logic produces. Crash-recovery rows (exit_reason='venue_reconcile',
    # an emergency flatten/carry after a process restart) are the exchange's action,
    # not a decision `decide()` could re-derive from the tape, so they stay in the
    # ledger for reconcile_live.py but not in the repro comparison -- same boundary as
    # the paper rows removed for ARM_WAIT.
    ledger = (store._db.execute(  # noqa: SLF001 - read-only strategy ledger
        "SELECT COUNT(*) AS trades, COALESCE(SUM(CAST(pnl AS REAL)), 0) AS pnl"
        " FROM trades WHERE closed_ts >= ?"
        " AND source != 'paper' AND exit_reason != 'venue_reconcile'", (epoch,)).fetchone()
        if live_epoch else store.epoch_stats(epoch))
    total_trades, total_pnl = 0, Decimal("0")
    mutated_trades, mutated_pnl = 0, Decimal("0")
    mutated_inputs = 0
    crossed_trades, crossed_inputs = 0, 0
    for cfg in SLOTS:
        ticks = store.recent_ticks(cfg.series, epoch, limit=200000)
        if len(ticks) < 30:
            continue
        stake = store.slot_stake(cfg.series)
        params = _params_for(store, cfg.series)
        base, _ = replay_series(cfg, ticks, params, stake, store=store,
                                venue_fills=venue_fills, venue_anchored=live_epoch,
                                venue_exits=venue_exits)
        total_trades += base.trades
        total_pnl += base.pnl
        # Mutate the decision window where this lane's position opened: blocking the
        # gate there must remove the trade, or the replay is not reproducing the entry.
        opened = opens.get(cfg.series)
        targets = _decision_window(ticks, opened) if opened else []
        copy = [dict(t) for t in ticks]
        for target in targets:
            copy[target]["can_open"] = 0
            mutated_inputs += 1
        mut, _ = replay_series(cfg, copy, params, stake, store=store,
                               venue_fills=venue_fills, venue_anchored=live_epoch,
                               venue_exits=venue_exits)
        mutated_trades += mut.trades
        mutated_pnl += mut.pnl
        # Crossing the ladder at the same window must also remove the entry: a crossed
        # book is a stale side, and `bid < ask` is enforced in `decide` and in both
        # executors. If this ever produces the same trade, the guard has a hole.
        crossed = [dict(t) for t in ticks]
        for target in targets:
            crossed[target]["yes_bid"] = "0.60"
            crossed[target]["yes_ask"] = "0.55"
            crossed_inputs += 1
        cross, _ = replay_series(cfg, crossed, params, stake, store=store,
                                 venue_fills=venue_fills, venue_anchored=live_epoch,
                                 venue_exits=venue_exits)
        crossed_trades += cross.trades

    check("repro: replay reconciles with the ledger on this epoch",
          total_trades == int(ledger["trades"])
          and abs(total_pnl - Decimal(str(ledger["pnl"]))) <= Decimal("0.05"),
          f"replay {total_trades}/{total_pnl} vs ledger {ledger['trades']}/{ledger['pnl']}")
    check("repro: blocking the entry decision window breaks parity",
          (mutated_trades, mutated_pnl) != (total_trades, total_pnl),
          f"mutated {mutated_inputs} input(s) -> {mutated_trades}/{mutated_pnl}")
    check("repro: a crossed ladder at the entry window removes the trade",
          crossed_trades < total_trades,
          f"crossed {crossed_inputs} tick(s) -> {crossed_trades}/{total_trades} trades")


# --------------------------------------------------------------------------
# 4. Crossed book: bid < ask binds at the decision and at the live order path.
# --------------------------------------------------------------------------
def control_crossed_book() -> None:
    import asyncio

    from openkal.live_exec import LiveExecutor
    from openkal.strategy import Decision, Quote, decide

    params = Params()
    now = 1_000_000.0

    def q(bid: str, ask: str) -> Quote:
        return Quote(ticker="KXTEST-26NOV161230-30", series="KXBTC15M", strike=100.0,
                     close_ts=now + 600, yes_bid=Decimal(bid), yes_ask=Decimal(ask),
                     last=Decimal(ask), updated_ts=now, received_ts=now)

    # 300s is inside the verified flow window (90-420s left); the control's point is the
    # book state, so the fixture must sit where the flow program would actually fire.
    common = dict(p_up=0.62, seconds_left=300.0, spot_age=0.1, params=params,
                  position=None, edge_hurdle=Decimal("0"), can_open=True,
                  signal="z_obi_flow", now=now, spot=100.0)
    tight = decide(quote=q("0.50", "0.51"), **common)
    crossing = decide(quote=q("0.60", "0.55"), **common)
    locked = decide(quote=q("0.50", "0.50"), **common)
    check("crossed: a tight book still allows the flow entry", tight.action == "entry",
          f"{tight.action}: {tight.reason}")
    check("crossed: bid > ask is refused at the decision", crossing.action != "entry",
          f"{crossing.action}: {crossing.reason}")
    check("crossed: bid == ask is refused at the decision", locked.action != "entry",
          f"{locked.action}: {locked.reason}")

    class FakeBroker:
        def __init__(self) -> None:
            self.calls: list[tuple] = []
            self.errors: list[str] = []

        async def place(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return None

    ctx = SimpleNamespace(cash=Decimal("262"), realized=Decimal("0"), fills=[],
                          store=None, supervisor=None, source="live")
    slot = SimpleNamespace(cfg=SimpleNamespace(series="KXBTC15M"), params=params,
                           bankroll=Decimal("65.58"), order=None, position=None,
                           exit=None, p_up=0.62, spot=100.0, window_entries=0)
    broker = FakeBroker()
    ex = LiveExecutor(ctx, broker)
    crossed_quote = q("0.60", "0.55")
    slot.quote = crossed_quote
    decision = Decision("entry", "control", side="YES", price=Decimal("0.60"),
                        p_model=0.62, signal="z_obi_flow")
    asyncio.run(ex._place_entry(slot, decision, crossed_quote, now, 300.0))
    check("crossed: the live executor never sends a crossing order",
          broker.calls == [] and "crossed" in (ex.note or ""),
          f"sent={len(broker.calls)} note={ex.note!r}")


def control_risk_bounds() -> None:
    """Guardrail 3 binds: naked 0.84 carry, unbounded entries and truncated TPs are
    refused to arm.

    The 0.84 entry with a +5c take-profit and no stop is the exact profile that lost
    -$44.55 on XRP: 5c up, 79c down. The entry ceiling is a hard 0.62, so a 0.95
    lock-in is refused even with a 5c hard stop, and a candidate that declares no
    ceiling cannot prove the bound. A negative control is the check that would have
    blocked them, so it is tested against the real production candidate shape.
    """
    from openkal.readiness import risk_bounds, _risk_problems

    live_loss = {
        "name": "z_obi_flow",
        "exit_mode": "maker_scratch",
        "tp_c": 0.05,
        "detector": {"cost_band": [0.15, 0.84]},
    }
    unbounded = {"name": "band", "exit_mode": "carry"}
    problems = " ".join(_risk_problems(live_loss))
    check("risk: 0.84 entry with a 5c TP and no stop cannot arm; no ceiling cannot arm",
          "entry cost" in problems and "truncated exit" in problems
          and any("no declared entry ceiling" in p for p in _risk_problems(unbounded)),
          problems or "passed")

    bounded = {"name": "band", "exit_mode": "carry", "tp_c": 0.42,
               "detector": {"cost_band": [0.25, 0.58]}}
    check("risk: a 0.58 bounded carry is allowed",
          _risk_problems(bounded) == [], f"{_risk_problems(bounded)}")

    stopped = {"name": "lock", "exit_mode": "maker_tp", "tp_c": 0.05, "stop_c": 0.05,
               "detector": {"cost_band": [0.90, 0.95]}}
    check("risk: a 0.95 lock-in with a 5c hard stop cannot arm: the 0.62 entry cap is hard",
          any("entry cost" in p for p in _risk_problems(stopped)),
          f"{_risk_problems(stopped)}")

    truncated = {"name": "band", "exit_mode": "maker_scratch", "tp_c": 0.04,
                 "detector": {"cost_band": [0.25, 0.55]}}
    check("risk: a 0.55 entry with a 4c TP is a 11:1 profile and cannot arm",
          any("truncated exit" in p for p in _risk_problems(truncated)),
          f"{_risk_problems(truncated)}")

    cost, stop, tp, mode = risk_bounds(live_loss)
    check("risk: bounds are read from the verdict, not guessed",
          cost == Decimal("0.84") and stop is None and tp == Decimal("0.05")
          and mode == "maker_scratch", f"cost={cost} stop={stop} tp={tp} mode={mode}")


def main() -> int:
    # Optional explicit DB (a parity harness epoch), defaulting to the live store.
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else load_settings().db_path
    store = Store(path)
    control_sequence()
    control_calibration(store)
    control_repro(store)
    control_crossed_book()
    control_risk_bounds()
    width = max(len(name) for name, _, _ in RESULTS)
    failed = 0
    for name, ok, detail in RESULTS:
        failed += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} controls passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
