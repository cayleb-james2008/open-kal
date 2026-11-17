"""Stop-Hypothesize-Reiterate.

A slot that fails does not wait for tomorrow and does not sit in a timed cooldown.
It stops, flattens, captures the exact ticks that failed, proposes bounded variants
of the EVOLVE-BLOCK, and resumes only when the immutable verifier approves one.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from . import verifier
from .store import Store
from .strategy import Params, clamp_params


@dataclass
class Hypothesis:
    name: str
    params: Params
    note: str
    score: float = 0.0
    profit_factor: float = 0.0
    ece: float = 0.0
    status: str = "proposed"       # proposed | rejected | promoted


@dataclass
class SlotFault:
    reason: str
    ts: float
    ticks: int


class Proposer:
    """Bounded mutations around the halted program.

    Deterministic here, so it needs no key and no network: each candidate changes one
    or two genes with a stated hypothesis. An LLM proposer can be dropped in behind
    `propose()` without touching the verifier.
    """

    def propose(self, base: Params, failure: dict[str, Any]) -> list[Hypothesis]:
        weak_fill = failure.get("fill_rate", 1.0) < 0.5
        losing = failure.get("win_rate", 0.5) < 0.5
        wide = failure.get("avg_spread", 0.03) > 0.02
        out = [
            Hypothesis("raise hurdle", base.variant(
                edge_hurdle=min(Decimal("0.06"), base.edge_hurdle + Decimal("0.005"))),
                "the losing entries were marginal: demand more edge before quoting"),
            Hypothesis("tighter band", base.variant(
                min_price=base.min_price + Decimal("0.02"),
                max_price=base.max_price - Decimal("0.02")),
                "trim both tails; the 20-85c band still holds the body"),
            Hypothesis("faster stop", base.variant(
                stop_loss=max(Decimal("0.015"), base.stop_loss - Decimal("0.005"))),
                "cut a broken thesis sooner, pay the taker fee less often"),
            Hypothesis("wider take-profit" if weak_fill else "thinner take-profit", base.variant(
                take_profit=min(Decimal("0.10"), base.take_profit + Decimal("0.01")) if weak_fill
                else max(Decimal("0.02"), base.take_profit - Decimal("0.005"))),
                "queue position is the bottleneck, not the exit price" if weak_fill
                else "bank the move while the maker queue is still ours"),
            Hypothesis("passive one more tick", base.variant(
                maker_offset_ticks=min(2, base.maker_offset_ticks + 1)),
                "buy queue priority with a tick instead of crossing"),
            Hypothesis("later entries", base.variant(
                min_seconds_left=min(Decimal("240"), Decimal(str(base.min_seconds_left)) + Decimal("30"))),
                "early-window noise was the loss source" if losing else "more of the window is tradeable"),
            Hypothesis("spread discipline", base.variant(
                max_spread=max(Decimal("0.02"), min(Decimal("0.08"), base.max_spread - Decimal("0.01")))),
                "only quote when the book is genuinely tight" if wide else "hold the current spread rule"),
        ]
        return out


@dataclass
class Supervisor:
    store: Store
    proposer: Proposer = field(default_factory=Proposer)
    state: str = "RUNNING"           # RUNNING | HALT_FLATTEN | CAPTURE | PROPOSE | VERIFY | SWAP | ABSTAIN
    fault: SlotFault | None = None
    candidates: list[Hypothesis] = field(default_factory=list)
    last_run: float = 0.0
    generation: int = 0
    promoted: int = 0
    rejected: int = 0
    timeline: list[dict[str, Any]] = field(default_factory=list)

    def note(self, state: str, detail: str) -> None:
        self.state = state
        self.timeline.append({"ts": time.time(), "state": state, "detail": detail})
        self.timeline = self.timeline[-40:]

    def triggered(self, slot: Any) -> str | None:
        if slot.consecutive_losses >= 3:
            return "3 consecutive net losses"
        if slot.peak_bankroll > 0 and slot.bankroll < slot.peak_bankroll * Decimal("0.88"):
            return "slot drawdown > 12%"
        if slot.ece > 0.08:
            return f"ECE {slot.ece:.3f} > 0.08"
        return None

    def start(self, slot: Any, reason: str) -> None:
        self.fault = SlotFault(reason=reason, ts=time.time(), ticks=0)
        slot.halted = True
        slot.flatten_requested = True
        self.note("HALT_FLATTEN", reason)

    def run(self, slot: Any, stake: Decimal, now: float) -> None:
        """One pass of capture -> propose -> verify -> swap/abstain. Runs in milliseconds."""
        if not slot.halted:
            return
        if now - self.last_run < 15.0:
            return
        self.last_run = now
        since = now - 3600
        ticks = self.store.recent_ticks(slot.cfg.series, since, limit=2400)
        failure_ticks, holdout_ticks = verifier.held_out(ticks)
        if self.fault:
            self.fault.ticks = len(ticks)

        self.note("CAPTURE", f"{len(ticks)} ticks, {len(failure_ticks)} in the failure slice")
        if len(failure_ticks) < 40 or len(holdout_ticks) < 40:
            self.note("ABSTAIN", "not enough tape to verify a replacement yet")
            return

        trades = self.store.recent_trades(60)
        won = sum(1 for t in trades if float(t["pnl"]) > 0)
        failure = {
            "win_rate": won / len(trades) if trades else 0.5,
            "fill_rate": 0.5,
            "avg_spread": sum(float(t.get("spread") or 0.03) for t in ticks[-200:]) / max(1, len(ticks[-200:])),
        }

        self.note("PROPOSE", "generating bounded EVOLVE-BLOCK variants")
        base = slot.params
        self.candidates = [c for c in self.proposer.propose(base, failure)]
        baseline = verifier.replay(failure_ticks, base, stake, store=self.store)
        baseline_holdout = verifier.replay(holdout_ticks, base, stake, store=self.store)

        self.note("VERIFY", f"replaying {len(self.candidates)} candidates across two slices")
        best: Hypothesis | None = None
        for cand in self.candidates:
            cand.params = clamp_params(cand.params)
            result = verifier.replay(failure_ticks, cand.params, stake, store=self.store)
            hold = verifier.replay(holdout_ticks, cand.params, stake, store=self.store)
            ok, why = verifier.passes(result, baseline)
            ok_hold, why_hold = verifier.passes(hold, baseline_holdout)
            cand.score = float(result.pnl)
            cand.profit_factor = result.profit_factor
            cand.ece = result.ece
            cand.status = "proposed" if not (ok and ok_hold) else "verified"
            cand.note = f"{cand.note} | {why}{'' if ok_hold else '; holdout: ' + why_hold}"
            if ok and ok_hold and (best is None or cand.score > best.score):
                best = cand
            self.store.add_program(
                ts=now, slot=slot.cfg.series, name=cand.name, params=cand.params.to_json(),
                status=cand.status, score=cand.score, profit_factor=cand.profit_factor,
                ece=cand.ece, note=cand.note, generation=self.generation + 1,
            )

        if best is not None:
            self.generation += 1
            slot.params = best.params
            slot.halted = False
            slot.flatten_requested = False
            slot.consecutive_losses = 0
            slot.peak_bankroll = slot.bankroll
            best.status = "promoted"
            self.promoted += 1
            self.note("SWAP", f"promoted '{best.name}' at +${best.score:.2f} replayed")
            self.fault = None
            return

        self.rejected += 1
        self.note("ABSTAIN", "no candidate cleared both slices; slot stays flat and keeps searching")
