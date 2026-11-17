"""Immutable verifier: separate process-plane, read-only, no path to order placement.

The recursive loop may write candidate parameters; it may never write here. Every
candidate is scored on recorded ticks only, with the same fee math the engine uses,
and it must beat the halted baseline on both the failure slice and a held-out slice.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .fees import maker_fee, taker_fee
from .features import expected_calibration_error
from .strategy import Params, contracts_for

TICK = Decimal("0.01")
# The same four numbers `readiness.py` arms on, in the process that produces them.
# Keeping one copy of the gate in the verifier and another in readiness is how a
# candidate could be "verified" at ECE 0.06 and then refused at 0.08, or worse,
# the other way round.
REQUIRED_PF = 1.15
REQUIRED_ECE = 0.08
MIN_OOS_TRADES = 20
MAX_ENTRY_COST = Decimal("0.62")


@dataclass(frozen=True)
class ReplayResult:
    trades: int
    pnl: Decimal
    profit_factor: float
    win_rate: float
    ece: float
    max_drawdown: Decimal
    crossed_fills: int = 0
    max_entry_cost: Decimal = Decimal("0")


def replay(ticks: list[dict], params: Params, stake: Decimal,
           store: object | None = None) -> ReplayResult:
    """Score a candidate with the engine's own execution core over recorded ticks.

    There is no second fill model: `replay.replay_series` drives `execution.Executor`,
    the same code that fills the live ledger. A candidate that looks good here is
    good under the exact path that would trade it.
    """
    from types import SimpleNamespace

    from .replay import replay_series

    cfg = SimpleNamespace(series=ticks[0].get("series", "REPLAY") if ticks else "REPLAY")
    # The store carries the venue print tape a passive entry needs to prove its fill.
    report, ctx = replay_series(cfg, ticks, params, stake, store=store)
    crossed = 0
    costs: list[Decimal] = []
    for fill in getattr(ctx, "fills", []):
        if fill.get("kind") == "entry":
            price = Decimal(str(fill.get("price") or "0"))
            costs.append(price if fill.get("side") == "YES" else Decimal(1) - price)
        elif "crossed" in str(fill.get("reason") or ""):
            # A crossed book is refused, not filtered: the refusal is the evidence
            # that the path saw one, and a live book that crossed under an order is
            # a fill the venue would never have given.
            crossed += 1
    return ReplayResult(
        trades=report.trades,
        pnl=report.pnl,
        profit_factor=report.profit_factor,
        win_rate=report.win_rate,
        ece=_replay_ece(ctx),
        max_drawdown=report.max_drawdown,
        crossed_fills=crossed,
        max_entry_cost=max(costs) if costs else Decimal("0"),
    )


def _replay_ece(ctx) -> float:
    """Calibration against what the trade actually did, not what the market did.

    `1 if net > 0 else 0` is the event the strategy's own probability claims to
    describe for a payoff-calibrated program, and it is the event this number
    measures for every program: a take-profit round trip that wins 90% of the
    time while settling 50% is well-calibrated about its *trades* and badly
    calibrated about the *market*, and those two are not the same claim. The
    probability read here is the side's own (`p_entry` if the trade was long YES,
    `1 - p_entry` if it was long NO); reading the raw YES probability for both
    scored a NO trade against the wrong event.
    """
    from .features import expected_calibration_error

    pairs = []
    for row in getattr(ctx, "trade_log", []):
        p = float(row.get("p_side", row.get("p")))
        pairs.append((p, 1 if Decimal(str(row["pnl"])) > 0 else 0))
    return expected_calibration_error(pairs) if pairs else 0.0


def passes(result: ReplayResult, baseline: ReplayResult,
           min_trades: int = MIN_OOS_TRADES) -> tuple[bool, str]:
    """The gate: a real sample, better than the halted baseline, no overfit, calibrated.

    The sample floor is 20 trades, not 8: below it the profit factor is three
    winners deciding a number that gets an account armed.
    """
    if result.trades < min_trades:
        return False, f"only {result.trades} replayed trades"
    if result.pnl <= baseline.pnl:
        return False, "does not beat the halted baseline"
    if result.profit_factor < REQUIRED_PF:
        return False, f"profit factor {result.profit_factor:.2f} < {REQUIRED_PF}"
    if result.ece > REQUIRED_ECE:
        return False, f"payoff ECE {result.ece:.3f} > {REQUIRED_ECE}"
    if result.crossed_fills != 0:
        return False, f"{result.crossed_fills} crossed-book fills"
    if result.max_entry_cost > MAX_ENTRY_COST:
        return False, f"entry cost {result.max_entry_cost} > {MAX_ENTRY_COST}"
    return True, "verified on both slices"


def held_out(ticks: list[dict], fraction: float = 0.4) -> tuple[list[dict], list[dict]]:
    """Split the tape: recent failure window for the proposer, older slice for the verifier."""
    if len(ticks) < 2:
        return ticks, []
    cut = int(len(ticks) * (1 - fraction))
    return ticks[cut:], ticks[:cut]
