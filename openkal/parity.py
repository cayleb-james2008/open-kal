"""Decision parity: the live detector's decision vs the canonical path's.

`tests/test_gate_controls.py` compares the live *ledger* to a replay. That control
can only see a disagreement after a fill, so a detector that changed its mind
between the hot path and the replay could go unnoticed for a whole window.

This module compares the decision itself. Every tick the engine writes carries the
inputs the live detector used (quote, spot, p_model, signal, hurdle, spot_age,
impulse, quote age, strike) and the decision it produced. `canonical_decision`
rebuilds those inputs and runs `strategy.decide` -- the function the replay drives --
so any drift between what the live loop decided and what the canonical path decides
for the same tape is a mismatch, not a rounding difference.

The check is deliberately loud: `lane_parity` returns every mismatch it found, the
readiness gate refuses to arm while any exist, and `tests/test_paper_guards.py`
injects a wrong decision and requires the checker to catch it.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any

from .strategy import Decision, Params, Quote, decide

# Programs whose flat-lane decision needs inputs the tape does not carry (the
# settlement program reads L2 size and two flow windows; the dispersion program two
# implied-vol ratios). They are counted as skipped rather than silently passed: a
# lane running one of them has no decision parity until those inputs are recorded.
UNREPRODUCIBLE = {"student_t_l2_settle", "cross_lane_iv_arb"}


def decision_tag(decision: Decision) -> str:
    """The tape's one-line form of a decision."""
    return f"{decision.action}:{decision.side or '-'}"


def _decimal(value: Any) -> Decimal | None:
    try:
        return Decimal(str(value))
    except Exception:  # noqa: BLE001 - a malformed tape row is not a decision
        return None


def canonical_decision(tick: dict[str, Any], params: Params, *,
                       series: str | None = None) -> Decision | None:
    """Rebuild the tick's inputs and run the canonical `decide`.

    None means the row cannot answer the question (missing quote fields, no
    strike). Callers count those as skipped, never as matches.
    """
    bid = _decimal(tick.get("yes_bid"))
    ask = _decimal(tick.get("yes_ask"))
    last = _decimal(tick.get("last"))
    if bid is None or ask is None or last is None:
        return None
    if tick.get("strike") in (None, ""):
        return None
    ts = float(tick["ts"])
    quote_age = float(tick.get("quote_age") or 0.0)
    quote = Quote(
        ticker=str(tick["ticker"]),
        series=series or str(tick["series"]),
        strike=float(tick["strike"]),
        close_ts=ts + float(tick["secs_left"]),
        yes_bid=bid,
        yes_ask=ask,
        last=last,
        # `age_s` is the field `decide` reads for the stale-quote refusal, and the
        # tape records it: setting updated_ts to the receive time (as the replay's
        # quote builder does) would make every tick look fresh and hide a real
        # live/lookalike difference.
        updated_ts=ts - quote_age,
        received_ts=ts,
    )
    hurdle = params.edge_hurdle
    if tick.get("hurdle"):
        hurdle = _decimal(tick["hurdle"]) or hurdle
    try:
        p_model = float(tick["p_model"])
    except (KeyError, TypeError, ValueError):
        return None
    return decide(
        quote=quote,
        p_up=p_model,
        seconds_left=float(tick["secs_left"]),
        spot_age=float(tick.get("spot_age") or 0.0),
        params=params,
        position=None,
        edge_hurdle=hurdle,
        can_open=bool(tick.get("can_open")),
        signal=str(tick.get("signal") or ""),
        now=ts,
        spot=float(tick.get("spot") or 0.0),
        impulse_bps=float(tick.get("impulse_bps") or 0.0),
    )


def lane_parity(series: str, ticks: list[dict[str, Any]], params: Params) -> dict[str, Any]:
    """Compare every recorded decision on one lane to the canonical decision."""
    checked = 0
    mismatches = 0
    skipped = 0
    examples: list[dict[str, Any]] = []
    for tick in ticks:
        tag = str(tick.get("decision") or "").strip()
        if not tag or ":" not in tag:
            skipped += 1
            continue
        if str(tick.get("signal") or "") in UNREPRODUCIBLE:
            skipped += 1
            continue
        canonical = canonical_decision(tick, params, series=series)
        if canonical is None:
            skipped += 1
            continue
        checked += 1
        want = decision_tag(canonical)
        if want != tag:
            mismatches += 1
            if len(examples) < 5:
                examples.append({"ts": float(tick["ts"]), "ticker": tick.get("ticker"),
                                 "signal": tick.get("signal"), "live": tag,
                                 "canonical": want, "reason": canonical.reason})
    return {"series": series, "checked": checked, "mismatches": mismatches,
            "skipped": skipped, "examples": examples, "ok": mismatches == 0}
