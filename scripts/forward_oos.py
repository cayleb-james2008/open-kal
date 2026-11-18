#!/usr/bin/env python3
"""Score the frozen F2 rule ONCE on the forward out-of-sample slice.

Everything here is frozen: the detector thresholds, the cost band, the 1s
execution lag, the +5c maker take-profit, the 7c two-second-confirmed stop, the
90s barrier, the payoff calibrator, and the desk's entry hurdle. This script
does not tune, and it must not be given a threshold to try.

Every fill and every exit is priced by `openkal/sim_core.py`, the one fill model
the live desk also reasons about: the order arrives a full second after its
trigger, a passive leg needs >= 50 contracts printed at or through its limit, and
a NO price is always derived from the YES ladder (`NO ask = 1 - yes_bid`) rather
than passed in. The feature side reads only the tape that existed at the trigger:
`obi_pool(..., spot_past=True)` takes the last settlement-index print at or
before the row, never a later one.

That matters because the earlier harness was biased. It priced an entry off a
quote that could be crossed, it read the spot print stamped up to 1.5s *after* the
trigger, and it let a passive NO leg fill against the wrong side of the tape. The
result was a clean-looking 25/25, +$41.38 forward score that could not be
reproduced live. The legacy scorer is gone; `tests/test_sim_core.py` (26 checks,
8 injected bugs each failing at least one) is what certifies the model now.

The forward slice is every settled window whose close is at or after
`FORWARD_OOS_START_TS`. It was recorded while nothing was armed, so it is the
first slice in this project that has not been read by a fit of any kind.

    .venv/bin/python scripts/forward_oos.py                      # default slice
    .venv/bin/python scripts/forward_oos.py --since 0 --out data/f2-repro.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openkal import features, sim_core  # noqa: E402
from openkal.config import (HARD_MAX_ENTRY_PRICE, REGIME_MAX_ECE,  # noqa: E402
                            REGIME_MIN_PROFIT_FACTOR, REGIME_MIN_PROFITABLE_LANES,
                            REGIME_MIN_TRADES, REGIME_WINDOW_HOURS)
from openkal.decision_gate import GATE_REFRESH_S  # noqa: E402
from openkal.strategy import (Params, Quote, decide, exit_cost,  # noqa: E402
                              taker_fee_per_contract)
from scripts import alpha_discover as AD  # noqa: E402
from scripts import alpha_search as A  # noqa: E402
from scripts.alpha_discover import Lane  # noqa: E402
from scripts.bounded_sweep import obi_pool, pick, summarize  # noqa: E402

LANES = ("KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M")
FORWARD_OOS_START_TS = 1794909600          # 2026-11-17 05:00:00 EST
OUT = ROOT / "data" / "forward-oos.json"
OUT_STUDENT_T = ROOT / "data" / "forward-oos-student-t-l2.json"
OUT_CROSS_LANE = ROOT / "data" / "forward-oos-cross-lane-iv.json"

# --- the frozen rule (scripts/f2_structural.py, unchanged) ------------------
Z_MIN = features.F2_Z_MIN
FLOW_MIN = features.F2_FLOW_MIN
REV_MIN = features.F2_REV_MIN
DELTA_BPS_MIN = features.F2_DELTA_BPS_MIN
COST_LO, COST_HI = 0.25, 0.60
CAP = 0.62
TP_C = 0.05
STOP_C = 0.07
STOP_CONFIRM_S = 2.0
BARRIER_S = 90.0
MIN_FILL = 50.0
MAX_QTY = 50
LAG_S = 1.0
T_LO, T_HI = 180.0, 780.0
COOLDOWN, PER_WINDOW = 90.0, 3
MIN_TRADES = 20
REQUIRED_PF = 1.15
REQUIRED_ECE = 0.08


def frozen_mask(pool: dict) -> tuple[np.ndarray, np.ndarray]:
    z = pool["z"]
    d = np.sign(z)
    ly = z > 0
    cost = np.where(ly, pool["ask"], 1.0 - pool["bid"])
    mask = (
        (np.abs(z) >= Z_MIN)
        & (pool["flow"] * d >= FLOW_MIN)
        & (pool["rev"] * d >= REV_MIN)
        & (pool["delta"] >= DELTA_BPS_MIN * 1e-4 * np.maximum(pool["spot"], 1.0))
        & (cost >= COST_LO - 1e-9)
        & (cost <= COST_HI + 1e-9)
    )
    return mask, ly


def triggers_for(lane: Lane, pool: dict, closes: set[float]) -> list[tuple]:
    """The frozen rule's triggers on the selected windows, at their evidence row.

    A trigger is only a timestamp, a side and the row's outcome. No price is
    carried out of here: `sim_core` derives the entry from the YES quote it meets
    one second later, which is why a stale or crossed arrival can no longer be
    priced as if it were tradeable.
    """
    mask, ly = frozen_mask(pool)
    if closes:
        mask = mask & np.isin(pool["close"], np.fromiter(closes, dtype=np.float64))
    out = []
    for i in pick(pool, mask, cooldown=COOLDOWN, per_window=PER_WINDOW):
        out.append((int(pool["tid"][i]), str(pool["ticker"][i]),
                    float(pool["ts"][i]), float(pool["close"][i]),
                    int(pool["y"][i]), bool(ly[i]), i))
    return out


def simulate(lane: Lane, tid: int, trigger: float, close: float, y: int,
             long_yes: bool):
    """One trigger through the canonical simulator: T+1s in, T+1s exit placement.

    The 90s barrier crosses out at the book (the rule's time stop), the +5c leg is
    a free resting sell that arrives a second after the fill and needs 50 prints
    through it, and the 7c stop is a two-second-confirmed breach whose exit order
    also arrives a second after confirmation. Crossed or empty arrival rows are
    counted skips, never walked back to a friendlier book.
    """
    return sim_core.run_trade(
        sim_core.Tape.from_lane(lane, tid, float(close), int(y)),
        side=sim_core.YES if long_yes else sim_core.NO,
        trigger_ts=float(trigger),
        entry_style=sim_core.TAKER,
        qty=MAX_QTY,
        lag_s=LAG_S,
        max_cost=Decimal(str(CAP)),
        cost_lo=Decimal(str(COST_LO)),
        tp_c=Decimal(str(TP_C)),
        stop_c=Decimal(str(STOP_C)),
        stop_confirm_s=STOP_CONFIRM_S,
        hold_s=BARRIER_S,
        barrier_exit="taker",
        min_print=MIN_FILL,
    )


def feature_row(pool: dict, i: int) -> list[float]:
    z = float(pool["z"][i])
    d = 1.0 if z > 0 else -1.0
    delta_bps = float(pool["delta"][i]) / max(float(pool["spot"][i]), 1.0) * 1e4
    return features.f2_features(z=z, flow=float(pool["flow"][i]) * d,
                                revision=float(pool["rev"][i]) * d,
                                delta_bps=delta_bps)


def _with_true_side_costs(summary: dict, rows: list[dict]) -> dict:
    """Correct the side-cost statistics for the forward tape's row convention.

    `bounded_sweep.summarize` reads `entry` as the YES-side price, which is what
    the research sweeps write. The forward tape writes the cost of the side
    actually held -- the premium at risk -- and `scripts/kelly_compound_sim.py`
    reads it that way. The two readings differ for a NO trade, and the cost
    ceiling is a check on the premium, so it has to be measured on the premium:
    a NO bought for 36c is a 36c premium, not a 64c one.
    """
    if not rows:
        return summary
    costs = [float(r["entry"]) for r in rows]
    summary["avg_entry"] = round(float(np.mean(costs)), 4)
    summary["max_side_cost"] = round(max(costs), 4)
    return summary


def gate_from_summary(summary: dict, payoff_ece: float,
                      lanes_profitable: int | None = None,
                      regime: bool = False) -> dict:
    """The frozen gate, as a pure function of the scored numbers.

    Every condition is checked on its own, in the order a reader would: a rule
    that fails one of them is refused, whatever the others say. There is no
    aggregate score and no fallback -- an average would let a spectacular PF
    pay for a broken calibration or a crossed fill, which is exactly the trade
    the gate exists to forbid. `scripts/day_supervisor.py` calls this again on
    the artifact before it promotes anything, so a hand-edited or stale result
    file cannot arm the engine on numbers nobody checked.

    `regime=True` is the rolling 48h gate: the same seven conditions, with the
    trade floor a 48h slice can reach (12), the PF a 2-3 day edge must show
    (1.25) and the ECE a short, decile starved slice can still meet (0.10). The
    crossed-fill and cost conditions are *not* relaxed: a fill model that
    crossed a book is wrong at any sample size. Only the window and the counts
    move, never the simulator.
    """
    min_trades = REGIME_MIN_TRADES if regime else MIN_TRADES
    required_pf = float(REGIME_MIN_PROFIT_FACTOR) if regime else REQUIRED_PF
    required_ece = float(REGIME_MAX_ECE) if regime else REQUIRED_ECE
    min_lanes = REGIME_MIN_PROFITABLE_LANES if regime else 2
    if lanes_profitable is None:
        lanes_profitable = sum(1 for v in (summary.get("by_lane") or {}).values()
                               if float(v.get("net") or 0) > 0)
    reasons: list[str] = []
    if int(summary.get("trades") or 0) < min_trades:
        reasons.append(f"{int(summary.get('trades') or 0)} forward trades < {min_trades}")
    if float(summary.get("net_pnl") or 0) <= 0:
        reasons.append(f"forward net ${float(summary.get('net_pnl') or 0):.2f} <= 0")
    pf = float(summary.get("pf") or 0)
    # Exactly the seven conditions the plan names. No eighth condition is added
    # here and no condition is combined with another: a PF of 99 on two losers is
    # still PF 99 for this gate, because the trade floor is the protection.
    if pf < required_pf:
        reasons.append(f"forward PF {pf:.3f} < {required_pf}")
    if float(payoff_ece) > required_ece:
        reasons.append(f"payoff ECE {float(payoff_ece):.4f} > {required_ece}")
    if int(summary.get("crossed_fills") or 0) != 0:
        reasons.append(f"{int(summary.get('crossed_fills') or 0)} crossed fills")
    if float(summary.get("max_side_cost") or 0) > CAP + 1e-9:
        reasons.append(f"max cost {float(summary.get('max_side_cost') or 0):.3f} > {CAP}")
    if int(lanes_profitable) < min_lanes:
        reasons.append(f"{int(lanes_profitable)} profitable lane(s) < {min_lanes}")
    return {"pass": not reasons, "reasons": reasons,
            "lanes_profitable": int(lanes_profitable),
            "mode": "regime-48h" if regime else "frozen",
            "min_trades": min_trades, "required_pf": required_pf,
            "required_ece": required_ece, "min_lanes": min_lanes}


def score(since: float, until: float | None = None, regime: bool = False) -> dict:
    """Score the frozen rule on windows closing in `[since, until)`.

    `until` exists for the pre-slice plumbing replay: it bounds the scorer to
    windows that closed before the forward tape began, so a harness check can never
    read (or write a result for) the live slice. The noon run passes no `until`,
    which keeps its behaviour byte-identical to the single-shot study.
    """
    cal = features.load_payoff_calibrator(ROOT / "data" / "f2-structural.json")
    params = Params()
    hurdle = max(Decimal(str(params.edge_hurdle)), Decimal(str(params.min_net_edge)))
    lanes = {name: Lane(name) for name in LANES}
    rows: list[dict] = []
    events = 0
    skips: dict[str, int] = {}
    windows: set[float] = set()
    pools_ok: dict[str, int] = {}
    for name, lane in lanes.items():
        closes = {float(m["close"]) for m in lane.meta.values()
                  if float(m["close"]) >= since
                  and (until is None or float(m["close"]) < until)}
        pools_ok[name] = len(closes)
        if not closes:
            continue
        sig = A.trailing_sigma(lane)
        # spot_past: the z of a row is computed from the last index print at or
        # before it. The nearest-neighbour read used elsewhere in research can see
        # 1.5s into the future, which is not information a trigger had.
        raw = obi_pool(lane, sig, t_lo=T_LO, t_hi=T_HI,
                       delta_bps_floor=DELTA_BPS_MIN, spot_past=True)
        if not raw.get("tid"):
            continue
        pool = {k: np.concatenate(v) for k, v in raw.items()}
        if "lane" not in pool:
            pool["lane"] = np.array([name] * len(pool["tid"]), dtype=object)
        pool["ticker"] = np.array([lane.tickers[i] for i in pool["tid"]], dtype=object)
        triggers = triggers_for(lane, pool, closes)
        events += len(triggers)
        for tid, ticker, trigger, close, outcome, long_yes, i in triggers:
            res = simulate(lane, tid, trigger, close, outcome, long_yes)
            if isinstance(res, sim_core.Skip):
                skips[res.reason] = skips.get(res.reason, 0) + 1
                continue
            windows.add(close)
            entry_ts = float(res.legs[0].ts)
            rows.append({
                "lane": name, "ticker": ticker, "tid": tid, "ts": float(res.order_ts),
                "fill_ts": entry_ts, "exit_ts": float(res.exit_ts),
                "slice": "oos", "side": res.side, "entry_style": res.entry_style,
                "entry": str(res.entry), "exit": str(res.exit_value), "qty": res.qty,
                "pnl": str(res.pnl), "fee": str(res.fees), "reason": res.reason,
                "crossed": res.crossed, "requotes": res.requotes,
                "close": close, "y": outcome,
                "p_side_raw": float(pool["p_up"][i]),
                "p_side": cal.apply(feature_row(pool, i)),
            })

    kept: list[dict] = []
    for r in rows:
        lane = lanes[str(r["lane"])]
        try:
            b, a = lane.quote_at(int(r["tid"]), float(r["ts"]), valid_only=True,
                                 lookback=2.0)
            side_ask = sim_core.entry_price(str(r["side"]), sim_core.TAKER, b, a)
        except (TypeError, sim_core.CrossedBook, sim_core.NoQuote):
            r["decide_takes"] = False
            continue
        gross = exit_cost(side_ask, params)
        edge = Decimal(str(r["p_side"])) - side_ask
        net = edge - gross - taker_fee_per_contract(side_ask)
        r["decide_edge_c"] = float(net)
        r["decide_takes"] = bool(net >= hurdle)
        if r["decide_takes"]:
            kept.append(r)

    hours = ((max(windows) - min(windows)) / 3600.0) if len(windows) > 1 else 0.0
    summary = summarize(kept, hours) if kept else summarize([], 0.0)
    summary = _with_true_side_costs(summary, kept)
    # A crossed arrival is a counted skip in sim_core, not a fill, so the gate's
    # crossed-fill condition stays a real check on the model's output.
    summary["crossed_fills"] = sum(int(r["crossed"]) for r in kept)
    payoff_ece = round(features.expected_calibration_error(
        [(float(r["p_side"]), 1 if float(r["pnl"]) > 0 else 0) for r in kept]), 4)
    lanes_pos = sum(1 for v in (summary.get("by_lane") or {}).values()
                    if float(v.get("net") or 0) > 0)
    gate = gate_from_summary(summary, payoff_ece, lanes_pos, regime=regime)
    return {
        "generated": datetime.now(timezone.utc).isoformat(),
        "score_model": "openkal.sim_core",
        "regime_mode": regime,
        "gate_mode": gate["mode"],
        "fill_rules": {
            "entry_lag_s": LAG_S,
            "exit_placement_lag_s": sim_core.LAG_S,
            "min_print_contracts": MIN_FILL,
            "no_ask": "1 - yes_bid",
            "spot_lookahead_s": 0.0,
        },
        "since_ts": since,
        "since_local": datetime.fromtimestamp(since).astimezone().isoformat() if since else "all",
        "until_ts": until,
        "until_local": (datetime.fromtimestamp(until).astimezone().isoformat()
                        if until else None),
        "windows_per_lane": pools_ok,
        "windows_scored": len(windows),
        "events": events,
        "skips": dict(sorted(skips.items())),
        "calibration": cal.to_json(),
        "hurdle_c": float(hurdle),
        "all_trades": summarize(rows, hours) if rows else summarize([], 0.0),
        "trades": summary,
        "payoff_ece": payoff_ece,
        "lanes_profitable": lanes_pos,
        "gate": gate,
        "rows": kept,
    }


# ---------------------------------------------------------------------------
# student_t_l2_settle: the settlement program, scored on the same slice
#
# The rule is the production one. `openkal/strategy.py` decides whether a row is
# an entry -- the cost band, the true-L2 imbalance, the signed aggressor flow
# and the net edge -- and `openkal/sim_core.py` prices the fill: a taker entry
# that arrives a full second after its trigger and a contract carried to
# settlement for a $0 exit fee. Nothing here re-implements either: a scorer that
# carried its own copy of the rule would score a program the engine does not run.
# ---------------------------------------------------------------------------

SL2_COST_LO, SL2_COST_HI = 0.35, 0.60
SL2_CAP = float(HARD_MAX_ENTRY_PRICE)
SL2_SECS_LO, SL2_SECS_HI = 60.0, 780.0
# The settlement program is sampled on a 2-second grid: the thesis is priced off
# an average that is left to print, not off a single book row, and a grid keeps
# the number of Student-t evaluations on the whole slice in the tens of thousands.
SL2_GRID_S = 2.0
# The resting size must be this fresh: an imbalance read off a ladder that has
# already moved is not the book the decision was made against.
SL2_MAX_L2_AGE_S = features.STUDENT_T_MAX_DEPTH_AGE_S


def _lane_depth(name: str) -> tuple[np.ndarray, np.ndarray]:
    """The true L2 resting sizes, aligned with the lane's book rows."""
    with np.load(AD.CACHE / f"book_{name}.npz", allow_pickle=False) as data:
        return (data["yes_bid_sz"].astype(np.float64),
                data["yes_ask_sz"].astype(np.float64))


def _flow_at(trade_ts: np.ndarray, cum_signed: np.ndarray, cum_size: np.ndarray,
             t: float, window_s: float) -> float:
    """Signed aggressor print flow over the trailing `window_s`.

    The same quantity as `features.print_flow`, computed on the whole tape at
    once: signed printed size over total printed size, on prints at or before
    `t` and no earlier than `t - window_s`.
    """
    lo = int(np.searchsorted(trade_ts, t - window_s, "left"))
    hi = int(np.searchsorted(trade_ts, t, "right"))
    if hi <= lo:
        return 0.0
    den = float(cum_size[hi] - cum_size[lo])
    if den <= 0:
        return 0.0
    return float((cum_signed[hi] - cum_signed[lo]) / den)


def _gate_verdicts(db_path: Path, since: float = float("-inf"),
                   until: float = float("inf"), min_confidence: float = 0.0) -> dict:
    """Replay only the current-ticker snapshot known to the decision at that tick."""
    by_ticker: dict[str, list[tuple[float, float, float]]] = {}
    if not db_path.exists():
        return {}
    con = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        for series in LANES:
            # Include empty ticks: otherwise the last available vote continues
            # to vote after the engine has invalidated it.
            for ticker, ts, gate in con.execute(
                "SELECT ticker, ts, gate FROM ticks WHERE series=? AND ts>=? AND ts<? ORDER BY ts, rowid",
                (series, since, until),
            ):
                allow = verdict_ts = float("nan")
                if gate:
                    try:
                        value = json.loads(gate)
                        stamp = float(value["ts"])
                        confidence = float(value["confidence"])
                        if (value["ticker"] == ticker and value["available"] is True
                                and isinstance(value["allow"], bool) and 0 <= ts - stamp < GATE_REFRESH_S
                                and 0 <= confidence <= 1):
                            allow = float(value["allow"] and confidence >= min_confidence)
                            verdict_ts = stamp
                    except (KeyError, TypeError, ValueError):
                        pass
                by_ticker.setdefault(ticker, []).append((ts, allow, verdict_ts))
    finally:
        con.close()
    return {ticker: np.asarray(items, dtype=np.float64).T for ticker, items in by_ticker.items()}


def _gate_allow(verdicts: dict, ticker: str, t: float) -> bool | None:
    snapshots = verdicts.get(ticker)
    if snapshots is None:
        return None
    ts, allow, verdict_ts = snapshots
    idx = int(np.searchsorted(ts, t, "right")) - 1
    if idx < 0 or not 0 <= t - verdict_ts[idx] < GATE_REFRESH_S:
        return None
    return bool(allow[idx])


def simulate_settlement(lane: Lane, trig: dict):
    """One settlement trigger through the canonical simulator.

    A taker entry that arrives a full second after its trigger, at the cost of
    the side held, capped at the desk's $0.62 ceiling and inside the $0.35-$0.60
    band, and a contract carried to expiry for a $0 exit fee. Every one of those
    is a rule, not a default: the cap refuses an arrival above it, and the exit
    is settlement rather than a crossing.
    """
    return sim_core.run_trade(
        sim_core.Tape.from_lane(lane, int(trig["tid"]), float(trig["close"]),
                                int(trig["y"])),
        side=trig["side"],
        trigger_ts=float(trig["ts"]),
        entry_style=sim_core.TAKER,
        qty=MAX_QTY,
        lag_s=LAG_S,
        max_cost=Decimal(str(SL2_CAP)),
        cost_lo=Decimal(str(SL2_COST_LO)),
        cost_hi=Decimal(str(SL2_COST_HI)),
        barrier_exit="settle",
        min_print=MIN_FILL,
    )


def _settlement_triggers(lane: Lane, name: str, closes: set[float],
                          index_tape: list[tuple[float, float]],
                          params: Params) -> tuple[list[dict], Counter]:
    """The settlement program's own entries on one lane.

    Every row is put to `strategy.decide`, so what is scored here is the program
    the engine runs, not a research copy of it. The volatility is
    `features.trailing_sigma` on the same settlement index the live engine reads:
    one estimator, one number, or the scored program is not the traded program.
    """
    nu, s = features.STUDENT_T_FIT[name]
    depth_bid, depth_ask = _lane_depth(name)
    skips: Counter = Counter()
    triggers: list[dict] = []
    grid = np.arange(SL2_SECS_HI, SL2_SECS_LO - 1e-9, -SL2_GRID_S)
    for tid, ticker, meta, a, b in A._segments(lane):
        close = float(meta["close"])
        if close not in closes:
            continue
        strike = float(meta["strike"])
        ts = lane.ts[a:b]
        bid = lane.bid[a:b]
        ask = lane.ask[a:b]
        spots = A.index_at_past(lane, close - grid, tol=params.max_spot_age_s)
        lo_tr, hi_tr = int(lane.trade_bounds[tid]), int(lane.trade_bounds[tid + 1])
        trade_ts = lane.trade_ts[lo_tr:hi_tr]
        size = lane.trade_sz[lo_tr:hi_tr]
        sizes = np.where(np.isfinite(size) & (size > 0), size, 0.0)
        signed = np.where(lane.trade_ty[lo_tr:hi_tr] == 1, sizes, -sizes)
        cum_signed = np.concatenate(([0.0], np.cumsum(signed)))
        cum_size = np.concatenate(([0.0], np.cumsum(sizes)))
        for k, secs in enumerate(grid):
            t = close - float(secs)
            j = int(np.searchsorted(ts, t, "right")) - 1
            if j < 0:
                skips["no_book"] += 1
                continue
            if t - float(ts[j]) > SL2_MAX_L2_AGE_S:
                skips["stale_l2"] += 1
                continue
            spot = float(spots[k])
            if not (spot > 0):
                skips["no_spot"] += 1
                continue
            try:
                sim_core.check_book(float(bid[j]), float(ask[j]))
            except sim_core.CrossedBook:
                skips["quote_crossed"] += 1
                continue
            except sim_core.NoQuote:
                skips["quote_no_quote"] += 1
                continue
            bid_sz, ask_sz = float(depth_bid[a + j]), float(depth_ask[a + j])
            obi = features.l2_imbalance(bid_sz, ask_sz)
            if obi is None:
                skips["no_l2"] += 1
                continue
            flow_fast = _flow_at(trade_ts, cum_signed, cum_size, t, params.sl2_flow_fast_s)
            flow_slow = _flow_at(trade_ts, cum_signed, cum_size, t, params.sl2_flow_slow_s)
            sigma = features.trailing_sigma(index_tape, t)
            signal = features.student_t_l2_settle(
                spot=spot, strike=strike, seconds_left=float(secs),
                sigma=sigma, nu=nu, s=s, l2_obi=obi, flow_fast=flow_fast, flow_slow=flow_slow)
            if signal is None:
                skips["no_settlement_signal"] += 1
                continue
            p_up = signal.p_up
            quote = Quote(
                ticker=ticker, series=name, strike=strike, close_ts=close,
                yes_bid=sim_core.px(bid[j]), yes_ask=sim_core.px(ask[j]),
                last=sim_core.px((float(bid[j]) + float(ask[j])) / 2.0),
                updated_ts=float(ts[j]), received_ts=t,
                yes_bid_sz=Decimal(str(bid_sz)), yes_ask_sz=Decimal(str(ask_sz)))
            decision = decide(
                quote=quote, p_up=p_up, seconds_left=float(secs), spot_age=0.0,
                params=params, position=None, edge_hurdle=params.edge_hurdle,
                can_open=True, signal="student_t_l2_settle", now=t, spot=spot,
                l2_obi=obi, flow_fast=flow_fast, flow_slow=flow_slow)
            if decision.action != "entry":
                skips["no_entry"] += 1
                continue
            triggers.append({
                "lane": name, "ticker": ticker, "tid": tid, "ts": t, "close": close,
                "y": int(meta["y"]), "side": decision.side, "price": decision.price,
                "edge": float(decision.edge),
                "p_yes": p_up,
                "p_side": p_up if decision.side == "YES" else 1.0 - p_up,
                "obi": obi, "flow_fast": flow_fast,
                "flow_slow": flow_slow, "secs_left": float(secs),
            })
    return triggers, skips


def score_student_t_l2(since: float, until: float | None = None,
                       gate_filter: bool = False, gate_min_confidence: float = 0.0,
                       db_path: Path | None = None, regime: bool = False) -> dict:
    """Score the settlement program on windows closing in `[since, until)`.

    The probability this program carries is a *settlement* probability, so the
    calibration error is measured against the contract's own settlement outcome,
    not against a round trip: the contract is carried to expiry and pays $1 or
    $0.

    `gate_filter=True` keeps only the trades whose trigger carried a LayaGate
    verdict of `allow == True` -- the same opinion the live lane saw, read off
    the recorded `ticks.gate` rows. A trigger the gate had no opinion on is
    dropped, because the live path refuses an absent opinion when the filter is
    binding; keeping the two the same is the point. `gate_min_confidence` is the
    same floor `Params.layagate_min_confidence` sets live.

    It is optional because the gate is still a shadow in this build; the filter
    is how a promoted strategy would enforce it without a code edit.
    """
    params = Params()
    # The gate verdicts are only read when the filter is asked for: the scan is
    # over every recorded tick, and a score that does not use the gate has no
    # business paying for it.
    verdicts = {}
    if gate_filter:
        verdicts = _gate_verdicts(db_path or ROOT / "data/openkal-prod.db", since - 900.0,
                                  until if until is not None else float("inf"), gate_min_confidence)
    lanes = {name: Lane(name) for name in LANES}
    rows: list[dict] = []
    skips: Counter = Counter()
    windows: set[float] = set()
    pools_ok: dict[str, int] = {}
    events = 0
    gate_counts = {"allowed": 0, "denied": 0, "no_opinion": 0} if gate_filter else {"evaluated": False}
    for name, lane in lanes.items():
        closes = {float(m["close"]) for m in lane.meta.values()
                  if float(m["close"]) >= since
                  and (until is None or float(m["close"]) < until)}
        pools_ok[name] = len(closes)
        if not closes:
            continue
        index_tape = ([(float(t), float(v)) for t, v in zip(lane.index["ts"], lane.index["value"])]
                      if lane.index else [])
        triggers, lane_skips = _settlement_triggers(lane, name, closes, index_tape, params)
        for key, value in lane_skips.items():
            skips[key] += value
        carried = set()
        for trig in triggers:
            if trig["tid"] in carried:
                continue
            allow = None
            if gate_filter:
                allow = _gate_allow(verdicts, str(trig["ticker"]), float(trig["ts"]))
                gate_counts["allowed" if allow is True else
                            "denied" if allow is False else "no_opinion"] += 1
                if allow is not True:
                    skips["gate_refused"] += 1
                    continue
            events += 1
            res = simulate_settlement(lane, trig)
            if isinstance(res, sim_core.Skip):
                skips[res.reason] += 1
                continue
            carried.add(trig["tid"])
            windows.add(float(trig["close"]))
            rows.append({
                "lane": name, "ticker": trig["ticker"], "tid": trig["tid"],
                "ts": float(res.order_ts), "fill_ts": float(res.legs[0].ts),
                "exit_ts": float(res.exit_ts), "slice": "oos", "side": res.side,
                "entry_style": res.entry_style, "entry": str(res.entry),
                "exit": str(res.exit_value), "qty": res.qty, "pnl": str(res.pnl),
                "fee": str(res.fees), "reason": res.reason, "crossed": res.crossed,
                "requotes": res.requotes, "close": trig["close"], "y": trig["y"],
                "p_side_raw": trig["p_side"], "p_side": trig["p_side"],
                "p_yes": trig["p_yes"],
                "edge_c": trig["edge"], "obi": trig["obi"],
                "flow_fast": trig["flow_fast"], "flow_slow": trig["flow_slow"],
                "secs_left": trig["secs_left"], "gate_allow": allow,
            })

    kept = rows
    hours = ((max(windows) - min(windows)) / 3600.0) if len(windows) > 1 else 0.0
    summary = summarize(kept, hours) if kept else summarize([], 0.0)
    summary = _with_true_side_costs(summary, kept)
    summary["crossed_fills"] = sum(int(r["crossed"]) for r in kept)
    # This program's probability describes the contract's own settlement, so the
    # calibration error is measured against the settlement outcome -- which is
    # what `summarize`'s own settlement ECE does.
    settlement_ece = float(summary["ece"]) if kept else 1.0
    lanes_pos = sum(1 for v in (summary.get("by_lane") or {}).values()
                    if float(v.get("net") or 0) > 0)
    gate = gate_from_summary(summary, settlement_ece, lanes_pos, regime=regime)
    return {
        "generated": datetime.now(timezone.utc).isoformat(),
        "score_model": "openkal.sim_core",
        "regime_mode": regime,
        "gate_mode": gate["mode"],
        "strategy": "student_t_l2_settle",
        "gate_filter": gate_filter,
        "gate_min_confidence": gate_min_confidence,
        "gate_counts": gate_counts,
        "program": {
            "settlement_model": "Student-t per lane (nu, s)",
            "student_t_fit": {k: list(v) for k, v in features.STUDENT_T_FIT.items()},
            "tau_offset_s": features.STUDENT_T_TAU_OFFSET_S,
            "l2_imbalance": "(yes_bid_sz - yes_ask_sz) / (yes_bid_sz + yes_ask_sz)",
            "flow_windows_s": [params.sl2_flow_fast_s, params.sl2_flow_slow_s],
            "min_flow": params.sl2_min_flow,
            "min_obi": params.sl2_min_obi,
            "cost_band": [SL2_COST_LO, SL2_COST_HI],
            "hard_cap": SL2_CAP,
            "entry_lag_s": LAG_S,
            "exit": "settlement ($0 exit fee)",
            "grid_s": SL2_GRID_S,
            "simultaneous_positions_per_lane": 1,
        },
        "fill_rules": {
            "entry_lag_s": LAG_S,
            "exit_placement_lag_s": sim_core.LAG_S,
            "min_print_contracts": MIN_FILL,
            "no_ask": "1 - yes_bid",
            "spot_lookahead_s": 0.0,
        },
        "since_ts": since,
        "since_local": datetime.fromtimestamp(since).astimezone().isoformat() if since else "all",
        "until_ts": until,
        "until_local": (datetime.fromtimestamp(until).astimezone().isoformat()
                        if until else None),
        "windows_per_lane": pools_ok,
        "windows_scored": len(windows),
        "events": events,
        "skips": dict(sorted(skips.items())),
        "hurdle_c": float(params.sl2_min_edge),
        "trades": summary,
        "settlement_ece": settlement_ece,
        "lanes_profitable": lanes_pos,
        "gate": gate,
        "rows": kept,
    }


# ---------------------------------------------------------------------------
# cross_lane_iv_arb: alt implied vol dispersion against BTC, same tick
#
# The comparison is per window and per tick: BTC's iv_ratio is read from the last
# sample at or before the alt's trigger, inside the *same* 15-minute contract, so
# nothing in this scorer can see a later BTC quote. `strategy.decide` owns the
# band, the $0.62 cap and the edge hurdle; `sim_core` owns the fill (a taker entry
# arriving a full second late) and the exit (settlement, $0 fee).
# ---------------------------------------------------------------------------

XIV_IV_SPIKE_MULT = 1.35
XIV_SECS_LO, XIV_SECS_HI = 120.0, 720.0
XIV_COST_LO, XIV_COST_HI = 0.35, 0.60
XIV_CAP = float(HARD_MAX_ENTRY_PRICE)
# The thesis is about minutes of dispersion, not one ladder row, and every grid
# point costs a Student-t inversion plus a trailing-sigma read. 10s bounds that
# cost without changing what the rule measures.
XIV_GRID_S = 10.0
# A BTC reading older than this is a different market, not the same tick.
XIV_MAX_BTC_AGE_S = 12.0
XIV_MAX_BOOK_AGE_S = features.STUDENT_T_MAX_DEPTH_AGE_S
XIV_GRID = np.arange(XIV_SECS_HI, XIV_SECS_LO - 1e-9, -XIV_GRID_S)


def _index_tape(lane: Lane) -> list[tuple[float, float]]:
    return ([(float(t), float(v)) for t, v in zip(lane.index["ts"], lane.index["value"])]
            if lane.index else [])


def _slice_windows(lane: Lane, since: float, until: float | None) -> list[tuple]:
    """`(tid, ticker, meta)` for windows closing inside `[since, until)`."""
    return [(tid, ticker, meta) for tid, ticker, meta, _a, _b in A._segments(lane)
            if float(meta["close"]) >= since
            and (until is None or float(meta["close"]) < until)]


def _iv_rows(lane: Lane, name: str, windows: list[tuple],
             index_tape: list[tuple[float, float]], nu: float, s: float,
             params: Params) -> list[dict]:
    """One implied-vol reading per usable grid tick, oldest first.

    The mid is the book's own, the sigma is `features.trailing_sigma` on the lane's
    settlement index (the same estimator the live engine prices with), and the
    inversion is `features.implied_vol_from_mid`. Rows whose book is crossed, empty
    or stale past the depth allowance are dropped rather than carried forward: an
    IV read off a ladder that has already moved is not a reading of this tick.
    """
    out: list[dict] = []
    for tid, ticker, meta in windows:
        close = float(meta["close"])
        strike = float(meta["strike"])
        a, b = int(lane.book_bounds[tid]), int(lane.book_bounds[tid + 1])
        ts = lane.ts[a:b]
        bid = lane.bid[a:b]
        ask = lane.ask[a:b]
        if not len(ts):
            continue
        spots = A.index_at_past(lane, close - XIV_GRID, tol=params.max_spot_age_s)
        for k, secs in enumerate(XIV_GRID):
            t = close - float(secs)
            j = int(np.searchsorted(ts, t, "right")) - 1
            if j < 0 or t - float(ts[j]) > XIV_MAX_BOOK_AGE_S:
                continue
            lo, hi = float(bid[j]), float(ask[j])
            try:
                sim_core.check_book(lo, hi)
            except (sim_core.CrossedBook, sim_core.NoQuote):
                continue
            mid = (lo + hi) / 2.0
            # Outside this band neither side can be inside the $0.35-$0.60 cost
            # band, so there is nothing to invert for.
            if not (0.30 <= mid <= 0.70):
                continue
            spot = float(spots[k])
            if not (spot > 0):
                continue
            sigma = features.trailing_sigma(index_tape, t)
            if sigma <= 0:
                continue
            iv = features.implied_vol_from_mid(mid, spot, strike, float(secs), nu, s, sigma)
            if iv is None or iv.iv_ratio <= 0:
                continue
            out.append({"lane": name, "ticker": ticker, "tid": tid, "close": close,
                        "y": int(meta["y"]), "strike": strike, "t": t,
                        "secs": float(secs), "bid": lo, "ask": hi, "spot": spot,
                        "sigma": sigma, "iv": iv})
    out.sort(key=lambda r: r["t"])
    return out


def _block_reason(text: str) -> str:
    """Which of the dispersion entry's own gates refused a signal.

    A research scorer that reports only "no entry" hides the difference between a
    rule whose edge is too thin and one whose band is wrong for the tape. The
    categories are the checks in `strategy._cross_lane_iv_entry`, in its order.
    """
    if "dispersion window" in text:
        return "window"
    if "outside" in text and "band" not in text:
        return "band"
    if "cap" in text:
        return "cap"
    if "wide for dispersion" in text:
        return "spread"
    if "under" in text:
        return "hurdle"
    return "other"


def _btc_ratio_tape(btc: Lane, windows: list[tuple], params: Params,
                    ) -> dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Window close -> sorted (ts, iv, iv_ratio) arrays for the BTC reference lane."""
    nu, s = features.STUDENT_T_FIT["KXBTC15M"]
    rows = _iv_rows(btc, "KXBTC15M", windows, _index_tape(btc), nu, s, params)
    grouped: dict[float, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["close"], []).append(row)
    out = {}
    for close, items in grouped.items():
        out[close] = (np.array([r["t"] for r in items], dtype=np.float64),
                      np.array([r["iv"].iv for r in items], dtype=np.float64),
                      np.array([r["iv"].iv_ratio for r in items], dtype=np.float64))
    return out


def score_cross_lane_iv_arb(since: float, until: float | None = None,
                            regime: bool = False) -> dict:
    """Score the dispersion program on windows closing in `[since, until)`.

    BTC is the reference, never a trade: the three alt lanes are scored, and each
    trigger is compared against the BTC ratio of the same window at the same tick.
    One position per lane per window, held to settlement, so the trade floor the
    gate applies is a floor on real, non-overlapping bets.
    """
    params = Params()
    lanes = {name: Lane(name) for name in LANES}
    btc = lanes["KXBTC15M"]
    btc_windows = _slice_windows(btc, since, until)
    btc_ratios = _btc_ratio_tape(btc, btc_windows, params)
    pools_ok = {"KXBTC15M": len(btc_windows)}
    rows: list[dict] = []
    skips: Counter = Counter()
    windows: set[float] = set()
    events = 0
    for name in ("KXETH15M", "KXSOL15M", "KXXRP15M"):
        lane = lanes[name]
        lane_windows = _slice_windows(lane, since, until)
        pools_ok[name] = len(lane_windows)
        if not lane_windows:
            continue
        nu, s = features.STUDENT_T_FIT[name]
        samples = _iv_rows(lane, name, lane_windows, _index_tape(lane), nu, s, params)
        carried: set[int] = set()
        for row in samples:
            if row["tid"] in carried:
                continue
            ref = btc_ratios.get(row["close"])
            if ref is None:
                skips["no_btc_window"] += 1
                continue
            bts, biv, bratio = ref
            i = int(np.searchsorted(bts, row["t"], "right")) - 1
            if i < 0 or row["t"] - float(bts[i]) > XIV_MAX_BTC_AGE_S:
                skips["no_btc_tick"] += 1
                continue
            btc_iv = features.ImpliedVol(iv=float(biv[i]), iv_ratio=float(bratio[i]),
                                        mid=0.0, p_model=0.0)
            signal = features.cross_lane_iv_arb(
                alt_iv=row["iv"], btc_iv=btc_iv, spot=row["spot"], strike=row["strike"],
                seconds_left=row["secs"], sigma=row["sigma"], nu=nu, s=s,
                iv_spike_mult=params.xiv_iv_spike_mult,
                min_seconds_left=params.xiv_min_seconds_left,
                max_seconds_left=params.xiv_max_seconds_left)
            if signal is None:
                skips["no_dispersion"] += 1
                continue
            quote = Quote(
                ticker=str(row["ticker"]), series=name, strike=float(row["strike"]),
                close_ts=float(row["close"]), yes_bid=sim_core.px(row["bid"]),
                yes_ask=sim_core.px(row["ask"]),
                last=sim_core.px((row["bid"] + row["ask"]) / 2.0),
                updated_ts=float(row["t"]), received_ts=float(row["t"]))
            decision = decide(
                quote=quote, p_up=signal.p_up, seconds_left=row["secs"], spot_age=0.0,
                params=params, position=None, edge_hurdle=params.edge_hurdle,
                can_open=True, signal="cross_lane_iv_arb", now=row["t"],
                spot=row["spot"], iv_alt_ratio=row["iv"].iv_ratio,
                iv_btc_ratio=float(bratio[i]))
            if decision.action != "entry":
                skips[f"no_entry_{_block_reason(decision.reason)}"] += 1
                continue
            events += 1
            trig = {"lane": name, "ticker": row["ticker"], "tid": row["tid"],
                    "ts": row["t"], "close": row["close"], "y": row["y"],
                    "side": decision.side, "price": decision.price,
                    "edge": float(decision.edge), "p_yes": signal.p_up,
                    "p_side": (signal.p_up if decision.side == "YES"
                               else 1.0 - signal.p_up),
                    "secs_left": row["secs"]}
            res = simulate_settlement(lane, trig)
            if isinstance(res, sim_core.Skip):
                skips[res.reason] += 1
                continue
            carried.add(row["tid"])
            windows.add(float(row["close"]))
            rows.append({
                "lane": name, "ticker": row["ticker"], "tid": row["tid"],
                "ts": float(res.order_ts), "fill_ts": float(res.legs[0].ts),
                "exit_ts": float(res.exit_ts), "slice": "oos", "side": res.side,
                "entry_style": res.entry_style, "entry": str(res.entry),
                "exit": str(res.exit_value), "qty": res.qty, "pnl": str(res.pnl),
                "fee": str(res.fees), "reason": res.reason, "crossed": res.crossed,
                "requotes": res.requotes, "close": row["close"], "y": row["y"],
                "p_side_raw": trig["p_side"], "p_side": trig["p_side"],
                "p_yes": trig["p_yes"], "edge_c": trig["edge"],
                "alt_iv_ratio": row["iv"].iv_ratio, "btc_iv_ratio": float(bratio[i]),
                "alt_iv": row["iv"].iv, "secs_left": row["secs"],
            })

    hours = ((max(windows) - min(windows)) / 3600.0) if len(windows) > 1 else 0.0
    summary = summarize(rows, hours) if rows else summarize([], 0.0)
    summary = _with_true_side_costs(summary, rows)
    summary["crossed_fills"] = sum(int(r["crossed"]) for r in rows)
    settlement_ece = float(summary["ece"]) if rows else 1.0
    lanes_pos = sum(1 for v in (summary.get("by_lane") or {}).values()
                    if float(v.get("net") or 0) > 0)
    gate = gate_from_summary(summary, settlement_ece, lanes_pos, regime=regime)
    return {
        "generated": datetime.now(timezone.utc).isoformat(),
        "score_model": "openkal.sim_core",
        "regime_mode": regime,
        "gate_mode": gate["mode"],
        "strategy": "cross_lane_iv_arb",
        "program": {
            "reference_lane": "KXBTC15M",
            "compared_lanes": ["KXETH15M", "KXSOL15M", "KXXRP15M"],
            "iv_spike_mult": params.xiv_iv_spike_mult,
            "seconds_left": [params.xiv_min_seconds_left, params.xiv_max_seconds_left],
            "cost_band": [XIV_COST_LO, XIV_COST_HI],
            "hard_cap": XIV_CAP,
            "entry_lag_s": LAG_S,
            "exit": "settlement ($0 exit fee)",
            "grid_s": XIV_GRID_S,
            "btc_max_age_s": XIV_MAX_BTC_AGE_S,
            "simultaneous_positions_per_lane": 1,
            "student_t_fit": {k: list(v) for k, v in features.STUDENT_T_FIT.items()},
        },
        "fill_rules": {
            "entry_lag_s": LAG_S,
            "exit_placement_lag_s": sim_core.LAG_S,
            "min_print_contracts": MIN_FILL,
            "no_ask": "1 - yes_bid",
            "spot_lookahead_s": 0.0,
        },
        "since_ts": since,
        "since_local": datetime.fromtimestamp(since).astimezone().isoformat() if since else "all",
        "until_ts": until,
        "until_local": (datetime.fromtimestamp(until).astimezone().isoformat()
                        if until else None),
        "windows_per_lane": pools_ok,
        "windows_with_btc_reference": len(btc_ratios),
        "windows_scored": len(windows),
        "events": events,
        "skips": dict(sorted(skips.items())),
        "hurdle_c": float(max(params.xiv_min_edge, params.edge_hurdle)),
        "trades": summary,
        "settlement_ece": settlement_ece,
        "lanes_profitable": lanes_pos,
        "gate": gate,
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    # None, not the frozen start: regime mode's default is a rolling window that
    # only exists at run time, and a non-regime run still resolves to the same
    # `FORWARD_OOS_START_TS` it always used.
    ap.add_argument("--since", type=float, default=None)
    # Drill only: bound the replay strictly below the forward start. The noon run
    # never passes it, so its slice is unchanged.
    ap.add_argument("--until", type=float, default=None)
    ap.add_argument("--out", default=str(OUT))
    # Which program to score. The default is the frozen F2 rule; the settlement
    # program is scored by the same canonical simulator on the same slice.
    ap.add_argument("--strategy",
                    choices=("f2", "student_t_l2_settle", "cross_lane_iv_arb"),
                    default="f2")
    # Optional LayaGate gating for the settlement program: keep only the trades
    # whose trigger carried an `allow == True` verdict in the recorded ticks.
    ap.add_argument("--gate-filter", action="store_true")
    ap.add_argument("--gate-min-confidence", type=float, default=0.0)
    # The fast-decay gate: a rolling 48h slice scored against a bar a 2-3 day
    # edge can reach. Same rule, same simulator, same $0.62 cap and 50-contract
    # print-through -- only the window and the counts move.
    ap.add_argument("--regime-mode", action="store_true",
                    help=f"score the rolling {REGIME_WINDOW_HOURS}h slice with the "
                         "decay gate instead of the frozen all-history gate")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    since = args.since
    if since is None:
        since = (time.time() - REGIME_WINDOW_HOURS * 3600 if args.regime_mode
                 else float(FORWARD_OOS_START_TS))
    if args.strategy == "student_t_l2_settle":
        if args.out == str(OUT):
            args.out = str(OUT_STUDENT_T)
        report = score_student_t_l2(since, args.until, gate_filter=args.gate_filter,
                                   gate_min_confidence=args.gate_min_confidence,
                                   regime=args.regime_mode)
    elif args.strategy == "cross_lane_iv_arb":
        if args.out == str(OUT):
            args.out = str(OUT_CROSS_LANE)
        report = score_cross_lane_iv_arb(since, args.until, regime=args.regime_mode)
    else:
        report = score(since, args.until, regime=args.regime_mode)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str))
    payload = {k: v for k, v in report.items() if k != "rows"}
    print(json.dumps(payload, indent=1, default=str))
    print(f"\nwrote {out}")
    return 0 if report["gate"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
