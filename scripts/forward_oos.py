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
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openkal import features, sim_core  # noqa: E402
from openkal.strategy import Params, exit_cost, taker_fee_per_contract  # noqa: E402
from scripts import alpha_search as A  # noqa: E402
from scripts.alpha_discover import Lane  # noqa: E402
from scripts.bounded_sweep import obi_pool, pick, summarize  # noqa: E402

LANES = ("KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M")
FORWARD_OOS_START_TS = 1794909600          # 2026-11-17 05:00:00 EST
OUT = ROOT / "data" / "forward-oos.json"

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


def gate_from_summary(summary: dict, payoff_ece: float,
                      lanes_profitable: int | None = None) -> dict:
    """The frozen gate, as a pure function of the scored numbers.

    Every condition is checked on its own, in the order a reader would: a rule
    that fails one of them is refused, whatever the others say. There is no
    aggregate score and no fallback -- an average would let a spectacular PF
    pay for a broken calibration or a crossed fill, which is exactly the trade
    the gate exists to forbid. `scripts/day_supervisor.py` calls this again on
    the artifact before it promotes anything, so a hand-edited or stale result
    file cannot arm the engine on numbers nobody checked.
    """
    if lanes_profitable is None:
        lanes_profitable = sum(1 for v in (summary.get("by_lane") or {}).values()
                               if float(v.get("net") or 0) > 0)
    reasons: list[str] = []
    if int(summary.get("trades") or 0) < MIN_TRADES:
        reasons.append(f"{int(summary.get('trades') or 0)} forward trades < {MIN_TRADES}")
    if float(summary.get("net_pnl") or 0) <= 0:
        reasons.append(f"forward net ${float(summary.get('net_pnl') or 0):.2f} <= 0")
    pf = float(summary.get("pf") or 0)
    # Exactly the seven conditions the plan names. No eighth condition is added
    # here and no condition is combined with another: a PF of 99 on two losers is
    # still PF 99 for this gate, because the trade floor is the protection.
    if pf < REQUIRED_PF:
        reasons.append(f"forward PF {pf:.3f} < {REQUIRED_PF}")
    if float(payoff_ece) > REQUIRED_ECE:
        reasons.append(f"payoff ECE {float(payoff_ece):.4f} > {REQUIRED_ECE}")
    if int(summary.get("crossed_fills") or 0) != 0:
        reasons.append(f"{int(summary.get('crossed_fills') or 0)} crossed fills")
    if float(summary.get("max_side_cost") or 0) > CAP + 1e-9:
        reasons.append(f"max cost {float(summary.get('max_side_cost') or 0):.3f} > {CAP}")
    if int(lanes_profitable) < 2:
        reasons.append(f"{int(lanes_profitable)} profitable lane(s) < 2")
    return {"pass": not reasons, "reasons": reasons, "lanes_profitable": int(lanes_profitable)}


def score(since: float, until: float | None = None) -> dict:
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
    # A crossed arrival is a counted skip in sim_core, not a fill, so the gate's
    # crossed-fill condition stays a real check on the model's output.
    summary["crossed_fills"] = sum(int(r["crossed"]) for r in kept)
    payoff_ece = round(features.expected_calibration_error(
        [(float(r["p_side"]), 1 if float(r["pnl"]) > 0 else 0) for r in kept]), 4)
    lanes_pos = sum(1 for v in (summary.get("by_lane") or {}).values()
                    if float(v.get("net") or 0) > 0)
    gate = gate_from_summary(summary, payoff_ece, lanes_pos)
    return {
        "generated": datetime.now(timezone.utc).isoformat(),
        "score_model": "openkal.sim_core",
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=float, default=float(FORWARD_OOS_START_TS))
    # Drill only: bound the replay strictly below the forward start. The noon run
    # never passes it, so its slice is unchanged.
    ap.add_argument("--until", type=float, default=None)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    report = score(args.since, args.until)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str))
    payload = {k: v for k, v in report.items() if k != "rows"}
    print(json.dumps(payload, indent=1, default=str))
    print(f"\nwrote {out}")
    return 0 if report["gate"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
