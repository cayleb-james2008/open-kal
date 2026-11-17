#!/usr/bin/env python3
"""Score the frozen F2 rule ONCE on the forward out-of-sample slice.

Everything here is frozen: the detector thresholds, the cost band, the 1s
execution lag, the +5c maker take-profit, the 7c two-second-confirmed stop, the
90s barrier, the payoff calibrator, and the desk's entry hurdle. This script
does not tune, and it must not be given a threshold to try.

The forward slice is every settled window whose close is at or after
`FORWARD_OOS_START_TS`. It was recorded while nothing was armed, so it is the
first slice in this project that has not been read by a fit of any kind.

    .venv/bin/python scripts/forward_oos.py                      # default slice
    .venv/bin/python scripts/forward_oos.py --since 0 --out data/f2-repro.json

With `--since 0` it re-scores the whole cache, which is how the harness is
validated: the totals must reproduce scripts/f2_structural.py (82 trades,
+$81.19 net, payoff-ECE 0.0764 on the in-sample half and 0.09 on the read OOS
half). A scorer that disagrees with the frozen study cannot certify anything.
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

from openkal import features  # noqa: E402
from openkal.strategy import Params, exit_cost, taker_fee_per_contract  # noqa: E402
from scripts import alpha_search as A  # noqa: E402
from scripts.alpha_discover import Lane  # noqa: E402
from scripts.bounded_sweep import _sim, obi_pool, pick, summarize  # noqa: E402

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


def limit_price(side_yes: bool, bid: float, ask: float) -> float:
    """The taker order is pinned to the ask of the side held, in YES terms."""
    return ask if side_yes else 1.0 - bid


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


def events_for(lane: Lane, pool: dict, closes: set[float]):
    """The frozen rule's events on the selected windows, priced at arrival (T+1s)."""
    mask, ly = frozen_mask(pool)
    if closes:
        mask = mask & np.isin(pool["close"], np.fromiter(closes, dtype=np.float64))
    idx = pick(pool, mask, cooldown=COOLDOWN, per_window=PER_WINDOW)
    out = []
    for i in idx:
        trigger = float(pool["ts"][i])
        close = float(pool["close"][i])
        arrival = trigger + LAG_S
        if close - arrival < 5.0:
            continue
        quote = lane.quote_at(int(pool["tid"][i]), arrival, valid_only=True, lookback=2.0)
        if quote is None:
            continue
        b, a = quote
        if not (0.0 < b < a < 1.0):
            continue
        long_yes = bool(ly[i])
        limit = limit_price(long_yes, b, a)
        cost = limit if long_yes else 1.0 - limit
        if not (COST_LO - 1e-9 <= cost <= CAP + 1e-9):
            continue
        out.append((A.Event(
            "F2", "F2_taker_1s", tid=int(pool["tid"][i]),
            ticker=str(pool["ticker"][i]), ts=arrival, close=close,
            # The shared fill model only simulates a slice it knows: "is" or "oos".
            # The forward slice *is* an out-of-sample slice, just a later one than
            # the 60/40 split, so it is scored as one rather than given a third
            # label the engine's fill model would silently drop.
            y=int(pool["y"][i]), slice="oos",
            side="YES" if long_yes else "NO", entry_style="taker", limit=limit,
            p_side=float(pool["p_up"][i])), i))
    return out


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
    crossed_total = 0
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
        raw = obi_pool(lane, sig, t_lo=T_LO, t_hi=T_HI, delta_bps_floor=DELTA_BPS_MIN)
        if not raw.get("tid"):
            continue
        pool = {k: (np.concatenate(v) if k != "lane" else np.concatenate(v))
                for k, v in raw.items()}
        if "lane" not in pool:
            pool["lane"] = np.array([name] * len(pool["tid"]), dtype=object)
        pool["ticker"] = np.array([lane.tickers[i] for i in pool["tid"]], dtype=object)
        evs = events_for(lane, pool, closes)
        events += len(evs)
        sim_rows, crossed = _sim(lane, [e for e, _ in evs], A.SimParams(
            tp_c=TP_C, barrier_s=BARRIER_S, stop_c=STOP_C,
            stop_confirm_s=STOP_CONFIRM_S, exit_mode="maker_tp",
            max_entry_price=CAP, min_fill_qty=MIN_FILL, max_qty=MAX_QTY))
        # Crossed books are counted per lane, not smeared over every row: the gate
        # needs the true count, and a zero must mean zero.
        crossed_total += crossed
        by_key = {(e.ticker, round(float(e.ts), 6)): (i, e.close, int(e.y))
                  for e, i in evs}
        for r in sim_rows:
            hit = by_key.get((r["ticker"], round(float(r["ts"]), 6)))
            if hit is None:
                # The sim can re-price an event (a stop, a barrier); without its
                # trigger row there is no frozen feature vector to calibrate.
                continue
            i, close, outcome = hit
            r["lane"] = name
            r["close"] = close
            r["y"] = outcome
            r["p_side_raw"] = float(r["p_side"])
            r["p_side"] = cal.apply(feature_row(pool, i))
            r["crossed"] = 0
            rows.append(r)
            windows.add(float(close))

    kept: list[dict] = []
    for r in rows:
        lane = lanes[str(r["lane"])]
        quote = lane.quote_at(int(r["tid"]), float(r["ts"]), valid_only=True, lookback=2.0)
        if quote is None:
            r["decide_takes"] = False
            continue
        bid, ask = quote
        side_ask = ask if r["side"] == "YES" else 1.0 - bid
        gross = exit_cost(Decimal(str(side_ask)), params)
        edge = Decimal(str(r["p_side"])) - Decimal(str(side_ask))
        net = edge - gross - taker_fee_per_contract(Decimal(str(side_ask)))
        r["decide_edge_c"] = float(net)
        r["decide_takes"] = bool(net >= hurdle)
        if r["decide_takes"]:
            kept.append(r)

    hours = ((max(windows) - min(windows)) / 3600.0) if len(windows) > 1 else 0.0
    summary = summarize(kept, hours) if kept else summarize([], 0.0)
    # The shared summary counts per-row flags; crossed books come from the fill
    # model, which reports them per lane.
    summary["crossed_fills"] = crossed_total
    payoff_ece = round(features.expected_calibration_error(
        [(float(r["p_side"]), 1 if float(r["pnl"]) > 0 else 0) for r in kept]), 4)
    lanes_pos = sum(1 for v in (summary.get("by_lane") or {}).values()
                    if float(v.get("net") or 0) > 0)
    gate = gate_from_summary(summary, payoff_ece, lanes_pos)
    return {
        "generated": datetime.now(timezone.utc).isoformat(),
        "since_ts": since,
        "since_local": datetime.fromtimestamp(since).astimezone().isoformat() if since else "all",
        "until_ts": until,
        "until_local": (datetime.fromtimestamp(until).astimezone().isoformat()
                        if until else None),
        "windows_per_lane": pools_ok,
        "windows_scored": len(windows),
        "events": events,
        "calibration": cal.to_json(),
        "hurdle_c": float(hurdle),
        "all_trades": summarize(rows, hours) if rows else summarize([], 0.0),
        "trades": summary,
        "payoff_ece": payoff_ece,
        "lanes_profitable": lanes_pos,
        "gate": gate,
        "rows": [{k: (str(v) if isinstance(v, Decimal) else v) for k, v in r.items()}
                 for r in kept],
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
