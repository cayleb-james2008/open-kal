#!/usr/bin/env python3
"""Frozen-parameter structural test of the F2 rule, with the mandatory 1s lag.

The F2 rule is frozen: `z >= 0.3`, printed flow `>= 0.1` in the trade's direction,
touch revisions `>= 0.0`, spot-to-strike delta `>= 3bp`, side cost in [0.25, 0.60],
a +5c maker take-profit, a 7c two-second-confirmed stop, and a 50-contract
print-through on every passive leg. Nothing here tunes a threshold.

Two structural variants are evaluated on the same 130-window 60/40 split:

  (a) F2_taker_1s    -- cross the ask one second after the trigger. The probability
                        the rule carries is calibrated on the in-sample trades to
                        predict the trade's own payoff event (net > 0), and the
                        calibrated probability is then put through the desk's own
                        entry test in `openkal/strategy.py:decide()`.
  (b) F2_maker_50pt  -- rest at the bid one second after the trigger and wait for
                        50 contracts to print through it. No entry fee.

The point of the exercise is diagnostic: the rule as swept filled at the price it
had already seen, and its reported probability described the market's settlement
rather than the trade's payoff. Both are fixed here, and the numbers are what they
are.

    .venv/bin/python scripts/f2_structural.py
"""
from __future__ import annotations

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
OUT = ROOT / "data" / "f2-structural.json"
TRADES_OUT = ROOT / "data" / "f2-structural-trades.jsonl"

# --- the frozen rule -------------------------------------------------------
Z_MIN = 0.30
FLOW_MIN = 0.10
REV_MIN = 0.0
DELTA_BPS_MIN = 3.0
COST_LO, COST_HI = 0.25, 0.60
TP_C = 0.05
STOP_C = 0.07
STOP_CONFIRM_S = 2.0
BARRIER_S = 90.0
CAP = 0.62
MIN_FILL = 50.0
MAX_QTY = 50
LAG_S = 1.0
T_LO, T_HI = 180.0, 780.0
COOLDOWN, PER_WINDOW = 90.0, 3


def side_cost(side_yes: bool, bid: float, ask: float, style: str) -> float:
    """What the side held actually costs, per contract.

    A YES long costs the YES ask (taker) or the YES bid (maker). A NO long costs
    `1 - bid` (taker: the NO ask) or `ask` (maker: the NO bid is `1 - ask`).
    """
    if style == "taker":
        return ask if side_yes else 1.0 - bid
    return bid if side_yes else ask


def limit_price(side_yes: bool, bid: float, ask: float, style: str) -> float:
    """The price the order is sent at, in YES terms."""
    if style == "taker":
        return ask if side_yes else 1.0 - bid
    return bid if side_yes else 1.0 - ask


# --- the trade-outcome probability model -----------------------------------
def fit_logistic(x: np.ndarray, y: np.ndarray, *, ridge: float = 1.0,
                 iters: int = 800) -> tuple[np.ndarray, np.ndarray]:
    """Ridge logistic regression, plain gradient descent, no tuning.

    `x` is standardised on the fit; the intercept is not penalised. The fit is
    deterministic: same rows in, same coefficients out.
    """
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd < 1e-12] = 1.0
    xs = (x - mu) / sd
    n, d = xs.shape
    w = np.zeros(d + 1)
    lr = 0.3
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(xs @ w[:d] + w[d])))
        grad = np.empty(d + 1)
        grad[:d] = xs.T @ (p - y) / n + ridge * w[:d] / n
        grad[d] = (p - y).mean()
        w -= lr * grad
    return w, np.concatenate((mu, sd))


def apply_logistic(x: np.ndarray, w: np.ndarray, norm: np.ndarray) -> np.ndarray:
    d = len(w) - 1
    xs = (x - norm[:d]) / norm[d:]
    return 1.0 / (1.0 + np.exp(-(xs @ w[:d] + w[d])))


# --- the pools and the events ----------------------------------------------
def build_pools(lanes: dict[str, Lane]) -> dict[str, dict]:
    pools: dict[str, dict] = {}
    for name, lane in lanes.items():
        sig = A.trailing_sigma(lane)
        raw = obi_pool(lane, sig, t_lo=T_LO, t_hi=T_HI, delta_bps_floor=3.0)
        p = {k: np.concatenate(v) for k, v in raw.items() if k != "lane"}
        p["lane"] = np.concatenate(raw["lane"])
        p["ticker"] = np.array([lane.tickers[i] for i in p["tid"]], dtype=object)
        p["slice"] = np.array([lane.slice_of(float(c)) for c in p["close"]], dtype=object)
        pools[name] = p
    return pools


def frozen_mask(pool: dict, style: str) -> tuple[np.ndarray, np.ndarray]:
    z = pool["z"]
    d = np.sign(z)
    ly = z > 0
    cost = np.where(ly, pool["ask"] if style == "taker" else pool["bid"],
                    1.0 - pool["bid"] if style == "taker" else pool["ask"])
    mask = (
        (np.abs(z) >= Z_MIN)
        & (pool["flow"] * d >= FLOW_MIN)
        & (pool["rev"] * d >= REV_MIN)
        & (pool["delta"] >= DELTA_BPS_MIN * 1e-4 * np.maximum(pool["spot"], 1.0))
        & (cost >= COST_LO - 1e-9)
        & (cost <= COST_HI + 1e-9)
    )
    return mask, ly


def events_for(lane: Lane, pool: dict, style: str, slice_name: str):
    """The frozen rule's events, priced when the order would arrive (T + 1s)."""
    mask, ly = frozen_mask(pool, style)
    mask = mask & (pool["slice"] == slice_name)
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
        limit = limit_price(long_yes, b, a, style)
        cost = limit if long_yes else 1.0 - limit
        if not (COST_LO - 1e-9 <= cost <= CAP + 1e-9):
            continue
        out.append((A.Event(
            "F2", f"F2_{style}_1s", tid=int(pool["tid"][i]),
            ticker=str(pool["ticker"][i]), ts=arrival, close=close,
            y=int(pool["y"][i]), slice=slice_name,
            side="YES" if long_yes else "NO", entry_style=style, limit=limit,
            p_side=float(pool["p_up"][i])), i))
    return out


def feature_row(pool: dict, i: int) -> list[float]:
    """The features the rule already uses, signed by the trade's direction."""
    z = float(pool["z"][i])
    d = 1.0 if z > 0 else -1.0
    delta_bps = float(pool["delta"][i]) / max(float(pool["spot"][i]), 1.0) * 1e4
    return [z, float(pool["flow"][i]) * d, float(pool["rev"][i]) * d, delta_bps]


def decide_filter(lane: Lane, rows: list[dict], params: Params) -> list[dict]:
    """The desk's own entry test, on the calibrated probability.

    `openkal/strategy.py:decide()` measures the edge against the ask even for a
    passive entry, charges the expected exit cost, and for a taker charges the
    venue's fee on top. A trade the desk would not have taken is dropped here.
    """
    hurdle = max(Decimal(str(params.edge_hurdle)), Decimal(str(params.min_net_edge)))
    kept: list[dict] = []
    for r in rows:
        quote = lane.quote_at(int(r["tid"]), float(r["ts"]), valid_only=True, lookback=2.0)
        if quote is None:
            r["decide_takes"] = False
            continue
        bid, ask = quote
        side_ask = ask if r["side"] == "YES" else 1.0 - bid
        entry = Decimal(str(r["entry"]))
        p_side = Decimal(str(r["p_side"]))
        gross = exit_cost(Decimal(str(side_ask)), params)
        edge = p_side - Decimal(str(side_ask))
        if r["entry_style"] == "taker":
            cost = gross + taker_fee_per_contract(Decimal(str(side_ask)))
            net = edge - cost
        else:
            net = edge - gross
        r["decide_edge_c"] = float(net)
        r["decide_hurdle_c"] = float(hurdle)
        r["decide_takes"] = bool(net >= hurdle)
        if r["decide_takes"]:
            kept.append(r)
    return kept


def main() -> int:
    lanes = {name: Lane(name) for name in LANES}
    pools = build_pools(lanes)
    any_lane = lanes[LANES[0]]
    split = {
        "rule": "chronological 60/40 by window close, the sweep's own per-lane split",
        "windows_per_lane": len(any_lane.is_closes) + len(any_lane.oos_closes),
        "is_windows_per_lane": len(any_lane.is_closes),
        "oos_windows_per_lane": len(any_lane.oos_closes),
        "pool_rows": {n: int(len(p["tid"])) for n, p in pools.items()},
    }
    print(json.dumps(split, indent=1))

    params = Params()
    report: dict = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "frozen_rule": {"z_min": Z_MIN, "flow_min": FLOW_MIN, "rev_min": REV_MIN,
                        "delta_bps_min": DELTA_BPS_MIN, "cost_band": [COST_LO, COST_HI],
                        "tp_c": TP_C, "stop_c": STOP_C, "stop_confirm_s": STOP_CONFIRM_S,
                        "barrier_s": BARRIER_S, "cap": CAP, "min_fill": MIN_FILL,
                        "max_qty": MAX_QTY, "lag_s": LAG_S},
        "split": split,
        "variants": {},
    }
    dump = TRADES_OUT.open("w")

    for style in ("taker", "maker"):
        rows_by_slice: dict[str, list[dict]] = {"is": [], "oos": []}
        events_by_slice: dict[str, int] = {"is": 0, "oos": 0}
        feats: dict[str, list[list[float]]] = {"is": [], "oos": []}
        for slice_name in ("is", "oos"):
            for name, lane in lanes.items():
                pool = pools[name]
                evs = events_for(lane, pool, style, slice_name)
                events_by_slice[slice_name] += len(evs)
                sim_rows, _ = _sim(lane, [e for e, _ in evs], A.SimParams(
                    tp_c=TP_C, barrier_s=BARRIER_S, stop_c=STOP_C,
                    stop_confirm_s=STOP_CONFIRM_S, exit_mode="maker_tp",
                    max_entry_price=CAP, min_fill_qty=MIN_FILL, max_qty=MAX_QTY))
                meta_y = {t: int(m["y"]) for t, m in lane.meta.items()}
                by_key = {(e.ticker, round(float(e.ts), 6)): i for e, i in evs}
                for r in sim_rows:
                    r["lane"] = name
                    r["y"] = meta_y.get(r["ticker"], 0)
                    rows_by_slice[slice_name].append(r)
                    i = by_key.get((r["ticker"], round(float(r["ts"]), 6)))
                    if i is not None:
                        feats[slice_name].append(feature_row(pool, i))
        n_is = len(rows_by_slice["is"])
        n_oos = len(rows_by_slice["oos"])
        x_is = np.array(feats["is"], dtype=np.float64) if feats["is"] else np.zeros((0, 4))
        x_oos = np.array(feats["oos"], dtype=np.float64) if feats["oos"] else np.zeros((0, 4))
        if len(x_is) != n_is or len(x_oos) != n_oos:
            print(f"  note: {style}: feature rows {len(x_is)}/{n_is} IS and "
                  f"{len(x_oos)}/{n_oos} OOS matched their trades")

        def payoff(rows: list[dict]) -> np.ndarray:
            return np.array([1.0 if float(r["pnl"]) > 0 else 0.0 for r in rows])

        cal: dict = {"fitted": False}
        if len(x_is) >= 20 and len(x_oos) == n_oos and len(x_is) == n_is:
            w, norm = fit_logistic(x_is, payoff(rows_by_slice["is"]))
            cal = {"fitted": True, "weights": w.tolist(), "normaliser": norm.tolist(),
                   "rows": int(len(x_is)),
                   "target": "P(trade pnl > 0)"}
            for rows, x in ((rows_by_slice["is"], x_is), (rows_by_slice["oos"], x_oos)):
                for r, xi in zip(rows, x):
                    r["p_side_raw"] = float(r["p_side"])
                    r["p_side"] = float(apply_logistic(xi[None, :], w, norm)[0])

        summary: dict = {}
        for slice_name in ("is", "oos"):
            rows = rows_by_slice[slice_name]
            s = summarize(rows, 1.0)
            s["events"] = events_by_slice[slice_name]
            s["fill_rate"] = round(len(rows) / max(1, events_by_slice[slice_name]), 4)
            s["payoff_event_ece"] = round(features.expected_calibration_error(
                [(float(r["p_side"]), 1 if float(r["pnl"]) > 0 else 0) for r in rows]), 4)
            s["settlement_ece"] = s["ece"]
            summary[slice_name] = s

        keep: dict[str, list[dict]] = {}
        decide: dict = {}
        for slice_name in ("is", "oos"):
            rows = rows_by_slice[slice_name]
            lane = lanes[str(rows[0]["lane"])] if rows else lanes[LANES[0]]
            # the filter needs the lane of each row
            kept = []
            hurdle = max(Decimal(str(params.edge_hurdle)), Decimal(str(params.min_net_edge)))
            for r in rows:
                lane_r = lanes[str(r["lane"])]
                quote = lane_r.quote_at(int(r["tid"]), float(r["ts"]), valid_only=True,
                                        lookback=2.0)
                if quote is None:
                    r["decide_takes"] = False
                    continue
                bid, ask = quote
                side_ask = ask if r["side"] == "YES" else 1.0 - bid
                gross = exit_cost(Decimal(str(side_ask)), params)
                edge = Decimal(str(r["p_side"])) - Decimal(str(side_ask))
                if r["entry_style"] == "taker":
                    net = edge - gross - taker_fee_per_contract(Decimal(str(side_ask)))
                else:
                    net = edge - gross
                r["decide_edge_c"] = float(net)
                r["decide_hurdle_c"] = float(hurdle)
                r["decide_takes"] = bool(net >= hurdle)
                if r["decide_takes"]:
                    kept.append(r)
            keep[slice_name] = kept
            s = summarize(kept, 1.0) if kept else {"trades": 0, "net_pnl": 0.0, "pf": 0.0}
            decide[slice_name] = {
                "trades": s.get("trades", 0), "net_pnl": s.get("net_pnl", 0.0),
                "pf": s.get("pf", 0.0),
                "payoff_event_ece": round(features.expected_calibration_error(
                    [(float(r["p_side"]), 1 if float(r["pnl"]) > 0 else 0) for r in kept]), 4),
                "by_lane": s.get("by_lane", {}),
                "taken_of": len(rows),
            }

        for slice_name in ("is", "oos"):
            for r in rows_by_slice[slice_name]:
                dump.write(json.dumps({
                    "variant": f"F2_{style}_1s", "slice": slice_name, "lane": r["lane"],
                    "ticker": r["ticker"], "ts": r["ts"], "side": r["side"],
                    "entry": str(r["entry"]), "qty": r["qty"], "exit": str(r["exit"]),
                    "pnl": float(r["pnl"]), "fee": float(r["fee"]), "reason": r["reason"],
                    "p_side_raw": r.get("p_side_raw"), "p_side": round(float(r["p_side"]), 4),
                    "y": r["y"], "decide_edge_c": r.get("decide_edge_c"),
                    "decide_takes": r.get("decide_takes")}, default=str) + "\n")

        name = f"F2_{style}_1s"
        report["variants"][name] = {
            "entry_style": style,
            "events": events_by_slice,
            "summary": summary,
            "calibration": cal,
            "decide_filter": decide,
            "hurdle_c": float(max(Decimal(str(params.edge_hurdle)),
                                  Decimal(str(params.min_net_edge)))),
        }
        print(f"\n=== {name} ===")
        for slice_name in ("is", "oos"):
            s = summary[slice_name]
            print(f"  {slice_name.upper():3s} events {s['events']:5d}  fills {s['trades']:5d} "
                  f"({s['fill_rate']:.1%} filled)  win {s['win_rate']:.3f}  "
                  f"net {s['net_pnl']:+9.2f}  PF {s['pf']:6.3f}  "
                  f"payoff-ECE {s['payoff_event_ece']:.4f}  "
                  f"settle-ECE {s['settlement_ece']:.4f}  maxDD {s['max_dd']:.2f}")
            print(f"       by lane: {json.dumps(s.get('by_lane', {}))}")
        if cal.get("fitted"):
            print(f"  calibrated on {cal['rows']} IS trades (target: trade pnl > 0)")
            for slice_name in ("is", "oos"):
                d = decide[slice_name]
                print(f"       decide() keeps {d['taken_of']} -> {d['trades']} {slice_name.upper()} "
                      f"trades, net {d['net_pnl']:+8.2f}, PF {d['pf']:.3f}, "
                      f"payoff-ECE {d['payoff_event_ece']:.4f}")
    dump.close()
    OUT.write_text(json.dumps(report, indent=1, default=str))
    print(f"\nwrote {OUT}\nwrote {TRADES_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
