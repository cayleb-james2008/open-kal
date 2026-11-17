#!/usr/bin/env python3
"""Track B: three upgrades to the frozen F2_taker_1s rule, IS-selected, OOS-scored once.

Upgrades, each tested alone and combined, on the same chronological 60/40 split:
  peg500 / peg1000  when spread >= 2c, rest post-only at bid + 1c (YES terms for a
                    long; NO bid + 1c for a short) for 0.5s / 1.0s, filling only on
                    >= 50 printed contracts through the rest. Unfilled after the peg
                    window, fall back to an IOC capped at the ask pinned at arrival.
  tp6 / tp7         take-profit 6c or 7c, a $0-fee maker exit, when the spot-to-strike
                    delta is >= 6bp. Below 6bp the take-profit stays at 5c.
  eth_d5 / eth_spr2 KXETH15M only: require delta >= 5bp, or spread <= 2c.

Two NO-side conventions are run on the same tape, because the baseline's is suspect:
  as_built    the sim's YES-level price for a NO order is 1 - (NO price), i.e. the
              existing `f2_structural.limit_price` / `events_for` convention.
  corrected   the YES-level price for a NO order is the YES price it rests at:
              taker NO = YES bid b, maker NO = YES ask a, peg NO = a - 1c.
YES-side handling is identical in both.

Selection is on IS only: among grid rows with >= 20 IS trades, >= 3 IS losers and zero
crossed fills, the highest IS net pnl wins. OOS is scored once for that row. The other
rows' OOS numbers are printed for transparency and were not used to choose.

Reads the cache at EVOLVE_CACHE (default data/frozen-cache), writes EVOLVE_OUT (default
data/evolve-f2-variants.json). Nothing under /home/cayleb/open-kal is read or written.

    .venv/bin/python scripts/evolve_f2_variants.py
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openkal import features  # noqa: E402
from scripts import alpha_discover as AD  # noqa: E402
from scripts import alpha_search as A  # noqa: E402
from scripts import f2_structural as F  # noqa: E402
from scripts.bounded_sweep import pick, summarize  # noqa: E402

AD.CACHE = Path(os.environ.get("EVOLVE_CACHE", ROOT / "data" / "frozen-cache"))
OUT = Path(os.environ.get("EVOLVE_OUT", ROOT / "data" / "evolve-f2-variants.json"))

PEG = {"taker": None, "peg500": 0.5, "peg1000": 1.0}
TP_HI = {"tp5": None, "tp6": 0.06, "tp7": 0.07}
ETH = ("none", "d5", "spr2")
ENTRY = ("taker", "peg500", "peg1000")
CONVENTIONS = ("as_built", "corrected")
PEG_OFFSET = 0.01
MIN_IS_TRADES, MIN_IS_LOSSES = 20, 3
LANE_ETH = "KXETH15M"


def ylimit(long_yes: bool, b: float, a: float, kind: str, conv: str) -> float:
    """The YES-level price the simulator is given for one order."""
    nos_as_built = conv == "as_built"
    if kind == "taker":
        if long_yes:
            return a
        return 1.0 - b if nos_as_built else b
    if kind == "rest":
        if long_yes:
            return b
        return 1.0 - a if nos_as_built else a
    if kind == "peg":
        if long_yes:
            return b + PEG_OFFSET
        return 1.0 - (a - PEG_OFFSET) if nos_as_built else a - PEG_OFFSET
    raise ValueError(kind)


def side_cost(long_yes: bool, px: float) -> float:
    return px if long_yes else 1.0 - px


def arrivals(lanes: dict[str, AD.Lane], pools: dict[str, dict], slice_name: str) -> list[dict]:
    """Frozen-rule triggers in one slice, with the quote at arrival (T + 1s)."""
    out: list[dict] = []
    for name, lane in lanes.items():
        pool = pools[name]
        mask, ly = F.frozen_mask(pool, "taker")
        mask = mask & (pool["slice"] == slice_name)
        for i in pick(pool, mask, cooldown=F.COOLDOWN, per_window=F.PER_WINDOW):
            trig = float(pool["ts"][i])
            close = float(pool["close"][i])
            arr = trig + F.LAG_S
            if close - arr < 5.0:
                continue
            tid = int(pool["tid"][i])
            quote = lane.quote_at(tid, arr, valid_only=True, lookback=2.0)
            if quote is None:
                continue
            b, a = quote
            if not (0.0 < b < a < 1.0):
                continue
            out.append({
                "lane": name, "tid": tid, "ticker": str(pool["ticker"][i]),
                "y": int(lane.meta.get(str(pool["ticker"][i]), {}).get("y", 0)),
                "arrival": arr, "close": close, "long_yes": bool(ly[i]),
                "b": b, "a": a, "spread": a - b,
                "d_bps": float(pool["delta"][i]) / max(float(pool["spot"][i]), 1.0) * 1e4,
                "feat": F.feature_row(pool, i), "p_side": float(pool["p_up"][i]),
            })
    return out


def _base_params(tp_c: float) -> A.SimParams:
    return A.SimParams(tp_c=tp_c, barrier_s=F.BARRIER_S, stop_c=F.STOP_C,
                       stop_confirm_s=F.STOP_CONFIRM_S, exit_mode="maker_tp",
                       max_entry_price=F.CAP, min_fill_qty=F.MIN_FILL, max_qty=F.MAX_QTY)


def _event(rec: dict, ts: float, px: float, style: str) -> A.Event:
    long_yes = rec["long_yes"]
    return A.Event("F2", f"F2_{style}_1s", tid=rec["tid"], ticker=rec["ticker"], ts=ts,
                   close=rec["close"], y=0, slice="is", side="YES" if long_yes else "NO",
                   entry_style=style, limit=px, p_side=rec["p_side"])


def _band_ok(long_yes: bool, px: float) -> bool:
    cost = side_cost(long_yes, px)
    return F.COST_LO - 1e-9 <= cost <= F.CAP + 1e-9


def sim_one(lane: AD.Lane, rec: dict, entry: str, tp_hi: float | None, conv: str) -> dict | None:
    """One trigger under one entry and take-profit rule. None when it does not trade."""
    tp = F.TP_C if (tp_hi is None or rec["d_bps"] < 6.0) else tp_hi
    base = _base_params(tp)
    long_yes, b, a, arr, tid = rec["long_yes"], rec["b"], rec["a"], rec["arrival"], rec["tid"]
    peg_s = PEG[entry]
    if peg_s is not None and rec["spread"] >= 0.02 - 1e-9:
        px = ylimit(long_yes, b, a, "peg", conv)
        if not (0.0 < px < 1.0) or not _band_ok(long_yes, px):
            return None
        end = min(arr + peg_s, rec["close"] - 20.0)
        if lane.first_fill(tid, arr + 1e-6, end, taker_yes=not long_yes, limit=px,
                           need=F.MIN_FILL) is not None:
            return _sim(lane, _event(rec, arr, px, "maker"), replace(base, patience_s=peg_s))
        # IOC fallback: the ask pinned at arrival caps the price; a worse ask does not fill.
        pinned = ylimit(long_yes, b, a, "taker", conv)
        t2 = arr + peg_s
        q = lane.quote_at(tid, t2, valid_only=True, lookback=2.0)
        if q is None or not (0.0 < q[0] < q[1] < 1.0):
            return None
        ioc = ylimit(long_yes, q[0], q[1], "taker", conv)
        if side_cost(long_yes, ioc) > side_cost(long_yes, pinned) + 1e-9:
            return None
        if not _band_ok(long_yes, ioc):
            return None
        return _sim(lane, _event(rec, t2, ioc, "taker"), base)
    px = ylimit(long_yes, b, a, "taker", conv)
    if not _band_ok(long_yes, px):
        return None
    return _sim(lane, _event(rec, arr, px, "taker"), base)


def _sim(lane: AD.Lane, ev: A.Event, params: A.SimParams) -> dict | None:
    try:
        return A._sim_event(lane, ev, params)
    except RuntimeError as exc:
        if "crossed" in str(exc):
            return {"crossed": True}
        raise


def run_config(recs: list[dict], lanes: dict[str, AD.Lane], entry: str, tp: str, eth: str,
               conv: str) -> list[dict]:
    rows: list[dict] = []
    for rec in recs:
        if rec["lane"] == LANE_ETH:
            if eth == "d5" and rec["d_bps"] < 5.0:
                continue
            if eth == "spr2" and rec["spread"] > 0.02 + 1e-9:
                continue
        row = sim_one(lanes[rec["lane"]], rec, entry, TP_HI[tp], conv)
        if row is None:
            continue
        if row.get("crossed"):
            rows.append({"crossed": True, "lane": rec["lane"]})
            continue
        row["lane"] = rec["lane"]
        row["y"] = rec["y"]
        row["feat"] = rec["feat"]
        row["p_side"] = rec["p_side"]
        rows.append(row)
    return rows


def _payoff_ece(p: np.ndarray, won: np.ndarray) -> float:
    return round(features.expected_calibration_error(list(zip(p.tolist(), won.astype(int).tolist()))), 4)


def score(rows_is: list[dict], rows_oos: list[dict]) -> dict:
    """Summaries, with the payoff model fitted on IS only and applied to OOS."""
    clean_is = [r for r in rows_is if not r.get("crossed")]
    clean_oos = [r for r in rows_oos if not r.get("crossed")]
    crossed = {"is": sum(1 for r in rows_is if r.get("crossed")),
               "oos": sum(1 for r in rows_oos if r.get("crossed"))}
    x_is = np.array([r["feat"] for r in clean_is], dtype=np.float64).reshape(-1, 4)
    y_is = np.array([float(r["pnl"]) > 0 for r in clean_is], dtype=np.float64)
    if len(clean_is) >= 20:
        w, norm = F.fit_logistic(x_is, y_is)
        p_is = F.apply_logistic(x_is, w, norm)
        p_oos = (F.apply_logistic(np.array([r["feat"] for r in clean_oos], dtype=np.float64)
                                  .reshape(-1, 4), w, norm) if clean_oos else np.zeros(0))
    else:
        p_is = np.zeros(0)
        p_oos = np.zeros(0)
    out: dict = {"crossed_fills": crossed}
    for label, rows, probs in (("is", clean_is, p_is), ("oos", clean_oos, p_oos)):
        s = summarize(rows, 1.0)
        s["crossed_fills"] = crossed[label]
        s["payoff_ece"] = _payoff_ece(probs, np.array([float(r["pnl"]) > 0 for r in rows])) if len(rows) else 0.0
        s["lanes_profitable"] = sum(1 for v in s["by_lane"].values() if v["net"] > 0)
        s.pop("trades_per_hour", None)
        out[label] = s
    return out


def gate_ok(oos: dict) -> dict:
    checks = {
        "oos_net_positive": oos["net_pnl"] > 0,
        "oos_pf_at_least_1_15": oos["pf"] >= 1.15,
        "oos_payoff_ece_at_most_0_08": oos["payoff_ece"] <= 0.08,
        "oos_trades_at_least_20": oos["trades"] >= 20,
        "oos_zero_crossed": oos["crossed_fills"] == 0,
        "oos_two_lanes_profitable": oos["lanes_profitable"] >= 2,
    }
    checks["pass"] = all(checks.values())
    return checks


def main() -> int:
    lanes = {name: AD.Lane(name) for name in F.LANES}
    pools = F.build_pools(lanes)
    recs = {s: arrivals(lanes, pools, s) for s in ("is", "oos")}
    grid = list(itertools.product(ENTRY, tuple(TP_HI), ETH))
    result: dict = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "cache": str(AD.CACHE),
        "split": "chronological 60/40 by window close, per the f2_structural split",
        "triggers": {s: len(recs[s]) for s in recs},
        "selection": "IS only: >=20 IS trades, >=3 IS losers, 0 crossed; max IS net",
        "conventions": {},
    }
    for conv in CONVENTIONS:
        table = {}
        for entry, tp, eth in grid:
            key = f"{entry}|{tp}|{eth}"
            rows_is = run_config(recs["is"], lanes, entry, tp, eth, conv)
            rows_oos = run_config(recs["oos"], lanes, entry, tp, eth, conv)
            table[key] = {"entry": entry, "tp": tp, "eth": eth, **score(rows_is, rows_oos)}
            print(f"[{conv}] {key:28s} IS n={table[key]['is']['trades']:3d} "
                  f"net={table[key]['is']['net_pnl']:+8.2f} | OOS n={table[key]['oos']['trades']:3d} "
                  f"net={table[key]['oos']['net_pnl']:+8.2f} PF={table[key]['oos']['pf']:.2f}",
                  flush=True)
        eligible = [k for k, v in table.items()
                    if v["is"]["trades"] >= MIN_IS_TRADES and v["is"]["losses"] >= MIN_IS_LOSSES
                    and v["is"]["crossed_fills"] == 0]
        chosen = max(eligible, key=lambda k: table[k]["is"]["net_pnl"]) if eligible else None
        result["conventions"][conv] = {
            "grid": table,
            "chosen_on_is": chosen,
            "chosen_oos": table[chosen]["oos"] if chosen else None,
            "gate_on_chosen_oos": gate_ok(table[chosen]["oos"]) if chosen else None,
            "baseline_row": table["taker|tp5|none"],
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1, default=str))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
