#!/usr/bin/env python3
"""Spot-to-RTI construction-lag discovery, strictly before the forward holdout.

The thesis: the Coinbase spot feed recorded in `ticks.npz` moves first, and the CF
Benchmarks RTI tape the contract settles on prints the same move later. When a fast
2-3 second spot impulse has been captured by the RTI by less than 50%, the Kalshi
ladder is pricing an average the reference feed has not caught up to yet.

Only pre-holdout rows are ever used. Every window is filtered by close time, the
settlement-index tapes are truncated with `index_before`, and the run asserts that
the largest signal timestamp is below `FORWARD_OOS_START_TS` (1794909600).

Mechanism `spot_rti_lag`:

  * fast spot impulse: `ticks.npz:spot` log return over a 2-3 second lookback,
    |move| >= 6 / 8 / 12 bp;
  * construction lag: the lane's own RTI index moved less than 50% of that spot
    move over the same interval (`rti_catchup_ratio < 0.50`);
  * entry: taker at the T+1s quote, side cost in [$0.25, $0.62];
  * exits: $0-fee +5c/+7c maker take-profit with >= 50-contract print-through, a
    7c two-second-confirmed taker stop at the book, or a 90s time barrier;
  * exact `Decimal` taker_fee on entry, stop and barrier; maker legs are $0.

Selection is strictly in-sample: at most one candidate, >= 20 IS trades, IS net > 0,
highest IS PF. The frozen pre-05:00 OOS slice is scored once for that candidate.

    .venv/bin/python scripts/rti_lag_search.py
"""
from __future__ import annotations

import json
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.novel_alpha_search as S  # noqa: E402
from scripts import alpha_search as A  # noqa: E402
from scripts.alpha_discover import LANES, Lane  # noqa: E402
from scripts.forward_oos import FORWARD_OOS_START_TS  # noqa: E402
from scripts.subsec_discovery import book_at, index_before  # noqa: E402

COST_LO = 0.25
COST_HI = 0.62
LAG_S = 1.0
LOOKBACK_S = 2.0
GAP_LO, GAP_HI = 1.9, 3.5          # the recorded spot tape is dense; 2-3s is a lookback
CATCHUP_MAX = 0.50
IMPULSE_GRID = (6.0, 8.0, 12.0)
TP_GRID = (0.05, 0.07)
STOP_C = 0.07
STOP_CONFIRM_S = 2.0
BARRIER_S = 90.0
COOLDOWN_S = 90.0
PER_WINDOW = 3
SECS_MIN = 6.0
SECS_MAX = 840.0
RTI_TOL_S = 1.5
OUT = ROOT / "data" / "rti-lag-search.json"

COLS = ("tid", "ts", "close", "y", "side_yes", "entry", "cost", "p_side", "is_is",
        "imp_bps", "rti_bps", "ratio", "spot")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _blank() -> dict:
    return {k: [] for k in COLS} | {"lane": []}


def _add(pool: dict, lane_name: str, arrays: dict) -> None:
    n = len(arrays["ts"])
    if n == 0:
        return
    for k in COLS:
        pool[k].append(np.asarray(arrays[k]))
    pool["lane"].append(np.full(n, lane_name, dtype=object))


def _finish(pool: dict) -> dict | None:
    if not pool["lane"]:
        return None
    return {k: np.concatenate(v) for k, v in pool.items()}


def _pick(pool: dict, mask: np.ndarray) -> list[int]:
    """Desk convention for this family: 90s cooldown, at most 3 entries per window."""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    tid = pool["tid"][idx]
    out: list[int] = []
    lo = 0
    while lo < len(idx):
        hi = int(np.searchsorted(tid, tid[lo], "right"))
        if hi <= lo:
            hi = lo + 1
        out.extend(A._dedupe([int(i) for i in idx[lo:hi]], pool["ts"],
                             cooldown=COOLDOWN_S, per_ticker=PER_WINDOW))
        lo = hi
    return out


def _rti_at(its: np.ndarray, ival: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Last RTI print at or before each query, within `RTI_TOL_S`."""
    j = np.searchsorted(its, query, "right") - 1
    jj = np.clip(j, 0, len(its) - 1)
    ok = (j >= 0) & ((query - its[jj]) <= RTI_TOL_S) & (ival[jj] > 0)
    return ival[jj], ok


def build_pool(lane: Lane, ticks: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict | None:
    index = index_before(lane.index)
    if not index:
        return None
    its = np.asarray(index["ts"], dtype=np.float64)
    ival = np.asarray(index["value"], dtype=np.float64)
    pool = _blank()
    for tid, name, meta, _a, _b in A._segments(lane):
        close = float(meta["close"])
        if close >= FORWARD_OOS_START_TS:
            continue
        if name not in ticks:
            continue
        tt_all, ss_all = ticks[name]
        lo = int(np.searchsorted(tt_all, close - SECS_MAX, "left"))
        hi = int(np.searchsorted(tt_all, close - SECS_MIN, "right"))
        tt, ss = tt_all[lo:hi], ss_all[lo:hi]
        if len(tt) < 10:
            continue
        n = len(tt)
        base = np.searchsorted(tt, tt - LOOKBACK_S, "right") - 1
        jj = np.clip(base, 0, n - 1)
        dt = tt - tt[jj]
        ok = (base >= 0) & (dt >= GAP_LO) & (dt <= GAP_HI) & (ss > 0) & (ss[jj] > 0)
        if not ok.any():
            continue
        spot_bps = np.where(ok, np.log(np.maximum(ss, 1e-12)
                                       / np.maximum(ss[jj], 1e-12)) * 1e4, 0.0)
        rti_now, ok_now = _rti_at(its, ival, tt)
        rti_then, ok_then = _rti_at(its, ival, tt[jj])
        rti_ok = ok_now & ok_then
        rti_bps = np.where(rti_ok, np.log(np.maximum(rti_now, 1e-12)
                                          / np.maximum(rti_then, 1e-12)) * 1e4, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(rti_ok & (np.abs(spot_bps) > 1e-9), rti_bps / spot_bps, 99.0)
        trigger = (ok & rti_ok & (np.abs(spot_bps) >= min(IMPULSE_GRID))
                   & (ratio < CATCHUP_MAX))
        if not trigger.any():
            continue
        sel = np.flatnonzero(trigger)
        t_sel = tt[sel]
        arrival = t_sel + LAG_S
        bid1, ask1 = book_at(lane, tid, arrival)
        fresh = S.fresh_quotes(lane, tid, arrival)
        direction = np.where(spot_bps[sel] >= 0, 1.0, -1.0)
        side_yes = direction > 0
        cost = np.where(side_yes, ask1, 1.0 - bid1)
        entry = np.where(side_yes, ask1, bid1)
        live = (fresh & (bid1 > 0) & (ask1 < 1) & (bid1 < ask1)
                & (cost >= COST_LO - S.EPS) & (cost <= COST_HI + S.EPS)
                & (close - arrival >= SECS_MIN))
        if not live.any():
            continue
        _add(pool, lane.lane, {
            "tid": np.full(int(live.sum()), tid, dtype=np.int32),
            "ts": t_sel[live], "close": np.full(int(live.sum()), close),
            "y": np.full(int(live.sum()), int(meta["y"]), dtype=np.int8),
            "side_yes": side_yes[live], "entry": entry[live], "cost": cost[live],
            "p_side": np.full(int(live.sum()), 0.5), "is_is": np.full(
                int(live.sum()), 1 if lane.slice_of(close) == "is" else 0, dtype=np.int8),
            "imp_bps": spot_bps[sel][live], "rti_bps": rti_bps[sel][live],
            "ratio": ratio[sel][live], "spot": ss[sel][live],
        })
    return _finish(pool)


def candidates() -> list[dict]:
    return [{"mechanism": "spot_rti_lag", "name": f"lag_i{imp:.0f}_tp{tp:.2f}",
             "imp_bps": imp, "tp_c": tp, "exit_mode": "maker_tp",
             "barrier_s": BARRIER_S, "stop_c": STOP_C, "stop_confirm_s": STOP_CONFIRM_S}
            for imp in IMPULSE_GRID for tp in TP_GRID]


def mask_for(pool: dict, cand: dict, slice_name: str) -> np.ndarray:
    return ((pool["is_is"] == (1 if slice_name == "is" else 0))
            & (pool["imp_bps"] >= cand["imp_bps"] - S.EPS))


def main() -> int:
    t0 = time.time()
    z = np.load(ROOT / "data" / "evolve-cache" / "ticks.npz", allow_pickle=True)
    ts_all = np.asarray(z["ts"], dtype=np.float64)
    tick_all = np.asarray(z["ticker"])
    spot_all = np.asarray(z["spot"], dtype=np.float64)
    pre = ts_all < FORWARD_OOS_START_TS
    order = np.argsort(ts_all[pre], kind="stable")
    ts_pre = ts_all[pre][order]
    tick_pre = tick_all[pre][order]
    spot_pre = spot_all[pre][order]
    ticks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in dict.fromkeys(tick_pre.tolist()):
        m = tick_pre == name
        ticks[name] = (ts_pre[m], spot_pre[m])
    _log(f"spot tape: {len(ts_pre):,} pre-holdout ticks, {len(ticks)} tickers")

    grid = candidates()
    is_rows: dict[str, list[dict]] = {c["name"]: [] for c in grid}
    pools: dict[str, dict] = {}
    lanes: dict[str, Lane] = {}
    max_ts = 0.0
    is_w = oos_w = 0
    is_span: list[float] = []
    oos_span: list[float] = []
    for name in LANES:
        lane = Lane(name, max_close=FORWARD_OOS_START_TS)
        lanes[name] = lane
        pool = build_pool(lane, ticks)
        if pool is None:
            raise RuntimeError(f"{name}: no lag rows")
        pools[name] = pool
        is_w += len(lane.is_closes)
        oos_w += len(lane.oos_closes)
        is_span.extend(lane.is_closes)
        oos_span.extend(lane.oos_closes)
        top = float(pool["ts"].max())
        if top >= FORWARD_OOS_START_TS:
            raise RuntimeError(f"{name}: signal at {top} is inside the holdout")
        max_ts = max(max_ts, top)
        _log(f"{name}: {len(pool['ts'])} lag rows across "
             f"{len(set(pool['tid'].tolist()))} windows")
        for cand in grid:
            idxs = _pick(pool, mask_for(pool, cand, "is"))
            if not idxs:
                continue
            is_rows[cand["name"]].extend(S.simulate(
                lane, pool, idxs, exit_mode=cand["exit_mode"], tp_c=cand["tp_c"],
                barrier_s=cand["barrier_s"], stop_c=cand["stop_c"],
                stop_confirm_s=cand["stop_confirm_s"]))
    if max_ts >= FORWARD_OOS_START_TS:
        raise RuntimeError(f"read a signal at {max_ts}, at/after the holdout start")

    summaries: list[dict] = []
    selected: dict | None = None
    for cand in grid:
        s = S.summarize(is_rows[cand["name"]])
        summaries.append({"name": cand["name"], "imp_bps": cand["imp_bps"],
                          "tp_c": cand["tp_c"], "is": s})
        if int(s["trades"]) >= 20 and Decimal(s["net"]) > 0:
            if selected is None or (s["pf"], Decimal(s["net"])) > (
                    selected["is"]["pf"], Decimal(selected["is"]["net"])):
                selected = {"name": cand["name"], "params": cand, "is": s,
                            "is_trades": is_rows[cand["name"]]}

    oos: dict | None = None
    if selected is not None:
        cand = selected["params"]
        rows: list[dict] = []
        for name, lane in lanes.items():
            pool = pools[name]
            idxs = _pick(pool, mask_for(pool, cand, "oos"))
            if not idxs:
                continue
            rows.extend(S.simulate(lane, pool, idxs, exit_mode=cand["exit_mode"],
                                   tp_c=cand["tp_c"], barrier_s=cand["barrier_s"],
                                   stop_c=cand["stop_c"],
                                   stop_confirm_s=cand["stop_confirm_s"]))
        oos = {"name": selected["name"], "oos": S.summarize(rows), "trades": rows}

    def clean(rows):
        return [{k: (str(v) if isinstance(v, Decimal) else v) for k, v in r.items()}
                for r in rows]

    payload = {
        "generated_ts": time.time(),
        "elapsed_s": round(time.time() - t0, 1),
        "guard": {
            "holdout_start_ts": FORWARD_OOS_START_TS,
            "rule": "no row at/after the holdout start enters a pool, fit or score",
            "max_signal_ts": round(max_ts, 3),
            "max_signal_below_holdout": max_ts < FORWARD_OOS_START_TS,
        },
        "split": {
            "rule": "chronological 60/40 by settled window close, within ts < holdout",
            "is_windows_per_lane": is_w // len(LANES),
            "oos_windows_per_lane": oos_w // len(LANES),
            "is_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(min(is_span))),
            "is_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(max(is_span))),
            "oos_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(min(oos_span))),
            "oos_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(max(oos_span))),
        },
        "mechanism": {
            "name": "spot_rti_lag",
            "fast_spot": f"ticks.npz:spot log move over {LOOKBACK_S:g}s, gap "
                         f"[{GAP_LO},{GAP_HI}]s, |move| >= {IMPULSE_GRID}",
            "lag": f"rti_catchup_ratio < {CATCHUP_MAX} over the same interval",
            "entry": f"taker at T+{LAG_S:g}s, side cost in [{COST_LO},{COST_HI}]",
            "exit": f"+{TP_GRID[0]:.2f}/+{TP_GRID[1]:.2f} maker TP (>=50 print-through), "
                    f"{STOP_CONFIRM_S:g}s confirmed {STOP_C:.2f} taker stop, "
                    f"{BARRIER_S:g}s barrier",
            "fill_rules": "exact Decimal taker_fee on taker legs; $0 maker legs; "
                          "crossed books abort; 50 contracts per trade",
            "cooldown_s": COOLDOWN_S, "per_window": PER_WINDOW,
        },
        "candidates_is": summaries,
        "selected": (None if selected is None else
                     {"name": selected["name"], "is": selected["is"],
                      "params": {k: v for k, v in selected["params"].items()
                                 if k not in ("name", "mechanism")},
                      "is_trades": clean(selected["is_trades"])}),
        "oos": (None if oos is None else
                {"name": oos["name"], "oos": oos["oos"], "trades": clean(oos["trades"])}),
        "selection": {"rule": "IS only: >=20 trades, net > 0, highest PF; OOS once",
                      "oos_scored_once": bool(oos is not None)},
        "tape": {"lanes": list(LANES),
                 "pool_rows": {n: int(len(p["ts"])) for n, p in pools.items()}},
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    _print(payload)
    print(f"\nwrote {OUT}")
    return 0


def _print(payload: dict) -> None:
    def line(label: str, s: dict) -> str:
        n = int(s["trades"] or 0)
        per = (float(s["net"]) / n) if n else 0.0
        return (f"{label:<22} {n:>4} {s['win_rate'] * 100:>6.1f} {s['net']:>9} "
                f"{s['pf']:>5.2f} {per:>6.2f} {s['fees']:>7} {s['max_side_cost']:>5.2f}")

    print(f"\n{'candidate':<22} {'n':>4} {'win%':>6} {'net$':>9} {'PF':>5} "
          f"{'$/tr':>6} {'fees$':>7} {'max$':>5}")
    print("-" * 70)
    for c in payload["candidates_is"]:
        print(line(f"{c['name']} IS", c["is"]))
    if payload["selected"]:
        print(line(f"{payload['selected']['name']} IS*", payload["selected"]["is"]))
        if payload["oos"]:
            print(line(f"{payload['selected']['name']} OOS", payload["oos"]["oos"]))
    else:
        print("no candidate cleared the 20-trade IS floor with positive IS net")


if __name__ == "__main__":
    raise SystemExit(main())
