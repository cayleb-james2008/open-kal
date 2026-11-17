#!/usr/bin/env python3
"""Two genuinely new positive-skew / fair-value mechanisms, scored strictly pre-holdout.

Nothing in this file may read a row at or after `FORWARD_OOS_START_TS`
(1794909600 == 2026-11-17 05:00:00 EST). The cache is materialised for the whole
recording, so the guards are explicit: every window is filtered by close time, the
settlement-index tapes are truncated with `index_before`, and the study asserts that
the largest timestamp it touched is below the holdout start.

  A. `fv_residual_snipe` -- spot/index fair-value residual snipe.
     `fair_p = Phi(log(spot/strike) / vol)` on each lane's own RTI index, with the
     settlement-average volatility law: inside the final minute the remaining
     settlement slice is a time average of Brownian motion, so its sd is
     `sigma * sqrt(T/3)`; before the final minute the plain Brownian prior
     `sigma * sqrt(T)` is used. A 3-second BTC impulse of >= 5 bp that agrees with
     the lane's own 3-second move is the trigger; the trade is only taken when the
     model's edge over the ask *one second later* clears a hurdle and that side
     costs between $0.25 and $0.62. Exits:
       (a) resting +6c/+8c maker take-profit, 60 s barrier, 2 s confirmed 10c stop;
       (b) hold to settlement (zero exit fee) at a >= 8c edge.

  B. `underdog_convex` -- cheap positive-skew convexity.
     A 5-second own-index momentum of >= 6 bp that leaves the model >= 5c above a
     $0.18-$0.42 ask. The maximum loss is the premium paid, there is no taker stop
     (so no stop-out fee bleed), and the exit is either hold-to-settlement or a
     resting +15c/+20c maker take-profit that carries to settlement when unfilled.

Physics enforced here, not configurable per variant:

  * every entry pays the book 1.0 s after the trigger (T+1s execution lag);
  * every entry costs <= $0.62 per contract for the side held;
  * a passive leg (take-profit) counts only when >= 50 contracts print at/through
    its limit -- a 1-lot touch is not liquidity;
  * a two-sided book is required (`0 < bid < ask < 1`); a crossed row aborts;
  * every taker leg pays `openkal.fees.taker_fee` in exact Decimal, maker legs $0;
  * 50 contracts per trade, the largest size the print-through rule can verify.

Selection is strictly chronological in-sample (60% of the pre-holdout settled
windows by close time): at most one candidate per mechanism, requiring >= 20 IS
trades and positive IS net, ranked by IS profit factor. Only that candidate's
frozen out-of-sample slice is scored, once. The F2_taker_1s row in the output is
read from the existing frozen study and is a reference, not a re-evaluation.

    .venv/bin/python scripts/novel_alpha_search.py
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

from openkal.fees import taker_fee  # noqa: E402
from scripts import alpha_search as A  # noqa: E402
from scripts.alpha_discover import EPS, LANES, Lane  # noqa: E402
from scripts.forward_oos import FORWARD_OOS_START_TS  # noqa: E402
from scripts.subsec_discovery import (Sigma, book_at, index_before,  # noqa: E402
                                      norm_cdf)

CAP = 0.62                 # hard price cap per contract, side held
MIN_FILL = 50.0            # contracts that must print at/through a passive limit
QTY = 50                   # contracts per trade
LAG_S = 1.0                # trigger -> order arrival
WINDOW_S = 60.0            # the settlement window
SECS_MIN = 6.0             # no entry with less than this left
SECS_MAX = 840.0           # no entry before this much of the window has run
QUOTE_MAX_AGE_S = 5.0      # the arrival quote must be live
COOLDOWN_S = 900.0         # one entry per window
PER_WINDOW = 1
A_MIN_BTC_BPS = 5.0        # BTC 3s impulse floor (mechanism A)
A_HURDLES = (0.05, 0.08, 0.10)
A_TP_CS = (0.06, 0.08)
A_HOLD_HURDLE = 0.08
A_STOP_C = 0.10            # 2s confirmed stop for the take-profit variants
A_STOP_CONFIRM_S = 2.0
A_BARRIER_S = 60.0
A_COST_LO = 0.25
B_MIN_MOM_BPS = 6.0        # 5s own-index momentum floor (mechanism B)
B_HURDLE = 0.05
B_COST_LO = 0.18
B_COST_HI = 0.42
B_TP_CS = (0.15, 0.20)
B_TP_BARRIER_S = 600.0
OUT = ROOT / "data" / "novel-alpha-search.json"
F2_ARTIFACT = ROOT / "data" / "f2-structural.json"
ZERO = Decimal("0")

A_COLS = ("tid", "ts", "close", "y", "side_yes", "entry", "cost", "p_side", "edge",
          "spot", "secs", "own_bps", "btc_bps", "is_is")
B_COLS = ("tid", "ts", "close", "y", "side_yes", "entry", "cost", "p_side", "edge",
          "spot", "secs", "own_bps", "is_is")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# ----------------------------------------------------------------------- pools
def _blank(cols: tuple[str, ...]) -> dict:
    return {k: [] for k in cols} | {"lane": []}


def pool_add(pool: dict, lane_name: str, cols: dict, mask: np.ndarray) -> None:
    n = int(mask.sum())
    if n == 0:
        return
    for k in pool:
        if k == "lane":
            continue
        pool[k].append(np.asarray(cols[k])[mask])
    pool["lane"].append(np.full(n, lane_name, dtype=object))


def pool_finish(pool: dict) -> dict | None:
    if not pool["lane"]:
        return None
    return {k: np.concatenate(v) for k, v in pool.items()}


def fresh_quotes(lane: Lane, tid: int, query: np.ndarray,
                 max_age: float = QUOTE_MAX_AGE_S) -> np.ndarray:
    """True where at least one two-sided book row exists within `max_age` of `query`."""
    lo, hi = int(lane.book_bounds[tid]), int(lane.book_bounds[tid + 1])
    if hi <= lo:
        return np.zeros(len(query), dtype=bool)
    seg_ts = lane.ts[lo:hi]
    valid = (lane.bid[lo:hi] > 0) & (lane.ask[lo:hi] < 1) & (lane.bid[lo:hi] < lane.ask[lo:hi])
    cs = np.concatenate(([0], np.cumsum(valid.astype(np.int64))))
    end = np.searchsorted(seg_ts, query, "right")
    start = np.searchsorted(seg_ts, query - max_age, "left")
    return (cs[end] - cs[start]) > 0


def build_pools(lane: Lane, btc_index: dict) -> tuple[dict | None, dict | None]:
    """Feature rows for both mechanisms, on the lane's own settlement-index grid.

    The index grid is the decision clock (5 Hz), not the raw ladder: the ladder has
    hundreds of rows per second and pooling them would be a memory problem rather
    than a discovery. The execution quote is still the ladder's, read at T+1s.
    """
    index = index_before(lane.index)
    if not index or not btc_index:
        return None, None
    sig = Sigma(index)
    its = np.asarray(index["ts"], dtype=np.float64)
    ival = np.asarray(index["value"], dtype=np.float64)
    bts = np.asarray(btc_index["ts"], dtype=np.float64)
    bval = np.asarray(btc_index["value"], dtype=np.float64)
    pool_a = _blank(A_COLS)
    pool_b = _blank(B_COLS)
    for tid, name, meta, _a, _b in A._segments(lane):
        close = float(meta["close"])
        if close >= FORWARD_OOS_START_TS:
            continue
        strike = float(meta["strike"])
        lo = int(np.searchsorted(its, close - SECS_MAX, "left"))
        hi = int(np.searchsorted(its, close - SECS_MIN, "right"))
        t, v = its[lo:hi], ival[lo:hi]
        if len(t) < 100 or strike <= 0:
            continue
        n = len(t)
        sg = sig.at(t)
        j3 = np.clip(np.searchsorted(t, t - 3.0, "right") - 1, 0, n - 1)
        j5 = np.clip(np.searchsorted(t, t - 5.0, "right") - 1, 0, n - 1)
        # Exact duplicate index timestamps are collapsed by Sigma but not here; a
        # zero-length base would divide by the same print, so guard the values.
        own3 = np.where(v[j3] > 0, (v / np.maximum(v[j3], 1e-12) - 1.0) * 1e4, 0.0)
        own5 = np.where(v[j5] > 0, (v / np.maximum(v[j5], 1e-12) - 1.0) * 1e4, 0.0)
        jb = np.clip(np.searchsorted(bts, t, "right") - 1, 0, len(bts) - 1)
        jb3 = np.clip(np.searchsorted(bts, t - 3.0, "right") - 1, 0, len(bts) - 1)
        fresh_btc = (t - bts[jb]) <= 1.5
        btc3 = np.where(bval[jb3] > 0,
                        (bval[jb] / np.maximum(bval[jb3], 1e-12) - 1.0) * 1e4, 0.0)
        secs = close - t
        vol = np.where(secs < WINDOW_S,
                       sg * np.sqrt(np.minimum(secs, WINDOW_S) / 3.0),
                       sg * np.sqrt(np.maximum(secs, 0.0)))
        good = (sg > 0) & (v > 0) & (vol > 0) & (secs >= SECS_MIN) & fresh_btc
        p = np.where(good, norm_cdf(np.log(np.maximum(v, 1e-12) / strike)
                                    / np.where(vol > 0, vol, 1.0)), 0.5)

        arrival = t + LAG_S
        bid1, ask1 = book_at(lane, tid, arrival)
        fresh = fresh_quotes(lane, tid, arrival)
        live = good & fresh & (bid1 > 0) & (ask1 < 1) & (bid1 < ask1)
        if not live.any():
            continue
        is_is = np.full(n, 1 if lane.slice_of(close) == "is" else 0, dtype=np.int8)

        direction_a = np.where(btc3 >= 0, 1.0, -1.0)
        aligned = (np.abs(btc3) >= A_MIN_BTC_BPS) & (direction_a * own3 > 0)
        side_yes_a = direction_a > 0
        cost_a = np.where(side_yes_a, ask1, 1.0 - bid1)
        p_a = np.where(side_yes_a, p, 1.0 - p)
        entry_a = np.where(side_yes_a, ask1, bid1)
        edge_a = p_a - cost_a
        mask_a = (live & aligned & (cost_a >= A_COST_LO - EPS)
                  & (cost_a <= CAP + EPS) & (edge_a >= min(A_HURDLES) - EPS))
        pool_add(pool_a, lane.lane, {
            "tid": np.full(n, tid, dtype=np.int32), "ts": t, "close": np.full(n, close),
            "y": np.full(n, int(meta["y"]), dtype=np.int8), "side_yes": side_yes_a,
            "entry": entry_a, "cost": cost_a, "p_side": p_a, "edge": edge_a,
            "spot": v, "secs": secs, "own_bps": own3, "btc_bps": btc3, "is_is": is_is,
        }, mask_a)

        direction_b = np.where(own5 >= 0, 1.0, -1.0)
        side_yes_b = direction_b > 0
        cost_b = np.where(side_yes_b, ask1, 1.0 - bid1)
        p_b = np.where(side_yes_b, p, 1.0 - p)
        entry_b = np.where(side_yes_b, ask1, bid1)
        edge_b = p_b - cost_b
        mask_b = (live & (np.abs(own5) >= B_MIN_MOM_BPS)
                  & (cost_b >= B_COST_LO - EPS) & (cost_b <= B_COST_HI + EPS)
                  & (edge_b >= B_HURDLE - EPS))
        pool_add(pool_b, lane.lane, {
            "tid": np.full(n, tid, dtype=np.int32), "ts": t, "close": np.full(n, close),
            "y": np.full(n, int(meta["y"]), dtype=np.int8), "side_yes": side_yes_b,
            "entry": entry_b, "cost": cost_b, "p_side": p_b, "edge": edge_b,
            "spot": v, "secs": secs, "own_bps": own5, "is_is": is_is,
        }, mask_b)
    return pool_finish(pool_a), pool_finish(pool_b)


def pick(pool: dict, mask: np.ndarray) -> list[int]:
    """One entry per window, first trigger wins, `COOLDOWN_S` apart within a window."""
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


# ------------------------------------------------------------------ simulation
def _first_confirmed(ts: np.ndarray, adverse: np.ndarray, px: np.ndarray,
                     confirm_s: float) -> tuple[float, float] | None:
    """First adverse touch that survives `confirm_s` unbroken (same as the desk's)."""
    bad = np.flatnonzero(adverse)
    if not len(bad):
        return None
    if confirm_s <= 0:
        return (float(ts[bad[0]]), float(px[bad[0]]))
    edges = np.flatnonzero(np.diff(np.concatenate(([0], adverse.astype(np.int8), [0]))))
    for s0, e0 in zip(edges[::2], edges[1::2]):
        start = float(ts[s0])
        if float(ts[e0 - 1]) - start < confirm_s - EPS:
            continue
        j = min(s0 + int(np.searchsorted(ts[s0:e0], start + confirm_s, "left")), e0 - 1)
        return (float(ts[j]), float(px[j]))
    return None


def simulate(lane: Lane, pool: dict, idxs: list[int], *, exit_mode: str,
             tp_c: float = 0.0, barrier_s: float = 0.0, stop_c: float = 0.0,
             stop_confirm_s: float = 0.0) -> list[dict]:
    """One trade at a time under one exit regime; every leg priced on the tape.

    Entry is a taker fill at the quote `LAG_S` after the trigger, so the fee is the
    exact `ceil(0.07 * Q * P * (1-P))` Decimal charge on the side actually held.
    A take-profit is passive and needs `MIN_FILL` contracts through its limit; when
    it never fills, `maker_tp` crosses out at the time barrier (A's stated 60s
    barrier) while `maker_tp_carry` carries the binary to settlement at $0 exit fee
    (B's no-taker-stop geometry).
    """
    out: list[dict] = []
    for i in idxs:
        tid = int(pool["tid"][i])
        arrival = float(pool["ts"][i])
        close = float(pool["close"][i])
        long_yes = bool(pool["side_yes"][i])
        entry = float(pool["entry"][i])
        cost = float(pool["cost"][i])
        if not (0.0 < entry < 1.0) or not (0.0 < cost <= CAP + EPS):
            continue
        if close - arrival < SECS_MIN:
            continue
        entry_dec = Decimal(str(round(entry, 4)))
        fee_in = taker_fee(Decimal(str(round(cost, 4))), QTY)
        reason = ""
        fee_out = ZERO
        if exit_mode == "hold_to_settlement":
            exit_dec = Decimal("1") if int(pool["y"][i]) == 1 else Decimal("0")
            reason = "hold_to_settlement"
        else:
            deadline = min(arrival + barrier_s, close - 5.0)
            if deadline <= arrival:
                continue
            target = round(entry + tp_c, 4) if long_yes else round(entry - tp_c, 4)
            take = None
            if 0.0 < target < 1.0:
                take = lane.first_fill(tid, arrival + 1e-6, deadline,
                                       taker_yes=long_yes, limit=target, need=MIN_FILL)
            hit = None
            if stop_c > 0:
                lo, hi = int(lane.book_bounds[tid]), int(lane.book_bounds[tid + 1])
                a0 = int(np.searchsorted(lane.ts[lo:hi], arrival + 1e-6, "right"))
                b0 = int(np.searchsorted(lane.ts[lo:hi], deadline, "right"))
                seg_ts = lane.ts[lo + a0:lo + b0]
                seg_bid = lane.bid[lo + a0:lo + b0]
                seg_ask = lane.ask[lo + a0:lo + b0]
                if long_yes:
                    hit = _first_confirmed(seg_ts, seg_bid <= entry - stop_c + EPS,
                                           seg_bid, stop_confirm_s)
                else:
                    hit = _first_confirmed(seg_ts, seg_ask >= entry + stop_c - EPS,
                                           seg_ask, stop_confirm_s)
            if take is not None and (hit is None or float(take[0]) <= hit[0]):
                exit_dec = Decimal(str(round(target, 4)))
                reason = "maker_tp"
            elif hit is not None and hit[1] > 0:
                exit_dec = Decimal(str(round(hit[1], 4)))
                reason = "stop_loss"
                fee_out = taker_fee(exit_dec, QTY)
            elif exit_mode == "maker_tp":
                quote = lane.quote_at(tid, deadline)
                if quote is None:
                    continue
                qb, qa = quote
                if not (0.0 < qb < qa < 1.0):
                    raise RuntimeError(f"crossed exit {lane.tickers[tid]} {qb}/{qa}")
                exit_dec = Decimal(str(round(qb if long_yes else qa, 4)))
                reason = "time_barrier"
                fee_out = taker_fee(exit_dec, QTY)
            else:
                exit_dec = Decimal("1") if int(pool["y"][i]) == 1 else Decimal("0")
                reason = "tp_unfilled_settle"
        gross = ((exit_dec - entry_dec) if long_yes else (entry_dec - exit_dec)) * QTY
        out.append({
            "ticker": lane.tickers[tid], "tid": tid, "ts": arrival, "close": close,
            "slice": "is" if int(pool["is_is"][i]) == 1 else "oos",
            "side": "YES" if long_yes else "NO", "entry": entry_dec, "exit": exit_dec,
            "cost": round(cost, 4), "qty": QTY, "p_side": float(pool["p_side"][i]),
            "fee": fee_in + fee_out, "pnl": gross - fee_in - fee_out,
            "reason": reason, "crossed": 0,
        })
    return out


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"trades": 0, "wins": 0, "win_rate": 0.0, "net": "0.00", "pf": 0.0,
                "avg_c": 0.0, "fees": "0.00", "max_side_cost": 0.0,
                "crossed_fills": 0, "reasons": {}, "lanes": {}}
    pnl = [r["pnl"] for r in rows]
    net = sum(pnl, ZERO)
    gross_win = sum((p for p in pnl if p > 0), ZERO)
    gross_loss = sum((-p for p in pnl if p <= 0), ZERO)
    pf = float(gross_win / gross_loss) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)
    fees = sum((r["fee"] for r in rows), ZERO)
    reasons: dict[str, int] = {}
    reasons_net: dict[str, Decimal] = {}
    lanes: dict[str, dict] = {}
    for r in rows:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        reasons_net[r["reason"]] = reasons_net.get(r["reason"], ZERO) + r["pnl"]
        b = lanes.setdefault(r["ticker"].split("-")[0], {"trades": 0, "net": ZERO})
        b["trades"] += 1
        b["net"] += r["pnl"]
    return {
        "trades": n,
        "wins": sum(1 for p in pnl if p > 0),
        "win_rate": round(sum(1 for p in pnl if p > 0) / n, 4),
        "net": str(net.quantize(Decimal("0.01"))),
        "pf": round(pf, 4),
        "avg_c": round(float(net) / (n * QTY), 3),
        "fees": str(fees.quantize(Decimal("0.01"))),
        "max_side_cost": round(max(r["cost"] for r in rows), 4),
        "crossed_fills": sum(int(r["crossed"]) for r in rows),
        "reasons": reasons,
        "reasons_net": {k: str(v.quantize(Decimal("0.01"))) for k, v in reasons_net.items()},
        "lanes": {k: {"trades": v["trades"], "net": str(v["net"].quantize(Decimal("0.01")))}
                  for k, v in sorted(lanes.items())},
    }


# ------------------------------------------------------------------ candidates
def candidate_grid() -> list[dict]:
    out: list[dict] = []
    for hurdle in A_HURDLES:
        for tp in A_TP_CS:
            out.append({"mechanism": "A", "name": f"A_tp{tp:.2f}_h{hurdle:.2f}",
                        "hurdle": hurdle, "exit_mode": "maker_tp", "tp_c": tp,
                        "barrier_s": A_BARRIER_S, "stop_c": A_STOP_C,
                        "stop_confirm_s": A_STOP_CONFIRM_S})
    out.append({"mechanism": "A", "name": f"A_hold_h{A_HOLD_HURDLE:.2f}",
                "hurdle": A_HOLD_HURDLE, "exit_mode": "hold_to_settlement",
                "tp_c": 0.0, "barrier_s": 0.0, "stop_c": 0.0, "stop_confirm_s": 0.0})
    out.append({"mechanism": "B", "name": "B_hold", "hurdle": B_HURDLE,
                "exit_mode": "hold_to_settlement", "tp_c": 0.0, "barrier_s": 0.0,
                "stop_c": 0.0, "stop_confirm_s": 0.0})
    for tp in B_TP_CS:
        out.append({"mechanism": "B", "name": f"B_tp{tp:.2f}", "hurdle": B_HURDLE,
                    "exit_mode": "maker_tp_carry", "tp_c": tp,
                    "barrier_s": B_TP_BARRIER_S, "stop_c": 0.0, "stop_confirm_s": 0.0})
    return out


def selection_mask(pool: dict, cand: dict, slice_name: str) -> np.ndarray:
    mask = pool["is_is"] == (1 if slice_name == "is" else 0)
    mask &= pool["edge"] >= cand["hurdle"] - EPS
    if cand["mechanism"] == "A":
        mask &= (pool["cost"] >= A_COST_LO - EPS) & (pool["cost"] <= CAP + EPS)
        mask &= (np.abs(pool["btc_bps"]) >= A_MIN_BTC_BPS - EPS)
        mask &= (np.sign(pool["btc_bps"]) == np.sign(pool["own_bps"]))
    else:
        mask &= (pool["cost"] >= B_COST_LO - EPS) & (pool["cost"] <= B_COST_HI + EPS)
        mask &= np.abs(pool["own_bps"]) >= B_MIN_MOM_BPS - EPS
    return mask


# --------------------------------------------------------------------- control
def f2_control() -> dict | None:
    """The frozen F2_taker_1s read-slice numbers, for the side-by-side table only."""
    try:
        raw = json.loads(F2_ARTIFACT.read_text())
    except (OSError, ValueError):
        return None
    summary = ((raw.get("variants") or {}).get("F2_taker_1s") or {}).get("summary") or {}
    if not summary:
        return None
    out = {"source": str(F2_ARTIFACT.relative_to(ROOT)),
           "note": "frozen pre-05:00 study; the forward holdout at noon is the verdict"}
    for name in ("is", "oos"):
        s = summary.get(name) or {}
        out[name] = {k: s.get(k) for k in
                     ("trades", "win_rate", "net_pnl", "pf", "payoff_event_ece",
                      "max_side_cost", "crossed_fills")}
    return out


# ----------------------------------------------------------------------- main
def main() -> int:
    t0 = time.time()
    btc_full = dict(np.load(ROOT / "data" / "evolve-cache" / "index_BRTI.npz"))
    btc_index = index_before(btc_full)
    del btc_full
    grid = candidate_grid()
    is_rows: dict[str, list[dict]] = {c["name"]: [] for c in grid}
    pools: dict[str, tuple[dict, dict]] = {}
    lanes: dict[str, Lane] = {}
    max_ts = 0.0
    is_windows = oos_windows = 0
    is_span: list[float] = []
    oos_span: list[float] = []

    for name in LANES:
        _log(f"{name}: loading pre-holdout tape")
        lane = Lane(name, max_close=FORWARD_OOS_START_TS)
        lanes[name] = lane
        if min(lane.meta.values(), key=lambda m: m["close"])["close"] >= FORWARD_OOS_START_TS:
            raise RuntimeError("no pre-holdout windows")
        pool_a, pool_b = build_pools(lane, btc_index)
        if pool_a is None or pool_b is None:
            raise RuntimeError(f"{name}: no index or no rows")
        pools[name] = (pool_a, pool_b)
        is_windows += len(lane.is_closes)
        oos_windows += len(lane.oos_closes)
        is_span.extend(lane.is_closes)
        oos_span.extend(lane.oos_closes)
        for pool in (pool_a, pool_b):
            top = float(pool["ts"].max())
            if top >= FORWARD_OOS_START_TS:
                raise RuntimeError(f"{name}: signal at {top} is inside the holdout")
            max_ts = max(max_ts, top)
        _log(f"{name}: {len(lane.meta)} windows, {len(pool_a['ts'])} A rows, "
             f"{len(pool_b['ts'])} B rows")
        # In-sample only: every candidate is priced on the same IS slice.
        for cand in grid:
            pool = pool_a if cand["mechanism"] == "A" else pool_b
            idxs = pick(pool, selection_mask(pool, cand, "is"))
            if not idxs:
                continue
            is_rows[cand["name"]].extend(simulate(
                lane, pool, idxs, exit_mode=cand["exit_mode"], tp_c=cand["tp_c"],
                barrier_s=cand["barrier_s"], stop_c=cand["stop_c"],
                stop_confirm_s=cand["stop_confirm_s"]))

    if max_ts >= FORWARD_OOS_START_TS:
        raise RuntimeError(f"read a row at {max_ts}, at/after the holdout start")

    candidates: list[dict] = []
    selected: dict[str, dict | None] = {"A": None, "B": None}
    for cand in grid:
        s = summarize(is_rows[cand["name"]])
        candidates.append({"name": cand["name"], "mechanism": cand["mechanism"],
                           "params": {k: v for k, v in cand.items()
                                      if k not in ("name", "mechanism")},
                           "is": s})
        eligible = int(s["trades"]) >= 20 and Decimal(s["net"]) > 0
        if eligible and (selected[cand["mechanism"]] is None
                         or (s["pf"], Decimal(s["net"]))
                         > (selected[cand["mechanism"]]["is"]["pf"],
                            Decimal(selected[cand["mechanism"]]["is"]["net"]))):
            selected[cand["mechanism"]] = {"name": cand["name"], "params": cand,
                                           "is": s}

    # The best IS candidate per mechanism is kept as a diagnostic even when it
    # misses the 20-trade floor. It is never promoted and its OOS slice is not read.
    best_is: dict[str, dict | None] = {"A": None, "B": None}
    for cand in grid:
        s = summarize(is_rows[cand["name"]])
        if int(s["trades"]) == 0:
            continue
        cur = best_is[cand["mechanism"]]
        if cur is None or (s["pf"], Decimal(s["net"])) > (cur["is"]["pf"],
                                                           Decimal(cur["is"]["net"])):
            best_is[cand["mechanism"]] = {"name": cand["name"], "is": s}

    # The frozen OOS slice is scored once, and only for the IS-selected candidate.
    oos_result: dict[str, dict | None] = {"A": None, "B": None}
    for mech in ("A", "B"):
        chosen = selected[mech]
        if chosen is None:
            continue
        cand = chosen["params"]
        rows: list[dict] = []
        for name, lane in lanes.items():
            pool = pools[name][0 if mech == "A" else 1]
            idxs = pick(pool, selection_mask(pool, cand, "oos"))
            if not idxs:
                continue
            rows.extend(simulate(lane, pool, idxs, exit_mode=cand["exit_mode"],
                                 tp_c=cand["tp_c"], barrier_s=cand["barrier_s"],
                                 stop_c=cand["stop_c"],
                                 stop_confirm_s=cand["stop_confirm_s"]))
        oos_result[mech] = {"name": cand["name"], "oos": summarize(rows),
                            "trades": [{k: (str(v) if isinstance(v, Decimal) else v)
                                        for k, v in r.items()} for r in rows]}

    is_trades = {mech: ([] if selected[mech] is None else [
        {k: (str(v) if isinstance(v, Decimal) else v) for k, v in r.items()}
        for r in is_rows[selected[mech]["name"]]]) for mech in ("A", "B")}

    payload = {
        "generated_ts": time.time(),
        "generated_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_s": round(time.time() - t0, 1),
        "guard": {
            "holdout_start_ts": FORWARD_OOS_START_TS,
            "holdout_start_local": "2026-11-17 05:00:00 EST",
            "rule": "no row at/after holdout_start_ts enters a pool, fit or score",
            "max_ts_used_signal": round(max_ts, 3),
            "max_ts_below_holdout": max_ts < FORWARD_OOS_START_TS,
        },
        "split": {
            "rule": "chronological 60/40 by settled window close, within ts < holdout",
            "is_windows_per_lane": is_windows // len(LANES),
            "oos_windows_per_lane": oos_windows // len(LANES),
            "is_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(min(is_span))),
            "is_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(max(is_span))),
            "oos_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(min(oos_span))),
            "oos_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(max(oos_span))),
        },
        "physics": {
            "cap": CAP, "min_fill": MIN_FILL, "qty": QTY, "lag_s": LAG_S,
            "settlement_window_s": WINDOW_S, "sigma_lookback_s": 300.0,
            "fair_value_vol": "sigma*sqrt(T/3) inside the final minute, sigma*sqrt(T) before it",
            "entry": "taker at the T+1s quote, exact Decimal taker_fee",
            "passive": ">=50 contracts printed at/through the limit",
            "no_entry_secs": [SECS_MIN, SECS_MAX], "cooldown_s": COOLDOWN_S,
            "per_window": PER_WINDOW,
        },
        "mechanisms": {
            "A": {
                "name": "fv_residual_snipe",
                "trigger": f"BTC 3s impulse >= {A_MIN_BTC_BPS:g}bp aligned with own 3s move; "
                           f"fair_p_side - ask_1s >= hurdle; cost in [{A_COST_LO},{CAP}]",
                "exits": f"(a) +6c/+8c maker TP, {A_BARRIER_S:g}s barrier, "
                         f"{A_STOP_CONFIRM_S:g}s confirmed {A_STOP_C:.2f} stop; "
                         f"(b) hold to settlement at hurdle >= {A_HOLD_HURDLE:.2f}",
                "selected": (None if selected["A"] is None else
                             {"name": selected["A"]["name"], "is": selected["A"]["is"],
                              "params": {k: v for k, v in selected["A"]["params"].items()
                                         if k not in ("name", "mechanism")},
                              "is_trades": is_trades["A"]}),
                "best_is_diagnostic": (None if best_is["A"] is None else
                                       {"name": best_is["A"]["name"],
                                        "is": best_is["A"]["is"],
                                        "note": "IS only; missed the 20-trade floor or "
                                                "lost to the selected candidate",
                                        "is_trades": [
                                            {k: (str(v) if isinstance(v, Decimal) else v)
                                             for k, v in r.items()}
                                            for r in is_rows[best_is["A"]["name"]]]}),
                "oos": oos_result["A"],
            },
            "B": {
                "name": "underdog_convex",
                "trigger": f"5s own-index momentum >= {B_MIN_MOM_BPS:g}bp; "
                           f"fair_p_side - ask_1s >= {B_HURDLE:.2f}; "
                           f"cost in [{B_COST_LO},{B_COST_HI}]",
                "exits": "hold to settlement, or resting +15c/+20c maker TP that "
                         "carries to settlement when unfilled (no taker stop)",
                "selected": (None if selected["B"] is None else
                             {"name": selected["B"]["name"], "is": selected["B"]["is"],
                              "params": {k: v for k, v in selected["B"]["params"].items()
                                         if k not in ("name", "mechanism")},
                              "is_trades": is_trades["B"]}),
                "best_is_diagnostic": (None if best_is["B"] is None else
                                       {"name": best_is["B"]["name"],
                                        "is": best_is["B"]["is"],
                                        "note": "IS only; missed the 20-trade floor or "
                                                "lost to the selected candidate",
                                        "is_trades": [
                                            {k: (str(v) if isinstance(v, Decimal) else v)
                                             for k, v in r.items()}
                                            for r in is_rows[best_is["B"]["name"]]]}),
                "oos": oos_result["B"],
            },
        },
        "candidates_is": candidates,
        "selection": {
            "rule": "at most one per mechanism on IS only: >=20 IS trades, IS net > 0, "
                    "highest IS PF; OOS scored once",
            "oos_scored_once": True,
        },
        "f2_control": f2_control(),
        "tape": {"lanes": list(LANES),
                 "book_rows_loaded": {n: int(len(l.ts)) for n, l in lanes.items()},
                 "pools": {n: {"A": int(len(p[0]["ts"])), "B": int(len(p[1]["ts"]))}
                           for n, p in pools.items()}},
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str))
    _print_table(payload)
    print(f"\nwrote {OUT}")
    return 0


def _print_table(payload: dict) -> None:
    def line(label: str, s: dict) -> str:
        n = int(s["trades"] or 0)
        per = (float(s["net"]) / n) if n else 0.0
        return (f"{label:<26} {n:>6} {s['win_rate'] * 100:>7.1f} "
                f"{s['net']:>10} {s['pf']:>6.2f} {per:>7.2f} "
                f"{s['fees']:>8} {s['max_side_cost']:>6.2f} {s['crossed_fills']:>4}")

    print(f"\n{'candidate':<26} {'n':>6} {'win%':>7} {'net$':>10} {'PF':>6} "
          f"{'$/tr':>7} {'fees$':>8} {'max$':>6} {'x':>4}")
    print("-" * 90)
    for mech in ("A", "B"):
        block = payload["mechanisms"][mech]
        chosen = block["selected"]
        if chosen is None:
            print(f"{block['name']:<26} no candidate cleared the 20-trade IS floor "
                  f"with positive IS net")
            best = block.get("best_is_diagnostic")
            if best is not None:
                print(line(f"{mech}: {best['name']} IS(diag)", best["is"]))
            continue
        print(line(f"{mech}: {chosen['name']} IS", chosen["is"]))
        oos = (block.get("oos") or {}).get("oos")
        if oos:
            print(line(f"{mech}: {chosen['name']} OOS", oos))
    f2 = payload.get("f2_control")
    if f2:
        for name in ("is", "oos"):
            s = f2.get(name) or {}
            print(line(f"F2_taker_1s {name.upper()}*", {
                "trades": s.get("trades") or 0,
                "win_rate": s.get("win_rate") or 0.0,
                "net": f"{float(s.get('net_pnl') or 0):.2f}",
                "pf": s.get("pf") or 0.0,
                "avg_c": 0.0, "fees": "n/a", "max_side_cost": s.get("max_side_cost") or 0.0,
                "crossed_fills": s.get("crossed_fills") or 0,
            }))
    print("* F2_taker_1s row is the existing frozen pre-05:00 study, not re-scored here.")


if __name__ == "__main__":
    raise SystemExit(main())
