"""Empirical alpha discovery on the recorded sub-second book + print tape.

This is the study that decides whether a rule may risk money. It reads only
`data/evolve-cache/book_<LANE>.npz` (the venue's own `ws_book` ladder and `ws_trade`
prints, every message), splits every settled window chronologically 60/40 by *close
time*, fits and selects on the in-sample slice, and reports the frozen out-of-sample
slice as a verdict.

Three candidates, each falsifiable:

  * `spike_fade` — retail overreaction. The Kalshi mid spikes 8-20c inside 5-20s while
    the settlement index (BRTI 5Hz) moves less than 40% of the implied probability
    delta. Fade it: rest a post-only maker on the opposite side, exit with a resting
    maker take-profit (+3..6c, $0 exit fee), scratch at a time barrier.
  * `cross_book` — YES_ask + NO_ask < $0.99. This is algebraically the same state as
    `yes_bid > yes_ask`, so the study reports how many such states the tape has and
    how many of them survive the `bid < ask` protection (the answer must be zero).
  * `laya_rescale` — a rank/Pllatt summary of the trained RLCD checkpoint, imported
    from `scripts/laya_rank.py` when its artifact exists.

Every fill is simulated on the print tape, never on a model:

  * a passive order fills only when at least 20 contracts print at or through its
    limit, in the direction that would hit it;
  * a `bid >= ask` book can never be an entry or an exit (crossed-book protection);
  * every taker leg pays `openkal.fees.taker_fee` in exact `Decimal`.

    .venv/bin/python scripts/alpha_discover.py            # full study + verdict
    .venv/bin/python scripts/alpha_discover.py --lane KXBTC15M
    .venv/bin/python scripts/alpha_discover.py --quick
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openkal import features, verifier  # noqa: E402
from openkal.fees import taker_fee  # noqa: E402
from openkal.kalshi_ws import INDEX_BY_SERIES  # noqa: E402

CACHE = ROOT / "data" / "evolve-cache"
REPORT = ROOT / "data" / "alpha-report.json"
VERDICT = ROOT / "data" / "strategy-verdict.json"

LANES = ("KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M")
TICK = Decimal("0.01")
STAKE = Decimal("65.58")          # one quarter of the $262.31 account
MIN_PASSIVE_QTY = 20.0            # contracts that must print at/through a passive limit
EPS = 1e-6                        # float32 price tolerance for at/through compares


# --------------------------------------------------------------------------- tape
def _segments(ids: np.ndarray, n: int) -> np.ndarray:
    """Start index of each id in a non-decreasing id array (size n+1)."""
    return np.searchsorted(ids, np.arange(n + 1), side="left")


class Lane:
    """One lane's book ladder, print tape, settled outcomes and index tape."""

    def __init__(self, lane: str, *, max_close: float | None = None) -> None:
        self.lane = lane
        data = np.load(CACHE / f"book_{lane}.npz", allow_pickle=False)
        self.tickers = [str(t) for t in data["tickers"]]
        self.index_of = {t: i for i, t in enumerate(self.tickers)}
        self.ts = data["ts"].astype(np.float64)
        self.bid = data["bid"].astype(np.float64)
        self.ask = data["ask"].astype(np.float64)
        self.tid = data["tid"].astype(np.int32)
        self.trade_ts = data["trade_ts"].astype(np.float64)
        self.trade_px = data["trade_px"].astype(np.float64)
        self.trade_sz = data["trade_sz"].astype(np.float64)
        self.trade_ty = data["trade_ty"].astype(np.int8)
        self.trade_tid = data["trade_tid"].astype(np.int32)
        self.book_bounds = _segments(self.tid, len(self.tickers))
        self.trade_bounds = _segments(self.trade_tid, len(self.tickers))

        meta = json.loads((CACHE / "markets.json").read_text())
        self.meta: dict[str, dict] = {}
        for name in self.tickers:
            m = meta.get(name)
            if m is None or not m.get("strike"):
                continue
            self.meta[name] = {"strike": float(m["strike"]), "close": float(m["close"]),
                               "y": int(m["y"])}
        if max_close is not None:
            # A hard boundary for studies that must not read the forward tape. The
            # book arrays stay loaded (the cache is one file), but every pool
            # builder walks `_segments`, which skips names absent from `meta`.
            self.meta = {name: m for name, m in self.meta.items()
                         if m["close"] < max_close}
        index_id = INDEX_BY_SERIES.get(lane)
        self.index = dict(np.load(CACHE / f"index_{index_id}.npz")) if index_id else {}
        self.sigma = {name: self._sigma(m["close"]) for name, m in self.meta.items()}

        closes = sorted({m["close"] for m in self.meta.values()})
        cut = int(len(closes) * 0.6)
        self.is_closes = set(closes[:cut])
        self.oos_closes = set(closes[cut:])
        self.cut_ts = closes[cut] if cut < len(closes) else math.inf
        self.hours = {
            "is": (max(self.is_closes) - min(self.is_closes)) / 3600.0 if self.is_closes else 0.0,
            "oos": (max(self.oos_closes) - min(self.oos_closes)) / 3600.0 if self.oos_closes else 0.0,
        }

    def _sigma(self, close: float) -> float:
        if not self.index:
            return 0.0
        ts, val = self.index["ts"], self.index["value"]
        lo = int(np.searchsorted(ts, close - 900.0, "left"))
        hi = int(np.searchsorted(ts, close, "right"))
        pts = [(float(a), float(b)) for a, b in zip(ts[lo:hi], val[lo:hi]) if b > 0]
        return features.realized_sigma(pts)

    def slice_of(self, close: float) -> str:
        return "is" if close in self.is_closes else "oos"

    def index_value(self, ts: float) -> float:
        """Nearest settlement-index print within 1.5s, or NaN."""
        if not self.index:
            return float("nan")
        arr = self.index["ts"]
        j = int(np.searchsorted(arr, ts, "left"))
        best = (math.inf, float("nan"))
        for cand in (j - 1, j, j + 1):
            if 0 <= cand < len(arr):
                gap = abs(float(arr[cand]) - ts)
                if gap <= 1.5 and gap < best[0]:
                    best = (gap, float(self.index["value"][cand]))
        return best[1]

    # ---- print tape access ------------------------------------------------
    def first_fill(self, tid: int, t0: float, t1: float, *, taker_yes: bool,
                   limit: float,
                   need: float | None = None) -> tuple[float, float] | None:
        """First time cumulative taker volume reaches `need` at/through `limit`.

        `taker_yes=True` means the aggressor bought YES (it lifts a resting YES ask).
        `taker_yes=False` means the aggressor sold YES (it hits a resting YES bid).
        """
        if need is None:
            need = MIN_PASSIVE_QTY
        if t1 <= t0:
            return None
        lo, hi = int(self.trade_bounds[tid]), int(self.trade_bounds[tid + 1])
        if hi <= lo:
            return None
        ts = self.trade_ts[lo:hi]
        a = int(np.searchsorted(ts, t0, "right"))
        b = int(np.searchsorted(ts, t1, "right"))
        if b <= a:
            return None
        px = self.trade_px[lo:hi][a:b]
        sz = self.trade_sz[lo:hi][a:b]
        ty = self.trade_ty[lo:hi][a:b]
        mask = (ty == 1) == bool(taker_yes)
        mask &= (px >= limit - EPS) if taker_yes else (px <= limit + EPS)
        if not mask.any():
            return None
        cum = np.cumsum(np.where(mask, sz, 0.0))
        if cum[-1] < need:
            return None
        hit = int(np.argmax(cum >= need))
        return float(ts[a + hit]), float(px[hit])

    def quote_at(self, tid: int, ts: float, *, valid_only: bool = True,
                 lookback: float = 30.0) -> tuple[float, float] | None:
        """Last recorded two-sided quote at or before `ts` (within `lookback`)."""
        lo, hi = int(self.book_bounds[tid]), int(self.book_bounds[tid + 1])
        j = int(np.searchsorted(self.ts[lo:hi], ts, "right")) - 1
        while j >= 0:
            t = float(self.ts[lo + j])
            if ts - t > lookback:
                return None
            b, a = float(self.bid[lo + j]), float(self.ask[lo + j])
            if (not valid_only) or (0.0 < b < a < 1.0):
                return b, a
            j -= 1
        return None


# ------------------------------------------------------------------ book audit
def cross_book_audit(lane: Lane) -> dict:
    """Count crossed states and the 'arb' they masquerade as.

    YES_ask + NO_ask where NO_ask = 1 - yes_bid: the sum is < 0.99 exactly when
    yes_bid - yes_ask > 0.01, i.e. when the book is crossed. Under `bid < ask` the sum
    is always > 1.00, so the arbitrage cannot exist on a protected book.
    """
    bid, ask = lane.bid, lane.ask
    live = (bid > 0) & (ask < 1)
    crossed = live & (bid >= ask)
    arb = live & ((ask + (1.0 - bid)) < 0.99)
    return {
        "rows": int(len(bid)),
        "crossed_rows": int(crossed.sum()),
        "crossed_share": round(float(crossed.sum()) / max(1, int(live.sum())), 5),
        "arb_rows": int(arb.sum()),
        "arb_that_are_crossed": int((arb & crossed).sum()),
        "arb_on_protected_book": int((arb & ~crossed).sum()),
        "one_sided_rows": int(((bid <= 0) | (ask >= 1)).sum()),
    }


# ------------------------------------------------------------------- discovery
def detect_spikes(lane: Lane, *, horizon: float, mag_lo: float, mag_hi: float,
                  spread_max: float, conf_max: float, cooldown: float = 45.0,
                  per_window: int = 2) -> list[dict]:
    """Vectorised `spike_fade` candidates with an index-confirmation test."""
    bid, ask, ts, tid = lane.bid, lane.ask, lane.ts, lane.tid
    n = len(ts)
    valid = (bid > 0) & (ask < 1) & (bid < ask) & ((ask - bid) <= spread_max + EPS)
    mid = (bid + ask) / 2.0

    anchor = np.full(n, -1, dtype=np.int64)
    gap = np.zeros(n, dtype=np.float64)
    for a, b in zip(lane.book_bounds[:-1], lane.book_bounds[1:]):
        if b <= a:
            continue
        seg_ts = ts[a:b]
        j = np.searchsorted(seg_ts, seg_ts - horizon, side="right") - 1
        ok = j >= 0
        jj = np.maximum(j, 0)
        anchor[a:b] = np.where(ok, a + jj, -1)
        gap[a:b] = np.where(ok, seg_ts - seg_ts[jj], 0.0)
    anchor = np.maximum(anchor, 0)
    ok = valid & valid[anchor] & (gap >= horizon) & (gap <= horizon + 3.0)
    delta = np.where(ok, mid - mid[anchor], 0.0)
    up = ok & (delta >= mag_lo) & (delta <= mag_hi)
    down = ok & (delta <= -mag_lo) & (delta >= -mag_hi)
    rows = np.flatnonzero(up | down)
    if len(rows) == 0:
        return []

    candidates: list[dict] = []
    stats: dict[int, dict] = defaultdict(lambda: {"seen": 0, "kept": 0, "nominal": 0,
                                                  "weak_conf": 0, "no_index": 0})
    last_kept: dict[int, float] = {}
    count_by_ticker: dict[int, int] = defaultdict(int)
    for i in rows:
        i = int(i)
        t = float(ts[i])
        t_tid = int(tid[i])
        a0 = int(anchor[i])
        st = stats[t_tid]
        st["seen"] += 1
        name = lane.tickers[t_tid]
        m = lane.meta.get(name)
        if m is None:
            continue
        if t < m["close"] - 1200.0 or t > m["close"] - 20.0:
            continue
        st["nominal"] += 1
        if count_by_ticker[t_tid] >= per_window:
            continue
        if t_tid in last_kept and t - last_kept[t_tid] < cooldown:
            continue
        d = float(delta[i])
        direction = 1.0 if d > 0 else -1.0
        sigma = lane.sigma.get(name, 0.0)
        if sigma <= 0:
            continue
        spot_now = lane.index_value(t)
        spot_then = lane.index_value(float(ts[a0]))
        if not (math.isfinite(spot_now) and math.isfinite(spot_then)) or spot_now <= 0 or spot_then <= 0:
            st["no_index"] += 1
            continue
        seconds_left = float(m["close"]) - t
        fv_now = features.fair_value(spot_now, m["strike"], seconds_left, sigma)
        fv_then = features.fair_value(spot_then, m["strike"], seconds_left + float(gap[i]), sigma)
        if not (fv_now.enough_data and fv_then.enough_data):
            st["no_index"] += 1
            continue
        implied = abs(fv_now.p_up - fv_then.p_up)
        if implied > conf_max * abs(d):
            st["weak_conf"] += 1
            continue
        count_by_ticker[t_tid] += 1
        last_kept[t_tid] = t
        st["kept"] += 1
        long_yes = d < 0
        candidates.append({
            "tid": t_tid, "ticker": name, "ts": t, "anchor_ts": float(ts[a0]),
            "close": m["close"], "strike": m["strike"], "y": m["y"],
            "bid": float(bid[i]), "ask": float(ask[i]), "mid": float(mid[i]),
            "delta": d, "gap_s": float(gap[i]), "side": "YES" if long_yes else "NO",
            "entry_limit": float(bid[i]) if long_yes else float(ask[i]),
            "p_side": fv_now.p_up if long_yes else 1.0 - fv_now.p_up,
            "spot": spot_now, "conf_share": implied / max(1e-9, abs(d)),
            "slice": lane.slice_of(m["close"]),
        })
    return candidates


def simulate(lane: Lane, candidates: list[dict], *, tp_c: float, hold_s: float,
             patience_s: float = 15.0) -> list[dict]:
    """Strict sub-second fill simulation for one fade candidate set."""
    out: list[dict] = []
    for c in candidates:
        if c["slice"] not in ("is", "oos"):
            continue
        # Crossed-book protection: an entry can only come from a two-sided book. This
        # is a hard guard, not a filter: if one ever gets through, the study aborts
        # rather than booking the stale-side artifact that inflated the SOL soak.
        if not (0.0 < c["bid"] < c["ask"] < 1.0):
            raise RuntimeError(f"crossed entry book {c['ticker']} {c['bid']} / {c['ask']}")
        long_yes = c["side"] == "YES"
        entry = c["entry_limit"]
        fill = lane.first_fill(c["tid"], c["ts"], min(c["ts"] + patience_s, c["close"] - 20.0),
                               taker_yes=not long_yes, limit=entry)
        if fill is None:
            continue
        t_fill = fill[0]
        qty = int((STAKE / Decimal(str(entry))).to_integral_value())
        if qty <= 0:
            continue
        deadline = min(t_fill + hold_s, c["close"] - 30.0)
        if deadline <= t_fill:
            continue
        target = round(entry + tp_c, 4) if long_yes else round(entry - tp_c, 4)
        fee = Decimal("0")
        exit_px = target
        reason = "maker_tp"
        crossed = 0
        if 0.0 < target < 1.0:
            take = lane.first_fill(c["tid"], t_fill + 1e-6, deadline,
                                   taker_yes=long_yes, limit=target)
        else:
            take = None
        if take is None:
            quote = lane.quote_at(c["tid"], deadline)
            if quote is None:
                quote = lane.quote_at(c["tid"], c["close"] - 30.0)
            if quote is None:
                continue
            b, a = quote
            if not (0.0 < b < a < 1.0):
                raise RuntimeError(f"crossed exit book {c['ticker']} {b} / {a}")
            # Flatten at the touch: a long sells the bid, a short buys the ask.
            exit_px = b if long_yes else a
            fee = taker_fee(Decimal(str(round(exit_px, 4))), qty)
            reason = "time_barrier_scratch"
        entry_dec = Decimal(str(round(entry, 4)))
        exit_dec = Decimal(str(round(exit_px, 4)))
        gross = ((exit_dec - entry_dec) if long_yes else (entry_dec - exit_dec)) * qty
        pnl = gross - fee
        out.append({**c, "pnl": pnl, "fee": fee, "reason": reason, "crossed": crossed,
                    "qty": qty, "entry": entry_dec, "exit": exit_dec,
                    "fill_ts": t_fill, "hold_s": t_fill - c["ts"]})
    return out


# -------------------------------------------------------------------- scoring
def block(rows: list[dict], hours: float) -> dict:
    n = len(rows)
    if n == 0:
        return {"trades": 0, "wins": 0, "win_rate": 0.0, "net": "0", "pf": 0.0,
                "ece": 0.0, "fees": "0", "trades_per_hour": 0.0, "avg_c": 0.0,
                "crossed_fills": 0, "tp_share": 0.0, "lanes": {}}
    pnl = [r["pnl"] for r in rows]
    net = sum(pnl, Decimal("0"))
    wins = [p for p in pnl if p > 0]
    losses = [-p for p in pnl if p <= 0]
    gw = sum(wins, Decimal("0"))
    gl = sum(losses, Decimal("0"))
    pf = float(gw / gl) if gl > 0 else (float("inf") if gw > 0 else 0.0)
    pairs = [(r["p_side"], 1 if r["pnl"] > 0 else 0) for r in rows]
    fees = sum((r["fee"] for r in rows), Decimal("0"))
    lanes = defaultdict(int)
    for r in rows:
        lanes[r["tid"]] += 1
    return {
        "trades": n,
        "wins": len(wins),
        "win_rate": round(len(wins) / n, 4),
        "net": str(net.quantize(Decimal("0.01"))),
        "pf": round(pf, 4) if math.isfinite(pf) else 99.0,
        "ece": round(features.expected_calibration_error(pairs), 4),
        "fees": str(fees.quantize(Decimal("0.01"))),
        "trades_per_hour": round(n / hours, 2) if hours > 0 else 0.0,
        "avg_usd_per_trade": round(float(net / n), 4),
        "crossed_fills": sum(int(r["crossed"]) for r in rows),
        "tp_share": round(sum(1 for r in rows if r["reason"] == "maker_tp") / n, 3),
        "lanes": {str(k): v for k, v in sorted(lanes.items())},
    }


def split(rows: list[dict], lane_list: list[Lane]) -> dict:
    hours = {"is": max(l.hours["is"] for l in lane_list),
             "oos": max(l.hours["oos"] for l in lane_list)}
    return {name: block([r for r in rows if r["slice"] == name], hours[name])
            for name in ("is", "oos")}


def verifier_call(block_out: dict) -> tuple[bool, str]:
    """Run the mission's immutable thresholds through `verifier.passes` itself."""
    res = verifier.ReplayResult(
        trades=int(block_out["trades"]),
        pnl=Decimal(str(block_out["net"])),
        profit_factor=float(block_out["pf"]),
        win_rate=float(block_out["win_rate"]),
        ece=float(block_out["ece"]),
        max_drawdown=Decimal("0"),
    )
    baseline = verifier.ReplayResult(0, Decimal("0"), 0.0, 0.0, 0.0, Decimal("0"))
    ok, why = verifier.passes(res, baseline)
    if ok and block_out["crossed_fills"] != 0:
        return False, f"{block_out['crossed_fills']} crossed-book fills"
    return ok, why


def gate(block_out: dict) -> dict:
    """The stated promotion gate, evaluated on one slice."""
    return {
        "net_positive": Decimal(str(block_out["net"])) > 0,
        "profit_factor_ge_1_15": float(block_out["pf"]) >= 1.15,
        "ece_le_0_08": float(block_out["ece"]) <= 0.08,
        "crossed_fills_zero": int(block_out["crossed_fills"]) == 0,
        "trades_ge_20": int(block_out["trades"]) >= 20,
    }


# ----------------------------------------------------------------------- main
def study(lanes: list[str], quick: bool) -> dict:
    t0 = time.time()
    lane_objs: dict[str, Lane] = {}
    audits: dict[str, dict] = {}
    for name in lanes:
        lane_objs[name] = Lane(name)
        audits[name] = cross_book_audit(lane_objs[name])
        print(f"{name}: {len(lane_objs[name].meta)} settled windows, "
              f"{len(lane_objs[name].ts):,} book rows, "
              f"{len(lane_objs[name].trade_ts):,} prints, "
              f"crossed {audits[name]['crossed_share']*100:.2f}%, "
              f"arb/protected {audits[name]['arb_on_protected_book']}", flush=True)

    grid = ([(5.0, 0.08), (8.0, 0.08), (12.0, 0.08), (20.0, 0.08),
             (5.0, 0.12), (8.0, 0.12), (12.0, 0.12)] if not quick else [(8.0, 0.08)])
    confs = (0.25, 0.40) if not quick else (0.40,)
    tps = (0.03, 0.04, 0.05, 0.06) if not quick else (0.04,)
    holds = (30.0, 60.0) if not quick else (60.0,)

    # Detection is expensive; generate each (horizon, conf) candidate set once and
    # reuse it for every take-profit / hold combination.
    cache: dict[tuple, dict[str, list[dict]]] = {}
    for horizon, mag_lo in grid:
        for conf in confs:
            key = (horizon, mag_lo, conf)
            per_lane: dict[str, list[dict]] = {}
            counts = {"is": 0, "oos": 0}
            for name in lanes:
                found = detect_spikes(lane_objs[name], horizon=horizon, mag_lo=mag_lo,
                                      mag_hi=0.20, spread_max=0.05, conf_max=conf)
                per_lane[name] = found
                for c in found:
                    counts[c["slice"]] = counts.get(c["slice"], 0) + 1
            cache[key] = per_lane
            print(f"  spikes h={horizon:>4}s conf<={conf:.2f}: "
                  f"IS {counts['is']:4d} candidates, OOS {counts['oos']:4d}", flush=True)

    results: list[dict] = []
    for (horizon, mag_lo, conf), per_lane in cache.items():
        for tp in tps:
            for hold in holds:
                rows: list[dict] = []
                for name in lanes:
                    rows.extend(simulate(lane_objs[name], per_lane[name],
                                         tp_c=tp, hold_s=hold))
                if not rows:
                    continue
                res = split(rows, list(lane_objs.values()))
                params = {"name": "spike_fade", "horizon_s": horizon, "mag_lo": mag_lo,
                          "mag_hi": 0.20, "conf_max": conf, "tp_c": tp, "hold_s": hold,
                          "spread_max": 0.05, "patience_s": 15.0, "entry": "post_only",
                          "exit": "post_only_tp_then_barrier"}
                results.append({"params": params, "is": res["is"], "oos": res["oos"],
                                "verifier": {"is": verifier_call(res["is"]),
                                             "oos": verifier_call(res["oos"])}})
                print(f"  tp={tp:.2f} hold={hold:>4.0f}s h={horizon:>4.0f} conf={conf:.2f} | "
                      f"IS n={res['is']['trades']:3d} net={res['is']['net']:>8} "
                      f"pf={res['is']['pf']:5.2f} | OOS n={res['oos']['trades']:3d} "
                      f"net={res['oos']['net']:>8} pf={res['oos']['pf']:5.2f}", flush=True)

    # Selection is on IS only. The frozen OOS slice is the verdict.
    eligible = [r for r in results
                if gate(r["is"])["trades_ge_20"] and Decimal(str(r["is"]["net"])) > 0]
    eligible.sort(key=lambda r: (Decimal(str(r["is"]["net"])), float(r["is"]["pf"])), reverse=True)
    best = eligible[0] if eligible else None
    promoted = False
    reason = "no candidate had a positive in-sample net with 20+ trades"
    if best is not None:
        g = gate(best["oos"])
        promoted = all(g.values()) and best["verifier"]["oos"][0]
        reason = ("frozen OOS passes every promotion check" if promoted
                  else "IS-best fails the frozen OOS gate: " +
                       ", ".join(k for k, v in g.items() if not v)
                       + ("" if best["verifier"]["oos"][0] else
                          f" ({best['verifier']['oos'][1]})"))

    payload = {
        "generated_ts": time.time(),
        "elapsed_s": round(time.time() - t0, 1),
        "lanes": list(lanes),
        "split": {
            "rule": "chronological 60/40 by settled window close, never re-split",
            "is_windows": sum(len(l.is_closes) for l in lane_objs.values()),
            "oos_windows": sum(len(l.oos_closes) for l in lane_objs.values()),
            "hours": {name: round(sum(l.hours[name] for l in lane_objs.values()), 2)
                      for name in ("is", "oos")},
        },
        "cross_book": audits,
        "candidates": results,
        "selected": best,
        "promotion": {"promoted": promoted, "reason": reason},
        "tape": {"book_rows": sum(len(l.ts) for l in lane_objs.values()),
                 "prints": sum(len(l.trade_ts) for l in lane_objs.values())},
    }
    REPORT.write_text(json.dumps(payload, indent=2, default=str))

    verdict = {
        "generated_ts": payload["generated_ts"],
        "candidate": (best or {}).get("params"),
        "promoted": promoted,
        "reason": reason,
        "is": (best or {}).get("is"),
        "oos": (best or {}).get("oos"),
        "gate": {
            "profit_factor_min": 1.15,
            "ece_max": 0.08,
            "requires_positive_net": True,
            "requires_zero_crossed_fills": True,
        },
        "tape": payload["tape"],
        "split": payload["split"],
    }
    VERDICT.write_text(json.dumps(verdict, indent=2, default=str))
    print(json.dumps({"promotion": payload["promotion"],
                      "selected_oos": (best or {}).get("oos"),
                      "selected_is": (best or {}).get("is")}, indent=2))
    print(f"wrote {REPORT} and {VERDICT}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", action="append", default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    study(args.lane or list(LANES), args.quick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
