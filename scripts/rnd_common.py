"""Shared plumbing for the rnd/alpha-evolve tracks: one cache, one split, one gate.

Every track imports this so that the split, the summary and the promotion gate are
the same code for all of them, committed before any track was run.

  * Cache: the frozen, hard-linked snapshot in data/frozen-cache (SHA256SUMS.txt).
  * Split: one global chronological cut at 60% of the distinct window close times
    across all four lanes, so a window is IS or OOS for every lane at once.
  * Gate: the strict promotion bar, applied to the OOS summary of one champion.
  * OOS ledger: `score_oos_once` appends every OOS touch to data/oos-ledger.jsonl and
    refuses a second scoring of the same family, so OOS cannot be used to choose.
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openkal import features  # noqa: E402
from openkal import sim_core as S  # noqa: E402
from scripts import alpha_discover as AD  # noqa: E402

CACHE = Path(os.environ.get("RND_CACHE", ROOT / "data" / "frozen-cache"))
AD.CACHE = CACHE
LANES = ("KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M")
STAKE = Decimal("65.58")
LEDGER = ROOT / "data" / "oos-ledger.jsonl"

GATE = {
    "oos_net_min": 0.0,          # strictly above
    "oos_pf_min": 1.25,
    "oos_payoff_ece_max": 0.08,
    "oos_trades_min": 25,
    "profitable_lanes_min": 2,
    "max_entry_cost": 0.62,
    "crossed_fills": 0,
}


def load_lanes(names=LANES) -> dict[str, AD.Lane]:
    return {n: AD.Lane(n) for n in names}


def global_cut(lanes: dict[str, AD.Lane]) -> float:
    """First OOS window close: 60% of the distinct closes across every lane."""
    closes = sorted({m["close"] for lane in lanes.values() for m in lane.meta.values()})
    return closes[int(len(closes) * 0.6)]


def slice_of(close: float, cut: float) -> str:
    return "is" if close < cut else "oos"


def tape(lane: AD.Lane, tid: int) -> S.Tape:
    m = lane.meta[lane.tickers[tid]]
    return S.Tape.from_lane(lane, tid, m["close"], int(m["y"]))


def trade_row(tr: S.Trade, *, lane: str, ticker: str, close: float, slice_name: str,
              feat: list[float] | None = None, **extra) -> dict:
    """A trade as a flat JSON-safe row; `entry` is the held side's contract cost."""
    fee_in = sum((leg.fee for leg in tr.legs if leg.role == "entry"), Decimal(0))
    row = {
        "lane": lane, "ticker": ticker, "close": close, "slice": slice_name,
        "side": tr.side, "entry_style": tr.entry_style, "qty": tr.qty,
        "trigger_ts": tr.trigger_ts, "order_ts": tr.order_ts, "exit_ts": tr.exit_ts,
        "entry": str(tr.entry), "exit_value": str(tr.exit_value), "reason": tr.reason,
        "pnl": str(tr.pnl), "fees": str(tr.fees), "fee_entry": str(fee_in),
        "fee_exit": str(tr.fees - fee_in), "crossed": tr.crossed, "requotes": tr.requotes,
        "feat": feat,
    }
    row.update(extra)
    return row


def summarize(rows: list[dict], skips: dict[str, int] | None = None) -> dict:
    """Net, PF, lanes, windows, reasons and fees from trade rows (Decimal sums)."""
    n = len(rows)
    pnl = [Decimal(r["pnl"]) for r in rows]
    net = sum(pnl, Decimal(0))
    gw = sum((p for p in pnl if p > 0), Decimal(0))
    gl = -sum((p for p in pnl if p <= 0), Decimal(0))
    by_lane: dict[str, dict] = {}
    by_window: dict[str, Decimal] = {}
    by_reason: dict[str, dict] = {}
    for r, p in zip(rows, pnl):
        d = by_lane.setdefault(r["lane"], {"trades": 0, "net": Decimal(0), "wins": 0})
        d["trades"] += 1
        d["net"] += p
        d["wins"] += int(p > 0)
        k = str(r["close"])
        by_window[k] = by_window.get(k, Decimal(0)) + p
        q = by_reason.setdefault(r["reason"], {"trades": 0, "net": Decimal(0)})
        q["trades"] += 1
        q["net"] += p
    best_lane = max((d["net"] for d in by_lane.values()), default=Decimal(0))
    best_window = max(by_window.values(), default=Decimal(0))
    return {
        "trades": n,
        "wins": sum(1 for p in pnl if p > 0),
        "losses": sum(1 for p in pnl if p <= 0),
        "win_rate": round(sum(1 for p in pnl if p > 0) / n, 4) if n else 0.0,
        "net": float(net),
        "pf": (float(gw / gl) if gl > 0 else (math.inf if gw > 0 else 0.0)),
        "avg_pnl": float(net / n) if n else 0.0,
        "max_entry_cost": max((float(r["entry"]) for r in rows), default=0.0),
        "fees_entry": float(sum((Decimal(r["fee_entry"]) for r in rows), Decimal(0))),
        "fees_exit": float(sum((Decimal(r["fee_exit"]) for r in rows), Decimal(0))),
        "crossed_fills": sum(int(r["crossed"]) for r in rows),
        "requotes": sum(int(r["requotes"]) for r in rows),
        "profitable_lanes": sum(1 for d in by_lane.values() if d["net"] > 0),
        "net_ex_best_lane": float(net - best_lane) if by_lane else 0.0,
        "net_ex_xrp": float(net - by_lane.get("KXXRP15M", {"net": Decimal(0)})["net"]),
        "net_ex_best_window": float(net - best_window) if by_window else 0.0,
        "windows": len(by_window),
        "by_lane": {k: {"trades": v["trades"], "wins": v["wins"], "net": float(v["net"])}
                    for k, v in sorted(by_lane.items())},
        "by_reason": {k: {"trades": v["trades"], "net": float(v["net"])}
                      for k, v in sorted(by_reason.items())},
        "skips": dict(sorted((skips or {}).items())),
    }


def fit_payoff_model(is_rows: list[dict]):
    """Ridge logistic P(pnl > 0) on IS feature rows; None below 20 rows."""
    from scripts.f2_structural import fit_logistic
    x = np.array([r["feat"] for r in is_rows], dtype=np.float64)
    if len(x) < 20 or x.ndim != 2:
        return None
    y = np.array([Decimal(r["pnl"]) > 0 for r in is_rows], dtype=np.float64)
    return fit_logistic(x, y)


def payoff_ece(model, rows: list[dict]) -> float | None:
    """ECE of the IS-fitted payoff model on `rows`. None when no model exists."""
    from scripts.f2_structural import apply_logistic
    if model is None or not rows:
        return None
    w, norm = model
    p = apply_logistic(np.array([r["feat"] for r in rows], dtype=np.float64), w, norm)
    won = [1 if Decimal(r["pnl"]) > 0 else 0 for r in rows]
    return round(features.expected_calibration_error(list(zip(p.tolist(), won))), 4)


def promotion_gate(oos: dict, ece: float | None) -> dict:
    """The strict bar. Every check must hold; nothing here is tunable per track."""
    checks = {
        "oos_net_positive": oos["net"] > GATE["oos_net_min"],
        "oos_pf_at_least_1_25": oos["pf"] >= GATE["oos_pf_min"],
        "oos_payoff_ece_at_most_0_08": ece is not None and ece <= GATE["oos_payoff_ece_max"],
        "oos_trades_at_least_25": oos["trades"] >= GATE["oos_trades_min"],
        "two_profitable_lanes": oos["profitable_lanes"] >= GATE["profitable_lanes_min"],
        "not_carried_by_one_lane": oos["net_ex_best_lane"] > 0,
        "net_ex_xrp_positive": oos["net_ex_xrp"] > 0,
        "not_carried_by_one_window": oos["net_ex_best_window"] > 0,
        "zero_crossed_fills": oos["crossed_fills"] == GATE["crossed_fills"],
        "max_entry_cost_at_most_0_62": oos["max_entry_cost"] <= GATE["max_entry_cost"] + 1e-9,
    }
    checks["pass"] = all(checks.values())
    return checks


def score_oos_once(family: str, config: dict, scorer, *, rescore: bool = False, before_record=None) -> dict:
    """Score one frozen champion on OOS, once per family, and log the touch.

    `scorer()` must return (oos_rows, oos_skips, payoff_model). The ledger is
    append-only. A "started" line is written before the scorer runs, so a crash
    mid-score still counts as a look. A family with any prior line is refused unless
    this is a rescore: `rescore=True`, or RND_OOS_RESCORE set to the reason, which is
    recorded with the entry.
    `before_record`, if supplied, can durably stage the calculated receipt before the scored append.
    """
    if os.environ.get("RND_OOS_UNSEAL") != "lead":
        raise RuntimeError("OOS is sealed: only the lead scores a champion (RND_OOS_UNSEAL=lead)")
    reason = os.environ.get("RND_OOS_RESCORE", "").strip() or None
    rescore = rescore or reason is not None
    prior = []
    if LEDGER.exists():
        prior = [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]
    if any(p["family"] == family for p in prior) and not rescore:
        raise RuntimeError(f"{family} was already scored on OOS; refusing a second look")
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as fh:
        fh.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "family": family,
                             "status": "started", "config": config, "rescore": rescore,
                             "rescore_reason": reason}, default=_json_default) + "\n")
    rows, skips, model = scorer()
    oos = summarize(rows, skips)
    ece = payoff_ece(model, rows)
    verdict = promotion_gate(oos, ece)
    entry = {"ts": datetime.now(timezone.utc).isoformat(), "family": family,
              "status": "scored", "config": config, "rescore": rescore,
              "rescore_reason": reason, "oos": oos, "payoff_ece": ece,
              "gate": verdict, "cache": str(CACHE)}
    if before_record is not None:
        before_record(entry)
    with LEDGER.open("a") as fh:
        fh.write(json.dumps(entry, default=_json_default) + "\n")
    return entry


def _json_default(o):
    if isinstance(o, Decimal):
        return str(o)
    if isinstance(o, float) and math.isinf(o):
        return "inf"
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=_json_default))
