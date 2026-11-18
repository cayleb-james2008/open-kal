#!/usr/bin/env python3
"""Audit the sealed forward fills and same-ticker cached verdicts; never run a replay."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import track2b_calibration_audit as A

CACHE = Path("/home/cayleb/open-kal/data/evolve-cache")
CUTOFF = 1794943521.0
MIN_WINDOWS_PER_LANE = 15


def gate_at(ts, ids, available, allow, tid, t):
    """Latest row for this ticker by trigger, without carrying past a missing verdict."""
    end = int(np.searchsorted(ts, t, side="right"))
    rows = np.flatnonzero(ids[:end] == tid)
    if not len(rows):
        return None, None
    j = rows[-1]
    verdict = bool(allow[j]) if available[j] == 1 and allow[j] in (0, 1) else None
    return verdict, float(ts[j])


def describe(rows, slope):
    p = np.array([r["p_side"] for r in rows])
    y = A.outcomes(rows)
    summary = A.economics(rows)
    if not rows:
        # 0 trades does not measure either win rate or profit factor.
        summary.update(win_rate=None, pf=None, max_entry_cost=None)
    summary.update(raw_p_side_ece=A.metrics(p, y)["ece"], platt_1p_ece=A.metrics(A.platt(p, slope), y)["ece"])
    return summary


def audit_forward(as_of):
    report_path = A.DATA / f"{A.STEM}-forward-holdout.json"
    frozen_markets = json.loads((A.DATA / "frozen-cache/markets.json").read_text())
    payload = (CACHE / "markets.json").read_bytes()
    markets = json.loads(payload)
    historical = json.loads(report_path.read_text())
    horizon = historical["holdout_window"]
    if horizon["reference_frozen_tape_max"] != CUTOFF:
        raise ValueError("prior look uses a different cutoff")
    rows = A.trades("forward")
    if any(not CUTOFF < r["close"] <= horizon["end_close_ts"] or r["ticker"] in frozen_markets for r in rows):
        raise ValueError("forward fill is outside the sealed calendar or is a frozen market")
    ledger = A.DATA / "oos-ledger.jsonl"
    families = ["flow_student_t_settle_forward_holdout", "flow_student_t_settle_forward_layagate"]
    logged = {}
    for line_no, line in enumerate(ledger.read_text().splitlines(), 1):
        entry = json.loads(line)
        if entry["family"] in families and entry["status"] == "scored":
            if entry["family"] in logged:
                raise ValueError("more than one scored record for a forward arm")
            logged[entry["family"]] = (line_no, entry)
    for family, key in zip(families, ("unfiltered_forward_holdout", "layagate_filtered_forward_holdout")):
        if logged[family][1]["oos"] != historical[key]:
            raise ValueError(f"{family}: prior forward summary does not match its ledger")

    counts, recorded_counts, gate_rows = {}, {}, []
    for lane in A.RC.LANES:
        live = {t: m["close"] for t, m in markets.items()
                if t.startswith(lane + "-") and t not in frozen_markets
                and CUTOFF < m["close"] <= as_of and m.get("y") in (0, 1) and m.get("strike", 0) > 0}
        counts[lane] = len(set(live.values()))
        recorded_counts[lane] = len({c for c in live.values() if c <= horizon["end_close_ts"]})
        if min(counts[lane], recorded_counts[lane]) < MIN_WINDOWS_PER_LANE:
            raise ValueError(f"{lane}: insufficient unseen settled windows for the requested audit")
        with (CACHE / f"book_{lane}.npz").open("rb") as f, np.load(f, allow_pickle=False) as data:
            names = {str(name): i for i, name in enumerate(data["tickers"])}
            ts, ids, av, allow = (data[k] for k in ("tick_ts", "tick_tid", "gate_available", "gate_allow"))
        if not (len(ts) == len(ids) == len(av) == len(allow)) or np.any(ts[:-1] > ts[1:]):
            raise ValueError(f"{lane}: gate columns are misaligned or out of time order")
        for r in (row for row in rows if row["lane"] == lane):
            if live.get(r["ticker"]) != r["close"]:
                raise ValueError(f"{r['ticker']}: forward settlement metadata no longer matches")
            gate, ts_hit = gate_at(ts, ids, av, allow, names[r["ticker"]], r["trigger_ts"])
            gate_rows.append({"ticker": r["ticker"], "trigger_ts": r["trigger_ts"], "gate_ts": ts_hit,
                              "gate_allow": gate, "recorded_gate_allow": r["gate_allow"]})
            r["gate_allow"] = gate

    train = A.trades("is")
    slope = A.fit_platt(np.array([r["p_side"] for r in train]), A.outcomes(train))
    unfiltered = describe(rows, slope)
    baseline = historical["unfiltered_forward_holdout"]
    fields = ("trades", "wins", "net", "pf", "by_lane", "net_ex_xrp", "max_entry_cost", "crossed_fills")
    if any(unfiltered[k] != baseline[k] for k in fields) or round(unfiltered["raw_p_side_ece"], 4) != baseline["raw_p_side_ece"]:
        raise ValueError("saved sim_core fills do not reproduce the recorded unfiltered metrics")
    groups = {state: [r for r in rows if r["gate_allow"] is state] for state in (True, False, None)}
    early = [r["ticker"] for r in rows if r["trigger_ts"] <= CUTOFF]
    return {
        "status": "audited_existing_single_look_not_replayed",
        "checked_at_utc": datetime.fromtimestamp(as_of, timezone.utc).isoformat(),
        "prior_score_utc": historical["timestamp_utc"],
        "cutoff_close_ts": CUTOFF,
        "cutoff_in_est": datetime.fromtimestamp(CUTOFF, ZoneInfo("America/New_York")).isoformat(),
        "historical_holdout": horizon,
        "prior_ledger_lines": {k: v[0] for k, v in logged.items()},
        "min_windows_per_lane": MIN_WINDOWS_PER_LANE,
        "settled_windows_by_lane_as_of_check": counts,
        "settled_windows_by_lane_through_prior_horizon_from_current_market_metadata": recorded_counts,
        "saved_fills_with_pre_cutoff_trigger": early,
        "source_sha256": {path.name: A.file_digest(path) for path in (report_path, A.trade_path("forward"), A.trade_path("is"), A.FIT)}
                         | {"cache_markets.json": hashlib.sha256(payload).hexdigest()},
        "platt_logit_slope_fit_on_is_only": slope,
        "unfiltered": unfiltered, "layagate_allowed": describe(groups[True], slope),
        "layagate_denied": describe(groups[False], slope), "unlogged": describe(groups[None], slope),
        "gate_counts": {name: len(groups[state]) for name, state in
                        (("allowed", True), ("denied", False), ("unlogged", None))},
        "gate_disagreements": sum(r["gate_allow"] is not r["recorded_gate_allow"] for r in gate_rows),
        "gate_source": "book_*.npz: latest same-ticker tick_tid/tick_ts <= trigger; gate_available=1; no DB re-query",
        "gate_rows": gate_rows,
    }


if __name__ == "__main__":
    out = A.DATA / "track2b-forward-holdout-audit.json"
    out.write_text(json.dumps(audit_forward(time.time()), indent=2, allow_nan=False) + "\n")
    print(out)
