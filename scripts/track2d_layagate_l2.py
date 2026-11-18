#!/usr/bin/env python3
"""Track 2D: fixed LayaGate + raw true-L2 + Student-t settlement replay."""
from __future__ import annotations

import datetime
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import track2c_true_l2_settle as C

OUT = ROOT / "data/track2d-layagate-l2.json"
IS_TRADES = ROOT / "data/track2d-layagate-l2-is-trades.jsonl"
OOS_TRADES = ROOT / "data/track2d-layagate-l2-oos-trades.jsonl"
FAMILY = "layagate_l2_settle"
CONFIG = "layagate_allow_or_conf055_raw_obi020_positive_fv_taker"
MIN_CONFIDENCE, MIN_ABS_OBI = .55, .20
FEATURES = ["p_side", "edge", "secs_left", "l2_obi", "gate_allow"]
GATE_COLUMNS = ("tick_ts", "tick_tid", "gate_available", "gate_allow", "gate_confidence")


def load_gates(data):
    if any(key not in data for key in GATE_COLUMNS):
        return {}
    ts, tid, available, allow, confidence = (data[key] for key in GATE_COLUMNS)
    if not (len(ts) == len(tid) == len(available) == len(allow) == len(confidence)) \
            or not np.isfinite(ts).all() or np.any(ts[:-1] > ts[1:]):
        raise ValueError("LayaGate cache arrays are misaligned or out of time order")
    order = np.argsort(tid, kind="stable")
    ids, starts = np.unique(tid[order], return_index=True)
    groups = np.split(order, starts[1:]) if len(ids) else []
    return {int(key): (ts[idx], available[idx] == 1, allow[idx] == 1, confidence[idx])
            for key, idx in zip(ids, groups) if key >= 0}


def gate_at(gate, times):
    """Latest same-ticker decision at or before T, including unavailable ticks."""
    if gate is None:
        return np.zeros(len(times), bool), np.zeros(len(times), bool), np.full(len(times), np.nan)
    ts, available, allow, confidence = gate
    idx = np.searchsorted(ts, times, side="right") - 1
    row = np.maximum(idx, 0)
    return (idx >= 0) & available[row], (idx >= 0) & allow[row], np.where(idx >= 0, confidence[row], np.nan)


def joint_coverage(cache, closes):
    """Count distinct shared closes with both inputs observable at the same decision."""
    markets = json.loads((cache / "markets.json").read_text())
    by_lane = {}
    for lane in C.RC.LANES:
        found = set()
        by_lane[lane] = []
        with np.load(cache / f"book_{lane}.npz", allow_pickle=False) as data:
            gates = load_gates(data)
            candidates = {}
            for tid, name in enumerate(data["tickers"]):
                m = markets.get(str(name))
                if (m and m.get("y") in (0, 1) and m.get("strike", 0) > 0 and m["close"] in closes and tid in gates
                        and gates[tid][1].any()):
                    times = m["close"] - C.DECISION_SECS
                    available, _, _ = gate_at(gates[tid], times)
                    if available.any():
                        candidates[tid] = (m["close"], times[available])
            if not candidates:
                continue
            ids, ts, bid, ask, micro = (data[key] for key in
                ("tid", "ts", "yes_bid_sz", "yes_ask_sz", "microprice"))
            for tid, (close, times) in candidates.items():
                a = np.searchsorted(ids, tid, side="left")
                b = np.searchsorted(ids, tid, side="right")
                if np.isfinite(C.recent_l2_obi(times, ts[a:b], bid[a:b], ask[a:b], micro[a:b])).any():
                    found.add(close)
        by_lane[lane] = sorted(found)
    return by_lane


def gate_pass(available, allow, confidence):
    return available & (allow | (np.isfinite(confidence) & (confidence >= MIN_CONFIDENCE)))


def find_entries(pack, available, allow, confidence):
    gate = gate_pass(available, allow, confidence)
    obi = pack["l2_obi"]
    cy, cn = pack["cost_yes_taker"], pack["cost_no_taker"]
    ey, en = pack["p_yes"] - cy, pack["p_no"] - cn
    yes = (obi >= MIN_ABS_OBI) & (cy >= C.COST_LO) & (cy <= C.COST_HI) & (ey > 0)
    no = (obi <= -MIN_ABS_OBI) & (cn >= C.COST_LO) & (cn <= C.COST_HI) & (en > 0)
    chosen = C.BS.pick({"tid": pack["tid"], "ts": pack["t"]}, gate & (yes | no),
                       cooldown=C.COOLDOWN, per_window=C.PER_WINDOW)
    return [(i, C.S.YES if yes[i] else C.S.NO,
             float(pack["p_yes" if yes[i] else "p_no"][i]),
             float(ey[i] if yes[i] else en[i]), float(obi[i]), float(allow[i])) for i in chosen]


def execute(pack, available, allow, confidence):
    scored = {"rows": [], "skips": Counter()}
    triggers = find_entries(pack, available, allow, confidence)
    C.run_l2_triggers(pack, triggers, CONFIG, "taker", 0., {CONFIG: scored}, family=FAMILY)
    lane = pack["lane"]
    by_trigger = {(lane.tickers[int(pack["tid"][i])], float(pack["t"][i])): i for i, *_ in triggers}
    for row in scored["rows"]:
        i = by_trigger[row["ticker"], row["trigger_ts"]]
        row.update(l2_obi=float(pack["l2_obi"][i]), gate_available=True, gate_allow=bool(allow[i]),
                   gate_confidence=float(confidence[i]) if np.isfinite(confidence[i]) else None)
    return scored["rows"], scored["skips"]


def score_slice(cache, by_lane, cut, split, fits):
    rows, skips = [], Counter()
    coverage = {}
    for name, closes in by_lane.items():
        closes = {close for close in closes if C.RC.slice_of(close, cut) == split}
        if not closes:
            continue
        lane = C.AD.Lane(name, max_close=cut if split == "is" else None)
        pack = C.collect_l2_lane_samples(lane, fits[name], closes, cut, split, cache, raw_only=True)
        with np.load(cache / f"book_{name}.npz", allow_pickle=False) as data:
            gates = load_gates(data)
        available, allow = np.zeros(len(pack["t"]), bool), np.zeros(len(pack["t"]), bool)
        confidence = np.full(len(pack["t"]), np.nan)
        for tid in np.unique(pack["tid"]):
            sel = pack["tid"] == tid
            available[sel], allow[sel], confidence[sel] = gate_at(gates.get(tid), pack["t"][sel])
        coverage[name] = {"priced_l2_samples": len(available), "gate_available_samples": int(available.sum()),
                          "gate_passing_samples": int(gate_pass(available, allow, confidence).sum())}
        fills, rejects = execute(pack, available, allow, confidence)
        rows.extend(fills)
        skips.update(pack["skips"])
        skips.update(rejects)
        del lane, pack
    return rows, dict(skips), coverage


def write_trades(path, rows):
    C.atomic_text(path, "".join(json.dumps(r, default=C.RC._json_default, allow_nan=False) + "\n" for r in rows))


def run_pinned(cache, c_receipt):
    """Called under Track 2C's single-writer lock; never replaces a spent look."""
    manifest, closes = c_receipt["frozen_input_sha256"], c_receipt["windows"]
    cut = c_receipt["cut"]
    finalized = OUT.exists()
    if not finalized:
        if os.environ.get("RND_OOS_UNSEAL") != "lead" or os.environ.get("RND_OOS_RESCORE"):
            raise RuntimeError("Track 2D needs the one-shot lead seal; rescores are forbidden")
    obj = json.loads(OUT.read_text()) if finalized else C.recover_scored_receipt(OUT, C.RC.LEDGER, FAMILY, manifest)
    if obj is not None and (obj.get("frozen_input_sha256"), obj.get("windows"), obj.get("cut")) != (manifest, closes, cut):
        raise RuntimeError("Track 2D receipt does not match the Track 2C pin")
    if obj is None:
        by_lane = joint_coverage(cache, set(closes))
        joint = set().union(*by_lane.values())
        obj = {
            "family": FAMILY, "config": CONFIG, "features": FEATURES, "cut": cut, "windows": closes,
            "frozen_input_sha256": manifest, "joint_windows_by_lane": by_lane,
            "joint_windows": len(joint), "is_windows": sum(close < cut for close in closes),
            "oos_windows": sum(close >= cut for close in closes), "joint_is_windows": sum(w < cut for w in joint),
            "joint_oos_windows": sum(w >= cut for w in joint), "is": None, "is_eligible": False,
            "oos_scored": None, "status": "Insufficient same-tape true-L2 + LayaGate coverage; OOS unspent",
            "rule": {
                "gate_available": True, "gate_allow_or_confidence_gte": MIN_CONFIDENCE,
                "raw_abs_obi_gte": MIN_ABS_OBI, "fv_edge_strictly_above": 0, "entry_style": "taker",
                "cost_band": [C.COST_LO, C.COST_HI], "lag_s": C.S.LAG_S, "exit": "settle",
            },
        }
        if len(joint) >= C.MIN_WINDOWS_REQUIRED:
            fits = c_receipt["fit_params"]  # Same Student-t calibration fitted only on Track 2C's IS index.
            C.AD.CACHE = cache
            C.RC.CACHE = C.EVOLVE_CACHE
            rows, skips, coverage = score_slice(cache, by_lane, cut, "is", fits)
            write_trades(IS_TRADES, rows)
            obj.update(is_coverage=coverage, fit_params=fits)
            obj["is"] = C.RC.summarize(rows, skips)
            obj["is_eligible"] = C.eligible(obj["is"])
            obj["status"] = "IS fail; OOS unspent"
            if obj["is_eligible"]:
                def scorer():
                    oos, skipped, oos_coverage = score_slice(cache, by_lane, cut, "oos", fits)
                    obj["oos_coverage"] = oos_coverage
                    write_trades(OOS_TRADES, oos)
                    return oos, skipped, C.RC.fit_payoff_model(rows)

                C.score_with_receipt(OUT, obj, FAMILY, {"config": CONFIG, "cut": cut, **obj["rule"],
                                                       "frozen_input_sha256": manifest}, scorer)
        obj["completed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if not finalized:
        if obj["oos_scored"] is not None:
            obj["status"] = "OOS scored"
        # Match existing metrics semantics without writing non-standard Infinity to JSON.
        for summary in (obj["is"], (obj["oos_scored"] or {}).get("oos")):
            if summary is not None and math.isinf(summary["pf"]):
                summary["pf"] = "inf"
        C.save_receipt(OUT, obj, strict=True)
    update_handoff(obj)
    return obj


def update_handoff(obj):
    score, best = obj["oos_scored"], obj["is"]
    next_data = json.loads(C.NEXT_JSON.read_text())
    next_data["updated"] = obj["completed_at"]
    state = next_data["rnd_overnight_shift"].setdefault("track2d_layagate_l2", {})
    state.update({
        "status": obj["status"], "config": CONFIG, "same_tape_as_track2c": True,
        "joint_windows": obj["joint_windows"], "joint_is_windows": obj["joint_is_windows"],
        "joint_oos_windows": obj["joint_oos_windows"], "is": best, "is_eligible": obj["is_eligible"],
        "oos_spent": score is not None, "oos": score["oos"] if score else None,
        "payoff_ece": score["payoff_ece"] if score else None, "gate_pass": score["gate"]["pass"] if score else False,
        "result_artifact": str(OUT.relative_to(ROOT)), "note": "Research only; no live promotion or trading.",
    })
    C.atomic_text(C.NEXT_JSON, json.dumps(next_data, indent=1, allow_nan=False) + "\n")

    header = "### Track 2D — LayaGate + true-L2 settlement result"
    report = C.RND_REPORT.read_text()
    if header not in report:
        lines = [
            header,
            f"- {obj['joint_windows']}/{len(obj['windows'])} distinct closes with joint coverage across lanes "
            f"on the same frozen tape and cut as Track 2C: {obj['is_windows']} IS / {obj['oos_windows']} OOS "
            f"({obj['joint_is_windows']} / {obj['joint_oos_windows']} jointly covered).",
            "- Taker, positive own-side Student-t edge at the trigger, "
            "live same-ticker verdict (`allow` or confidence ≥ 0.55), signed raw OBI ≥ 0.20 in magnitude, "
            "$0.35–$0.60, T+1s, and binary $1/$0 settlement via `sim_core.py`.",
        ]
        if best is not None:
            admitted = sum(c["gate_passing_samples"] for c in obj["is_coverage"].values())
            coverage = sum(c["gate_available_samples"] for c in obj["is_coverage"].values())
            pf = best["pf"] if best["trades"] else "n/a"
            lines.append(f"- IS: {best['trades']} trades, net ${best['net']:+.2f}, PF {pf}, "
                         f"{best['profitable_lanes']} profitable lanes; {admitted}/{coverage} "
                         "available decision snapshots passed LayaGate before the OBI/FV filter.")
        if score is not None:
            m = score["oos"]
            lines.append(f"- One sealed OOS look: {m['trades']} trades, net ${m['net']:+.2f}, PF {m['pf']}, "
                         f"payoff-ECE {score['payoff_ece']}; research gate pass: {score['gate']['pass']}.")
        else:
            lines.append(f"- {obj['status']}. No new OOS ledger entry.")
        lines.append(f"- Receipt: `{OUT.relative_to(ROOT)}`; no live promotion.")
        C.atomic_text(C.RND_REPORT, report.rstrip() + "\n\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    if sys.argv[1:] != ["--check-only"]:
        raise SystemExit("Run Track 2C to replay the shared pin, or use --check-only to inspect cache coverage.")
    audit = C.audit_l2_depth()
    print(json.dumps({"true_l2_windows": audit["windows_with_l2"], "joint_windows_by_lane":
                      joint_coverage(C.EVOLVE_CACHE, set(audit.get("l2_window_closes", [])))}, indent=2))
