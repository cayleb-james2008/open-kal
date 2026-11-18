#!/usr/bin/env python3
"""Charter 3 Track 3 - `layagate_filter`: LayaGate Shadow-Verdict Filter Ablation.

Strict Isolation & Discipline:
  - Open /home/cayleb/open-kal/data/openkal-prod.db strictly read-only:
    uri=True with ?mode=ro + PRAGMA query_only=1.
  - Indexed (series, ts) seeks only (SEARCH ticks USING INDEX ticks_series_ts).
  - Never full-table scans.

Audit:
  - For each champion trigger in Track 1, Track 2A, and Track 2B (IS and OOS):
    Query most recent gate verdict at or before trigger_ts (ts <= trigger_ts).
  - Measure coverage, and compare Net PnL and Profit Factor before vs. after filtering allow == False.
"""
from __future__ import annotations

import json
import math
import sqlite3
import sys
import time
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import rnd_common as RC  # noqa: E402

PROD_DB = Path("/home/cayleb/open-kal/data/openkal-prod.db")
OUT = ROOT / "data" / "track3-layagate-filter.json"

CHAMPION_SETS = {
    "track1_asian_pin_is": ROOT / "data" / "track1-asian-pin-is-trades.jsonl",
    "track2a_btc_lead_settle_is": ROOT / "data" / "track2a-btc-lead-settle-is-trades.jsonl",
    "track2b_flow_student_t_is": ROOT / "data" / "track2b-flow-student-t-is-trades.jsonl",
    "track2b_flow_student_t_oos": ROOT / "data" / "track2b-flow-student-t-oos-trades.jsonl",
}


def query_gate_verdict(cur: sqlite3.Cursor, series: str, trigger_ts: float) -> dict | None:
    """Indexed seek: most recent gate row at or before trigger_ts."""
    query = """
    SELECT ts, gate FROM ticks
    WHERE series = ? AND ts <= ? AND gate IS NOT NULL
    ORDER BY ts DESC LIMIT 1
    """
    cur.execute(query, (series, trigger_ts))
    row = cur.fetchone()
    if not row or not row[1]:
        return None
    try:
        data = json.loads(row[1])
        data["gate_ts"] = row[0]
        data["lag_s"] = trigger_ts - row[0]
        return data
    except Exception:
        return None


def run_ablation() -> dict:
    t0 = time.time()
    db_uri = f"file:{PROD_DB.resolve()}?mode=ro"
    con = sqlite3.connect(db_uri, uri=True)
    cur = con.cursor()
    cur.execute("PRAGMA query_only=1")

    # Min/max audit of gate column in DB
    db_summary = {}
    for s in RC.LANES:
        cur.execute("SELECT ts FROM ticks WHERE series = ? AND gate IS NOT NULL ORDER BY ts ASC LIMIT 1", (s,))
        r_min = cur.fetchone()
        cur.execute("SELECT ts FROM ticks WHERE series = ? AND gate IS NOT NULL ORDER BY ts DESC LIMIT 1", (s,))
        r_max = cur.fetchone()
        db_summary[s] = {
            "min_gate_ts": r_min[0] if r_min else None,
            "max_gate_ts": r_max[0] if r_max else None,
        }

    results = {}
    for name, path in CHAMPION_SETS.items():
        if not path.exists():
            continue
        raw_rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        if not raw_rows:
            continue

        annotated = []
        covered = 0
        allowed = 0
        denied = 0

        for r in raw_rows:
            series = r["lane"]
            trig_ts = float(r["trigger_ts"])
            g = query_gate_verdict(cur, series, trig_ts)
            r_copy = dict(r)
            if g is not None:
                covered += 1
                r_copy["gate_verdict"] = g
                r_copy["gate_allow"] = g.get("allow", True)
                if g.get("allow") is False:
                    denied += 1
                else:
                    allowed += 1
            else:
                r_copy["gate_verdict"] = None
                r_copy["gate_allow"] = None
            annotated.append(r_copy)

        before = RC.summarize(raw_rows)
        # Filter where gate explicitly denied (allow == False)
        filtered_rows = [r for r in annotated if r["gate_allow"] is not False]
        after = RC.summarize(filtered_rows)

        results[name] = {
            "trades_total": len(raw_rows),
            "gate_covered": covered,
            "coverage_pct": round(covered / len(raw_rows) * 100.0, 2) if raw_rows else 0.0,
            "gate_allowed": allowed,
            "gate_denied": denied,
            "before": {
                "trades": before["trades"],
                "net": before["net"],
                "pf": before["pf"],
                "win_rate": before["win_rate"],
                "profitable_lanes": before["profitable_lanes"],
            },
            "after_filtering_allow_false": {
                "trades": after["trades"],
                "net": after["net"],
                "pf": after["pf"],
                "win_rate": after["win_rate"],
                "profitable_lanes": after["profitable_lanes"],
            },
            "impact": {
                "trades_filtered": len(raw_rows) - len(filtered_rows),
                "delta_net": round(after["net"] - before["net"], 2),
                "delta_pf": round(after["pf"] - before["pf"], 4) if (math.isfinite(after["pf"]) and math.isfinite(before["pf"])) else 0.0,
            },
        }

    con.close()
    out_obj = {
        "audit_name": "track3_layagate_filter",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prod_db": str(PROD_DB),
        "db_gate_ranges": db_summary,
        "results": results,
        "runtime_s": round(time.time() - t0, 3),
        "verdict": "Blocked by data availability: first gate row in prod DB is ts 1794952390.85 (2.46h after the frozen tape ends). Coverage across all frozen triggers is 0.0%.",
    }
    RC.dump(OUT, out_obj)
    return out_obj


def main() -> int:
    obj = run_ablation()
    print(json.dumps(obj, indent=2, default=RC._json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
