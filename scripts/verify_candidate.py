"""Run the frozen 60/40 verification through the immutable verifier and the SHR.

Two planes, on purpose:

  * `verifier.replay` drives the engine's own `replay_series`, so the numbers here
    come from the same execution core that would trade the candidate.
  * `shr.Proposer` mutates the EVOLVE-BLOCK, and every variant is scored on the
    in-sample slice and the frozen out-of-sample slice. Nothing is promoted unless
    both slices clear the immutable thresholds.

Splits are chronological by *window close* over every settled window in the tape,
never by candidate count.

    .venv/bin/python scripts/verify_candidate.py
"""
from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openkal import features, shr, verifier  # noqa: E402
from openkal.config import SLOTS  # noqa: E402
from openkal.store import Store  # noqa: E402
from openkal.strategy import Params, clamp_params  # noqa: E402
from scripts.micro_study import connect, market_meta  # noqa: E402

OUT = ROOT / "data" / "verify-report.json"
STAKE = Decimal("65")


def main() -> int:
    markets = market_meta()
    db = connect()
    lanes = [c for c in SLOTS if c.feed_symbol]
    result: dict = {"lanes": {}, "summary": {}}
    totals = {"is": {"trades": 0, "pnl": Decimal("0")},
              "oos": {"trades": 0, "pnl": Decimal("0")}}
    for cfg in lanes:
        rows = db.execute(
            "SELECT ts,ticker FROM ticks WHERE series=? ORDER BY ts", (cfg.series,)
        ).fetchall()
        closes = sorted({markets[r["ticker"]]["close"] for r in rows
                         if r["ticker"] in markets})
        if len(closes) < 2:
            continue
        cut = int(len(closes) * 0.6)
        is_closes = set(closes[:cut])
        ticks = Store(ROOT / "data" / "openkal-prod.db").recent_ticks(
            cfg.series, 0.0, limit=200000)
        buckets = {"is": [], "oos": []}
        for t in ticks:
            m = markets.get(t["ticker"])
            if m is None:
                continue
            buckets["is" if m["close"] in is_closes else "oos"].append(t)
        baseline = verifier.ReplayResult(0, Decimal("0"), 0.0, 0.0, 0.0, Decimal("0"))
        base = Params()
        lane_out = {"ticks": {k: len(v) for k, v in buckets.items()}, "windows": len(closes)}
        for name in ("is", "oos"):
            slice_ticks = buckets[name]
            if len(slice_ticks) < 30:
                lane_out[name] = {"skipped": "not enough ticks"}
                continue
            res = verifier.replay(slice_ticks, base, STAKE)
            ok, why = verifier.passes(res, baseline)
            lane_out[name] = {
                "ticks": len(slice_ticks), "trades": res.trades,
                "pnl": str(res.pnl.quantize(Decimal("0.01"))),
                "win_rate": round(res.win_rate, 3),
                "profit_factor": round(res.profit_factor, 3),
                "ece": round(res.ece, 4), "max_drawdown": str(res.max_drawdown),
                "verifier_pass": ok, "verifier_note": why,
            }
            totals[name]["trades"] += res.trades
            totals[name]["pnl"] += res.pnl

        # SHR: propose bounded variants and let the same verifier judge both slices.
        failure = {"win_rate": lane_out.get("is", {}).get("win_rate", 0.5),
                   "fill_rate": 0.5, "avg_spread": 0.02}
        candidates = shr.Proposer().propose(base, failure)
        judged = []
        baseline_is = verifier.replay(buckets["is"], base, STAKE) if len(buckets["is"]) >= 30 else None
        baseline_oos = verifier.replay(buckets["oos"], base, STAKE) if len(buckets["oos"]) >= 30 else None
        for cand in candidates:
            cand.params = clamp_params(cand.params)
            if baseline_is is None or baseline_oos is None:
                judged.append({"name": cand.name, "status": "unverified"})
                continue
            r_is = verifier.replay(buckets["is"], cand.params, STAKE)
            r_oos = verifier.replay(buckets["oos"], cand.params, STAKE)
            ok_is, why_is = verifier.passes(r_is, baseline_is)
            ok_oos, why_oos = verifier.passes(r_oos, baseline_oos)
            judged.append({"name": cand.name, "status": "verified" if ok_is and ok_oos else "rejected",
                           "is_trades": r_is.trades, "is_pnl": str(r_is.pnl.quantize(Decimal("0.01"))),
                           "oos_trades": r_oos.trades, "oos_pnl": str(r_oos.pnl.quantize(Decimal("0.01"))),
                           "note": f"{why_is}; holdout: {why_oos}"})
        lane_out["shr"] = {"proposed": len(candidates), "candidates": judged,
                           "promoted": sum(1 for j in judged if j["status"] == "verified")}
        result["lanes"][cfg.series] = lane_out
        print(f"{cfg.series}: IS {lane_out.get('is', {}).get('trades', 0)} trades "
              f"{lane_out.get('is', {}).get('pnl', '-')} | OOS "
              f"{lane_out.get('oos', {}).get('trades', 0)} trades "
              f"{lane_out.get('oos', {}).get('pnl', '-')} | SHR promoted "
              f"{lane_out['shr']['promoted']}/{len(candidates)}")

    result["summary"] = {
        "is_trades": totals["is"]["trades"],
        "is_pnl": str(totals["is"]["pnl"].quantize(Decimal("0.01"))),
        "oos_trades": totals["oos"]["trades"],
        "oos_pnl": str(totals["oos"]["pnl"].quantize(Decimal("0.01"))),
        "promoted": sum(v["shr"]["promoted"] for v in result["lanes"].values()),
        "verdict": "rejected" if any(
            not v.get(name, {}).get("verifier_pass", False)
            for v in result["lanes"].values() for name in ("is", "oos")
        ) or totals["oos"]["pnl"] <= 0 else "passed",
    }
    OUT.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result["summary"], indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
