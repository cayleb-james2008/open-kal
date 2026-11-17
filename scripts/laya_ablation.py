#!/usr/bin/env python3
"""Track C: LayaGate ablation on F2_taker_1s, frozen tape, same chronological 60/40 split.

LayaGate arm: BLOCKED. No historical per-trigger LayaGate verdicts exist on disk for the
frozen tape, and the only fine-tuned checkpoint was trained on labels written after the
OOS start (see `gate_findings` in the output). The live endpoint is not called.

What runs instead, under both NO-side conventions:
  baseline  F2_taker_1s, tp5, no ETH filter (the frozen rule as swept).
  oracle    the same triggers minus those the sim stopped out (exit reason `stop_loss`,
            the 2s-confirmed 7c stop). HINDSIGHT ONLY: it uses the realised path, so it
            is a ceiling on any rejection filter, NOT a deployable filter.

Reads data/frozen-cache (via scripts/evolve_f2_variants.py). Writes data/laya-ablation.json.

    .venv/bin/python scripts/laya_ablation.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import alpha_discover as AD  # noqa: E402
from scripts import evolve_f2_variants as E  # noqa: E402
from scripts import f2_structural as F  # noqa: E402

AD.CACHE = ROOT / "data" / "frozen-cache"
OUT = ROOT / "data" / "laya-ablation.json"
CONVENTIONS = ("as_built", "corrected")
STOP_REASON = "stop_loss"

GATE_FINDINGS = {
    "status": "blocked: no per-trigger LayaGate verdicts exist on disk for the frozen triggers. "
              "Not contaminated for OOS by time.",
    "verdict_shape": "openkal/decision_gate.py Verdict(available, allow, confidence=P(enter), "
                     "note, model, latency_ms, error); allow = P(enter) >= 0.55. Live mode is "
                     "'shadow' (no ONNX), so engine._gate_entry logs the verdict and never vetoes.",
    "historical_verdicts_on_disk": "none in this worktree: data/laya-rank.json and data/laya-rlcd/ "
                                   "are absent here. The main repo has data/laya-rank.json, which "
                                   "scores post-cutoff prod ticks from openkal-prod.db (not F2 "
                                   "triggers, per laya_rank.py); it was not opened.",
    "model_training_data": "data/laya-rlcd/labels.jsonl (mtime 2026-11-16 00:57 UTC), then "
                           "labels.csv, the fine-tune input (mtime 2026-11-16 10:05 UTC). Built from "
                           "prod-DB ticks. Training text contains the realised 30s post-fee return, "
                           "so the label is readable from the prompt (leakage, per laya_rank.py).",
    "oos_start_utc": "last IS close 2026-11-17 02:00 UTC, first OOS close 2026-11-17 02:15 UTC "
                     "(Lane 60/40 split, see split).",
    "contamination": "NOT contaminated for OOS by time: the training labels end about 16 h before "
                     "the first OOS close. IS windows that start before 2026-11-16 10:05 UTC may "
                     "overlap training data; the IS start was not recorded by this run.",
    "fine_tuned_checkpoint_present_in_worktree": False,
    "laya_onnx_present": False,
    "which_checkpoint_the_gate_serves": "unverified: the endpoint at OPENKAL_GATE_URL is not in "
                                        "this worktree and was not called.",
    "base_checkpoint": "convaiinnovations/laya typed-decisions in the HF hub cache. Its training "
                       "cutoff was not verified. It is zero-shot, about 1.7s per item on CPU, and "
                       "was not run here.",
    "still_needed": "per-trigger LayaGate scores for the 43 OOS triggers, from offline scoring "
                    "with the fine-tuned checkpoint (not run here), or from the engine's logged "
                    "verdicts (not in this worktree).",
}


def _trades(run: list[dict]) -> tuple[list[dict], list[dict]]:
    clean = [r for r in run if not r.get("crossed")]
    return clean, [r for r in run if r.get("crossed")]


def arm(recs: dict[str, list[dict]], lanes: dict, conv: str, oracle: bool) -> dict:
    rows = {s: E.run_config(recs[s], lanes, "taker", "tp5", "none", conv) for s in ("is", "oos")}
    removed = {}
    if oracle:
        for s in ("is", "oos"):
            clean, crossed = _trades(rows[s])
            kept = [r for r in clean if r["reason"] != STOP_REASON]
            removed[s] = len(clean) - len(kept)
            rows[s] = kept + crossed
    else:
        removed = {"is": 0, "oos": 0}
    out = E.score(rows["is"], rows["oos"])
    out["removed_stop_loss"] = removed
    out["gate_arm"] = "blocked"
    return out


def main() -> int:
    lanes = {name: AD.Lane(name) for name in F.LANES}
    any_lane = lanes[F.LANES[0]]
    split = {
        "is_close_max_utc": datetime.fromtimestamp(float(max(any_lane.is_closes)),
                                                   timezone.utc).isoformat(),
        "oos_close_min_utc": datetime.fromtimestamp(float(min(any_lane.oos_closes)),
                                                    timezone.utc).isoformat(),
    }
    pools = F.build_pools(lanes)
    recs = {s: E.arrivals(lanes, pools, s) for s in ("is", "oos")}
    result: dict = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "cache": str(AD.CACHE),
        "split": split,
        "triggers": {s: len(recs[s]) for s in recs},
        "gate_findings": GATE_FINDINGS,
        "rows": {},
    }
    for conv in CONVENTIONS:
        for label, oracle in (("baseline", False), ("oracle_hindsight", True)):
            key = f"{label}_{conv}"
            result["rows"][key] = arm(recs, lanes, conv, oracle)
            s = result["rows"][key]
            print(f"{key:28s} IS n={s['is']['trades']:3d} net={s['is']['net_pnl']:+9.2f} "
                  f"PF={s['is']['pf']:.3f} | OOS n={s['oos']['trades']:3d} "
                  f"net={s['oos']['net_pnl']:+9.2f} PF={s['oos']['pf']:.3f} "
                  f"win={s['oos']['win_rate']:.3f} removed={s['removed_stop_loss']}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1, default=str))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
