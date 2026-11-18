#!/usr/bin/env python3
"""Track 10 addendum, EXPLORATORY and not registered: F2 with its calibrated-probability gate, on the post-freeze windows.

The registered Track 10 scores the ungated F2 trigger set. The live program also gates every trigger through a payoff calibrator
(ridge logistic on z, signed flow, signed revision and delta in bp, target P(trade pnl > 0)) and the desk's `decide()` hurdle. This
reproduces that gate with nothing from the post-freeze windows: the calibrator is fitted on the pre-freeze trades at the same latency,
then each post-freeze trade is kept when p - ask - exit_cost - taker fee >= max(edge_hurdle, min_net_edge). It was designed after the
registered result was known, it is one row per latency, it sits outside the Holm family and decides nothing; it exists to compare the
gated rule, which is what runs live, with the ungated trigger set.

    flock data/.heavy.lock nice -n 10 .venv/bin/python scripts/track10_f2_gated.py
"""
from __future__ import annotations

import gc
import json
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openkal.strategy import Params, exit_cost, taker_fee_per_contract  # noqa: E402
from scripts import f2_structural as F  # noqa: E402
from scripts import rnd_common as RC  # noqa: E402
from scripts import track10_champion_rescore as R  # noqa: E402
from scripts import track4_common as T  # noqa: E402

OUT = ROOT / "data" / "track10-f2-gated-exploratory.json"
MIN_FIT = 20                       # f2_structural's own floor for fitting the calibrator
REASON = "Track 10 exploratory addendum: F2 with its calibrated gate on post-freeze windows (not registered, outside Holm)"


def gate(pre_rows: list[dict], post_rows: list[dict], params: Params | None = None) -> dict:
    """Fit the calibrator on `pre_rows` only, keep the `post_rows` the desk's entry test would take."""
    params = params or Params()
    if len(pre_rows) < MIN_FIT:
        return {"fitted": False, "fit_rows": len(pre_rows), "kept": []}
    x = np.array([r["feat"] for r in pre_rows], dtype=np.float64)
    y = np.array([1.0 if Decimal(r["pnl"]) > 0 else 0.0 for r in pre_rows])
    w, norm = F.fit_logistic(x, y)
    hurdle = max(Decimal(str(params.edge_hurdle)), Decimal(str(params.min_net_edge)))
    kept = []
    for r in post_rows:
        p = float(F.apply_logistic(np.array([r["feat"]], dtype=np.float64), w, norm)[0])
        ask = Decimal(r["entry"])                                   # a taker pays the side's ask, which is the row's entry cost
        net = Decimal(str(p)) - ask - exit_cost(ask, params) - taker_fee_per_contract(ask)
        if net >= hurdle:
            kept.append({**r, "p_side": p, "gate_net": float(net)})
    return {"fitted": True, "fit_rows": len(pre_rows), "hurdle": float(hurdle), "kept": kept}


def run(tape: Path = R.TAPE, pool_dir: Path = R.POOL, out: Path = OUT, registered: Path | None = None) -> dict:
    t0 = time.time()
    if out.exists():
        raise RuntimeError(f"{out.name} exists: the exploratory row is made once")
    T.pin_tape(tape)
    closes = T.window_closes(tape)
    post = [c for c in closes if c > R.FREEZE_CLOSE]
    lanes = RC.load_lanes()
    triggers = []
    for name in R.LANES:
        pool = R.load_pool(name, pool_dir)
        triggers += R.f2_triggers(name, lanes[name], pool)
        del pool
        gc.collect()
    tapes = {(t["lane"], t["tid"]): RC.tape(lanes[t["lane"]], t["tid"]) for t in triggers}
    main = json.loads((registered or next((ROOT / "data").glob("track10-rescore-*.json"))).read_text())
    cells = {}
    for lag in R.LATENCIES:
        R.ledger_append({"ts": datetime.now(timezone.utc).isoformat(), "family": f"rescore10x:F2_taker_1s_gated@{lag:g}s", "status": "started",
                         "config": {"latency_s": lag, "exploratory": True}, "rescore": True, "rescore_reason": REASON})
    for lag in R.LATENCIES:
        rows, _ = R.score_cell(triggers, tapes, lag)
        pre = [r for r in rows if r["slice"] == "pre"]
        post_rows = [r for r in rows if r["slice"] == "post"]
        g = gate(pre, post_rows)
        gated_pre = gate(pre, pre)                                   # in sample: the calibrator scored on the trades it was fitted on
        reg = main["cells"]["F2_taker_1s"][f"{lag:g}"]["post"]
        if (len(post_rows), round(RC.summarize(post_rows)["net"], 2)) != (reg["trades"], round(reg["net"], 2)):
            raise RuntimeError("the ungated post-freeze score does not reproduce the registered Track 10 cell")
        cells[f"{lag:g}"] = {"ungated_post": R.short(RC.summarize(post_rows)), "gated_post": R.short(RC.summarize(g["kept"])),
                             "gated_in_sample": R.short(RC.summarize(gated_pre["kept"])), "pre_trades": len(pre),
                             "kept_of_post": f"{len(g['kept'])} of {len(post_rows)}", "fitted": g["fitted"], "hurdle": g.get("hurdle"),
                             "gated_post_by_lane": RC.summarize(g["kept"])["by_lane"]}
        R.ledger_append({"ts": datetime.now(timezone.utc).isoformat(), "family": f"rescore10x:F2_taker_1s_gated@{lag:g}s", "status": "scored",
                         "config": {"latency_s": lag, "exploratory": True}, "rescore": True, "rescore_reason": REASON,
                         "post": cells[f"{lag:g}"]["gated_post"], "cache": str(tape)})
    result = {"track": "champion_rescore_f2_gated_exploratory", "registered": False, "outside_holm": True, "freeze_close": R.FREEZE_CLOSE,
              "post_freeze_windows": len(post), "cells": cells, "seconds": round(time.time() - t0, 1)}
    RC.dump(out, result)
    return result


def table(result: dict) -> str:
    lines = [f"{'latency':8s} {'ungated post n/net/PF':26s} {'gated post n/net/PF/win':30s} {'gated in-sample n/net':22s} kept"]
    for lag, c in result["cells"].items():
        u, g, i = c["ungated_post"], c["gated_post"], c["gated_in_sample"]
        lines.append(f"{lag + ' s':8s} {u['trades']:4d} {u['net']:+9.2f} {u['pf']:6.2f}      {g['trades']:4d} {g['net']:+9.2f} {g['pf']:6.2f} {g['win_rate']:.2f}"
                     f"      {i['trades']:4d} {i['net']:+9.2f}   {c['kept_of_post']}")
    return "\n".join(lines)


def main() -> int:
    print(table(run()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
