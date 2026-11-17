"""Targeted z_obi_flow scaling sweep on the frozen 60/40 split.

Sweeps the exact deployed exit (maker_scratch, 0.30-0.60 cost band, 25s entry patience,
0.05 TP / 45s barrier family) across:
  a) entry window T_remaining [360,840] vs expanded [240,860]
  b) TP {0.05, 0.06, 0.07} x barrier {35s, 45s}
  c) max_window_trades {2, 3} with per-window loss halt and seq_cap_halt enforced

Selection is in-sample only (highest IS profit factor, 20+ IS trades, positive IS net);
the frozen OOS slice is scored once per variant. Promotion is reported, never applied
by this script.
"""
from __future__ import annotations

import json
import sys
import time
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.alpha_search import (LANES, SimParams, ZOBI_REV_MIN, block2, cal_ece,
                                  gate, platt_fit, sequence_rows, slim2, simulate,
                                  trailing_sigma, z_obi_flow_events)
from scripts.alpha_discover import Lane

HOURS = {"is": 0.0, "oos": 0.0}
started = time.time()
print("sweep start", time.strftime("%H:%M:%S"), flush=True)

# Load lanes once, build both event windows once.
by_window: dict[tuple[float, float], dict[str, tuple]] = {}
for name in LANES:
    lane = Lane(name)
    sig = trailing_sigma(lane)
    for sl in ("is", "oos"):
        HOURS[sl] += lane.hours[sl]
    for t_lo, t_hi in ((360.0, 840.0), (240.0, 860.0)):
        evs = []
        for z_min in (0.6, 0.8, 1.0, 1.2):
            for flow_min in (0.15, 0.25, 0.35):
                evs.extend(z_obi_flow_events(lane, sig, z_min=z_min, flow_min=flow_min,
                                             rev_min=ZOBI_REV_MIN,
                                             t_lo=t_lo, t_hi=t_hi,
                                             cost_lo=0.30, cost_hi=0.60))
        by_window.setdefault((t_lo, t_hi), {})[name] = (lane, evs)
        print(f"{name} window [{t_lo:.0f},{t_hi:.0f}] events={len(evs)} "
              f"({time.time() - started:.0f}s)", flush=True)
    del lane

variants = []
for (t_lo, t_hi), bylane in sorted(by_window.items()):
    for tp_c in (0.05, 0.06, 0.07):
        for barrier_s in (35.0, 45.0):
            for cap in (2, 3):
                rows: list[dict] = []
                for name, (lane, evs) in bylane.items():
                    params = SimParams(tp_c=tp_c, barrier_s=barrier_s,
                                       exit_mode="maker_scratch",
                                       scratch_concession_c=0.0,
                                       scratch_patience_s=20.0)
                    rows.extend(simulate(lane, evs, params))
                for r in rows:
                    if r["crossed"] != 0:
                        raise RuntimeError(f"crossed fill: {r}")
                seq = sequence_rows(rows, max_window_trades=cap, halt_after_loss=True)
                is_rows = [r for r in seq if r["slice"] == "is"]
                oos_rows = [r for r in seq if r["slice"] == "oos"]
                if not is_rows or not oos_rows:
                    print(f"[{t_lo:.0f},{t_hi:.0f}] tp{tp_c:.2f} b{barrier_s:.0f} cap{cap} "
                          f"SKIP (no IS or OOS rows)", flush=True)
                    continue
                is_block = block2(is_rows, HOURS["is"])
                oos_block = block2(oos_rows, HOURS["oos"])
                pa, pb = platt_fit(is_rows)
                is_block["ece_raw"] = is_block.get("ece")
                is_block["ece"] = cal_ece(is_rows, pa, pb)
                oos_block["ece_raw"] = oos_block.get("ece")
                oos_block["ece"] = cal_ece(oos_rows, pa, pb)
                row = {"window": [t_lo, t_hi], "tp_c": tp_c, "barrier_s": barrier_s,
                       "cap": cap, "is": slim2(is_block), "oos": slim2(oos_block),
                       "gate": gate(oos_block),
                       "platt": {"a": round(pa, 4), "b": round(pb, 4)}}
                variants.append(row)
                print(f"[{t_lo:.0f},{t_hi:.0f}] tp{tp_c:.2f} b{barrier_s:.0f} cap{cap} "
                      f"IS {is_block['trades']}/{is_block['net']} pf {is_block['pf']} "
                      f"-> OOS {oos_block['trades']}/{oos_block['net']} pf {oos_block['pf']} "
                      f"ece {oos_block['ece']} pass={all(row['gate'].values())}", flush=True)

out = {"lanes": LANES, "hours": HOURS, "variants": variants,
       "elapsed_s": round(time.time() - started, 1)}
Path("data/sweep-fast.json").write_text(json.dumps(out, indent=1, default=str))
print("sweep done", out["elapsed_s"], "s, variants", len(variants),
      "-> data/sweep-fast.json", flush=True)
