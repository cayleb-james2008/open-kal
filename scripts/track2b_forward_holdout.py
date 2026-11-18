#!/usr/bin/env python3
"""Single-Shot Unseen Forward Holdout Evaluation for Track 2B champion + LayaGate Filter Check.

Champion:
  - flow_student_t_settle: h0.05_w5_15_th0.30_taker
  - Frozen IS Student-t parameters from data/track2b-flow-student-t-is.json (ZERO refitting)
  - Frozen IS payoff model from data/track2b-flow-student-t-is-trades.jsonl (44 IS trades)

Target Data:
  - /home/cayleb/open-kal/data/evolve-cache/ (STRICTLY READ-ONLY)
  - Strictly unseen windows closing AFTER the frozen tape (close > 1794943521.0)
  - Live verdicts evaluated:
    (a) Unfiltered forward holdout
    (b) LayaGate-filtered forward holdout (gate_allow == True)
"""
from __future__ import annotations

import datetime
import itertools
import json
import math
import os
import sqlite3
import sys
import time
from collections import Counter
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path("/home/cayleb/open-kal-rnd")
sys.path.insert(0, str(ROOT))

from openkal import features  # noqa: E402
from openkal import sim_core as S  # noqa: E402
from scripts import alpha_discover as AD  # noqa: E402
from scripts import bounded_sweep as BS  # noqa: E402
from scripts import f2_structural as F2  # noqa: E402
from scripts import kurtosis_fv_snipe as KF  # noqa: E402
from scripts import rnd_common as RC  # noqa: E402

FROZEN_TAPE_MAX_TS = 1794943521.0
EVOLVE_CACHE = Path("/home/cayleb/open-kal/data/evolve-cache")
PROD_DB = Path("/home/cayleb/open-kal/data/openkal-prod.db")

IS_JSON = ROOT / "data" / "track2b-flow-student-t-is.json"
IS_TRADES = ROOT / "data" / "track2b-flow-student-t-is-trades.jsonl"
OUT_JSON = ROOT / "data" / "track2b-flow-student-t-forward-holdout.json"
OUT_TRADES = ROOT / "data" / "track2b-flow-student-t-forward-trades.jsonl"
LEDGER = ROOT / "data" / "oos-ledger.jsonl"

FAMILY = "flow_student_t_settle"
CONFIG_NAME = "h0.05_w5_15_th0.30_taker"
HURDLE = 0.05
FLOW_WINDOW = (5.0, 15.0)
FLOW_THRESH = 0.30
STYLE = "taker"

COST_LO, COST_HI = 0.35, 0.60
COST_LO_D, COST_HI_D = Decimal("0.35"), Decimal("0.60")
COOLDOWN, PER_WINDOW = 60.0, 2
FEATURES = ["p_side", "edge", "secs_left", "flow_fast", "flow_slow"]


def run_forward_holdout():
    if LEDGER.exists() and any(
        json.loads(line)["family"] in {"flow_student_t_settle_forward_holdout", "flow_student_t_settle_forward_layagate"}
        for line in LEDGER.read_text().splitlines() if line.strip()
    ):
        raise RuntimeError("forward holdout already scored; audit the saved fills without replaying")
    t0 = time.time()
    
    # Verify strict isolation
    assert EVOLVE_CACHE.exists(), f"Missing cache dir: {EVOLVE_CACHE}"
    assert PROD_DB.exists(), f"Missing prod db: {PROD_DB}"
    
    # 1. Load Frozen IS Configuration and Parameters
    is_meta = json.loads(IS_JSON.read_text())
    fit_params = is_meta["fit_params"]
    print("Loaded frozen IS Student-t parameters:")
    for lane_name, p in fit_params.items():
        print(f"  {lane_name}: nu={p['nu']}, s={p['s']}")
        
    is_trades_raw = [json.loads(line) for line in IS_TRADES.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(is_trades_raw)} frozen IS trades for payoff model calibration.")
    
    x_is = np.array([r["feat"] for r in is_trades_raw], dtype=np.float64)
    y_is = np.array([1 if Decimal(r["pnl"]) > 0 else 0 for r in is_trades_raw], dtype=np.float64)
    payoff_model = F2.fit_logistic(x_is, y_is)
    print(f"Frozen IS Payoff Model weights: {payoff_model[0]}")
    
    # 2. Point AD.CACHE to EVOLVE_CACHE
    AD.CACHE = EVOLVE_CACHE
    lanes = {n: AD.Lane(n) for n in RC.LANES}
    
    # Open prod db strictly read-only for gate verdicts
    db_uri = f"file:{PROD_DB.resolve()}?mode=ro"
    con = sqlite3.connect(db_uri, uri=True)
    cur = con.cursor()
    cur.execute("PRAGMA query_only=1")
    
    all_forward_rows = []
    skips = Counter()
    
    # Also load gate arrays from book_*.npz for cross-check
    gate_data = {}
    for lane_name in RC.LANES:
        npz_p = EVOLVE_CACHE / f"book_{lane_name}.npz"
        npz_d = np.load(npz_p, allow_pickle=False)
        gate_data[lane_name] = {
            "tick_ts": npz_d["tick_ts"],
            "gate_allow": npz_d["gate_allow"],
        }

    total_unseen_windows = set()
    
    for lane_name, lane in lanes.items():
        fp = fit_params[lane_name]
        nu = fp["nu"]
        s_fat = fp["s"]
        table = KF.t_cdf_table(nu)
        
        # Spot & Sigma causal prefixes
        its, v = KF.clean_index(lane.index, before=None)
        tv, C, N = KF.sigma_prefix(its, v)
        
        # Filter tickers strictly closing AFTER frozen tape
        unseen_tickers = []
        for tid, name in enumerate(lane.tickers):
            m = lane.meta.get(name)
            if m is None or m.get("close", 0) <= FROZEN_TAPE_MAX_TS:
                continue
            if m.get("y") is None or not m.get("strike"):
                continue
            unseen_tickers.append((tid, name, m))
            total_unseen_windows.add(m["close"])
            
        print(f"Lane {lane_name}: found {len(unseen_tickers)} unseen markets closing > {FROZEN_TAPE_MAX_TS}")
        
        # Pre-collect samples for unseen markets
        tid_l, t_l, secs_l, close_l = [], [], [], []
        cy_taker_l, cn_taker_l = [], []
        p_yes_l, p_no_l = [], []
        f_fast_l, f_slow_l = [], []
        
        secs_grid = np.arange(780.0, 59.0, -2.0)
        
        for tid, name, m in unseen_tickers:
            tape = RC.tape(lane, tid)
            close, strike = m["close"], m["strike"]
            
            lo, hi = int(lane.trade_bounds[tid]), int(lane.trade_bounds[tid + 1])
            tts = lane.trade_ts[lo:hi]
            sz = lane.trade_sz[lo:hi]
            signed = np.where(lane.trade_ty[lo:hi] == 1, sz, -sz)
            cs = np.concatenate(([0.0], np.cumsum(signed)))
            ct = np.concatenate(([0.0], np.cumsum(sz)))
            n_trades = len(tts)
            
            for secs in secs_grid:
                t = close - secs
                spot = KF.index_last(its, v, t)
                if not math.isfinite(spot) or spot <= 0:
                    skips["no_spot"] += 1
                    continue
                    
                try:
                    b, a = tape.quote(t)
                except S.CrossedBook:
                    skips["quote_crossed"] += 1
                    continue
                except S.NoQuote:
                    skips["quote_no_quote"] += 1
                    continue
                    
                sig = float(KF.sigma_at(t, tv, C, N))
                tau_eff = max(secs - 40.0, 1.0)
                vol = sig * math.sqrt(tau_eff)
                if vol <= 0:
                    skips["zero_vol"] += 1
                    continue
                    
                x = math.log(strike / spot) / vol
                p_yes = 1.0 - float(np.interp(x / s_fat, KF.CDF_X, table))
                p_no = 1.0 - p_yes
                
                cy_t = float(S.entry_price(S.YES, S.TAKER, b, a))
                cn_t = float(S.entry_price(S.NO, S.TAKER, b, a))
                
                tid_l.append(tid)
                t_l.append(t)
                secs_l.append(secs)
                close_l.append(close)
                cy_taker_l.append(cy_t)
                cn_taker_l.append(cn_t)
                p_yes_l.append(p_yes)
                p_no_l.append(p_no)
                
                if n_trades > 0:
                    af = int(np.searchsorted(tts, t - FLOW_WINDOW[0], "left"))
                    bf = int(np.searchsorted(tts, t, "right"))
                    denf = ct[bf] - ct[af]
                    ff = float((cs[bf] - cs[af]) / denf) if denf > 0 else 0.0
                    
                    asl = int(np.searchsorted(tts, t - FLOW_WINDOW[1], "left"))
                    bsl = int(np.searchsorted(tts, t, "right"))
                    dens = ct[bsl] - ct[asl]
                    fs = float((cs[bsl] - cs[asl]) / dens) if dens > 0 else 0.0
                else:
                    ff, fs = 0.0, 0.0
                    
                f_fast_l.append(ff)
                f_slow_l.append(fs)
                
        if not tid_l:
            continue
            
        tid_arr = np.asarray(tid_l, dtype=np.int64)
        t_arr = np.asarray(t_l)
        secs_arr = np.asarray(secs_l)
        close_arr = np.asarray(close_l)
        cy_arr = np.asarray(cy_taker_l)
        cn_arr = np.asarray(cn_taker_l)
        py_arr = np.asarray(p_yes_l)
        pn_arr = np.asarray(p_no_l)
        ff_arr = np.asarray(f_fast_l)
        fs_arr = np.asarray(f_slow_l)
        
        ey = py_arr - cy_arr
        en = pn_arr - cn_arr
        
        oky = (cy_arr >= COST_LO) & (cy_arr <= COST_HI) & (ey >= HURDLE) & (ff_arr >= FLOW_THRESH) & (fs_arr >= FLOW_THRESH)
        okn = (cn_arr >= COST_LO) & (cn_arr <= COST_HI) & (en >= HURDLE) & (ff_arr <= -FLOW_THRESH) & (fs_arr <= -FLOW_THRESH)
        
        mask = oky | okn
        if not np.any(mask):
            continue
            
        use_y = oky & (~okn | (ey >= en))
        keep = BS.pick({"tid": tid_arr, "ts": t_arr}, mask, cooldown=COOLDOWN, per_window=PER_WINDOW)
        
        print(f"Lane {lane_name}: {len(keep)} candidate triggers passed threshold & cooldown.")
        
        # Simulate trades through sim_core
        for idx in keep:
            side = S.YES if use_y[idx] else S.NO
            p_side = float(py_arr[idx]) if side == S.YES else float(pn_arr[idx])
            cost = float(cy_arr[idx]) if side == S.YES else float(cn_arr[idx])
            edge = p_side - cost
            ff = float(ff_arr[idx])
            fs = float(fs_arr[idx])
            secs = float(secs_arr[idx])
            t = float(t_arr[idx])
            close = float(close_arr[idx])
            tid = int(tid_arr[idx])
            ticker_name = lane.tickers[tid]
            feat = [p_side, edge, secs, ff, fs]
            
            tape = RC.tape(lane, tid)
            try:
                bq, aq = tape.quote(t + S.LAG_S)
                cost_arr = S.entry_price(side, S.TAKER, bq, aq)
                qty = S.stake_qty(cost_arr, RC.STAKE)
            except (S.CrossedBook, S.NoQuote):
                qty = S.MAX_QTY
                
            res = S.run_trade(tape, side=side, trigger_ts=t, entry_style=S.TAKER, qty=qty,
                              patience_s=0.0, cost_lo=COST_LO_D, cost_hi=COST_HI_D)
                              
            if isinstance(res, S.Skip):
                skips[res.reason] += 1
                continue
                
            # Query gate verdict from DB and NPZ
            cur.execute("""
                SELECT ts, gate FROM ticks
                WHERE series = ? AND ts <= ? AND gate IS NOT NULL
                ORDER BY ts DESC LIMIT 1
            """, (lane_name, t))
            db_row = cur.fetchone()
            gate_dict = None
            db_allow = None
            if db_row and db_row[1]:
                try:
                    gate_dict = json.loads(db_row[1])
                    db_allow = gate_dict.get("allow")
                except Exception:
                    pass
                    
            # NPZ check
            tts_lane = gate_data[lane_name]["tick_ts"]
            ga_lane = gate_data[lane_name]["gate_allow"]
            npz_idx = int(np.searchsorted(tts_lane, t, side="right") - 1)
            npz_allow = None
            if npz_idx >= 0:
                raw_ga = ga_lane[npz_idx]
                if raw_ga == 1:
                    npz_allow = True
                elif raw_ga == 0:
                    npz_allow = False
                    
            allow_verdict = db_allow if db_allow is not None else npz_allow
            
            row = RC.trade_row(
                res, lane=lane_name, ticker=ticker_name, close=close,
                slice_name="unseen_forward_holdout", feat=feat, config=CONFIG_NAME,
                family=FAMILY, hurdle=HURDLE, p_side=p_side, edge=round(edge, 6)
            )
            row["gate_dict"] = gate_dict
            row["gate_allow"] = allow_verdict
            row["gate_allow_db"] = db_allow
            row["gate_allow_npz"] = npz_allow
            all_forward_rows.append(row)
            
    print(f"\nExecution finished in {time.time() - t0:.1f}s.")
    print(f"Total unseen forward windows evaluated: {len(total_unseen_windows)}")
    print(f"Total unfiltered forward holdout trades: {len(all_forward_rows)}")
    
    # Save all forward trades
    OUT_TRADES.parent.mkdir(parents=True, exist_ok=True)
    OUT_TRADES.write_text("".join(json.dumps(r, default=RC._json_default) + "\n" for r in all_forward_rows))
    
    # 3. Analyze (a) Unfiltered forward holdout
    unfiltered_summary = RC.summarize(all_forward_rows, dict(skips))
    
    # Compute Payoff-ECE using frozen IS payoff model
    if all_forward_rows:
        x_fwd = np.array([r["feat"] for r in all_forward_rows], dtype=np.float64)
        p_pred_5f = F2.apply_logistic(x_fwd, payoff_model[0], payoff_model[1])
        won_fwd = [1 if Decimal(r["pnl"]) > 0 else 0 for r in all_forward_rows]
        ece_unfiltered = round(features.expected_calibration_error(list(zip(p_pred_5f.tolist(), won_fwd))), 4)
        
        p_raw_fwd = [r["p_side"] for r in all_forward_rows]
        ece_unfiltered_raw = round(features.expected_calibration_error(list(zip(p_raw_fwd, won_fwd))), 4)
    else:
        ece_unfiltered = None
        ece_unfiltered_raw = None
        
    # Net ex-XRP
    net_ex_xrp = sum(
        v["net"] for k, v in unfiltered_summary["by_lane"].items() if k != "KXXRP15M"
    )
    unfiltered_summary["net_ex_xrp"] = round(net_ex_xrp, 2)
    unfiltered_summary["payoff_ece"] = ece_unfiltered
    unfiltered_summary["raw_p_side_ece"] = ece_unfiltered_raw
    
    # 4. Analyze (b) LayaGate-filtered forward holdout (gate_allow == True)
    # Note: trades where gate_allow is True
    gate_allowed_rows = [r for r in all_forward_rows if r["gate_allow"] is True]
    gate_denied_rows = [r for r in all_forward_rows if r["gate_allow"] is False]
    gate_unlogged_rows = [r for r in all_forward_rows if r["gate_allow"] is None]
    
    print(f"\nGate classification:")
    print(f"  Allowed (gate_allow == True):   {len(gate_allowed_rows)}")
    print(f"  Denied  (gate_allow == False):  {len(gate_denied_rows)}")
    print(f"  Unlogged/Offline:               {len(gate_unlogged_rows)}")
    
    filtered_summary = RC.summarize(gate_allowed_rows)
    if gate_allowed_rows:
        x_filt = np.array([r["feat"] for r in gate_allowed_rows], dtype=np.float64)
        p_pred_filt = F2.apply_logistic(x_filt, payoff_model[0], payoff_model[1])
        won_filt = [1 if Decimal(r["pnl"]) > 0 else 0 for r in gate_allowed_rows]
        ece_filtered = round(features.expected_calibration_error(list(zip(p_pred_filt.tolist(), won_filt))), 4)
        
        p_raw_filt = [r["p_side"] for r in gate_allowed_rows]
        ece_filtered_raw = round(features.expected_calibration_error(list(zip(p_raw_filt, won_filt))), 4)
    else:
        ece_filtered = None
        ece_filtered_raw = None
        
    net_ex_xrp_filt = sum(
        v["net"] for k, v in filtered_summary["by_lane"].items() if k != "KXXRP15M"
    )
    filtered_summary["net_ex_xrp"] = round(net_ex_xrp_filt, 2)
    filtered_summary["payoff_ece"] = ece_filtered
    filtered_summary["raw_p_side_ece"] = ece_filtered_raw
    
    # Check gate verdicts on denied rows
    denied_summary = RC.summarize(gate_denied_rows)
    
    # 5. Build report structure
    report = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "holdout_window": {
            "start_close_ts": min(total_unseen_windows) if total_unseen_windows else None,
            "end_close_ts": max(total_unseen_windows) if total_unseen_windows else None,
            "distinct_windows": len(total_unseen_windows),
            "reference_frozen_tape_max": FROZEN_TAPE_MAX_TS,
        },
        "config": {
            "family": FAMILY,
            "config": CONFIG_NAME,
            "hurdle": HURDLE,
            "flow_window": list(FLOW_WINDOW),
            "flow_thresh": FLOW_THRESH,
            "style": STYLE,
            "exit": "settle",
        },
        "unfiltered_forward_holdout": unfiltered_summary,
        "layagate_filtered_forward_holdout": filtered_summary,
        "layagate_denied_trades": denied_summary,
        "gate_counts": {
            "total_trades": len(all_forward_rows),
            "allowed": len(gate_allowed_rows),
            "denied": len(gate_denied_rows),
            "unlogged": len(gate_unlogged_rows),
        },
    }
    
    RC.dump(OUT_JSON, report)
    
    # 6. Log single-shot unseen holdout evaluation to data/oos-ledger.jsonl
    # Log unfiltered entry
    ledger_entry_unfiltered = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "family": "flow_student_t_settle_forward_holdout",
        "status": "scored",
        "config": report["config"],
        "rescore": False,
        "rescore_reason": "unseen_forward_holdout_1620_present",
        "oos": unfiltered_summary,
        "payoff_ece": ece_unfiltered,
        "gate": RC.promotion_gate(unfiltered_summary, ece_unfiltered),
        "cache": str(EVOLVE_CACHE),
    }
    
    # Log filtered entry
    ledger_entry_filtered = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "family": "flow_student_t_settle_forward_layagate",
        "status": "scored",
        "config": report["config"],
        "rescore": False,
        "rescore_reason": "unseen_forward_holdout_layagate_filtered",
        "oos": filtered_summary,
        "payoff_ece": ece_filtered,
        "gate": RC.promotion_gate(filtered_summary, ece_filtered),
        "cache": str(EVOLVE_CACHE),
    }
    
    with LEDGER.open("a") as f:
        f.write(json.dumps(ledger_entry_unfiltered, default=RC._json_default) + "\n")
        f.write(json.dumps(ledger_entry_filtered, default=RC._json_default) + "\n")
        
    print("\nSuccessfully logged both entries to data/oos-ledger.jsonl!")
    return report

if __name__ == "__main__":
    rep = run_forward_holdout()
    print(json.dumps(rep, indent=2, default=RC._json_default))
