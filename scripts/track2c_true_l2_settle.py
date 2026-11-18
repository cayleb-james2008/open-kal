#!/usr/bin/env python3
"""Charter 3 Track 2C - `true_l2_settle`: True L2 Book-Imbalance + Settlement Fair Value.

Mechanism:
  - Ingest true L2 depth (yes_bid_sz, yes_ask_sz, microprice) from evolve-cache.
  - Require >= 24 settled windows (>= 6 hours) of non-NaN true-L2 depth.
  - Chronological 60% IS / 40% OOS split on true-L2 windows.
  - True L2 order book imbalance (z_l2_obi) + Student-t settlement fair value.
  - Entry cost in [$0.35, $0.60], hold to $1.00 settlement ($0 exit fee).

Evaluation:
  - Strict chronological 60% IS (>= 30 IS trades, IS PF >= 1.25, >= 2 profitable lanes).
  - Promotion bar: OOS net > 0, OOS PF >= 1.25, payoff-ECE <= 0.08, >= 25 OOS trades across >= 2 profitable coins, profitable excluding XRP (net_ex_xrp > 0), 0 crossed fills, max_entry_cost <= $0.62.
  - Single sealed OOS look logged to data/oos-ledger.jsonl.
  - Updates RND_REPORT.md + NEXT.json and commits to git.
"""
from __future__ import annotations

import argparse
import datetime
import itertools
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openkal import sim_core as S  # noqa: E402
from scripts import alpha_discover as AD  # noqa: E402
from scripts import bounded_sweep as BS  # noqa: E402
from scripts import kurtosis_fv_snipe as KF  # noqa: E402
from scripts import rnd_common as RC  # noqa: E402

EVOLVE_CACHE = Path("/home/cayleb/open-kal/data/evolve-cache")
OUT = ROOT / "data" / "track2c-true-l2-settle-is.json"
TRADES = ROOT / "data" / "track2c-true-l2-settle-is-trades.jsonl"
LEDGER = ROOT / "data" / "oos-ledger.jsonl"
RND_REPORT = ROOT / "RND_REPORT.md"
NEXT_JSON = ROOT / "NEXT.json"

FAMILY = "true_l2_settle"
HURDLES = (0.03, 0.05, 0.07)
OBI_THRESHOLDS = (0.15, 0.30, 0.45)
FLOW_WINDOWS = ((5.0, 15.0), (10.0, 30.0))
STYLES = ("taker", "maker")

COST_LO, COST_HI = 0.35, 0.60
COST_LO_D, COST_HI_D = Decimal("0.35"), Decimal("0.60")
COOLDOWN, PER_WINDOW = 60.0, 2
MIN_TRADES, MIN_PF, MIN_LANES = 30, 1.25, 2
MIN_WINDOWS_REQUIRED = 24  # >= 24 settled windows (>= 6 hours)
FEATURES = ["p_side", "edge", "secs_left", "l2_obi", "flow_fast"]


def config_name(hurdle: float, obi_th: float, fw: tuple[float, float], style: str) -> str:
    return f"h{hurdle:.2f}_obi{obi_th:.2f}_w{int(fw[0])}_{int(fw[1])}_{style}"


def parse_config(name: str) -> dict:
    parts = name.split("_")
    h = float(parts[0][1:])
    obi_th = float(parts[1][3:])
    w0 = float(parts[2][1:])
    w1 = float(parts[3])
    style = parts[4]
    return {"hurdle": h, "obi_thresh": obi_th, "flow_windows": (w0, w1), "style": style}


def audit_l2_depth(cache_dir: Path = EVOLVE_CACHE) -> dict:
    """Inspect book_KXBTC15M.npz for true L2 depth columns and count windows with non-NaN values."""
    btc_path = cache_dir / "book_KXBTC15M.npz"
    if not btc_path.exists():
        return {"status": "no_file", "path": str(btc_path), "windows_with_l2": 0}

    data = np.load(btc_path, allow_pickle=False)
    keys = list(data.keys())
    has_keys = all(k in keys for k in ("yes_bid_sz", "yes_ask_sz", "microprice"))
    if not has_keys:
        return {"status": "missing_keys", "keys": keys, "windows_with_l2": 0}

    bid_sz = data["yes_bid_sz"]
    ask_sz = data["yes_ask_sz"]
    micro = data["microprice"]

    non_nan_mask = np.isfinite(bid_sz) & np.isfinite(ask_sz) & (bid_sz > 0) & (ask_sz > 0)
    total_valid_rows = int(np.sum(non_nan_mask))

    if total_valid_rows == 0:
        return {
            "status": "zero_valid_l2_rows",
            "total_rows": len(bid_sz),
            "valid_l2_rows": 0,
            "windows_with_l2": 0,
        }

    # Count distinct settled windows with L2 data
    tid = data["tid"]
    tids_with_l2 = np.unique(tid[non_nan_mask])
    tickers = [str(t) for t in data["tickers"]]

    markets_path = cache_dir / "markets.json"
    windows = set()
    if markets_path.exists():
        meta = json.loads(markets_path.read_text())
        for idx in tids_with_l2:
            if idx < len(tickers):
                t_name = tickers[idx]
                if t_name in meta and "close" in meta[t_name] and meta[t_name].get("y") is not None:
                    windows.add(meta[t_name]["close"])

    return {
        "status": "ready" if len(windows) >= MIN_WINDOWS_REQUIRED else "accumulating",
        "total_rows": len(bid_sz),
        "valid_l2_rows": total_valid_rows,
        "windows_with_l2": len(windows),
        "windows_required": MIN_WINDOWS_REQUIRED,
        "l2_window_closes": sorted(windows),
    }


def fit_all_lanes_l2(lanes: dict, cut: float) -> dict:
    """Fit Student-t parameters using causal index history before cut."""
    fit = {}
    for name in RC.LANES:
        its, v = KF.clean_index(lanes[name].index, before=cut)
        res = KF.fit_lane(its, v, cut)
        fit[name] = res
    return fit


def collect_l2_lane_samples(lane, fit_params: dict, valid_closes: set[float], cut: float, split: str) -> dict:
    its, v = KF.clean_index(lane.index, before=cut if split == "is" else None)
    tv, C, N = KF.sigma_prefix(its, v)

    nu = fit_params["nu"]
    s_fat = fit_params["s"]
    table = KF.t_cdf_table(nu)

    # Load L2 depth arrays from npz
    npz_data = np.load(EVOLVE_CACHE / f"book_{lane.lane}.npz", allow_pickle=False)
    bid_sz_all = npz_data["yes_bid_sz"]
    ask_sz_all = npz_data["yes_ask_sz"]

    skips: Counter = Counter()
    tid_l, t_l, secs_l, close_l = [], [], [], []
    cy_taker_l, cn_taker_l = [], []
    cy_maker_l, cn_maker_l = [], []
    p_yes_l, p_no_l = [], []
    obi_l = []
    f_fast_l, f_slow_l = {}, {}
    for fw in FLOW_WINDOWS:
        f_fast_l[fw] = []
        f_slow_l[fw] = []

    secs_grid = np.arange(780.0, 59.0, -2.0)
    n_tickers = 0

    for tid, name in enumerate(lane.tickers):
        m = lane.meta.get(name)
        if m is None or m.get("y") is None or not m.get("strike"):
            continue
        close = float(m["close"])
        if close not in valid_closes:
            continue
        is_split = "is" if close < cut else "oos"
        if is_split != split:
            continue

        n_tickers += 1
        tape = RC.tape(lane, tid)
        strike = float(m["strike"])

        a_bk, b_bk = int(lane.book_bounds[tid]), int(lane.book_bounds[tid + 1])
        bk_ts = lane.ts[a_bk:b_bk]
        bk_bid_sz = bid_sz_all[a_bk:b_bk]
        bk_ask_sz = ask_sz_all[a_bk:b_bk]

        lo_tr, hi_tr = int(lane.trade_bounds[tid]), int(lane.trade_bounds[tid + 1])
        tts = lane.trade_ts[lo_tr:hi_tr]
        sz = lane.trade_sz[lo_tr:hi_tr]
        signed = np.where(lane.trade_ty[lo_tr:hi_tr] == 1, sz, -sz)
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

            # Look up L2 depth at t
            j_bk = int(np.searchsorted(bk_ts, t, side="right")) - 1
            if j_bk < 0:
                skips["no_l2"] += 1
                continue
            bsz = float(bk_bid_sz[j_bk])
            asz = float(bk_ask_sz[j_bk])
            if not (math.isfinite(bsz) and math.isfinite(asz) and bsz > 0 and asz > 0):
                skips["nan_l2"] += 1
                continue

            l2_obi = (bsz - asz) / (bsz + asz)

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
            cy_m = float(S.entry_price(S.YES, S.MAKER, b, a))
            cn_m = float(S.entry_price(S.NO, S.MAKER, b, a))

            tid_l.append(tid)
            t_l.append(t)
            secs_l.append(secs)
            close_l.append(close)
            cy_taker_l.append(cy_t)
            cn_taker_l.append(cn_t)
            cy_maker_l.append(cy_m)
            cn_maker_l.append(cn_m)
            p_yes_l.append(p_yes)
            p_no_l.append(p_no)
            obi_l.append(l2_obi)

            for fw in FLOW_WINDOWS:
                w_fast, w_slow = fw
                if n_trades > 0:
                    af = int(np.searchsorted(tts, t - w_fast, "left"))
                    bf = int(np.searchsorted(tts, t, "right"))
                    denf = ct[bf] - ct[af]
                    ff = float((cs[bf] - cs[af]) / denf) if denf > 0 else 0.0

                    asl = int(np.searchsorted(tts, t - w_slow, "left"))
                    bsl = int(np.searchsorted(tts, t, "right"))
                    dens = ct[bsl] - ct[asl]
                    fs = float((cs[bsl] - cs[asl]) / dens) if dens > 0 else 0.0
                else:
                    ff, fs = 0.0, 0.0
                f_fast_l[fw].append(ff)
                f_slow_l[fw].append(fs)

    return {
        "lane": lane,
        "split": split,
        "tickers": n_tickers,
        "tid": np.asarray(tid_l, dtype=np.int64),
        "t": np.asarray(t_l),
        "secs": np.asarray(secs_l),
        "close": np.asarray(close_l),
        "cost_yes_taker": np.asarray(cy_taker_l),
        "cost_no_taker": np.asarray(cn_taker_l),
        "cost_yes_maker": np.asarray(cy_maker_l),
        "cost_no_maker": np.asarray(cn_maker_l),
        "p_yes": np.asarray(p_yes_l),
        "p_no": np.asarray(p_no_l),
        "l2_obi": np.asarray(obi_l),
        "flow_fast": {fw: np.asarray(f_fast_l[fw]) for fw in FLOW_WINDOWS},
        "flow_slow": {fw: np.asarray(f_slow_l[fw]) for fw in FLOW_WINDOWS},
        "skips": skips,
    }


def find_l2_triggers(pack: dict, hurdle: float, obi_th: float,
                     fw: tuple[float, float], style: str) -> list[tuple[int, str, float, float, float, float]]:
    p_yes = pack["p_yes"]
    p_no = pack["p_no"]
    obi = pack["l2_obi"]
    ff = pack["flow_fast"][fw]

    if style == "taker":
        cy = pack["cost_yes_taker"]
        cn = pack["cost_no_taker"]
    else:
        cy = pack["cost_yes_maker"]
        cn = pack["cost_no_maker"]

    ey = p_yes - cy
    en = p_no - cn

    # YES condition: cost in [0.35, 0.60], edge >= hurdle, L2 OBI >= obi_th, flow >= 0
    oky = (cy >= COST_LO) & (cy <= COST_HI) & (ey >= hurdle) & (obi >= obi_th) & (ff >= 0.0)
    # NO condition: cost in [0.35, 0.60], edge >= hurdle, L2 OBI <= -obi_th, flow <= 0
    okn = (cn >= COST_LO) & (cn <= COST_HI) & (en >= hurdle) & (obi <= -obi_th) & (ff <= 0.0)

    mask = oky | okn
    if not np.any(mask):
        return []

    use_y = oky & (~okn | (ey >= en))
    keep = BS.pick({"tid": pack["tid"], "ts": pack["t"]}, mask,
                   cooldown=COOLDOWN, per_window=PER_WINDOW)

    out = []
    for i in keep:
        side = S.YES if use_y[i] else S.NO
        p_side = float(p_yes[i]) if side == S.YES else float(p_no[i])
        cost = float(cy[i]) if side == S.YES else float(cn[i])
        edge = p_side - cost
        out.append((i, side, p_side, edge, float(obi[i]), float(ff[i])))
    return out


def run_l2_triggers(pack: dict, triggers: list[tuple[int, str, float, float, float, float]],
                    cfg_name: str, style: str, hurdle: float, out_dict: dict) -> None:
    lane = pack["lane"]
    entry_style = S.TAKER if style == "taker" else S.MAKER
    patience = 5.0 if style == "maker" else 0.0

    for i, side, p_side, edge, obi, ff in triggers:
        tid = int(pack["tid"][i])
        t = float(pack["t"][i])
        name = lane.tickers[tid]
        feat = [p_side, edge, float(pack["secs"][i]), obi, ff]
        tape = RC.tape(lane, tid)

        try:
            bq, aq = tape.quote(t + S.LAG_S)
            cost_arr = S.entry_price(side, entry_style, bq, aq)
            qty = S.stake_qty(cost_arr, RC.STAKE)
        except (S.CrossedBook, S.NoQuote):
            qty = S.MAX_QTY

        res = S.run_trade(tape, side=side, trigger_ts=t, entry_style=entry_style, qty=qty,
                          patience_s=patience, cost_lo=COST_LO_D, cost_hi=COST_HI_D)

        if isinstance(res, S.Skip):
            out_dict[cfg_name]["skips"][res.reason] += 1
            continue

        out_dict[cfg_name]["rows"].append(RC.trade_row(
            res, lane=lane.lane, ticker=name, close=float(pack["close"][i]),
            slice_name=pack["split"], feat=feat, config=cfg_name, family=FAMILY,
            hurdle=hurdle, p_side=p_side, edge=round(edge, 6)))


def eligible(s: dict) -> bool:
    return (s["trades"] >= MIN_TRADES and s["pf"] >= MIN_PF
            and s["profitable_lanes"] >= MIN_LANES and s["crossed_fills"] == 0)


def pick_champion(configs: dict) -> dict:
    el = [c for c in configs if eligible(configs[c]["is"])]
    if el:
        best = max(el, key=lambda c: configs[c]["is"]["net"])
        return {"config": best, "is_eligible": True, "is": configs[best]["is"]}
    best = max(configs, key=lambda c: configs[c]["is"]["net"])
    return {"config": best, "is_eligible": False, "is": configs[best]["is"]}


def run_full_l2_sweep(audit: dict) -> dict:
    t0 = time.time()
    valid_closes = set(audit["l2_window_closes"])
    sorted_closes = sorted(valid_closes)
    cut_idx = int(len(sorted_closes) * 0.6)
    cut = sorted_closes[cut_idx]

    AD.CACHE = EVOLVE_CACHE
    lanes = {n: AD.Lane(n) for n in RC.LANES}

    fit_params = fit_all_lanes_l2(lanes, cut)
    packs = [collect_l2_lane_samples(lane, fit_params[lane.lane], valid_closes, cut, "is")
             for lane in lanes.values()]

    all_combos = list(itertools.product(HURDLES, OBI_THRESHOLDS, FLOW_WINDOWS, STYLES))
    cfg_names = [config_name(h, obi, fw, st) for h, obi, fw, st in all_combos]
    out_dict = {cn: {"rows": [], "skips": Counter()} for cn in cfg_names}

    for pack in packs:
        for h, obi, fw, st in all_combos:
            cn = config_name(h, obi, fw, st)
            trig = find_l2_triggers(pack, h, obi, fw, st)
            run_l2_triggers(pack, trig, cn, st, h, out_dict)

    configs = {}
    for h, obi, fw, st in all_combos:
        cn = config_name(h, obi, fw, st)
        configs[cn] = {
            "family": FAMILY,
            "hurdle": h,
            "obi_thresh": obi,
            "flow_windows": list(fw),
            "style": st,
            "is": RC.summarize(out_dict[cn]["rows"], dict(out_dict[cn]["skips"])),
        }

    champion = pick_champion(configs)
    champ_rows = out_dict[champion["config"]]["rows"]
    TRADES.parent.mkdir(parents=True, exist_ok=True)
    TRADES.write_text("".join(json.dumps(r, default=RC._json_default) + "\n" for r in champ_rows))

    obj = {
        "family": FAMILY,
        "split": f"IS only: {cut_idx} windows before cut={cut}",
        "cut": cut,
        "total_l2_windows": len(sorted_closes),
        "is_windows": cut_idx,
        "oos_windows": len(sorted_closes) - cut_idx,
        "lanes": list(lanes),
        "fit_params": fit_params,
        "configs": configs,
        "champion": champion,
        "runtime_s": round(time.time() - t0, 1),
    }
    RC.dump(OUT, obj)

    # If eligible, score single sealed OOS look
    if champion["is_eligible"]:
        print(f"IS Champion {champion['config']} is eligible! Scoring single sealed OOS look...")
        cfg = parse_config(champion["config"])
        oos_packs = [collect_l2_lane_samples(lane, fit_params[lane.lane], valid_closes, cut, "oos")
                     for lane in lanes.values()]
        oos_dict = {champion["config"]: {"rows": [], "skips": Counter()}}
        for pack in oos_packs:
            trig = find_l2_triggers(pack, cfg["hurdle"], cfg["obi_thresh"], cfg["flow_windows"], cfg["style"])
            run_l2_triggers(pack, trig, champion["config"], cfg["style"], cfg["hurdle"], oos_dict)

        oos_rows = oos_dict[champion["config"]]["rows"]
        oos_skips = dict(oos_dict[champion["config"]]["skips"])
        payoff_model = RC.fit_payoff_model(champ_rows)
        oos_summary = RC.summarize(oos_rows, oos_skips)
        ece = RC.payoff_ece(payoff_model, oos_rows)
        gate_res = RC.promotion_gate(oos_summary, ece)

        entry = {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "family": FAMILY,
            "status": "scored",
            "config": {"config": champion["config"], **cfg, "exit": "settle"},
            "rescore": False,
            "rescore_reason": None,
            "oos": oos_summary,
            "payoff_ece": ece,
            "gate": gate_res,
            "cache": str(EVOLVE_CACHE),
        }
        with LEDGER.open("a") as f:
            f.write(json.dumps(entry, default=RC._json_default) + "\n")
        obj["oos_scored"] = entry
    else:
        print(f"IS Champion {champion['config']} did not meet eligibility bar ({MIN_TRADES}+ trades, PF >= {MIN_PF}, {MIN_LANES}+ lanes). OOS look unspent.")
        obj["oos_scored"] = None

    # Update NEXT.json
    try:
        next_data = json.loads(NEXT_JSON.read_text())
        next_data["rnd_overnight_shift"]["track2c_true_l2_loop"] = {
            "status": "Swept" if champion["is_eligible"] else "IS fail",
            "current_windows": len(sorted_closes),
            "required_windows": MIN_WINDOWS_REQUIRED,
            "champion": champion["config"],
            "is_net": champion["is"]["net"],
            "is_pf": champion["is"]["pf"],
            "is_trades": champion["is"]["trades"],
            "oos_spent": champion["is_eligible"],
            "gate_pass": obj["oos_scored"]["gate"]["pass"] if obj.get("oos_scored") else False,
        }
        RC.dump(NEXT_JSON, next_data)
    except Exception as e:
        print(f"Failed to update NEXT.json: {e}")

    # Append to RND_REPORT.md
    try:
        with RND_REPORT.open("a") as f:
            f.write(f"\n\n### Track 2C — `true_l2_settle` Sweep ({len(sorted_closes)} Windows Accumulated)\n")
            f.write(f"- Champion: `{champion['config']}` (IS: {champion['is']['trades']} trades, net +${champion['is']['net']:.2f}, PF {champion['is']['pf']:.2f})\n")
            f.write(f"- Eligibility: {champion['is_eligible']}\n")
            if obj.get("oos_scored"):
                oos_res = obj["oos_scored"]["oos"]
                f.write(f"- OOS: {oos_res['trades']} trades, net +${oos_res['net']:.2f}, PF {oos_res['pf']:.2f}, ECE {obj['oos_scored']['payoff_ece']}\n")
                f.write(f"- Gate Verdict: {obj['oos_scored']['gate']}\n")
            else:
                f.write("- OOS look preserved unspent.\n")
    except Exception as e:
        print(f"Failed to append to RND_REPORT.md: {e}")

    # Commit & push
    try:
        subprocess.run(["git", "add", "data/", "NEXT.json", "RND_REPORT.md", "scripts/"], cwd=str(ROOT), check=True)
        subprocess.run(["git", "commit", "-m", f"Track 2C true-L2 sweep: {champion['config']}"], cwd=str(ROOT), check=True)
        subprocess.run(["git", "push", "origin", "rnd/alpha-evolve"], cwd=str(ROOT), check=True)
        print("Successfully committed and pushed Track 2C results to origin/rnd/alpha-evolve!")
    except Exception as e:
        print(f"Git sync notice: {e}")

    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    audit = audit_l2_depth()
    print(json.dumps(audit, indent=2, default=RC._json_default))

    if audit["windows_with_l2"] < MIN_WINDOWS_REQUIRED:
        print(f"Waiting for >= {MIN_WINDOWS_REQUIRED} true-L2 settled windows (currently {audit['windows_with_l2']} windows with non-NaN depth).")
        return 0

    print(f"Prerequisite met: {audit['windows_with_l2']} >= {MIN_WINDOWS_REQUIRED} windows of true L2 depth present! Launching full sweep...")
    run_full_l2_sweep(audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
