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
import fcntl
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from zipfile import BadZipFile
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
OOS_TRADES = ROOT / "data" / "track2c-true-l2-settle-oos-trades.jsonl"
LEDGER = ROOT / "data" / "oos-ledger.jsonl"
RND_REPORT = ROOT / "RND_REPORT.md"
NEXT_JSON = ROOT / "NEXT.json"
PRIVATE_GIT = ROOT / "data" / "overnight-rnd.git"
BRANCH = "rnd/alpha-evolve"

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
FEATURES = ["p_side", "edge", "secs_left", "z_l2_obi", "flow_fast"]
OBI_LOOKBACK_S, MIN_OBI_OBS, MAX_L2_AGE_S = 150.0, 5, 2.0


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


def valid_size(size):
    return np.isfinite(size) & (size > 0)


def valid_microprice(micro):
    return np.isfinite(micro) & (micro > 0) & (micro < 1)


def audit_l2_depth(cache_dir: Path = EVOLVE_CACHE, *, as_of: float | None = None) -> dict:
    """Count BTC settles only where both sizes and microprice coexist on a book row."""
    as_of = time.time() if as_of is None else as_of
    btc_path = cache_dir / "book_KXBTC15M.npz"
    if not btc_path.exists():
        return {"status": "no_file", "path": str(btc_path), "windows_with_l2": 0}

    with btc_path.open("rb") as f, np.load(f, allow_pickle=False) as data:
        needed = {"yes_bid_sz", "yes_ask_sz", "microprice", "tid", "tickers"}
        if needed.difference(data.files):
            return {"status": "missing_keys", "keys": data.files, "windows_with_l2": 0}
        valid = valid_size(data["yes_bid_sz"])
        valid &= valid_size(data["yes_ask_sz"])
        valid &= valid_microprice(data["microprice"])
        total_valid_rows = int(valid.sum())
        if total_valid_rows:
            ids = np.unique(data["tid"][valid])
            tickers = data["tickers"]
        total_rows = len(valid)

    if total_valid_rows == 0:
        return {
            "status": "zero_valid_l2_rows",
            "total_rows": total_rows,
            "valid_l2_rows": 0,
            "windows_with_l2": 0,
        }

    markets_path = cache_dir / "markets.json"
    if not markets_path.exists():
        return {"status": "no_markets", "valid_l2_rows": total_valid_rows, "windows_with_l2": 0}
    meta = json.loads(markets_path.read_text())
    if int(ids[0]) < 0 or int(ids[-1]) >= len(tickers):
        raise ValueError(f"{btc_path}: invalid ticker IDs on true-depth rows")
    windows = {float(m["close"]) for idx in ids if (m := meta.get(str(tickers[idx])))
               and m.get("y") in (0, 1) and 0 < m["close"] <= as_of}

    return {
        "status": "ready" if len(windows) >= MIN_WINDOWS_REQUIRED else "accumulating",
        "total_rows": total_rows,
        "valid_l2_rows": total_valid_rows,
        "windows_with_l2": len(windows),
        "windows_required": MIN_WINDOWS_REQUIRED,
        "l2_window_closes": sorted(windows),
    }


def causal_z_l2_obi(ts, obi):
    """Standardize current imbalance against valid *prior* samples within 150s."""
    good = np.isfinite(obi)
    x = np.where(good, obi, 0.0)
    left = np.searchsorted(ts, ts - OBI_LOOKBACK_S, side="right")

    def prior_sum(a):
        prefix = np.r_[0, np.cumsum(a)]
        return prefix[:-1] - prefix[left]

    n = prior_sum(good)
    mean = prior_sum(x) / np.maximum(n, 1)
    sd = np.sqrt(np.maximum(prior_sum(x * x) / np.maximum(n, 1) - mean * mean, 0.0))
    z = np.full(len(ts), np.nan)
    ok = good & (n >= MIN_OBI_OBS) & (sd > 1e-8)
    z[ok] = (obi[ok] - mean[ok]) / sd[ok]
    return z


def collect_l2_lane_samples(lane, fit_params: dict, valid_closes: set[float], cut: float,
                            split: str, cache: Path) -> dict:
    its, v = KF.clean_index(lane.index, before=cut if split == "is" else None)
    tv, C, N = KF.sigma_prefix(its, v)

    nu = fit_params["nu"]
    s_fat = fit_params["s"]
    table = KF.t_cdf_table(nu)

    with (cache / f"book_{lane.lane}.npz").open("rb") as f, np.load(f, allow_pickle=False) as data:
        bid_sz_all, ask_sz_all, micro_all = (data[k] for k in ("yes_bid_sz", "yes_ask_sz", "microprice"))
    if not len(bid_sz_all) == len(ask_sz_all) == len(micro_all) == len(lane.ts):
        raise ValueError(f"{lane.lane}: depth arrays do not match book rows")

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
        if not len(bk_ts):
            skips["no_l2"] += len(secs_grid)
            continue
        ts_grid = close - secs_grid
        j = np.searchsorted(bk_ts, ts_grid, side="right") - 1
        row = a_bk + np.maximum(j, 0)
        bid_sz, ask_sz, micro = bid_sz_all[row], ask_sz_all[row], micro_all[row]
        good = ((j >= 0) & (ts_grid - bk_ts[np.maximum(j, 0)] <= MAX_L2_AGE_S)
                & valid_size(bid_sz) & valid_size(ask_sz) & valid_microprice(micro))
        obi = np.full(len(secs_grid), np.nan)
        obi[good] = (bid_sz[good] - ask_sz[good]) / (bid_sz[good] + ask_sz[good])
        z_grid = causal_z_l2_obi(ts_grid, obi)

        lo_tr, hi_tr = int(lane.trade_bounds[tid]), int(lane.trade_bounds[tid + 1])
        tts = lane.trade_ts[lo_tr:hi_tr]
        sz = lane.trade_sz[lo_tr:hi_tr]
        signed = np.where(lane.trade_ty[lo_tr:hi_tr] == 1, sz, -sz)
        cs = np.concatenate(([0.0], np.cumsum(signed)))
        ct = np.concatenate(([0.0], np.cumsum(sz)))
        n_trades = len(tts)

        for t, secs, z_l2_obi in zip(ts_grid, secs_grid, z_grid):
            if not math.isfinite(z_l2_obi):
                skips["no_causal_l2_z"] += 1
                continue

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
            tau_eff = max(secs - KF.TAU_OFFSET, 1.0)  # grid stops at 60s: no locked fixings
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
            obi_l.append(z_l2_obi)

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
        "z_l2_obi": np.asarray(obi_l),
        "flow_fast": {fw: np.asarray(f_fast_l[fw]) for fw in FLOW_WINDOWS},
        "flow_slow": {fw: np.asarray(f_slow_l[fw]) for fw in FLOW_WINDOWS},
        "skips": skips,
    }


def find_l2_triggers(pack: dict, hurdle: float, obi_th: float,
                     fw: tuple[float, float], style: str) -> list[tuple[int, str, float, float, float, float]]:
    p_yes = pack["p_yes"]
    p_no = pack["p_no"]
    obi = pack["z_l2_obi"]
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


def run_full_l2_sweep(audit: dict, cache: Path, manifest: dict) -> dict:
    t0 = time.time()
    valid_closes = set(audit["l2_window_closes"])
    sorted_closes = sorted(valid_closes)
    cut_idx = int(len(sorted_closes) * 0.6)
    cut = sorted_closes[cut_idx]
    AD.CACHE = cache
    RC.CACHE = EVOLVE_CACHE  # Ledger provenance; all scoring reads the pinned copy.

    all_combos = list(itertools.product(HURDLES, OBI_THRESHOLDS, FLOW_WINDOWS, STYLES))
    cfg_names = [config_name(h, obi, fw, st) for h, obi, fw, st in all_combos]
    out_dict = {cn: {"rows": [], "skips": Counter()} for cn in cfg_names}
    fit_params, coverage = {}, {}
    for name in RC.LANES:
        lane = AD.Lane(name, max_close=cut)
        its, v = KF.clean_index(lane.index, before=cut - KF.FIT_LEAD)
        start = int(np.searchsorted(its, sorted_closes[0] - KF.FIT_LEAD))
        fit_params[name] = KF.fit_lane(its[start:], v[start:], cut)
        pack = collect_l2_lane_samples(lane, fit_params[name], valid_closes, cut, "is", cache)
        coverage[name] = {"is_windows_with_priced_depth": len(set(pack["close"])), "is_samples": len(pack["t"])}
        for h, obi, fw, st in all_combos:
            cn = config_name(h, obi, fw, st)
            trig = find_l2_triggers(pack, h, obi, fw, st)
            run_l2_triggers(pack, trig, cn, st, h, out_dict)
        del pack, lane

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
    TRADES.write_text("".join(json.dumps(r, default=RC._json_default) + "\n" for r in champ_rows))
    obj = {
        "family": FAMILY,
        "split": "First 60% and last 40% of distinct qualifying BTC closes, shared across lanes",
        "cut": cut,
        "total_l2_windows": len(sorted_closes),
        "is_windows": cut_idx,
        "oos_windows": len(sorted_closes) - cut_idx,
        "windows": sorted_closes,
        "features": FEATURES,
        "coverage": coverage,
        "frozen_input_sha256": manifest,
        "fit_params": fit_params,
        "configs": configs,
        "champion": champion,
        "oos_scored": None,
    }

    if champion["is_eligible"]:
        cfg = parse_config(champion["config"])

        def scorer():
            oos_dict = {champion["config"]: {"rows": [], "skips": Counter()}}
            for name in RC.LANES:
                lane = AD.Lane(name)
                pack = collect_l2_lane_samples(lane, fit_params[name], valid_closes, cut, "oos", cache)
                trig = find_l2_triggers(pack, cfg["hurdle"], cfg["obi_thresh"], cfg["flow_windows"], cfg["style"])
                run_l2_triggers(pack, trig, champion["config"], cfg["style"], cfg["hurdle"], oos_dict)
                del pack, lane
            scored = oos_dict[champion["config"]]
            OOS_TRADES.write_text("".join(json.dumps(r, default=RC._json_default) + "\n" for r in scored["rows"]))
            return scored["rows"], dict(scored["skips"]), RC.fit_payoff_model(champ_rows)

        # The ledger writes "started" before invoking scorer; no score after a crash or rerun.
        obj["oos_scored"] = RC.score_oos_once(
            FAMILY, {"config": champion["config"], **cfg, "exit": "settle", "cut": cut,
                     "input_markets_sha256": manifest["markets.json"]}, scorer)
    obj["runtime_s"] = round(time.time() - t0, 1)
    obj["completed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return obj


def snapshot_cache(destination: Path) -> dict | None:
    names = ["markets.json", *(f"book_{n}.npz" for n in RC.LANES),
             *(f"index_{AD.INDEX_BY_SERIES[n]}.npz" for n in RC.LANES)]
    paths = [EVOLVE_CACHE / name for name in names]

    def signature():
        stats = [p.stat() for p in paths]
        return [(s.st_ino, s.st_size, s.st_mtime_ns) for s in stats]

    before = signature()
    subprocess.run(["cp", "--reflink=auto", "--", *paths, destination], check=True, timeout=120)
    if signature() != before:
        print("Cache files changed while taking the snapshot; no score, retry on the next slot.")
        return None
    digests = {}
    for name in names:
        with (destination / name).open("rb") as f:
            digests[name] = hashlib.file_digest(f, "sha256").hexdigest()
    return digests


def atomic_text(path, text):
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as f:
        f.write(text)
        temp = Path(f.name)
    temp.replace(path)


def update_handoff(obj):
    champion, score = obj["champion"], obj["oos_scored"]
    eligible = champion["is_eligible"]
    if eligible != (score is not None):
        raise ValueError("incomplete Track 2C result")
    best = champion["is"]
    gate = score["gate"] if score else None
    next_data = json.loads(NEXT_JSON.read_text())
    next_data["updated"] = obj["completed_at"]
    state = next_data["rnd_overnight_shift"]["track2c_true_l2_loop"]
    state.update(status="OOS scored" if eligible else "IS fail; OOS unspent",
                 current_windows=obj["total_l2_windows"], champion=champion["config"],
                 is_net=best["net"], is_pf=best["pf"] if math.isfinite(best["pf"]) else "inf",
                 is_trades=best["trades"], oos_spent=eligible,
                 gate_pass=gate["pass"] if gate else False, gate=gate,
                 result_artifact=str(OUT.relative_to(ROOT)),
                 note="Research result only; no live promotion or trading.")
    atomic_text(NEXT_JSON, json.dumps(next_data, indent=1, allow_nan=False) + "\n")

    report = RND_REPORT.read_text()
    header = "### Track 2C — overnight true-L2 result"
    if header not in report:
        lines = [
            header,
            f"- {obj['total_l2_windows']} qualifying BTC windows; chronological split "
            f"{obj['is_windows']} IS / {obj['oos_windows']} OOS, cut `{obj['cut']}`.",
            f"- IS champion `{champion['config']}`: {best['trades']} trades, net ${best['net']:+.2f}, "
            f"PF {best['pf']:.2f}, {best['profitable_lanes']} profitable lanes.",
        ]
        if score:
            m = score["oos"]
            lines.append(f"- One OOS look: {m['trades']} trades, net ${m['net']:+.2f}, PF {m['pf']:.2f}, "
                         f"payoff-ECE {score['payoff_ece']}, ex-XRP ${m['net_ex_xrp']:+.2f}, "
                         f"{m['profitable_lanes']} profitable lanes, {m['crossed_fills']} crossed fills, "
                         f"max entry ${m['max_entry_cost']:.2f}. Gate pass: {gate['pass']}.")
        else:
            lines.append("- IS eligibility missed; OOS left sealed and no OOS entry appended.")
        lines.append(f"- Receipt: `{OUT.relative_to(ROOT)}`; no live promotion.")
        atomic_text(RND_REPORT, report.rstrip() + "\n\n" + "\n".join(lines) + "\n")


def publish_results(obj):
    if not PRIVATE_GIT.is_dir() or not PRIVATE_GIT.resolve().is_relative_to(ROOT):
        raise RuntimeError("cannot publish without isolated R&D Git metadata")
    env = {k: v for k, v in os.environ.items() if k not in
           ("GIT_COMMON_DIR", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES")}
    env["GIT_TERMINAL_PROMPT"] = "0"

    def git(*args):
        return subprocess.run(["git", f"--git-dir={PRIVATE_GIT}", f"--work-tree={ROOT}", *args],
                              cwd=ROOT, env=env, check=True, text=True, stdout=subprocess.PIPE,
                              timeout=90).stdout.strip()

    if git("branch", "--show-current") != BRANCH:
        raise RuntimeError("refusing to publish from a different branch")
    paths = [str(p.relative_to(ROOT)) for p in (RND_REPORT, NEXT_JSON, OUT, TRADES, LEDGER, OOS_TRADES) if p.exists()]
    if set(git("diff", "--cached", "--name-only").splitlines()).difference(paths):
        raise RuntimeError("refusing to commit unrelated staged files")
    git("add", "-f", "--", *paths)
    if git("diff", "--cached", "--name-only"):
        git("diff", "--cached", "--check")
        git("commit", "-m", f"Track 2C true-L2 sweep: {obj['champion']['config']}")
    ref = f"refs/heads/{BRANCH}"
    git("push", "origin", f"{ref}:{ref}")
    if git("ls-remote", "origin", ref).split()[0] != git("rev-parse", "HEAD"):
        raise RuntimeError("remote R&D ref did not verify")
    print(f"Published Track 2C on {BRANCH}: {git('rev-parse', '--short', 'HEAD')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--publish", action="store_true", help="Commit and push the completed research receipt")
    args = ap.parse_args()

    if args.check_only:
        print(json.dumps(audit_l2_depth(), indent=2))
        return 0
    with (ROOT / "data/overnight-track2c.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if OUT.exists():
            obj = json.loads(OUT.read_text())
            if "frozen_input_sha256" not in obj or "completed_at" not in obj:
                raise RuntimeError("ambiguous prior Track 2C output; refusing to rescore")
        else:
            if any(json.loads(line)["family"] == FAMILY for line in LEDGER.read_text().splitlines() if line.strip()):
                raise RuntimeError("Track 2C OOS touched without a complete receipt; refusing to rescore")
            if os.environ.get("RND_OOS_UNSEAL") != "lead" or os.environ.get("RND_OOS_RESCORE"):
                raise RuntimeError("Track 2C needs the one-shot lead seal; rescores are forbidden")
            with tempfile.TemporaryDirectory(prefix="track2c-tape-", dir=ROOT / "data") as temp:
                cache = Path(temp)
                try:
                    manifest = snapshot_cache(cache)
                    if manifest is not None:
                        audit = audit_l2_depth(cache)
                except (FileNotFoundError, BadZipFile) as e:
                    print(f"Cache publication incomplete; no score: {e}", file=sys.stderr)
                    return 75
                if manifest is None:
                    return 75
                if audit["windows_with_l2"] < MIN_WINDOWS_REQUIRED:
                    print(f"Frozen tape has {audit['windows_with_l2']}/{MIN_WINDOWS_REQUIRED} eligible windows; no score.")
                    return 75
                print(f"Pinned {audit['windows_with_l2']} qualifying BTC closes. Running Track 2C.", flush=True)
                obj = run_full_l2_sweep(audit, cache, manifest)
                atomic_text(OUT, json.dumps(obj, indent=2, default=RC._json_default) + "\n")
        update_handoff(obj)
        if args.publish:
            try:
                publish_results(obj)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                print(f"Publication failed; retain result and retry only publishing: {e}", file=sys.stderr)
                return 75
        print(f"Track 2C: {obj['champion']['config']}; OOS {'scored' if obj['oos_scored'] else 'unspent'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
