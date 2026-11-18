#!/usr/bin/env python3
"""Track 9 premise test `coinbase_lead_lag_premise`: does the RTI lag pure Coinbase prints, and is there anything to trade?

Everything here is fixed by `scripts/track9_prereg.json` (registered, then amended once to v2 before any code or data;
the file is read at run time and its sha256 goes into the receipt). Three criteria, all in sample only:
  lead_lag  exact-lag correlations of 1-second returns, A_k = corr(r_cb(t), r_rti(t+k)) - corr(r_rti(t), r_cb(t+k)),
            primary S = A_1 + A_2 with a 15-minute block bootstrap; 3 of 4 lanes must show S >= 0.10 with a lower bound above 0;
  funnel    at least 100 (lane, window) events with |Coinbase / RTI - 1| >= 8 bps in the last 90 s, in >= 40 distinct windows;
  brier     in the last 60 s the Coinbase settlement probability must beat the market mid, bootstrap upper bound below 0.
The premise passes only if all three do; only then may a sweep be designed, on IS, with OOS still sealed.

Inputs are a pinned tape directory (pin.json digests; build one with `pin_new_tape`) holding the evolve-cache files the other tracks use
plus `spot_ticks.npz` from `scripts/export_spot_ticks.py`. The run gate is >= 288 eligible windows (about three days): below it the script
writes `insufficient_data` and computes no statistic. A decided receipt is never overwritten: the test is run once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import alpha_discover as AD  # noqa: E402
from scripts import kurtosis_fv_snipe as KF  # noqa: E402
from scripts import rnd_common as RC  # noqa: E402
from scripts import track2c_true_l2_settle as C  # noqa: E402
from scripts import track4_common as T  # noqa: E402
from scripts import track8_coinbase_rti_settle as M8  # noqa: E402
from scripts.asian_rti import AsianRTI  # noqa: E402
from scripts.track6_gbdt_l2_meta import last_vals  # noqa: E402

SPEC_PATH = Path(__file__).with_name("track9_prereg.json")
SPEC_BYTES = SPEC_PATH.read_bytes()
SPEC = json.loads(SPEC_BYTES)
SPEC_SHA256 = hashlib.sha256(SPEC_BYTES).hexdigest()
LL, FUN, BR = SPEC["lead_lag"], SPEC["funnel"], SPEC["brier"]
LANE_SYMBOL = SPEC["data"]["lane_symbols"]
RECEIPT = ROOT / "data" / "track9-premise.json"
TAPE = ROOT / "data" / "track9-tape"
SPOT_FILE = "spot_ticks.npz"
LAGS = tuple(LL["lags_s"])
PRIMARY = (1, 2)
ELIG_WINDOW_S, ELIG_MIN_PRINTS, ELIG_MAX_GAP_S = 150.0, 30, 3.0       # the registered eligibility rule


def check_constants() -> None:
    """The funnel and Brier reuse Track 8's sampler; refuse to run if its constants no longer match the registration."""
    assert M8.DIVERGENCE_BPS == FUN["min_abs_divergence_bps"], "divergence threshold differs from the registration"
    assert (float(M8.SECS[0]), float(M8.SECS[-1])) == (float(FUN["secs_left"][1]), float(FUN["secs_left"][0])), "decision grid differs"
    assert M8.QUOTE_MAX_AGE_S == FUN["quote_fresh_s"], "quote age differs"
    assert KF.INDEX_TOL == FUN["cb_fresh_s"] == LL["fresh_s"], "freshness differs"


# ------------------------------------------------------------------ inputs
def coinbase_series(spot: dict, symbol: str) -> tuple[np.ndarray, np.ndarray]:
    """(ts, price) of the venue-labelled Coinbase prints for one symbol, sorted; Kraken and REST prints are excluded."""
    m = (spot["source"] == "coinbase") & (spot["symbol"] == symbol) & np.isfinite(spot["price"]) & (spot["price"] > 0)
    order = np.argsort(spot["ts"][m], kind="stable")
    return spot["ts"][m][order], spot["price"][m][order]


def eligible_closes(closes: list[float], series: dict) -> list[float]:
    """Closes where every lane has >= 30 Coinbase prints in the last 150 s with no gap over 3 s (edges count as gaps)."""
    out = []
    for close in closes:
        ok = True
        for ts, _ in series.values():
            seg = ts[np.searchsorted(ts, close - ELIG_WINDOW_S, "left"):np.searchsorted(ts, close, "right")]
            if len(seg) < ELIG_MIN_PRINTS or np.diff(np.r_[close - ELIG_WINDOW_S, seg, close]).max() > ELIG_MAX_GAP_S:
                ok = False
                break
        if ok:
            out.append(close)
    return out


def split_cut(eligible: list[float]) -> float:
    return eligible[int(len(eligible) * 0.6)]


# ----------------------------------------------------------------- bootstrap
def bootstrap(stats: np.ndarray, fn, draws: int, seed: int, ci: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Point estimate and percentile interval of `fn(column totals)` over blocks resampled with replacement."""
    rng = np.random.default_rng(seed)
    blocks = len(stats)
    point = np.atleast_1d(fn(stats.sum(axis=0)))
    vals = np.empty((draws, len(point)))
    for i in range(draws):
        vals[i] = fn(np.bincount(rng.integers(0, blocks, blocks), minlength=blocks) @ stats)
    a = (1.0 - ci) / 2.0 * 100.0
    return point, np.nanpercentile(vals, a, axis=0), np.nanpercentile(vals, 100.0 - a, axis=0)


def corr_from(n, sa, sb, sab, saa, sbb) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        return float((n * sab - sa * sb) / np.sqrt((n * saa - sa * sa) * (n * sbb - sb * sb)))


def pair_sums(a: np.ndarray, b: np.ndarray, block: np.ndarray, blocks: int) -> list[np.ndarray]:
    ok = np.isfinite(a) & np.isfinite(b)
    a, b, block = a[ok], b[ok], block[ok]
    cols = (np.ones(len(a)), a, b, a * b, a * a, b * b)
    return [np.bincount(block, weights=c, minlength=blocks) for c in cols]


# ----------------------------------------------------------------- lead-lag
def second_returns(grid: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(grid[1:] / grid[:-1])
    return np.where(np.isfinite(r), r, np.nan)


def lane_passes(s: dict) -> bool:
    """Registered lane rule: the primary statistic is at least 0.10 and its bootstrap lower bound is above zero."""
    return bool(s["point"] >= 0.10 and s["lo"] > 0.0)


def lead_lag(cb: tuple, rti: tuple, t_start: float, t_end: float) -> dict:
    """A_k for each registered lag and the primary S, with block-bootstrap intervals, from data strictly before `t_end`."""
    grid_t = np.arange(np.ceil(t_start), np.floor(t_end), 1.0)
    r_cb, r_rti = second_returns(last_vals(*cb, grid_t)), second_returns(last_vals(*rti, grid_t))
    block = ((grid_t[1:] // LL["bootstrap"]["block_s"]) - (grid_t[1] // LL["bootstrap"]["block_s"])).astype(np.int64)
    blocks = int(block.max()) + 1
    cols = []
    for k in LAGS:
        n = len(r_cb) - k
        cols += pair_sums(r_cb[:n], r_rti[k:], block[:n], blocks)            # Coinbase leads: r_cb(t) with r_rti(t+k)
        cols += pair_sums(r_rti[:n], r_cb[k:], block[:n], blocks)            # RTI leads:      r_rti(t) with r_cb(t+k)
    stats = np.column_stack(cols)
    stats = stats[stats[:, 0::6].sum(axis=1) > 0]

    def fn(t):
        a = [corr_from(*t[12 * i:12 * i + 6]) - corr_from(*t[12 * i + 6:12 * i + 12]) for i in range(len(LAGS))]
        return np.array(a + [sum(a[LAGS.index(k)] for k in PRIMARY)])

    b = LL["bootstrap"]
    point, lo, hi = bootstrap(stats, fn, b["draws"], b["seed"], b["ci"])
    out = {f"A_{k}": {"point": float(point[i]), "lo": float(lo[i]), "hi": float(hi[i])} for i, k in enumerate(LAGS)}
    out["S"] = {"point": float(point[-1]), "lo": float(lo[-1]), "hi": float(hi[-1])}
    out.update(rows=int(stats[:, 0].sum()), blocks=int(len(stats)))
    out["passes"] = lane_passes(out["S"])
    return out


def lead_lag_criterion(lanes: dict) -> dict:
    passed = [n for n, v in lanes.items() if v["passes"]]
    return {"lanes_passing": passed, "required": 3, "passes": len(passed) >= 3}


# ------------------------------------------------------------ funnel, Brier
def funnel_criterion(packs: dict) -> dict:
    events = {(n, c) for n, p in packs.items() for c in p["close"][M8.diverged(p)]}
    closes = {c for _, c in events}
    return {"events": len(events), "distinct_closes": len(closes), "needs": {"events": 100, "distinct_closes": 40},
            "passes": len(events) >= 100 and len(closes) >= 40}


def brier_criterion(packs: dict) -> dict:
    """Mean of (p_cb - y)^2 - (mid - y)^2 over the last-60s rows, bootstrapped by window close."""
    parts = []
    for p in packs.values():
        s = p["secs"] <= BR["secs_left"][1] + 1e-9
        mid = (p["bid_m"][s] + p["ask_m"][s]) / 2000.0
        y = p["y"][s]
        parts.append(np.column_stack([p["close"][s], (p["p_cb"][s] - y) ** 2, (p["p_rti"][s] - y) ** 2, (mid - y) ** 2]))
    rows = np.concatenate(parts) if parts else np.zeros((0, 4))
    if not len(rows):
        return {"rows": 0, "passes": False}
    closes, inv = np.unique(rows[:, 0], return_inverse=True)
    stats = np.column_stack([np.bincount(inv, minlength=len(closes))] + [np.bincount(inv, weights=rows[:, j], minlength=len(closes)) for j in (1, 2, 3)])
    fn = lambda t: np.array([(t[1] - t[3]) / t[0], t[1] / t[0], t[2] / t[0], t[3] / t[0]])
    b = BR["bootstrap"]
    point, lo, hi = bootstrap(stats, fn, b["draws"], b["seed"], b["ci"])
    return {"rows": int(len(rows)), "windows": int(len(closes)), "mean_d": float(point[0]), "d_lo": float(lo[0]), "d_hi": float(hi[0]),
            "brier_cb": float(point[1]), "brier_rti_spot": float(point[2]), "brier_mid": float(point[3]), "passes": bool(point[0] < 0 and hi[0] < 0)}


def decide(lead: dict, funnel: dict, brier: dict) -> dict:
    ok = bool(lead["passes"] and funnel["passes"] and brier["passes"])
    return {"premise_passes": ok, "criteria": {"lead_lag": lead["passes"], "funnel": funnel["passes"], "brier": brier["passes"]},
            "next": SPEC["decision"]["on_pass" if ok else "on_fail"]}


# --------------------------------------------------------------------- run
def load_inputs(tape: Path) -> dict:
    """Pin-checked tape contents: window closes, Coinbase series per lane, RTI series per lane."""
    pin = T.pin_tape(tape)
    spot = dict(np.load(tape / SPOT_FILE, allow_pickle=False))
    cb = {lane: coinbase_series(spot, sym) for lane, sym in LANE_SYMBOL.items()}
    rti = {lane: KF.clean_index(dict(np.load(tape / f"index_{AD.INDEX_BY_SERIES[lane]}.npz"))) for lane in RC.LANES}
    return {"pin": pin, "closes": T.window_closes(tape), "cb": cb, "rti": rti}


def gather_packs(inp: dict, cut: float) -> dict:
    packs = {}
    for lane in RC.LANES:
        lane_obj = AD.Lane(lane)
        model = AsianRTI(lane_obj.index["ts"], lane_obj.index["value"], lane_obj.index["accum"], lane_obj.index["k"])
        packs[lane] = M8.collect(lane_obj, model, inp["cb"][lane], cut, "is")
        del lane_obj, model
    return packs


def run(tape: Path = TAPE, receipt: Path = RECEIPT, *, check_only: bool = False) -> dict:
    check_constants()
    if receipt.exists() and "decision" in json.loads(receipt.read_text()):
        raise RuntimeError("the premise test has already been decided; it is run once")
    inp = load_inputs(tape)
    eligible = eligible_closes(inp["closes"], inp["cb"])
    gate = SPEC["run_gate"]["min_eligible_windows"]
    head = {"spec_sha256": SPEC_SHA256, "spec_version": SPEC["version"], "frozen_input_sha256": inp["pin"]["sha256"],
            "eligible_windows": len(eligible), "run_gate": gate}
    if check_only:
        return {**head, "ready": len(eligible) >= gate}
    if len(eligible) < gate:
        obj = {**head, "status": "insufficient_data"}
        C.save_receipt(receipt, T.clean(obj), strict=True)
        return obj
    cut = split_cut(eligible)
    t_start = max(max(v[0][0] for v in inp["cb"].values()), max(v[0][0] for v in inp["rti"].values()))
    lanes = {lane: lead_lag(inp["cb"][lane], inp["rti"][lane], t_start, cut) for lane in RC.LANES}
    packs = gather_packs(inp, cut)
    lead, funnel, brier = lead_lag_criterion(lanes), funnel_criterion(packs), brier_criterion(packs)
    obj = {**head, "status": "decided", "cut": cut, "is_windows": sum(c < cut for c in eligible), "oos_windows_unread": sum(c >= cut for c in eligible),
           "lead_lag": {"lanes": lanes, **lead}, "funnel": funnel, "brier": brier, "decision": decide(lead, funnel, brier)}
    C.save_receipt(receipt, T.clean(obj), strict=True)
    return obj


def pin_new_tape(dest: Path, files: list[Path]) -> dict:
    """Copy `files` into a new directory and write pin.json with their sha256 digests (refuses to touch an existing pin)."""
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "pin.json").exists():
        raise RuntimeError(f"{dest} is already pinned")
    for f in files:
        shutil.copyfile(f, dest / Path(f).name)
    digests = C.snapshot_digests(dest, sorted(p.name for p in dest.iterdir()))
    (dest / "pin.json").write_text(json.dumps({"sha256": digests, "source": [str(f) for f in files]}) + "\n")
    return digests


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tape", type=Path, default=TAPE)
    ap.add_argument("--check-only", action="store_true", help="count eligible windows against the run gate; compute nothing else")
    args = ap.parse_args()
    out = run(args.tape, check_only=args.check_only)
    print(json.dumps(T.clean(out), indent=1, default=RC._json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
