#!/usr/bin/env python3
"""Track 2: synthetic intra-window pair-locking on the frozen F2 triggers.

Every entry goes through `sim_core.run_trade`. A $0-fee post-only lock on the opposite
contract (lock_c) and/or a $0-fee take-profit (tp_c) race on the same tape; no stop.

Default mode is IS-only: triggers, fills and metrics exist only for windows whose close
is before the global cut. `--score-oos` scores the IS champion once on OOS (lead only).

    flock data/.heavy.lock timeout 1800 nice -n 10 .venv/bin/python scripts/synthetic_lock.py
    .venv/bin/python scripts/synthetic_lock.py --lanes KXSOL15M --out <scratch>/dev.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openkal import sim_core as S  # noqa: E402
from scripts import f2_structural as F  # noqa: E402
from scripts import rnd_common as R  # noqa: E402
from scripts.bounded_sweep import pick  # noqa: E402

FAMILY = "synthetic_lock"
# Added by the lead after the first IS run and before any OOS look: the overall
# champion was the no-lock settle control, so the lock thesis gets its own family,
# chosen on IS among the configs that rest a lock leg, and its own single OOS look.
FAMILY_LOCKED = "synthetic_lock_locked_only"
COST_HI = Decimal("0.58")
COOLDOWN, PER_WINDOW = 90.0, 3
PATIENCE_S = 5.0
POOL_DIR = ROOT / "data" / "rnd-pool"
POOL_KEYS = ("tid", "ts", "close", "bid", "ask", "spot", "delta", "z", "flow", "rev")
FEAT_NAMES = ["z", "flow_signed", "rev_signed", "delta_bps"]
STYLES = ("taker", "maker")
LOCKS = (None, Decimal("0.04"), Decimal("0.06"), Decimal("0.08"))
TPS = (None, Decimal("0.06"))
EQUIV_LOCKS = (Decimal("0.04"), Decimal("0.06"), Decimal("0.08"))
DEFAULT_OUT = ROOT / "data" / "synthetic-lock-is.json"
OOS_OUT = ROOT / "data" / "synthetic-lock-oos.json"


def name_of(style: str, lock: Decimal | None, tp: Decimal | None) -> str:
    return f"{style}_lock{lock or 'none'}_tp{tp or 'none'}"


def cut_all_lanes() -> float:
    """The global cut over all four lanes, from the cache metadata (no book arrays).

    Mirrors `rnd_common.global_cut` exactly: a ticker counts when markets.json has a
    strike for it. Lets a single-lane debug run use the real cut without loading books.
    """
    meta = json.loads((R.CACHE / "markets.json").read_text())
    closes: set[float] = set()
    for lane in R.LANES:
        tickers = np.load(R.CACHE / f"book_{lane}.npz", allow_pickle=False)["tickers"]
        for t in tickers:
            m = meta.get(str(t))
            if m is not None and m.get("strike"):
                closes.add(float(m["close"]))
    ordered = sorted(closes)
    return ordered[int(len(ordered) * 0.6)]


def size(tape: S.Tape, side: str, ts: float, style: str) -> int:
    """Contracts a stake buys at the cost the order would meet. Crossed or empty books
    leave qty at MAX_QTY so that run_trade reports the skip itself."""
    try:
        b, a = tape.quote(ts + S.LAG_S)
    except (S.CrossedBook, S.NoQuote):
        return S.MAX_QTY
    return S.stake_qty(S.entry_price(side, style, b, a), R.STAKE)


def build_triggers(lanes: dict, cut: float, slice_name: str) -> list[dict]:
    """Frozen-F2 triggers in one slice: IS is close < cut, OOS is close >= cut."""
    trigs: list[dict] = []
    for name, lane in lanes.items():
        with np.load(POOL_DIR / f"slim_{name}.npz", allow_pickle=False) as f:
            pool = {k: f[k] for k in POOL_KEYS}
        mask, long_yes = F.frozen_mask(pool, "taker")
        mask &= (pool["close"] < cut) if slice_name == "is" else (pool["close"] >= cut)
        for i in pick(pool, mask, cooldown=COOLDOWN, per_window=PER_WINDOW):
            tid = int(pool["tid"][i])
            tape = R.tape(lane, tid)
            ts = float(pool["ts"][i])
            side = "YES" if long_yes[i] else "NO"
            trigs.append({
                "lane": name, "tid": tid, "ticker": lane.tickers[tid],
                "close": float(pool["close"][i]), "ts": ts, "side": side, "tape": tape,
                "qty": {st: size(tape, side, ts, st) for st in STYLES},
                "feat": F.feature_row(pool, i),
            })
        del pool
    return trigs


def run_one(t: dict, style: str, lock: Decimal | None, tp: Decimal | None):
    return S.run_trade(t["tape"], side=t["side"], trigger_ts=t["ts"], entry_style=style,
                       qty=t["qty"][style], patience_s=PATIENCE_S, cost_hi=COST_HI,
                       lock_c=lock, tp_c=tp)


def run_config(trigs: list[dict], style: str, lock, tp, slice_name: str = "is"):
    """(Trade, row) for every fill, and skip counts by reason for the rest."""
    fills: list[tuple[S.Trade, dict]] = []
    skips: dict[str, int] = {}
    for t in trigs:
        res = run_one(t, style, lock, tp)
        if isinstance(res, S.Skip):
            skips[res.reason] = skips.get(res.reason, 0) + 1
            continue
        row = R.trade_row(res, lane=t["lane"], ticker=t["ticker"], close=t["close"],
                          slice_name=slice_name, feat=t["feat"])
        fills.append((res, row))
    return fills, skips


def mix_of(fills: list[tuple[S.Trade, dict]], lock: Decimal | None) -> dict:
    trades = [tr for tr, _ in fills]

    def bucket(sel: list[S.Trade]) -> dict:
        return {"count": len(sel), "net": float(sum((t.pnl for t in sel), Decimal(0)))}

    settled = [t for t in trades if t.reason == "settle"]
    mix = {
        "lock": bucket([t for t in trades if t.reason == "lock"]),
        "tp": bucket([t for t in trades if t.reason == "tp"]),
        "settle": {**bucket(settled),
                   "won": bucket([t for t in settled if t.pnl > 0]),
                   "lost": bucket([t for t in settled if t.pnl <= 0])},
    }
    if lock is not None:
        px = [S.ONE - t.entry - lock for t in trades]
        pxf = [float(p) for p in px]
        mix["lock_leg_price"] = {
            "over": "every filled entry (the lock rests at fill)",
            "n": len(px),
            "min": min(pxf) if px else None,
            "median": statistics.median(pxf) if px else None,
            "max": max(pxf) if px else None,
            "above_0_62": sum(1 for p in px if p > S.CAP),
        }
    return mix


def equivalence(trigs: list[dict]) -> dict:
    """Lock at L versus take-profit at L: same fills, same pnl, same exit time."""
    def key(x):
        return (x.pnl, x.exit_ts) if isinstance(x, S.Trade) else ("skip", x.reason)

    def describe(x):
        if isinstance(x, S.Trade):
            return {"reason": x.reason, "pnl": str(x.pnl), "exit_ts": x.exit_ts}
        return {"skip": x.reason}

    report: dict = {}
    for L in EQUIV_LOCKS:
        compared = filled = 0
        examples: list[dict] = []
        for t in trigs:
            for st in STYLES:
                a = run_one(t, st, L, None)     # lock only
                b = run_one(t, st, None, L)     # take-profit only
                compared += 1
                filled += isinstance(a, S.Trade) and isinstance(b, S.Trade)
                if key(a) != key(b):
                    examples.append({"entry_style": st, "ticker": t["ticker"],
                                     "trigger_ts": t["ts"], "lock": describe(a),
                                     "tp": describe(b)})
        report[str(L)] = {"compared": compared, "filled_pairs": filled,
                          "differ": len(examples), "examples": examples}
    return report


def choose(configs: dict, locked_only: bool = False) -> tuple[str, bool]:
    def eligible(c: dict) -> bool:
        i = c["is"]
        return i["trades"] >= 30 and i["pf"] >= 1.20 and i["crossed_fills"] == 0

    names = [k for k, c in configs.items() if not locked_only or c["lock_c"] is not None]
    ok = [k for k in names if eligible(configs[k])]
    best = max(ok or names, key=lambda k: configs[k]["is"]["net"])
    return best, best in ok


def run_is(lanes: dict, cut: float, out_path: Path) -> None:
    trigs = build_triggers(lanes, cut, "is")
    print(f"IS triggers: {len(trigs)}  cut: {cut}")
    eq = equivalence(trigs)
    for L, v in eq.items():
        print(f"equiv lock={L}: compared {v['compared']} differ {v['differ']}")

    configs: dict = {}
    fills_of: dict = {}
    for style in STYLES:
        for lock in LOCKS:
            for tp in TPS:
                name = name_of(style, lock, tp)
                fills, skips = run_config(trigs, style, lock, tp)
                fills_of[name] = fills
                configs[name] = {
                    "entry_style": style,
                    "lock_c": str(lock) if lock is not None else None,
                    "tp_c": str(tp) if tp is not None else None,
                    "is": R.summarize([r for _, r in fills], skips),
                    "mix": mix_of(fills, lock),
                }
                s = configs[name]["is"]
                print(f"{name:32s} trades {s['trades']:5d} win {s['win_rate']:.3f} "
                      f"net {s['net']:+10.2f} PF {s['pf']:6.3f} skips {s['skips']}")

    champ, eligible = choose(configs)
    champ_l, eligible_l = choose(configs, locked_only=True)
    trades_path = out_path.with_name(out_path.stem + "-trades.jsonl")
    with trades_path.open("w") as fh:
        for fam, name in ((FAMILY, champ), (FAMILY_LOCKED, champ_l)):
            for _, row in fills_of[name]:
                fh.write(json.dumps(dict(row, family=fam, config=name),
                                    default=R._json_default) + "\n")
    report = {
        "family": FAMILY,
        "cut": cut,
        "lanes": list(lanes),
        "cost_hi": str(COST_HI),
        "feat": FEAT_NAMES,
        "triggers_is": len(trigs),
        "equivalence_check": eq,
        "configs": configs,
        "n_configs": len(configs),
        "champion": {"family": FAMILY, "config": champ, "is_eligible": eligible,
                     "is": configs[champ]["is"]},
        "champion_locked_only": {"family": FAMILY_LOCKED, "config": champ_l,
                                 "is_eligible": eligible_l, "is": configs[champ_l]["is"]},
        "champion_trades": str(trades_path),
    }
    R.dump(out_path, report)
    print(f"champion {champ} is_eligible={eligible}")
    print(f"locked-only champion {champ_l} is_eligible={eligible_l}")
    print(f"wrote {out_path}\nwrote {trades_path}")


def score_oos(lanes: dict, cut: float, out_path: Path) -> None:
    """Lead only. Scores each family's IS champion once. score_oos_once refuses without
    RND_OOS_UNSEAL=lead, and it runs the scorer only after that check passes."""
    rep = json.loads(out_path.read_text())
    all_is = [json.loads(x) for x in Path(rep["champion_trades"]).read_text().splitlines()
              if x.strip()]
    trigs = None
    entries = {}
    for key in ("champion", "champion_locked_only"):
        fam, name = rep[key]["family"], rep[key]["config"]
        cfg = rep["configs"][name]
        style = cfg["entry_style"]
        lock = Decimal(cfg["lock_c"]) if cfg["lock_c"] is not None else None
        tp = Decimal(cfg["tp_c"]) if cfg["tp_c"] is not None else None
        champ_is = [r for r in all_is if r.get("family") == fam]

        def scorer(style=style, lock=lock, tp=tp, champ_is=champ_is):
            nonlocal trigs
            if trigs is None:
                trigs = build_triggers(lanes, cut, "oos")
            fills, skips = run_config(trigs, style, lock, tp, slice_name="oos")
            return [r for _, r in fills], skips, R.fit_payoff_model(champ_is)

        config = {"name": name, "entry_style": style, "lock_c": cfg["lock_c"],
                  "tp_c": cfg["tp_c"]}
        entries[fam] = R.score_oos_once(fam, config, scorer)
        print(f"[{fam}] scored {name} once on OOS; gate pass={entries[fam]['gate']['pass']}")
    R.dump(OOS_OUT, entries)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Track 2 synthetic pair-locking, IS-only by default.")
    ap.add_argument("--lanes", default=",".join(R.LANES),
                    help="comma-separated lanes; the cut is always global over all four")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--score-oos", action="store_true",
                    help="lead only: score the IS champion once (needs RND_OOS_UNSEAL=lead)")
    args = ap.parse_args(argv)
    names = tuple(x.strip() for x in args.lanes.split(",") if x.strip())
    unknown = set(names) - set(R.LANES)
    if unknown:
        ap.error(f"unknown lanes {sorted(unknown)}")

    cut = cut_all_lanes()
    lanes = R.load_lanes(names)
    if set(names) == set(R.LANES) and cut != R.global_cut(lanes):
        raise RuntimeError("cut mismatch with rnd_common.global_cut")

    if args.score_oos:
        score_oos(lanes, cut, args.out)
    else:
        run_is(lanes, cut, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
