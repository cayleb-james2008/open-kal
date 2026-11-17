#!/usr/bin/env python3
"""Read-only Monte Carlo: 2 vs 4 crypto slots against the 20% trailing floor.

The OOS evidence is 36 pre-05:00 trades at 94.4% wins and 20 forward trades at
100%, with the frozen F2 payoff of +$3.45 on a take-profit and -$6.50 on a
confirmed stop. This study asks a single risk question: does opening four
crypto slots instead of two change the probability of touching the 20%
trailing drawdown floor?

Two arrival models are reported because they bracket the truth:

  * ``observed`` - the tape's own concurrency. Peak concurrent positions were
    2 in-sample and 1 out-of-sample, so a fourth slot adds no filled trades.
    Both caps therefore see exactly the same trades.
  * ``independent`` - each of four lanes signals independently at the observed
    trade rate. This is deliberately unkind to the two-slot cap: it assumes the
    cap would throw away half the trades, which the capacity audit shows it
    does not.
  * ``simultaneous`` - an adversarial window where all four lanes fire at
    once. It is the only model in which the fourth slot can lose money in the
    same settlement window as the other three.

The edge-erosion sweep repeats the simultaneous model at 90%, 80% and 70%
win rates. The OOS 94.4% is a point estimate, not a guarantee; the sweep shows
where a fourth slot starts to matter to the floor.

No engine state is read or written. Output: data/mc-drawdown.json.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "mc-drawdown.json"

START_EQUITY = 164.00
FLOOR_BASE = 131.20          # 20% below the 15:01 peak; re-derived from peak below
FLOOR_MIN = 105.00
TRAIL = 0.20
WIN = 3.45
LOSS = -6.50
P_WIN_OOS = 34 / 36           # 94.4% pre-05:00
P_WIN_FORWARD = 1.0           # 20/20 forward
Q_LANE_WINDOW = 0.14375       # 2.3 trades/h / 4 lanes / 4 windows per hour
OBSERVED_WINDOW_COUNTS = [3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]


def simulate(
    slots: int,
    p_win: float,
    windows: int,
    paths: int,
    seed: int,
    arrival: str,
) -> dict:
    rng = random.Random(seed)
    breaches = 0
    finals: list[float] = []
    drawdowns: list[float] = []
    trade_counts: list[int] = []

    for _ in range(paths):
        equity = START_EQUITY
        peak = equity
        max_dd = 0.0
        trades = 0
        breached = False
        for w in range(windows):
            if arrival == "simultaneous":
                n = 4
            elif arrival == "independent":
                n = sum(1 for _ in range(slots) if rng.random() < Q_LANE_WINDOW)
            else:
                # The observed groups are sequential entries inside one close
                # window, not simultaneous positions; peak concurrency was 2
                # in-sample and 1 out-of-sample. A cap never truncated them.
                n = OBSERVED_WINDOW_COUNTS[w % len(OBSERVED_WINDOW_COUNTS)]
            if arrival != "observed":
                n = min(n, slots)
            pnl = 0.0
            for _ in range(n):
                pnl += WIN if rng.random() < p_win else LOSS
            equity += pnl
            trades += n
            peak = max(peak, equity)
            max_dd = max(max_dd, peak - equity)
            floor = max(FLOOR_MIN, (1.0 - TRAIL) * peak)
            if equity < floor:
                breached = True
                break
        breaches += int(breached)
        finals.append(equity)
        drawdowns.append(max_dd)
        trade_counts.append(trades)

    finals.sort()
    drawdowns.sort()

    def q(xs: list[float], p: float) -> float:
        return xs[min(len(xs) - 1, int(p * len(xs)))]

    return {
        "slots": slots,
        "arrival": arrival,
        "p_win": p_win,
        "paths": paths,
        "windows": windows,
        "breach_prob": breaches / paths,
        "median_final": round(q(finals, 0.50), 2),
        "p05_final": round(q(finals, 0.05), 2),
        "p95_final": round(q(finals, 0.95), 2),
        "median_max_dd": round(q(drawdowns, 0.50), 2),
        "p95_max_dd": round(q(drawdowns, 0.95), 2),
        "median_trades": sorted(trade_counts)[len(trade_counts) // 2],
        "floor_start": FLOOR_BASE,
        "start_equity": START_EQUITY,
        "win": WIN,
        "loss": LOSS,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=20_000)
    ap.add_argument("--bootstrap-paths", type=int, default=100_000)
    ap.add_argument("--windows", type=int, default=250)
    ap.add_argument("--seed", type=int, default=20261117)
    args = ap.parse_args()

    results = []
    for slots in (2, 4):
        results.append(simulate(slots, P_WIN_OOS, args.windows, args.paths,
                                args.seed + slots, "observed"))
    for slots in (2, 4):
        results.append(simulate(slots, P_WIN_OOS, args.windows, args.paths,
                                args.seed + 10 + slots, "independent"))
    for slots in (2, 4):
        results.append(simulate(slots, P_WIN_OOS, 500, args.paths,
                                args.seed + 20 + slots, "simultaneous"))

    sensitivity = []
    for p in (0.90, 0.80, 0.70):
        for slots in (2, 4):
            sensitivity.append(simulate(
                slots, p, args.windows, max(5_000, args.paths // 4),
                args.seed + int(p * 1000) + slots, "simultaneous"))

    # Bootstrap the OOS payoff directly: 94.4% of 36 trades win; a finite
    # sample can draw more losses than the point estimate. The floor is applied
    # to the same starting equity over a 56-trade epoch.
    rng = random.Random(args.seed + 99)
    boot_breach = 0
    boot_finals: list[float] = []
    for _ in range(args.bootstrap_paths):
        equity = START_EQUITY
        peak = equity
        breached = False
        for _ in range(56):
            equity += WIN if rng.random() < P_WIN_OOS else LOSS
            peak = max(peak, equity)
            if equity < max(FLOOR_MIN, (1.0 - TRAIL) * peak):
                breached = True
                break
        boot_breach += int(breached)
        boot_finals.append(equity)
    boot_finals.sort()
    bootstrap = {
        "trades": 56,
        "paths": args.bootstrap_paths,
        "breach_prob": boot_breach / args.bootstrap_paths,
        "median_final": round(boot_finals[len(boot_finals) // 2], 2),
        "p05_final": round(boot_finals[int(0.05 * len(boot_finals))], 2),
    }

    payload = {
        "generated": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
        "method": "read-only bootstrap/Monte Carlo; fixed canonical payoff",
        "start_equity": START_EQUITY,
        "floor_base": FLOOR_BASE,
        "trail": TRAIL,
        "p_win_oos": P_WIN_OOS,
        "p_win_forward": P_WIN_FORWARD,
        "results": results,
        "sensitivity": sensitivity,
        "bootstrap_oos": bootstrap,
        "recommendation": (
            "Keep max_concurrent_crypto=2. Observed peak concurrency is 2 "
            "in-sample and 1 out-of-sample, so a fourth slot adds no filled "
            "trades; it only doubles the worst simultaneous stop from $13.00 "
            "to $26.00 against a ~$32.80 buffer. The two-slot cap never binds "
            "on the OOS tape."
        ),
    }
    OUT.write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
