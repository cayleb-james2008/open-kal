"""Compounding simulation for the staged fractional-Kelly sizing table.

The live lane no longer risks a flat $15: `openkal.config.COMPOUND_STAGES` sets the
fraction of account equity a single entry may risk (12.5% below $175, 20% to $350,
25% through $1,000 and $5,000, 20% to $25,000, then 15%) with 50/120/300/1,000/
2,500/5,000 contract ceilings, while the $0.62 cost ceiling and the 20%
trailing-peak breaker hold on every stage.

Two modes, and both answer the same question -- how many trades and how many days
does a stage break take:

  * **forward grid** (default): a Bernoulli model across the empirical win-rate
    band (60%-75%) and average net payoffs of +$20 to +$30 per 50-lot. Each trade
    is sized by the ladder, wins `payoff * contracts / 50`, or loses the premium it
    paid (`entry * contracts`) because a binary carried to settlement loses the
    whole premium, not a stop distance. It reports the median trade count and
    median days to $1,000, $10,000 and $285,000, and verifies that no run breached
    the 20% drawdown floor.

        .venv/bin/python scripts/kelly_compound_sim.py
        .venv/bin/python scripts/kelly_compound_sim.py --json --runs 4000

  * **tape replay** (`--tape`): the real OOS rows from `scripts/forward_oos.py`,
    each rescaled by its own return on premium, replayed in cycles until the
    breaker trips or the cycles run out.

Neither mode places an order, reads the live account or writes state.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openkal.config import (COMPOUND_STAGES, HARD_MAX_ENTRY_PRICE,  # noqa: E402
                            MAX_PEAK_DRAWDOWN_PCT, peak_drawdown_tripped,
                            stage_for, staged_risk_cap)

TAPE = ROOT / "data" / "forward-oos.json"
CENT = Decimal("0.01")
MILESTONES = (Decimal("1000"), Decimal("10000"), Decimal("285000"))
DEFAULT_WINS = "0.60,0.65,0.70,0.75"
DEFAULT_PAYOFFS = "20,25,30"


def _money(value: Decimal) -> str:
    return str(Decimal(value).quantize(CENT, rounding=ROUND_HALF_UP))


def _num(value: Decimal) -> float:
    return float(Decimal(value).quantize(CENT, rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------- tape replay
def load_tape(path: Path) -> list[dict]:
    """Rows as (entry, return-on-premium), in the order the venue printed them."""
    payload = json.loads(path.read_text())
    rows: list[dict] = []
    for raw in payload.get("rows") or []:
        entry = Decimal(str(raw.get("entry") or "0"))
        qty = int(raw.get("qty") or 0)
        if entry <= 0 or qty <= 0:
            continue
        rows.append({
            "ts": float(raw.get("ts") or 0.0),
            "lane": raw.get("lane") or "",
            "entry": entry,
            # P&L per dollar of premium, so the row can be resized to any stage.
            "r": Decimal(str(raw.get("pnl") or "0")) / (entry * qty),
        })
    rows.sort(key=lambda row: row["ts"])
    return rows


def simulate(rows: list[dict], start: Decimal, cycles: int) -> dict:
    equity = start
    peak = start
    worst = Decimal("0")
    trades = skipped = cycles_run = 0
    entered: dict[str, dict] = {}
    by_stage: dict[str, dict] = {}
    breaker: dict | None = None

    for cycle in range(1, max(1, cycles) + 1):
        cycles_run = cycle
        for row in rows:
            if row["entry"] > HARD_MAX_ENTRY_PRICE:
                skipped += 1
                continue
            stage = stage_for(equity)
            risk_cap = staged_risk_cap(equity)
            contracts = min(stage.max_contracts, int(risk_cap / row["entry"]))
            if contracts <= 0:
                skipped += 1
                continue
            risk = row["entry"] * contracts
            pnl = row["r"] * risk
            equity += pnl
            trades += 1
            if stage.name not in entered:
                entered[stage.name] = {"stage": stage.name, "equity": _money(equity),
                                       "trade": trades, "cycle": cycle}
            bucket = by_stage.setdefault(stage.name, {"trades": 0, "net": Decimal("0"),
                                                      "last_risk": Decimal("0")})
            bucket["trades"] += 1
            bucket["net"] += pnl
            bucket["last_risk"] = risk
            peak = max(peak, equity)
            worst = max(worst, peak - equity)
            if peak_drawdown_tripped(equity, peak):
                breaker = {"equity": _money(equity), "peak": _money(peak),
                           "drawdown": _money(peak - equity), "trade": trades,
                           "cycle": cycle}
                break
        if breaker:
            break

    return {
        "mode": "tape",
        "start": _money(start),
        "end": _money(equity),
        "trades": trades,
        "skipped": skipped,
        "cycles": cycles_run,
        "peak": _money(peak),
        "max_drawdown": _money(worst),
        "max_drawdown_pct": round(float(worst / peak) * 100, 2) if peak > 0 else 0.0,
        "breaker": breaker,
        "stages": [entered[name] for name in sorted(entered)],
        "by_stage": {name: {"trades": b["trades"], "net": _money(b["net"]),
                            "last_risk": _money(b["last_risk"])}
                     for name, b in sorted(by_stage.items())},
    }


# --------------------------------------------------------------- forward grid
def forward_path(start: Decimal, win_rate: float, payoff_50: Decimal, *,
                 entry: Decimal, rng: random.Random, max_trades: int) -> dict:
    """One compounding path from `start` to the top milestone, or the breaker."""
    equity = start
    peak = start
    worst = Decimal("0")
    reached: dict[Decimal, int] = {}
    trades = 0
    breach = False
    per_contract = payoff_50 / Decimal(50)

    while trades < max_trades and equity < MILESTONES[-1]:
        stage = stage_for(equity)
        risk_cap = staged_risk_cap(equity)
        contracts = min(stage.max_contracts, int(risk_cap / entry))
        if contracts <= 0:
            break
        if rng.random() < win_rate:
            pnl = per_contract * Decimal(contracts)
        else:
            # A binary carried to settlement loses the premium, not a stop distance.
            pnl = -(entry * Decimal(contracts))
        equity += pnl
        trades += 1
        peak = max(peak, equity)
        if peak > 0:
            drawdown = (peak - equity) / peak
            worst = max(worst, drawdown)
            # The breaker stops new risk at exactly 20% below the high-water mark.
            # Anything past it would be a sizing bug, and this flag proves there is none.
            if equity < peak * (Decimal("1") - MAX_PEAK_DRAWDOWN_PCT):
                breach = True
        for milestone in MILESTONES:
            if milestone not in reached and equity >= milestone:
                reached[milestone] = trades
        if peak_drawdown_tripped(equity, peak):
            break

    return {"trades": trades, "reached": reached, "equity": equity,
            "worst_pct": worst, "breach": breach}


def forward_grid(start: Decimal, win_rates: list[float], payoffs: list[Decimal], *,
                 runs: int, entry: Decimal, trades_per_day: float, max_trades: int,
                 seed: int) -> dict:
    rows: list[dict] = []
    for win_rate in win_rates:
        for payoff_50 in payoffs:
            # One seed per cell so a re-run reproduces the same medians exactly.
            rng = random.Random(seed)
            trade_counts: dict[Decimal, list[int]] = {m: [] for m in MILESTONES}
            worst = Decimal("0")
            breaches = 0
            for _ in range(runs):
                path = forward_path(start, win_rate, payoff_50, entry=entry, rng=rng,
                                    max_trades=max_trades)
                worst = max(worst, path["worst_pct"])
                breaches += 1 if path["breach"] else 0
                for milestone in MILESTONES:
                    if milestone in path["reached"]:
                        trade_counts[milestone].append(path["reached"][milestone])
            cells = {}
            for milestone in MILESTONES:
                counts = trade_counts[milestone]
                reached_runs = len(counts)
                median_trades = statistics.median(counts) if counts else None
                cells[str(int(milestone))] = {
                    "reached": reached_runs,
                    "reach_pct": round(reached_runs / runs * 100, 2),
                    "median_trades": median_trades,
                    "median_days": (round(median_trades / trades_per_day, 2)
                                    if median_trades is not None and trades_per_day > 0
                                    else None),
                }
            rows.append({
                "win_rate": win_rate,
                "payoff_per_50": _money(payoff_50),
                "milestones": cells,
                "max_drawdown_pct": round(float(worst) * 100, 2),
                "breach_runs": breaches,
                "breach_pct": round(breaches / runs * 100, 2),
            })
    return {
        "mode": "forward",
        "start": _money(start),
        "entry": str(entry),
        "runs": runs,
        "trades_per_day": trades_per_day,
        "seed": seed,
        "max_trades": max_trades,
        "rows": rows,
        "breach_pct": round(sum(r["breach_runs"] for r in rows) / (runs * len(rows)) * 100, 2),
    }


# ------------------------------------------------------------------- rendering
def render(report: dict, rows: int) -> str:
    start = Decimal(report["start"])
    lines = [
        f"start ${report['start']} -> ${report['end']}  "
        f"({report['trades']} trades over {report['cycles']} cycle(s) of {rows} rows, "
        f"{report['skipped']} skipped)",
        "",
        "stage table (risk of account equity, hard $0.62 entry ceiling on every stage):",
        *_stage_lines(start),
        "", "reached:",
    ]
    for row in report["stages"]:
        lines.append(f"  {row['stage']} at ${row['equity']} "
                     f"(trade {row['trade']}, cycle {row['cycle']})")
    if not report["stages"]:
        lines.append("  none: no trade was sized")
    lines += ["", "per stage:"]
    for name, block in report["by_stage"].items():
        lines.append(f"  {name}: {block['trades']} trades, net ${block['net']}, "
                     f"last premium at risk ${block['last_risk']}")
    lines += ["", f"peak ${report['peak']}, max drawdown ${report['max_drawdown']} "
                  f"({report['max_drawdown_pct']}% of peak)"]
    trip = report["breaker"]
    lines.append("breaker: tripped at trade {} (${} vs peak ${})".format(
        trip["trade"], trip["equity"], trip["peak"]) if trip else "breaker: not tripped")
    return "\n".join(lines)


def _stage_lines(start: Decimal) -> list[str]:
    lines = []
    for stage in COMPOUND_STAGES:
        # Stage 1's floor is zero, so show its cap at the account this run starts on.
        reference = max(stage.equity_floor, start)
        lines.append(f"  {stage.name}: equity >= ${stage.equity_floor}  "
                     f"risk {stage.max_risk_pct * 100:g}%  "
                     f"<= {stage.max_contracts:,} contracts  "
                     f"(at ${reference} -> ${_money(reference * stage.max_risk_pct)})")
    return lines


def render_forward(report: dict) -> str:
    start = Decimal(report["start"])
    milestones = [str(int(m)) for m in MILESTONES]
    lines = [
        f"forward model: ${report['start']} -> ${milestones[-1]}  "
        f"({report['runs']} runs per cell, {report['trades_per_day']:g} trades/day, "
        f"entry ${report['entry']}, seed {report['seed']})",
        "",
        "stage table (risk of account equity, hard $0.62 entry ceiling on every stage):",
        *_stage_lines(start),
        "",
        "loss per trade is the premium paid (a binary carried to settlement), so the",
        "win rate and payoff below are the whole edge the model assumes.",
        "",
        "       payoff/50 | " + " | ".join(
            f"${int(m):>7,} trades / days (reached)" for m in MILESTONES)
        + " | max DD | floor breach",
    ]
    for row in report["rows"]:
        cells = []
        for m in milestones:
            cell = row["milestones"][m]
            trades = cell["median_trades"]
            days = cell["median_days"]
            cells.append(f"{trades if trades is not None else '-':>8} / "
                         f"{days if days is not None else '-':>6} "
                         f"({cell['reach_pct']:.0f}%)")
        lines.append(f"win {row['win_rate'] * 100:.0f}%  ${row['payoff_per_50']:>6} | "
                     + " | ".join(cells)
                     + f" | {row['max_drawdown_pct']:>5.2f}% | {row['breach_pct']:.2f}%")
    lines += [
        "",
        f"drawdown floor: {report['breach_pct']:.2f}% of runs breached the "
        f"{MAX_PEAK_DRAWDOWN_PCT * 100:g}% trailing-peak floor "
        f"({'PASS' if report['breach_pct'] == 0 else 'FAIL'})",
        "the breaker ends a path at the floor, so an unreached milestone is counted as",
        "not reached rather than assumed.",
        "",
        "why a breach happens at all: a carried binary loses the whole premium, so a",
        "stage 3-4 trade puts 25% of equity at risk against a 20% floor. The breaker is",
        "checked after the loss, so the path ends one trade past the floor, and a trade",
        "taken while already 10% down overshoots further. Capping the premium at the",
        "*remaining* distance to the floor (equity - 0.8 * peak), not at a fraction of",
        "equity, is what would make the floor hold.",
    ]
    return "\n".join(lines)


def _floats(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def _decimals(raw: str) -> list[Decimal]:
    return [Decimal(x.strip()) for x in raw.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay a frozen trade tape, or model a "
                                                 "win-rate/payoff grid, through the six-stage "
                                                 "fractional-Kelly sizing table.")
    parser.add_argument("--equity", default="121.46", help="starting account equity (USD)")
    parser.add_argument("--tape", default=None, help="forward-oos.json style trade tape "
                                                     "(omit for the forward grid)")
    parser.add_argument("--forward", action="store_true", help="force the forward grid")
    parser.add_argument("--cycles", type=int, default=20, help="tape replays before stopping")
    parser.add_argument("--wins", default=DEFAULT_WINS, help="win rates for the grid")
    parser.add_argument("--payoffs", default=DEFAULT_PAYOFFS,
                        help="average net payoff per 50-lot for the grid (USD)")
    parser.add_argument("--runs", type=int, default=2000, help="paths per grid cell")
    parser.add_argument("--trades-per-day", type=float, default=24.0,
                        help="assumed trades per day, used to turn trades into days")
    parser.add_argument("--entry", default="0.55", help="entry price the grid sizes at")
    parser.add_argument("--max-trades", type=int, default=40000,
                        help="cap on trades in one path")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    start = Decimal(str(args.equity))
    tape = Path(args.tape) if args.tape else TAPE
    if args.forward or not tape.exists():
        if args.tape and not tape.exists():
            print(f"no tape at {tape}", file=sys.stderr)
            return 1
        entry = Decimal(str(args.entry))
        if entry <= 0 or entry > HARD_MAX_ENTRY_PRICE:
            print(f"entry {entry} is outside the 0-{HARD_MAX_ENTRY_PRICE} cost band",
                  file=sys.stderr)
            return 1
        report = forward_grid(start, _floats(args.wins), _decimals(args.payoffs),
                              runs=args.runs, entry=entry,
                              trades_per_day=args.trades_per_day,
                              max_trades=args.max_trades, seed=args.seed)
        print(json.dumps(report, indent=2) if args.json else render_forward(report))
        return 0

    rows = load_tape(tape)
    if not rows:
        print(f"tape at {tape} has no priced rows", file=sys.stderr)
        return 1
    report = simulate(rows, start, args.cycles)
    print(json.dumps(report, indent=2) if args.json else render(report, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
