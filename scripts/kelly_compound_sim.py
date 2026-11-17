"""Compounding simulation for the staged fractional-Kelly sizing table.

The live lane no longer risks a flat $15: `openkal.config.COMPOUND_STAGES` sets the
fraction of account equity a single entry may risk (12.5% below $175, 20% up to
$350, 25% above) with 50/120/250 contract ceilings, while the $0.62 cost ceiling and
the 20% trailing-peak breaker hold on every stage.

This replays a frozen trade tape through that table so the path to $175 and $350 is
measured rather than assumed:

    .venv/bin/python scripts/kelly_compound_sim.py --equity 121.46
    .venv/bin/python scripts/kelly_compound_sim.py --json --cycles 50

The tape is `data/forward-oos.json` -- the real OOS rows from `scripts/forward_oos.py`,
each carrying its entry, qty and net pnl. Every row is rescaled by its own return on
premium (`pnl / (entry * qty)`) to the stage's risk budget, so the simulation carries
the tape's payoff distribution instead of a fitted win rate, and is replayed in
cycles until the breaker trips or the cycles run out. Nothing here places an order,
reads the live account or writes state.
"""
from __future__ import annotations

import argparse
import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openkal.config import (COMPOUND_STAGES, HARD_MAX_ENTRY_PRICE,  # noqa: E402
                            peak_drawdown_tripped, stage_for, staged_risk_cap)

TAPE = ROOT / "data" / "forward-oos.json"
CENT = Decimal("0.01")


def _money(value: Decimal) -> str:
    return str(Decimal(value).quantize(CENT, rounding=ROUND_HALF_UP))


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


def render(report: dict, rows: int) -> str:
    start = Decimal(report["start"])
    lines = [
        f"start ${report['start']} -> ${report['end']}  "
        f"({report['trades']} trades over {report['cycles']} cycle(s) of {rows} rows, "
        f"{report['skipped']} skipped)",
        "",
        "stage table (risk of account equity, hard $0.62 entry ceiling on every stage):",
    ]
    for stage in COMPOUND_STAGES:
        # Stage 1's floor is zero, so show its cap at the account this run starts on.
        reference = max(stage.equity_floor, start)
        lines.append(f"  {stage.name}: equity >= ${stage.equity_floor}  "
                     f"risk {stage.max_risk_pct * 100:g}%  "
                     f"<= {stage.max_contracts} contracts  "
                     f"(at ${reference} -> ${_money(reference * stage.max_risk_pct)})")
    lines += ["", "reached:"]
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay a frozen trade tape through the "
                                                 "staged fractional-Kelly sizing table.")
    parser.add_argument("--equity", default="121.46", help="starting account equity (USD)")
    parser.add_argument("--tape", default=str(TAPE), help="forward-oos.json style trade tape")
    parser.add_argument("--cycles", type=int, default=20, help="tape replays before stopping")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    path = Path(args.tape)
    if not path.exists():
        print(f"no tape at {path}", file=sys.stderr)
        return 1
    rows = load_tape(path)
    if not rows:
        print(f"tape at {path} has no priced rows", file=sys.stderr)
        return 1
    report = simulate(rows, Decimal(str(args.equity)), args.cycles)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render(report, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
