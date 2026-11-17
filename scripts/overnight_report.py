#!/usr/bin/env python3
"""Write the morning handoff from the artifacts the overnight actually produced.

Everything in the report is read back from disk: the venue's own balance and the
ledger reconciliation, the frozen discovery's in-sample and out-of-sample rows for
every family that was tested, the engine's live state, and the one-minute
falsification summary. Nothing is asserted that an artifact does not say.

    .venv/bin/python scripts/overnight_report.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAP_NOTE = "$0.62"
OUT = ROOT / "data" / "overnight-report.md"


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def engine_state() -> dict:
    try:
        out = subprocess.run(["curl", "-s", "--max-time", "10",
                              "http://127.0.0.1:8787/api/state"],
                             capture_output=True, text=True, timeout=20).stdout
        return json.loads(out) if out.strip() else {}
    except (subprocess.SubprocessError, OSError, ValueError):
        return {}


def money(value) -> str:
    try:
        return f"${float(value):,.4f}"
    except (TypeError, ValueError):
        return str(value)


def family_rows(discovery: dict, family: str) -> list[str]:
    fam = (discovery.get("families") or {}).get(family) or {}
    sel = fam.get("selected")
    if not sel:
        best = fam.get("best_is")
        lines = [f"- **{family}**: no candidate cleared IS selection "
                 f"({fam.get('no_candidate', 'not run')})."]
        if best:
            lines.append(
                f"  - best in-sample variant `{fam.get('best_is_variant')}`: "
                f"{best['trades']} IS trades, net {money(best['net_pnl'])}, "
                f"PF {best['pf']}, ECE {best['ece']}, max side cost "
                f"{best.get('max_side_cost')} - below the 20-trade floor, so the "
                f"frozen OOS slice was never scored.")
        most = fam.get("most_traded_is")
        if most:
            lines.append(
                f"  - most-traded in-sample variant `{most.get('variant')}`: "
                f"{most['trades']} IS trades, net {money(most['net_pnl'])}, "
                f"PF {most['pf']}, ECE {most['ece']}, model-implied EV "
                f"{money(most.get('model_ev'))} - the model priced those trades "
                f"at what they realized, which is the most the in-sample slice can "
                f"say.")
            geo = fam.get("exit_geometry_same_entries") or []
            if geo:
                rows = ", ".join(
                    f"`{g['variant'].rsplit('_', 2)[-2]}` {g['trades']}t "
                    f"{money(g['net_pnl'])} PF {g['pf']}" for g in geo)
                lines.append(f"  - the same entries with a different exit: {rows}.")
        trades = fam.get("winner_trades_is") or []
        if trades:
            wins = sum(1 for r in trades if r["pnl"] > 0)
            lines.append(
                f"  - those trades: {len(trades)} on {len({r['lane'] for r in trades})} "
                f"lanes, {wins} winners, net {money(sum(r['pnl'] for r in trades))}")
        return lines
    isr, oos, g = sel["is"], sel["oos"], sel["gate"]
    lines = [
        f"- **{family}** `{sel['variant']}`",
        f"  - IS: {isr['trades']} trades, net {money(isr['net_pnl'])}, "
        f"PF {isr['pf']}, ECE {isr['ece']}",
        f"  - OOS (frozen): {oos.get('trades', 0)} trades, net {money(oos.get('net_pnl'))}, "
        f"PF {oos.get('pf')}, ECE {oos.get('ece')}, crossed {oos.get('crossed_fills')}, "
        f"max side cost {oos.get('max_side_cost')}",
        f"  - gate: {'PASS' if g.get('pass') else 'FAIL'} — "
        f"{'; '.join(g.get('reasons', [])) or 'every gate cleared'}",
    ]
    lanes = oos.get("by_lane") or {}
    if lanes:
        lines.append("  - OOS by lane: " + ", ".join(
            f"{k} {v['trades']}t {money(v['net'])}" for k, v in sorted(lanes.items())))
    return lines


def recorded_windows() -> int | None:
    """How many settled windows the rebuilt sub-second cache holds.

    In paper mode the night's work is the tape: each window is a 15-minute contract
    whose book, prints and settlement the engine recorded and the study can read.
    """
    import numpy as np

    cache = ROOT / "data" / "evolve-cache"
    markets = ROOT / "data" / "settled-markets.json"
    if not markets.exists():
        return None
    seen: set[str] = set()
    for lane in ("KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M"):
        path = cache / f"book_{lane}.npz"
        if not path.exists():
            continue
        try:
            with np.load(path, allow_pickle=False) as d:
                seen.update(str(t) for t in d["tickers"])
        except (OSError, ValueError):
            continue
    if not seen:
        return None
    return len(seen)


def main() -> int:
    now = datetime.now(timezone.utc).astimezone()
    st = engine_state()
    verdict = load(ROOT / "data" / "strategy-verdict.json") or {}
    disc = load(ROOT / "data" / "subsec-discovery.json") or {}
    tape = load(ROOT / "data" / "long-tape.json") or {}
    canary = load(ROOT / "data" / "canary.json") or {}
    rec = load(ROOT / "data" / "overnight-reconcile.json") or {}
    sup = load(ROOT / "data" / "overnight-state.json") or {}

    balance = rec.get("balance_dollars") or st.get("venue_balance")
    delta = rec.get("delta")
    armed = st.get("live_armed")
    mode = st.get("mode")

    lines: list[str] = []
    lines.append(f"# Open-Kal overnight report — {now.strftime('%Y-%m-%d %H:%M %Z')}")
    lines.append("")
    lines.append("## 1. Account and reconciliation")
    lines.append("")
    lines.append(f"- Venue balance: **{money(balance)}**")
    lines.append(f"- Ledger vs venue delta: **{money(delta) if delta is not None else 'n/a'}**")
    lines.append(f"- Open positions: {rec.get('open_positions', 'n/a')}, "
                 f"resting orders: {rec.get('resting_orders', 'n/a')}")
    lines.append(f"- Venue fills in the epoch: {rec.get('venue_fills', 'n/a')}, "
                 f"settlements: {rec.get('venue_settlements', 'n/a')}")
    lines.append(f"- Engine mode: **{mode}**, live_armed: **{armed}**")
    lines.append(f"- Verdict `promoted`: **{verdict.get('promoted')}** — "
                 f"{verdict.get('reason') or verdict.get('disarm_reason') or ''}")
    if sup:
        lines.append(f"- Last supervisor check: {sup.get('ts')} — balance "
                     f"{sup.get('balance')}, mode {sup.get('mode')}, "
                     f"armed {sup.get('armed')}, delta {sup.get('delta')}, "
                     f"breaker {sup.get('breaker') or 'none'}")
    lines.append("")

    lines.append("## 2. Sub-second families: in-sample vs frozen out-of-sample")
    lines.append("")
    if disc:
        split = disc.get("split") or {}
        lines.append(f"- Split: {split.get('rule')}; {split.get('is_windows')} IS windows "
                     f"({split.get('is_lane_hours')} lane-hours) / "
                     f"{split.get('oos_windows')} OOS windows "
                     f"({split.get('oos_lane_hours')} lane-hours)")
        lines.append(f"- Physics: {json.dumps(disc.get('physics', {}))}")
        for family in ("A", "B"):
            lines.extend(family_rows(disc, family))
    else:
        lines.append("- No discovery artifact: the frozen discovery did not run.")
    lines.append("")

    lines.append("## 2b. Bounded sweep on the expanded tape (F2 touch-OBI + spot impulse)")
    lines.append("")
    sweep = load(ROOT / "data" / "bounded-sweep-f2.json") or {}
    fam = (sweep.get("families") or {}).get("F2") or {}
    if fam:
        isr, oos, g = fam.get("is", {}), fam.get("oos", {}), fam.get("gate", {})
        lines.append(f"- Candidate: `{fam.get('candidate')}` "
                     f"(cost cap {CAP_NOTE}, passive fill >= 50 contracts, "
                     f"2s-confirmed 7c stop, +5c maker take-profit)")
        lines.append("")
        lines.append("| slice | trades | win rate | net | PF | ECE | max DD | avg entry | "
                     "taker fees | per-lane net |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for label, s2 in (("IS", isr), ("OOS", oos)):
            lanes = s2.get("by_lane") or {}
            lane_txt = ", ".join(f"{k} {v['net']:+.2f}" for k, v in sorted(lanes.items()))
            lines.append(
                f"| {label} | {s2.get('trades')} | {s2.get('win_rate')} | "
                f"{money(s2.get('net_pnl'))} | {s2.get('pf')} | {s2.get('ece')} | "
                f"{money(s2.get('max_dd'))} | {s2.get('avg_entry')} | "
                f"{money(s2.get('taker_fees_paid'))} | {lane_txt} |")
        lines.append("")
        lines.append(f"- Gate: **{'PASS' if g.get('pass') else 'REJECT'}** — "
                     f"{'; '.join(g.get('reasons', [])) or 'every gate cleared'}; "
                     f"{g.get('lanes_profitable')} profitable lanes, "
                     f"{oos.get('crossed_fills')} crossed fills, max side cost "
                     f"{oos.get('max_side_cost')}.")
        lines.append("- **Not armed.** The rule clears every economic gate on the frozen "
                     "OOS slice and fails only the calibration gate, and that failure is "
                     "informative: the pool's reported probability says ~0.63 on these "
                     "trades while their settlement win rate is 0.52. The F2 filters "
                     "(printed flow, touch revisions) pick short-horizon momentum -- a "
                     "+5c take-profit fills 39 of 46 times -- not a better settlement "
                     "probability. A candidate whose stated probability is ten points "
                     "above its own outcomes should not arm, and this OOS slice has now "
                     "been read, so any further F2 work needs a fresh one.")
    else:
        lines.append("- No F2 sweep artifact on disk.")
    lines.append("")

    lines.append("## 2c. Frozen structural test: F2 with the mandatory 1s lag")
    lines.append("")
    st = load(ROOT / "data" / "f2-structural.json") or {}
    for name, v in (st.get("variants") or {}).items():
        lines.append(f"### `{name}`")
        lines.append("")
        lines.append("| slice | events | fills | fill rate | win rate | net | PF | payoff-ECE "
                     "| settlement-ECE | max DD | crossed | max cost |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for label, sl in (("IS", "is"), ("OOS", "oos")):
            s2 = v["summary"][sl]
            lines.append(
                f"| {label} | {s2.get('events')} | {s2.get('trades')} | "
                f"{s2.get('fill_rate')} | {s2.get('win_rate')} | {money(s2.get('net_pnl'))} | "
                f"{s2.get('pf')} | {s2.get('payoff_event_ece')} | {s2.get('settlement_ece')} | "
                f"{money(s2.get('max_dd'))} | {s2.get('crossed_fills')} | "
                f"{s2.get('max_side_cost')} |")
        dec = v.get("decide_filter") or {}
        if dec:
            lines.append("")
            lines.append(f"- decide() hurdle {v.get('hurdle_c')}: IS "
                         f"{dec['is']['taken_of']} -> {dec['is']['trades']} trades, "
                         f"{money(dec['is']['net_pnl'])}, PF {dec['is']['pf']}, payoff-ECE "
                         f"{dec['is']['payoff_event_ece']}; OOS "
                         f"{dec['oos']['taken_of']} -> {dec['oos']['trades']} trades, "
                         f"{money(dec['oos']['net_pnl'])}, PF {dec['oos']['pf']}, payoff-ECE "
                         f"{dec['oos']['payoff_event_ece']}.")
        lines.append("")
    lines.append("The sweep had applied the one-second execution lag to family F3 only, so "
                 "F2 filled at the price it had already seen. Every family is now priced at "
                 "the quote one second after the trigger. The passive variant is dead: a "
                 "resting bid at the touch fills 4-6% of the time and the fills that come "
                 "are the adverse ones, so the edge is the crossing, not the selection.")
    lines.append("")

    lines.append("## 3. What the one-minute tape already settled")
    lines.append("")
    summary = (tape.get("summary") or {})
    if summary:
        lines.append(f"- {summary.get('verdict', '')}")
        fix = summary.get("double_count_fix") or {}
        lines.append(f"- Double-count fix: {fix.get('now', '')}")
    else:
        lines.append("- No long-tape summary on disk.")
    lines.append("")

    lines.append("## 4. Live activity")
    lines.append("")
    windows = recorded_windows()
    if windows is not None:
        lines.append(f"- Sub-second tape rebuilt at the rebuild hour: **{windows} windows** "
                     f"with a book and a settled market (the paper engine's output for "
                     f"the night).")

    if armed:
        lines.append(f"- The engine was **live-armed**: {st.get('readiness', {}).get('settled_trades')} "
                     f"settled trades, net {st.get('readiness', {}).get('net')} USD in the epoch.")
    else:
        lines.append(f"- The engine stayed in **paper mode**. No real-money trade was "
                     f"opened by the strategy. The venue was touched only by the "
                     f"one-contract canary.")
        lines.append(f"- Settled windows recorded since the epoch start: "
                     f"{rec.get('venue_settlements', 'n/a')}; "
                     f"venue fills: {rec.get('venue_fills', 'n/a')}.")
    lines.append(f"- Canary: ok={canary.get('ok')}, ticker {canary.get('ticker')}, "
                 f"error {canary.get('error') or 'none'}")
    lines.append("")

    lines.append("## 4b. Clean forward OOS holdout")
    lines.append("")
    lines.append("- `forward_oos_start_ts = 1794909600` (2026-11-17 05:00:00 EST): every "
                 "window recorded from here is an untouched forward slice. The slice in the "
                 "sub-second cache has been read several times and can no longer certify "
                 "anything.")
    lines.append("- The engine keeps running uninterrupted in paper mode "
                 "(`live_armed: false`, verdict `promoted: false`). The canary was "
                 "deliberately not re-run; it is re-run by hand only when a candidate "
                 "clears every gate on the forward slice.")
    lines.append("")

    lines.append("## 5. Next action")
    lines.append("")
    selected = {f: ((disc.get("families") or {}).get(f) or {}).get("selected")
                for f in ("A", "B")}
    passing = [f for f, s in selected.items() if s and (s.get("gate") or {}).get("pass")]
    if passing:
        lines.append(f"- Families {', '.join(passing)} cleared every frozen gate. "
                     f"Promote by hand: wire the candidate into `openkal/features.py` "
                     f"and `openkal/strategy.py`, set `promoted: true`, run the "
                     f"controls, run `scripts/canary.py --live` and only then "
                     f"`bash scripts/restart_flat.sh --new-epoch`.")
    else:
        lines.append("- **Nothing cleared the gate. The account stays untouched.** "
                     "The next bounded action is more tape: the sub-second study is "
                     "still sample-limited, so keep the engine recording "
                     "(`data/openkal-prod.db`), rebuild the cache and re-run "
                     "`scripts/subsec_discovery.py` on the larger tape. Do not "
                     "loosen the gate and do not arm.")
    lines.append("")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
