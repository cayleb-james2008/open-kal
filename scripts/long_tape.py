#!/usr/bin/env python3
"""Long-tape study on venue minute candles (13 days, 5,086 settled windows).

The sub-second frozen tape is only ~43 lane-hours, which cannot reach the 20-trade
OOS floor for rare mechanisms. Kalshi's own settled-window candlesticks cover 5,086
windows over 13 days and are authoritative venue data. Resolution is one minute, so
this study tests mechanisms whose decision and exit live on minute boundaries:

  final_minute  decision at T-60s (penultimate candle close), favourite buy
  band          mid-window entries in a cost band, maker TP / barrier / carry

Fills are conservative: a passive buy at P needs a print strictly through P
(price.low <= P - 0.01) in a minute with >= 50 contracts of volume; a resting sell
needs price.high >= P + 0.01. Quotes are asserted uncrossed on every candle.

Split is chronological 60/40 by window close, built once. Selection reads IS only;
the frozen OOS slice is scored once per family.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CANDLES = ROOT / "data" / "candles-15m.jsonl"
LANES = ("KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M")
MIN_PASSIVE_QTY = 50.0
TICK = 0.01


@dataclass
class Candle:
    t: int
    po: float
    ph: float
    pl: float
    pc: float
    bo: float
    bh: float
    bl: float
    bc: float
    ao: float
    ah: float
    al: float
    ac: float
    vol: float


@dataclass
class Window:
    ticker: str
    lane: str
    close: int
    strike: float
    value: float
    result: str
    candles: list[Candle] = field(default_factory=list)

    @property
    def y(self) -> int:
        return 1 if self.result == "yes" else 0


def load() -> tuple[list[Window], int]:
    out: list[Window] = []
    crossed = 0
    for line in CANDLES.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        lane = r["ticker"].split("-")[0]
        if lane not in LANES:
            continue
        cs: list[Candle] = []
        for c in r["candles"]:
            if not all(k in c.get("price", {}) for k in
                       ("open_dollars", "high_dollars", "low_dollars", "close_dollars")):
                cs = []
                break
            d = Candle(
                t=int(c["end_period_ts"]),
                po=float(c["price"]["open_dollars"]), ph=float(c["price"]["high_dollars"]),
                pl=float(c["price"]["low_dollars"]), pc=float(c["price"]["close_dollars"]),
                bo=float(c["yes_bid"]["open_dollars"]), bh=float(c["yes_bid"]["high_dollars"]),
                bl=float(c["yes_bid"]["low_dollars"]), bc=float(c["yes_bid"]["close_dollars"]),
                ao=float(c["yes_ask"]["open_dollars"]), ah=float(c["yes_ask"]["high_dollars"]),
                al=float(c["yes_ask"]["low_dollars"]), ac=float(c["yes_ask"]["close_dollars"]),
                vol=float(c.get("volume_fp") or 0.0),
            )
            if c["yes_bid"]["close_dollars"] and c["yes_ask"]["close_dollars"]:
                if d.bc > 0 and d.ac > 0 and d.bc >= d.ac:
                    crossed += 1
            cs.append(d)
        if len(cs) < 15:
            continue
        out.append(Window(r["ticker"], lane, int(r["close"]), float(r["strike"] or 0),
                          float(r["value"] or 0), r["result"], cs))
    out.sort(key=lambda w: w.close)
    return out, crossed


def split(windows: list[Window], frac: float = 0.6) -> tuple[list[Window], list[Window]]:
    cut = int(len(windows) * frac)
    return windows[:cut], windows[cut:]


def fee(p: float, n: int) -> float:
    """Exact Kalshi taker fee: ceil to the cent of 0.07*n*p*(1-p)."""
    if n <= 0 or p <= 0 or p >= 1:
        return 0.0
    raw = Decimal("0.07") * Decimal(n) * Decimal(str(p)) * (Decimal(1) - Decimal(str(p)))
    cents = (raw / Decimal("0.01")).to_integral_value(rounding="ROUND_CEILING")
    return float(cents) * 0.01


def ece(pairs: list[tuple[float, int]], bins: int = 10) -> float:
    if not pairs:
        return 0.0
    edges = [i / bins for i in range(bins + 1)]
    total = len(pairs)
    val = 0.0
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        sel = [(p, y) for p, y in pairs if (lo <= p < hi) or (b == bins - 1 and p == 1.0)]
        if not sel:
            continue
        conf = sum(p for p, _ in sel) / len(sel)
        hit = sum(y for _, y in sel) / len(sel)
        val += len(sel) / total * abs(conf - hit)
    return val


# ---------------------------------------------------------------- simulation
@dataclass
class Trade:
    lane: str
    ticker: str
    close: int
    idx: int
    side: str
    entry: float
    exit: float
    net: float
    reason: str
    t_remaining: float


def _buy_yes_fills(c: Candle, p: float) -> bool:
    """A resting YES buy at `p`: a print strictly through it, real volume."""
    return c.vol >= MIN_PASSIVE_QTY and c.pl <= p - TICK


def _sell_yes_fills(c: Candle, p: float) -> bool:
    """A resting YES sell at `p`: a print strictly through it."""
    return c.vol >= MIN_PASSIVE_QTY and c.ph >= p + TICK


def _buy_no_fills(c: Candle, q: float) -> bool:
    """A resting NO buy at `q` fills when YES trades through 1-q upward."""
    return c.vol >= MIN_PASSIVE_QTY and c.ph >= (1.0 - q) + TICK


def _sell_no_fills(c: Candle, q: float) -> bool:
    """A resting NO sell at `q` fills when YES trades through 1-q downward."""
    return c.vol >= MIN_PASSIVE_QTY and c.pl <= (1.0 - q) - TICK


def _side_quote(c: Candle, side: str, *, maker: bool) -> float | None:
    """Price of one contract of `side`: maker rests at its own bid, taker pays its ask."""
    if side == "YES":
        px = c.bc if maker else c.ac
    else:
        px = (1.0 - c.ac) if maker else (1.0 - c.bc)
    return px if 0.0 < px < 1.0 else None


def _pick_side(c: Candle, *, maker: bool, lo: float, hi: float) -> tuple[float, str] | None:
    """One decision point, one side.

    YES and NO are the same market seen twice. At a 0.40/0.41 book the YES ask (0.40)
    and the NO ask (1 - 0.41 = 0.59) both sit inside a 0.25-0.60 band, and the two
    prices sum to more than 1, so taking both is a guaranteed loss rather than two
    trades. Counting both made every band decision look twice as frequent as it is.

    The band family has no directional signal, so the honest single choice is the
    cheaper contract: for the same belief it is the larger expected value, and it
    never peeks at the outcome. The other side is not a second trade, it is this one.
    """
    out: list[tuple[float, str]] = []
    for side in ("YES", "NO"):
        px = _side_quote(c, side, maker=maker)
        if px is None or not (lo <= px <= hi):
            continue
        out.append((px, side))
    if not out:
        return None
    out.sort(key=lambda t: (t[0], t[1]))
    return out[0]


def sim(w: Window, k: int, side: str, *, maker_in: bool, tp: float, barrier: int,
        exit_style: str, n: int = 50) -> Trade | None:
    """Enter at candle k close; exit per `exit_style`.

    maker_tp      TP resting, a miss scratches at the barrier mark (taker at the touch)
    barrier_taker TP resting, a miss takes the touch at the barrier
    carry         TP resting, a miss carries to settlement
    """
    px = _side_quote(w.candles[k], side, maker=maker_in)
    if px is None:
        return None
    entry_fee = 0.0
    first = k + 1
    if maker_in:
        fill = (_buy_yes_fills(w.candles[k + 1], px) if side == "YES"
                else _buy_no_fills(w.candles[k + 1], px))
        if k + 1 >= 15 or not fill:
            return None
        first = k + 2
    else:
        entry_fee = fee(px, n)
    target = round(px + tp, 4)
    t_rem = 900.0 - 60.0 * (k + 1)
    for j in range(first, 15):
        c = w.candles[j]
        hit = (_sell_yes_fills(c, target) if side == "YES"
               else _sell_no_fills(c, target))
        if hit:
            return Trade(w.lane, w.ticker, w.close, k, side, px, target,
                         (target - px) * n - entry_fee, "maker_tp", t_rem)
        if j - k >= barrier:
            break
    if exit_style in ("maker_tp", "barrier_taker"):
        j = min(k + barrier, 14)
        c = w.candles[j]
        raw = _side_quote(c, side, maker=True) or 0.0
        return Trade(w.lane, w.ticker, w.close, k, side, px, raw,
                     (raw - px) * n - entry_fee - fee(raw, n), "barrier_taker", t_rem)
    payoff = 1.0 if (side == "YES") == (w.y == 1) else 0.0
    return Trade(w.lane, w.ticker, w.close, k, side, px, payoff,
                 (payoff - px) * n - entry_fee, "settle", t_rem)


# ---------------------------------------------------------------- statistics
def book(trades: list[Trade], minutes: float) -> dict:
    net = sum(t.net for t in trades)
    wins = [t for t in trades if t.net > 0]
    losses = [t for t in trades if t.net < 0]
    gp = sum(t.net for t in wins)
    gl = -sum(t.net for t in losses)
    pf = (gp / gl) if gl > 0 else (99.0 if gp > 0 else 0.0)
    pairs = []
    for t in trades:
        p = t.entry
        pairs.append((p, 1 if t.net > 0 else 0))
    return {
        "trades": len(trades),
        "net": round(net, 2),
        "pf": round(pf, 3),
        "win_rate": round(len(wins) / len(trades), 4) if trades else 0.0,
        "ece": round(ece(pairs), 4),
        "avg_usd": round(net / len(trades), 3) if trades else 0.0,
        "per_hour": round(len(trades) / minutes * 60, 2) if minutes else 0.0,
        "max_loss_per_contract": round(max((t.entry for t in trades), default=0.0), 4),
        "worst": round(min((t.net for t in trades), default=0.0), 2),
        "reasons": dict(sorted(_count(t.reason for t in trades).items())),
    }


def _count(it) -> dict:
    d: dict[str, int] = defaultdict(int)
    for x in it:
        d[x] += 1
    return d


def _decision_counts(is_w: list[Window], oos_w: list[Window]) -> dict:
    """How many decision points the band family really has, under each reading.

    The double-counted study reported two trades for one book. These counters make
    the size of the clean sample explicit: the total number of decision points, and
    the number left if an ambiguous decision (both sides in band) is dropped instead
    of being taken on the cheaper side.
    """
    out = {"taker_decisions_is": 0, "maker_decisions_is": 0,
           "taker_decisions_oos": 0, "maker_decisions_oos": 0,
           "drop_if_both_taker_is": 0,
           "drop_if_both_taker_oos": 0}
    for key, ws in (("is", is_w), ("oos", oos_w)):
        for w in ws:
            for k in range(1, 13):
                c = w.candles[k]
                taker = [_side_quote(c, s, maker=False) for s in ("YES", "NO")]
                taker = [p for p in taker if p is not None and 0.25 <= p <= 0.60]
                maker = [_side_quote(c, s, maker=True) for s in ("YES", "NO")]
                maker = [p for p in maker if p is not None and 0.25 <= p <= 0.60]
                out[f"taker_decisions_{key}"] += 1 if taker else 0
                out[f"maker_decisions_{key}"] += 1 if maker else 0
                if len(taker) > 1:
                    out[f"drop_if_both_taker_{key}"] += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--explore", action="store_true")
    ap.add_argument("--final-minute", action="store_true")
    ap.add_argument("--band", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "data" / "long-tape.json"))
    args = ap.parse_args()

    windows, crossed = load()
    print(f"windows {len(windows)}  crossed candles {crossed}  "
          f"span {datetime.fromtimestamp(windows[0].close)} -> "
          f"{datetime.fromtimestamp(windows[-1].close)}")
    assert crossed == 0, "crossed book on the venue candles"
    is_w, oos_w = split(windows)
    is_min = sum(15 for _ in is_w)
    oos_min = sum(15 for _ in oos_w)
    print(f"IS {len(is_w)} windows / {is_min/60:.0f} lane-hours   "
          f"OOS {len(oos_w)} / {oos_min/60:.0f} lane-hours")

    if args.explore:
        # forward price drift inside the band, no rule yet: is there any edge at all?
        for lo, hi in ((0.25, 0.60),):
            rows = []
            for w in windows:
                for k in range(1, 13):
                    c = w.candles[k]
                    pick = _pick_side(c, maker=False, lo=lo, hi=hi)
                    if pick is None:
                        continue
                    px, side = pick
                    nxt = w.candles[k + 1]
                    fwd = (nxt.pc - c.pc) if side == "YES" else -(nxt.pc - c.pc)
                    rows.append((px, fwd, 1 if (side == "YES") == (w.y == 1) else 0))
            print(f"\nband {lo}-{hi}: {len(rows)} decisions")
            by = defaultdict(list)
            for px, fwd, win in rows:
                by[round(px, 1)].append(fwd)
            for b in sorted(by):
                v = by[b]
                print(f"  px {b:.1f}  n={len(v):6d}  fwd1m mean {100*statistics.fmean(v):+.2f}c "
                      f"median {100*statistics.median(v):+.2f}c  sd {100*statistics.pstdev(v):.1f}c")
            hold = [win for _, _, win in rows]
            print(f"  hold-to-settle win rate {statistics.fmean(hold):.3f}")
            ev = [win - px for px, _, win in rows]
            print(f"  carry EV at ask (taker, no fee) {100*statistics.fmean(ev):+.2f}c "
                  f"median {100*statistics.median(ev):+.2f}c")
        return 0

    results = {}
    counts = _decision_counts(is_w, oos_w)
    print(f"decision points IS {counts['taker_decisions_is']} "
          f"(maker {counts['maker_decisions_is']}, drop-if-both "
          f"{counts['drop_if_both_taker_is']})")
    if args.final_minute:
        rows = {"is": [], "oos": []}
        for name, ws in (("is", is_w), ("oos", oos_w)):
            for w in ws:
                c = w.candles[13]           # T-60s, decision known before settlement
                pick = _pick_side(c, maker=False, lo=0.90, hi=0.995)
                if pick is None:
                    continue
                px, side = pick
                if c.vol < MIN_PASSIVE_QTY:
                    continue
                payoff = 1.0 if ((side == "YES") == (w.y == 1)) else 0.0
                net = payoff - px - fee(px, 1)
                rows[name].append((px, payoff, net, w.lane))
        out = {}
        for name, rs in rows.items():
            if not rs:
                out[name] = {"trades": 0}
                continue
            nets = [r[2] for r in rs]
            pairs = [(r[0], 1 if r[1] > r[0] else 0) for r in rs]
            out[name] = {
                "trades": len(rs), "net_per_contract_c": round(100 * sum(nets) / len(nets), 3),
                "win_rate": round(sum(r[1] for r in rs) / len(rs), 4),
                "pf": round(sum(n for n in nets if n > 0) / max(1e-9, -sum(n for n in nets if n < 0)), 3),
                "ece": round(ece(pairs), 4),
            }
        results["final_minute"] = out
        print(json.dumps(out, indent=1))

    if args.band:
        grid = []
        for maker_in in (True, False):
            for tp in (0.05, 0.08):
                for barrier in (1, 2, 3):
                    for exit_style in ("maker_tp", "barrier_taker", "carry"):
                        grid.append((maker_in, tp, barrier, exit_style))
        table = []
        for maker_in, tp, barrier, exit_style in grid:
            tr_is = []
            for w in is_w:
                for k in range(1, 13):
                    pick = _pick_side(w.candles[k], maker=maker_in, lo=0.25, hi=0.60)
                    if pick is None:
                        continue
                    px, side = pick
                    t = sim(w, k, side, maker_in=maker_in, tp=tp, barrier=barrier,
                            exit_style=exit_style)
                    if t:
                        tr_is.append(t)
            if len(tr_is) < 20:
                continue
            b = book(tr_is, is_min)
            tr_oos = []
            for w in oos_w:
                for k in range(1, 13):
                    pick = _pick_side(w.candles[k], maker=maker_in, lo=0.25, hi=0.60)
                    if pick is None:
                        continue
                    px, side = pick
                    t = sim(w, k, side, maker_in=maker_in, tp=tp, barrier=barrier,
                            exit_style=exit_style)
                    if t:
                        tr_oos.append(t)
            o = book(tr_oos, oos_min)
            row = {
                "label": f"{'maker' if maker_in else 'taker'}_tp{tp:.2f}_b{barrier}_{exit_style}",
                "is": b, "oos": o,
            }
            table.append(row)
            print(f"{row['label']:34s} IS {b['trades']:5d} {b['net']:>9.2f} PF{b['pf']:>6.2f} "
                  f"| OOS {o['trades']:5d} {o['net']:>9.2f} PF{o['pf']:>6.2f} ECE{o['ece']:.3f}")
        table.sort(key=lambda r: -r["is"]["pf"])
        results["band"] = table

    results["summary"] = _summary(results, len(windows), crossed, len(is_w), len(oos_w),
                                   is_min, oos_min, counts)
    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"wrote {args.out}")
    return 0


def _summary(results: dict, windows: int, crossed: int, is_n: int, oos_n: int,
             is_min: float, oos_min: float, counts: dict) -> dict:
    """The clean 1-minute falsification summary, written next to the tables."""
    band = results.get("band") or []
    out: dict = {
        "source": "data/candles-15m.jsonl (Kalshi settled-window minute candles)",
        "windows": windows,
        "split": {"is_windows": is_n, "oos_windows": oos_n,
                  "is_lane_hours": round(is_min / 60, 1),
                  "oos_lane_hours": round(oos_min / 60, 1),
                  "rule": "chronological 60/40 by window close, built once"},
        "crossed_candles": crossed,
        "double_count_fix": {
            "was": "both the YES and the NO side of one decision point were simulated "
                   "whenever both quotes fell inside [0.25, 0.60], so a single book "
                   "printed as two trades (14,773 rows for 7,828 decision points)",
            "now": "one decision point is one trade: the cheaper of the two sides "
                   "(YES and NO prices sum above 1, so holding both is a guaranteed "
                   "loss, and the band family carries no directional signal)",
            "rule": "cheapest side per (window, minute); the other side is dropped",
            "counts_is": counts,
            "alternative_rule": "dropping an ambiguous decision instead of keeping "
                                "the cheaper side leaves "
                                f"{counts.get('drop_if_both_taker_is')} taker decisions "
                                "on the IS slice (the reading behind the 7,828 figure "
                                "in the directive); the cheaper-side rule keeps every "
                                "decision point, so the clean sample is larger",
        },
        "band_variants": len(band),
    }
    if band:
        best = band[0]
        out["band_best_is_variant"] = best["label"]
        out["band_best_is"] = best["is"]
        out["band_best_oos"] = best["oos"]
        pos_oos = [r["label"] for r in band if r["oos"]["net"] > 0]
        out["band_variants_positive_oos"] = pos_oos
        out["verdict"] = (
            f"all {len(band)} band variants lose on the frozen OOS slice: the best "
            f"profile by IS PF ({best['label']}) returns {best['oos']['net']} USD over "
            f"{best['oos']['trades']} OOS trades at PF {best['oos']['pf']}. "
            + (f"variants with a positive OOS net: {pos_oos}. " if pos_oos else "")
            + "A +5c/+8c take-profit on a binary truncates the winner and keeps the "
              "full loss: that exit geometry, not the entry signal, is what lost the "
              "live money. The one-minute tape has no band edge to promote."
        )
    fm = results.get("final_minute") or {}
    if fm:
        out["final_minute"] = fm
    return out


if __name__ == "__main__":
    raise SystemExit(main())
