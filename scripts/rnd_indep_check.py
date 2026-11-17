#!/usr/bin/env python3
"""Independent re-derivation of sampled IS trades from the raw npz tape, without sim_core.

For each trade in data/indep-check-pick.json (13 rows sampled at random, seed 7, from
the Track B and Track A champion IS trades) this recomputes, from the book and print
arrays directly: the book row at trigger + 1s, the maker entry price (YES: yes_bid,
NO: 1 - yes_ask), whether >= 50 contracts printed through it within the patience
window, the take-profit fill (prints after fill + 1s), settlement from markets.json,
and the PnL. It shares no code with openkal/sim_core.py.

    flock data/.heavy.lock .venv/bin/python -I scripts/rnd_indep_check.py data/indep-check-pick.json
"""
import json, sys
from decimal import Decimal, ROUND_CEILING
import numpy as np
C = "/home/cayleb/open-kal-rnd/data/frozen-cache"
meta = json.load(open(f"{C}/markets.json"))
def fee(p, q):
    raw = Decimal("0.07") * q * p * (1 - p)
    return (raw / Decimal("0.01")).to_integral_value(rounding=ROUND_CEILING) * Decimal("0.01")
def check(lane, rows):
    d = np.load(f"{C}/book_{lane}.npz")
    names = [str(t) for t in d["tickers"]]
    ts, bid, ask, tid = d["ts"], d["bid"].astype(float), d["ask"].astype(float), d["tid"]
    tts, tpx, tsz, tty, ttid = d["trade_ts"], d["trade_px"].astype(float), d["trade_sz"].astype(float), d["trade_ty"], d["trade_tid"]
    for r in rows:
        i = names.index(r["ticker"]); side = r["side"]
        bm = tid == i; bts, bb, ba = ts[bm], bid[bm], ask[bm]
        tm = ttid == i; pts, ppx, psz, pty = tts[tm], tpx[tm], tsz[tm], tty[tm]
        o = float(r["trigger_ts"]) + 1.0
        j = np.searchsorted(bts, o, "right") - 1
        b, a = round(float(bb[j]), 3), round(float(ba[j]), 3)
        cost = Decimal(str(b)) if side == "YES" else Decimal(1) - Decimal(str(a))
        # maker buy: YES bid hit by YES sellers at <= b; NO bid (= YES offer at a) lifted by YES buyers at >= a
        patience = float(r.get("patience_s") or 10.0)
        w = (pts > o) & (pts <= o + patience)
        hit = w & ((pty == 0) & (ppx <= b + 1e-6) if side == "YES" else (pty == 1) & (ppx >= a - 1e-6))
        cum = np.cumsum(np.where(hit, psz, 0)); filled = cum[-1] >= 50 if len(cum) else False
        fill_t = float(pts[np.argmax(cum >= 50)]) if filled else None
        y = int(meta[r["ticker"]]["y"]); won = (y == 1) == (side == "YES")
        qty = int(r["qty"]); reason = r["reason"]
        if reason == "settle":
            pnl = ((Decimal(1) if won else Decimal(0)) - cost) * qty
        elif reason == "tp":
            tp = cost + Decimal(r["exit_value"]) - cost
            lim = float(tp) if side == "YES" else 1 - float(tp)
            w2 = (pts > fill_t + 1.0)
            hit2 = w2 & ((pty == 1) & (ppx >= lim - 1e-6) if side == "YES" else (pty == 0) & (ppx <= lim + 1e-6))
            c2 = np.cumsum(np.where(hit2, psz, 0)); ok2 = len(c2) and c2[-1] >= 50
            pnl = (Decimal(r["exit_value"]) - cost) * qty if ok2 else None
        else:
            pnl = None
        print(f"{lane} {r['ticker']} {side} {reason}: entry sim={r['entry']} indep={cost} | filled indep={filled} | pnl sim={r['pnl']} indep={pnl} | won={won}")
want = json.loads(open(sys.argv[1]).read())
for lane, rows in want.items():
    check(lane, rows)
