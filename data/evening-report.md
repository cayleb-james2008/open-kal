# Open-Kal 24/7 report

Updated 2026-11-17T15:46:31-05:00 by `scripts/daemon_247.py`.

## State

- Decision: **promoted**
- Engine: mode=live live_armed=True strategy=F2_taker_1s
- Equity: 164.43 (start 163.9971, peak 164.43, realized 0.43)
- Venue balance: 164.4269
- Trailing floor: 131.54 (peak 164.4269)
- Reconcile delta: 0.00 (cap $0.05)
- Open lanes: 0

## Forward OOS

- Windows scored: 16 (events 22)
- Post-decide trades: 20 (20W/0L)
- Net: $33.13  PF: 99.0  payoff ECE: 0.057
- Gate: PASS 

## Recent 15-minute checks

| Time | Armed | Equity | Balance | Delta | Floor |
|---|---|---|---|---|---|
| 2026-11-17T14:46:17-05:00 | True | 164.00 | 163.9971 | 0.00 | 131.20 |
| 2026-11-17T15:01:20-05:00 | True | 164.00 | 163.9971 | 0.00 | 131.20 |
| 2026-11-17T15:16:25-05:00 | True | 164.43 | 164.4269 | 0.00 | 131.54 |
| 2026-11-17T15:31:27-05:00 | True | 164.43 | 164.4269 | 0.00 | 131.54 |
| 2026-11-17T15:46:31-05:00 | True | 164.43 | 164.4269 | 0.00 | 131.54 |

## Afternoon audit · 15:08–18:00 EST (live, updates each window)

- Baseline equity: $164.00 (15:01 check; audit starts 15:08)
- Equity at 2026-11-17T15:46:31-05:00: $164.43 → P&L $+0.43
- Account realized: $0.43 · equity per lane (equity/4): $41.11
- Fills since 15:08: 21 (1 entries, 1 exits, exit P&L $+0.43)
- Venue balance: $164.4269 · floor: $131.54 · reconcile: $0.00 · open lanes: 0

## Capacity decision · 2 vs 4 crypto slots

- OOS: 36 trades, 34W/2L (94.4%), +$42.65. Forward: 20/20, +$33.13.
- Peak concurrent: 2 in-sample / 1 out-of-sample; skipped fills at cap 2: 0 ($0.00).
- MC observed concurrency: 20k paths, 250 windows; K=2 breach 0.000% vs K=4 0.000%; median P&L $1,200 vs $1,200.
- Edge erosion (simultaneous, breach K=2 vs K=4): 90% 0.020% vs 0.000%; 80% 1.540% vs 0.940%; 70% 62.340% vs 60.380%.
- **Recommendation: keep `max_concurrent_crypto=2`.** A fourth slot adds no filled trades on the observed tape; revisit only if forward peak concurrency exceeds 2 with skipped fills, or the 94.4% OOS win rate holds through a full 30-day forward sample. The MC does not justify 4: at 94.4% both caps never touch the $131.20 floor, and below 80% both breach; 4 is marginally safer only because more positive-EV trades compound the buffer faster, which is a counterfactual the tape does not support.

## Events

- 2026-11-17T14:41:13-05:00 **live armed**: 
- 2026-11-17T14:41:13-05:00 **report**: promotion chain completed; live compounding under supervision
- 2026-11-17T14:46:17-05:00 **report**: 15-minute window check
- 2026-11-17T15:01:20-05:00 **report**: 15-minute window check
- 2026-11-17T15:16:25-05:00 **report**: 15-minute window check
- 2026-11-17T15:31:27-05:00 **report**: 15-minute window check
- 2026-11-17T15:46:31-05:00 **report**: 15-minute window check
