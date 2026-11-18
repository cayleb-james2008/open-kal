# Open-Kal 24/7 report

Updated 2026-11-18T01:46:29-05:00 by `scripts/daemon_247.py`.

## State

- Decision: **waiting**
- Engine: mode=None live_armed=None strategy=None
- Equity: None (start None, peak None, realized None)
- Venue balance: 121.4563
- Trailing floor: 131.54 (peak 164.4269)
- Reconcile delta: 0.00 (cap $0.05)
- Open lanes: 0

## Forward OOS

- Windows scored: 30 (events 45)
- Post-decide trades: 34 (22W/12L)
- Net: $-22.61  PF: 0.6422  payoff ECE: 0.382
- Gate: FAIL forward net $-22.61 <= 0 forward PF 0.642 < 1.15 payoff ECE 0.3820 > 0.08 1 profitable lane(s) < 2

## Recent 15-minute checks

| Time | Armed | Equity | Balance | Delta | Floor |
|---|---|---|---|---|---|
| 2026-11-17T20:31:17-05:00 | False | 121.46 | 121.4563 | 0.00 | 131.54 |
| 2026-11-17T20:46:20-05:00 | False | 121.46 | 121.4563 | 0.00 | 131.54 |
| 2026-11-17T21:00:23-05:00 | False | 121.46 | 121.4563 | 0.00 | 131.54 |
| 2026-11-17T21:46:06-05:00 | False | 121.46 | 121.4563 | 0.00 | 131.54 |
| 2026-11-17T22:00:10-05:00 | False | 121.46 | 121.4563 | 0.00 | 131.54 |
| 2026-11-17T22:46:20-05:00 | False | 121.46 | 121.4563 | 0.00 | 131.54 |
| 2026-11-17T23:00:24-05:00 | False | 121.46 | 121.4563 | 0.00 | 131.54 |
| 2026-11-17T23:46:21-05:00 | False | 121.46 | 121.4563 | 0.00 | 131.54 |
| 2026-11-18T00:00:24-05:00 | False | 121.46 | 121.4563 | 0.00 | 131.54 |
| 2026-11-18T00:46:33-05:00 | False | 121.46 | 121.4563 | 0.00 | 131.54 |
| 2026-11-18T01:00:36-05:00 | False | 121.46 | 121.4563 | 0.00 | 131.54 |
| 2026-11-18T01:46:29-05:00 | False | None | 121.4563 | 0.00 | 131.54 |

## Afternoon audit · 15:08–18:00 EST (final 18:00 snapshot)

- Baseline equity: $164.00 (15:01 check; audit starts 15:08)
- Equity at 2026-11-17T18:00:19-05:00: $121.46 → P&L $-42.54
- Account realized: $-42.54 · equity per lane (equity/4): $30.36
- Fills since 15:08: 0 (0 entries, 0 exits, exit P&L $+0.00)
- Venue balance: $121.4563 · floor: $131.54 · reconcile: $0.00 · open lanes: 0

## Capacity decision · 2 vs 4 crypto slots

- OOS: 36 trades, 34W/2L (94.4%), +$42.65. Forward: 20/20, +$33.13.
- Peak concurrent: 2 in-sample / 1 out-of-sample; skipped fills at cap 2: 0 ($0.00).
- MC observed concurrency: 20k paths, 250 windows; K=2 breach 0.000% vs K=4 0.000%; median P&L $1,200 vs $1,200.
- Edge erosion (simultaneous, breach K=2 vs K=4): 90% 0.020% vs 0.000%; 80% 1.540% vs 0.940%; 70% 62.340% vs 60.380%.
- **Recommendation: keep `max_concurrent_crypto=2`.** A fourth slot adds no filled trades on the observed tape; revisit only if forward peak concurrency exceeds 2 with skipped fills, or the 94.4% OOS win rate holds through a full 30-day forward sample. The MC does not justify 4: at 94.4% both caps never touch the $131.20 floor, and below 80% both breach; 4 is marginally safer only because more positive-EV trades compound the buffer faster, which is a counterfactual the tape does not support.

## Events

- 2026-11-17T23:00:24-05:00 **report**: unarmed 15-minute reconcile
- 2026-11-17T23:36:18-05:00 **gate refused**: ['forward net $-17.52 <= 0', 'forward PF 0.690 < 1.15', 'payoff ECE 0.3752 > 0.08', '1 profitable lane(s) < 2']
- 2026-11-17T23:36:18-05:00 **report**: gate refused at the hourly check
- 2026-11-17T23:46:21-05:00 **report**: unarmed 15-minute reconcile
- 2026-11-18T00:00:24-05:00 **report**: unarmed 15-minute reconcile
- 2026-11-18T00:40:29-05:00 **gate refused**: ['forward net $-22.61 <= 0', 'forward PF 0.642 < 1.15', 'payoff ECE 0.3820 > 0.08', '1 profitable lane(s) < 2']
- 2026-11-18T00:40:29-05:00 **report**: gate refused at the hourly check
- 2026-11-18T00:46:33-05:00 **report**: unarmed 15-minute reconcile
- 2026-11-18T01:00:36-05:00 **report**: unarmed 15-minute reconcile
- 2026-11-18T01:37:25-05:00 **gate refused**: ['forward net $-22.61 <= 0', 'forward PF 0.642 < 1.15', 'payoff ECE 0.3820 > 0.08', '1 profitable lane(s) < 2']
- 2026-11-18T01:37:25-05:00 **report**: gate refused at the hourly check
- 2026-11-18T01:46:29-05:00 **report**: unarmed 15-minute reconcile
