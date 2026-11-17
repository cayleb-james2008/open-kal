# Open-Kal 24/7 report

Updated 2026-11-17T15:16:25-05:00 by `scripts/daemon_247.py`.

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

## Events

- 2026-11-17T14:41:13-05:00 **live armed**: 
- 2026-11-17T14:41:13-05:00 **report**: promotion chain completed; live compounding under supervision
- 2026-11-17T14:46:17-05:00 **report**: 15-minute window check
- 2026-11-17T15:01:20-05:00 **report**: 15-minute window check
- 2026-11-17T15:16:25-05:00 **report**: 15-minute window check
