# Strategy factory (owner directive 2026-11-18 12:44, corrected 12:45)

Source of truth for this build. Isolation holds: this tree only reads the evolve cache and pinned tapes, never touches `/home/cayleb/open-kal`, `.env` or `keys/`. Nothing here arms, orders or sends: the promotion ladder's live steps are the owner's to approve and run on Session 1's engine.

## Owner's design (what was asked)
1. **Generator: pure code, no LLM.** A strategy is a rule tree of indicators (the Decision Agent list) with thresholds, side, entry window, size and exit parameters. Random + genetic search (hand-rolled numpy GA) and Optuna tuning. The only learned part is the local LightGBM decision model, used as one indicator/gate. Every indicator is precomputed once per lane into a feature matrix (parquet); each candidate is one vectorised numba pass. Target >= 10,000 candidates/hour.
2. **Simulator:** polars + numba over every recorded lane's book and trades; 5 s latency; IOC at seen depth; the real fee formula; queue position for maker orders.
3. **Judge:** walk-forward. Promote only if OOS EV > 0 with lower CI > 0, >= 200 trades, max DD < 20%, and it survives 2x latency + 1 cent slippage. Log every candidate to a leaderboard.
4. **Evolve:** keep the top 10%, mutate/crossover, feed failures and their reasons back, run 24/7.
5. **Promotion ladder (automatic; owner approves arming):** leaderboard -> live paper on Session 1's engine for 50 trades -> 1-contract canary for 20 trades -> size x2 per step while live P&L stays positive, up to quarter-Kelly.
6. **Data:** keep recording every lane's book and trades across ALL Kalshi series plus public spot feeds; build any missing collector.
7. **Report every 2 hours:** candidates scored, top 5 by OOS EV with CIs, anything promoted.

## Where reality bends the design (decided, disclosed)
- **Data is 2.7 days, not weeks** (238 settled book-backed windows, 2026-11-15 17:45 to 11-18 11:00, 46 of them with touch sizes). Walk-forward therefore uses 6-hour blocks until 10 days exist; 200 OOS trades is reachable only by rules that fire often. Statistical power is the binding constraint, not compute.
- **10,000 candidates/hour makes a per-candidate 95% CI meaningless**: about 250 would clear "lower CI > 0" by luck every hour. The judge keeps the owner's bars as necessary conditions and adds a multiple-testing correction (lower bound at alpha / number of candidates scored on the same OOS blocks). The leaderboard shows both bounds.
- **Evolving on OOS results leaks**: a candidate is selected on data before its epoch and judged only on blocks after; parents for epoch e are chosen from data before epoch e's test blocks, never from that epoch's OOS.
- **The indicator list "Session 1 has" is not on this machine** (not in any repo file, note or transcript I can read). The registry in `factory/indicators.py` is PROVISIONAL (the engine's own features: z, flow, revision, shift, spread, mid, momentum, vol, time left, touch imbalance, cross-lane returns). It is versioned; every candidate records the registry version; swapping in the real list is a registry change, not a rewrite.
- **Maker orders need the queue model in numba**: v1 of the simulator is taker-only (entry, take-profit, stop, time barrier, settlement); the maker queue port (sim_core's `queue_fill`) is milestone 2. Cross-market arb (ladders, S&P/NDX, weather, econ) and non-crypto series need collectors that do not exist yet: milestone 3.
- Candidates keep the standing entry cost ceiling of $0.62 (`readiness.MAX_NAKED_COST`).

## Milestones
M1 pre-registered judge + feature matrix + numba taker simulator (parity-tested against sim_core) + leaderboard + GA/Optuna loop, first leaderboard.
M2 maker queue in numba; LightGBM gate (needs the Decision Agent definition); 2-hour reporter.
M3 collectors for other Kalshi series and spot feeds; cross-market families.
M4 paper-ladder export for Session 1 (specs only; arming is the owner's).
