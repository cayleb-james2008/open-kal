# RND report — rnd/alpha-evolve

Worktree `/home/cayleb/open-kal-rnd`, branch `rnd/alpha-evolve`, based on origin/master `d9fd4ec`. Data: the frozen, hard-linked snapshot `data/frozen-cache/` (books from the 14:07 vintage; `SHA256SUMS.txt`), plus a copy of the live `ticks.npz` (16:20 vintage, hash appended). Nothing under `/home/cayleb/open-kal` was written, signalled or restarted. No `.env`, `keys/`, `canary.py --live` or `restart_flat.sh` was touched, and the prod DB was never opened.

## Charter 2 — zero taker fees, hold to settlement (latest)

**Neither architecture passes. Nothing should be promoted or traded.** Each family champion was scored once on OOS (`data/oos-ledger.jsonl`), and both champions met the IS bar first.

| Family | IS champion | IS n / W-L / net / PF | OOS n | OOS W/L | OOS net | OOS PF | payoff-ECE | OOS by lane | net − best lane | Gate |
|---|---|---|---|---|---|---|---|---|---|---|
| A `passive_mm_fade` | `flow_thr2_vq0-1_tp0.06_scratch_hold180`, eligible | 43 / 36-7 / +69.00 / 2.55 | 46 | 33/13 | −57.50 | 0.63 | 0.149 | BTC 40 trades −56.50; ETH 6, −1.00 | −56.50 | **fail** |
| B `student_t_settle_v2` | `h0.03_s0.02_v0.5-1.0_p10`, eligible | 216 / 87-129 / +556.50 / 1.27 | 126 | 43/83 | −192.50 | 0.87 | 0.108 | BTC −110.50, ETH +12.00, SOL −18.50, XRP −75.50 | −204.50 | **fail** |

Every leg in both tracks is maker or settlement: $0 entry and exit fees in every OOS trade, 0 crossed fills, max entry cost 0.55 (A) and 0.58 (B).

### Owner rulings applied (commit `85cc73e`)
- **$0.62 on every buy leg.** A lock leg priced above $0.62 is never placed (test `LockLegCap`). No earlier result changes. Checked two ways:
  - every Track 2 config's own count of lock legs above $0.62 (over all its fills, not only the champion's) is 0; the cheapest lock leg was $0.34;
  - a rerun of Track 2 IS on the capped core reproduces all 16 IS summaries exactly.
  - Tracks 3 and 4 have no lock legs.
- **50-contract print-through floor** kept, unchanged.
- New `$0`-fee **maker scratch** exit: `barrier_exit="scratch"` at `fill + hold_s` or `close − scratch_by_s`. It is a post-only sell at max(entry − scratch_c, side bid + 1c), arriving 1s after the barrier and filled on the 50-print rule; unfilled, the position settles (5 tests).
- `tests/test_sim_core.py` is now 33 tests. Before the change, 5 of the 7 new ones failed.

### Track A — passive book-imbalance fade (`scripts/passive_mm_fade.py`)
- **Proxy, stated plainly.** The book cache holds top-of-book *prices* only, no sizes, so a depth OBI or microprice cannot be computed without the prod DB, which is off-limits. z_obi is the pool's 10s signed print-flow imbalance (`flow`) or 5s touch-revision imbalance (`rev`), z-scored per lane on IS rows only.
- **The fade:** z_obi ≥ +thr rests a NO bid (at 1 − yes_ask); z_obi ≤ −thr rests a YES bid. Maker entry, 10s patience, cost ≤ $0.55; maker TP +4c/+6c arriving 1s after the fill; scratch (after 180s, or 120s before close) or settle.
- **Grid:** 48 configs (2 sources × 2 thresholds × 2 vol bands × 2 TPs × 3 exits). Two were IS-eligible.
- **The champion has no volatility filter.** Its 0–1 band admits every row; it is the grid's unfiltered control.
  - None of the 24 bounded-band (0.2–0.8) configs was IS-eligible.
  - The best bounded config with ≥ 30 IS trades made +$15.00 at PF 1.02 (257 trades). The best bounded config overall made +$66.00 on 24 trades.
  - So the charter's bounded-volatility version of Track A has no candidate even before OOS.

| Source, threshold | IS trades (range over 12 configs) | IS net range | Lanes traded |
|---|---|---|---|
| flow, 1.5 | 257–375 | −$431.25 to +$15.00 | all 4 |
| flow, 2.0 | 24–43 | −$4.50 to +$69.00 | BTC, ETH only |
| rev, 1.5 | 421–569 | −$581.50 to −$63.00 | all 4 |
| rev, 2.0 | 373–549 | −$787.00 to −$214.50 | all 4 |

- **Structural limit:** `flow` is bounded in [−1, 1], so |z| ≥ 2 is reachable only where a lane's IS std is small. The reachable maximum is BTC 2.71, ETH 1.86 (−2.00 at the bound), SOL 1.61, XRP 1.69. So every flow-2.0 config trades almost only BTC, and the two-lane gate was out of reach for them by construction.
- **The pattern in every config:** take-profits fill often and pay +$2–3, and the few positions that never revert lose the whole contract at settlement.
  - OOS champion: 33 TPs made +$99.00, 4 scratches made $0.00, and 9 settled positions lost −$156.50.
  - The +4/+6c maker exit cannot pay for the tail.
- `rev` fades lose heavily everywhere: the touch-revision skew keeps going, it does not revert.

### Track B — multi-lane Student-t settlement (`scripts/student_t_settle_v2.py`)
- Same IS-only Student-t fit as Track 3 (ν 6–10 per lane), causal spot, `secs − 40` settlement variance.
- Maker entry at the side's bid, 1s after the trigger, capped at p − hurdle, cost $0.22–0.58. Filled only on the 50-print rule, held to settlement, no exit leg.
- **Grid:** 48 configs (hurdle 3/5/7/9c × spread ≤ 2/4c × vol band all/low/high × patience 10/30s). Vol quantiles come from IS samples. 9 of 48 were IS-eligible, and the champion was the best of them.
- On IS it looked broad: all 4 lanes profitable (BTC +316.00, ETH +61.50, SOL +104.50, XRP +74.50), +$482.00 without XRP, fill rate 0.44.
- **OOS it reversed.** Win rate went from 0.40 to 0.34, 3 of 4 lanes lost, net without XRP was −$117.00, and payoff-ECE was 0.108.
  - It cleared the ≥ 25-trade bar (126 trades).
  - It was not XRP-dependent in the sense the charter meant: it lost money without XRP too (−$117.00).
  - The gate's "not carried by one lane" check still fails, because net is negative even before a lane is removed.
  - It lost money everywhere except ETH (+$12.00).
- A reading, not tested here: a maker bid fills when sellers push through it, which may select the moments when the market knows more than the model.

### Outside checks for Charter 2
- **13 sampled IS trades re-derived from the raw npz tape, without sim_core:** 9 from Track B across SOL, XRP and ETH, 4 from Track A on BTC.
  - Arrival quote, entry price, 50-print fill, TP fill and settlement side were each recomputed.
  - All 13 matched sim_core exactly in entry, fill and PnL.
  - Script `scripts/rnd_indep_check.py`, sample `data/indep-check-pick.json`, output `data/indep-check.txt`.
- The IS champions and both scripts were committed (`88a4d6a`) before the OOS look; OOS was committed in `d8e92f8`.
- A fresh Haiku review re-derived every Charter 2 number from the data files and confirmed the scratch and cap logic, the timing and IS-only behaviour. It found one false statement, now corrected: the XRP wording above. Its other findings:
  - The champion has no vol filter; now stated above.
  - The lock-cap claim was unproven; now checked above.
  - The trade check was unreproducible; now committed.
  - Two `sim_core` edge cases with no effect on any reported number, left as they are:
    - a confirmed stop that finds no valid quote before the end falls through to the barrier exit or settlement;
    - a scratch that arrives into a crossed book is not placed, and the position settles.
  - Track B is maker-only by construction: `entry_style=MAKER` and no exit legs. It has no runtime guard against taker legs; every OOS trade shows $0.00 fees.

---

## Charter 1 bottom line

**No candidate passes the promotion gate. Nothing here should be promoted or traded.**

- Eight families each have one champion, and all eight fail on OOS.
- Only one of the eight champions (`gauss_fv_control`) met the IS eligibility bar (≥ 30 IS trades, PF ≥ 1.20). The other seven are best-IS-net fallbacks, flagged ineligible; under a strict reading of the charter they would not be OOS-scored at all. Their OOS rows are diagnostics, not candidates.
- The best OOS net is Track 3's fat-tail sniper: +$94.20 on 18 trades. Removing its best lane (XRP) leaves −$2.65. Its IS sample was too small to fit a payoff model.
- Two defects were found and fixed in the research pipeline before any OOS look:
  - **pool look-ahead:** spot was read up to 1.5s after the trigger;
  - **float32 boundary compares:** cost-$0.62 trades were silently dropped.
- Both defects also live in the legacy scripts behind earlier results. The legacy scripts were not changed.
- An independent review then found a third defect, in sim_core itself: take-profit and lock legs were live the instant the entry filled, with no 1s placement lag. It was fixed in `104b04e`. Every IS grid was rerun, and every family was re-scored on OOS with the re-score and its reason recorded in the ledger. The verdict did not change.

## Commits on this branch

| Commit | What | Cherry-pick to master? |
|---|---|---|
| `8a5c9b6` | tests: daemon git sync, drawdown floor, 18:00 freeze, mc floor model | **yes** (tests only) |
| `9e39dee` | earlier research: F2 peg/TP/ETH variants, LayaGate ablation | no (research) |
| `0a26266` | `openkal/sim_core.py` + `tests/test_sim_core.py` | **yes**: new module and tests, nothing existing changed |
| `e8f34f4` | `scripts/rnd_common.py` (split, gate, sealed OOS ledger), `scripts/sim_core_parity.py`, parity JSON | no (research; adds a `data/` file) |
| `5bf301b` | `scripts/rnd_pool_slim.py` | no; superseded and deleted in `b27c27f` |
| `b27c27f` | Tracks 2–4 scripts, causal pool builder, frozen IS results | no (research) |
| `a27edf7` | OOS ledger for Tracks 2 and 4 | no (research) |
| `37dde5f` | first version of this report | no |
| `104b04e` | sim_core: take-profit and lock legs arrive 1s after the entry fill (+2 tests) | **yes**, together with `0a26266` |
| `9e19bcc` | rnd_common: the ledger writes a "started" line before scoring; re-scores carry a reason | no (research) |

| `90e326b` | Charter 1 rerun after the exit-leg fix; report revision | no |
| `85cc73e` | sim_core: $0.62 cap on lock legs; $0-fee maker scratch exit (+7 tests) | **yes**, with `0a26266` and `104b04e` |
| `88a4d6a` | Charter 2 scripts (`passive_mm_fade`, `student_t_settle_v2`), IS champions frozen | no (research) |
| `d8e92f8` | Charter 2 OOS ledger entries | no (research) |

This revision of the report is in the commit after `d8e92f8`.

```bash
git -C /home/cayleb/open-kal cherry-pick 8a5c9b6 0a26266 104b04e 85cc73e
```

## The promotion gate (`scripts/rnd_common.py:promotion_gate`, committed in `e8f34f4` before any track ran)

- OOS net > 0
- OOS PF ≥ 1.25
- OOS payoff-ECE ≤ 0.08, from a ridge-logistic P(pnl > 0) fitted on IS trades only. With fewer than 20 IS trades there is no model, and the check fails.
- ≥ 25 OOS trades
- ≥ 2 profitable lanes
- net still > 0 after removing the best lane, and after removing the best window (windows grouped by close time across lanes)
- 0 crossed fills, max entry cost ≤ $0.62

Split: one global chronological cut at 60% of the distinct window closes (`1794881700` = 2026-11-17 02:15 UTC). All four lanes' own cuts coincide with it.

IS selection rule, for every track: eligible means IS trades ≥ 30, IS PF ≥ 1.20 and 0 crossed fills; the champion is the eligible config with the highest IS net. If no config is eligible, the highest-IS-net config is taken and flagged ineligible. It is still scored once, so every family has an OOS row.

OOS is sealed. `score_oos_once` refuses without `RND_OOS_UNSEAL=lead` and refuses a second scoring of a family unless it is a recorded re-score. Since `9e19bcc` it also writes a "started" line before the scorer runs, so a crash mid-score still counts as a look. The Haiku workers built the scripts in IS-only mode; only the lead ran OOS.

Every champion look is in `data/oos-ledger.jsonl`. Not every OOS computation is: `scripts/sim_core_parity.py` computes the existing F2 rule's OOS row (below) without the ledger. F2 is the rule already live, not a candidate chosen here, and its OOS had already been computed in the earlier study (`9e39dee`). The Track 2 grid was written before that parity result existed.

## OOS ledger: every champion

The ledger holds two rounds:

1. **First round:** 8 entries, one per family, scored with the pre-fix core.
2. **Re-score:** 8 "started" lines and 8 "scored" entries with `rescore: true` and the reason "sim_core 104b04e: TP/lock legs now arrive 1s after the entry fill".
   - The re-score uses the fixed core and the IS champions from the IS rerun.
   - Tracks 2 and 3 kept the same champions.
   - In Track 4 the corrected IS metrics changed the search path, so three of four champions are new genomes. Their OOS looks are therefore first looks at those genomes but second looks at the families.

The table shows the re-score; the last column is the first-round OOS net.

| Track / family | Champion (IS) | IS n / net / PF | OOS n | OOS W/L | OOS net | OOS PF | payoff-ECE | lanes > 0 | net − best lane | net − best window | entry / exit fees | Gate | first-round OOS net |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 `synthetic_lock` | `maker_locknone_tpnone` (settle-only control), ineligible | 29 / +384.00 / 3.72 | 14 | 8/6 | +4.00 | 1.02 | 0.323 | 1 | −12.00 | −40.50 | 0.00 / 0.00 | **fail** | +4.00 |
| 2 `synthetic_lock_locked_only` | `taker_lock0.08_tpnone`, ineligible | 28 / +87.82 / ∞ | 14 | 11/3 | −36.08 | 0.59 | 0.210 | 1 | −36.26 | −56.22 | 12.08 / 0.00 | **fail** | −53.08 |
| 3 `kurtosis_fv_snipe` | `fat_h0.09_settle`, ineligible | 19 / +22.25 / 1.12 | 18 | 7/11 | +94.20 | 1.64 | none (19 IS trades < 20) | 2 | −2.65 | +56.32 | 12.30 / 0.00 | **fail** | +94.20 |
| 3 `gauss_fv_control` | `gauss_h0.09_settle`, **eligible** | 33 / +99.08 / 1.32 | 35 | 10/25 | +13.29 | 1.04 | 0.213 | 1 | −67.25 | −59.88 | 23.71 / 0.00 | **fail** | +13.29 |
| 4 `depletion` | `depletion:4.11`, ineligible | 11 / +40.64 / ∞ | 3 | 1/2 | −14.29 | 0.13 | none | 1 | −16.45 | −16.45 | 2.55 / 1.74 | **fail** | −7.25 (`depletion:3.2`) |
| 4 `divergence` | `divergence:4.8`, ineligible | 4 / +40.62 / ∞ | 2 | 1/1 | −6.08 | 0.16 | none | 1 | −7.24 | −7.24 | 1.71 / 0.87 | **fail** | −23.00 (`divergence:3.2`) |
| 4 `spread_regime` | `spread_regime:4.7`, ineligible | 13 / +37.45 / 8.15 | 3 | 1/2 | −14.29 | 0.13 | none | 1 | −16.45 | −16.45 | 2.55 / 1.74 | **fail** | −2.38 (`spread_regime:3.3`) |
| 4 `tod_vol` | `tod_vol:4.2`, ineligible | 6 / +9.22 / 1.86 | 0 | — | 0.00 | — | none | 0 | — | — | — | **fail** | 0.00 |

- PF ∞ means no IS losers.
- Every OOS row has 0 crossed fills and max entry cost ≤ $0.62. Per-lane OOS tables are in the ledger.
- `depletion:4.11` and `spread_regime:4.7` took the same 3 OOS trades.

**Multiple testing.** Charter 1 had 16 champion scorings across 8 families, plus the F2 parity computation; Charter 2 adds 2 more (10 families, 18 scorings in all). No row passes, so no multiplicity correction can change the verdict. Any future pass found by this same process would need confirmation on a fresh forward tape, not on this OOS slice.

---

## Track 1 — canonical simulator `openkal/sim_core.py`

New standalone module; no existing file changed. Callers pass `side`, `style` and the YES quote, and the core derives every price, so the earlier "limit in YES terms" bug class cannot be expressed through it.

| Rule | How it is enforced | Test |
|---|---|---|
| YES taker / maker entry = `yes_ask` / `yes_bid` | `entry_price` | `test_entry_prices` |
| NO taker / maker entry = `1 − yes_bid` / `1 − yes_ask` | `entry_price` | `test_entry_prices`, `test_no_taker_entry_is_never_the_yes_bid`, `test_no_taker_trade_charges_one_minus_bid` |
| YES taker / maker exit = `yes_bid` / `yes_ask`; NO = `1 − yes_ask` / `1 − yes_bid` | `exit_price` | `test_exit_prices`, `test_no_taker_stop_sells_at_one_minus_ask` |
| Crossed book raises; the trade skips; never walks back to an older row | `check_book`, `Tape.quote` | `CrossedBooks` (3 tests) |
| Order arrives ≥ 1s after the trigger; lag < 1s raises; stop and barrier exits also arrive 1s after the decision | `run_trade` | `Arrival` (5 tests); the stop test checks `exit_ts = breach + 2s + 1s` |
| Passive fill needs ≥ 50 contracts at/through the limit, after arrival, from the aggressor side that can hit it; a floor < 50 raises | `Tape.print_fill`, `print_rule` | `PrintThrough` (6 tests) |
| Exact `Decimal` `openkal.fees.taker_fee` on taker legs, $0 on maker legs | `run_trade` | fee asserts in 5 tests |
| Cap ≤ $0.62 on the held side's cost; a ceiling > 0.62 raises; qty ≤ 50 | `run_trade` | `test_entry_above_cap_is_skipped`, `test_cap_and_print_floor_cannot_be_loosened`, `test_qty_is_capped` |
| Lock ≡ maker take-profit at the same distance; one-cancels-other | `run_trade` | `PairLock` (3 tests) |
| Take-profit and lock legs arrive 1s after the entry fill that sends them (`104b04e`) | `run_trade` (`place_ts = fill_ts + lag_s`) | `ExitLegArrival` (2 tests; both failed on the pre-fix code) |

`tests/test_sim_core.py`: 26 tests, all pass. Mutation check (on the first 24 tests): each of 8 injected bugs fails at least one test. The bugs were: NO taker entry = `yes_bid`; NO taker exit = `1 − yes_bid`; print floor 20; no arrival lag; crossed book accepted; wrong aggressor side for a NO bid; maker fee charged; stop with no exit lag. The independent reviewer reproduced five of these in memory.

### Fill-model limits (still optimistic; disclosed, not modelled)
- **No queue position.** A passive leg fills once 50 contracts print at or through its limit after it arrives. Prints *at* the limit may have filled orders queued ahead of it.
- **Taker legs fill the full quantity (≤ 50) at the touch.** The cache stores top of book only, so depth beyond the touch cannot be checked.
- **Trigger pre-filter.** `f2_structural.frozen_mask` and `alpha_evolve_v2.prep` compute the favoured side's cost from bid/ask at the trigger, to filter triggers before the cooldown dedupe. sim_core prices every order again at arrival, and enforces the band there. No fill price or PnL is computed outside sim_core.

These limits bias fills in the strategy's favour. Every verdict here is negative, so they cannot have produced a false pass. A future promotion candidate needs them modelled first.

### Outside check: F2 `taker|tp5|none` rerun through sim_core (`scripts/sim_core_parity.py`)

| | Prior corrected row (legacy simulator) | sim_core |
|---|---|---|
| IS trades / net | 58 / −$79.31 | 61 / −$74.14 (PF 0.46) |
| OOS trades / net | 34 / −$73.48 | 39 / −$87.04 (PF 0.28); −$87.52 before the exit-leg fix |

The prior row is copied from `data/evolve-f2-variants.json` (corrected convention, `taker|tp5|none`). `data/sim-core-parity.json` was regenerated with the current code.

Trigger-by-trigger diff of the 109 frozen-rule triggers (legacy `evolve_f2_variants.sim_one` vs `sim_core.run_trade`, run before the exit-leg fix):
- 92 trade in both, with the same exit reason every time. PnL differs by +$7.55 in total (−$1.50 to +$5.52 a trade), from the 1s lag sim_core puts on stop and barrier exits.
- 9 are skipped by both (cost band).
- 8 trade only in sim_core. All 8 cost exactly $0.62 (six NO at `1 − 0.38`, two YES at ask 0.62). The legacy band test computes `1 − float32(0.38) = 0.6200000048 > 0.62 + 1e-9` and drops them. The cap is inclusive (`entry_price <= $0.62`), so sim_core is right to keep them. 5 of the 8 lost.
- Arrival quotes are identical in both for all 109 triggers.

The parity script uses the legacy (nearest-spot) pools on purpose: it checks the simulator, not the signal.

## Pipeline defects found before any OOS look

1. **Spot look-ahead in the trigger pools.**
   - `bounded_sweep.obi_pool` reads spot through `alpha_search.index_at`, which returns the *nearest* index print within 1.5s on either side of a book row. So z, delta and p_up at a trigger could use a print from up to T + 1.5s, which is 0.5s after the order's own T + 1s arrival.
   - `scripts/rnd_pool_build.py` rebuilds the pools with the last print at or before the row. Flow, revision imbalance and the trailing sigma were already causal.
   - With causal spot, only 61 of 109 F2 triggers are unchanged; the rest move in time, appear or disappear. IS triggers stay at 66; OOS goes from 43 to 45.
   - Tracks 2 and 4 were rerun on IS with the causal pools. The pre-fix runs are kept as `data/*-nearestspot.json`.
   - Track 3 builds its own causal samples and was not affected.
   - Every earlier study built on `obi_pool` (including the F2 rule's own sweep) carries this look-ahead.
2. **float32 boundary compares.**
   - NumPy compares a float32 array with a Python float in float32.
   - A float32 trigger pool gave 137 F2 triggers instead of 109 (an ask of 0.60 passes `<= 0.60 + 1e-9` in float32 and fails in float64).
   - The legacy float64 band drops $0.62 entries (above).
   - The pools now keep every threshold column in float64. sim_core quantises prices to the $0.001 tick in `Decimal`.
3. **A time-bucket thinning of the pools** (one row per 0.25s) lost 4 of 9 ETH F2 triggers, because a qualifying row is often a single book update. It was not used. The pools keep every row with |z| ≥ 0.10.

## Track 2 — synthetic intra-window pair-locking (`scripts/synthetic_lock.py`)

Signals: the frozen F2 triggers, unchanged; 66 IS triggers. Entry cost ≤ $0.58, taker or maker (5s patience). No stop: anything still open settles at $0 exit fee.

**Equivalence, measured on real data.** A resting NO bid at `1 − c − L` is the same order as a resting YES offer at `c + L`. Over 132 IS trigger×style pairs per L, `lock_c = L` and `tp_c = L` gave identical PnL and exit time in every case: 0 differences, 57 filled pairs at each of L = 4, 6 and 8c. So the "lock" is a $0-fee maker take-profit at +L under another name:
- with a +6c TP, a 4c lock always fills first;
- a 6c lock is the same order as the TP;
- an 8c lock never fills before the TP.

IS grid on causal pools with the fixed core. The last column is the same config on the old nearest-spot pools with the pre-fix core.

| Config | IS n | W/L | IS net | PF | lock / tp / settled-won / settled-lost | entry fees | lock legs priced > 0.62 | net, nearest-spot, pre-fix |
|---|---|---|---|---|---|---|---|---|
| `taker_locknone_tpnone` | 28 | 23/5 | +330.82 | 3.28 | 0/0/23/5 | 24.18 | — | +280.32 |
| `taker_locknone_tp0.06` | 28 | 28/0 | +59.82 | ∞ | 0/28/0/0 | 24.18 | — | +27.82 |
| `taker_lock0.04_tpnone` | 28 | 28/0 | +31.82 | ∞ | 28/0/0/0 | 24.18 | 0 | +0.82 |
| `taker_lock0.04_tp0.06` | 28 | 28/0 | +34.82 | ∞ | 25/3/0/0 | 24.18 | 0 | +3.82 |
| `taker_lock0.06_tpnone` | 28 | 28/0 | +59.82 | ∞ | 28/0/0/0 | 24.18 | 0 | +27.82 |
| `taker_lock0.06_tp0.06` | 28 | 28/0 | +59.82 | ∞ | 0/28/0/0 | 24.18 | 0 | +27.82 |
| `taker_lock0.08_tpnone` | 28 | 28/0 | +87.82 | ∞ | 28/0/0/0 | 24.18 | 0 | +54.82 |
| `taker_lock0.08_tp0.06` | 28 | 28/0 | +59.82 | ∞ | 0/28/0/0 | 24.18 | 0 | +27.82 |
| `maker_locknone_tpnone` | 29 | 24/5 | +384.00 | 3.72 | 0/0/24/5 | 0.00 | — | +306.00 |
| `maker_locknone_tp0.06` | 29 | 28/1 | +55.00 | 2.90 | 0/28/0/1 | 0.00 | — | +26.50 |
| `maker_lock0.04_tpnone` | 29 | 27/2 | +17.50 | 1.31 | 26/0/1/2 | 0.00 | 0 | +29.50 |
| `maker_lock0.04_tp0.06` | 29 | 28/1 | +37.00 | 2.28 | 18/10/0/1 | 0.00 | 0 | +36.50 |
| `maker_lock0.06_tpnone` | 29 | 28/1 | +55.00 | 2.90 | 28/0/0/1 | 0.00 | 0 | +26.50 |
| `maker_lock0.06_tp0.06` | 29 | 28/1 | +55.00 | 2.90 | 0/28/0/1 | 0.00 | 0 | +26.50 |
| `maker_lock0.08_tpnone` | 29 | 28/1 | +83.00 | 3.86 | 28/0/0/1 | 0.00 | 0 | +54.50 |
| `maker_lock0.08_tp0.06` | 29 | 28/1 | +55.00 | 2.90 | 0/28/0/1 | 0.00 | 0 | +26.50 |

- No config reaches 30 IS trades, so both champions are flagged ineligible.
- The exit-leg fix changed only the two `maker_lock0.04` rows. A 4c lock is close enough to the maker entry that prints in the first second after the fill used to complete it.
- On IS, holding to settlement beats every lock with the same entry style by $243–367: locking caps a winner at +L while the F2 favourite usually settles at $1.
- `synthetic_lock_locked_only` was added as a second family after the first IS run, before any Track 2 OOS look, so that the lock thesis got its own OOS row.
- OOS: the settle-only control fell to +$4.00 on 14 trades (IS win 0.83 → OOS 0.57).
- OOS, unfilled-second-leg expiry loss: the 8c lock locked 10 of 14 trades (+$31.37). The 4 whose lock never filled went to settlement and lost −$67.45, for a net of −$36.08. Before the fix the split was 11 locked (+$34.51) and 3 unfilled (−$87.59), net −$53.08.
- Every entry here is ≤ $0.58, so no lock leg (`1 − entry − L`) was priced above $0.62. Whether the cap should bind a risk-reducing lock leg was never tested.

Verdict: the thesis fails. Pair-locking does not remove expiry risk when the lock does not fill, and when it does fill it is a take-profit.

## Track 3 — fat-tail fair value (`scripts/kurtosis_fv_snipe.py`)

Fit, IS only. Per lane, 15-second samples of τ-second index log returns (τ ∈ 60…780s) are scaled by trailing 150s realised vol. The fit uses index points before cut − 900s only. A scaled Student-t is fitted by grid MLE against a Gaussian. The t CDF is a numerical table (scipy is not installed); its unit check passes (CDF(0) error ≤ 5e-13; ν = 100 CDF(1.96) = 0.97361).

| Lane | ν | t scale s | Gaussian s | excess kurtosis | log-lik t vs Gaussian |
|---|---|---|---|---|---|
| BTC | 8 | 1.13 | 1.31 | 1.63 | −43,638 vs −44,130 |
| ETH | 6 | 0.78 | 0.95 | 2.58 | −34,899 vs −35,638 |
| SOL | 10 | 0.68 | 0.76 | 1.06 | −29,544 vs −29,776 |
| XRP | 10 | 1.06 | 1.19 | 1.23 | −41,285 vs −41,580 |

The tails are real: the t beats the Gaussian on every lane.
- Fair value is p_yes = 1 − F_t(log(K/S) / (s·σ·√(secs − 40)); ν). The `secs − 40` is the variance of the 60s settlement average.
- Spot is causal: the last print at or before t.
- Settlement calibration over all 29,550 in-band IS quotes: fat ECE 0.012, Gaussian 0.023.
- Entries: taker at T+1s, as an IOC capped at min(0.58, p_side − hurdle), cost ∈ [0.22, 0.58]. No stop.

| Config | IS n | IS net | PF | win | lanes > 0 | entry fees | | Config | IS n | IS net | PF | win |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fat h0.06 tp6 | 98 | −229.54 | 0.43 | 0.77 | 0 | 72.54 | | gauss h0.06 tp6 | 199 | −300.73 | 0.57 | 0.78 |
| fat h0.06 tp8 | 98 | −202.04 | 0.54 | 0.73 | 0 | 72.54 | | gauss h0.06 tp8 | 199 | −328.73 | 0.59 | 0.74 |
| fat h0.06 settle | 98 | +13.96 | 1.01 | 0.35 | 3 | 72.54 | | gauss h0.06 settle | 199 | −86.23 | 0.96 | 0.31 |
| fat h0.09 tp6 | 19 | −68.75 | 0.31 | 0.74 | 1 | 14.75 | | gauss h0.09 tp6 | 33 | −95.92 | 0.36 | 0.73 |
| fat h0.09 tp8 | 19 | −54.75 | 0.45 | 0.74 | 1 | 14.75 | | gauss h0.09 tp8 | 33 | −89.42 | 0.46 | 0.70 |
| **fat h0.09 settle** | 19 | +22.25 | 1.12 | 0.42 | 2 | 14.75 | | **gauss h0.09 settle** | 33 | +99.08 | 1.32 | 0.39 |
| fat h0.12 tp6 | 10 | −66.23 | 0.17 | 0.60 | 1 | 7.73 | | gauss h0.12 tp6 | 12 | −57.94 | 0.24 | 0.67 |
| fat h0.12 tp8 | 10 | −60.23 | 0.24 | 0.60 | 1 | 7.73 | | gauss h0.12 tp8 | 12 | −49.94 | 0.34 | 0.67 |
| fat h0.12 settle | 10 | −96.73 | 0.31 | 0.20 | 1 | 7.73 | | gauss h0.12 settle | 12 | −116.94 | 0.27 | 0.17 |

- After the exit-leg fix, only `gauss h0.06 tp6` changed (−$335.73 → −$300.73). Both champions are settle-only and did not move.
- Maker take-profits lose everywhere. The taker entry fee (about $0.75 a trade) plus a capped +6c/+8c upside does not pay for the tickets that settle at $0.
- OOS, the fat champion made +$94.20 on 18 trades. It fails three checks: under 25 trades, no payoff model (19 IS trades), and it is carried by XRP (4 trades, +$96.85; the other three lanes −$2.65).
- OOS, the Gaussian control made +$13.29 at PF 1.04 and is also carried by XRP.

Verdict: better-calibrated probabilities do not turn into a tradeable edge at these hurdles on this tape.

## Track 4 — evolutionary search (`scripts/alpha_evolve_v2.py`)

The run had 4 families × (1 seed + 4 generations × 12 mutations) = 196 evaluations, 152 of them distinct genomes, all on IS, on causal pools with the fixed core. Every evaluation is logged in `data/alpha-evolve-v2.json`.

- The seed is the F2 rule: taker, tp 5c, stop 7c, 90s taker barrier. It made 57 trades on IS, −$78.45.
- Primitives: depletion (signed flow / revision imbalance / 5s mid shift), divergence, spread regime, and time-of-day/vol regime.
- Divergence is the engine's spot feed minus RTI: the last tick at or before the trigger within 1.5s, 98% coverage. `ticks.npz` spot is a different feed from BRTI (median −0.13bp), so this primitive is testable.
- The book cache stores top-of-book only, so true order-book depletion velocity cannot be measured. The touch-revision imbalance stands in for it.
- **No evaluation in any family was IS-eligible.** Every genome with ≥ 30 IS trades lost money on IS. The best of those per family:
  - depletion: 37 trades, −$37.36;
  - divergence: 30, −$22.41;
  - spread_regime: 49, −$47.38;
  - tod_vol: 58, −$43.04.
- With nothing eligible, selection drifted to small, lucky samples. The champions have 4–13 IS trades.
- Hour-of-day genes have few independent samples on a tape of about two days.

| Family | Champion | IS n | W/L | IS net | PF | Genome (changes from seed) |
|---|---|---|---|---|---|---|
| depletion | `depletion:4.11` | 11 | 11/0 | +40.64 | ∞ | z ≥ 0.4, flow ≥ 0.15, shift ≥ −2c, cost 0.20–0.60, tp 6c, stop 10c, no barrier |
| divergence | `divergence:4.8` | 4 | 4/0 | +40.62 | ∞ | z ≥ 0.4, div ≥ 0.5bp, cost 0.30–0.60, tp 4c, stop 10c, 180s settle barrier |
| spread_regime | `spread_regime:4.7` | 13 | 12/1 | +37.45 | 8.15 | z ≥ 0.4, cost 0.20–0.60, tp 6c, stop 10c, 180s taker barrier (its own spread/time genes ended back at the seed) |
| tod_vol | `tod_vol:4.2` | 6 | 5/1 | +9.22 | 1.86 | maker 5s, δ ≥ 4bp, tp 8c, cost 0.25–0.55 (its own time/vol genes ended back at the seed) |

OOS, the four champions made 0–3 trades each, net −$14.29 to $0.00 (ledger above).

## Tests (run in the worktree after the last code change)

| Suite | Baseline (before any change) | Final |
|---|---|---|
| `compileall openkal scripts tests` | rc 0 | rc 0 |
| `tests/test_sim_core.py` | — (new) | 33/33 |
| `tests/test_daemon_247.py` | 20 run, OK, 2 expected failures | 20 run, OK, 2 expected failures |
| `tests/test_live_exit.py` | 52/52 | 52/52 |
| `tests/test_gate_controls.py` on parity copy `/tmp/openkal-rnd/parity.db` | 18/18 | 18/18 |
| `tests/test_day_pipeline.py` | 74/74 | 74/74 |

## How this was run

- The lead wrote `sim_core`, its tests, `rnd_common`, the pool builder and the gate.
- Three Haiku 5.5 workers each built one track script against written briefs. They ran IS only, with heavy runs serialised on `data/.heavy.lock`.
- The lead read each script in full and found no OOS leak, no hand-computed prices and no gate bypass. The lead then reran Tracks 2 and 4 on the corrected pools and ran every OOS score.
- Peak memory of the full pool build was about 19 GB, on the machine that runs the live engine. Hence the lock and the lane-by-lane builder.
- A fresh Haiku reviewer, given only the bar and the artifacts, re-derived the report from the data files and code. Every table row and every ledger row matched. It confirmed in code that IS mode never simulates an OOS trade, that every decision feature is causal, and that the payoff models are fitted on IS rows only. It raised 7 findings, and each was reproduced before acting:
  - **High:** the parity script's unlogged F2 OOS computation. Disclosed above.
  - **Medium:** 7 of 8 champions are IS-ineligible. Now stated in the bottom line.
  - **Medium:** fill optimism. Exit-leg lag fixed in `104b04e`; queue position and depth disclosed.
  - **Low:** look count and the added family. Multiple-testing note added.
  - **Low:** trigger-band pricing outside sim_core. Disclosed.
  - **Low:** stale parity JSON. Regenerated.
  - **Low:** ledger written only after scoring. Fixed in `9e19bcc`.

## Open questions for the owner

1. **Lock-leg cap.** Should the $0.62 cap bind a risk-reducing opposite-side lock leg (`1 − entry − L`)? With entries ≤ $0.58 it never came up here.
2. **`MIN_PASSIVE_QTY = 20`** in production `openkal/execution.py` and in `scripts/alpha_discover.py`, against the 50-contract research floor. Left unchanged.
3. **Look-ahead in legacy studies.** The legacy `obi_pool` look-ahead (nearest spot, up to +1.5s) affects every earlier study built on it, including the sweep that produced the live F2 rule. Re-validating F2 on causal pools is the natural next step; Track 2's settle-only row is that test, and it decays to +$4.00 OOS.
4. **Exit lag in the legacy simulator.** `alpha_search._sim_event` also places take-profit legs at the fill instant, and earlier studies used it. sim_core no longer does.

## Earlier work on this branch (from the previous report, commit `9e39dee`)

- Track A: `tests/test_daemon_247.py` (commit `8a5c9b6`). It has 2 expected failures for known gaps:
  - a nested `scripts/.env` would be pushed by git_sync;
  - a root `.venv/` path is not matched by `_forbidden`.
- Track B: the F2 NO-side convention bug. The legacy simulator charged a NO taker `yes_bid` instead of `1 − yes_bid`. Under the corrected convention every F2 variant loses on IS and OOS. sim_core makes this convention impossible to get wrong (Track 1).
- Track C: the LayaGate ablation was blocked; there are no historical per-trigger verdicts. Oracle rows are hindsight ceilings only.
- The live `data/evolve-cache/` was mid-rebuild during this run: at 16:36, only a partial `book_KXBTC15M.npz` was present. This study never read it.
