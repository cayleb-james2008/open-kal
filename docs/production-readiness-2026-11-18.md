# Production readiness, 2026-11-18 (01:00–05:30 EST)

Three deliverables landed on `master` while the overnight L2 supervisor kept
accumulating true-L2 windows. Nothing was armed: `data/strategy-verdict.json`
still reads `"promoted": false`, `F2_taker_1s` is still the default program, and
the venue was flat at $121.46 throughout.

## 1. The LayaGate is off the entry path

A forward pass on this CPU costs 610–650ms. The entry path used to `await` it,
so every tick that wanted a second opinion paid for it in latency.

* `openkal/decision_gate.py` keeps one verdict per ticker (`LayaGate.cache`),
  refreshes it in a background task (`LayaGate.schedule`), and says when the
  cache is stale (`LayaGate.stale`): every 15s, or as soon as the book's mid has
  moved 2c.
* `openkal/engine.py` reads that cache on the entry path
  (`Engine._gate_veto`): a dictionary read, 0ms, no connection opened, no
  `await`. `Engine._gate_refresh` schedules the background review and does
  nothing when the cache is still current.
* The shadow verdicts keep landing in `ticks.gate`. `store.set_tick_gate` now
  takes the ticker, so a review still in flight when the window rolls cannot be
  filed against the next contract's tick row.
* `Params.require_layagate_allow` (default `false`) and
  `Params.layagate_min_confidence` (default `0.0`) turn the cached opinion into
  a hard filter for a promoted strategy, with no code edit. An offline gate
  never blocks: the kernel's numeric edge decides.

Measured on the unit bench: the cached read takes 0.0014ms and the whole entry
path 0.0076ms, with the HTTP client booby-trapped to raise on construction.

## 2. `student_t_l2_settle`

* `openkal/features.py`: an exact Student-t CDF (Lentz continued fraction, no
  lookup table), the frozen per-lane `(nu, s)` fit from the Track 2B in-sample
  slice, `tau_eff = max(secs_left - 40, 1)`, the true-L2 imbalance
  `(yes_bid_sz - yes_ask_sz) / (yes_bid_sz + yes_ask_sz)`, and the signed
  aggressor print flow on a fast and a slow window. The CDF reproduces the
  published 0.975 quantiles to 1.7e-06.
* `openkal/strategy.py`: the entry — the $0.35–$0.60 band under the $0.62 hard
  cap, the imbalance and both flow windows aligned with the side, the net edge
  after the taker fee — and a hold-to-settlement exit that pays no exit fee.
* `openkal/engine.py`: `Engine._settlement_signal` prices it off the
  settlement index, the resting size and the aggressor tape.
  `Params.strategy_mode` selects the program; the default is still
  `F2_taker_1s`.
* `scripts/forward_oos.py --strategy student_t_l2_settle` scores it on the same
  forward slice, priced only through `openkal/sim_core.py`, with the production
  strategy deciding the entries — the scorer and the live lane cannot drift
  apart. `--gate-filter` keeps only the trades whose trigger carried an
  `allow == True` verdict in the recorded ticks.

**The score on today's slice is 2 trades, -$38.13, and the gate fails on the
20-trade floor.** The true-L2 depth exists in only 16–20 of the 81 forward
windows per lane, so the sample is not there yet. Both triggers were denied by
the recorded gate verdict — the same shield the R&D holdout measured (12/12
losing trades denied, -$99.54).

One correction worth recording: `bounded_sweep.summarize` reads a row's `entry`
as the YES-side price, which is what the research sweeps write, while the
forward tape writes the cost of the side actually held — the premium at risk.
The two differ for a NO trade, and the cost ceiling is a check on the premium.
`scripts/forward_oos.py` now measures the side cost itself
(`_with_true_side_costs`).

## 3. Checks

| Suite | Before | After |
| --- | --- | --- |
| `tests/test_sim_core.py` | 33/33 | 33/33 |
| `tests/test_live_exit.py` | 97/97 | **146/146** |
| `tests/test_daemon_247.py` | 20/20 | 20/20 |
| `tests/test_gate_controls.py` | 18/18 | 18/18 |
| `tests/test_day_pipeline.py` | 76/76 | 76/76 |

Dashboard: `npm run build` ok, `npm run role-coverage` ok, `render-check`
**PASSED** (role coverage, one source for motion values, all five job questions
in the first fold at 1440x900 and 390x844, spacing, no inert control, contrast
in both themes, reduced motion).

The new checks cover: the 0ms cached read and the refresh cadence; the filter
on, off, with a minimum confidence, against an offline gate and against an
absent opinion; a verdict filed on the contract it judged; the Student-t
quantiles and symmetry; the imbalance; the band floor and ceiling; the $0.62
cap; the imbalance and flow alignment on both flow windows and on both sides;
the net edge; the hold-to-settlement exit; the shared volatility estimator
against a vectorised reference; the live signal's own evidence; and the
scorer's pricing -- a taker that arrives one second late, capped at 62c, inside
the band, and settled with no exit fee.

## What an independent review changed

A fresh reviewer (read-only, given the bar and the diff, not this account) found
four defects that the unit bench could not see. All four are fixed and pinned by
a check now:

* **The live engine and the forward scorer priced the same contract with two
  different volatilities.** The engine used `realized_sigma` over its whole index
  buffer (about nine minutes) and the scorer used a 150-second trailing window:
  0.61x apart, and 7-11c of settlement probability at the same instant on the
  same tape -- against a 5c entry hurdle. The scored program was not the traded
  program. There is now one estimator, `features.trailing_sigma`, used by both,
  and a check pins it against a vectorised reference.
* **A binding gate filter failed open.** With `require_layagate_allow=True` and
  no verdict cached for the contract yet, the entry went through -- and the
  forward filter dropped exactly those trades. The two were the inverse of each
  other on the no-opinion case. The live path now refuses an absent opinion when
  the filter is binding, which is what the forward score does.
* **The forward gate filter was not the live filter.** It ignored the verdict's
  `available` flag and the confidence floor, and it keyed the verdict in force on
  the lane rather than on the contract. It now reads the contract's own verdict
  (with the lane as the fallback for rows written before the recorder pinned
  verdicts to their ticker), applies the same confidence floor, and reports the
  allowed, denied and no-opinion counts separately.
* **The NO side of the settlement filter was untested.** Every test took the YES
  side, so mirroring the imbalance or the flow out of the NO side changed
  nothing any suite could see. The NO side now has its own checks: a NO entry
  with a bid-heavy book is refused, and a NO entry with YES-leaning aggressors is
  refused.

Also fixed while the review was open: the settlement program's caps now come from
`Params` instead of a second copy in the scorer; the evolve loop can no longer
push the settlement band outside its bounds or turn the gate filter on; the
live executor records the settlement band's refusal in the ledger; the dead
`slot.gate_ticker` is gone; and the gate verdicts are only scanned when a score
actually asks for the gate filter.

One difference is left in place and named: the **T+1s entry lag is the scoring
model's**, not the live path's. `openkal/sim_core.py` prices every order as
arriving a full second after its trigger, which is the floor for the canonical
simulator; the live lane sends an IOC and the venue fills it at the venue's own
latency, and the paper ledger fills it at the decision price. The live executor
pins the order to the arrival ask (the same discipline the F2 program uses), so
the live fill is priced at the arrival book rather than at the trigger's book,
but the one-second delay itself is not reproduced live. Closing that gap means
a delayed-order queue in the live executor, which is a larger change to a
live-money path than this work asked for.

## Incident: the engine and the L2 supervisor died at ~01:24 EST

`run.py` (pid 1034236) and `scripts/overnight_l2_supervisor.py` (pid 1425934)
both stopped at about 01:24 EST, 2026-11-18.

* The last tick in `data/openkal-prod.db` is ts 1794983052.95 = 01:24:12 EST,
  and the first "connection refused" from the monitor is in the 01:24 minute.
* `data/engine.log` has no new session banner, so `restart_flat.sh` did not
  run: the engine was not restarted, it died.
* No OOM line in the kernel journal, and no coredump.
* The first source edit of this session is `openkal/features.py` at 01:29:17 —
  five minutes *after* the last engine tick — so the change did not cause it.
* `daemon_247.py` (214533) and `laya-serve` (4129892) are still running.

Consequence: `http://127.0.0.1:8787` went down, no new depth rows or gate
verdicts were being written, and the 24/24 true-L2 window target could not be
met while the engine was down.

The owner approved the restart at 05:01 EST, and both are back:

```bash
bash scripts/restart_flat.sh          # paper, unarmed, venue flat -> came back at $121.46
systemd-run --user --unit=openkal-l2-supervisor --property=Restart=on-failure \
  --working-directory=/home/cayleb/open-kal -- \
  .venv/bin/python -B scripts/overnight_l2_supervisor.py --until 2026-11-18T08:00:00-05:00
```

Verified after the restart: `/api/state` returns `mode=paper`, `live_armed=false`,
`venue_balance=121.46`, zero open positions and zero resting orders; new
`ws_book` rows carry non-null `yes_bid_sz`/`yes_ask_sz`; and `ticks.gate`
verdicts are being written again. The R&D watcher unit was also extended from
06:00 to 08:00 EST with its fences, `RND_OOS_UNSEAL=lead` and `--publish`
unchanged, so it outlives the 24/24-window cache export and runs Track 2C (which
chains Track 2D) on its own.

Then re-score:

```bash
.venv/bin/python scripts/forward_oos.py --strategy student_t_l2_settle
.venv/bin/python scripts/forward_oos.py --strategy student_t_l2_settle --gate-filter
```

The program stays unarmed until it clears the same seven gate conditions as
every other candidate.
