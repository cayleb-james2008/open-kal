# Production readiness, 2026-11-18 (01:00–02:05 EST)

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
| `tests/test_live_exit.py` | 97/97 | **130/130** |
| `tests/test_daemon_247.py` | 20/20 | 20/20 |
| `tests/test_gate_controls.py` | 18/18 | 18/18 |
| `tests/test_day_pipeline.py` | 76/76 | 76/76 |

Dashboard: `npm run build` ok, `npm run role-coverage` ok, `render-check`
**PASSED** (role coverage, one source for motion values, all five job questions
in the first fold at 1440x900 and 390x844, spacing, no inert control, contrast
in both themes, reduced motion).

The new checks cover: the 0ms cached read and the refresh cadence; the filter
on, off, with a minimum confidence, and against an offline gate; a verdict
filed on the contract it judged; the Student-t quantiles and symmetry; the
imbalance; the band floor and ceiling; the $0.62 cap; the imbalance and flow
alignment on both flow windows; the net edge; and the hold-to-settlement exit.

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

Consequence: `http://127.0.0.1:8787` is down, no new depth rows or gate
verdicts are being written, and the 24/24 true-L2 window target cannot be met
while the engine is down.

Neither process was restarted here: the session guardrails say to keep
`run.py` untouched and not to restart the L2 supervisor. The exact steps, ready
to run, are:

```bash
bash scripts/restart_flat.sh          # paper, unarmed, venue flat -> comes back at $121.46
.venv/bin/python scripts/overnight_l2_supervisor.py --until 2026-11-18T06:00:00-05:00
```

Then re-score:

```bash
.venv/bin/python scripts/forward_oos.py --strategy student_t_l2_settle
.venv/bin/python scripts/forward_oos.py --strategy student_t_l2_settle --gate-filter
```

The program stays unarmed until it clears the same seven gate conditions as
every other candidate.
