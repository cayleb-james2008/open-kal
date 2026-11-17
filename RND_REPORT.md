# RND report — rnd/alpha-evolve

Branch base: origin/master d9fd4ec. Worktree: /home/cayleb/open-kal-rnd (outside the live repo).
Live engine and `scripts/daemon_247.py` in /home/cayleb/open-kal were not read for writes, signalled, or restarted.

## Track A — deterministic tests (commit 8a5c9b6, tests only)

`tests/test_daemon_247.py`: 20 tests, 18 pass, 2 `expectedFailure` (known gaps, not fixed here).

| Check | Result |
|---|---|
| git_sync allowlist: allowlisted change committed and pushed; untouched paths not staged | pass |
| git_sync refuses `openkal/*.db`, `scripts/keys/*.pem`, `data/*` (non-report), oversized >5 MB; index reset | pass |
| 5,000,000-byte file (at limit) is not refused | pass |
| Root `.env` never staged | pass |
| Nested `scripts/.env` is refused | **expectedFailure: pushed** |
| Root-level `.venv/…` matched by `_forbidden` | **expectedFailure: not matched** (unreachable via GIT_PATHS) |
| floor = max(105.00, 0.80·peak), incl. exact 105.00 boundary and rounding | pass |
| float floor in mc_drawdown agrees with Decimal floor_for on 0.01 grid | pass |
| 18:00 freeze: live before 18:00, frozen once, later windows return the same numbers | pass |
| mc_drawdown: p_win=1 → 0 breach; p_win=0 on observed schedule → 1 breach; breach non-increasing in p_win (3 arrival models); observed schedule ignores slot cap | pass |

Existing suites, run in the worktree (parity DB copied to /tmp/openkal-rnd/parity.db):

| Suite | Result | Note |
|---|---|---|
| compileall openkal scripts tests | rc 0 | |
| tests/test_live_exit.py | 52/52 | matches requested count |
| tests/test_gate_controls.py parity.db | 18/18 | matches requested count |
| tests/test_day_pipeline.py | **74/74** | request said 76/76; the suite now reports 74 |

## Track B — F2 upgrades, and a NO-side convention bug

### B1. Baseline reproduction does not match the request

- Request: OOS +$49.56 / PF 99.00 / payoff-ECE 0.0595.
- `data/f2-structural.json` (generated 2026-11-17 05:18 EST, older cache): `F2_taker_1s` OOS +$42.65, 36 trades, PF 4.17.
- README forward OOS (live): 20 trades, 20W/0L, +$33.13, PF 99.00, payoff-ECE 0.057. PF 99.00 is the no-loss cap in `summarize`, not a measured PF.
- Rerun of the existing `scripts/f2_structural.py` on the frozen snapshot (`data/frozen-cache`, books from the 14:07 vintage): IS 63 trades +$59.16 PF 2.70; OOS 41 trades **+$40.20, PF 2.77**, win 0.927, payoff-ECE 0.0926.

None of the stored figures reproduce exactly. The request's +$49.56 is not in any file on disk.

### B2. Suspected NO-side convention bug (verified on data)

`scripts/f2_structural.py:limit_price` and `events_for` pass a NO taker order to the simulator at `1 - bid`. The simulator (`alpha_search._sim_event`) reads its price argument as the YES-level price, so `side_cost = 1 - price` is charged as `bid`, not `1 - bid`.

Checked against the book at each trade's arrival (`data/frozen-cache`, 79 NO trades in the frozen run):

- 79/79 NO trades store `entry = 1 - bid` (the true NO ask). 0/79 store `bid`.
- Consequence: a NO trade is costed at `bid`, not at its real cost `1 - bid`. Example: bid 0.44, ask 0.45 → stored entry 0.56, which is the real NO cost. The simulator charges 0.44.
- 7 OOS NO trades have a real cost above the 0.62 cap (`entry_price <= $0.62` is a stated risk gate).
- NO take-profit targets are set at `entry - 5c` in the mirrored price, so exits are checked against prints about 5c from the real level. Stops are checked at the mirrored ask.
- The OOS win rate (0.927) and PF (2.77) are in this mirrored frame. Their size is not yet shown to be real.

`scripts/evolve_f2_variants.py` runs every variant under both conventions: `as_built` (the existing mapping) and `corrected` (YES-level price for NO = the YES price it rests at: taker → bid, maker → ask, peg → ask − 1c). The corrected run is the one to trust until the owner decides.

### B3. Variant grid (27 configs × 2 conventions; IS selects, OOS scored once)

Source: `scripts/evolve_f2_variants.py` → `data/evolve-f2-variants.json`. Frozen snapshot, 66 IS / 43 OOS triggers. Selection rule: max IS net among rows with ≥20 IS trades, ≥3 IS losers, 0 crossed fills.

| Convention | Row | IS n / net | OOS n / net | OOS PF | OOS win | payoff-ECE | Gate |
|---|---|---|---|---|---|---|---|
| as_built | baseline `taker\|tp5\|none` | 63 / +59.16 | 41 / +40.20 | 2.77 | 0.927 | 0.093 | — |
| as_built | **IS-chosen** `taker\|tp7\|none` | 63 / +76.16 | 41 / +56.20 | 3.48 | 0.927 | 0.093 | fails ECE (0.093 > 0.08) |
| corrected | baseline `taker\|tp5\|none` | 58 / −79.31 | 34 / −73.48 | 0.29 | 0.529 | 0.115 | fails |
| corrected | **IS-chosen** `peg1000\|tp5\|d5` | 54 / −57.93 | 33 / −64.21 | 0.33 | 0.546 | 0.203 | fails net, PF, ECE |

Findings:
- The as_built edge is produced by the NO-side convention. Under the corrected convention, every one of the 27 variants loses money on IS and OOS (OOS range −$54 to −$75).
- Under as_built, take-profit 7c is the only upgrade that helps (OOS +$56 vs +$40). Peg entry hurts OOS in that frame (peg500 +$32, peg1000 +$35 at 5c TP). The ETH filter (d≥5bp or spread ≤2c) changes little; OOS ETH trades are 4–5 per run.
- Adaptive TP here is "tp6/tp7 when d≥6bp". I ran it as a flat TP for every d≥6bp trade, not split by band. The variant had no separate effect on the 3–6bp band, since that band keeps 5c.
- Peg fills need ≥50 printed contracts through the rest, so fills are rare and fallback IOC carries most entries. The `peg` rows are close to baseline in trade count (36–38 of 41).
- Not a deployable result: the chosen rows fail the gate under both conventions. The OOS sample is 33–41 trades.

## Track C — LayaGate ablation

**Blocked on availability, not on contamination.** Oracle rows are hindsight ceilings, not filters.

- LayaGate returns `Verdict(allow = P(enter) >= 0.55, confidence, …)`. The live mode is `shadow` (no ONNX), so `engine._gate_entry` only logs the verdict and never vetoes. Today the gate does not filter trades even in production.
- No historical per-trigger LayaGate verdicts exist on disk for the frozen tape, so the gated arm cannot be scored without calling the endpoint or running the checkpoint offline. Neither was done.
- Training cutoff: `labels.csv` (fine-tune input) ends 2026-11-16 10:05 UTC. The frozen OOS starts 2026-11-17 02:15 UTC, about 16 h later, so the OOS slice is not contaminated by time. IS may overlap training data; the IS start was not recorded. The training text contains realised outcomes, so there is leakage inside training.
- Not verified: which checkpoint the live endpoint serves, its base cutoff, and the content of the main repo's `data/laya-rank.json` (it scores post-cutoff prod ticks, not F2 triggers).

| Row (taker, tp5, no ETH filter) | IS trades / net / PF / win | OOS trades / net / PF / win | Removed IS / OOS |
|---|---|---|---|
| baseline, as_built | 63 / +59.16 / 2.70 / 0.905 | 41 / +40.20 / 2.77 / 0.927 | 0 / 0 |
| oracle (hindsight), as_built | 58 / +91.82 / 43.1 / 0.983 | 38 / +62.89 / 99.0 / 1.000 | 5 / 3 |
| baseline, corrected | 58 / −79.31 / 0.43 / 0.638 | 34 / −73.48 / 0.29 / 0.529 | 0 / 0 |
| oracle (hindsight), corrected | 38 / +58.75 / 28.0 / 0.974 | 18 / +29.68 / 99.0 / 1.000 | 20 / 16 |

- PF 99.0 is the code cap when a row has no losers, not a measured value.
- In every oracle row, the removed set is exactly the losing trades. That is a ceiling on what a perfect filter could add, not evidence that LayaGate adds anything.
- Crossed fills: 0 in all rows. Samples are small (34–41 OOS trades).
- Files: `scripts/laya_ablation.py`, `data/laya-ablation.json` (both in the worktree; `data/` is gitignored).

## Cherry-pick guidance

- Safe for master: commit 8a5c9b6 (tests only; no production code changed).
- Not yet decided: evolve_f2_variants.py and any data JSON. The JSON is under `data/`, which the live daemon's `_forbidden` rule excludes from sync, so it needs `git add -f`.
- Do not cherry-pick any F2 variant result as a live change until the NO-side convention question is settled.

## Live-tree issues found along the way (not changed)

- `/home/cayleb/open-kal/data/evolve-cache/` has **no `book_*.npz`** since the 14:24 rebuild. The four book files are in `prev-20261117-142456/`. The study used a hard-linked snapshot of those files in `data/frozen-cache/` (worktree only).
- The daemon in the process list is PID 3729291, not 3572268 as stated in the request.
