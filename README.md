# Open-Kal

**A four-lane Kalshi 15-minute crypto microstructure engine, with a 24/7 arming supervisor and a live React dashboard.**

Open-Kal trades the 15-minute crypto settlement windows listed on Kalshi (BTC, ETH, SOL, XRP). It combines a live exchange book, sub-second spot feeds and a frozen microstructure rule; it sizes each lane from live equity, enters with a pinned IOC, rests a `+5c` post-only take-profit, and uses a confirmed `7c` stop and a `90s` barrier. A separate supervisor owns the evidence gate, the live canary, the flat restart and the hard risk breakers.

> **Status (2026-11-17):** live-armed, `F2_taker_1s`, epoch started 14:41 EST at **$163.9971**. Forward OOS: **20 trades, 20W/0L, +$33.13, PF 99.00, payoff-ECE 0.057**, 0 crossed fills, max entry cost 0.59.

---

## Why this exists

Most prediction-market bots optimize a backtest. Open-Kal is built around the opposite question: **what would make an out-of-sample result untrustworthy, and how do we make that condition impossible to ignore?**

The answer is a small set of frozen gates, an execution path that reproduces the venue rather than the model, and a supervisor that refuses to arm when any gate is missing. The same code that trades also proves the replay, the fees, the reconciliation and the exit structure.

## Architecture

```text
                    ┌──────────────────────────────┐
  Kalshi WS ───────►│  openkal/kalshi_ws.py         │
  (book, trades)    │  sequence + gap recovery      │
                    └──────────────┬───────────────┘
  Coinbase/Kraken ──► openkal/feeds.py ──► spot tape
                                   │
                    ┌──────────────▼───────────────┐
                    │  openkal/engine.py           │
                    │  4 lane slots · 15m windows  │
                    │  sizing = equity / 4         │
                    └──────┬─────────────┬─────────┘
                           │             │
              openkal/strategy.py    openkal/live_exec.py
              F2 signal + exits      IOC entry · post-only TP
                           │             │
                    ┌──────▼─────────────▼─────────┐
                    │  Kalshi venue                 │
                    └──────────────┬───────────────┘
                                   │
        scripts/daemon_247.py ─────┤ hourly gate · canary · new epoch
        (24/7 supervisor)          │ 15m checks · 20% trailing floor
                                   │ $0.05 reconcile guard
                    ┌──────────────▼───────────────┐
                    │  /api/state · React dashboard│
                    └──────────────────────────────┘
```

### The four lanes

Each lane is an independent 15-minute contract with its own budget, risk state and window counter. The engine re-reads the venue balance on every window roll and re-slices **equity / 4** into each lane, so compounding is a property of the window loop, not a separate optimizer.

### The signal

`F2_taker_1s` is a single frozen configuration, not a parameter union:

- z-score ≥ `0.30`, flow ≥ `0.10`, revision ≥ `0.0`, 2-second move ≥ `3bp`
- entry cost band `0.25–0.62`, seconds-left `180–780`, max spread `0.04`
- one execution lag: **T+1s**
- side: YES iff spot ≥ strike
- probability: a payoff calibrator fitted on the in-sample 60% only

The live path and the study compute the same net edge:

```text
net = p_side - ask - exit_cost(ask) - taker_fee(ask)
hurdle = max(0.015, min_net_edge)
```

The settlement calibrator is identity on every lane, and the live ECE pressure can only make the hurdle stricter.

### Exits

- **Take-profit:** post-only ask at `entry_fill + 0.05` (the frozen structure).
- **Gap rule:** if the bid already pays more than the TP, the TP rests for `3s`; if it is still unlifted, the engine takes the bid. This is how the first live trade turned a `+5c` target into **+$33.54**.
- **Stop:** `entry - 0.07`, confirmed for `2s`; reduce-only IOC at the best bid.
- **Barrier:** flat at `90s`, the same horizon the study scored.

### The 24/7 supervisor

`scripts/daemon_247.py` has one job: never let a good result rot into an unverified live position.

- **Unarmed:** on every hourly boundary it rebuilds `data/evolve-cache`, scores the frozen rule once on the forward slice (`--since 1794909600`), and only if the gate passes with **≥20 trades** runs the audited chain: verdict → unit tests → dashboard build/render → flat check → one-contract live canary → `restart_flat.sh --new-epoch`.
- **Armed:** one minute after every quarter-hour close it records equity, venue balance, reconcile delta and the trailing floor, then writes `NEXT.json` and `data/evening-report.md`.
- **Breakers:** venue balance below `max(105.00, 0.80 × peak)` or an absolute reconcile delta above `$0.05` demotes the verdict, waits for a flat venue and restarts flat once.
- **Cache:** rebuilds `data/evolve-cache` every 6 hours, only while every lane is flat.

## Statistical gates

The gate is seven independent conditions. There is no aggregate score and no fallback.

| # | Condition | Threshold | Forward OOS (2026-11-17) |
|---|---|---|---|
| 1 | Post-decide trades | ≥ 20 | **20** |
| 2 | Net P&L | > $0 | **+$33.13** |
| 3 | Profit factor | ≥ 1.15 | **99.00** |
| 4 | Payoff ECE | ≤ 0.08 | **0.057** |
| 5 | Crossed fills | 0 | **0** |
| 6 | Max entry cost | ≤ 0.62 | **0.59** |
| 7 | Profitable lanes | ≥ 2 | **3** (BTC, ETH, XRP) |

The pre-05:00 OOS distribution carried a **94.4%** win rate; the forward post-decide slice is **20/20** with an average entry of 0.4255. A `PF 99` is a zero-loss artifact, not a claim of infinite edge; the 20-trade floor is what protects the gate from a lucky short sample.

## Risk controls

| Control | Value |
|---|---|
| Per-lane sizing | live equity / 4 |
| Hard entry cap | $0.62 |
| Trailing drawdown floor | `max($105.00, 0.80 × peak epoch balance)` |
| Current floor | **$131.20** (peak $163.9971) |
| Reconciliation guard | absolute delta ≤ **$0.05** |
| Live canary | 1 contract, reconciled, flattened |
| Breaker action | demote verdict → wait flat → one restart |
| Re-arm | fresh epoch only after a passing gate |

## Verified evidence

| Suite | Command | Result |
|---|---|---|
| Live exit structure | `tests/test_live_exit.py` | **52/52** |
| Gate controls (production DB) | `tests/test_gate_controls.py data/openkal-prod.db` | **18/18** |
| Gate controls (parity DB) | `tests/test_gate_controls.py data/openkal-parity.db` | **18/18** |
| Day pipeline | `tests/test_day_pipeline.py` | **76/76** |
| Controls bind | `tests/test_controls_bind.py data/openkal-parity.db` | pass |

The controls include a **replay-vs-ledger** check. For live epochs, taker entries are anchored to the venue's own fills (side, price, size) while exits, fees and P&L are re-derived from the tape. That is what makes a live fill reproducible without pretending an IOC always fills.

## Repository map

```text
openkal/            engine, strategy, execution, live executor, replay, store
scripts/            daemon_247.py, forward_oos.py, day_supervisor.py,
                    rebuild_cache.sh, canary.py, reconcile_live.py,
                    restart_flat.sh, research tools
tests/              live exit, gate controls, day pipeline, bind checks
dashboard/          React 19 + Vite + Tailwind dashboard ("house" UI)
docs/               static GitHub Pages showcase
data/               local tape, caches and databases (ignored)
```

## Local setup

```bash
git clone https://github.com/cayleb-james2008/open-kal.git
cd open-kal

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cd dashboard
npm install
npm run build
cd ..

cp .env.example .env        # if present; otherwise create .env
# KALSHI_API_KEY_ID=...
# KALSHI_PRIVATE_KEY_PATH=keys/kalshi-private.pem
# OPENKAL_ARM_LIVE=0        # keep paper until the gate passes

.venv/bin/python run.py     # engine + dashboard on http://127.0.0.1:8787
```

The engine re-execs into `.venv` automatically. `data/` and `keys/` are ignored; neither is required to read the code or run the tests.

### Operating the 24/7 supervisor

```bash
nohup setsid .venv/bin/python scripts/daemon_247.py \
  --since 1794909600 >> data/daemon-247.log 2>&1 < /dev/null &
```

It is a detached process, not a boot service. After a machine reboot it must be relaunched; the engine and supervisor are separate so a supervisor restart never touches a live position.

## Security

- `.env`, `keys/`, `*.pem`, databases, caches, `node_modules`, build output and agent scratch are ignored.
- No key material is committed. The only private key on the working machine is `keys/kalshi-private.pem`, which git ignores.
- The dashboard reads `/api/state`; it never receives exchange credentials.
- The live canary is one contract and is reconciled to the venue before every arm.

## Limitations

Open-Kal is research and execution infrastructure, not financial advice. Prediction markets are binary and can settle against a position quickly. The forward sample is small (20 trades); the gate is deliberately strict about that, and the trailing drawdown breaker exists because a good calibration does not remove tail risk. Live results depend on venue liquidity, fees and the quality of the spot feeds.
