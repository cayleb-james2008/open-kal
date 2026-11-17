"""Prove the gate controls *bind*: break the code, and they must go red.

`tests/test_gate_controls.py` asserts every control is green on real evidence. This
file asserts the other half — that each control is capable of failing. It injects one
real regression at a time into the same code the controls exercise and requires the
matching control to fail, then restores the code and requires green again. A control
that survives its own regression is measuring nothing.

Run: .venv/bin/python tests/test_controls_bind.py [data/openkal-parity.db]

The default is the parity harness epoch: it carries the deployed program and the
replayable fills the controls bind on. The older `openkal-gate11.db` fixture has a
single trade and no `deployed_params`, so the replay control cannot go green there.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import openkal.execution as execution_mod  # noqa: E402
import openkal.replay as replay_mod  # noqa: E402
import openkal.strategy as strategy_mod  # noqa: E402
from openkal.store import Store  # noqa: E402

WINDOW = "repro: blocking the entry decision window breaks parity"
CROSSED_REPLAY = "repro: a crossed ladder at the entry window removes the trade"
CROSSED_DECISION = "crossed: bid > ask is refused at the decision"
CROSSED_LOCKED = "crossed: bid == ask is refused at the decision"
CROSSED_LIVE = "crossed: the live executor never sends a crossing order"


def load_controls():
    path = Path(__file__).with_name("test_gate_controls.py")
    spec = importlib.util.spec_from_file_location("gate_controls", path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    db = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "openkal-parity.db"
    store = Store(db)
    gc = load_controls()

    def run() -> dict[str, bool]:
        gc.RESULTS.clear()
        gc.control_repro(store)
        gc.control_crossed_book()
        return {name: ok for name, ok, _ in gc.RESULTS}

    real_replay_decide = replay_mod.decide
    real_strategy_decide = strategy_mod.decide
    real_book_open = execution_mod.Executor.__dict__["book_open"].__func__
    failures = 0

    def expect(label: str, baseline: dict[str, bool], injected: dict[str, bool],
               control: str, restored: dict[str, bool]) -> None:
        nonlocal failures
        ok = (baseline.get(control) is True and injected.get(control) is False
              and restored.get(control) is True)
        failures += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'}  {label:<44} "
              f"baseline={baseline.get(control)} injected={injected.get(control)} "
              f"restored={restored.get(control)}")

    baseline = run()
    green = all(baseline.values())
    failures += 0 if green else 1
    print(f"{'PASS' if green else 'FAIL'}  {'controls green on real evidence':<44} "
          f"{sum(baseline.values())}/{len(baseline)}")
    if not green:
        for name, ok in baseline.items():
            if not ok:
                print(f"      red: {name}")

    # 1. The engine stops enforcing its own gate: every signal may open.
    def gate_off(**kwargs: Any):
        return real_strategy_decide(**{**kwargs, "can_open": True})

    replay_mod.decide = gate_off
    strategy_mod.decide = gate_off
    injected = run()
    replay_mod.decide = real_replay_decide
    strategy_mod.decide = real_strategy_decide
    restored = run()
    expect("regression: can_open ignored -> window control red",
           baseline, injected, WINDOW, restored)

    # 2. The crossed-book guard inside `decide` is deleted: a crossed ladder is
    #    laundered into a normal one, which is exactly what the guard prevents.
    def crossed_off(**kwargs: Any):
        quote = kwargs.get("quote")
        if quote is not None and quote.yes_bid >= quote.yes_ask:
            kwargs = {**kwargs, "quote": replace(quote, yes_ask=quote.yes_bid + Decimal("0.01"))}
        return real_strategy_decide(**kwargs)

    replay_mod.decide = crossed_off
    strategy_mod.decide = crossed_off
    injected = run()
    for control in (CROSSED_DECISION, CROSSED_LOCKED):
        expect(f"regression: no crossed guard in decide -> {control.split(':')[1].strip()[:22]}",
               baseline, injected, control, baseline)
    # The replay-level crossed control is guarded twice (decide *and* the executor),
    # so it is expected to stay green while only one guard is gone — that is the
    # defense-in-depth claim. Remove both and it must go red.
    execution_mod.Executor.book_open = staticmethod(lambda quote: True)  # type: ignore[assignment]
    both_off = run()
    execution_mod.Executor.book_open = staticmethod(real_book_open)  # type: ignore[assignment]
    replay_mod.decide = real_replay_decide
    strategy_mod.decide = real_strategy_decide
    restored = run()
    expect("regression: no crossed guard anywhere -> replay red",
           baseline, both_off, CROSSED_REPLAY, restored)

    # 3. The executor's last gate is deleted. `decide` is intact, so only the live
    #    order path may be the thing that refuses — and it must be.
    execution_mod.Executor.book_open = staticmethod(lambda quote: True)  # type: ignore[assignment]
    injected = run()
    execution_mod.Executor.book_open = staticmethod(real_book_open)  # type: ignore[assignment]
    restored = run()
    expect("regression: no executor book gate -> live control red",
           baseline, injected, CROSSED_LIVE, restored)

    print(f"\n{'all controls bind' if not failures else f'{failures} binding check(s) failed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
