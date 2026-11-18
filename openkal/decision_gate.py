"""The Laya decision gate.

The plan puts a small decision model in front of the deterministic kernel. This is the
real one: Laya served locally over the Jev `/v1/systemone` wire protocol
(`python -m laya.serve`, typed-decisions checkpoint, CPU). It never invents numbers —
it reads the kernel's own state and answers one question about it.

If the server is down or slow, the gate reports itself unavailable and the kernel's
numeric edge decides. That is a visible, logged fallback, not a silent substitution:
the dashboard shows which of the two made the call on every entry.

The gate never sits in the entry path. A forward pass on this CPU costs 610-650ms,
and a tick loop that waits for one is a tick loop that cannot price a 15-minute
binary. So the review runs in a background task, its verdict is cached per ticker,
and the entry decision reads that cache: a dictionary lookup, 0ms. The cache is
refreshed every `GATE_REFRESH_S` seconds, or sooner when the book's mid has moved
`GATE_MID_MOVE_C`, because the question Laya was asked ("should it open now?")
describes a book that has since moved.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import httpx

# How long a cached verdict stays current.
GATE_REFRESH_S = 15.0
# A mid move of this size makes the cached opinion stale. Two cents is the venue's
# own tick: the book has moved a whole tick since Laya looked at it.
GATE_MID_MOVE_C = 0.02


@dataclass
class Verdict:
    available: bool
    allow: bool
    confidence: float
    note: str
    model: str
    latency_ms: float
    error: str | None = None
    # Which ticker this verdict was about, and when it was taken. The cache is keyed
    # by ticker, so a verdict from the previous window can never answer for this one.
    ticker: str = ""
    ts: float = 0.0


class LayaGate:
    def __init__(self, url: str = "http://127.0.0.1:8000", timeout: float = 6.0,
                 mode: str = "auto", refresh_s: float = GATE_REFRESH_S,
                 mid_move_c: float = GATE_MID_MOVE_C) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.last: Verdict | None = None
        self.calls = 0
        self.failures = 0
        # A 1.7s zero-shot CPU answer cannot sit in the entry path. Until a fine-tuned
        # ONNX model exists, Laya runs in shadow: its probability and lane verdict are
        # logged for the dashboard and for RLCD training, while the deterministic
        # kernel decides.
        self.mode = "onnx" if (mode in {"auto", "onnx"} and self._onnx_ready()) else "shadow"
        self.refresh_s = refresh_s
        self.mid_move_c = mid_move_c
        # The cached verdicts, one per ticker. This is what the entry path reads.
        self.cache: dict[str, Verdict] = {}
        # The mid each cached verdict was taken against, so a book that has moved a
        # tick can ask for a fresh opinion without waiting for the timer.
        self.mid_ref: dict[str, float] = {}
        # The in-flight background review per ticker. One at a time: a slow server
        # must not be able to pile up work behind the tick loop.
        self.tasks: dict[str, asyncio.Task] = {}

    @staticmethod
    def _onnx_ready() -> bool:
        try:
            import onnxruntime  # noqa: F401
            from pathlib import Path
            return Path("data/laya-rlcd/laya.onnx").exists()
        except Exception:  # noqa: BLE001
            return False

    # ---------- the 0ms path ----------
    def cached(self, ticker: str) -> Verdict | None:
        """The last verdict for `ticker`. A dictionary read: no I/O, no await."""
        return self.cache.get(ticker)

    def stale(self, ticker: str, mid: float | None = None) -> bool:
        """True when the cached verdict for `ticker` should be refreshed."""
        verdict = self.cache.get(ticker)
        if verdict is None:
            return True
        if time.time() - verdict.ts >= self.refresh_s:
            return True
        ref = self.mid_ref.get(ticker)
        if mid is not None and ref is not None and abs(float(mid) - ref) >= self.mid_move_c:
            return True
        return False

    # ---------- the background path ----------
    def schedule(self, slot: Any, quote: Any, seconds_left: float, decision: Any,
                 on_done: Any = None) -> None:
        """Refresh one ticker's verdict in the background. The hot path never waits.

        At most one review per ticker is in flight. A second caller while one is
        running is dropped: the running review will produce a verdict from a book
        that is at most one tick old, and queuing reviews behind a 650ms forward
        pass would rebuild the stall this module exists to remove.
        """
        ticker = str(getattr(quote, "ticker", "") or "")
        if not ticker:
            return
        running = self.tasks.get(ticker)
        if running is not None and not running.done():
            return

        async def run() -> None:
            try:
                verdict = await self.review(slot, quote, seconds_left, decision)
            except Exception:  # noqa: BLE001 - a shadow failure is not a trading event
                self.failures += 1
                return
            verdict.ticker = ticker
            verdict.ts = time.time()
            self.cache[ticker] = verdict
            self.last = verdict
            self.mid_ref[ticker] = _mid_of(quote)
            if on_done is not None:
                try:
                    on_done(verdict)
                except Exception:  # noqa: BLE001 - a logging failure is not a trading event
                    pass

        try:
            self.tasks[ticker] = asyncio.create_task(run(), name=f"laya-gate-{ticker}")
        except RuntimeError:  # no running loop: there is nothing to schedule onto
            return

    async def review(self, slot: Any, quote: Any, seconds_left: float, decision: Any) -> Verdict:
        state = self._state(slot, quote, seconds_left, decision)
        question = {
            "type": "choice",
            "instructions": (
                "A market-making bot is considering opening the position described in the "
                "state. Should it open now, or skip this trade?"
            ),
            "criteria": {
                "enter": "Open the position now: the model edge is real and the book supports it.",
                "skip": "Skip this trade: the edge looks like noise, the spread or the timing is bad.",
            },
        }
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.url}/v1/systemone",
                    json={"state": state, "questions": {"open": question}},
                )
            elapsed = (time.perf_counter() - started) * 1000
            self.calls += 1
            if resp.status_code != 200:
                self.failures += 1
                return self._down(f"http {resp.status_code}", elapsed)
            payload = resp.json()
            answer = (payload.get("answers") or {}).get("open") or {}
            choice = str(answer.get("choice") or "").strip().lower()
            probs = answer.get("probabilities") or {}
            p_enter = float(probs.get("enter") or (1.0 if "enter" in choice else 0.0))
            note = str(answer.get("noul") or "").strip()[:180]
            # `confidence` on a choice is normalized entropy; the decision is the
            # probability the model actually assigned to "enter".
            allow = p_enter >= 0.55
            verdict = Verdict(True, allow, p_enter, note or f"{choice} p={p_enter:.2f}",
                              str(payload.get("model") or "laya"), elapsed)
            self.last = verdict
            return verdict
        except Exception as exc:  # noqa: BLE001 - a gate that throws must not stop trading
            self.failures += 1
            return self._down(f"{type(exc).__name__}: {exc}", (time.perf_counter() - started) * 1000)

    def _down(self, error: str, elapsed: float) -> Verdict:
        verdict = Verdict(False, True, 0.0, "gate offline: kernel edge decides", "laya", elapsed, error)
        self.last = verdict
        return verdict

    @staticmethod
    def _state(slot: Any, quote: Any, seconds_left: float, decision: Any) -> str:
        pos = slot.position
        p_side = slot.p_up if (decision.side or "YES") == "YES" else 1 - slot.p_up
        return (
            f"lane {slot.cfg.series} ({slot.cfg.display}). "
            f"spot {slot.spot:.4f}, model fair probability for the side being bought "
            f"{p_side:.3f}, best bid {quote.yes_bid} and ask {quote.yes_ask}, "
            f"spread {quote.spread:.2f} dollars, {seconds_left:.0f} seconds until the contract "
            f"expires, model edge over the ask {decision.edge:.3f} dollars. "
            f"Recent lane record: {slot.wins} wins and {slot.losses} losses, "
            f"calibration error {slot.ece:.3f}. "
            f"Position: {'none' if pos is None else f'{pos.contracts} contracts at {pos.entry}'}."
        )


def _mid_of(quote: Any) -> float:
    """The quote's mid as a float, 0.0 when the quote cannot say."""
    try:
        return float(quote.mid)
    except (AttributeError, TypeError, ValueError):
        return 0.0


def verdict_payload(verdict: Verdict | None) -> dict[str, Any] | None:
    if verdict is None:
        return None
    return {
        "available": verdict.available, "allow": verdict.allow,
        "confidence": round(verdict.confidence, 3), "note": verdict.note,
        "model": verdict.model, "latency_ms": round(verdict.latency_ms, 1),
        "error": verdict.error, "ticker": verdict.ticker, "ts": verdict.ts,
    }
