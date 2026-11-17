"""The Laya decision gate.

The plan puts a small decision model in front of the deterministic kernel. This is the
real one: Laya served locally over the Jev `/v1/systemone` wire protocol
(`python -m laya.serve`, typed-decisions checkpoint, CPU). It never invents numbers —
it reads the kernel's own state and answers one question about it.

If the server is down or slow, the gate reports itself unavailable and the kernel's
numeric edge decides. That is a visible, logged fallback, not a silent substitution:
the dashboard shows which of the two made the call on every entry.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class Verdict:
    available: bool
    allow: bool
    confidence: float
    note: str
    model: str
    latency_ms: float
    error: str | None = None


class LayaGate:
    def __init__(self, url: str = "http://127.0.0.1:8000", timeout: float = 6.0,
                 mode: str = "auto") -> None:
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

    @staticmethod
    def _onnx_ready() -> bool:
        try:
            import onnxruntime  # noqa: F401
            from pathlib import Path
            return Path("data/laya-rlcd/laya.onnx").exists()
        except Exception:  # noqa: BLE001
            return False

    def shadow(self, slot: Any, quote: Any, seconds_left: float, decision: Any,
               on_done: Any = None) -> None:
        """Non-blocking review. The hot path never waits for this."""
        self.calls += 1

        async def run() -> None:
            try:
                verdict = await self.review(slot, quote, seconds_left, decision)
                self.last = verdict
                if on_done is not None:
                    on_done(verdict)
            except Exception:  # noqa: BLE001 - a shadow failure is not a trading event
                self.failures += 1

        asyncio.create_task(run(), name="laya-shadow")

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


def verdict_payload(verdict: Verdict | None) -> dict[str, Any] | None:
    if verdict is None:
        return None
    return {
        "available": verdict.available, "allow": verdict.allow,
        "confidence": round(verdict.confidence, 3), "note": verdict.note,
        "model": verdict.model, "latency_ms": round(verdict.latency_ms, 1),
        "error": verdict.error,
    }
