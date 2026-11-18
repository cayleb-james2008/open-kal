"""Kalshi V2 REST client: Ed25519 signing, a token bucket, and jittered 429 backoff.

Signing rule, verified against the live venue: the message is
`timestamp_ms + METHOD + full_path_including_/trade-api/v2_without_query`.
Getting that wrong is a 401, not a hint.
"""
from __future__ import annotations

import asyncio
import base64
import json
import random
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx

from .config import API_PREFIX, Settings


class TokenBucket:
    """Simple refill meter so a burst can never trip the venue's read/write caps."""

    def __init__(self, rate: float, capacity: float) -> None:
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def take(self, amount: float = 1.0) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= amount:
                    self.tokens -= amount
                    return
                await asyncio.sleep((amount - self.tokens) / self.rate)


class KalshiClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._key = self._load_key(settings.key_path)
        # The key is on Kalshi's Advanced tier (300 tokens/s, 900 burst; verified
        # 2026-11-18).  These request-count buckets stay deliberately below it:
        # venue costs are 1-30 tokens per request, so one request is not one token.
        self._read = TokenBucket(rate=40.0, capacity=80.0)
        self._write = TokenBucket(rate=15.0, capacity=30.0)
        self._client = httpx.AsyncClient(
            base_url=settings.base_url,
            timeout=httpx.Timeout(10.0, connect=5.0),
            headers={"accept": "application/json", "content-type": "application/json"},
        )
        self.last_latency_ms: float = 0.0
        self.last_error: str | None = None
        self.calls = 0

    @staticmethod
    def _load_key(path: Path):
        from cryptography.hazmat.primitives import serialization

        return serialization.load_pem_private_key(path.read_bytes(), password=None)

    def _sign(self, message: bytes) -> bytes:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        name = type(self._key).__name__
        if name == "Ed25519PrivateKey":
            return self._key.sign(message)
        return self._key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32),
            hashes.SHA256(),
        )

    def headers(self, method: str, path: str) -> dict[str, str]:
        signed_path = API_PREFIX + path.split("?")[0]
        ts = str(int(time.time() * 1000))
        sig = base64.b64encode(self._sign((ts + method.upper() + signed_path).encode())).decode()
        return {
            "KALSHI-ACCESS-KEY": self.settings.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": sig,
            "KALSHI-ACCESS-TIMESTAMP": ts,
        }

    async def request(self, method: str, path: str, *, params: dict[str, Any] | None = None,
                      body: dict[str, Any] | None = None, write: bool = False) -> tuple[int, Any]:
        bucket = self._write if write else self._read
        await bucket.take(1.0)
        delay = 0.4
        for attempt in range(4):
            started = time.perf_counter()
            headers = self.headers(method, path)
            try:
                response = await self._client.request(
                    method, API_PREFIX + path, params=params, json=body, headers=headers
                )
            except httpx.HTTPError as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                if attempt == 3:
                    return 0, {"error": self.last_error}
                await asyncio.sleep(delay + random.uniform(0, delay))
                delay *= 2
                continue
            self.calls += 1
            self.last_latency_ms = (time.perf_counter() - started) * 1000
            if response.status_code == 429:
                # Kalshi does not send Retry-After; back off with jitter.
                await asyncio.sleep(delay + random.uniform(0, delay))
                delay *= 2
                continue
            try:
                payload = response.json() if response.content else {}
            except ValueError:
                payload = {"raw": response.text[:500]}
            if response.status_code >= 400:
                self.last_error = f"HTTP {response.status_code}: {str(payload)[:200]}"
            return response.status_code, payload
        self.last_error = "rate limited after retries"
        return 429, {"error": self.last_error}

    async def get(self, path: str, params: dict[str, Any] | None = None) -> tuple[int, Any]:
        return await self.request("GET", path, params=params)

    async def post(self, path: str, body: dict[str, Any]) -> tuple[int, Any]:
        return await self.request("POST", path, body=body, write=True)

    async def delete(self, path: str, params: dict[str, Any] | None = None) -> tuple[int, Any]:
        return await self.request("DELETE", path, params=params, write=True)

    async def balance(self) -> tuple[int, Any]:
        return await self.get("/portfolio/balance")

    async def close(self) -> None:
        await self._client.aclose()


def money(value: Any, default: str = "0") -> Decimal:
    """Parse a venue decimal string into Decimal without a float in the middle."""
    if value in (None, ""):
        return Decimal(default)
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal(default)


def dump(payload: Any, limit: int = 800) -> str:
    return json.dumps(payload, default=str)[:limit]
