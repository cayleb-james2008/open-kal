"""DEMO RFQ/combo quote capability probe.

A production key is not a demo key.  Kalshi's demo environment has its own
account and key namespace; the production key is rejected at authentication with

    HTTP 401
    {"error":{"code":"authentication_error",
              "message":"We could not authenticate your request",
              "details":"NOT_FOUND"}}

The probe prefers `KALSHI_DEMO_API_KEY_ID` and `KALSHI_DEMO_PRIVATE_KEY_PATH`
from `.env`/the environment.  If they are absent it falls back to the configured
production key so the exact rejection is recorded rather than guessed.  It never
touches a production order endpoint.

When demo credentials are present it lists open combo RFQs, posts a deliberately
far 0.01/0.01 post-only quote, records the exact response, and cancels the quote.
It does not create an RFQ.

    .venv/bin/python -m openkal.maker.demo_probe
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..auth import KalshiClient
from ..config import ROOT, Settings, load_env, load_settings

OUT = ROOT / "data" / "rfq-demo-probe.json"


def demo_settings(settings: Settings) -> tuple[Settings, str]:
    env = load_env()
    key_id = (env.get("KALSHI_DEMO_API_KEY_ID") or "").strip()
    key_path = (env.get("KALSHI_DEMO_PRIVATE_KEY_PATH") or "").strip()
    if key_id and key_path:
        path = Path(key_path)
        if not path.is_absolute():
            path = ROOT / path
        return replace(settings, env_name="demo", api_key_id=key_id,
                       key_path=path), "demo_key"
    return replace(settings, env_name="demo"), "prod_fallback"


async def probe(settings: Settings, *, client: KalshiClient | None = None,
                save: bool = True) -> dict[str, Any]:
    demo, source = demo_settings(settings)
    own = client is None
    if client is None:
        try:
            client = KalshiClient(demo)
        except Exception as exc:  # noqa: BLE001 - missing demo key file is a result
            result = {"can_post": False, "reason": "demo_credential_error",
                      "credential_source": source,
                      "error": f"{type(exc).__name__}: {exc}"}
            if save:
                _save(result)
            return result

    responses: list[dict[str, Any]] = []

    async def call(method: str, path: str, params: dict[str, Any] | None = None,
                   body: dict[str, Any] | None = None) -> tuple[int, Any]:
        if method == "GET":
            code, payload = await client.get(path, params)
        elif method == "POST":
            code, payload = await client.post(path, body or {})
        elif method == "DELETE":
            code, payload = await client.delete(path, params)
        else:
            raise ValueError(method)
        responses.append({"method": method, "path": path,
                          "status": code, "body": payload})
        return code, payload

    try:
        await call("GET", "/exchange/status")
        code, payload = await call("GET", "/communications/rfqs", {"limit": 100})
        if code != 200:
            # Ask the quote endpoint directly.  Auth is checked before RFQ
            # validation, so a dummy id is safe: it cannot create anything.
            await call(
                "POST", "/communications/quotes",
                body={"rfq_id": "00000000-0000-0000-0000-000000000000",
                      "yes_bid": "0.0100", "no_bid": "0.0100",
                      "rest_remainder": False, "post_only": True})
            result = {
                "can_post": False,
                "reason": "demo_auth_rejected",
                "credential_source": source,
                "detail": ("The configured key does not authenticate to the demo "
                           "environment. Demo keys are separate from production keys; "
                           "the quote endpoint rejects this key before RFQ validation."),
                "responses": responses,
            }
            if save:
                _save(result)
            return result

        rfqs = payload.get("rfqs") or []
        combos = [r for r in rfqs
                  if r.get("status") == "open"
                  and (r.get("mve_collection_ticker") or r.get("mve_selected_legs"))]
        if not combos:
            result = {
                "can_post": None,
                "reason": "no_open_combo_rfq",
                "credential_source": source,
                "detail": "Demo auth works, but there was no open combo RFQ to quote.",
                "rfqs_seen": len(rfqs),
                "responses": responses,
            }
            if save:
                _save(result)
            return result

        rfq = combos[0]
        quote_code, quote_payload = await call(
            "POST", "/communications/quotes",
            body={"rfq_id": rfq.get("id"), "yes_bid": "0.0100",
                  "no_bid": "0.0100", "rest_remainder": False,
                  "post_only": True})
        quote_id = (quote_payload or {}).get("id") if quote_code == 201 else None
        cancel_code = None
        if quote_id:
            cancel_code, _ = await call("DELETE", f"/communications/quotes/{quote_id}")
        took = quote_code in (200, 201)
        result = {
            "can_post": took,
            "reason": "quote_created" if took else "quote_rejected",
            "credential_source": source,
            "rfq": {"id": rfq.get("id"), "market_ticker": rfq.get("market_ticker"),
                    "target_cost_dollars": rfq.get("target_cost_dollars"),
                    "mve_collection_ticker": rfq.get("mve_collection_ticker")},
            "quote_id": quote_id,
            "cancel_status": cancel_code,
            "responses": responses,
        }
        if save:
            _save(result)
        return result
    finally:
        if own:
            await client.close()


def _save(result: dict[str, Any]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, default=str) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="demo_probe", description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = asyncio.run(probe(load_settings()))
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"can_post={result.get('can_post')} reason={result.get('reason')} "
              f"source={result.get('credential_source')}")
        for row in result.get("responses") or []:
            print(f"  {row['method']} {row['path']} -> {row['status']}: "
                  f"{json.dumps(row['body'], default=str)[:500]}")
        print(f"saved {OUT.relative_to(ROOT)}")
    return 0 if result.get("can_post") else 1


if __name__ == "__main__":
    raise SystemExit(main())
