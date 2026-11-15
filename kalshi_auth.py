"""Kalshi API auth: sign a request with the key in this folder, read the account.

Kalshi signs `timestamp + method + path`, where `path` is the FULL request path
INCLUDING `/trade-api/v2` and EXCLUDING the query string. Getting that wrong is
the difference between 200 and `401 INCORRECT_API_KEY_SIGNATURE` — verified
against the live venue on 2026-11-15:
  "/portfolio/balance"           -> 401 INCORRECT_API_KEY_SIGNATURE
  "/trade-api/v2/portfolio/balance" -> 200

Ed25519 and RSA keys are both supported by Kalshi; which one this is comes from
the key bytes, not from a setting here. The signature is base64 of the raw
signature bytes over the UTF-8 message.

Read-only. No order, cancel or trade path lives in this file.
"""
import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE_URL = "https://api.elections.kalshi.com"
API_PREFIX = "/trade-api/v2"
HERE = Path(__file__).resolve().parent


def _load_env(path=HERE / ".env"):
    env = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _load_key(env):
    from cryptography.hazmat.primitives import serialization

    raw = Path(env["KALSHI_PRIVATE_KEY_PATH"]).read_bytes()
    return serialization.load_pem_private_key(raw, password=None)


def key_type(env=None):
    """'Ed25519' or 'RSA', read from the key bytes."""
    return type(_load_key(env or _load_env())).__name__.replace("PrivateKey", "")


def _sign(key, message: bytes) -> bytes:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    if key.__class__.__name__ == "Ed25519PrivateKey":
        return key.sign(message)
    return key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32),
        hashes.SHA256(),
    )


def headers(method, path, env=None, key=None):
    """Signed headers for one call. `path` is the API path, e.g. '/portfolio/balance'."""
    env = env or _load_env()
    key = key or _load_key(env)
    kid = env.get("KALSHI_API_KEY_ID") or env["KALSHI_KEY_ID"]
    signed_path = API_PREFIX + path.split("?")[0]
    ts = str(int(time.time() * 1000))
    sig = base64.b64encode(_sign(key, (ts + method.upper() + signed_path).encode())).decode()
    return {
        "KALSHI-ACCESS-KEY": kid,
        "KALSHI-ACCESS-SIGNATURE": sig,
        "KALSHI-ACCESS-TIMESTAMP": ts,
    }


def get(path, params=None):
    """Authenticated GET. Returns (status, payload)."""
    url = BASE_URL + API_PREFIX + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=headers("GET", path))
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return e.code, json.loads(body or b"{}")
        except ValueError:
            return e.code, {"raw": body.decode(errors="replace")}


def balance():
    """The venue's own balance payload."""
    return get("/portfolio/balance")


if __name__ == "__main__":
    print(json.dumps(balance()[1], indent=2))
