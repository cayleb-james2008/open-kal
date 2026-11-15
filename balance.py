#!/usr/bin/env python3
"""Print the Kalshi account balance from an authenticated call. Read-only."""
import json
import sys

import kalshi_auth

status, payload = kalshi_auth.balance()
if status != 200:
    print(f"HTTP {status}: {json.dumps(payload)}", file=sys.stderr)
    sys.exit(1)

print(f"HTTP {status} | cash ${float(payload['balance_dollars']):.4f} "
      f"| portfolio ${float(payload.get('portfolio_value') or 0):.2f}")
