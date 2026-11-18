"""V2 order groups as the maker quote kill switch.

Verified against production on 2026-11-18:

  * `POST /portfolio/order_groups/create` returns 201 and a group id.
  * The single-group read is served by `external-api.kalshi.com`; the
    `api.elections.kalshi.com` host returns 404 for it.
  * `is_auto_cancel_enabled` is **false while the group is active**, true after a
    trigger, and false again after reset.  It reports the triggered state, not
    whether the rolling limit is armed.  The create-body field of the same name is
    ignored; auto-cancel on the rolling 15-second limit is inherent to the group.
  * `PUT /trigger`, `PUT /reset` and `DELETE` all return 200.

So a newly created group is already armed.  A manual trigger is the operator kill
switch; the venue produces the same state automatically once matched contracts in
the rolling 15-second window exceed the limit.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx

from ..auth import KalshiClient
from ..config import API_PREFIX

ALT_BASE = "https://external-api.kalshi.com"
Getter = Callable[[KalshiClient, str, dict[str, Any] | None], Awaitable[tuple[int, Any]]]


async def alt_get(client: KalshiClient, path: str,
                  params: dict[str, Any] | None = None) -> tuple[int, Any]:
    """GET against the host that actually serves the single-group read."""
    async with httpx.AsyncClient(base_url=ALT_BASE, timeout=10.0) as http:
        response = await http.get(API_PREFIX + path,
                                  headers=client.headers("GET", path), params=params)
    try:
        payload = response.json() if response.content else {}
    except ValueError:
        payload = {"raw": response.text[:500]}
    return response.status_code, payload


@dataclass(frozen=True)
class GroupStatus:
    id: str
    contracts_limit: str
    triggered: bool
    orders: tuple[str, ...] = ()
    exchange_index: int = 0

    @property
    def state(self) -> str:
        return "triggered" if self.triggered else "active"


class OrderGroupKillSwitch:
    """Create, inspect, trigger, reset and delete one order group.

    `getter` is injectable so the lifecycle can be tested without a network.
    """

    def __init__(self, client: KalshiClient, contracts_limit: int, *,
                 subaccount: int = 0, exchange_index: int = 0,
                 getter: Getter | None = None, status_timeout: float = 4.0) -> None:
        if not 1 <= int(contracts_limit) <= 1_000_000:
            raise ValueError("contracts_limit must be between 1 and 1,000,000")
        self.client = client
        self.contracts_limit = int(contracts_limit)
        self.subaccount = int(subaccount)
        self.exchange_index = int(exchange_index)
        self.group_id: str | None = None
        self.error: str | None = None
        self._getter = getter or alt_get
        self.status_timeout = float(status_timeout)

    async def create(self) -> GroupStatus | None:
        body: dict[str, Any] = {"contracts_limit": self.contracts_limit}
        if self.subaccount:
            body["subaccount"] = self.subaccount
        if self.exchange_index:
            body["exchange_index"] = self.exchange_index
        status, payload = await self.client.post("/portfolio/order_groups/create", body)
        if status != 201:
            self.error = f"create {status}: {str(payload)[:200]}"
            return None
        group_id = payload.get("order_group_id")
        if not group_id:
            self.error = f"create returned no group id: {str(payload)[:200]}"
            return None
        self.group_id = str(group_id)
        return await self.status()

    async def status(self) -> GroupStatus | None:
        if not self.group_id:
            return None
        path = f"/portfolio/order_groups/{self.group_id}"
        params = {"subaccount": self.subaccount, "exchange_index": self.exchange_index}
        for attempt in range(3):
            code, payload = await self._getter(self.client, path, params)
            if code == 200:
                return GroupStatus(
                    id=self.group_id,
                    contracts_limit=str(payload.get("contracts_limit_fp") or ""),
                    triggered=bool(payload.get("is_auto_cancel_enabled")),
                    orders=tuple(str(o) for o in payload.get("orders") or []),
                    exchange_index=int(payload.get("exchange_index") or 0),
                )
            if code == 404 and attempt < 2:
                await asyncio.sleep(0.3)
                continue
            self.error = f"status {code}: {str(payload)[:200]}"
            return None
        return None

    async def trigger(self) -> bool:
        if not self.group_id:
            return False
        status, payload = await self.client.request(
            "PUT", f"/portfolio/order_groups/{self.group_id}/trigger",
            body={}, write=True)
        if status != 200:
            self.error = f"trigger {status}: {str(payload)[:200]}"
            return False
        return True

    async def reset(self) -> bool:
        if not self.group_id:
            return False
        status, payload = await self.client.request(
            "PUT", f"/portfolio/order_groups/{self.group_id}/reset",
            body={}, write=True)
        if status != 200:
            self.error = f"reset {status}: {str(payload)[:200]}"
            return False
        return True

    async def delete(self) -> bool:
        if not self.group_id:
            return False
        status, payload = await self.client.delete(
            f"/portfolio/order_groups/{self.group_id}")
        if status != 200:
            self.error = f"delete {status}: {str(payload)[:200]}"
            return False
        return True

    async def _wait_for(self, triggered: bool, timeout: float | None = None) -> GroupStatus | None:
        """Poll through the read replica's short lag after a write.

        The trigger/reset writes are applied immediately, but the single-group
        read can return the previous state for a moment.  Waiting for the
        expected state is still a check: a trigger that never lands fails.
        """
        deadline = asyncio.get_event_loop().time() + (self.status_timeout if timeout is None else timeout)
        last: GroupStatus | None = None
        while True:
            last = await self.status()
            if last is not None and last.triggered == triggered:
                return last
            if asyncio.get_event_loop().time() >= deadline:
                return last
            await asyncio.sleep(0.25)

    async def verify_cycle(self) -> tuple[bool, list[str], dict[str, Any]]:
        """Create a disposable group and prove the trigger/reset cycle.

        No orders are placed.  This is the readiness check: the flag must be
        false on a fresh group, true after a manual trigger, and false after a
        reset.  A failure is reported, never assumed away.
        """
        problems: list[str] = []
        group = await self.create()
        if group is None:
            return False, [self.error or "create failed"], {}
        if group.triggered:
            problems.append("new group is already triggered")
        if group.orders:
            problems.append(f"new group already has orders: {group.orders}")
        if not await self.trigger():
            problems.append(self.error or "trigger failed")
        after_trigger = await self._wait_for(True)
        if after_trigger is None:
            problems.append(self.error or "status after trigger failed")
        elif not after_trigger.triggered:
            problems.append("trigger did not set is_auto_cancel_enabled")
        if not await self.reset():
            problems.append(self.error or "reset failed")
        after_reset = await self._wait_for(False)
        if after_reset is None:
            problems.append(self.error or "status after reset failed")
        elif after_reset.triggered:
            problems.append("reset did not clear is_auto_cancel_enabled")
        if not await self.delete():
            problems.append(self.error or "delete failed")
        return (not problems), problems, {
            "group_id": group.id,
            "contracts_limit": group.contracts_limit,
            "create_state": group.state,
            "after_trigger": None if after_trigger is None else after_trigger.state,
            "after_reset": None if after_reset is None else after_reset.state,
        }
