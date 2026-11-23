"""Smoke check, sends NOTHING to the manager.

- connects with the saved session (no interactive login),
- resolves MANAGER_USERNAME,
- posts a test note to the agent's own Saved Messages (operator channel),
- prints today's booking target in the restaurant time zone.

Run:  python -m scripts.check_telegram
"""

from __future__ import annotations

import asyncio
import sys

from agent.config import get_settings
from agent.tg import build_client, resolve_manager, send_text
from agent.timeutil import human_date, minutes_until, target_datetime


async def main() -> int:
    s = get_settings()
    client = build_client(s)
    await client.connect()
    if not await client.is_user_authorized():
        print("Not logged in. Run: python -m scripts.login")
        return 1

    me = await client.get_me()
    print(f"Agent account: {me.first_name} (@{me.username or '-'})")

    manager = await resolve_manager(client, s.manager_username)
    print(f"Manager resolved: @{s.manager_username}, id={manager.id}, name={manager.first_name}")

    target = target_datetime(s.tz, s.booking_time)
    mins = minutes_until(target)
    print(f"Booking target: {human_date(target)}, {target:%H:%M} {s.booking_tz} "
          f"({'in %.0f min' % mins if mins > 0 else 'ALREADY PASSED'})")

    await send_text(client, "me", "🤖 [agent] Smoke check OK — operator channel works.")
    print("Test note sent to Saved Messages.")
    await client.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
