"""One-time interactive login for the AGENT account.

Run:  python -m scripts.login
Telethon asks for the phone number, the login code (and 2FA password if set)
in this terminal. The resulting .session file is git-ignored.
"""

from __future__ import annotations

import asyncio

from agent.config import get_settings
from agent.tg import build_client


async def main() -> None:
    s = get_settings()
    client = build_client(s)
    await client.start()  # interactive prompts on first run
    me = await client.get_me()
    print(f"Logged in as {me.first_name} (@{me.username or '-'}), id={me.id}")
    print(f"Session saved to {s.tg_session}.session")
    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
