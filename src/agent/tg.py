"""Thin Telethon wrapper: client construction, peer resolution, safe sending.

Kept deliberately small; the agent loop (stage 3) builds on these helpers.
"""

from __future__ import annotations

import asyncio
import logging

from telethon import TelegramClient, errors
from telethon.tl.types import User

from .config import Settings

log = logging.getLogger(__name__)


def build_client(s: Settings) -> TelegramClient:
    s.ensure_dirs()
    return TelegramClient(
        str(s.tg_session),
        s.tg_api_id,
        s.tg_api_hash.get_secret_value(),
        # Survive flaky networks: Telethon reconnects transparently.
        connection_retries=-1,  # infinite
        retry_delay=5,
        auto_reconnect=True,
        # Telethon auto-sleeps on FloodWait up to this many seconds.
        flood_sleep_threshold=120,
        device_model="tg-booking-agent",
        app_version="0.1.0",
    )


async def resolve_manager(client: TelegramClient, username: str) -> User:
    entity = await client.get_entity(username)
    if not isinstance(entity, User) or entity.bot:
        raise ValueError(f"@{username} is not a regular user account")
    return entity


async def send_text(client: TelegramClient, peer, text: str, *, retries: int = 3):
    """Send with bounded retries on transient errors. Never raises on FloodWait < limit."""
    for attempt in range(1, retries + 1):
        try:
            return await client.send_message(peer, text)
        except errors.FloodWaitError as e:
            log.warning("FloodWait %ss on send (attempt %d)", e.seconds, attempt)
            await asyncio.sleep(e.seconds + 1)
        except (ConnectionError, OSError, asyncio.TimeoutError) as e:
            log.warning("Network error on send (attempt %d): %s", attempt, e)
            await asyncio.sleep(2 * attempt)
    raise RuntimeError(f"Failed to send message after {retries} attempts")
