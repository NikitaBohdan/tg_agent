"""Entry point:  python -m agent [--reset] [--force] [--operator console]

Starts (or resumes) the booking conversation with MANAGER_USERNAME.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import sys

from .brain import Brain
from .config import get_settings
from .core import Agent
from .decision_log import DecisionLog
from .llm import build_chain
from .logging_setup import setup_logging
from .runner import Runner, now_factory
from .state import ConversationState, Goal, StateStore
from .tg import build_client, resolve_manager
from .timeutil import human_date, minutes_until, target_datetime

log = logging.getLogger("agent")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m agent", description="Telegram table-booking agent")
    ap.add_argument("--reset", action="store_true", help="archive saved state and start a new conversation")
    ap.add_argument("--force", action="store_true", help="start even if the slot is too close or passed")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation before writing first")
    ap.add_argument("--operator", choices=["saved_messages", "console"], help="override OPERATOR_CHANNEL")
    return ap.parse_args()


async def amain(args: argparse.Namespace) -> int:
    s = get_settings()
    if args.operator:
        s.operator_channel = args.operator
    s.ensure_dirs()
    setup_logging(s.log_dir, s.log_level)

    now = now_factory(s)
    target = target_datetime(s.tz, s.booking_time)
    mins = minutes_until(target)
    log.info("Goal: %d people, %s at %s %s (%.0f min from now)", s.booking_party_size,
             human_date(target), f"{s.booking_time:%H:%M}", s.booking_tz, mins)

    chain = build_chain(s)  # fail fast on LLM misconfiguration, before touching Telegram
    client = build_client(s)
    await client.connect()
    if not await client.is_user_authorized():
        log.error("Telegram session not authorized. Run: python -m scripts.login")
        return 1
    manager = await resolve_manager(client, s.manager_username)
    log.info("Manager: @%s (id=%s)", s.manager_username, manager.id)

    goal_date = target.date().isoformat()
    store = StateStore(s.data_dir / f"conversation_{manager.id}_{goal_date}.json")
    if args.reset and store.path.exists():
        archived = store.path.with_suffix(f".{now():%H%M%S}.bak.json")
        shutil.move(store.path, archived)
        log.info("Previous state archived to %s", archived)
    state = store.load()

    if state is None:
        if mins < s.start_guard_min and not args.force:
            log.error("Only %.0f min until the slot (guard: %.0f). Not starting a new conversation. "
                      "Use --force to override.", mins, s.start_guard_min)
            await client.disconnect()
            return 2
        if not args.yes:
            # Safety: opening a conversation is outward-facing (the real manager in the final run).
            typed = await asyncio.to_thread(
                input,
                f"\nAbout to START a new conversation with @{s.manager_username} "
                f"({manager.first_name}). Type the username to confirm: ",
            )
            if typed.strip().lstrip("@").lower() != s.manager_username.lower():
                log.error("Not confirmed. Nothing was sent.")
                await client.disconnect()
                return 3
        state = ConversationState(goal=Goal(date=goal_date, time=f"{s.booking_time:%H:%M}",
                                            party_size=s.booking_party_size, tz=s.booking_tz))
        store.save(state)

    agent = Agent(Brain(chain, now_fn=now), state, now)
    runner = Runner(s, client, manager, agent, store, DecisionLog(s.log_dir))
    await runner.start()
    log.info("Agent running (operator: %s). Ctrl+C to stop.", s.operator_channel)

    try:
        while True:  # Telethon reconnects itself; this loop covers a hard disconnect
            await client.run_until_disconnected()
            log.warning("Disconnected from Telegram; reconnecting in 10s")
            await asyncio.sleep(10)
            try:
                await client.connect()
            except Exception as e:
                log.error("reconnect failed: %s", e)
    finally:
        await runner.stop()
        await client.disconnect()


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(amain(args))
    except KeyboardInterrupt:
        print("\nStopped. State saved; run again to resume.")
        return 0
    except ValueError as e:  # configuration errors
        print(f"Configuration error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
