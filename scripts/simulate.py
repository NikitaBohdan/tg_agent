"""Offline simulator: YOU play the manager (and the operator) in the console.

No Telegram involved. Uses the same Agent core as the live runner, so prompt
and guard behaviour can be tried cheaply.

Run:  python -m scripts.simulate
Commands at the MANAGER> prompt:  /state  — show booking state,  /quit — exit
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime

from agent.brain import Brain, BrainError
from agent.config import get_settings
from agent.core import Agent, Outcome
from agent.llm import build_chain
from agent.logging_setup import setup_logging
from agent.state import ConversationState, Goal
from agent.timeutil import target_datetime

DIM, BOLD, CYAN, YEL, RED, RST = "\033[2m", "\033[1m", "\033[36m", "\033[33m", "\033[31m", "\033[0m"


async def ainput(prompt: str) -> str:
    return (await asyncio.to_thread(input, prompt)).strip()


def show(o: Outcome) -> None:
    t = o.turn
    print(f"{DIM}  understood: {t.understood}")
    print(f"  intent={t.manager_intent}  action={t.action}  missing={t.missing_facts}  via {o.provider}")
    print(f"  decision: {t.decision}{RST}")
    for n in o.guard_notes:
        print(f"{RED}  guard: {n}{RST}")


async def main() -> int:
    s = get_settings()
    setup_logging(s.log_dir, "WARNING")
    tz = s.tz
    now = lambda: datetime.now(tz)  # noqa: E731
    target = target_datetime(tz, s.booking_time)
    state = ConversationState(
        goal=Goal(date=target.date().isoformat(), time=f"{s.booking_time:%H:%M}",
                  party_size=s.booking_party_size, tz=s.booking_tz)
    )
    agent = Agent(Brain(build_chain(s), now_fn=now), state, now)
    print(f"{BOLD}Simulator{RST}: you are the MANAGER. Goal: {state.goal.model_dump()}\n")

    step = agent.start
    while True:
        try:
            o = await step()
        except BrainError as e:
            print(f"{YEL}!! LLM failed: {e}{RST}")
            if (await ainput("retry? [Y/n] ")).lower() == "n":
                return 1
            continue

        show(o)
        if o.message_to_manager:
            print(f"{CYAN}AGENT to MANAGER:{RST} {o.message_to_manager}")

        if o.operator_question:
            answer = await ainput(f"{YEL}OPERATOR? {o.operator_question}\nOPERATOR> {RST}")
            step = lambda a=answer: agent.on_operator_answer(a)  # noqa: E731
            continue

        if o.finished:
            print(f"\n{BOLD}Finished.{RST} Booking: {state.booking.model_dump(mode='json')}")
            return 0

        while True:
            msg = await ainput("MANAGER> ")
            if msg == "/quit":
                return 0
            if msg == "/state":
                print(state.booking.model_dump(mode="json"), state.facts, "recap_sent:", state.recap_sent)
                continue
            if msg:
                break
        step = lambda m=msg: agent.on_manager_message(m)  # noqa: E731


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except (KeyboardInterrupt, EOFError):
        print()
