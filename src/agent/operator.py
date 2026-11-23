"""Human-in-the-loop channel. Answers are delivered as events via a callback,
so waiting for the operator never blocks the conversation with the manager.

saved_messages: questions are posted to the agent account's Saved Messages;
    the operator answers from any device logged into that account — ideally as
    a reply to the question, but any non-agent message counts when a question
    is open.
console: questions are printed; answers are typed in the terminal (debug).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from telethon import TelegramClient, events

from .tg import send_text

log = logging.getLogger(__name__)

AGENT_PREFIX = "🤖"
AnswerCallback = Callable[[str], Awaitable[None]]


def format_question(question: str, context: str | None = None) -> str:
    """Question for the operator, with the manager's message quoted so the operator
    sees everything the manager said, not only the part the model asked about."""
    parts = [f"{AGENT_PREFIX} QUESTION from the booking agent:"]
    if context:
        parts.append(f"The manager wrote:\n«{context}»\n")
    parts.append(question)
    parts.append("\n↩️ Reply to this message with the answer.")
    return "\n".join(parts)


class SavedMessagesOperator:
    def __init__(
        self, client: TelegramClient, on_answer: AnswerCallback, open_question_id: Callable[[], int | None]
    ):
        self.client = client
        self.on_answer = on_answer
        self.open_question_id = open_question_id  # None = no open question
        self._own_ids: set[int] = set()

    def register(self) -> None:
        self.client.add_event_handler(self._on_saved, events.NewMessage(chats="me"))

    async def ask(self, question: str, context: str | None = None) -> int | None:
        msg = await send_text(self.client, "me", format_question(question, context))
        self._own_ids.add(msg.id)
        return msg.id

    async def notify(self, text: str) -> None:
        try:
            msg = await send_text(self.client, "me", f"{AGENT_PREFIX} {text}")
            self._own_ids.add(msg.id)
        except Exception as e:  # a failed notice must not break the agent
            log.error("operator notify failed: %s", e)

    async def _on_saved(self, event: events.NewMessage.Event) -> None:
        m = event.message
        text = (m.raw_text or "").strip()
        if m.id in self._own_ids or not text or text.startswith(AGENT_PREFIX):
            return
        open_id = self.open_question_id()
        reply_to = m.reply_to_msg_id
        if reply_to and reply_to not in (open_id, None) and reply_to in self._own_ids:
            await self.notify("That question is already closed — your answer was not used. "
                              "Reply to my latest question, or send a new message (not a reply) "
                              "to give me an instruction.")
            return
        if open_id is None:
            log.info("operator sent an instruction in Saved Messages (msg %s)", m.id)
            await self.notify("Got it — instruction received.")
        else:
            log.info("operator answered in Saved Messages (msg %s)", m.id)
        await self.on_answer(text)


class ConsoleOperator:
    def __init__(self, on_answer: AnswerCallback):
        self.on_answer = on_answer
        self._task: asyncio.Task | None = None

    def register(self) -> None:
        pass

    async def ask(self, question: str, context: str | None = None) -> int | None:
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = asyncio.create_task(self._read(format_question(question, context)))
        return None

    async def _read(self, text: str) -> None:
        answer = await asyncio.to_thread(input, f"\n{text}\nOPERATOR> ")
        if answer.strip():
            await self.on_answer(answer.strip())

    async def notify(self, text: str) -> None:
        print(f"\n{AGENT_PREFIX} {text}")
