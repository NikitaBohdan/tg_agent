"""Live Telegram runner.

All inputs become Events on one asyncio queue: manager messages (a quick burst
is merged into one event), edits of earlier messages, operator answers and
silence timers. A single consumer takes them one at a time, so the agent never
races with itself. For each event the Agent core decides what to do; the runner
then sends the reply to the manager or the question to the operator, writes a
line to the decision log and saves the state file.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime

from pathlib import Path

from telethon import TelegramClient, events
from telethon.tl.types import DocumentAttributeAudio, User

from .brain import BrainError
from .config import Settings
from .core import Agent, Outcome
from .decision_log import DecisionLog
from .operator import ConsoleOperator, SavedMessagesOperator
from .schema import Status
from .state import Role, StateStore
from .tg import send_text
from .voice import VoiceError, build_stt, synthesize_voice_note, temp_dir

log = logging.getLogger(__name__)

BRAIN_RETRY_DELAYS = (20, 60, 180)  # seconds; then give up on this event and tell the operator
_DONE = (Status.confirmed, Status.declined, Status.abandoned)
_PERSONAL_KEYS = {"guest_name", "name", "booking_name", "phone", "phone_number",
                  "contact_number", "contact", "email"}


@dataclass
class Event:
    kind: str  # start | manager | edit | operator | system
    text: str = ""
    msg_ids: list[int] = field(default_factory=list)
    voice: bool = False  # the manager spoke, so answer with a voice note


class Runner:
    def __init__(
        self, s: Settings, client: TelegramClient, manager: User, agent: Agent,
        store: StateStore, dlog: DecisionLog,
    ):
        self.s = s
        self.client = client
        self.manager = manager
        self.agent = agent
        self.state = agent.state
        self.store = store
        self.dlog = dlog
        self.queue: asyncio.Queue[Event] = asyncio.Queue()
        # (msg id, text or pending transcription task, is_voice) — kept in arrival order
        self._buffer: list[tuple[int, str | asyncio.Task, bool]] = []
        self.stt = build_stt(s) if s.voice_enabled else None
        self._debounce: asyncio.Task | None = None
        self._tasks: list[asyncio.Task] = []
        if s.operator_channel == "saved_messages":
            self.operator = SavedMessagesOperator(client, self._on_operator_answer, self._open_question_id)
        else:
            self.operator = ConsoleOperator(self._on_operator_answer)

    # --- wiring --------------------------------------------------------------

    async def start(self) -> None:
        mid = self.manager.id
        self.client.add_event_handler(self._on_manager_msg, events.NewMessage(chats=mid, incoming=True))
        self.client.add_event_handler(self._on_manager_edit, events.MessageEdited(chats=mid, incoming=True))
        self.operator.register()

        if not self.state.history:
            await self.queue.put(Event("start"))
            self.dlog.note(self.agent.now(), f"session start, new conversation with @{self.s.manager_username}")
        else:
            self.dlog.note(self.agent.now(), "session resumed from saved state")
            await self._catch_up()
            if self.state.pending_question:
                await self.operator.notify(f"Still waiting for your answer: {self.state.pending_question.question}")

        self._tasks = [
            asyncio.create_task(self._consume(), name="consume"),
            asyncio.create_task(self._watch_silence(), name="silence"),
        ]
        if self.stt is not None and hasattr(self.stt, "warmup"):
            self._tasks.append(asyncio.create_task(self._warmup_stt(), name="stt-warmup"))

    async def _warmup_stt(self) -> None:
        try:
            await self.stt.warmup()
            log.info("speech-to-text model ready")
        except Exception as e:  # voice is optional; text keeps working
            log.error("STT warm-up failed (voice notes will show as untranscribed): %s", e)

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        self.store.save(self.state)

    # --- inputs --------------------------------------------------------------

    async def _on_manager_msg(self, ev: events.NewMessage.Event) -> None:
        if not ev.is_private:
            return
        self._buffer_message(ev.message)
        if self._debounce and not self._debounce.done():
            self._debounce.cancel()
        self._debounce = asyncio.create_task(self._flush_later())

    def _buffer_message(self, m) -> None:
        if m.voice and self.stt:
            # Transcribe in the background; flush awaits it, so order is preserved.
            self._buffer.append((m.id, asyncio.create_task(self._transcribe(m)), True))
        else:
            self._buffer.append((m.id, _describe(m), False))

    async def _transcribe(self, m) -> str:
        try:
            with temp_dir() as d:
                path = await self.client.download_media(m, file=str(Path(d) / "in.ogg"))
                if not path:
                    raise VoiceError("download failed")
                text = await self.stt.transcribe(Path(path))
        except Exception as e:  # never lose the message because STT failed
            log.error("transcription failed: %s", e)
            return "[voice message — could not be transcribed]"
        log.info("voice transcribed: %r", text)
        return f'(voice message, transcribed) "{text}"' if text else "[voice message — silent or unintelligible]"

    async def _flush_later(self) -> None:
        try:
            await asyncio.sleep(self.s.debounce_s)
        except asyncio.CancelledError:
            return
        await self._flush()

    async def _flush(self) -> None:
        if not self._buffer:
            return
        items, self._buffer = self._buffer, []
        texts = [await t if isinstance(t, asyncio.Task) else t for _, t, _ in items]
        await self.queue.put(Event(
            "manager", "\n".join(texts), [i for i, _, _ in items], voice=any(v for _, _, v in items)
        ))

    async def _on_manager_edit(self, ev: events.MessageEdited.Event) -> None:
        text = _describe(ev.message)
        # Telegram also emits "edits" for reactions etc. — ignore if text unchanged.
        if any(e.role == Role.manager and text in e.text for e in self.state.history):
            return
        await self.queue.put(Event("edit", f'The manager edited an earlier message; it now reads: "{text}"'))

    async def _on_operator_answer(self, text: str) -> None:
        await self.queue.put(Event("operator", text))

    def _open_question_id(self) -> int | None:
        """0 = open question without a Telegram id; None = no open question."""
        pq = self.state.pending_question
        return None if pq is None else (pq.tg_msg_id or 0)

    async def _catch_up(self) -> None:
        """Messages the manager sent while the agent was offline."""
        last = self.state.last_manager_msg_id()
        missed = [
            m for m in await self.client.get_messages(self.manager, min_id=last, limit=50)
            if not m.out and m.id > last
        ]
        for m in reversed(missed):
            self._buffer_message(m)
        if missed:
            log.info("caught up %d missed manager message(s)", len(missed))
            await self._flush()

    # --- processing ----------------------------------------------------------

    async def _consume(self) -> None:
        while True:
            ev = await self.queue.get()
            try:
                await self._process(ev)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # last line of defence: log, tell operator, keep running
                log.exception("unexpected error while processing %s", ev.kind)
                self.dlog.note(self.agent.now(), f"ERROR processing {ev.kind}: {e!r}")
                await self.operator.notify(f"⚠️ Internal error ({type(e).__name__}); I'm still running.")
            finally:
                self.store.save(self.state)

    async def _process(self, ev: Event) -> None:
        call = {
            "start": lambda: self.agent.start(),
            "manager": lambda: self.agent.on_manager_message(ev.text, max(ev.msg_ids, default=None)),
            "edit": lambda: self.agent.on_system_event(ev.text),
            "operator": lambda: self.agent.on_operator_answer(ev.text),
            "system": lambda: self.agent.on_system_event(ev.text),
        }[ev.kind]

        try:
            outcome = await call()
        except BrainError as e:
            outcome = await self._retry_brain(e)
            if outcome is None:
                return

        if ev.kind in ("manager", "edit"):
            try:
                await self.client.send_read_acknowledge(self.manager)
            except Exception as e:
                log.debug("read ack failed: %s", e)
        if ev.kind == "manager":
            self.state.manager_uses_voice = ev.voice
        context = ev.text if ev.kind in ("manager", "edit") else None
        await self._apply(outcome, reply_voice=self._reply_by_voice(ev), context=context,
                          after_operator=ev.kind == "operator")
        kind = f"{ev.kind}/voice" if ev.voice else ev.kind
        line = self.dlog.write(self.agent.now(), kind, ev.text or None, outcome)
        log.info(line)

    def _has_operator_details(self, text: str, include_name: bool) -> bool:
        """True if the message hands over details from the operator: a phone number or
        email anywhere, or the guest's name when it is being passed on right after the
        operator's answer. A later mention like "under Nikita" stays voice.
        Operator instructions such as "only 5pm works" are not details."""
        low = text.lower()
        for key, value in self.state.facts.items():
            if not set(key.split("+")) & _PERSONAL_KEYS:
                continue
            for part in value.replace(",", " ").split():
                part = part.strip().lower()
                if len(part) < 3 or part not in low:
                    continue
                is_contact = any(c.isdigit() for c in part) or "@" in part
                if is_contact or include_name:
                    return True
        return False

    def _reply_by_voice(self, ev: Event) -> bool:
        """VOICE_MODE=after_opening: the opening message is text, every later reply is a
        voice note whatever the manager uses. VOICE_MODE=mirror: voice only while the
        manager's latest message was a voice note."""
        if not self.s.voice_enabled:
            return False
        if self.s.voice_mode == "after_opening":
            return ev.kind != "start"
        return self.state.manager_uses_voice

    async def _retry_brain(self, err: BrainError) -> Outcome | None:
        log.error("brain failed: %s", err)
        self.dlog.note(self.agent.now(), f"LLM failure: {err}")
        await self.operator.notify("⚠️ LLM is failing; retrying automatically.")
        for delay in BRAIN_RETRY_DELAYS:
            await asyncio.sleep(delay)
            try:
                return await self.agent.retry()
            except BrainError as e:
                log.error("brain retry failed: %s", e)
        await self.operator.notify("❌ LLM still failing — this event was NOT answered. Check logs/agent.log.")
        return None

    async def _apply(
        self, o: Outcome, reply_voice: bool = False, context: str | None = None,
        after_operator: bool = False,
    ) -> None:
        if o.closed_question:
            await self.operator.notify(
                f"✅ No need to answer my earlier question anymore:\n«{o.closed_question}»\n"
                f"Reason: {o.turn.understood}"
            )
        if o.message_to_manager:
            sent_voice = False
            # Text: the opening message, the final confirmation, and messages that hand
            # over the guest's details (phone/email, or the name right after the
            # operator gave it), so they stay readable and exact. Everything else: voice.
            details = self._has_operator_details(o.message_to_manager, include_name=after_operator)
            if reply_voice and not o.is_recap and not details:
                sent_voice = await self._send_voice_to_manager(o.message_to_manager)
            if not sent_voice:  # text reply, recap, or voice failed
                await self._send_to_manager(o.message_to_manager)
        if o.operator_question:
            qid = await self.operator.ask(o.operator_question, context)
            if self.state.pending_question:
                self.state.pending_question.tg_msg_id = qid
        if o.finished:
            b = self.state.booking
            summary = (f"✅ Done. Booking status: {b.status}. {b.restaurant or '?'}, "
                       f"{b.date}, {b.time}, {b.party_size} people.")
            await self.operator.notify(summary)
            log.info(summary)

    async def _send_to_manager(self, text: str) -> None:
        if self.s.typing_simulation:
            delay = min(1.0 + len(text) / 35, 7.0)  # roughly human typing speed
            try:
                async with self.client.action(self.manager, "typing"):
                    await asyncio.sleep(delay)
            except Exception as e:
                log.debug("typing action failed: %s", e)
        msg = await send_text(self.client, self.manager, text)
        # attach the Telegram id to the history entry the core just added
        for e in reversed(self.state.history):
            if e.role == Role.agent and e.text == text and e.tg_msg_id is None:
                e.tg_msg_id = msg.id
                break

    async def _send_voice_to_manager(self, text: str) -> bool:
        """Send `text` as a voice note. Returns False (caller falls back to text) on failure."""
        try:
            with temp_dir() as d:
                async with self.client.action(self.manager, "record-audio"):
                    ogg, duration = await synthesize_voice_note(text, self.s.tts_voice, Path(d))
                msg = await self.client.send_file(
                    self.manager, str(ogg), voice_note=True,
                    attributes=[DocumentAttributeAudio(duration=duration, voice=True)],
                )
        except Exception as e:
            log.error("voice reply failed, falling back to text: %s", e)
            self.dlog.note(self.agent.now(), f"voice reply failed ({e}); sent as text")
            return False
        for e in reversed(self.state.history):
            if e.role == Role.agent and e.text == text and e.tg_msg_id is None:
                e.tg_msg_id = msg.id
                e.text = f"{text} (sent as a voice message)"
                break
        return True

    # --- silence watchdog ------------------------------------------------------

    async def _watch_silence(self) -> None:
        while True:
            await asyncio.sleep(30)
            st = self.state
            if st.booking.status in _DONE or st.pending_question or st.followup_sent or not st.history:
                continue
            last = st.history[-1]
            if last.role != Role.agent:
                continue
            minutes = (self.agent.now() - last.at).total_seconds() / 60
            if minutes >= self.s.followup_after_min:
                st.followup_sent = True
                await self.queue.put(Event(
                    "system",
                    f"The manager has not replied for {minutes:.0f} minutes. "
                    "Decide whether one short, polite follow-up is appropriate.",
                ))


def _describe(m) -> str:
    """Text of a Telegram message; non-text content becomes a [bracketed] note."""
    text = (m.raw_text or "").strip()
    kind = None
    if m.voice:
        kind = "voice message"
    elif m.video_note or m.video:
        kind = "video"
    elif m.sticker:
        alt = getattr(m.file, "emoji", None) if m.file else None
        kind = f"sticker {alt}" if alt else "sticker"
    elif m.photo:
        kind = "photo"
    elif m.geo:
        kind = "location"
    elif m.contact:
        kind = "contact card"
    elif m.media:
        kind = "attachment"
    if kind:
        return f"{text} [{kind}]".strip()
    return text


def now_factory(s: Settings):
    tz = s.tz
    return lambda: datetime.now(tz)
