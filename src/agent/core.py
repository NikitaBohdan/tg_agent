"""IO-free agent core: takes an event, asks the Brain, applies code guards, returns an Outcome.

Shared by the console simulator and the Telegram runner, so both behave
identically. The LLM proposes; this module enforces the rules that must not
depend on the model:

- never ask the operator for a fact that is already known;
- confirmed booking fields default to the goal (the agent may accept nothing
  else without the operator, see the prompt), so the record is never empty;
- the final confirmation is rendered from state by a template, so the details
  repeated to the manager cannot be hallucinated;
- the agent cannot "finish" a confirmed booking before that recap is sent.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime

from .brain import Brain
from .schema import Action, AgentTurn, Intent, Status, norm_key
from .state import ConversationState, PendingQuestion, Role
from .timeutil import human_date

log = logging.getLogger(__name__)

_NUMBERS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}


@dataclass
class Outcome:
    turn: AgentTurn
    message_to_manager: str | None = None
    operator_question: str | None = None
    finished: bool = False
    guard_notes: list[str] = field(default_factory=list)
    provider: str | None = None
    is_recap: bool = False  # message is the template confirmation summary
    closed_question: str | None = None  # open operator question that no longer needs an answer


class Agent:
    def __init__(self, brain: Brain, state: ConversationState, now_fn: Callable[[], datetime]):
        self.brain = brain
        self.state = state
        self.now = now_fn
        self._last_event: str | None = None
        self._operator_answer: tuple[list[str], str] | None = None

    # --- events ------------------------------------------------------------
    # Each method records the event in history exactly once, then decides.
    # If deciding fails (BrainError), call retry() — it does not re-record.

    async def start(self) -> Outcome:
        return await self._handle(
            "START: the conversation has not started. Write your opening message to the manager."
        )

    async def on_manager_message(self, text: str, tg_msg_id: int | None = None) -> Outcome:
        self.state.add(Role.manager, text, self.now(), tg_msg_id)
        self.state.followup_sent = False
        return await self._handle(f"MANAGER wrote: {text}")

    async def on_operator_answer(self, answer: str) -> Outcome:
        pq = self.state.pending_question
        keys = pq.keys if pq and pq.keys else ["operator_note"]
        question = pq.question if pq else "(no open question)"
        # Raw answer under the requested key(s) as a fallback; the model then splits it
        # into individual facts (facts_from_operator), verified verbatim in _handle.
        self.state.facts["+".join(keys)] = answer
        self.state.pending_question = None
        # Kept for the whole conversation, so a decision like "only 5pm works" is
        # applied to the next similar offer instead of asking the operator again.
        self.state.operator_instructions.append(answer)
        self.state.add(Role.operator, answer, self.now())
        self._operator_answer = (keys, answer)  # consumed by _handle (also on retry)
        if pq is None:  # the operator stepped in on their own
            return await self._handle(
                f"OPERATOR sent you an instruction (not an answer to a question): {answer}"
            )
        return await self._handle(f"OPERATOR answered your question ({question!r}): {answer}")

    async def on_system_event(self, text: str) -> Outcome:
        self.state.add(Role.system, text, self.now())
        return await self._handle(f"SYSTEM: {text}")

    async def retry(self) -> Outcome:
        if self._last_event is None:
            raise RuntimeError("nothing to retry")
        return await self._handle(self._last_event)

    # --- core --------------------------------------------------------------

    async def _handle(self, event: str) -> Outcome:
        self._last_event = event
        notes: list[str] = []
        turn = await self.brain.decide(self.state, event)

        # Guard 1: never ask the operator twice for a known fact.
        if turn.operator_question and turn.missing_facts and self._all_known(turn.missing_facts):
            notes.append(f"blocked repeat question for known facts {turn.missing_facts}")
            known = {k: v for k, v in self.state.facts.items() if self._covers(k, turn.missing_facts)}
            turn = await self.brain.decide(
                self.state,
                f"{event}\nSYSTEM: you already know {known}. Do NOT ask the operator; use these facts.",
            )
            if turn.operator_question and self._all_known(turn.missing_facts):
                notes.append("model insisted; operator question dropped")
                turn.operator_question = None
                turn.action = Action.send_message if turn.message_to_manager else Action.wait

        # Guard 5: an unconfirmed booking must not stall. Rehearsal: the manager only
        # gave the restaurant name, the agent replied "Thank you ever so much." and
        # the chat ended with no confirmation and no recap.
        if self._stalls(turn, event):
            notes.append("reply would stall an unconfirmed booking; asked model to steer to confirmation")
            turn = await self.brain.decide(
                self.state,
                f"{event}\nSYSTEM: the booking is NOT confirmed yet and your reply asks nothing, so the "
                "chat may stall. If the manager asked you to wait or promised to get back to you, or "
                "you already made this request in one of your last messages, keep your reply as it "
                "is (or use action wait). Otherwise restate the terms (restaurant if known, today, "
                "5pm, two people, guest name if known) and ask the manager to confirm the booking.",
            )

        # Guard 6: the agent told the manager it would come back; after the operator
        # answers, it must actually reply. Live test: "I'll come back to you in a
        # moment", then the operator said "Only 5pm" and the agent stayed silent.
        if (
            self._operator_answer
            and self.state.owes_manager_reply
            and not turn.message_to_manager
            and not turn.operator_question
        ):
            notes.append("manager was promised a reply; asked model to answer now")
            turn = await self.brain.decide(
                self.state,
                f"{event}\nSYSTEM: you told the manager you would come back to them after "
                "checking. The operator has answered, so reply to the manager NOW (action "
                "send_message), applying the operator's instructions.",
            )

        # Guard 8: never cancel, give up or end the conversation without the operator.
        # Final run: the manager claimed "your client just rang us and cancelled"; the
        # agent believed it, marked the booking abandoned and finished in silence.
        if not event.startswith("OPERATOR") and self._gives_up(turn):
            notes.append("tried to cancel/close without the operator; asked model to check first")
            turn = await self.brain.decide(
                self.state,
                f"{event}\nSYSTEM: you may NOT accept a cancellation, give up or end the "
                "conversation on your own. Do not change the booking status. Ask the operator "
                "what to do, saying exactly what the manager claimed, and tell the manager "
                "politely that you'll check and come back (action send_and_ask_operator).",
            )
            if self._gives_up(turn) or not turn.operator_question:
                notes.append("model still did not ask; fallback question to the operator")
                turn = AgentTurn(
                    understood=turn.understood,
                    manager_intent=turn.manager_intent,
                    decision="Closing or cancelling needs the operator's decision.",
                    action=Action.send_and_ask_operator,
                    message_to_manager="I'm sorry, I think there may be a misunderstanding. "
                    "Let me just check with the guest and come back to you.",
                    operator_question="The manager's message would cancel or end the booking. "
                    "Is it true? How should I respond?",
                )

        notes += self._store_operator_facts(turn)
        self._operator_answer = None

        b = self.state.booking
        prev_status = b.status
        b.apply(turn.state_update)
        # A booking confirmed again (e.g. reinstated after a dispute) gets a fresh recap.
        if b.status == Status.confirmed and prev_status != Status.confirmed:
            self.state.recap_sent = False

        # Guard 2a: when the model marks the booking confirmed but leaves the agreed terms
        # empty, ask it to state them. Filling them from the goal would be wrong when the
        # operator approved a different time (e.g. 17:30), and the recap would lie.
        if turn.state_update.status == Status.confirmed and (b.time is None or b.date is None):
            notes.append("confirmed without recorded terms; asked model to state them")
            fix = await self.brain.decide(
                self.state,
                f"{event}\nSYSTEM: you marked the booking confirmed but did not record the AGREED "
                "terms. Return the same decision with state_update.status='confirmed' and the "
                "date (YYYY-MM-DD), time (HH:MM) and party_size the manager actually agreed to.",
            )
            b.apply(fix.state_update)
            turn = fix

        # Guard 2b: a confirmed booking always has full terms recorded.
        if b.status == Status.confirmed:
            g = self.state.goal
            for f, v in (("date", g.date), ("time", g.time), ("party_size", g.party_size)):
                if getattr(b, f) is None:
                    setattr(b, f, v)
                    notes.append(f"filled confirmed {f} from goal")

        msg = turn.message_to_manager
        is_recap = False
        # Guard 3: deterministic recap once confirmed AND the restaurant is known.
        if b.status == Status.confirmed and b.restaurant and not self.state.recap_sent:
            msg = self.render_recap()
            is_recap = True
            self.state.recap_sent = True
            notes.append("confirmation recap rendered from state")

        finished = turn.action == Action.finish
        if finished and b.status == Status.confirmed and not self.state.recap_sent:
            finished = False
            notes.append("finish blocked: recap not sent yet (restaurant unknown)")

        if msg:
            self.state.add(Role.agent, msg, self.now())

        # Guard 4: an open operator question that became irrelevant is closed, so a
        # late answer is not applied to the wrong context and no stale reminder is sent.
        closed = None
        old = self.state.pending_question
        if old and turn.operator_question and turn.operator_question != old.question:
            closed = old.question
            notes.append("previous operator question superseded by a new one")
        elif old and turn.resolved_operator_question and not turn.operator_question:
            closed = old.question
            self.state.pending_question = None
            notes.append("operator question closed: no longer needed")

        if turn.operator_question:
            self.state.pending_question = PendingQuestion(
                question=turn.operator_question, keys=turn.missing_facts, asked_at=self.now()
            )
            self.state.owes_manager_reply = True  # the manager now waits for our answer
        elif msg:
            self.state.owes_manager_reply = False
        for n in notes:
            log.info("guard: %s", n)
        return Outcome(
            turn=turn,
            message_to_manager=msg,
            operator_question=turn.operator_question,
            finished=finished,
            guard_notes=notes,
            provider=self.brain.last_provider,
            is_recap=is_recap,
            closed_question=closed,
        )

    def render_recap(self) -> str:
        b, g = self.state.booking, self.state.goal
        d = date.fromisoformat(b.date or g.date)
        day = human_date(datetime.combine(d, datetime.min.time())).rsplit(" ", 1)[0]  # no year
        today = "today, " if d.isoformat() == g.date else ""
        size = b.party_size or g.party_size
        people = _NUMBERS.get(size, str(size))
        parts = [f"a table for {people} at {b.restaurant}, {today}{day}, at {_time12(b.time or g.time)}"]
        name = self._fact("guest_name") or self._fact("name") or self._fact("booking_name")
        if name:
            parts.append(f"under the name {name}")
        # Only the agreed terms: model-written notes ("5pm fully booked; offered…")
        # cluttered the final run's recap, so they stay in state and the log only.
        recap = "Perfect. Just to confirm: " + ", ".join(parts) + "."
        return recap + " Thank you, we look forward to it."

    # --- helpers -----------------------------------------------------------

    def _gives_up(self, turn: AgentTurn) -> bool:
        """The turn would cancel, abandon or end the conversation. Ending is fine only
        after a confirmed booking whose recap was sent (normal goodbye)."""
        new = turn.state_update.status
        if new in (Status.declined, Status.abandoned):
            return True
        if turn.action == Action.finish:
            # Finishing a confirmed booking is a normal goodbye (the recap guard still
            # holds the finish until the recap is sent); anything else is giving up.
            confirmed = self.state.booking.status == Status.confirmed or new == Status.confirmed
            return not (confirmed and new in (None, Status.confirmed))
        return False

    def _stalls(self, turn: AgentTurn, event: str) -> bool:
        status = turn.state_update.status or self.state.booking.status
        return (
            status not in (Status.confirmed, Status.declined, Status.abandoned)
            and turn.action == Action.send_message
            and not turn.operator_question
            and self.state.pending_question is None
            and "?" not in (turn.message_to_manager or "")
            and event.startswith(("MANAGER wrote", "OPERATOR answered"))
            # answering the manager's own question ("are you a bot?") is not a stall
            and turn.manager_intent not in (Intent.question, Intent.smalltalk, Intent.off_topic)
        )

    def _store_operator_facts(self, turn: AgentTurn) -> list[str]:
        """Guard: keep only facts whose value literally appears in the operator's answer."""
        if not self._operator_answer or not turn.facts_from_operator:
            return []
        keys, answer = self._operator_answer
        haystack = _norm(answer)
        notes, stored = [], set()
        for f in turn.facts_from_operator:
            key, value = norm_key(f.key), f.value.strip()
            if key and value and _norm(value) in haystack:
                self.state.facts[key] = value
                stored.add(key)
            else:
                notes.append(f"rejected fact {key}={value!r}: not in operator answer")
        combined = "+".join(keys)
        if len(keys) > 1 and set(keys) <= stored:
            self.state.facts.pop(combined, None)  # fully split; drop the raw blob
        return notes

    @staticmethod
    def _covers(fact_key: str, wanted: list[str]) -> bool:
        return any(w in fact_key.split("+") for w in wanted)

    def _all_known(self, keys: list[str]) -> bool:
        stored = [k for fk in self.state.facts for k in fk.split("+")]
        return all(k in stored for k in keys)

    def _fact(self, key: str) -> str | None:
        if key in self.state.facts:
            return self.state.facts[key]
        for fk, v in self.state.facts.items():
            if key in fk.split("+"):
                return v
        return None


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def _time12(hhmm: str) -> str:
    h, m = map(int, hhmm.split(":"))
    suffix = "am" if h < 12 else "pm"
    h12 = h % 12 or 12
    return f"{h12}{suffix}" if m == 0 else f"{h12}:{m:02d}{suffix}"
