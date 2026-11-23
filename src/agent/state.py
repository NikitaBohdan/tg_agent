"""Conversation memory persisted to disk so a restart never loses the dialogue.

- Goal:    what we want (fixed, from config + London date).
- Booking: what the manager has stated so far (updated from AgentTurn.state_update).
- Facts:   operator-provided facts (guest_name, phone, ...) — cached, never asked twice.
- History: transcript of manager / agent / operator events.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from .schema import StateUpdate, Status

log = logging.getLogger(__name__)


class Role(StrEnum):
    manager = "manager"
    agent = "agent"
    operator = "operator"  # answer from the human operator
    system = "system"  # internal events: start, timeout, guard notes


class Entry(BaseModel):
    role: Role
    text: str
    at: datetime
    tg_msg_id: int | None = None


class Goal(BaseModel):
    date: str  # YYYY-MM-DD in restaurant tz
    time: str  # HH:MM
    party_size: int
    tz: str


class Booking(BaseModel):
    restaurant: str | None = None
    date: str | None = None
    time: str | None = None
    party_size: int | None = None
    status: Status = Status.not_started
    notes: list[str] = Field(default_factory=list)

    def apply(self, upd: StateUpdate) -> None:
        for f in ("restaurant", "date", "time", "party_size", "status"):
            v = getattr(upd, f)
            if v is not None:
                setattr(self, f, v)
        if upd.notes and upd.notes not in self.notes:
            self.notes.append(upd.notes)


class PendingQuestion(BaseModel):
    question: str
    keys: list[str] = Field(default_factory=list)
    tg_msg_id: int | None = None  # id of the question in Saved Messages
    asked_at: datetime


class ConversationState(BaseModel):
    goal: Goal
    booking: Booking = Field(default_factory=Booking)
    facts: dict[str, str] = Field(default_factory=dict)
    history: list[Entry] = Field(default_factory=list)
    recap_sent: bool = False
    pending_question: PendingQuestion | None = None
    followup_sent: bool = False
    manager_uses_voice: bool = False  # modality of the manager's latest message
    # Every operator answer, verbatim: standing decisions such as "only 5pm works".
    operator_instructions: list[str] = Field(default_factory=list)
    # The agent told the manager it would come back after checking with the operator.
    owes_manager_reply: bool = False

    def last_manager_msg_id(self) -> int:
        ids = [e.tg_msg_id for e in self.history if e.role == Role.manager and e.tg_msg_id]
        return max(ids, default=0)

    def add(self, role: Role, text: str, at: datetime, tg_msg_id: int | None = None) -> Entry:
        e = Entry(role=role, text=text, at=at, tg_msg_id=tg_msg_id)
        self.history.append(e)
        return e


class StateStore:
    """JSON file per conversation, written atomically (tmp + replace)."""

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> ConversationState | None:
        if not self.path.exists():
            return None
        try:
            return ConversationState.model_validate_json(self.path.read_text("utf-8"))
        except (OSError, ValueError) as e:
            log.error("State file %s unreadable (%s); starting fresh", self.path, e)
            return None

    def save(self, state: ConversationState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2), "utf-8")
        os.replace(tmp, self.path)
