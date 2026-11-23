"""Data contracts: what the LLM must return (AgentTurn) and what the agent remembers.

AgentTurn is validated with Pydantic; anything that does not fit is rejected and
the LLM gets one chance to repair it (see brain.py). The decision log is built
from these fields, so the log is written by the agent itself.
"""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Intent(StrEnum):
    none = "none"  # no manager message (e.g. opening, operator answer)
    greeting = "greeting"
    question = "question"
    request_info = "request_info"  # manager asks for name/phone/etc.
    offer_alternative = "offer_alternative"  # different time/date/size
    confirm = "confirm"
    reject = "reject"
    smalltalk = "smalltalk"
    off_topic = "off_topic"
    unclear = "unclear"


class Action(StrEnum):
    send_message = "send_message"
    ask_operator = "ask_operator"
    send_and_ask_operator = "send_and_ask_operator"
    wait = "wait"
    finish = "finish"


class Status(StrEnum):
    not_started = "not_started"
    negotiating = "negotiating"
    pending_confirmation = "pending_confirmation"
    confirmed = "confirmed"
    declined = "declined"
    abandoned = "abandoned"


def norm_key(k: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", k.strip().lower()).strip("_")


_TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class StateUpdate(BaseModel):
    """Only facts stated by the MANAGER in the conversation. null = no change."""

    model_config = ConfigDict(extra="forbid")

    restaurant: str | None = None
    date: str | None = Field(None, description="YYYY-MM-DD")
    time: str | None = Field(None, description="HH:MM, 24h, restaurant local time")
    party_size: int | None = Field(None, ge=1, le=50)
    status: Status | None = None
    notes: str | None = Field(None, description="Conditions from the manager, e.g. table back by 19:00")

    @field_validator("time")
    @classmethod
    def _time(cls, v: str | None) -> str | None:
        if v is not None and not _TIME_RE.match(v):
            raise ValueError("time must be HH:MM (24h)")
        return v

    @field_validator("date")
    @classmethod
    def _date(cls, v: str | None) -> str | None:
        if v is not None and not _DATE_RE.match(v):
            raise ValueError("date must be YYYY-MM-DD")
        return v


class Fact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(description="snake_case, e.g. guest_name, phone")
    value: str = Field(description="copied verbatim from the operator's answer")


class AgentTurn(BaseModel):
    """One decision of the agent. Must be returned by the LLM as JSON."""

    model_config = ConfigDict(extra="forbid")

    understood: str = Field(description="What the agent understood from the latest event, one sentence")
    manager_intent: Intent
    state_update: StateUpdate = Field(default_factory=StateUpdate)
    missing_facts: list[str] = Field(
        default_factory=list,
        description="snake_case keys of facts the agent needs but does not know, e.g. guest_name, phone",
    )
    facts_from_operator: list[Fact] = Field(
        default_factory=list,
        description="Only when the latest event is an operator answer: each fact it contains, value verbatim",
    )
    resolved_operator_question: bool = Field(
        default=False,
        description="True if the currently open operator question is no longer needed",
    )
    decision: str = Field(description="What the agent decided and why, one sentence")
    action: Action
    message_to_manager: str | None = None
    operator_question: str | None = None

    @field_validator("manager_intent", mode="before")
    @classmethod
    def _lenient_intent(cls, v):
        # Intent is informational (log only); an invented label like "statement"
        # should not cost a whole repair round with the LLM.
        return v if v in Intent._value2member_map_ else Intent.unclear

    @field_validator("missing_facts")
    @classmethod
    def _norm_keys(cls, v: list[str]) -> list[str]:
        out = []
        for k in v:
            k = norm_key(k)
            if k and k not in out:
                out.append(k)
        return out

    @model_validator(mode="after")
    def _action_payload(self) -> AgentTurn:
        sends = self.action in (Action.send_message, Action.send_and_ask_operator)
        asks = self.action in (Action.ask_operator, Action.send_and_ask_operator)
        if sends and not (self.message_to_manager or "").strip():
            raise ValueError(f"action={self.action} requires message_to_manager")
        if asks and not (self.operator_question or "").strip():
            raise ValueError(f"action={self.action} requires operator_question")
        if not sends:
            self.message_to_manager = None
        if not asks:
            self.operator_question = None
        return self
