import asyncio
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from agent.brain import Brain, BrainError, load_system_prompt, parse_turn
from agent.llm import LLMConfigError, LLMRateLimited, LLMUnavailable
from agent.schema import Action, AgentTurn, StateUpdate, Status
from agent.state import Booking, ConversationState, Goal, Role, StateStore

TZ = ZoneInfo("Europe/London")
NOW = datetime(2026, 11, 21, 14, 0, tzinfo=TZ)

GOOD = {
    "understood": "Conversation not started",
    "manager_intent": "none",
    "state_update": {},
    "missing_facts": [],
    "decision": "Open politely",
    "action": "send_message",
    "message_to_manager": "Good afternoon!",
    "operator_question": None,
}


def state():
    return ConversationState(goal=Goal(date="2026-11-21", time="17:00", party_size=2, tz="Europe/London"))


class Fake:
    def __init__(self, name, outputs):
        self.name = name
        self.outputs = list(outputs)
        self.calls = 0

    async def complete_json(self, system, user, json_schema):
        self.calls += 1
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


async def _nosleep(_):
    return None


def run(coro):
    return asyncio.run(coro)


# --- schema ---------------------------------------------------------------

def test_send_requires_message():
    with pytest.raises(ValidationError):
        AgentTurn.model_validate({**GOOD, "message_to_manager": ""})


def test_ask_requires_question():
    with pytest.raises(ValidationError):
        AgentTurn.model_validate({**GOOD, "action": "ask_operator", "message_to_manager": None})


def test_payload_cleared_when_not_needed():
    t = AgentTurn.model_validate({**GOOD, "action": "wait", "operator_question": "x"})
    assert t.message_to_manager is None and t.operator_question is None


def test_bad_time_rejected():
    with pytest.raises(ValidationError):
        StateUpdate(time="5pm")


def test_missing_facts_normalized():
    t = AgentTurn.model_validate({**GOOD, "missing_facts": ["Guest Name", "guest_name", "phone"]})
    assert t.missing_facts == ["guest_name", "phone"]


def test_parse_turn_tolerates_fences_and_chatter():
    raw = "Sure!\n```json\n" + json.dumps(GOOD) + "\n```"
    assert parse_turn(raw).action == Action.send_message


# --- state ----------------------------------------------------------------

def test_booking_apply_only_non_null():
    b = Booking(time="17:00")
    b.apply(StateUpdate(restaurant="The Ivy", status=Status.negotiating))
    assert (b.restaurant, b.time, b.status) == ("The Ivy", "17:00", Status.negotiating)


def test_state_store_roundtrip(tmp_path):
    st = state()
    st.facts["guest_name"] = "Nikita"
    st.add(Role.manager, "Hello", NOW)
    store = StateStore(tmp_path / "s.json")
    store.save(st)
    assert store.load() == st


def test_corrupt_state_file_returns_none(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("{broken", "utf-8")
    assert StateStore(p).load() is None


def test_system_prompt_renders_goal():
    p = load_system_prompt(state())
    assert "Saturday 21 November 2026" in p and "17:00" in p and "{" not in p.split("# Output")[0]


# --- brain resilience -------------------------------------------------------

def test_repair_after_invalid_output():
    p = Fake("p", ["not json at all", json.dumps(GOOD)])
    brain = Brain([p], now_fn=lambda: NOW, sleep=_nosleep)
    assert run(brain.decide(state(), "START")).message_to_manager == "Good afternoon!"
    assert p.calls == 2


def test_fallback_on_long_rate_limit():
    p1 = Fake("p1", [LLMRateLimited("429", retry_after=600)])
    p2 = Fake("p2", [json.dumps(GOOD)])
    brain = Brain([p1, p2], now_fn=lambda: NOW, sleep=_nosleep)
    run(brain.decide(state(), "START"))
    assert brain.last_provider == "p2" and p1.calls == 1


def test_retry_on_network_error_then_success():
    p = Fake("p", [LLMUnavailable("timeout"), json.dumps(GOOD)])
    brain = Brain([p], now_fn=lambda: NOW, sleep=_nosleep)
    run(brain.decide(state(), "START"))
    assert p.calls == 2


def test_config_error_skips_provider_without_retry():
    p1 = Fake("p1", [LLMConfigError("404 model retired")])
    p2 = Fake("p2", [json.dumps(GOOD)])
    brain = Brain([p1, p2], now_fn=lambda: NOW, sleep=_nosleep)
    run(brain.decide(state(), "START"))
    assert p1.calls == 1 and brain.last_provider == "p2"


def test_all_failed_raises_brain_error():
    p = Fake("p", [LLMUnavailable("down")] * 3)
    brain = Brain([p], now_fn=lambda: NOW, sleep=_nosleep)
    with pytest.raises(BrainError):
        run(brain.decide(state(), "START"))


def test_prompt_keeps_ai_disclosure_honest():
    p = load_system_prompt(state())
    assert "Never claim or imply that you are a human" in p
    assert "Yes, I'm Nikita's AI assistant" in p


def test_prompt_no_self_introduction_unless_asked():
    p = load_system_prompt(state())
    assert "Do not introduce yourself or say whose assistant you are unless the manager asks" in p
