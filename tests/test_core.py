"""Agent-core guards, replaying the exact failure seen in the first simulation."""

import asyncio
import json
from datetime import datetime
from zoneinfo import ZoneInfo

from agent.brain import Brain
from agent.core import Agent, _time12
from agent.schema import Status
from agent.state import ConversationState, Goal

TZ = ZoneInfo("Europe/London")
NOW = datetime(2026, 11, 21, 14, 0, tzinfo=TZ)


def turn(**kw):
    base = {
        "understood": "u", "manager_intent": "none", "state_update": {}, "missing_facts": [],
        "decision": "d", "action": "send_message", "message_to_manager": "Thank you. Could you confirm?",
        "operator_question": None,
    }
    base.update(kw)
    return json.dumps(base)


class Scripted:
    name = "scripted"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    async def complete_json(self, system, user, schema):
        self.calls += 1
        return self.outputs.pop(0)


def make(outputs):
    st = ConversationState(goal=Goal(date="2026-11-21", time="17:00", party_size=2, tz="Europe/London"))
    p = Scripted(outputs)
    return Agent(Brain([p], now_fn=lambda: NOW), st, lambda: NOW), p


def run(c):
    return asyncio.run(c)


def test_recap_sent_when_name_arrives_even_if_model_says_finish():
    # Replay of the first sim: model confirmed without time/date (even when asked again),
    # then finished silently. Terms fall back to the goal only after the re-ask.
    confirm = turn(manager_intent="confirm", state_update={"status": "confirmed", "party_size": 2},
                   message_to_manager="Lovely — could you remind me of the restaurant's name?")
    agent, _ = make([
        confirm, confirm,
        turn(manager_intent="none", state_update={"restaurant": "The Ivy, Covent Garden"},
             action="finish", message_to_manager=None),
    ])
    agent.state.facts["guest_name"] = "Nikita"

    o1 = run(agent.on_manager_message("You're all booked for 5pm today, table for two!"))
    assert not agent.state.recap_sent and "name" in o1.message_to_manager
    assert agent.state.booking.time == "17:00" and agent.state.booking.date == "2026-11-21"

    o2 = run(agent.on_manager_message("This is The Ivy, Covent Garden."))
    assert o2.finished and agent.state.recap_sent
    assert o2.message_to_manager == (
        "Perfect. Just to confirm: a table for two at The Ivy, Covent Garden, "
        "today, Saturday 21 November, at 5pm, under the name Nikita. Thank you, we look forward to it."
    )


def test_confirmed_alternative_time_is_not_overwritten_by_goal():
    # Final run: operator approved 17:30; the model confirmed without recording the time.
    agent, p = make([
        turn(manager_intent="confirm", state_update={"status": "confirmed", "restaurant": "Dishoom"}),
        turn(manager_intent="confirm", state_update={"status": "confirmed", "restaurant": "Dishoom",
                                                     "date": "2026-11-21", "time": "17:30", "party_size": 2}),
    ])
    o = run(agent.on_manager_message("Yes, half five is booked for you."))
    assert p.calls == 2 and agent.state.booking.time == "17:30"
    assert "at 5:30pm" in o.message_to_manager


def test_finish_blocked_until_restaurant_known():
    t = turn(state_update={"status": "confirmed", "date": "2026-11-21", "time": "17:00"},
             action="finish", message_to_manager=None)
    agent, _ = make([t])
    o = run(agent.on_manager_message("Booked!"))
    assert not o.finished and agent.state.booking.status == Status.confirmed
    assert any("finish blocked" in n for n in o.guard_notes)


def test_operator_not_asked_twice_for_known_fact():
    ask = dict(action="send_and_ask_operator", missing_facts=["guest_name"],
               message_to_manager="One moment please.", operator_question="Guest name?")
    agent, p = make([turn(**ask), turn(**ask), turn()])  # insists twice, then steered to confirm
    agent.state.facts["guest_name"] = "Nikita"
    o = run(agent.on_manager_message("Name again, please?"))
    assert o.operator_question is None and p.calls == 3
    assert any("dropped" in n for n in o.guard_notes)


def test_operator_answer_stored_under_requested_key():
    agent, _ = make([
        turn(action="send_and_ask_operator", missing_facts=["phone"],
             message_to_manager="One moment.", operator_question="Phone?"),
        turn(),
    ])
    run(agent.on_manager_message("Phone number?"))
    assert agent.state.pending_question.keys == ["phone"]
    run(agent.on_operator_answer("+44 20 0000 0000"))
    assert agent.state.facts == {"phone": "+44 20 0000 0000"}
    assert agent.state.pending_question is None


def test_retry_does_not_duplicate_history():
    from agent.llm import LLMUnavailable

    class Flaky(Scripted):
        async def complete_json(self, system, user, schema):
            self.calls += 1
            out = self.outputs.pop(0)
            if isinstance(out, Exception):
                raise out
            return out

    st = ConversationState(goal=Goal(date="2026-11-21", time="17:00", party_size=2, tz="Europe/London"))

    async def nosleep(_):
        return None

    p = Flaky([LLMUnavailable("x")] * 3 + [turn()])
    agent = Agent(Brain([p], now_fn=lambda: NOW, sleep=nosleep), st, lambda: NOW)
    import pytest

    from agent.brain import BrainError

    with pytest.raises(BrainError):
        run(agent.on_manager_message("Hello"))
    run(agent.retry())
    assert [e.role for e in st.history] == ["manager", "agent"]


def test_combined_operator_answer_split_into_verified_facts():
    # Replay of the live run: "Nikita +447700900123" ended up in the recap as the name.
    agent, _ = make([
        turn(action="send_and_ask_operator", missing_facts=["guest_name", "phone"],
             message_to_manager="One moment.", operator_question="Name and phone?"),
        turn(facts_from_operator=[{"key": "guest_name", "value": "Nikita"},
                                  {"key": "phone", "value": "+447700900123"},
                                  {"key": "email", "value": "made@up.com"}]),
        turn(manager_intent="confirm", state_update={"status": "confirmed", "restaurant": "Dishoom, Shoreditch",
                                                     "date": "2026-11-21", "time": "17:00", "party_size": 2}),
    ])
    run(agent.on_manager_message("Name and number?"))
    o = run(agent.on_operator_answer("Nikita +447700900123"))
    assert agent.state.facts == {"guest_name": "Nikita", "phone": "+447700900123"}
    assert any("rejected fact email" in n for n in o.guard_notes)
    o = run(agent.on_manager_message("All booked!"))
    assert "under the name Nikita." in o.message_to_manager


def test_stale_operator_question_closed_when_resolved():
    # Live test: manager offered 6pm (operator asked), then edited it to 5pm.
    agent, _ = make([
        turn(action="send_and_ask_operator", message_to_manager="One moment.",
             operator_question="They offer 6pm instead of 5pm. Accept?"),
        turn(resolved_operator_question=True, message_to_manager="Perfect, 5pm please."),
    ])
    run(agent.on_manager_message("We have a table at 6pm today"))
    o = run(agent.on_system_event('The manager edited an earlier message; it now reads: "5pm"'))
    assert agent.state.pending_question is None
    assert o.closed_question == "They offer 6pm instead of 5pm. Accept?"


def test_new_operator_question_supersedes_old():
    agent, _ = make([
        turn(action="send_and_ask_operator", message_to_manager="One moment.", operator_question="Accept 6pm?"),
        turn(action="send_and_ask_operator", missing_facts=["guest_name"],
             message_to_manager="One moment.", operator_question="Guest name?"),
    ])
    run(agent.on_manager_message("6pm?"))
    o = run(agent.on_manager_message("Actually 5pm is fine. Name?"))
    assert o.closed_question == "Accept 6pm?" and agent.state.pending_question.question == "Guest name?"


def test_unconfirmed_booking_does_not_stall():
    # Rehearsal: manager only replied "Burger King"; agent said "Thank you ever so much." and stopped.
    agent, p = make([
        turn(state_update={"restaurant": "Burger King"}, message_to_manager="Thank you ever so much."),
        turn(message_to_manager="So that's a table for two at Burger King today at 5pm. Could you confirm?"),
    ])
    o = run(agent.on_manager_message("Burger King"))
    assert p.calls == 2 and o.message_to_manager.endswith("Could you confirm?")
    assert any("stall" in n for n in o.guard_notes)


def test_no_stall_guard_once_confirmed():
    agent, p = make([turn(manager_intent="smalltalk", message_to_manager="Thank you, see you then.")])
    agent.state.booking.status = Status.confirmed
    agent.state.recap_sent = True
    run(agent.on_manager_message("See you!"))
    assert p.calls == 1


def test_unknown_intent_label_does_not_fail_validation():
    from agent.schema import AgentTurn, Intent

    t = AgentTurn.model_validate(json.loads(turn(manager_intent="statement")))
    assert t.manager_intent == Intent.unclear


def test_time12():
    assert [_time12(t) for t in ("17:00", "17:30", "12:00", "00:15")] == ["5pm", "5:30pm", "12pm", "12:15am"]


def test_answering_managers_question_is_not_a_stall():
    # Final run: "am I chatting to a person or a bot?" got the 5pm request appended again.
    agent, p = make([turn(manager_intent="question",
                          message_to_manager="I'm an AI assistant booking on behalf of the guest.")])
    o = run(agent.on_manager_message("Out of curiosity, am I chatting to a person or a bot?"))
    assert p.calls == 1 and not any("stall" in n for n in o.guard_notes)


def test_agent_replies_after_operator_answer_when_manager_waits():
    # Live test: "I'll come back to you in a moment", operator said "Only 5 pm",
    # the model chose to wait and the manager never heard back.
    agent, p = make([
        turn(action="send_and_ask_operator", message_to_manager="I'm just checking, back in a moment.",
             operator_question="Accept half five or six?"),
        turn(action="wait", message_to_manager=None),
        turn(message_to_manager="I'm afraid only 5pm works for us. Do let us know if it opens up?"),
    ])
    run(agent.on_manager_message("5 is gone. It's half five or six."))
    o = run(agent.on_operator_answer("Only 5 pm"))
    assert p.calls == 3 and o.message_to_manager.startswith("I'm afraid only 5pm")
    assert agent.state.owes_manager_reply is False
    assert any("promised a reply" in n for n in o.guard_notes)


def test_operator_answers_kept_as_standing_instructions():
    agent, _ = make([
        turn(action="send_and_ask_operator", missing_facts=["guest_name", "phone"],
             message_to_manager="One moment.", operator_question="Name, phone, accept 5:30?"),
        turn(facts_from_operator=[{"key": "guest_name", "value": "Nikita"},
                                  {"key": "phone", "value": "+447700900123"}]),
    ])
    run(agent.on_manager_message("5 is full, half five? Name and number?"))
    run(agent.on_operator_answer("Nikita, +447700900123. Only 5pm works."))
    # the blob was split into facts, but the decision itself must survive
    assert agent.state.operator_instructions == ["Nikita, +447700900123. Only 5pm works."]
    from agent.brain import build_user_prompt
    assert "Only 5pm works" in build_user_prompt(agent.state, "x", NOW)


def test_every_alternative_offer_goes_to_operator_again():
    # User's rule: each time the manager offers other times, the operator is asked,
    # even if they answered "only 5pm" before. No code guard may suppress the question.
    agent, p = make([
        turn(action="send_and_ask_operator", message_to_manager="Bear with me a moment.",
             operator_question="5pm is gone; 5:30 or 6pm. Earlier you said only 5pm. Still hold?"),
    ])
    agent.state.operator_instructions.append("Only 5pm works, please don't accept other times.")
    o = run(agent.on_manager_message("No, 5 is gone. It's half five or six."))
    assert p.calls == 1 and o.operator_question.startswith("5pm is gone")
    assert agent.state.pending_question is not None


def test_claimed_cancellation_goes_to_operator():
    # Final run: "your client just rang us and cancelled" - the agent believed it,
    # set abandoned and finished silently.
    agent, p = make([
        turn(manager_intent="reject", state_update={"status": "abandoned"}, action="finish",
             message_to_manager=None),
        turn(manager_intent="reject", state_update={"status": "abandoned"}, action="finish",
             message_to_manager=None),  # model insists -> fallback
    ])
    agent.state.booking.status = Status.confirmed
    agent.state.recap_sent = True
    o = run(agent.on_manager_message("Ah, sorry, your client just rang us and cancelled."))
    assert not o.finished and o.operator_question and "misunderstanding" in o.message_to_manager
    assert agent.state.booking.status == Status.confirmed  # status not changed by the claim
    assert any("fallback" in n for n in o.guard_notes)


def test_normal_goodbye_after_recap_is_allowed():
    agent, p = make([turn(manager_intent="smalltalk", action="finish", message_to_manager=None)])
    agent.state.booking.status = Status.confirmed
    agent.state.recap_sent = True
    o = run(agent.on_manager_message("See you at 5!"))
    assert o.finished and p.calls == 1


def test_operator_instruction_without_question_and_recap_after_reinstatement():
    agent, p = make([
        turn(message_to_manager="The guest hasn't cancelled. Could you reinstate the 5:30pm table?"),
        turn(manager_intent="confirm",
             state_update={"status": "confirmed", "date": "2026-11-21", "time": "17:30", "party_size": 2}),
    ])
    agent.state.booking.restaurant = "The Olive Room"
    agent.state.booking.status = Status.abandoned
    agent.state.recap_sent = True  # recap of the first confirmation was sent earlier
    o = run(agent.on_operator_answer("We did not cancel. Ask them to reinstate the booking."))
    assert "reinstate" in o.message_to_manager
    o = run(agent.on_manager_message("Apologies, reinstated."))
    assert o.is_recap and "at 5:30pm" in o.message_to_manager and "Noted" not in o.message_to_manager
