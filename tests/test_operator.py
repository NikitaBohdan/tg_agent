import asyncio
from types import SimpleNamespace

from agent.operator import SavedMessagesOperator


def _setup(open_id):
    answers, notices = [], []

    async def on_answer(text):
        answers.append(text)

    op = SavedMessagesOperator(client=None, on_answer=on_answer, open_question_id=lambda: open_id)

    async def notify(text):
        notices.append(text)

    op.notify = notify
    return op, answers, notices


def _event(text, reply_to=None, id_=500):
    return SimpleNamespace(message=SimpleNamespace(id=id_, raw_text=text, reply_to_msg_id=reply_to))


def test_reply_to_open_question_is_used():
    op, answers, _ = _setup(open_id=10)
    asyncio.run(op._on_saved(_event("Nikita", reply_to=10)))
    assert answers == ["Nikita"]


def test_reply_to_closed_question_is_rejected():
    op, answers, notices = _setup(open_id=20)
    op._own_ids.add(10)  # 10 was an earlier question of the agent, now closed
    asyncio.run(op._on_saved(_event("No, 6pm is fine", reply_to=10)))
    assert answers == [] and "already closed" in notices[0]


def test_message_without_open_question_is_an_instruction():
    # Final run: the operator could not step in after the agent wrongly closed the chat.
    op, answers, notices = _setup(open_id=None)
    asyncio.run(op._on_saved(_event("We did not cancel. Ask them to reinstate.")))
    assert answers == ["We did not cancel. Ask them to reinstate."]
    assert "instruction received" in notices[0]


def test_agent_own_messages_ignored():
    op, answers, notices = _setup(open_id=10)
    asyncio.run(op._on_saved(_event("🤖 QUESTION ...")))
    assert answers == [] and notices == []
