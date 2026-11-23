"""Runner wiring with a fake Telegram client (no network)."""

import asyncio
import json
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from agent.brain import Brain
from agent.config import Settings
from agent.core import Agent
from agent.decision_log import DecisionLog
from agent.runner import Event, Runner, _describe
from agent.state import ConversationState, Goal, StateStore

TZ = ZoneInfo("Europe/London")
NOW = datetime(2026, 11, 21, 14, 0, tzinfo=TZ)


def turn(**kw):
    base = {"understood": "u", "manager_intent": "none", "state_update": {}, "missing_facts": [],
            "decision": "d", "action": "send_message", "message_to_manager": "Good afternoon!",
            "operator_question": None}
    base.update(kw)
    return json.dumps(base)


class Scripted:
    name = "scripted"

    def __init__(self, outputs):
        self.outputs = list(outputs)

    async def complete_json(self, *a):
        return self.outputs.pop(0)


class FakeClient:
    def __init__(self):
        self.sent = []

    async def send_message(self, peer, text):
        self.sent.append((peer, text))
        return SimpleNamespace(id=100 + len(self.sent))

    async def send_read_acknowledge(self, peer):
        pass

    def add_event_handler(self, *a):
        pass


def make(tmp_path, outputs, voice=False, voice_mode="mirror"):
    s = Settings(tg_api_id=1, tg_api_hash="x", manager_username="m", operator_channel="console",
                 typing_simulation=False, voice_enabled=voice, voice_mode=voice_mode,
                 log_dir=tmp_path / "logs", data_dir=tmp_path, _env_file=None)
    st = ConversationState(goal=Goal(date="2026-11-21", time="17:00", party_size=2, tz="Europe/London"))
    agent = Agent(Brain([Scripted(outputs)], now_fn=lambda: NOW), st, lambda: NOW)
    client = FakeClient()
    manager = SimpleNamespace(id=42)
    r = Runner(s, client, manager, agent, StateStore(tmp_path / "s.json"), DecisionLog(s.log_dir))
    asked = []

    async def ask(q, context=None):
        asked.append(q)
        r.asked_context = context
        return None

    r.operator.ask = ask
    return r, client, asked


def test_start_sends_opening_and_logs(tmp_path):
    r, client, _ = make(tmp_path, [turn()])
    asyncio.run(r._process(Event("start")))
    assert client.sent == [(r.manager, "Good afternoon!")]
    assert r.state.history[-1].tg_msg_id == 101
    md = (tmp_path / "logs" / "decisions.md").read_text("utf-8")
    assert "IN(start)" in md and 'OUT: "Good afternoon!"' in md


def test_operator_question_routed_and_answer_resumes(tmp_path):
    r, client, asked = make(tmp_path, [
        turn(action="send_and_ask_operator", missing_facts=["guest_name"],
             message_to_manager="One moment, please.", operator_question="Guest name?"),
        turn(message_to_manager="It's under Nikita. Could you confirm?"),
    ])
    asyncio.run(r._process(Event("manager", "Name please?", [7])))
    assert asked == ["Guest name?"] and r.state.pending_question is not None
    asyncio.run(r._process(Event("operator", "Nikita")))
    assert client.sent[-1][1] == "It's under Nikita. Could you confirm?" and r.state.facts == {"guest_name": "Nikita"}


def _msg(id_, text="", voice=False):
    return SimpleNamespace(id=id_, raw_text=text, voice=voice, video_note=None, video=None, sticker=None,
                           photo=None, geo=None, contact=None, media=object() if voice else None, file=None)


def test_voice_in_is_transcribed_in_order_and_answered_by_voice(tmp_path):
    r, client, _ = make(tmp_path, [turn(message_to_manager="Lovely, could you confirm?")], voice=True)

    async def download_media(m, file):
        open(file, "wb").write(b"OggS")
        return file

    class FakeSTT:
        async def transcribe(self, path):
            await asyncio.sleep(0.05)  # slower than the text message that follows
            return "Table for two at five is fine"

    voiced = []

    async def send_voice(text):
        voiced.append(text)
        return True

    client.download_media = download_media
    r.stt = FakeSTT()
    r._send_voice_to_manager = send_voice

    async def scenario():
        r._buffer_message(_msg(1, voice=True))
        r._buffer_message(_msg(2, "Anything else?"))
        await r._flush()
        ev = await r.queue.get()
        assert ev.voice and ev.text.splitlines() == [
            '(voice message, transcribed) "Table for two at five is fine"', "Anything else?"]
        await r._process(ev)

    asyncio.run(scenario())
    assert voiced == ["Lovely, could you confirm?"] and client.sent == []  # voice only, no duplicate text
    assert "manager/voice" in (tmp_path / "logs" / "decisions.md").read_text("utf-8")


def test_reply_after_operator_answer_keeps_voice_modality(tmp_path):
    # Live run: manager asked by voice, operator answered, the reply went out as text.
    r, client, _ = make(tmp_path, [
        turn(action="send_and_ask_operator", missing_facts=["guest_name"],
             message_to_manager="One moment, please.", operator_question="Guest name?"),
        turn(message_to_manager="The name is Nikita. Could you confirm?"),
        turn(message_to_manager="Lovely, shall I confirm?"),
    ], voice=True)
    voiced = []

    async def send_voice(text):
        voiced.append(text)
        return True

    r._send_voice_to_manager = send_voice
    asyncio.run(r._process(Event("manager", '(voice message, transcribed) "Name?"', [1], voice=True)))
    asyncio.run(r._process(Event("operator", "Nikita")))
    # the reply carrying the operator's name goes as text, the hold message stays voice
    assert voiced == ["One moment, please."]
    assert client.sent[-1][1] == "The name is Nikita. Could you confirm?"
    asyncio.run(r._process(Event("manager", "Thanks!", [2])))  # manager switched to text
    assert client.sent[-1][1] == "Lovely, shall I confirm?"


def test_stt_failure_keeps_message(tmp_path):
    r, client, _ = make(tmp_path, [], voice=True)

    async def download_media(m, file):
        raise ConnectionError("boom")

    client.download_media = download_media
    assert asyncio.run(r._transcribe(_msg(1, voice=True))) == "[voice message — could not be transcribed]"


def test_describe_media():
    m = SimpleNamespace(raw_text="", voice=True, video_note=None, video=None, sticker=None,
                        photo=None, geo=None, contact=None, media=object(), file=None)
    assert _describe(m) == "[voice message]"


def test_after_opening_mode_voice_for_every_reply_but_recap_is_text(tmp_path):
    # Final run request: opening as text, then voice whatever the manager uses; recap as text.
    r, client, _ = make(tmp_path, [
        turn(message_to_manager="Good afternoon, could I book a table for two today at 5pm?"),
        turn(message_to_manager="I'm afraid only 5pm works for us. Could you let me know if it frees up?"),
        turn(manager_intent="confirm", message_to_manager="Lovely.",
             state_update={"status": "confirmed", "restaurant": "Dishoom",
                           "date": "2026-11-21", "time": "17:00", "party_size": 2}),
    ], voice=True, voice_mode="after_opening")
    voiced = []

    async def send_voice(text):
        voiced.append(text)
        return True

    r._send_voice_to_manager = send_voice
    asyncio.run(r._process(Event("start")))
    asyncio.run(r._process(Event("manager", "5 is gone, half five?", [1])))  # manager writes text
    asyncio.run(r._process(Event("manager", "A 5pm table freed up, booked!", [2])))
    texts = [t for _, t in client.sent]
    assert texts[0].startswith("Good afternoon")                      # opening: text
    assert voiced == ["I'm afraid only 5pm works for us. Could you let me know if it frees up?"]
    assert texts[-1].startswith("Perfect. Just to confirm") and len(texts) == 2  # recap: text only


def test_message_with_operator_details_goes_as_text(tmp_path):
    r, client, _ = make(tmp_path, [
        turn(message_to_manager="The name is Nikita and the number is +447700900123. Could you confirm?"),
        turn(message_to_manager="Of course, I'll wait. Could you let me know?"),
    ], voice=True, voice_mode="after_opening")
    r.state.facts.update({"guest_name+phone": "Nikita +447700900123",
                          "operator_note": "Only 5pm works, politely decline"})
    voiced = []

    async def send_voice(text):
        voiced.append(text)
        return True

    r._send_voice_to_manager = send_voice
    asyncio.run(r._process(Event("manager", "Name and number?", [1])))
    asyncio.run(r._process(Event("manager", "Nothing at 5 yet, sorry.", [2])))
    assert client.sent[0][1].startswith("The name is Nikita")       # details: text
    assert voiced == ["Of course, I'll wait. Could you let me know?"]  # operator note is not a detail


def test_operator_question_quotes_managers_message(tmp_path):
    # Live test: manager said "5 is fully booked, half five or six. Name and number?",
    # the operator was asked only for name and phone and never saw the time change.
    msg = "I'm afraid 5 is fully booked today. I could do half five or six. Could I take a name?"
    r, _, asked = make(tmp_path, [
        turn(action="send_and_ask_operator", missing_facts=["guest_name", "phone"],
             message_to_manager="One moment, please.", operator_question="Guest name and phone?"),
    ])
    asyncio.run(r._process(Event("manager", msg, [1])))
    assert asked == ["Guest name and phone?"] and r.asked_context == msg


def test_format_question_includes_context():
    from agent.operator import format_question

    text = format_question("Accept 5:30?", "5 is fully booked, half five or six.")
    assert "The manager wrote:\n«5 is fully booked, half five or six.»" in text and "Accept 5:30?" in text


def test_later_mention_of_name_stays_voice_but_phone_is_text(tmp_path):
    # Live test: "...for two under Nikita, so please do let me know..." went out as
    # text only because it mentioned the name. Name alone later on: voice.
    r, client, _ = make(tmp_path, [
        turn(message_to_manager="We can only do 5pm for two under Nikita. Could you let me know?"),
        turn(message_to_manager="Of course, the number is +447700900123. Could you confirm?"),
    ], voice=True, voice_mode="after_opening")
    r.state.facts.update({"guest_name": "Nikita", "phone": "+447700900123"})
    voiced = []

    async def send_voice(text):
        voiced.append(text)
        return True

    r._send_voice_to_manager = send_voice
    asyncio.run(r._process(Event("manager", "5 is gone, half five?", [1])))
    asyncio.run(r._process(Event("manager", "What's the number again?", [2])))
    assert voiced == ["We can only do 5pm for two under Nikita. Could you let me know?"]
    assert client.sent[-1][1].startswith("Of course, the number is")   # phone: text
