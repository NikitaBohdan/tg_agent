# Telegram table-booking agent

An AI agent that books **a table for two at 17:00 today (Europe/London)** by chatting with a
restaurant manager on Telegram. It opens the conversation itself in polite British English,
reacts to whatever the manager says, never invents facts (it asks its human operator instead),
confirms the booking back in the chat and writes its own decision log.

Bonus: it understands the manager's **voice notes** and talks back with generated
British-voice voice notes.

- Submission notes (tools, AI use, what broke): [NOTES.md](NOTES.md)
- Full history of what broke during testing and how it was fixed: [docs/NOTES.md](docs/NOTES.md)
- Decision log of the final conversation: [docs/decision-log.md](docs/decision-log.md)
- Screenshots of the final conversation: [docs/screenshots/](docs/screenshots/)

## How it works

The agent has three layers:

- **Runner** (`runner.py`) talks to Telegram. It receives the manager's messages, edits and
  voice notes, merges messages sent in a quick burst, transcribes voice notes, watches for
  silence, and picks up messages that arrived while the agent was offline. Everything goes
  into one queue and is handled one event at a time.
- **Core** (`core.py`) decides what to do with each event. It asks the brain for a decision,
  then applies the rules that must not depend on the model: it writes the final confirmation
  itself, never asks the operator twice for the same fact, and checks every fact from the
  operator against what the operator actually wrote.
- **Brain** (`brain.py`) builds the prompt from the goal, the booking so far, the known facts
  and the transcript, calls the LLM, and checks its JSON answer against the schema. A broken
  answer gets one repair attempt, and an unavailable provider is retried or replaced by the
  fallback provider.

After each event the runner sends a text or voice reply to the manager or a question to the
operator, writes a line to the decision log and saves the state file.

1. **Why a user account, not a bot.** The Bot API cannot message someone who has not pressed
   `/start`, and the manager never writes first. The agent therefore runs on a regular Telegram
   account via [Telethon](https://docs.telethon.dev) (MTProto).
2. **One decision per event.** Every event (start, manager message, operator answer, silence)
   goes to the LLM with the goal, the booking so far, the known facts and the transcript. The
   model must return one JSON object (`schema.py`):

   ```json
   {"understood": "Manager offers 18:30 instead of 17:00", "manager_intent": "offer_alternative",
    "state_update": {"time": null}, "missing_facts": [], "facts_from_operator": [],
    "decision": "Only 17:00 may be accepted without the operator, so ask and hold politely",
    "action": "send_and_ask_operator",
    "message_to_manager": "Thank you, bear with me a moment while I check.",
    "operator_question": "They offer 6:30pm instead of 5pm. Accept?"}
   ```

   The decision log is built from these fields, so the agent writes it itself.
3. **Rules the model cannot break** are enforced in code (`core.py`):

   | Rule | How |
   |---|---|
   | The final confirmation repeats restaurant, day and date, time, people and name | rendered from state by a template, not by the LLM |
   | The conversation cannot end before that confirmation is sent | `finish` is blocked until then |
   | Only the exact goal (2 people, today, 17:00) is accepted without the operator | prompt + operator question on any change |
   | The operator is never asked twice for a known fact (name, phone) | repeat question blocked, model re-asked |
   | Every change of terms (time, date, people) goes to the operator, with their earlier decision quoted | prompt; the question in Saved Messages always quotes the manager's message |
   | After the operator answers, the manager always gets a reply | a "wait" after a promised reply is sent back to the model |
   | A question that became irrelevant is closed and the operator is told | late replies to closed questions are rejected |
   | An unconfirmed booking never ends on a plain "thank you" | a reply with no question is sent back to the model to ask for confirmation |
   | The agent never cancels, gives up or ends the chat on its own, and checks claims about the guest with the operator | otherwise the operator is asked, with a fallback question if the model refuses |
   | Facts from the operator are not embellished | each value must appear verbatim in the operator's answer |
   | No new conversation too close to the slot | refuses < `START_GUARD_MIN` minutes before 17:00 |
   | No accidental message to the real manager | username must be typed to confirm the first message |

4. **Resilience.** Invalid JSON gets one repair round. Rate limits and outages are retried with
   backoff, then the optional fallback provider takes over, then delayed retries with an alert
   to the operator. Telegram reconnects automatically. State is saved after every step, and a
   restart resumes the conversation and reads messages missed while offline. Voice failures
   degrade to text.

## Setup

Requirements: Python 3.11+, `ffmpeg` on PATH (voice notes only), a **separate Telegram account**
for the agent, and an LLM API key (OpenAI by default; Gemini and Groq also supported).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,voice]"
copy .env.example .env          # then fill it in
python -m scripts.login          # one-time Telegram login of the AGENT account
```

Key settings in `.env` (all documented in [.env.example](.env.example)):

| Variable | Meaning |
|---|---|
| `TG_API_ID`, `TG_API_HASH` | from <https://my.telegram.org>, section "API development tools" |
| `MANAGER_USERNAME` | who the agent writes to: a test account first, `vlad_dev_7` for the real run |
| `BOOKING_TZ`, `BOOKING_TIME`, `BOOKING_PARTY_SIZE` | the goal; the date is always *today* in `BOOKING_TZ` |
| `LLM_PROVIDER`, `LLM_MODEL`, `*_API_KEY` | `openai` / `gemini` / `groq`; list usable models with `python -m scripts.list_models` |
| `LLM_FALLBACK_PROVIDER`, `LLM_FALLBACK_MODEL` | optional second provider |
| `OPERATOR_CHANNEL` | `saved_messages` (answer from the phone) or `console` |
| `STT_PROVIDER`, `TTS_VOICE` | speech-to-text (`local` faster-whisper is free) and edge-tts voice |

## Run

```powershell
python -m agent                      # start, or resume today's conversation
python -m agent --reset              # archive saved state and start over
python -m agent --operator console   # answer operator questions in the terminal
python -m agent --force              # start even if 17:00 London is too close
```

- When the agent needs something it does not know (guest name, phone, a different time), it
  posts a 🤖 question to the agent account's **Saved Messages**, quoting the manager's message.
  Reply to it, for example from the phone with the agent account added as a second account.
- You can also step in at any time: a plain message (not a reply) in Saved Messages is taken
  as an instruction, e.g. "Send one polite reminder" or "We did not cancel, ask them to
  reinstate the booking".
- `logs/decisions.md` holds one line per event (in / understood / decided / out) and
  `logs/decisions.jsonl` the full records. Times are in the restaurant's time zone.
  The debug log is `logs/agent.log`.

## Voice notes (bonus)

- **Voice from the manager:** the voice note is downloaded, decoded with ffmpeg and transcribed by
  faster-whisper (`small.en`, local and free). The agent sees
  `(voice message, transcribed) "…"` and is told to double-check key details.
- **Voice from the agent:** the opening message is text; every later reply is a voice note,
  whatever the manager uses (`VOICE_MODE=after_opening`; `mirror` answers in the manager's
  modality instead): edge-tts `en-GB-RyanNeural`, converted to OGG/Opus, shown with "recording audio…".
  The final booking confirmation, any message with the guest's phone or email, and the message
  that first passes on the guest's name are sent as text so the details stay readable and exact.

## Testing

```powershell
pytest                                   # unit tests, no network
python -m scripts.check_telegram         # Telegram session + manager lookup; sends nothing to the manager
python -m scripts.list_models --ping     # which models your key can use + a test call
python -m scripts.simulate               # play the manager in the console (same core as live)
python -m scripts.voice_check            # speak a phrase, convert it, transcribe it back
```

Live test: set `MANAGER_USERNAME` to your second account, run `python -m agent`, and play the
manager. Try offering another time, asking for a name and phone, sending a voice note, sending
several messages in a row, going silent, or stopping and restarting the agent mid-conversation.

## Project layout

```
src/agent/
  __main__.py       CLI entry point (python -m agent)
  config.py         typed settings from .env (pydantic-settings)
  runner.py         live Telegram loop: event queue, debounce, typing, silence watchdog, resume
  core.py           decides what to do with each event, no I/O (shared by sim and live)
  brain.py          prompt building, LLM call, JSON repair / retry / fallback
  prompts/system.md the agent's instructions (British manners, no-invention rules)
  schema.py         AgentTurn — the LLM's validated JSON contract
  state.py          goal, booking, operator facts, transcript (persisted JSON)
  llm/              provider adapters: openai_provider.py, gemini.py, groq_provider.py
  operator.py       human-in-the-loop: Saved Messages or console
  voice.py          STT (faster-whisper / OpenAI / Groq) and TTS (edge-tts + ffmpeg)
  decision_log.py   decisions.md / decisions.jsonl written by the agent
  tg.py             Telethon client, peer resolution, resilient send
  timeutil.py       "today 17:00" in the restaurant time zone
  logging_setup.py  console + rotating file debug log
scripts/            login, check_telegram, list_models, simulate, voice_check
tests/              unit tests (fake LLM and fake Telegram client)
docs/               final conversation (screenshots/, decision-log.md) and the full NOTES.md
```

## Security

Secrets live only in `.env`. `.env`, `*.session`, `data/` and `logs/` are git-ignored. A Telethon
`.session` file is as good as being logged in, so never share it.
