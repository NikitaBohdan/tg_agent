"""The agent's decision step: sends the context to the LLM and returns a validated AgentTurn.

Resilience policy:
- invalid JSON or schema mismatch: one repair round with the validation error;
- rate limit: wait (if short) and retry, otherwise switch to fallback provider;
- network error, 5xx or timeout: exponential backoff, then fallback provider;
- everything failed: BrainError (caller decides; the agent never crashes).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable
from datetime import date, datetime
from importlib import resources

from pydantic import ValidationError

from .llm import LLMBadOutput, LLMConfigError, LLMProvider, LLMRateLimited, LLMUnavailable
from .schema import AgentTurn
from .state import ConversationState, Role
from .timeutil import human_date

log = logging.getLogger(__name__)

MAX_ATTEMPTS_PER_PROVIDER = 3
MAX_RATE_LIMIT_WAIT_S = 30.0
TRANSCRIPT_LIMIT = 60  # entries; a booking chat is far shorter


class BrainError(Exception):
    """All providers failed to produce a valid AgentTurn."""


def load_system_prompt(state: ConversationState) -> str:
    tpl = resources.files("agent.prompts").joinpath("system.md").read_text("utf-8")
    g = state.goal
    values = {
        "party_size": g.party_size,
        "goal_date": g.date,
        "goal_date_human": human_date(datetime.combine(date.fromisoformat(g.date), datetime.min.time())),
        "goal_time": g.time,
    }
    # Plain replacement, not str.format: the prompt contains literal JSON braces.
    for k, v in values.items():
        tpl = tpl.replace("{" + k + "}", str(v))
    return tpl


def build_user_prompt(state: ConversationState, event: str, now: datetime) -> str:
    lines = []
    for e in state.history[-TRANSCRIPT_LIMIT:]:
        t = e.at.astimezone(now.tzinfo).strftime("%H:%M")
        label = {
            Role.manager: "MANAGER",
            Role.agent: "YOU",
            Role.operator: "OPERATOR (private, manager cannot see)",
            Role.system: "SYSTEM",
        }[e.role]
        lines.append(f"[{t}] {label}: {e.text}")
    transcript = "\n".join(lines) or "(empty — the conversation has not started yet)"

    ctx = {
        "now_restaurant_time": now.strftime("%A %d %B %Y, %H:%M"),
        "goal": state.goal.model_dump(),
        "booking_so_far": state.booking.model_dump(mode="json"),
        "known_facts": state.facts or {},
        "operator_instructions": state.operator_instructions,
        "confirmation_summary_already_sent": state.recap_sent,
        "awaiting_operator_answer_to": state.pending_question.question if state.pending_question else None,
    }
    return (
        f"CONTEXT:\n{json.dumps(ctx, ensure_ascii=False, indent=2)}\n\n"
        f"TRANSCRIPT:\n{transcript}\n\n"
        f"LATEST EVENT: {event}\n\n"
        "Decide your next step and return the JSON object."
    )


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


def parse_turn(raw: str) -> AgentTurn:
    """Parse model text into AgentTurn. Raises ValueError/ValidationError."""
    text = _FENCE.sub("", raw.strip())
    if not text.startswith("{"):  # tolerate leading chatter
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found")
        text = text[start : end + 1]
    return AgentTurn.model_validate(json.loads(text))


class Brain:
    def __init__(
        self,
        chain: list[LLMProvider],
        now_fn: Callable[[], datetime],
        sleep: Callable[[float], object] = asyncio.sleep,
    ):
        if not chain:
            raise ValueError("empty provider chain")
        self.chain = chain
        self.now_fn = now_fn
        self._sleep = sleep
        self._schema = AgentTurn.model_json_schema()
        self.last_provider: str | None = None

    async def decide(self, state: ConversationState, event: str) -> AgentTurn:
        system = load_system_prompt(state)
        base_user = build_user_prompt(state, event, self.now_fn())
        errors: list[str] = []

        for provider in self.chain:
            user = base_user
            repaired = False
            attempt = 0
            while attempt < MAX_ATTEMPTS_PER_PROVIDER:
                attempt += 1
                try:
                    raw = await provider.complete_json(system, user, self._schema)
                except LLMRateLimited as e:
                    errors.append(str(e))
                    wait = e.retry_after if e.retry_after is not None else 5.0 * attempt
                    if wait > MAX_RATE_LIMIT_WAIT_S:
                        log.warning("%s rate-limited for %.0fs, switching provider", provider.name, wait)
                        break
                    log.warning("%s rate-limited, retry in %.0fs", provider.name, wait)
                    await self._sleep(wait)
                    continue
                except LLMConfigError as e:
                    errors.append(str(e))
                    log.error("%s misconfigured, skipping it: %s", provider.name, e)
                    break
                except LLMUnavailable as e:
                    errors.append(str(e))
                    log.warning("%s unavailable (attempt %d): %s", provider.name, attempt, e)
                    await self._sleep(2.0**attempt)
                    continue
                except LLMBadOutput as e:
                    errors.append(str(e))
                    log.warning("%s bad output: %s", provider.name, e)
                    continue

                try:
                    turn = parse_turn(raw)
                except (ValueError, ValidationError) as e:
                    errors.append(f"{provider.name}: invalid turn: {e}")
                    if repaired:
                        log.warning("%s still invalid after repair, next provider", provider.name)
                        break
                    log.info("%s returned invalid turn, asking it to repair: %s",
                             provider.name, _short(" ".join(str(e).split()), 300))
                    repaired = True
                    user = (
                        f"{base_user}\n\nYour previous output was INVALID:\n{_short(str(e))}\n"
                        f"Previous output:\n{_short(raw, 2000)}\n"
                        "Return the corrected JSON object only."
                    )
                    continue

                self.last_provider = provider.name
                return turn

        raise BrainError("; ".join(errors[-6:]) or "no provider produced a result")


def _short(s: str, n: int = 800) -> str:
    return s if len(s) <= n else s[:n] + "…"
