"""Decision log written by the agent itself: one line per processed event.

- decisions.jsonl — full structured record (machine-readable)
- decisions.md    — one human-readable line: what came in, what the agent
                    understood, what it decided, what it sent

Times are in the restaurant's time zone (e.g. GMT for London in winter);
the console debug log uses the machine's local time.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from .core import Outcome

log = logging.getLogger(__name__)


def _q(s: str | None, n: int = 300) -> str:
    if not s:
        return "—"
    s = " ".join(s.split())
    return f'"{s if len(s) <= n else s[:n] + "…"}"'


class DecisionLog:
    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = log_dir / "decisions.jsonl"
        self.md = log_dir / "decisions.md"

    def write(self, at: datetime, event_kind: str, incoming: str | None, o: Outcome) -> str:
        t = o.turn
        record = {
            "at": at.isoformat(timespec="seconds"),
            "event": event_kind,
            "incoming": incoming,
            "understood": t.understood,
            "manager_intent": t.manager_intent.value,
            "decision": t.decision,
            "action": t.action.value,
            "sent_to_manager": o.message_to_manager,
            "asked_operator": o.operator_question,
            "missing_facts": t.missing_facts,
            "state_update": t.state_update.model_dump(exclude_none=True),
            "guards": o.guard_notes,
            "provider": o.provider,
        }
        line = (
            f"- {at:%H:%M:%S %Z} | IN({event_kind}): {_q(incoming)} | UNDERSTOOD: {t.understood} | "
            f"DECIDED: {t.decision} | ACTION: {t.action.value} | OUT: {_q(o.message_to_manager)}"
        )
        if o.operator_question:
            line += f" | ASKED OPERATOR: {_q(o.operator_question)}"
        if o.guard_notes:
            line += f" | GUARDS: {'; '.join(o.guard_notes)}"
        try:
            with self.jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            with self.md.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as e:  # logging must never break the conversation
            log.error("decision log write failed: %s", e)
        return line

    def note(self, at: datetime, text: str) -> None:
        """Non-LLM events: errors, operator notices, session start/stop."""
        try:
            with self.md.open("a", encoding="utf-8") as f:
                f.write(f"- {at:%H:%M:%S %Z} | NOTE: {text}\n")
        except OSError as e:
            log.error("decision log write failed: %s", e)
