"""Provider-neutral LLM interface and error taxonomy.

Providers return RAW TEXT (expected to be JSON). Parsing/validation lives in
brain.py so it is identical for every provider.
"""

from __future__ import annotations

from typing import Protocol


class LLMError(Exception):
    """Base class for all LLM failures."""


class LLMRateLimited(LLMError):
    def __init__(self, msg: str, retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


class LLMUnavailable(LLMError):
    """Network error, timeout, 5xx, auth/config problem — try another provider."""


class LLMConfigError(LLMError):
    """Retrying won't help: unknown/retired model, bad key, no access (HTTP 400/401/403/404)."""


class LLMBadOutput(LLMError):
    """Model answered, but not with usable JSON (after repair attempts)."""


class LLMProvider(Protocol):
    name: str  # e.g. "gemini:gemini-x-flash"

    async def complete_json(self, system: str, user: str, json_schema: dict) -> str:
        """Return the model's JSON text for the given prompt."""
        ...
