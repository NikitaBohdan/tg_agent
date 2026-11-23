from __future__ import annotations

import logging
import re

from google import genai
from google.genai import errors, types

from .base import LLMBadOutput, LLMConfigError, LLMRateLimited, LLMUnavailable

log = logging.getLogger(__name__)


class GeminiProvider:
    def __init__(
        self, api_key: str, model: str, temperature: float | None, timeout_s: float, thinking_level: str = ""
    ):
        self.model = model
        self.name = f"gemini:{model}"
        self._temperature = temperature
        # Thinking models can blow the deadline on long reasoning; a chat reply
        # needs little. Dropped automatically if the model does not support it.
        self._thinking_level = thinking_level.upper() or None
        self._client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000))
        )
        # Constrained decoding by JSON schema; disabled automatically if the
        # model/API rejects the schema (we still validate with Pydantic).
        self._use_schema = True

    async def complete_json(self, system: str, user: str, json_schema: dict) -> str:
        cfg = dict(system_instruction=system, response_mime_type="application/json")
        if self._temperature is not None:
            cfg["temperature"] = self._temperature
        if self._use_schema:
            cfg["response_json_schema"] = json_schema
        if self._thinking_level:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_level=self._thinking_level)
        try:
            resp = await self._client.aio.models.generate_content(
                model=self.model, contents=user, config=types.GenerateContentConfig(**cfg)
            )
        except errors.APIError as e:
            code = getattr(e, "code", None)
            if code == 429:
                raise LLMRateLimited(f"{self.name}: {e}", _retry_after(str(e))) from e
            if code == 400 and self._use_schema and "schema" in str(e).lower():
                log.warning("%s rejected JSON schema, falling back to plain JSON mode", self.name)
                self._use_schema = False
                return await self.complete_json(system, user, json_schema)
            if code == 400 and self._thinking_level and "think" in str(e).lower():
                log.warning("%s does not support thinking_level, dropping it", self.name)
                self._thinking_level = None
                return await self.complete_json(system, user, json_schema)
            if code in (400, 401, 403, 404):
                raise LLMConfigError(f"{self.name}: HTTP {code}: {e}") from e
            raise LLMUnavailable(f"{self.name}: HTTP {code}: {e}") from e
        except Exception as e:  # network/transport errors (httpx/httpx2/OS) — SDK-version neutral
            raise LLMUnavailable(f"{self.name}: {type(e).__name__}: {e}") from e

        text = (resp.text or "").strip()
        if not text:
            reason = None
            if resp.candidates:
                reason = resp.candidates[0].finish_reason
            raise LLMBadOutput(f"{self.name}: empty response (finish_reason={reason})")
        return text


def _retry_after(msg: str) -> float | None:
    # Gemini puts e.g. "retryDelay": "17s" into the error body.
    m = re.search(r"retry(?:Delay)?\D{0,5}(\d+(?:\.\d+)?)s", msg, re.I)
    return float(m.group(1)) if m else None
