from __future__ import annotations

import json
import logging

import openai

from .base import LLMBadOutput, LLMConfigError, LLMRateLimited, LLMUnavailable

log = logging.getLogger(__name__)


class OpenAIProvider:
    """OpenAI Chat Completions in JSON mode.

    Model families differ in accepted params (e.g. reasoning models reject a
    custom `temperature`, older ones reject `reasoning_effort`). Instead of a
    hard-coded model table, an unsupported param is dropped once on HTTP 400
    and the request is retried — so LLM_MODEL can be any chat model.
    """

    def __init__(
        self, api_key: str, model: str, temperature: float | None, timeout_s: float, reasoning_effort: str = ""
    ):
        self.model = model
        self.name = f"openai:{model}"
        self._params: dict = {} if temperature is None else {"temperature": temperature}
        if reasoning_effort:
            self._params["reasoning_effort"] = reasoning_effort
        # SDK retries disabled: retry/fallback policy lives in brain.py.
        self._client = openai.AsyncOpenAI(api_key=api_key, timeout=timeout_s, max_retries=0)

    async def complete_json(self, system: str, user: str, json_schema: dict) -> str:
        system_full = (
            f"{system}\n\nReturn ONLY a JSON object matching this JSON Schema:\n"
            f"{json.dumps(json_schema, ensure_ascii=False)}"
        )
        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_full},
                    {"role": "user", "content": user},
                ],
                **self._params,
            )
        except openai.BadRequestError as e:
            dropped = self._drop_unsupported_param(str(e))
            if dropped:
                log.warning("%s does not accept %r, dropping it", self.name, dropped)
                return await self.complete_json(system, user, json_schema)
            raise LLMConfigError(f"{self.name}: {e}") from e
        except openai.RateLimitError as e:
            if "insufficient_quota" in str(e):
                raise LLMConfigError(f"{self.name}: API balance exhausted (insufficient_quota)") from e
            ra = e.response.headers.get("retry-after") if e.response is not None else None
            raise LLMRateLimited(f"{self.name}: {e}", float(ra) if ra else None) from e
        except (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError) as e:
            raise LLMConfigError(f"{self.name}: {e}") from e
        except (openai.APIConnectionError, openai.APIStatusError) as e:
            raise LLMUnavailable(f"{self.name}: {e}") from e

        choice = resp.choices[0] if resp.choices else None
        text = (choice.message.content or "").strip() if choice else ""
        if not text:
            reason = choice.finish_reason if choice else None
            raise LLMBadOutput(f"{self.name}: empty response (finish_reason={reason})")
        if resp.usage:
            log.debug("%s usage: in=%s out=%s", self.name, resp.usage.prompt_tokens, resp.usage.completion_tokens)
        return text

    def _drop_unsupported_param(self, err: str) -> str | None:
        for p in list(self._params):
            if p in err:
                del self._params[p]
                return p
        return None
