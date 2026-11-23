from __future__ import annotations

import json

import groq

from .base import LLMBadOutput, LLMConfigError, LLMRateLimited, LLMUnavailable


class GroqProvider:
    def __init__(self, api_key: str, model: str, temperature: float | None, timeout_s: float):
        self.model = model
        self.name = f"groq:{model}"
        self._temperature = temperature
        # SDK retries are disabled: retry/fallback policy lives in the chain.
        self._client = groq.AsyncGroq(api_key=api_key, timeout=timeout_s, max_retries=0)

    async def complete_json(self, system: str, user: str, json_schema: dict) -> str:
        # JSON mode (widely supported on Groq); the schema goes into the prompt
        # and the result is validated by Pydantic in brain.py.
        system_full = (
            f"{system}\n\nReturn ONLY a JSON object matching this JSON Schema:\n"
            f"{json.dumps(json_schema, ensure_ascii=False)}"
        )
        try:
            extra = {} if self._temperature is None else {"temperature": self._temperature}
            resp = await self._client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_full},
                    {"role": "user", "content": user},
                ],
                **extra,
            )
        except groq.RateLimitError as e:
            ra = e.response.headers.get("retry-after") if e.response is not None else None
            raise LLMRateLimited(f"{self.name}: {e}", float(ra) if ra else None) from e
        except groq.BadRequestError as e:
            # json_validate_failed => the model produced invalid JSON
            if "json" in str(e).lower():
                raise LLMBadOutput(f"{self.name}: {e}") from e
            raise LLMConfigError(f"{self.name}: {e}") from e
        except (groq.AuthenticationError, groq.PermissionDeniedError, groq.NotFoundError) as e:
            raise LLMConfigError(f"{self.name}: {e}") from e
        except (groq.APIConnectionError, groq.APIStatusError) as e:
            raise LLMUnavailable(f"{self.name}: {e}") from e

        text = (resp.choices[0].message.content or "").strip() if resp.choices else ""
        if not text:
            raise LLMBadOutput(f"{self.name}: empty response")
        return text
