import asyncio
from types import SimpleNamespace

import httpx
import openai
import pytest

from agent.llm import LLMConfigError
from agent.llm.openai_provider import OpenAIProvider

REQ = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _err(cls, status, msg):
    return cls(msg, response=httpx.Response(status, request=REQ), body=None)


def _ok(text):
    msg = SimpleNamespace(content=text)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=None)


def _provider(outcomes):
    p = OpenAIProvider("sk-test", "some-model", 0.4, 10, reasoning_effort="low")
    calls = []

    async def create(**kw):
        calls.append(kw)
        out = outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    p._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return p, calls


def test_drops_unsupported_temperature_and_retries():
    p, calls = _provider([
        _err(openai.BadRequestError, 400, "Unsupported parameter: 'temperature' is not supported with this model."),
        _ok('{"ok": true}'),
    ])
    assert asyncio.run(p.complete_json("s", "u", {})) == '{"ok": true}'
    assert "temperature" in calls[0] and "temperature" not in calls[1]
    assert calls[1]["reasoning_effort"] == "low"


def test_insufficient_quota_is_config_error():
    p, _ = _provider([_err(openai.RateLimitError, 429, "You exceeded your current quota: insufficient_quota")])
    with pytest.raises(LLMConfigError, match="balance exhausted"):
        asyncio.run(p.complete_json("s", "u", {}))
