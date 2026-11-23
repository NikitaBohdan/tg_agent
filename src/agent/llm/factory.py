"""Build the provider chain [primary, fallback?] from settings."""

from __future__ import annotations

from ..config import Settings
from .base import LLMProvider


def _build(provider: str, model: str, s: Settings) -> LLMProvider:
    if not model:
        raise ValueError(f"Model for provider '{provider}' is empty (LLM_MODEL / LLM_FALLBACK_MODEL)")
    key = s.api_key_for(provider)
    if provider == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(
            key, model, s.llm_temperature, s.llm_timeout_s, s.openai_reasoning_effort
        )
    if provider == "gemini":
        from .gemini import GeminiProvider

        return GeminiProvider(
            key, model, s.llm_temperature, s.llm_timeout_s, s.gemini_thinking_level
        )
    if provider == "groq":
        from .groq_provider import GroqProvider

        return GroqProvider(key, model, s.llm_temperature, s.llm_timeout_s)
    raise ValueError(f"Unknown LLM provider: {provider}")


def build_chain(s: Settings) -> list[LLMProvider]:
    chain = [_build(s.llm_provider, s.llm_model, s)]
    if s.llm_fallback_provider:
        chain.append(_build(s.llm_fallback_provider, s.llm_fallback_model, s))
    return chain
