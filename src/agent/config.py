"""Typed, validated configuration loaded from environment / .env.

All secrets live in .env (git-ignored). Import `get_settings()` instead of
reading os.environ anywhere else.
"""

from __future__ import annotations

from datetime import time
from functools import lru_cache
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Telegram
    tg_api_id: int = Field(gt=0)
    tg_api_hash: SecretStr
    tg_session: Path = Path("sessions/agent")
    manager_username: str = Field(min_length=1)

    # Booking goal
    booking_tz: str = "Europe/London"
    booking_time: time = time(17, 0)
    booking_party_size: int = Field(default=2, ge=1, le=20)

    # Operator
    operator_channel: Literal["saved_messages", "console"] = "saved_messages"

    # Live behaviour
    debounce_s: float = Field(default=4, ge=0)  # merge manager messages sent in a burst
    followup_after_min: float = Field(default=10, gt=0)  # one polite nudge after silence
    start_guard_min: float = Field(default=20, ge=0)  # refuse to start this close to the slot
    typing_simulation: bool = True  # show "typing…" proportional to message length

    # Voice (reply in voice only to the manager's voice messages)
    voice_enabled: bool = True
    # after_opening: first message text, then every reply is a voice note
    # mirror: voice only while the manager's latest message is a voice note
    voice_mode: Literal["after_opening", "mirror"] = "after_opening"
    stt_provider: Literal["local", "openai", "groq"] = "local"
    stt_model: str = ""  # empty = provider default (local: small.en)
    tts_voice: str = "en-GB-RyanNeural"

    # LLM (validated lazily — not needed for Telegram-only scripts)
    llm_provider: Literal["openai", "gemini", "groq"] = "openai"
    llm_model: str = ""
    llm_fallback_provider: Literal["openai", "gemini", "groq", ""] = ""
    llm_fallback_model: str = ""
    # None = provider default (some reasoning models reject a custom temperature)
    llm_temperature: float | None = Field(default=None, ge=0, le=2)

    @field_validator("llm_temperature", mode="before")
    @classmethod
    def _empty_temp(cls, v):
        return None if v == "" else v
    llm_timeout_s: float = Field(default=120, gt=0)
    gemini_thinking_level: Literal["", "minimal", "low", "medium", "high"] = "low"
    openai_reasoning_effort: Literal["", "minimal", "low", "medium", "high"] = "low"
    openai_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None

    # Runtime
    data_dir: Path = Path("data")
    log_dir: Path = Path("logs")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("manager_username")
    @classmethod
    def _strip_at(cls, v: str) -> str:
        return v.strip().lstrip("@")

    @field_validator("booking_tz")
    @classmethod
    def _valid_tz(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except ZoneInfoNotFoundError as e:
            raise ValueError(f"Unknown time zone: {v}") from e
        return v

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.booking_tz)

    def api_key_for(self, provider: str) -> str:
        key = {
            "openai": self.openai_api_key,
            "gemini": self.gemini_api_key,
            "groq": self.groq_api_key,
        }.get(provider)
        if key is None or not key.get_secret_value().strip():
            raise ValueError(f"{provider.upper()}_API_KEY is not set in .env")
        return key.get_secret_value().strip()

    def ensure_dirs(self) -> None:
        for d in (self.tg_session.parent, self.data_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # values come from env
