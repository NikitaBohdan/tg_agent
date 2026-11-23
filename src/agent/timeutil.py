"""Booking-time helpers. 'Today 17:00' always means the restaurant's local time."""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo


def now_in(tz: ZoneInfo) -> datetime:
    return datetime.now(tz)


def target_datetime(tz: ZoneInfo, at: time, now: datetime | None = None) -> datetime:
    """Today's date in `tz` combined with `at` (tz-aware)."""
    now = now.astimezone(tz) if now else now_in(tz)
    return datetime.combine(now.date(), at, tzinfo=tz)


def minutes_until(target: datetime, now: datetime | None = None) -> float:
    now = now or datetime.now(target.tzinfo)
    return (target - now).total_seconds() / 60


def human_date(dt: datetime) -> str:
    """'Saturday 21 November 2026' — British style, unambiguous."""
    return f"{dt:%A} {dt.day} {dt:%B %Y}"
