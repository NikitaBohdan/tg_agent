from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

import pytest

from agent.config import Settings
from agent.timeutil import human_date, target_datetime

LONDON = ZoneInfo("Europe/London")


def _settings(**kw):
    base = dict(tg_api_id=1, tg_api_hash="x", manager_username="@vlad_dev_7", _env_file=None)
    base.update(kw)
    return Settings(**base)


def test_username_at_is_stripped():
    assert _settings().manager_username == "vlad_dev_7"


def test_invalid_timezone_rejected():
    with pytest.raises(ValueError):
        _settings(booking_tz="Mars/Olympus")


def test_target_uses_london_date_not_local():
    # 23:30 UTC on 21 Nov == 23:30 London (GMT), but already 22 Nov in Kyiv.
    now = datetime(2026, 11, 21, 23, 30, tzinfo=timezone.utc)
    t = target_datetime(LONDON, time(17, 0), now=now)
    assert (t.year, t.month, t.day, t.hour) == (2026, 11, 21, 17)


def test_target_respects_bst():
    # In July London is UTC+1: 17:00 London == 16:00 UTC.
    now = datetime(2026, 7, 10, 9, 0, tzinfo=timezone.utc)
    t = target_datetime(LONDON, time(17, 0), now=now)
    assert t.astimezone(timezone.utc).hour == 16


def test_human_date():
    assert human_date(datetime(2026, 11, 21, tzinfo=LONDON)) == "Saturday 21 November 2026"
