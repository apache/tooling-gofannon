"""Unit tests for time helpers."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from time_utils import naive_utc_now


pytestmark = pytest.mark.unit


def test_naive_utc_now_returns_naive_utc_datetime():
    before = datetime.now(timezone.utc).replace(tzinfo=None)

    value = naive_utc_now()

    after = datetime.now(timezone.utc).replace(tzinfo=None)
    assert value.tzinfo is None
    assert before <= value <= after


def test_user_service_has_no_deprecated_datetime_calls():
    service_root = Path(__file__).resolve().parents[2]
    needle = "datetime." + "utc" + "now"

    offenders = [
        path.relative_to(service_root)
        for path in service_root.rglob("*.py")
        if needle in path.read_text(encoding="utf-8")
    ]

    assert offenders == []
