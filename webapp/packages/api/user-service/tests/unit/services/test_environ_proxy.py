"""Unit tests for the per-agent env_var overlay proxy (ISSUE-008)."""
from __future__ import annotations

import asyncio
import os

import pytest


pytestmark = pytest.mark.unit


def test_install_environ_proxy_is_idempotent():
    from services.environ_proxy import install_environ_proxy, _EnvironProxy

    install_environ_proxy()
    first = os.environ
    install_environ_proxy()
    assert os.environ is first
    assert isinstance(os.environ, _EnvironProxy)


def test_no_overlay_passes_through_to_real_environ(monkeypatch):
    from services.environ_proxy import install_environ_proxy

    install_environ_proxy()
    monkeypatch.setenv("ISSUE_008_TEST_KEY", "real-value")
    assert os.environ.get("ISSUE_008_TEST_KEY") == "real-value"


def test_overlay_takes_precedence_over_real():
    from services.environ_proxy import install_environ_proxy, env_overlay

    install_environ_proxy()
    os.environ["ISSUE_008_TEST_KEY"] = "real-value"
    try:
        with env_overlay({"ISSUE_008_TEST_KEY": "overlay-value"}):
            assert os.environ.get("ISSUE_008_TEST_KEY") == "overlay-value"
            assert "ISSUE_008_TEST_KEY" in os.environ
        # Outside the overlay, real value visible again.
        assert os.environ.get("ISSUE_008_TEST_KEY") == "real-value"
    finally:
        del os.environ["ISSUE_008_TEST_KEY"]


def test_overlay_missing_key_falls_back_to_real():
    from services.environ_proxy import install_environ_proxy, env_overlay

    install_environ_proxy()
    os.environ["ISSUE_008_FALLBACK_KEY"] = "real"
    try:
        with env_overlay({"OTHER_KEY": "other"}):
            assert os.environ.get("ISSUE_008_FALLBACK_KEY") == "real"
            assert os.environ.get("OTHER_KEY") == "other"
    finally:
        del os.environ["ISSUE_008_FALLBACK_KEY"]


def test_concurrent_overlays_are_isolated():
    """Two asyncio tasks see their own overlays — no cross-contamination."""
    from services.environ_proxy import install_environ_proxy, env_overlay

    install_environ_proxy()
    results: dict[str, str | None] = {}

    async def runner(label: str, val: str):
        with env_overlay({"ISO_TEST_KEY": val}):
            await asyncio.sleep(0)  # context-switch
            results[label] = os.environ.get("ISO_TEST_KEY")

    async def main():
        await asyncio.gather(runner("a", "alpha"), runner("b", "beta"))

    asyncio.run(main())
    assert results == {"a": "alpha", "b": "beta"}


def test_overlay_does_not_leak_after_exit():
    from services.environ_proxy import install_environ_proxy, env_overlay

    install_environ_proxy()
    with env_overlay({"NEW_OVERLAY_KEY": "x"}):
        assert "NEW_OVERLAY_KEY" in os.environ
    # Outside the overlay, the key shouldn't be reachable
    # (unless something else set it in the real environ — guard against that).
    assert os.environ.get("NEW_OVERLAY_KEY") is None
