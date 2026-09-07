"""Tests for how the backend is configured from the environment.

The mock timings are the one thing an evaluation run adjusts without touching
code, so a bad value must fall back to the default rather than crash the
server on startup: the MCP client would only see the connection close.
"""

import pytest

from video_mcp.video import (
    DEFAULT_QUEUED_SECONDS,
    DEFAULT_RUNNING_SECONDS,
    MockVideoBackend,
    _float_env,
    build_backend,
)

QUEUED_VAR = "AI_VIDEO_MOCK_QUEUED_SECONDS"
RUNNING_VAR = "AI_VIDEO_MOCK_RUNNING_SECONDS"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Keep the developer's own timing overrides out of these tests."""
    monkeypatch.delenv(QUEUED_VAR, raising=False)
    monkeypatch.delenv(RUNNING_VAR, raising=False)


def test_an_unset_variable_uses_the_default():
    assert _float_env("AI_VIDEO_MOCK_NOT_SET", 4.5) == 4.5


def test_a_valid_value_is_used(monkeypatch):
    monkeypatch.setenv(QUEUED_VAR, "1.5")

    assert _float_env(QUEUED_VAR, 4.5) == 1.5


def test_zero_is_a_valid_value(monkeypatch):
    monkeypatch.setenv(QUEUED_VAR, "0")

    assert _float_env(QUEUED_VAR, 4.5) == 0.0


@pytest.mark.parametrize("raw", ["", "fast", "5 seconds", "-1"])
def test_an_unusable_value_falls_back_to_the_default(monkeypatch, raw):
    monkeypatch.setenv(QUEUED_VAR, raw)

    assert _float_env(QUEUED_VAR, 4.5) == 4.5


def test_the_built_backend_uses_the_defaults():
    backend = build_backend()

    assert isinstance(backend, MockVideoBackend)
    assert backend._queued_seconds == DEFAULT_QUEUED_SECONDS
    assert backend._running_seconds == DEFAULT_RUNNING_SECONDS


def test_the_built_backend_honours_the_environment(monkeypatch):
    monkeypatch.setenv(QUEUED_VAR, "5")
    monkeypatch.setenv(RUNNING_VAR, "600")

    backend = build_backend()

    assert backend._queued_seconds == 5.0
    assert backend._running_seconds == 600.0
