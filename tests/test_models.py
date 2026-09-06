"""Tests for the request/response models.

The lifecycle tests in test_video.py cover the invalid side of validation
through the backend. These cover the boundaries themselves: the values the
interface promises to accept, and the terminal-state contract the backend
relies on when it decides whether a job can still change.
"""

import pytest
from pydantic import ValidationError

from video_mcp.models import (
    MAX_DURATION_SECONDS,
    MIN_DURATION_SECONDS,
    JobStatus,
    VideoRequest,
)

ASPECT_RATIOS = ["16:9", "9:16", "1:1", "4:3", "21:9"]


@pytest.mark.parametrize("duration", [MIN_DURATION_SECONDS, MAX_DURATION_SECONDS])
def test_duration_limits_are_inclusive(duration):
    assert VideoRequest(prompt="a city", duration=duration).duration == duration


@pytest.mark.parametrize(
    "duration", [MIN_DURATION_SECONDS - 1, MAX_DURATION_SECONDS + 1]
)
def test_duration_outside_the_limits_is_rejected(duration):
    with pytest.raises(ValidationError):
        VideoRequest(prompt="a city", duration=duration)


@pytest.mark.parametrize("aspect_ratio", ASPECT_RATIOS)
def test_every_advertised_aspect_ratio_is_accepted(aspect_ratio):
    request = VideoRequest(prompt="a city", aspect_ratio=aspect_ratio)

    assert request.aspect_ratio == aspect_ratio


def test_defaults_match_the_documented_tool_signature():
    request = VideoRequest(prompt="a city")

    assert request.duration == 5
    assert request.aspect_ratio == "16:9"


def test_a_prompt_at_the_length_limit_is_accepted():
    prompt = "a" * 2000

    assert VideoRequest(prompt=prompt).prompt == prompt


def test_a_prompt_over_the_length_limit_is_rejected():
    with pytest.raises(ValidationError):
        VideoRequest(prompt="a" * 2001)


@pytest.mark.parametrize(
    ("status", "terminal"),
    [
        (JobStatus.QUEUED, False),
        (JobStatus.RUNNING, False),
        (JobStatus.COMPLETED, True),
        (JobStatus.CANCELLED, True),
        (JobStatus.FAILED, True),
    ],
)
def test_terminal_states_are_the_ones_that_cannot_change(status, terminal):
    assert status.is_terminal is terminal
