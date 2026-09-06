"""Tests for the MCP tool layer, exercised through an in-memory MCP client.

These cover what the LLM actually sees: the registered tool set, the structured
results, and the error messages returned when a call cannot be satisfied.
"""

import asyncio

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from tests.conftest import QUEUED_SECONDS, RUNNING_SECONDS, FakeClock
from video_mcp import server
from video_mcp.models import JobStatus
from video_mcp.video import MockVideoBackend

EXPECTED_TOOLS = {
    "create_video",
    "get_video_status",
    "get_video_result",
    "cancel_video",
}


@pytest.fixture(autouse=True)
def backend(monkeypatch, clock: FakeClock) -> MockVideoBackend:
    """Give every test its own backend on a clock it controls."""
    stub = MockVideoBackend(
        clock=clock,
        queued_seconds=QUEUED_SECONDS,
        running_seconds=RUNNING_SECONDS,
    )
    monkeypatch.setattr(server, "backend", stub)
    return stub


def call(tool: str, **arguments):
    """Call one tool through an in-memory MCP client and return its data."""

    async def _call():
        async with Client(server.mcp) as client:
            result = await client.call_tool(tool, arguments)
            return result.data

    return asyncio.run(_call())


def list_tools():
    async def _list():
        async with Client(server.mcp) as client:
            return await client.list_tools()

    return asyncio.run(_list())


# -- tool registration ---------------------------------------------------


def test_server_exposes_the_phase_1_tools():
    assert {tool.name for tool in list_tools()} == EXPECTED_TOOLS


def test_every_tool_is_described_for_the_llm():
    for tool in list_tools():
        assert tool.description, f"{tool.name} has no description"


def test_create_video_schema_advertises_the_valid_aspect_ratios():
    schema = next(t for t in list_tools() if t.name == "create_video").input_schema

    assert set(schema["required"]) == {"prompt"}
    assert schema["properties"]["duration"]["default"] == 5
    assert schema["properties"]["aspect_ratio"]["default"] == "16:9"


# -- happy path ----------------------------------------------------------


def test_create_video_returns_a_queued_job():
    job = call("create_video", prompt="a futuristic city at night", duration=10)

    assert job.status == JobStatus.QUEUED
    assert job.job_id
    assert job.duration == 10
    assert job.aspect_ratio == "16:9"


def test_full_workflow_from_prompt_to_video(clock):
    job = call("create_video", prompt="a futuristic city at night", duration=10)
    job_id = job.job_id

    clock.advance(QUEUED_SECONDS + 1.0)
    assert call("get_video_status", job_id=job_id).status == JobStatus.RUNNING

    clock.advance(RUNNING_SECONDS)
    assert call("get_video_status", job_id=job_id).status == JobStatus.COMPLETED

    result = call("get_video_result", job_id=job_id)
    assert result.video_path.endswith(f"{job_id}.mp4")


def test_cancel_video_stops_a_running_job(clock):
    job_id = call("create_video", prompt="a slow render").job_id
    clock.advance(QUEUED_SECONDS + 1.0)

    assert call("cancel_video", job_id=job_id).status == JobStatus.CANCELLED


# -- errors the LLM has to recover from ----------------------------------


def test_unknown_job_id_explains_how_to_get_a_valid_one():
    with pytest.raises(ToolError) as exc:
        call("get_video_status", job_id="does-not-exist")

    assert "create_video" in str(exc.value)


def test_requesting_a_result_too_early_says_to_keep_polling():
    job_id = call("create_video", prompt="a futuristic city").job_id

    with pytest.raises(ToolError) as exc:
        call("get_video_result", job_id=job_id)

    assert "get_video_status" in str(exc.value)


def test_cancelling_a_completed_job_is_rejected(clock):
    job_id = call("create_video", prompt="a futuristic city").job_id
    clock.advance(QUEUED_SECONDS + RUNNING_SECONDS)

    with pytest.raises(ToolError):
        call("cancel_video", job_id=job_id)


@pytest.mark.parametrize(
    "arguments",
    [
        {"prompt": ""},
        {"prompt": "a city", "duration": 0},
        {"prompt": "a city", "duration": 61},
        {"prompt": "a city", "aspect_ratio": "5:4"},
    ],
    ids=["empty prompt", "duration too short", "duration too long", "bad ratio"],
)
def test_invalid_requests_are_rejected(arguments):
    with pytest.raises(ToolError):
        call("create_video", **arguments)


def test_a_rejected_request_creates_no_job(backend):
    with pytest.raises(ToolError):
        call("create_video", prompt="a city", duration=999)

    assert backend._jobs == {}
