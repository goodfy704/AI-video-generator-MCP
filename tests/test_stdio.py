"""Tests that launch the server the way an MCP client does.

Every other test imports the server in-process. These spawn it as a real
subprocess over stdio, which is the only place two whole classes of breakage
show up: the server failing to start as a module, and anything printing to
stdout, where it would corrupt the JSON-RPC stream and close the connection.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from tests.test_server import EXPECTED_TOOLS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# A client's first message. The server must answer this and nothing else on
# stdout.
INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "1"},
    },
}


def test_the_server_starts_as_a_module_and_serves_its_tools():
    """`python -m video_mcp.server` from the project root, as mcp.json runs it."""

    async def _drive():
        transport = StdioTransport(
            command=sys.executable,
            args=["-m", "video_mcp.server"],
            cwd=str(PROJECT_ROOT),
            env={
                "AI_VIDEO_MOCK_QUEUED_SECONDS": "0",
                "AI_VIDEO_MOCK_RUNNING_SECONDS": "0",
            },
        )
        async with Client(transport) as client:
            tools = {tool.name for tool in await client.list_tools()}
            job = (await client.call_tool("create_video", {"prompt": "a city"})).data
            result = (
                await client.call_tool("get_video_result", {"job_id": job.job_id})
            ).data
            return tools, result

    tools, result = asyncio.run(asyncio.wait_for(_drive(), timeout=60))

    assert tools == EXPECTED_TOOLS
    assert result.video_path.endswith(".mp4")


def test_nothing_but_json_rpc_reaches_stdout():
    """A stray print() in the tool layer would break every MCP client."""
    process = subprocess.run(
        [sys.executable, "-m", "video_mcp.server"],
        cwd=PROJECT_ROOT,
        input=(json.dumps(INITIALIZE) + "\n").encode(),
        capture_output=True,
        timeout=60,
    )

    stdout = process.stdout.decode("utf-8")
    lines = [line for line in stdout.splitlines() if line.strip()]

    assert lines, "the server wrote nothing to stdout"
    for line in lines:
        json.loads(line)  # raises if anything non-JSON was printed

    assert json.loads(lines[0])["result"]["serverInfo"]["name"] == "AI Video Generator"


@pytest.mark.parametrize("variable", ["AI_VIDEO_MOCK_QUEUED_SECONDS"])
def test_an_unusable_timing_value_does_not_stop_the_server_starting(variable):
    """A bad env value must degrade to the default, not kill the connection."""

    async def _list():
        transport = StdioTransport(
            command=sys.executable,
            args=["-m", "video_mcp.server"],
            cwd=str(PROJECT_ROOT),
            env={variable: "not-a-number"},
        )
        async with Client(transport) as client:
            return {tool.name for tool in await client.list_tools()}

    assert asyncio.run(asyncio.wait_for(_list(), timeout=60)) == EXPECTED_TOOLS
