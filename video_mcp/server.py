"""FastMCP server exposing the video-generation tools.

This layer is intentionally thin: it validates arguments, delegates to the
backend, and translates backend errors into tool errors the LLM can act on.
All generation logic lives in `video_mcp.video`.
"""

from collections.abc import Generator
from contextlib import contextmanager

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import ValidationError

from video_mcp.models import VideoJob, VideoRequest, VideoResult
from video_mcp.video import VideoServiceError, build_backend

mcp = FastMCP("AI Video Generator")

backend = build_backend()


@mcp.tool
def create_video(
    prompt: str,
    duration: int = 5,
    aspect_ratio: str = "16:9",
) -> VideoJob:
    """Start generating a video from a text prompt.

    Returns immediately with a job_id; the video is not ready yet. Poll
    get_video_status with that job_id until the status is 'completed', then
    call get_video_result to retrieve the video.

    Args:
        prompt: Description of the video to generate.
        duration: Length in seconds, from 1 to 60.
        aspect_ratio: One of '16:9', '9:16', '1:1', '4:3', '21:9'.
    """
    request = _validate(prompt=prompt, duration=duration, aspect_ratio=aspect_ratio)
    return backend.create(request)


@mcp.tool
def get_video_status(job_id: str) -> VideoJob:
    """Check the progress of a video-generation job.

    Use the job_id returned by create_video. Statuses are 'queued', 'running',
    'completed' and 'cancelled'. Keep polling while the job is queued or
    running.

    Args:
        job_id: Identifier returned by create_video.
    """
    with _tool_errors():
        return backend.status(job_id)


@mcp.tool
def get_video_result(job_id: str) -> VideoResult:
    """Retrieve the finished video for a completed job.

    Only works once get_video_status reports 'completed'. Calling it earlier
    is an error, not a wait.

    Args:
        job_id: Identifier returned by create_video.
    """
    with _tool_errors():
        return backend.result(job_id)


@mcp.tool
def cancel_video(job_id: str) -> VideoJob:
    """Cancel a video-generation job that has not finished yet.

    Jobs that are already completed or cancelled cannot be cancelled.

    Args:
        job_id: Identifier returned by create_video.
    """
    with _tool_errors():
        return backend.cancel(job_id)


def _validate(prompt: str, duration: int, aspect_ratio: str) -> VideoRequest:
    """Build a validated request, reporting problems in terms the LLM can fix."""
    try:
        return VideoRequest(prompt=prompt, duration=duration, aspect_ratio=aspect_ratio)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        raise ToolError(f"Invalid video request ({problems}).") from exc


@contextmanager
def _tool_errors() -> Generator[None]:
    """Translate backend errors into ToolError so the LLM sees the reason."""
    try:
        yield
    except VideoServiceError as exc:
        raise ToolError(str(exc)) from exc


if __name__ == "__main__":
    mcp.run(transport="stdio")
