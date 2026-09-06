"""Video-generation backend.

Phase 1 ships a mock backend only. It performs no real generation, but it does
keep genuine job state and advance it over time, so that an LLM driving the MCP
tools has to complete the full workflow: create a job, poll it while it is still
running, retrieve the result once it is ready, and recover from errors such as
an unknown job id.

The backend is accessed through the `VideoBackend` protocol so a real
implementation (local model, ComfyUI, remote API) can replace the mock without
touching the MCP tool layer.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from video_mcp.models import JobStatus, VideoJob, VideoRequest, VideoResult

# Default mock timings. Kept short enough that an interactive LLM session does
# not stall, but long enough that the model must actually poll for status
# instead of getting a completed job on the first call.
DEFAULT_QUEUED_SECONDS = 2.0
DEFAULT_RUNNING_SECONDS = 8.0

OUTPUT_DIR = "outputs"


class VideoServiceError(Exception):
    """Base class for errors the MCP layer turns into tool errors."""


class JobNotFoundError(VideoServiceError):
    def __init__(self, job_id: str) -> None:
        super().__init__(
            f"No video job exists with id {job_id!r}. "
            f"Call create_video first and use the job_id it returns."
        )
        self.job_id = job_id


class JobNotReadyError(VideoServiceError):
    def __init__(self, job_id: str, status: JobStatus) -> None:
        super().__init__(
            f"Job {job_id!r} is {status.value}, not completed. "
            f"Call get_video_status until the status is 'completed', "
            f"then request the result again."
        )
        self.job_id = job_id
        self.status = status


class JobNotCancellableError(VideoServiceError):
    def __init__(self, job_id: str, status: JobStatus) -> None:
        super().__init__(
            f"Job {job_id!r} cannot be cancelled because it is already {status.value}."
        )
        self.job_id = job_id
        self.status = status


@dataclass
class _JobRecord:
    """Internal bookkeeping for a single mock job."""

    job_id: str
    request: VideoRequest
    created_at: float
    cancelled: bool = field(default=False)


class VideoBackend(Protocol):
    """The operations the MCP tool layer depends on."""

    def create(self, request: VideoRequest) -> VideoJob: ...

    def status(self, job_id: str) -> VideoJob: ...

    def result(self, job_id: str) -> VideoResult: ...

    def cancel(self, job_id: str) -> VideoJob: ...


class MockVideoBackend:
    """In-memory backend that simulates a video-generation queue.

    Job state is derived from elapsed time rather than stored, so progress
    advances on its own between polls without any background task.
    """

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        queued_seconds: float = DEFAULT_QUEUED_SECONDS,
        running_seconds: float = DEFAULT_RUNNING_SECONDS,
    ) -> None:
        self._clock = clock
        self._queued_seconds = queued_seconds
        self._running_seconds = running_seconds
        self._jobs: dict[str, _JobRecord] = {}

    # -- queries ---------------------------------------------------------

    def _get(self, job_id: str) -> _JobRecord:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise JobNotFoundError(job_id) from None

    def _evaluate(self, record: _JobRecord) -> tuple[JobStatus, int]:
        """Return the job's current status and progress percentage."""
        if record.cancelled:
            return JobStatus.CANCELLED, 0

        elapsed = self._clock() - record.created_at
        if elapsed < self._queued_seconds:
            return JobStatus.QUEUED, 0

        if self._running_seconds <= 0:
            return JobStatus.COMPLETED, 100

        running_elapsed = elapsed - self._queued_seconds
        if running_elapsed >= self._running_seconds:
            return JobStatus.COMPLETED, 100

        # Report 1-99 while running, so progress is never mistaken for either
        # "not started" or "done".
        fraction = running_elapsed / self._running_seconds
        return JobStatus.RUNNING, max(1, min(99, int(fraction * 100)))

    def _to_job(self, record: _JobRecord) -> VideoJob:
        status, progress = self._evaluate(record)
        return VideoJob(
            job_id=record.job_id,
            status=status,
            progress=progress,
            prompt=record.request.prompt,
            duration=record.request.duration,
            aspect_ratio=record.request.aspect_ratio,
            message=_describe(status, progress),
        )

    # -- backend operations ----------------------------------------------

    def create(self, request: VideoRequest) -> VideoJob:
        record = _JobRecord(
            job_id=str(uuid.uuid4()),
            request=request,
            created_at=self._clock(),
        )
        self._jobs[record.job_id] = record
        return self._to_job(record)

    def status(self, job_id: str) -> VideoJob:
        return self._to_job(self._get(job_id))

    def result(self, job_id: str) -> VideoResult:
        record = self._get(job_id)
        status, _ = self._evaluate(record)
        if status is not JobStatus.COMPLETED:
            raise JobNotReadyError(job_id, status)

        return VideoResult(
            job_id=record.job_id,
            status=status,
            video_path=f"{OUTPUT_DIR}/{record.job_id}.mp4",
            prompt=record.request.prompt,
            duration=record.request.duration,
            aspect_ratio=record.request.aspect_ratio,
            message=(
                "Mock video generation completed. No file was written; this is "
                "a placeholder path for the Phase 1 proof of concept."
            ),
        )

    def cancel(self, job_id: str) -> VideoJob:
        record = self._get(job_id)
        status, _ = self._evaluate(record)
        if status.is_terminal:
            raise JobNotCancellableError(job_id, status)

        record.cancelled = True
        return self._to_job(record)


def _describe(status: JobStatus, progress: int) -> str:
    match status:
        case JobStatus.QUEUED:
            return "Job is queued and has not started yet."
        case JobStatus.RUNNING:
            return f"Job is generating the video ({progress}% complete)."
        case JobStatus.COMPLETED:
            return "Job is complete. Call get_video_result to retrieve the video."
        case JobStatus.CANCELLED:
            return "Job was cancelled."
        case _:
            return "Job failed."


def _float_env(name: str, default: float) -> float:
    """Read a non-negative float from the environment, falling back on error."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def build_backend() -> VideoBackend:
    """Create the backend the server uses.

    Mock timings can be shortened or lengthened for LLM evaluation runs via
    AI_VIDEO_MOCK_QUEUED_SECONDS and AI_VIDEO_MOCK_RUNNING_SECONDS.
    """
    return MockVideoBackend(
        queued_seconds=_float_env(
            "AI_VIDEO_MOCK_QUEUED_SECONDS", DEFAULT_QUEUED_SECONDS
        ),
        running_seconds=_float_env(
            "AI_VIDEO_MOCK_RUNNING_SECONDS", DEFAULT_RUNNING_SECONDS
        ),
    )
