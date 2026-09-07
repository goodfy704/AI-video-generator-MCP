"""Data models for the MCP video-generation interface.

These models define the contract between the LLM and the server. They are
deliberately independent of any particular video-generation backend, so the
same shapes hold whether the work is done by a mock, a local GPU model, or a
remote API.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

# The aspect ratios the interface accepts. Declared as a Literal so the value
# set is visible in the tool schema the LLM reads.
AspectRatio = Literal["16:9", "9:16", "1:1", "4:3", "21:9"]

MIN_DURATION_SECONDS = 1
MAX_DURATION_SECONDS = 60


class JobStatus(StrEnum):
    """Lifecycle of a video-generation job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        """Whether no further status change is possible."""
        return self in (JobStatus.COMPLETED, JobStatus.CANCELLED, JobStatus.FAILED)


class VideoRequest(BaseModel):
    """A validated request to generate a video."""

    prompt: str = Field(
        min_length=1,
        max_length=2000,
        description="Description of the video to generate.",
    )
    duration: int = Field(
        default=5,
        ge=MIN_DURATION_SECONDS,
        le=MAX_DURATION_SECONDS,
        description="Length of the video in seconds.",
    )
    aspect_ratio: AspectRatio = Field(
        default="16:9",
        description="Aspect ratio of the generated video.",
    )


class VideoJob(BaseModel):
    """The current state of a video-generation job."""

    job_id: str = Field(description="Identifier used to track this job.")
    status: JobStatus = Field(description="Current lifecycle state of the job.")
    progress: int = Field(
        ge=0,
        le=100,
        description="Completion percentage, 0-100.",
    )
    prompt: str = Field(description="Prompt the job was created from.")
    duration: int = Field(description="Requested video length in seconds.")
    aspect_ratio: str = Field(description="Requested aspect ratio.")
    message: str = Field(description="Human-readable summary of the job state.")


class VideoResult(BaseModel):
    """The output of a completed video-generation job."""

    job_id: str = Field(description="Identifier of the completed job.")
    status: JobStatus = Field(description="Always 'completed' for a result.")
    video_path: str = Field(description="Path to the generated video file.")
    prompt: str = Field(description="Prompt the video was generated from.")
    duration: int = Field(description="Video length in seconds.")
    aspect_ratio: str = Field(description="Aspect ratio of the video.")
    message: str = Field(description="Human-readable summary of the result.")
