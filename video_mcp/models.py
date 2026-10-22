from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

Prompt = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=2000,
    ),
]
VideoDuration = Literal[5, 10, 15, 30]
FiveSeconds = Annotated[int, Field(strict=True, ge=5, le=5)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ImageAsset(Contract):
    asset_id: UUID
    mime_type: Literal["image/jpeg"] = "image/jpeg"
    width: Literal[1280] = 1280
    height: Literal[720] = 720
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class GenerationPlan(Contract):
    asset_id: UUID
    prompt: Prompt
    model: Literal["ltx-2-5-fast"] = "ltx-2-5-fast"
    duration: VideoDuration = 5
    story_beats: list[Prompt] = Field(default_factory=list, max_length=6)
    aspect_ratio: Literal["16:9"] = "16:9"
    audio: Literal["silent"] = "silent"

    @field_validator("duration", mode="before")
    @classmethod
    def validate_duration_type(cls, value: object) -> int:
        if type(value) is not int or value not in (5, 10, 15, 30):
            raise ValueError("duration must be one of 5, 10, 15 or 30 seconds.")
        return value

    @field_validator("story_beats")
    @classmethod
    def validate_beats(cls, value: list[str], info) -> list[str]:
        duration = info.data.get("duration", 5)
        if (value or duration > 5) and len(value) != duration // 5:
            raise ValueError("story_beats must contain one beat per five seconds.")
        return value


class VideoEstimate(Contract):
    currency: Literal["USD"] = "USD"
    estimated_cost: float = 0.54
    # The LTX API bills a 6-second minimum; self-hosted renders exactly 5 s.
    generated_seconds: int = 6
    delivered_seconds: VideoDuration = 5
    basis: str = "LTX direct API, 720p; excludes taxes."


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class VideoJob(Contract):
    job_id: UUID
    plan: GenerationPlan
    status: JobStatus
    comfy_id: str | None = None
    provider_job_id: str | None = None
    progress: int | None = Field(default=None, ge=0, le=100)
    message: str = ""
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )


class VideoResult(Contract):
    job_id: UUID
    status: Literal["completed"] = "completed"
    duration: VideoDuration = 5
    aspect_ratio: Literal["16:9"] = "16:9"
    video_path: str
    video_url: str | None = None
