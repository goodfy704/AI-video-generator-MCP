"""Image-to-video on a RunPod serverless endpoint running LTX-2.5 in ComfyUI.

The endpoint runs the LTX 2.5 ComfyUI worker (github.com/vavo/LTX2.5-serverless).
Its handler takes a ComfyUI API workflow plus base64 input images on /run and
returns the rendered video inline as base64. ltx25_i2v_workflow.json is that
worker's checked-in graph with the Gemma prompt-enhancer branch and the audio
decode removed: the MCP client's LLM already writes the prompt, the contract
is silent video, and both would otherwise be paid GPU seconds.

RunPod bills worker time (cold start + render + idle timeout), not output
length, so this backend refuses a second submission while one is active and
caps submissions per day.
"""

import base64
import contextlib
import copy
import json
import logging
import os
import random
import shutil
import struct
import subprocess
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import httpx

from video_mcp.models import GenerationPlan, JobStatus, VideoEstimate, VideoJob
from video_mcp.video import ComfyBackend, VideoServiceError

logger = logging.getLogger(__name__)

API_URL = "https://api.runpod.ai/v2"
WORKFLOW = json.loads(
    Path(__file__).with_name("ltx25_i2v_workflow.json").read_text(encoding="utf-8")
)
NODES = {
    "image": "395",
    "prompt": "398:376",
    "width": "398:372",
    "height": "398:360",
    "seconds": "398:362",
    "fps": "398:361",
    "resize": "398:351",
    "low_res_noise": "398:339",
    "full_res_noise": "398:338",
    "output": "75",
}
SECONDS = 5
FPS = 24
ACTIVE = {JobStatus.QUEUED, JobStatus.RUNNING}


def parse_resolution(value: str) -> tuple[int, int]:
    """Parse WIDTHxHEIGHT for the two-stage graph.

    Stage one samples at half size and the latent upscaler doubles it, so
    each side must be divisible by 64 for the half to stay a multiple of 32.
    """
    try:
        width, height = (int(part) for part in value.lower().split("x"))
    except ValueError as exc:
        raise ValueError(f"RUNPOD_RESOLUTION must look like 1024x576: {value}") from exc

    if (
        width % 64
        or height % 64
        or not 512 <= width <= 1920
        or abs(width * 9 - height * 16) > height * 16 * 0.04
    ):
        raise ValueError(
            "RUNPOD_RESOLUTION must be a landscape ~16:9 size from 512 to 1920 "
            f"pixels wide with both sides divisible by 64: {value}"
        )
    return width, height


def build_workflow(
    *,
    job_id: UUID,
    prompt: str,
    image_name: str,
    width: int,
    height: int,
    seeds: tuple[int, int],
    prompt_suffix: str = "",
) -> dict:
    workflow = copy.deepcopy(WORKFLOW)
    values = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "seconds": SECONDS,
        "fps": FPS,
    }
    for name, value in values.items():
        workflow[NODES[name]]["inputs"]["value"] = value

    workflow[NODES["image"]]["inputs"]["image"] = image_name
    # The asset is already 16:9; resize straight to the render size.
    workflow[NODES["resize"]]["inputs"]["resize_type.longer_size"] = width
    workflow[NODES["low_res_noise"]]["inputs"]["noise_seed"] = seeds[0]
    workflow[NODES["full_res_noise"]]["inputs"]["noise_seed"] = seeds[1]
    workflow[NODES["output"]]["inputs"]["filename_prefix"] = f"avg/{job_id}{prompt_suffix}"
    return workflow


def _boxes(content: bytes, start: int, end: int):
    position = start
    while position + 8 <= end:
        size, kind = struct.unpack_from(">I4s", content, position)
        header = 8
        if size == 1:
            (size,) = struct.unpack_from(">Q", content, position + 8)
            header = 16
        elif size == 0:
            size = end - position
        if size < header or position + size > end:
            return
        yield kind, position + header, position + size
        position += size


def probe_mp4(content: bytes) -> tuple[float, int, int]:
    """Return (duration in seconds, width, height) read from an MP4's moov box."""
    top = {kind: (start, end) for kind, start, end in _boxes(content, 0, len(content))}
    if b"ftyp" not in top or b"moov" not in top:
        raise ValueError("not an MP4 file")

    duration = 0.0
    width = height = 0
    for kind, start, end in _boxes(content, *top[b"moov"]):
        if kind == b"mvhd":
            if content[start] == 1:
                timescale, length = struct.unpack_from(">IQ", content, start + 20)
            else:
                timescale, length = struct.unpack_from(">II", content, start + 12)
            duration = length / timescale if timescale else 0.0
        elif kind == b"trak":
            for child, _, child_end in _boxes(content, start, end):
                if child == b"tkhd":
                    track_width, track_height = struct.unpack_from(
                        ">II", content, child_end - 8
                    )
                    if track_width and track_height:
                        width, height = track_width >> 16, track_height >> 16

    if duration <= 0 or not width or not height:
        raise ValueError("MP4 has no video track or duration")
    return duration, width, height


class RunPodBackend(ComfyBackend):
    def __init__(self, root: Path | None = None):
        super().__init__(root)
        self.api_key = os.getenv("RUNPOD_API_KEY", "").strip()
        self.endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID", "").strip()
        self.width, self.height = parse_resolution(
            os.getenv("RUNPOD_RESOLUTION", "1024x576")
        )
        # 48 GB Ampere (A40 / RTX A6000) flex price, per second of worker time.
        self.price_per_second = float(os.getenv("RUNPOD_PRICE_PER_SECOND", "0.00034"))
        # Worker start + model load + render. The first measured cold job
        # billed 106 s ($0.037) at 1024x576; the margin covers slower hosts.
        self.expected_seconds = float(os.getenv("RUNPOD_EXPECTED_SECONDS", "150"))
        self.daily_limit = int(os.getenv("RUNPOD_MAX_JOBS_PER_DAY", "10"))
        self.execution_timeout = int(
            os.getenv("RUNPOD_EXECUTION_TIMEOUT_SECONDS", "600")
        )
        self.job_ttl = int(os.getenv("RUNPOD_JOB_TTL_SECONDS", "1800"))
        (self.root / "runpod").mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._watcher: threading.Thread | None = None

    @property
    def estimated_cost(self) -> float:
        return round(self.expected_seconds * self.price_per_second, 3)

    def describe(self) -> dict:
        return {
            "model": "ltx-2-5-fast",
            "engine": "LTX-2.5 22B distilled INT8, ComfyUI on RunPod serverless",
            "duration": SECONDS,
            "durations": [5, 10, 15, 30],
            "aspect_ratio": "16:9",
            "resolution": f"{self.width}x{self.height}",
            "audio": "silent",
            "generated_seconds": SECONDS,
            "estimated_usd": self.estimated_cost,
            "backend": "runpod",
            "notice": (
                "Generates a real video from the registered image. RunPod bills "
                "GPU worker time, including a cold start of a few minutes."
            ),
        }

    def _api(self, method: str, path: str, *, timeout: float = 30, **kwargs):
        if not self.api_key or not self.endpoint_id:
            raise VideoServiceError(
                "Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID for the RunPod backend."
            )
        return httpx.request(
            method,
            f"{API_URL}/{self.endpoint_id}{path}",
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=timeout,
            **kwargs,
        )

    def health(self) -> dict:
        try:
            response = self._api("GET", "/health", timeout=15)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise VideoServiceError(
                f"RunPod health check returned HTTP {exc.response.status_code}."
            ) from exc
        except (httpx.RequestError, ValueError) as exc:
            raise VideoServiceError("Cannot reach the RunPod API.") from exc

    def _jobs(self):
        for path in (self.root / "jobs").glob("*.json"):
            try:
                yield VideoJob.model_validate_json(path.read_text())
            except ValueError:
                logger.warning("Skipping unreadable job file %s.", path.name)

    def _record(self, job_id: UUID, **fields):
        path = self.root / "runpod" / f"{job_id}.json"
        record = json.loads(path.read_text()) if path.is_file() else {}
        record.update(fields)
        self._write(path, json.dumps(record, indent=2).encode())

    def _check_spend_limits(self, requested_segments: int = 1):
        jobs = list(self._jobs())
        active = next(
            (job for job in jobs if job.status in ACTIVE and job.provider_job_id), None
        )
        if active:
            raise VideoServiceError(
                f"RunPod job {active.job_id} is still {active.status.value}. Wait for "
                "it to finish or cancel it before submitting another."
            )

        since = datetime.now(UTC) - timedelta(days=1)
        recent = 0
        for prior in jobs:
            if prior.created_at < since or not (prior.provider_job_id or prior.status is JobStatus.UNKNOWN):
                continue
            path = self.root / "runpod" / f"{prior.job_id}.json"
            details = json.loads(path.read_text()) if path.is_file() else {}
            recent += int(details.get("segment", 0)) + 1
        if recent + requested_segments > self.daily_limit:
            raise VideoServiceError(
                f"Daily RunPod limit reached ({self.daily_limit} jobs in 24 hours). "
                "Raise RUNPOD_MAX_JOBS_PER_DAY to allow more."
            )

        queue = self.health().get("jobs", {})
        busy = queue.get("inQueue", 0) + queue.get("inProgress", 0)
        if busy:
            raise VideoServiceError(
                f"The RunPod endpoint already has {busy} queued or running job(s). "
                "Wait for them so this request does not pay for a second render."
            )

    def estimate(self, plan: GenerationPlan) -> VideoEstimate:
        self.validate_plan(plan)
        return VideoEstimate(
            estimated_cost=round(self.estimated_cost * (plan.duration // SECONDS), 3),
            generated_seconds=plan.duration,
            delivered_seconds=plan.duration,
            basis=(
                f"{plan.duration // SECONDS} sequential RunPod renders, about "
                f"{self.expected_seconds:.0f} worker seconds each at "
                f"${self.price_per_second}/s; excludes taxes."
            ),
        )

    def create(self, plan: GenerationPlan) -> VideoJob:
        plan, asset, content = self.validate_plan(plan)
        if plan.duration > SECONDS and not shutil.which("ffmpeg"):
            raise VideoServiceError("ffmpeg is required for videos longer than five seconds.")

        with self._lock:
            self._check_spend_limits(plan.duration // SECONDS)

            job = VideoJob(
                job_id=uuid4(),
                plan=plan,
                status=JobStatus.UNKNOWN,
                message="Submission outcome is not yet recorded.",
            )
            self._save("jobs", job.job_id, job)

            return self._submit_segment(job, content, 0)

    def _submit_segment(self, job: VideoJob, content: bytes, segment: int) -> VideoJob:
            plan = job.plan
            rng = random.SystemRandom()
            seeds = (rng.randrange(1, 2**48), rng.randrange(1, 2**48))
            image_name = f"{plan.asset_id}.jpg" if segment == 0 else f"{job.job_id}-{segment}.jpg"
            beats = plan.story_beats or [plan.prompt] * (plan.duration // SECONDS)
            payload = {
                "input": {
                    "workflow": build_workflow(
                        job_id=job.job_id,
                        prompt=beats[segment],
                        image_name=image_name,
                        width=self.width,
                        height=self.height,
                        seeds=seeds,
                        prompt_suffix=f"-{segment + 1}" if plan.duration > SECONDS else "",
                    ),
                    "images": [{
                        "name": image_name,
                        "image": "data:image/jpeg;base64,"
                        + base64.b64encode(content).decode(),
                    }],
                },
                "policy": {
                    "executionTimeout": self.execution_timeout * 1000,
                    "ttl": self.job_ttl * 1000,
                },
            }
            self._record(
                job.job_id,
                endpoint_id=self.endpoint_id,
                submitted_at=datetime.now(UTC).isoformat(),
                width=self.width,
                height=self.height,
                frames=SECONDS * FPS + 1,
                fps=FPS,
                seeds=list(seeds),
                segment=segment,
                submission_pending_segment=segment,
                segment_count=plan.duration // SECONDS,
                prompt=beats[segment],
            )

            unknown = (
                "RunPod did not confirm the submission, so it may be running. "
                "Check the endpoint's requests in the RunPod console before "
                "creating another job."
            )
            try:
                response = self._api("POST", "/run", json=payload, timeout=60)
            except httpx.RequestError:
                job = job.model_copy(update={"status": JobStatus.UNKNOWN, "message": unknown})
            else:
                provider_id = None
                if response.is_success:
                    with contextlib.suppress(ValueError):
                        provider_id = response.json().get("id")

                if provider_id:
                    job = job.model_copy(update={
                        "provider_job_id": provider_id,
                        "status": JobStatus.QUEUED,
                        "progress": 0,
                        "message": (
                            f"Generating segment {segment + 1} of {plan.duration // SECONDS}."
                        ),
                    })
                    self._record(job.job_id, provider_job_id=provider_id, submission_pending_segment=None)
                elif 400 <= response.status_code < 500:
                    self._record(job.job_id, submission_pending_segment=None)
                    job = job.model_copy(update={
                        "status": JobStatus.FAILED,
                        "message": (
                            f"RunPod rejected the submission (HTTP "
                            f"{response.status_code}); nothing was billed. Check "
                            "RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID."
                        ),
                    })
                else:
                    job = job.model_copy(update={"status": JobStatus.UNKNOWN, "message": unknown})

            self._save("jobs", job.job_id, job)
            return job

    def status(self, job_id: UUID) -> VideoJob:
        with self._lock:
            job = self._load("jobs", job_id, VideoJob)
            record_path = self.root / "runpod" / f"{job.job_id}.json"
            record = json.loads(record_path.read_text()) if record_path.is_file() else {}
            recorded_provider = record.get("provider_job_id")
            if recorded_provider and recorded_provider != job.provider_job_id and record.get("submission_pending_segment") is None:
                job = job.model_copy(update={"provider_job_id": recorded_provider, "status": JobStatus.QUEUED, "message": "Resuming the next storyboard segment."})
                self._save("jobs", job.job_id, job)
            elif record.get("submission_pending_segment") is not None and int(record["submission_pending_segment"]) > 0:
                job = job.model_copy(update={"status": JobStatus.UNKNOWN, "message": "A segment submission was interrupted. Check RunPod before retrying."})
                self._save("jobs", job.job_id, job)
                return job
            if job.status not in ACTIVE or not job.provider_job_id:
                return job

            try:
                response = self._api(
                    "GET", f"/status/{job.provider_job_id}", timeout=120
                )
            except httpx.RequestError as exc:
                raise VideoServiceError(
                    "Cannot reach the RunPod API. The job is unaffected; try again."
                ) from exc

            data = {}
            if response.status_code == 404:
                job = job.model_copy(update={
                    "status": JobStatus.UNKNOWN,
                    "progress": None,
                    "message": (
                        "RunPod no longer has this job. Results expire 30 minutes "
                        "after completion; check the RunPod console."
                    ),
                })
            elif not response.is_success:
                raise VideoServiceError(
                    f"RunPod status check returned HTTP {response.status_code}."
                )
            else:
                data = response.json()
                job = self._apply_remote_state(job, data)

            self._save("jobs", job.job_id, job)
            if job.status not in ACTIVE:
                self._record(
                    job.job_id,
                    final_status=job.status.value,
                    delay_ms=data.get("delayTime"),
                    execution_ms=data.get("executionTime"),
                    worker_id=data.get("workerId"),
                    message=job.message,
                )
            return job

    def _apply_remote_state(self, job: VideoJob, data: dict) -> VideoJob:
        state = data.get("status")
        if state == "IN_QUEUE":
            return job.model_copy(update={
                "status": JobStatus.QUEUED,
                "progress": 0,
                "message": "Waiting for a RunPod worker to start.",
            })
        if state == "IN_PROGRESS":
            return job.model_copy(update={
                "status": JobStatus.RUNNING,
                "progress": None,
                "message": "Rendering on RunPod.",
            })
        if state == "COMPLETED":
            return self._store_output(job, data)
        if state == "CANCELLED":
            return job.model_copy(update={
                "status": JobStatus.CANCELLED,
                "progress": None,
                "message": "Cancelled on RunPod. Worker time already used is billed.",
            })
        if state in {"FAILED", "TIMED_OUT"}:
            output = data.get("output")
            detail = data.get("error") or (
                output.get("error") if isinstance(output, dict) else None
            )
            return job.model_copy(update={
                "status": JobStatus.FAILED,
                "progress": None,
                "message": (
                    f"RunPod job {state.lower().replace('_', ' ')}"
                    + (f": {str(detail)[:500]}" if detail else ".")
                    + " Worker time already used is billed."
                ),
            })
        return job

    def _store_output(self, job: VideoJob, data: dict) -> VideoJob:
        output = data.get("output")

        def failed(message: str) -> VideoJob:
            return job.model_copy(update={
                "status": JobStatus.FAILED,
                "progress": None,
                "message": message + " Worker time already used is billed.",
            })

        if not isinstance(output, dict) or output.get("status") != "success":
            detail = output.get("error") if isinstance(output, dict) else output
            return failed(f"The RunPod worker reported an error: {str(detail)[:500]}")

        files = output.get("output") or {}
        entries = [*files.get("videos", []), *files.get("images", [])]
        video = next(
            (
                entry
                for entry in entries
                if str(entry.get("media_type", "")).startswith("video/")
                or str(entry.get("filename", "")).endswith(".mp4")
            ),
            None,
        )
        if video is None:
            return failed("The RunPod job finished without a video file.")
        if video.get("type") != "base64":
            return failed("RunPod returned the video by URL; expected inline base64.")

        content = base64.b64decode(video.get("data", ""))
        try:
            duration, width, height = probe_mp4(content)
        except (ValueError, struct.error):
            self._write(self.root / "outputs" / f"{job.job_id}.unverified.mp4", content)
            return failed(
                "RunPod returned a file that is not a readable MP4; it was kept as "
                f"{job.job_id}.unverified.mp4."
            )

        record_path = self.root / "runpod" / f"{job.job_id}.json"
        record = json.loads(record_path.read_text())
        segment = int(record.get("segment", 0))
        count = job.plan.duration // SECONDS
        if not 4.8 <= duration <= 5.3:
            return failed(f"Segment {segment + 1} had unexpected duration {duration:.2f} s.")
        if segment == 0 and count == 1:
            self._write(self.root / "outputs" / f"{job.job_id}.mp4", content)
        else:
            segment_path = self.root / "outputs" / f"{job.job_id}-{segment}.mp4"
            self._write(segment_path, content)
            if segment + 1 < count:
                if not shutil.which("ffmpeg"):
                    return failed("ffmpeg is required to continue the storyboard.")
                frame_path = self.root / "runpod" / f"{job.job_id}-{segment}.jpg"
                try:
                    subprocess.run(["ffmpeg", "-y", "-v", "error", "-sseof", "-0.1",
                                    "-i", str(segment_path), "-frames:v", "1", str(frame_path)],
                                   check=True, capture_output=True, timeout=60)
                    frame = frame_path.read_bytes()
                except (OSError, subprocess.SubprocessError) as exc:
                    return failed(f"Could not prepare segment {segment + 2}: {exc}")
                # The completed segment is durable before the next paid submission.
                self._record(job.job_id, completed_segment=segment)
                return self._submit_segment(job, frame, segment + 1)
            try:
                self._join_segments(job.job_id, count)
            except (OSError, ValueError, struct.error, subprocess.SubprocessError) as exc:
                return failed(f"Could not assemble the completed video: {exc}")
        waited = (data.get("delayTime") or 0) / 1000
        rendered = (data.get("executionTime") or 0) / 1000
        cost = (waited + rendered) * self.price_per_second
        return job.model_copy(update={
            "status": JobStatus.COMPLETED,
            "progress": 100,
            "message": (
                f"{job.plan.duration}-second video generated from the image is ready "
                f"({width}x{height}, {duration:.2f} s). RunPod waited {waited:.0f} s "
                f"for a worker and rendered for {rendered:.0f} s, at most about "
                f"${cost:.2f} of GPU time."
            ),
        })

    def _join_segments(self, job_id: UUID, count: int) -> None:
        if not shutil.which("ffmpeg"):
            raise OSError("ffmpeg is required to assemble the storyboard")
        output = self.root / "outputs" / f"{job_id}.mp4"
        command = ["ffmpeg", "-y", "-v", "error"]
        for index in range(count):
            command += ["-i", str(self.root / "outputs" / f"{job_id}-{index}.mp4")]
        streams = "".join(f"[{index}:v:0]fps=24,trim=duration=5,setpts=PTS-STARTPTS[v{index}];" for index in range(count))
        streams += "".join(f"[v{index}]" for index in range(count)) + f"concat=n={count}:v=1:a=0[out]"
        command += ["-filter_complex", streams, "-map", "[out]", "-an", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
        subprocess.run(command, check=True, capture_output=True, timeout=600)
        checked = probe_mp4(output.read_bytes())
        if abs(checked[0] - count * SECONDS) > 0.1:
            output.unlink(missing_ok=True)
            raise OSError("assembled video duration does not match the request")

    def cancel(self, job_id: UUID) -> VideoJob:
        with self._lock:
            job = self.status(job_id)
            if job.status not in ACTIVE or not job.provider_job_id:
                raise VideoServiceError(
                    f"Job is {job.status.value} and cannot be cancelled."
                )

            try:
                response = self._api(
                    "POST", f"/cancel/{job.provider_job_id}", timeout=30
                )
            except httpx.RequestError as exc:
                raise VideoServiceError(
                    "Cannot reach the RunPod API to cancel; try again."
                ) from exc
            if not response.is_success:
                raise VideoServiceError(
                    f"RunPod refused the cancellation (HTTP {response.status_code})."
                )

            # The job may have finished before the cancel landed.
            return self.status(job_id)

    def start_watcher(self, interval: float | None = None):
        """Poll active jobs in the background.

        Results are deleted 30 minutes after completion, so a video the client
        stopped polling for would otherwise be paid for and never saved.
        """
        if self._watcher is not None:
            return
        interval = interval or float(os.getenv("RUNPOD_POLL_SECONDS", "10"))

        def watch():
            while True:
                time.sleep(interval)
                for job in self._jobs():
                    if job.status in ACTIVE and job.provider_job_id:
                        try:
                            self.status(job.job_id)
                        except Exception:
                            logger.warning(
                                "Background status check for %s failed.",
                                job.job_id,
                                exc_info=True,
                            )

        self._watcher = threading.Thread(
            target=watch, name="runpod-watcher", daemon=True
        )
        self._watcher.start()
