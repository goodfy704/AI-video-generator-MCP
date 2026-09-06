"""Tests for the mock video backend."""

import pytest

from tests.conftest import QUEUED_SECONDS, RUNNING_SECONDS, FakeClock
from video_mcp.models import JobStatus, VideoRequest
from video_mcp.video import (
    JobNotCancellableError,
    JobNotFoundError,
    JobNotReadyError,
    MockVideoBackend,
)


@pytest.fixture
def backend(clock: FakeClock) -> MockVideoBackend:
    return MockVideoBackend(
        clock=clock,
        queued_seconds=QUEUED_SECONDS,
        running_seconds=RUNNING_SECONDS,
    )


@pytest.fixture
def request_() -> VideoRequest:
    return VideoRequest(
        prompt="a futuristic city at night", duration=10, aspect_ratio="16:9"
    )


def test_new_job_starts_queued(backend, request_):
    job = backend.create(request_)

    assert job.status is JobStatus.QUEUED
    assert job.progress == 0
    assert job.prompt == "a futuristic city at night"
    assert job.duration == 10
    assert job.aspect_ratio == "16:9"


def test_each_job_gets_a_distinct_id(backend, request_):
    first = backend.create(request_)
    second = backend.create(request_)

    assert first.job_id != second.job_id


def test_job_progresses_through_the_full_lifecycle(backend, clock, request_):
    job = backend.create(request_)

    clock.advance(QUEUED_SECONDS - 0.1)
    assert backend.status(job.job_id).status is JobStatus.QUEUED

    clock.advance(0.2)
    running = backend.status(job.job_id)
    assert running.status is JobStatus.RUNNING
    assert 0 < running.progress < 100

    clock.advance(RUNNING_SECONDS)
    completed = backend.status(job.job_id)
    assert completed.status is JobStatus.COMPLETED
    assert completed.progress == 100


def test_progress_increases_while_running(backend, clock, request_):
    job = backend.create(request_)
    clock.advance(QUEUED_SECONDS + 1.0)
    early = backend.status(job.job_id).progress

    clock.advance(RUNNING_SECONDS / 2)
    later = backend.status(job.job_id).progress

    assert later > early


def test_status_of_unknown_job_is_an_error(backend):
    with pytest.raises(JobNotFoundError):
        backend.status("does-not-exist")


def test_result_is_unavailable_until_the_job_completes(backend, clock, request_):
    job = backend.create(request_)

    with pytest.raises(JobNotReadyError) as queued:
        backend.result(job.job_id)
    assert queued.value.status is JobStatus.QUEUED

    clock.advance(QUEUED_SECONDS + 1.0)
    with pytest.raises(JobNotReadyError) as running:
        backend.result(job.job_id)
    assert running.value.status is JobStatus.RUNNING


def test_result_is_available_once_the_job_completes(backend, clock, request_):
    job = backend.create(request_)
    clock.advance(QUEUED_SECONDS + RUNNING_SECONDS)

    result = backend.result(job.job_id)

    assert result.status is JobStatus.COMPLETED
    assert result.job_id == job.job_id
    assert result.video_path.endswith(f"{job.job_id}.mp4")
    assert result.prompt == request_.prompt


def test_result_of_unknown_job_is_an_error(backend):
    with pytest.raises(JobNotFoundError):
        backend.result("does-not-exist")


def test_cancelling_a_running_job_stops_it(backend, clock, request_):
    job = backend.create(request_)
    clock.advance(QUEUED_SECONDS + 1.0)

    cancelled = backend.cancel(job.job_id)
    assert cancelled.status is JobStatus.CANCELLED

    # Cancellation is sticky: time passing does not complete the job.
    clock.advance(RUNNING_SECONDS * 2)
    assert backend.status(job.job_id).status is JobStatus.CANCELLED


def test_cancelled_job_has_no_result(backend, request_):
    job = backend.create(request_)
    backend.cancel(job.job_id)

    with pytest.raises(JobNotReadyError):
        backend.result(job.job_id)


def test_completed_job_cannot_be_cancelled(backend, clock, request_):
    job = backend.create(request_)
    clock.advance(QUEUED_SECONDS + RUNNING_SECONDS)

    with pytest.raises(JobNotCancellableError):
        backend.cancel(job.job_id)


def test_job_cannot_be_cancelled_twice(backend, request_):
    job = backend.create(request_)
    backend.cancel(job.job_id)

    with pytest.raises(JobNotCancellableError):
        backend.cancel(job.job_id)


def test_cancelling_an_unknown_job_is_an_error(backend):
    with pytest.raises(JobNotFoundError):
        backend.cancel("does-not-exist")
