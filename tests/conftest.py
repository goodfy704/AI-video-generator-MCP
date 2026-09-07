"""Shared test fixtures."""

import pytest

# Mock timings the tests pin, independent of the server's runtime defaults.
QUEUED_SECONDS = 2.0
RUNNING_SECONDS = 8.0


class FakeClock:
    """A manually advanced clock, so job progression is deterministic."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()
