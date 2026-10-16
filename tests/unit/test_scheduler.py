"""IntervalScheduler (PLAN §5.7, B4): fire-on-start, interval + jitter, overlap guard, error handling,
`last_run_at` only on successful runs, graceful stop, metrics."""

from __future__ import annotations

import asyncio
import random
import time

import pytest

from support_agent.observability import metrics as m
from tests.conftest import require

IntervalScheduler = require("support_agent.pipeline.scheduler", "IntervalScheduler")


def metric(name: str) -> float:
    return m.REGISTRY.get_sample_value(name) or 0.0


class Job:
    def __init__(self, duration: float = 0.0, fail: bool | int = False) -> None:
        self.duration = duration
        self.fail = fail  # True = always, int = first N calls
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self.completed = 0

    async def __call__(self):
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.duration:
                await asyncio.sleep(self.duration)
            if self.fail is True or (isinstance(self.fail, int) and self.fail and self.calls <= self.fail):
                raise RuntimeError(f"poll failed #{self.calls}")
            self.completed += 1
        finally:
            self.active -= 1


@pytest.fixture
async def sched_factory():
    made = []

    def make(interval=0.05, jitter=0.0, job=None, **kw):
        s = IntervalScheduler(interval, jitter, job or Job(), **kw)
        made.append(s)
        return s

    yield make
    for s in made:
        await s.stop()


# --------------------------------------------------------------------------- construction + jitter


@pytest.mark.parametrize("interval", [0, -1])
def test_invalid_interval(interval):
    with pytest.raises(ValueError):
        IntervalScheduler(interval, 0.1, Job())


def test_jitter_clamped():
    assert IntervalScheduler(1, -0.5, Job()).jitter_ratio == 0.0
    assert IntervalScheduler(1, 5, Job()).jitter_ratio < 1.0


def test_next_delay_within_jitter_bounds():
    s = IntervalScheduler(15, 0.1, Job(), rng=random.Random(42))
    delays = [s.next_delay() for _ in range(2000)]
    assert min(delays) >= 13.5 and max(delays) <= 16.5
    assert max(delays) - min(delays) > 2.0  # actually spreads
    assert abs(sum(delays) / len(delays) - 15) < 0.2


def test_no_jitter_is_exact():
    s = IntervalScheduler(2.5, 0.0, Job())
    assert {s.next_delay() for _ in range(50)} == {2.5}


def test_seeded_rng_is_deterministic():
    a = IntervalScheduler(10, 0.1, Job(), rng=random.Random(7))
    b = IntervalScheduler(10, 0.1, Job(), rng=random.Random(7))
    assert [a.next_delay() for _ in range(10)] == [b.next_delay() for _ in range(10)]


# --------------------------------------------------------------------------- ticking


async def test_fires_immediately_on_start(sched_factory):
    job = Job()
    s = sched_factory(interval=10, job=job)
    await s.start()
    await asyncio.sleep(0.05)
    assert job.calls == 1 and s.ticks == 1


async def test_fires_every_interval(sched_factory):
    job = Job()
    s = sched_factory(interval=0.05, job=job)
    ticks0 = metric("scheduler_ticks_total")
    await s.start()
    await asyncio.sleep(0.33)
    await s.stop()
    assert 5 <= job.calls <= 9
    assert metric("scheduler_ticks_total") - ticks0 == s.ticks


async def test_start_is_idempotent(sched_factory):
    job = Job()
    s = sched_factory(interval=10, job=job)
    await s.start()
    await s.start()
    await asyncio.sleep(0.05)
    assert job.calls == 1


async def test_sync_job_supported(sched_factory):
    calls = []
    s = sched_factory(interval=10, job=lambda: calls.append(1))
    await s.start()
    await asyncio.sleep(0.05)
    assert calls == [1] and s.last_run_at is not None


# --------------------------------------------------------------------------- overlap guard


async def test_never_overlaps_and_counts_skips(sched_factory):
    job = Job(duration=0.2)
    s = sched_factory(interval=0.03, job=job)
    skipped0 = metric("scheduler_skipped_total")
    await s.start()
    await asyncio.sleep(0.5)
    await s.stop()
    assert job.max_active == 1
    assert s.skipped >= 5
    assert s.ticks == s.skipped + job.calls
    assert metric("scheduler_skipped_total") - skipped0 == s.skipped


async def test_run_now_respects_guard(sched_factory):
    job = Job(duration=0.2)
    s = sched_factory(interval=10, job=job)
    await s.start()
    await asyncio.sleep(0.02)
    assert s.running
    assert await s.run_now() is False
    await asyncio.sleep(0.25)
    assert not s.running
    assert await s.run_now() is True
    await asyncio.sleep(0.25)
    assert job.calls == 2


# --------------------------------------------------------------------------- errors + last_run_at


async def test_job_error_does_not_stop_loop(sched_factory):
    job = Job(fail=True)
    s = sched_factory(interval=0.03, job=job)
    await s.start()
    await asyncio.sleep(0.2)
    assert s.started
    assert job.calls >= 3 and s.errors == job.calls
    assert s.last_error.startswith("RuntimeError: poll failed")


async def test_last_run_at_only_for_successful_runs(sched_factory):
    job = Job(fail=True)
    s = sched_factory(interval=0.03, job=job)
    await s.start()
    await asyncio.sleep(0.15)
    assert s.last_run_at is None  # every run raised
    assert s.last_finished_at is not None  # but runs did finish
    assert s.runs >= 2


async def test_success_after_errors_sets_last_run_and_clears_error(sched_factory):
    job = Job(fail=2)
    s = sched_factory(interval=0.03, job=job)
    before = time.time()
    await s.start()
    await asyncio.sleep(0.2)
    assert job.completed >= 1
    assert s.last_error is None
    assert s.last_run_at is not None and s.last_run_at >= before
    assert metric("scheduler_last_run_timestamp") == pytest.approx(s.last_run_at)


async def test_failure_after_success_keeps_last_run_at(sched_factory):
    state = {"n": 0}

    async def job():
        state["n"] += 1
        if state["n"] > 1:
            raise RuntimeError("later failure")

    s = sched_factory(interval=0.04, job=job)
    await s.start()
    await asyncio.sleep(0.02)
    first_ok = s.last_run_at
    assert first_ok is not None
    await asyncio.sleep(0.15)
    assert state["n"] >= 3
    assert s.last_run_at == first_ok  # failures never advance it
    assert s.last_error == "RuntimeError: later failure"


async def test_last_duration_recorded(sched_factory):
    s = sched_factory(interval=10, job=Job(duration=0.05))
    await s.start()
    await asyncio.sleep(0.1)
    assert s.last_duration_s >= 0.04


# --------------------------------------------------------------------------- stop


async def test_stop_waits_for_inflight_run(sched_factory):
    job = Job(duration=0.15)
    s = sched_factory(interval=10, job=job, stop_timeout_s=2)
    await s.start()
    await asyncio.sleep(0.02)
    await s.stop()
    assert job.completed == 1 and not s.running and not s.started


async def test_stop_cancels_after_timeout(sched_factory):
    job = Job(duration=5)
    s = sched_factory(interval=10, job=job, stop_timeout_s=0.1)
    await s.start()
    await asyncio.sleep(0.02)
    t0 = time.perf_counter()
    await s.stop()
    assert time.perf_counter() - t0 < 1.0
    assert job.completed == 0 and job.active == 0


async def test_no_ticks_after_stop(sched_factory):
    job = Job()
    s = sched_factory(interval=0.03, job=job)
    await s.start()
    await asyncio.sleep(0.1)
    await s.stop()
    n = job.calls
    await asyncio.sleep(0.12)
    assert job.calls == n


async def test_stop_without_start_and_restart(sched_factory):
    job = Job()
    s = sched_factory(interval=10, job=job)
    await s.stop()
    await s.start()
    await asyncio.sleep(0.03)
    await s.stop()
    await s.start()
    await asyncio.sleep(0.03)
    assert job.calls == 2
