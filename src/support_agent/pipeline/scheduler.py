"""IntervalScheduler — runs an async job every `interval_s` seconds (±jitter), never overlapping (PLAN §5.7).

* Fires once immediately on `start()`, then every `interval_s * (1 ± jitter_ratio)` seconds.
* Each tick increments `scheduler_ticks_total`. If the previous run is still in progress the tick is
  skipped (`scheduler_skipped_total`); runs never overlap.
* A job that raises is logged (and kept in `last_error`); the loop keeps going.
* `scheduler_last_run_timestamp` / `last_run_at` = wall-clock time of the last run that completed
  without raising (used by `/ready`).
* `stop()` stops ticking, waits up to `stop_timeout_s` for an in-flight run, then cancels it.

Generic: it knows nothing about the poller (the poller writes `scheduler_runs` rows itself).
"""

from __future__ import annotations

import asyncio
import inspect
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any

from support_agent.observability import metrics as m
from support_agent.observability.logging import get_logger

log = get_logger("scheduler")

Job = Callable[[], Awaitable[Any]]


class IntervalScheduler:
    def __init__(
        self,
        interval_s: float,
        jitter_ratio: float,
        job: Job,
        *,
        name: str = "scheduler",
        stop_timeout_s: float = 5.0,
        rng: random.Random | None = None,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be > 0")
        self.interval_s = float(interval_s)
        self.jitter_ratio = min(max(float(jitter_ratio), 0.0), 0.99)
        self.job = job
        self.name = name
        self.stop_timeout_s = stop_timeout_s
        self._rng = rng or random.Random()

        self._loop_task: asyncio.Task[None] | None = None
        self._job_task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()

        self.ticks = 0
        self.skipped = 0
        self.runs = 0  # completed runs (ok or error)
        self.errors = 0
        self.last_run_at: float | None = None  # unix ts of last successful completion
        self.last_finished_at: float | None = None  # unix ts of last completion, ok or error
        self.last_error: str | None = None
        self.last_duration_s: float | None = None

    # ------------------------------------------------------------------ state

    @property
    def running(self) -> bool:
        """True while a job run is in progress."""
        return self._job_task is not None and not self._job_task.done()

    @property
    def started(self) -> bool:
        return self._loop_task is not None and not self._loop_task.done()

    def next_delay(self) -> float:
        """Seconds until the next tick: interval * (1 + U(-jitter, +jitter))."""
        j = self.jitter_ratio
        return self.interval_s * (1.0 + self._rng.uniform(-j, j))

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        if self.started:
            return
        self._stopping.clear()
        self._loop_task = asyncio.create_task(self._loop(), name=f"{self.name}.loop")

    async def stop(self) -> None:
        self._stopping.set()
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None
        job = self._job_task
        if job is not None and not job.done():
            done, _ = await asyncio.wait({job}, timeout=self.stop_timeout_s)
            if not done:
                log.warning(
                    "scheduler.stop_cancel_inflight", scheduler=self.name, timeout_s=self.stop_timeout_s
                )
                job.cancel()
                try:
                    await job
                except asyncio.CancelledError:
                    pass

    async def run_now(self) -> bool:
        """Trigger a tick outside the timer (same overlap guard). Returns False if skipped."""
        return self._tick()

    # ------------------------------------------------------------------ internals

    async def _loop(self) -> None:
        while not self._stopping.is_set():
            self._tick()
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=self.next_delay())
            except TimeoutError:
                pass

    def _tick(self) -> bool:
        self.ticks += 1
        m.SCHEDULER_TICKS.inc()
        if self.running:
            self.skipped += 1
            m.SCHEDULER_SKIPPED.inc()
            log.info("scheduler.tick_skipped", scheduler=self.name, reason="previous run still in progress")
            return False
        self._job_task = asyncio.create_task(self._run_job(), name=f"{self.name}.job")
        return True

    async def _run_job(self) -> None:
        t0 = time.perf_counter()
        try:
            result = self.job()
            if inspect.isawaitable(result):
                await result
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.errors += 1
            self.last_error = f"{type(e).__name__}: {e}"
            log.exception("scheduler.job_failed", scheduler=self.name, error=self.last_error)
        else:
            self.last_error = None
            self.last_run_at = time.time()
            m.SCHEDULER_LAST_RUN.set(self.last_run_at)
        finally:
            self.runs += 1
            self.last_duration_s = time.perf_counter() - t0
            self.last_finished_at = time.time()
