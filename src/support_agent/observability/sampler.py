"""Periodic metrics sampler: persists `metrics.snapshot()` into `metric_samples` for the dashboard.

The runtime starts it as a task:
    asyncio.create_task(run_metric_sampler(repo, settings.metrics_sample_interval_seconds, stop_event))
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from . import metrics
from .logging import get_logger

_log = get_logger("metrics_sampler")


async def sample_once(repo: Any) -> int:
    """Take one snapshot and persist it. Returns the number of samples written."""
    samples = metrics.snapshot()
    await repo.record_metric_samples(samples)
    return len(samples)


async def run_metric_sampler(repo: Any, interval_s: float, stop: asyncio.Event) -> None:
    """Every `interval_s` seconds write a metrics snapshot until `stop` is set.

    Errors are logged and swallowed so a DB hiccup never kills the runtime. A final
    sample is written when `stop` fires so the dashboard sees the end state.
    """
    interval_s = max(0.05, float(interval_s))
    _log.info("sampler.start", interval_s=interval_s)
    while not stop.is_set():
        try:
            await sample_once(repo)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log.warning("sampler.failed", error=f"{type(e).__name__}: {e}")
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
    try:
        await sample_once(repo)
    except Exception as e:
        _log.warning("sampler.final_failed", error=f"{type(e).__name__}: {e}")
    _log.info("sampler.stop")
