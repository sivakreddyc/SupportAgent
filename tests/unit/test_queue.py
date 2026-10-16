"""BoundedTicketQueue (PLAN §5.6): default 1000, bounds, try_put/put, backpressure metrics.

Skips at module level until B3 lands. `try_put` / `put` / `get` may be sync or async; `_call` handles both.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from support_agent.common.config import Settings
from support_agent.common.models import WorkItem
from support_agent.observability import metrics as m
from tests.conftest import require

BoundedTicketQueue = require("support_agent.pipeline.queue", "BoundedTicketQueue")


async def _call(fn, *args, **kwargs):
    r = fn(*args, **kwargs)
    if inspect.isawaitable(r):
        r = await r
    return r


def _item(i: int) -> WorkItem:
    return WorkItem(ticket_id=f"T-{i}", trace_id=f"tr{i}")


def _metric(name: str) -> float:
    return m.REGISTRY.get_sample_value(name) or 0.0


@pytest.mark.parametrize("size", [None, 0, -1, -100])
def test_default_maxsize_is_1000(size):
    assert BoundedTicketQueue(size).maxsize == 1000


@pytest.mark.parametrize("size", [1, 100, 5000])
def test_custom_maxsize(size):
    assert BoundedTicketQueue(size).maxsize == size


@pytest.mark.parametrize("raw", [None, "", "0", "-4"])
def test_maxsize_from_settings(raw):
    assert BoundedTicketQueue(Settings(queue_max_size=raw).queue_max_size).maxsize == 1000


async def test_try_put_false_when_full():
    q = BoundedTicketQueue(3)
    for i in range(3):
        assert await _call(q.try_put, _item(i)) is True
    assert q.full()
    assert q.qsize() == 3
    assert await _call(q.try_put, _item(99)) is False
    assert q.qsize() == 3


async def test_fifo_and_task_done():
    q = BoundedTicketQueue(5)
    for i in range(3):
        await _call(q.try_put, _item(i))
    got = [(await _call(q.get)).ticket_id for _ in range(3)]
    assert got == ["T-0", "T-1", "T-2"]
    for _ in range(3):
        q.task_done()
    assert q.qsize() == 0 and not q.full()


async def test_get_frees_capacity():
    q = BoundedTicketQueue(1)
    assert await _call(q.try_put, _item(1))
    assert not await _call(q.try_put, _item(2))
    await _call(q.get)
    q.task_done()
    assert await _call(q.try_put, _item(2))


async def test_put_waits_for_space():
    q = BoundedTicketQueue(1)
    await _call(q.try_put, _item(1))

    async def consume_later():
        await asyncio.sleep(0.05)
        await _call(q.get)
        q.task_done()

    consumer = asyncio.create_task(consume_later())
    await _call(q.put, _item(2), timeout=2)
    await consumer
    assert q.qsize() == 1
    assert (await _call(q.get)).ticket_id == "T-2"


async def test_put_times_out_when_full():
    q = BoundedTicketQueue(1)
    await _call(q.try_put, _item(1))
    try:
        r = await _call(q.put, _item(2), timeout=0.05)
    except (TimeoutError, asyncio.QueueFull):
        r = False
    assert r in (False, None)
    assert q.qsize() == 1


async def test_concurrent_producers_never_exceed_capacity():
    q = BoundedTicketQueue(100)
    peak = 0

    async def producer(start: int) -> tuple[int, int]:
        nonlocal peak
        ok = rej = 0
        for i in range(start, start + 100):
            if await _call(q.try_put, _item(i)):
                ok += 1
            else:
                rej += 1
            peak = max(peak, q.qsize())
            await asyncio.sleep(0)
        return ok, rej

    results = await asyncio.gather(*(producer(k * 1000) for k in range(10)))
    accepted = sum(r[0] for r in results)
    rejected = sum(r[1] for r in results)
    assert accepted == 100 and rejected == 900
    assert peak <= 100 and q.qsize() == 100


async def test_rejections_and_enqueued_counted():
    q = BoundedTicketQueue(2)
    enq0, rej0 = _metric("queue_enqueued_total"), _metric("queue_rejections_total")
    for i in range(5):
        await _call(q.try_put, _item(i))
    assert _metric("queue_enqueued_total") - enq0 == 2
    assert _metric("queue_rejections_total") - rej0 == 3


async def test_capacity_and_depth_gauges():
    q = BoundedTicketQueue(7)
    assert _metric("queue_capacity") == 7
    await _call(q.try_put, _item(1))
    await _call(q.try_put, _item(2))
    assert _metric("queue_depth") == 2
    await _call(q.get)
    q.task_done()
    assert _metric("queue_depth") == 1


async def test_queue_wait_observed_on_get():
    q = BoundedTicketQueue(3)
    before = _metric("queue_wait_seconds_count")
    await _call(q.try_put, _item(1))
    await asyncio.sleep(0.02)
    await _call(q.get)
    q.task_done()
    assert _metric("queue_wait_seconds_count") - before == 1
