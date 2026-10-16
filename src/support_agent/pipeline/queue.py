"""BoundedTicketQueue — the in-memory work buffer between the poller and the workers (PLAN §5.6).

SQLite is the durable record; this queue only buffers work. Besides the asyncio.Queue it tracks the set
of ticket ids it currently *holds*: an id is held from a successful put until `task_done(ticket_id)`
(i.e. while it is queued or being processed). Rules:

* Putting an id that is already held is a **duplicate**: `try_put` / `put` return False immediately and it
  is NOT counted in `queue_rejections_total` (that counter means "rejected because full").
* `try_put` never blocks; when full it returns False and increments `queue_rejections_total`.
* `put(item, timeout)` waits for space; on timeout it returns False and also counts a rejection.
* `task_done()` without an id still works (asyncio semantics) but leaves the id held; workers pass the id.

Metrics: `queue_capacity` (set on construction), `queue_depth`, `queue_enqueued_total`,
`queue_rejections_total`, `queue_wait_seconds` (item.enqueued_at -> get).
"""

from __future__ import annotations

import asyncio
from datetime import UTC

from support_agent.common.config import DEFAULT_QUEUE_MAX_SIZE
from support_agent.common.models import WorkItem, utcnow
from support_agent.observability import metrics as m
from support_agent.observability.logging import get_logger

log = get_logger("queue")


class BoundedTicketQueue:
    def __init__(self, maxsize: int | None = None) -> None:
        size = DEFAULT_QUEUE_MAX_SIZE if maxsize is None or maxsize <= 0 else int(maxsize)
        self._maxsize = size
        self._q: asyncio.Queue[WorkItem] = asyncio.Queue(maxsize=size)
        self._held: set[str] = set()
        m.QUEUE_CAPACITY.set(size)
        m.QUEUE_DEPTH.set(0)

    # ------------------------------------------------------------------ introspection

    @property
    def maxsize(self) -> int:
        return self._maxsize

    def qsize(self) -> int:
        return self._q.qsize()

    def full(self) -> bool:
        return self._q.full()

    def empty(self) -> bool:
        return self._q.empty()

    def contains(self, ticket_id: str) -> bool:
        """True while the ticket is queued or being processed (put .. task_done(ticket_id))."""
        return ticket_id in self._held

    def held_ids(self) -> frozenset[str]:
        return frozenset(self._held)

    def _update_depth(self) -> None:
        m.QUEUE_DEPTH.set(self._q.qsize())

    # ------------------------------------------------------------------ producers

    def try_put(self, item: WorkItem) -> bool:
        """Non-blocking enqueue. False if the id is already held (duplicate) or the queue is full."""
        if item.ticket_id in self._held:
            log.debug("queue.duplicate", ticket_id=item.ticket_id)
            return False
        try:
            self._q.put_nowait(item)
        except asyncio.QueueFull:
            m.QUEUE_REJECTIONS.inc()
            return False
        self._held.add(item.ticket_id)
        m.QUEUE_ENQUEUED.inc()
        self._update_depth()
        return True

    async def put(self, item: WorkItem, timeout: float | None = None) -> bool:  # noqa: ASYNC109 (spec API)
        """Enqueue, waiting for space up to `timeout` seconds (None = forever).

        Returns True on success; False for a duplicate or on timeout (timeout counts as a rejection).
        """
        if item.ticket_id in self._held:
            log.debug("queue.duplicate", ticket_id=item.ticket_id)
            return False
        self._held.add(item.ticket_id)  # reserve, so a concurrent put of the same id is a duplicate
        try:
            if timeout is None:
                await self._q.put(item)
            else:
                await asyncio.wait_for(self._q.put(item), timeout)
        except TimeoutError:
            self._held.discard(item.ticket_id)
            m.QUEUE_REJECTIONS.inc()
            return False
        except BaseException:
            self._held.discard(item.ticket_id)
            raise
        m.QUEUE_ENQUEUED.inc()
        self._update_depth()
        return True

    # ------------------------------------------------------------------ consumers

    async def get(self) -> WorkItem:
        item = await self._q.get()
        self._update_depth()
        enq = item.enqueued_at
        if enq.tzinfo is None:
            enq = enq.replace(tzinfo=UTC)
        wait = (utcnow() - enq).total_seconds()
        m.QUEUE_WAIT.observe(max(wait, 0.0))
        return item

    def task_done(self, ticket_id: str | None = None) -> None:
        """Mark one item processed; pass `ticket_id` to release it from the held set."""
        if ticket_id is not None:
            self._held.discard(ticket_id)
        self._q.task_done()

    async def join(self) -> None:
        """Wait until every enqueued item has been marked task_done (used for graceful drain)."""
        await self._q.join()
