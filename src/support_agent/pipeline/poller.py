"""DiscoveryPoller — one discovery pass per scheduler tick (PLAN §5.7).

`run_once()`:
  1. `portal.list_open_tickets()`; unseen ids -> `repo.upsert_discovered(summary, new_trace_id())`.
     A portal error is logged + counted (`poller_runs_total{result="error"}`), recorded on the
     `scheduler_runs` row, and the enqueue passes below still run. It never raises.
  2. Orphan pass: tickets already in QUEUED that the in-memory queue does not hold (a DLQ requeue from the
     dashboard, a rare failed `try_put` from step 3, or anything left by a restart) -> `try_put`.
  3. DISCOVERED pass (oldest first, includes leftovers from earlier polls): skip ids the queue already holds;
     stop as soon as the queue is full (the rest stay DISCOVERED = backpressure, counted in skipped_full);
     otherwise transition DISCOVERED -> QUEUED **first**, then `try_put`. Transition-first means a worker
     can never dequeue a ticket that is still DISCOVERED. If `try_put` still fails (capacity taken by a
     concurrent retry), the ticket is QUEUED-but-not-held and step 2 of the next poll enqueues it.
  Tickets past DISCOVERED are never transitioned by the poller.

Returns {"found", "new", "enqueued", "requeued", "skipped_full", "error"}.
"""

from __future__ import annotations

import time
from typing import Any

from support_agent.common.contracts import Repository, TicketPortal
from support_agent.common.models import WorkItem, new_trace_id
from support_agent.common.states import InvalidTransition, TicketState
from support_agent.observability import metrics as m
from support_agent.observability.logging import get_logger
from support_agent.observability.tracing import span

from .queue import BoundedTicketQueue

log = get_logger("poller")
S = TicketState


async def enqueue_orphaned_queued(
    repo: Repository, queue: BoundedTicketQueue, rows: list[dict[str, Any]] | None = None
) -> int:
    """`try_put` every QUEUED ticket the queue doesn't hold (no state change). Returns how many were put."""
    if rows is None:
        rows = await repo.tickets_in_states({S.QUEUED})
    n = 0
    for row in rows:
        tid = row["ticket_id"]
        if queue.contains(tid):
            continue
        if queue.full():
            break
        if queue.try_put(
            WorkItem(ticket_id=tid, trace_id=row["trace_id"], attempt=int(row.get("attempt") or 0))
        ):
            n += 1
    return n


class DiscoveryPoller:
    def __init__(self, portal: TicketPortal, repo: Repository, queue: BoundedTicketQueue) -> None:
        self.portal = portal
        self.repo = repo
        self.queue = queue
        self.last_result: dict[str, Any] | None = None
        self.last_ok_at: float | None = None  # unix ts of the last poll that listed the portal without error

    async def run_once(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        res: dict[str, Any] = {
            "found": 0,
            "new": 0,
            "enqueued": 0,
            "requeued": 0,
            "skipped_full": 0,
            "error": None,
        }
        run_id: int | None = None
        try:
            run_id = await self.repo.start_scheduler_run()
        except Exception as e:  # DB trouble must not kill the scheduler either
            log.warning("poller.run_row_failed", error=str(e))

        async with span("poller.run", run_id=run_id) as sp:
            try:
                summaries = await self.portal.list_open_tickets()
            except Exception as e:
                res["error"] = f"{type(e).__name__}: {e}"
                log.error("poller.list_failed", error=res["error"])
                summaries = []
            else:
                res["found"] = len(summaries)
                m.POLLER_TICKETS_FOUND.set(len(summaries))

            try:
                for s in summaries:
                    if await self.repo.upsert_discovered(s, new_trace_id()):
                        res["new"] += 1
                        m.TICKETS_DISCOVERED.inc()
                res["requeued"] = await enqueue_orphaned_queued(self.repo, self.queue)
                await self._enqueue_discovered(res)
            except Exception as e:
                res["error"] = res["error"] or f"{type(e).__name__}: {e}"
                log.exception("poller.enqueue_failed", error=str(e))

            if res["error"]:
                sp.status = "error"
            sp.set(**{k: v for k, v in res.items() if v is not None})

        m.POLLER_RUNS.labels(result="error" if res["error"] else "ok").inc()
        m.POLLER_DURATION.observe(time.perf_counter() - t0)
        if run_id is not None:
            try:
                await self.repo.finish_scheduler_run(
                    run_id,
                    found=res["found"],
                    enqueued=res["enqueued"] + res["requeued"],
                    skipped_full=res["skipped_full"],
                    error=res["error"],
                )
            except Exception as e:
                log.warning("poller.run_row_failed", error=str(e))
        log.info("poller.run", **res)
        self.last_result = res
        if not res["error"]:
            self.last_ok_at = time.time()
        return res

    async def _enqueue_discovered(self, res: dict[str, Any]) -> None:
        pending = await self.repo.tickets_in_states({S.DISCOVERED})
        for i, row in enumerate(pending):
            tid = row["ticket_id"]
            if self.queue.contains(tid):
                continue
            if self.queue.full():
                res["skipped_full"] += len(pending) - i
                m.QUEUE_REJECTIONS.inc(len(pending) - i)
                break
            try:
                await self.repo.transition(tid, S.QUEUED, detail={"source": "poller"})
            except InvalidTransition:  # moved on concurrently; not ours any more
                continue
            item = WorkItem(ticket_id=tid, trace_id=row["trace_id"], attempt=int(row.get("attempt") or 0))
            if self.queue.try_put(item):
                res["enqueued"] += 1
            else:  # capacity taken during the transition await; next poll's orphan pass picks it up
                res["skipped_full"] += 1
