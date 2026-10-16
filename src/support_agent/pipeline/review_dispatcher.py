"""ReviewDispatcher — turns dashboard decisions (SQLite) into portal actions (PLAN §5.12 item 3).

Every `interval_s` seconds, `run_once()`:
  1. Posts approved drafts: for each `repo.approved_pending_post()` row -> `portal.post_draft` (stored
     idempotency key) -> `mark_draft_posted` -> `portal.set_status("pending")` -> APPROVED -> DRAFT_POSTED -> DONE.
     `drafts_posted_total{source}` and `reviews_total{action}` (approve for llm drafts, edit for human_edit).
     On error: APPROVED -> FAILED (attempt+1, detail.step="review_post"), `tickets_failed_total{step="review_post"}`.
  2. Retries failed posts (`repo.failed_review_posts()`): attempt >= max_retries -> DEAD_LETTER, otherwise after
     `min(base * 2**attempt, max)` seconds since the failure -> FAILED -> APPROVED (posted on the next pass).
  3. Enqueues DLQ requeues: QUEUED tickets whose last transition was DEAD_LETTER -> QUEUED and that the queue
     doesn't hold -> `try_put` (the poller's orphan pass is the slower safety net for the same thing).
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

from support_agent.common.config import Settings
from support_agent.common.contracts import Repository, TicketPortal
from support_agent.common.models import utcnow
from support_agent.common.states import InvalidTransition, TicketState
from support_agent.observability import metrics as m
from support_agent.observability.logging import bind_context, get_logger, unbind_context
from support_agent.observability.tracing import span
from support_agent.storage.db import parse_iso

from .poller import enqueue_orphaned_queued
from .queue import BoundedTicketQueue

log = get_logger("review_dispatcher")
S = TicketState
WORKER_ID = "review_dispatcher"


class ReviewDispatcher:
    def __init__(
        self,
        repo: Repository,
        portal: TicketPortal,
        interval_s: float,
        *,
        queue: BoundedTicketQueue | None = None,
        max_retries: int = 3,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 30.0,
    ) -> None:
        self.repo = repo
        self.portal = portal
        self.interval_s = max(float(interval_s), 0.01)
        self.queue = queue
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()
        self.last_result: dict[str, int] | None = None
        self.last_error: str | None = None

    @classmethod
    def from_settings(
        cls,
        repo: Repository,
        portal: TicketPortal,
        settings: Settings,
        queue: BoundedTicketQueue | None = None,
    ) -> ReviewDispatcher:
        return cls(
            repo,
            portal,
            settings.review_dispatch_interval_seconds,
            queue=queue,
            max_retries=settings.max_retries,
            backoff_base_s=settings.retry_backoff_base_seconds,
            backoff_max_s=settings.retry_backoff_max_seconds,
        )

    # ------------------------------------------------------------------ lifecycle

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        if self.running:
            return
        self._stopping.clear()
        self._task = asyncio.create_task(self._loop(), name="review_dispatcher")

    async def stop(self) -> None:
        self._stopping.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        while not self._stopping.is_set():
            try:
                await self.run_once()
                self.last_error = None
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                log.exception("review_dispatcher.pass_failed", error=self.last_error)
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=self.interval_s)
            except TimeoutError:
                pass

    # ------------------------------------------------------------------ one pass

    async def run_once(self) -> dict[str, int]:
        res = {"posted": 0, "failed": 0, "retried": 0, "dead_lettered": 0, "requeued": 0}
        for row in await self.repo.approved_pending_post():
            if await self._post(row):
                res["posted"] += 1
            else:
                res["failed"] += 1
        await self._retry_failed(res)
        if self.queue is not None:
            rows = await self.repo.requeued_from_dead_letter()  # type: ignore[attr-defined]
            if rows:
                res["requeued"] = await enqueue_orphaned_queued(self.repo, self.queue, rows)
        if any(res.values()):
            log.info("review_dispatcher.pass", **res)
        self.last_result = res
        return res

    async def _post(self, row: dict[str, Any]) -> bool:
        tid = row["ticket_id"]
        bind_context(trace_id=row["trace_id"], ticket_id=tid, worker_id=WORKER_ID)
        try:
            async with span(
                "review.post",
                trace_id=row["trace_id"],
                ticket_id=tid,
                draft_id=row["draft_id"],
                source=row["source"],
            ):
                await self.portal.post_draft(tid, row["body"], idempotency_key=row["idempotency_key"])
                await self.repo.mark_draft_posted(row["draft_id"])
                await self.portal.set_status(tid, "pending")
                await self.repo.transition(
                    tid,
                    S.DRAFT_POSTED,
                    worker_id=WORKER_ID,
                    detail={"draft_id": row["draft_id"], "source": row["source"]},
                )
                await self.repo.transition(tid, S.DONE, worker_id=WORKER_ID)
            m.DRAFTS_POSTED.labels(source=row["source"]).inc()
            m.REVIEWS.labels(action="edit" if row["source"] == "human_edit" else "approve").inc()
            log.info("review_dispatcher.posted", draft_id=row["draft_id"], source=row["source"])
            return True
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            m.TICKETS_FAILED.labels(step="review_post").inc()
            log.warning("review_dispatcher.post_failed", error=msg)
            try:
                await self.repo.transition(
                    tid,
                    S.FAILED,
                    worker_id=WORKER_ID,
                    error=msg,
                    bump_attempt=True,
                    detail={"step": "review_post"},
                )
            except InvalidTransition as e2:  # e.g. already DRAFT_POSTED: finish it
                if e2.from_state is S.DRAFT_POSTED:
                    await self.repo.transition(tid, S.DONE, worker_id=WORKER_ID)
                    return True
                log.error("review_dispatcher.cannot_fail", error=str(e2))
            return False
        finally:
            unbind_context("trace_id", "ticket_id", "worker_id")

    def _delay(self, attempt: int) -> float:
        return min(self.backoff_base_s * (2**attempt), self.backoff_max_s)

    async def _retry_failed(self, res: dict[str, int]) -> None:
        rows = await self.repo.failed_review_posts()  # type: ignore[attr-defined]
        now = utcnow()
        for row in rows:
            tid, attempt = row["ticket_id"], int(row["attempt"] or 0)
            try:
                if attempt >= self.max_retries:
                    await self.repo.transition(
                        tid,
                        S.DEAD_LETTER,
                        worker_id=WORKER_ID,
                        detail={"reason": "max_retries", "step": "review_post", "attempt": attempt},
                    )
                    m.TICKETS_DEAD_LETTER.inc()
                    m.TICKETS_PROCESSED.labels(outcome="dead_letter").inc()
                    res["dead_lettered"] += 1
                    continue
                failed_at = parse_iso(row["updated_at"])
                if failed_at is not None and now - failed_at < timedelta(seconds=self._delay(attempt)):
                    continue
                await self.repo.transition(
                    tid,
                    S.APPROVED,
                    worker_id=WORKER_ID,
                    detail={"retry_attempt": attempt, "step": "review_post"},
                )
                res["retried"] += 1
            except InvalidTransition as e:
                log.warning("review_dispatcher.retry_skipped", ticket_id=tid, error=str(e))
