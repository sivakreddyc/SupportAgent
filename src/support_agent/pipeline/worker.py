"""Workers (PLAN §5.8): get -> process -> task_done(ticket_id), with retries, dead-lettering and a timeout.

Per ticket (each step = span `worker.<step>` + `step_seconds{step}` + a state transition):
  claim        QUEUED -> PROCESSING (tickets not in QUEUED are skipped: duplicate / already handled)
  fetch        portal.get_ticket -> ticket_snapshots
  user_lookup  user_client.get_user_by_email -> user_lookups            -> ENRICHED
  llm          llm.analyze -> llm_calls (also on error)                  -> ANALYZED
  policy       policy.evaluate -> policy_decisions, policy_* metrics     -> POLICY_CHECKED
  post         auto_approve (non-empty draft):
                 create_draft(llm, posted) -> post_draft -> mark_draft_posted -> set_status("pending")
                 -> DRAFT_POSTED -> DONE
               otherwise (human_review, reject, or an empty draft):
                 create_draft(llm, pending_review) -> post_internal_note("Escalated for human review: ...")
                 -> set_status("escalated") -> NEEDS_REVIEW

Failures: any exception (or the per-ticket timeout) -> FAILED (error, attempt+1), `tickets_failed_total{step}`.
If retryable (`LLMError.retryable`; every other exception counts as retryable except InvalidTransition) and
attempt < max_retries: after `min(base * 2**attempt, max)` s, FAILED -> QUEUED, then a blocking `queue.put`.
Otherwise -> DEAD_LETTER. A worker never dies because of a ticket.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from contextlib import asynccontextmanager
from typing import Any

from support_agent.common.config import Settings
from support_agent.common.contracts import (
    LLMClient,
    LLMError,
    PolicyGate,
    Repository,
    TicketPortal,
    UserInfoClient,
)
from support_agent.common.models import LLMResult, WorkItem, utcnow
from support_agent.common.states import InvalidTransition, TicketState
from support_agent.observability import metrics as m
from support_agent.observability.logging import bind_context, get_logger, unbind_context
from support_agent.observability.tracing import span
from support_agent.storage.db import parse_iso

from .queue import BoundedTicketQueue

log = get_logger("worker")
S = TicketState

_CAN_FAIL = frozenset({S.QUEUED, S.PROCESSING, S.ENRICHED, S.ANALYZED, S.POLICY_CHECKED, S.APPROVED})


def draft_idempotency_key(ticket_id: str, body: str) -> str:
    return f"{ticket_id}:{hashlib.sha256(body.encode()).hexdigest()[:16]}"


def escalation_note(reasons: list[str]) -> str:
    return f"Escalated for human review: {', '.join(reasons) if reasons else 'policy'}"


def retry_delay(settings: Settings, attempt: int) -> float:
    return min(settings.retry_backoff_base_seconds * (2**attempt), settings.retry_backoff_max_seconds)


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, LLMError):
        return bool(exc.retryable)
    if isinstance(exc, InvalidTransition):
        return False
    return True


class _Run:
    """Mutable per-ticket context (current step for failure attribution)."""

    def __init__(self, item: WorkItem) -> None:
        self.item = item
        self.step = "claim"
        self.t0 = time.perf_counter()


class Worker:
    def __init__(
        self,
        worker_id: str,
        queue: BoundedTicketQueue,
        repo: Repository,
        portal: TicketPortal,
        user_client: UserInfoClient,
        llm: LLMClient,
        policy: PolicyGate,
        settings: Settings,
        *,
        pool: WorkerPool | None = None,
    ) -> None:
        self.worker_id = worker_id
        self.queue = queue
        self.repo = repo
        self.portal = portal
        self.user_client = user_client
        self.llm = llm
        self.policy = policy
        self.settings = settings
        self.pool = pool
        self._retry_tasks: set[asyncio.Task[None]] = pool.retry_tasks if pool is not None else set()
        self.processed = 0

    # ------------------------------------------------------------------ loop

    async def run(self) -> None:
        log.info("worker.started", worker_id=self.worker_id)
        while True:
            item = await self.queue.get()
            if self.pool is not None:
                self.pool._busy_inc()
            try:
                await self.process(item)
            except asyncio.CancelledError:
                raise
            except Exception:  # belt and braces: process() already handles everything
                log.exception("worker.unexpected_error", worker_id=self.worker_id, ticket_id=item.ticket_id)
            finally:
                if self.pool is not None:
                    self.pool._busy_dec()
                self.queue.task_done(item.ticket_id)

    async def process(self, item: WorkItem) -> str | None:
        """Process one work item. Returns the outcome (auto_approve | human_review | failed | dead_letter),
        or None if the ticket was skipped (not in QUEUED)."""
        bind_context(trace_id=item.trace_id, ticket_id=item.ticket_id, worker_id=self.worker_id)
        run = _Run(item)
        deadline = asyncio.timeout(self.settings.ticket_timeout_seconds)
        try:
            async with deadline:
                return await self._pipeline(run)
        except asyncio.CancelledError:
            raise
        except TimeoutError as e:
            err: BaseException = e
            if deadline.expired():
                err = TimeoutError(
                    f"ticket timed out after {self.settings.ticket_timeout_seconds}s in step {run.step}"
                )
            return await self._handle_failure(run, err)
        except Exception as e:
            return await self._handle_failure(run, e)
        finally:
            self.processed += 1
            unbind_context("trace_id", "ticket_id", "worker_id")

    # ------------------------------------------------------------------ pipeline

    @asynccontextmanager
    async def _step(self, run: _Run, name: str, **attrs: Any):
        run.step = name
        t0 = time.perf_counter()
        try:
            async with span(
                f"worker.{name}",
                trace_id=run.item.trace_id,
                ticket_id=run.item.ticket_id,
                worker_id=self.worker_id,
                attempt=run.item.attempt,
                **attrs,
            ) as sp:
                yield sp
        finally:
            m.STEP_SECONDS.labels(step=name).observe(time.perf_counter() - t0)

    async def _transition(self, tid: str, to: TicketState, **kw: Any) -> None:
        await self.repo.transition(tid, to, worker_id=self.worker_id, **kw)

    async def _pipeline(self, run: _Run) -> str | None:
        item = run.item
        tid = item.ticket_id

        row = await self.repo.get_ticket_row(tid)
        if row is None or row["state"] != S.QUEUED.value:
            log.warning("worker.skip_not_queued", state=row["state"] if row else None)
            return None
        await self._transition(tid, S.PROCESSING, detail={"attempt": item.attempt})

        # BUG-2: an earlier attempt already reached the auto-approve post (the llm draft row is created as
        # 'posted' right before portal.post_draft). Never ask the LLM again: re-post that same draft with its
        # stored key (the portal dedups on it) so a non-deterministic LLM can't cause a second reply.
        prior = await self.repo.latest_draft(tid)
        if prior is not None and prior["source"] == "llm" and prior["status"] == "posted":
            return await self._resume_post(run, row, prior)

        async with self._step(run, "fetch"):
            ticket = await self.portal.get_ticket(tid)
            await self.repo.save_snapshot(ticket)

        async with self._step(run, "user_lookup") as sp:
            t0 = time.perf_counter()
            user = await self.user_client.get_user_by_email(ticket.customer_email)
            latency_ms = (time.perf_counter() - t0) * 1000
            await self.repo.record_user_lookup(tid, user, latency_ms)
            sp.set(found=user is not None)
            await self._transition(tid, S.ENRICHED, detail={"user_found": user is not None})

        async with self._step(run, "llm") as sp:
            try:
                result: LLMResult = await self.llm.analyze(ticket, user)
            except Exception as e:
                await self.repo.record_llm_call(tid, None, error=f"{type(e).__name__}: {e}")
                raise
            await self.repo.record_llm_call(tid, result)
            sp.set(
                model=result.model, stop_reason=result.stop_reason, has_analysis=result.analysis is not None
            )
            await self._transition(
                tid,
                S.ANALYZED,
                detail={
                    "model": result.model,
                    "stop_reason": result.stop_reason,
                    "has_analysis": result.analysis is not None,
                },
            )

        async with self._step(run, "policy") as sp:
            decision = self.policy.evaluate(ticket, result, user)
            await self.repo.record_policy_decision(tid, decision)
            m.POLICY_DECISIONS.labels(outcome=decision.outcome).inc()
            for rule in decision.reasons:
                m.POLICY_RULE_HITS.labels(rule=rule).inc()
            sp.set(outcome=decision.outcome, reasons=decision.reasons)
            await self._transition(
                tid,
                S.POLICY_CHECKED,
                detail={
                    "outcome": decision.outcome,
                    "reasons": decision.reasons,
                    "priority": decision.priority,
                },
            )

        body = result.analysis.draft_reply if result.analysis is not None else ""
        key = draft_idempotency_key(tid, body)
        async with self._step(run, "post") as sp:
            if decision.outcome == "auto_approve" and body.strip():
                draft_id = await self.repo.create_draft(
                    tid, body, source="llm", status="posted", idempotency_key=key
                )
                await self.portal.post_draft(tid, body, idempotency_key=key)
                await self.repo.mark_draft_posted(draft_id)
                m.DRAFTS_POSTED.labels(source="llm").inc()
                await self.portal.set_status(tid, "pending")
                sp.set(action="posted", draft_id=draft_id)
                await self._transition(tid, S.DRAFT_POSTED, detail={"draft_id": draft_id, "source": "llm"})
                await self._transition(tid, S.DONE)
                outcome = "auto_approve"
            else:
                reasons = list(decision.reasons)
                if decision.outcome == "auto_approve":  # auto_approve with an empty draft: never post ""
                    reasons.append("EMPTY_DRAFT")
                elif not reasons:
                    reasons.append(f"POLICY_{decision.outcome.upper()}")
                draft_id = await self.repo.create_draft(
                    tid, body, source="llm", status="pending_review", idempotency_key=key
                )
                note = escalation_note(reasons)
                if note not in ticket.notes:  # a retried ticket doesn't re-post the same note
                    await self.portal.post_internal_note(tid, note)
                await self.portal.set_status(tid, "escalated")
                sp.set(action="escalated", draft_id=draft_id, reasons=reasons)
                await self._transition(
                    tid,
                    S.NEEDS_REVIEW,
                    detail={"draft_id": draft_id, "reasons": reasons, "policy_outcome": decision.outcome},
                )
                outcome = "human_review"

        self._finish(run, outcome, row.get("discovered_at"))
        log.info("worker.ticket_done", outcome=outcome)
        return outcome

    async def _resume_post(self, run: _Run, row: dict[str, Any], prior: dict[str, Any]) -> str:
        tid = run.item.ticket_id
        detail = {"resumed_from_draft": prior["id"]}
        for st in (S.ENRICHED, S.ANALYZED, S.POLICY_CHECKED):  # walk the state machine, no re-analysis
            await self._transition(tid, st, detail=detail)
        async with self._step(run, "post") as sp:
            await self.portal.post_draft(tid, prior["body"], idempotency_key=prior["idempotency_key"])
            if prior["posted_at"] is None:
                await self.repo.mark_draft_posted(prior["id"])
                m.DRAFTS_POSTED.labels(source="llm").inc()
            await self.portal.set_status(tid, "pending")
            sp.set(action="posted", draft_id=prior["id"], resumed=True)
            await self._transition(
                tid, S.DRAFT_POSTED, detail={"draft_id": prior["id"], "source": "llm", "resumed": True}
            )
            await self._transition(tid, S.DONE)
        self._finish(run, "auto_approve", row.get("discovered_at"))
        log.info("worker.ticket_done", outcome="auto_approve", resumed=True)
        return "auto_approve"

    def _finish(self, run: _Run, outcome: str, discovered_at: str | None) -> None:
        m.TICKETS_PROCESSED.labels(outcome=outcome).inc()
        m.TICKET_PROCESSING.observe(time.perf_counter() - run.t0)
        disc = parse_iso(discovered_at)
        if disc is not None:
            m.TICKET_E2E.observe(max((utcnow() - disc).total_seconds(), 0.0))

    # ------------------------------------------------------------------ failures / retries

    async def _handle_failure(self, run: _Run, exc: BaseException) -> str:
        tid = run.item.ticket_id
        msg = f"{type(exc).__name__}: {exc}"
        m.TICKETS_FAILED.labels(step=run.step).inc()
        log.warning("worker.ticket_failed", step=run.step, error=msg)
        try:
            row = await self.repo.get_ticket_row(tid)
            state = S(row["state"]) if row else None
            if state not in _CAN_FAIL:
                if state is S.DRAFT_POSTED:  # draft already posted; only DONE was missing
                    await self._transition(tid, S.DONE, detail={"recovered_from_error": msg})
                    return "auto_approve"
                log.error("worker.cannot_fail", state=state, error=msg)
                return "failed"
            await self._transition(tid, S.FAILED, error=msg, bump_attempt=True, detail={"step": run.step})
            attempt = int(row["attempt"]) + 1
            if is_retryable(exc) and attempt < self.settings.max_retries:
                delay = retry_delay(self.settings, attempt)
                log.info("worker.retry_scheduled", attempt=attempt, delay_s=delay)
                task = asyncio.create_task(
                    self._requeue_later(tid, run.item.trace_id, attempt, delay), name=f"retry:{tid}"
                )
                self._retry_tasks.add(task)
                task.add_done_callback(self._retry_tasks.discard)
                return "failed"
            reason = "max_retries" if is_retryable(exc) else "non_retryable"
            await self._transition(
                tid, S.DEAD_LETTER, detail={"reason": reason, "attempt": attempt, "error": msg}
            )
            m.TICKETS_DEAD_LETTER.inc()
            self._finish(run, "dead_letter", row.get("discovered_at"))
            log.error("worker.dead_letter", reason=reason, attempt=attempt, error=msg)
            return "dead_letter"
        except Exception:
            log.exception("worker.failure_handling_failed", error=msg)
            return "failed"

    async def _requeue_later(self, tid: str, trace_id: str, attempt: int, delay: float) -> None:
        await asyncio.sleep(delay)
        try:
            await self.repo.transition(
                tid, S.QUEUED, worker_id=self.worker_id, detail={"retry_attempt": attempt}
            )
        except InvalidTransition as e:
            log.warning("worker.retry_skipped", ticket_id=tid, error=str(e))
            return
        except Exception:
            log.exception("worker.retry_requeue_failed", ticket_id=tid)
            return
        # Blocking put (waits for space). False = already held (an orphan pass got there first): fine.
        await self.queue.put(WorkItem(ticket_id=tid, trace_id=trace_id, attempt=attempt))


class WorkerPool:
    def __init__(
        self,
        n: int,
        queue: BoundedTicketQueue,
        repo: Repository,
        portal: TicketPortal,
        user_client: UserInfoClient,
        llm: LLMClient,
        policy: PolicyGate,
        settings: Settings,
    ) -> None:
        self.size = max(1, int(n))
        self.queue = queue
        self.retry_tasks: set[asyncio.Task[None]] = set()
        self.workers = [
            Worker(f"w{i}", queue, repo, portal, user_client, llm, policy, settings, pool=self)
            for i in range(self.size)
        ]
        self._tasks: list[asyncio.Task[None]] = []
        self.busy = 0

    def _busy_inc(self) -> None:
        self.busy += 1
        m.WORKERS_BUSY.set(self.busy)

    def _busy_dec(self) -> None:
        self.busy -= 1
        m.WORKERS_BUSY.set(self.busy)

    @property
    def running(self) -> bool:
        return any(not t.done() for t in self._tasks)

    async def start(self) -> None:
        if self._tasks:
            return
        m.WORKERS_CONFIGURED.set(self.size)
        m.WORKERS_BUSY.set(0)
        self._tasks = [asyncio.create_task(w.run(), name=f"worker:{w.worker_id}") for w in self.workers]

    async def stop(self) -> None:
        """Cancel workers and pending retry timers. Interrupted tickets stay in SQLite for startup recovery."""
        for t in [*self._tasks, *self.retry_tasks]:
            t.cancel()
        await asyncio.gather(*self._tasks, *list(self.retry_tasks), return_exceptions=True)
        self._tasks = []
        self.retry_tasks.clear()
        self.busy = 0
        m.WORKERS_BUSY.set(0)
