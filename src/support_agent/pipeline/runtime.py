"""Runtime wiring (PLAN §5.7–5.8, §5.11): repo, portal, clients, queue, scheduler+poller, workers,
review dispatcher, metric sampler and the ops API in one asyncio process.

    await run(settings)                      # CLI entrypoint (`support-agent run`), handles SIGINT/SIGTERM
    rt = Runtime(settings, portal=fake, ...)  # tests: inject any component, then
    await rt.start(); ...; await rt.stop()

Startup recovery (`Runtime.recover()`, run by `start()` after the workers start):
  * QUEUED                                    -> enqueue (no state change)
  * PROCESSING / ENRICHED / ANALYZED / POLICY_CHECKED (interrupted mid-pipeline)
                                              -> FAILED (attempt+1, "interrupted by restart") -> QUEUED -> enqueue,
                                                 or -> DEAD_LETTER if attempt >= max_retries (poison-ticket guard)
  * FAILED (pipeline failure)                 -> QUEUED -> enqueue, or DEAD_LETTER if attempt >= max_retries
  * FAILED after a review post (APPROVED->FAILED), APPROVED -> left to the review dispatcher
  * DRAFT_POSTED -> DONE (the draft was posted; only the last transition was lost)
  * DISCOVERED -> left to the poller
Enqueue uses transition-first then `try_put`; anything that doesn't fit stays QUEUED and the poller's orphan
pass enqueues it once there is room.

Shutdown (`stop()`): scheduler -> review dispatcher -> drain queue (up to shutdown_grace_seconds) -> cancel
workers and retry timers -> sampler -> ops API -> close portal, user client, LLM client, repo.
"""

from __future__ import annotations

import asyncio
import importlib
import signal
import time
from typing import Any

from support_agent.common.config import Settings
from support_agent.common.contracts import LLMClient, PolicyGate, Repository, TicketPortal, UserInfoClient
from support_agent.common.models import WorkItem
from support_agent.common.states import RECOVERABLE_STATES, InvalidTransition, TicketState
from support_agent.observability import metrics as m
from support_agent.observability.logging import configure_logging, get_logger
from support_agent.observability.tracing import set_span_sink

from .poller import DiscoveryPoller
from .queue import BoundedTicketQueue
from .review_dispatcher import ReviewDispatcher
from .scheduler import IntervalScheduler
from .worker import WorkerPool

log = get_logger("runtime")
S = TicketState
_MID_PIPELINE = frozenset({S.PROCESSING, S.ENRICHED, S.ANALYZED, S.POLICY_CHECKED})
RECOVERY_WORKER_ID = "recovery"


# ---------------------------------------------------------------------- component factories (lazy imports)


def _load(module: str, attr: str) -> Any:
    return getattr(importlib.import_module(module), attr)


def build_repository(settings: Settings) -> Repository:
    from support_agent.storage.repository import SqliteRepository

    return SqliteRepository(settings.db_path)


def build_portal(settings: Settings) -> TicketPortal:
    return _load("support_agent.browser.ticket_portal", "PlaywrightTicketPortal")(settings)


def build_user_client(settings: Settings) -> UserInfoClient:
    return _load("support_agent.clients.user_info", "HttpUserInfoClient")(settings)


def build_llm(settings: Settings) -> LLMClient:
    if settings.llm_provider == "mock":
        return _load("support_agent.llm.mock_client", "MockLLMClient")(settings)
    return _load("support_agent.llm.anthropic_client", "AnthropicLLMClient")(settings)


def build_policy(settings: Settings) -> PolicyGate:
    return _load("support_agent.policy.gate", "RuleBasedPolicyGate").from_file(settings.policy_file)


async def _maybe_close(obj: Any) -> None:
    for name in ("close", "aclose"):
        fn = getattr(obj, name, None)
        if fn is not None:
            res = fn()
            if asyncio.iscoroutine(res):
                await res
            return


# ---------------------------------------------------------------------- runtime


class Runtime:
    def __init__(
        self,
        settings: Settings,
        *,
        repo: Repository | None = None,
        portal: TicketPortal | None = None,
        user_client: UserInfoClient | None = None,
        llm: LLMClient | None = None,
        policy: PolicyGate | None = None,
        queue: BoundedTicketQueue | None = None,
        serve_ops_api: bool = True,
        run_sampler: bool = True,
    ) -> None:
        self.settings = settings
        self.repo: Repository | None = repo
        self.portal: TicketPortal | None = portal
        self.user_client: UserInfoClient | None = user_client
        self.llm: LLMClient | None = llm
        self.policy: PolicyGate | None = policy
        self.queue: BoundedTicketQueue | None = queue
        self.serve_ops_api = serve_ops_api
        self.run_sampler = run_sampler

        self.pool: WorkerPool | None = None
        self.poller: DiscoveryPoller | None = None
        self.scheduler: IntervalScheduler | None = None
        self.dispatcher: ReviewDispatcher | None = None
        self.ops_server: Any = None
        self.portal_started = False
        self.portal_error: str | None = None
        self.started = False
        self.started_at: float | None = None
        self.recovery: dict[str, int] = {}
        self._sampler_stop = asyncio.Event()
        self._sampler_task: asyncio.Task[None] | None = None
        self._portal_lock = asyncio.Lock()

    # ------------------------------------------------------------------ properties used by the ops API

    @property
    def ops_port(self) -> int | None:
        return getattr(self.ops_server, "port", None)

    @property
    def last_poll_at(self) -> float | None:
        """Unix ts of the last poll that listed the portal without error (drives /ready)."""
        return self.poller.last_ok_at if self.poller else None

    # ------------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        if self.started:
            return
        s = self.settings
        try:
            if self.repo is None:
                self.repo = build_repository(s)
            await self.repo.init()
            set_span_sink(self.repo.record_span)

            if self.portal is None:
                self.portal = build_portal(s)
            if self.user_client is None:
                self.user_client = build_user_client(s)
            if self.llm is None:
                self.llm = build_llm(s)
            if self.policy is None:
                self.policy = build_policy(s)
            if self.queue is None:
                self.queue = BoundedTicketQueue(s.queue_max_size)

            await self._ensure_portal()

            self.pool = WorkerPool(
                s.worker_count, self.queue, self.repo, self.portal, self.user_client, self.llm, self.policy, s
            )
            await self.pool.start()
            self.recovery = await self.recover()

            self.poller = DiscoveryPoller(self.portal, self.repo, self.queue)
            self.scheduler = IntervalScheduler(
                s.poll_interval_seconds, s.poll_jitter_ratio, self._poll_job, name="poller"
            )
            await self.scheduler.start()

            self.dispatcher = ReviewDispatcher.from_settings(self.repo, self.portal, s, self.queue)
            await self.dispatcher.start()

            if self.run_sampler:
                self._start_sampler()
            if self.serve_ops_api:
                from support_agent.ops_api.app import OpsServer

                self.ops_server = OpsServer(self, s.ops_api_host, s.ops_api_port)
                await self.ops_server.start()
        except BaseException:
            log.exception("runtime.start_failed")
            await self.stop()
            raise
        self.started = True
        self.started_at = time.time()
        log.info(
            "runtime.started",
            workers=s.worker_count,
            queue_max_size=self.queue.maxsize,
            poll_interval_s=s.poll_interval_seconds,
            ops_port=self.ops_port,
            recovery=self.recovery,
        )

    async def stop(self) -> None:
        s = self.settings
        log.info("runtime.stopping")

        async def _safe(name: str, coro: Any) -> None:
            try:
                await coro
            except Exception:
                log.exception("runtime.stop_step_failed", step=name)

        if self.scheduler is not None:
            await _safe("scheduler", self.scheduler.stop())
        if self.dispatcher is not None:
            await _safe("dispatcher", self.dispatcher.stop())
        if self.queue is not None and self.pool is not None and self.pool.running:
            try:
                await asyncio.wait_for(self.queue.join(), timeout=s.shutdown_grace_seconds)
                log.info("runtime.drained")
            except TimeoutError:
                log.warning(
                    "runtime.drain_timeout", remaining=self.queue.qsize(), grace_s=s.shutdown_grace_seconds
                )
        if self.pool is not None:
            await _safe("workers", self.pool.stop())
        if self._sampler_task is not None:
            self._sampler_stop.set()
            try:
                await asyncio.wait_for(self._sampler_task, timeout=5)
            except (TimeoutError, Exception):
                self._sampler_task.cancel()
            self._sampler_task = None
        if self.ops_server is not None:
            await _safe("ops_api", self.ops_server.stop())
        if self.portal is not None:
            await _safe("portal", self.portal.close())
            self.portal_started = False
        if self.user_client is not None:
            await _safe("user_client", _maybe_close(self.user_client))
        if self.llm is not None:
            await _safe("llm", _maybe_close(self.llm))
        if self.repo is not None:
            set_span_sink(None)
            await _safe("repo", self.repo.close())
        self.started = False
        log.info("runtime.stopped")

    # ------------------------------------------------------------------ pieces

    async def _ensure_portal(self) -> bool:
        """Start the portal if needed. Failure is logged, not raised: the poll job keeps retrying."""
        if self.portal_started:
            return True
        async with self._portal_lock:
            if self.portal_started:
                return True
            try:
                await self.portal.start()
            except Exception as e:
                self.portal_error = f"{type(e).__name__}: {e}"
                log.error("runtime.portal_start_failed", error=self.portal_error)
                return False
            self.portal_started = True
            self.portal_error = None
            return True

    async def _poll_job(self) -> None:
        if not await self._ensure_portal():
            m.POLLER_RUNS.labels(result="error").inc()
            raise RuntimeError(f"portal not started: {self.portal_error}")
        await self.poller.run_once()

    def _start_sampler(self) -> None:
        try:
            from support_agent.observability.sampler import run_metric_sampler
        except ImportError as e:
            log.warning("runtime.sampler_unavailable", error=str(e))
            return
        self._sampler_stop.clear()
        self._sampler_task = asyncio.create_task(
            run_metric_sampler(self.repo, self.settings.metrics_sample_interval_seconds, self._sampler_stop),
            name="metric_sampler",
        )

    async def recover(self) -> dict[str, int]:
        """Startup recovery (see module docstring). Returns counts by action."""
        repo, q, s = self.repo, self.queue, self.settings
        out = {"enqueued": 0, "requeued": 0, "dead_lettered": 0, "finished": 0, "left_queued": 0}
        review_failures = {r["ticket_id"] for r in await repo.failed_review_posts()}  # type: ignore[attr-defined]
        rows = await repo.tickets_in_states(set(RECOVERABLE_STATES) | {S.DRAFT_POSTED})

        for row in rows:
            tid, state = row["ticket_id"], S(row["state"])
            attempt = int(row["attempt"] or 0)
            try:
                if state is S.DRAFT_POSTED:
                    await repo.transition(
                        tid, S.DONE, worker_id=RECOVERY_WORKER_ID, detail={"recovered": True}
                    )
                    out["finished"] += 1
                    continue
                if state is S.FAILED and tid in review_failures:
                    continue  # the review dispatcher retries it
                if state in _MID_PIPELINE:
                    await repo.transition(
                        tid,
                        S.FAILED,
                        worker_id=RECOVERY_WORKER_ID,
                        error=f"interrupted by restart in {state.value}",
                        bump_attempt=True,
                        detail={"recovered_from": state.value},
                    )
                    attempt += 1
                    state = S.FAILED
                if state is S.FAILED:
                    if attempt >= s.max_retries:
                        await repo.transition(
                            tid,
                            S.DEAD_LETTER,
                            worker_id=RECOVERY_WORKER_ID,
                            detail={"reason": "max_retries", "attempt": attempt, "recovered": True},
                        )
                        out["dead_lettered"] += 1
                        continue
                    await repo.transition(
                        tid, S.QUEUED, worker_id=RECOVERY_WORKER_ID, detail={"recovered": True}
                    )
                    out["requeued"] += 1
                if q.contains(tid):
                    continue
                if q.try_put(WorkItem(ticket_id=tid, trace_id=row["trace_id"], attempt=attempt)):
                    out["enqueued"] += 1
                else:
                    out["left_queued"] += 1  # poller's orphan pass enqueues it when there is room
            except InvalidTransition as e:
                log.warning("runtime.recovery_skip", ticket_id=tid, error=str(e))
        if rows:
            log.info("runtime.recovered", **out)
        return out

    # ------------------------------------------------------------------ status (ops API)

    async def status(self) -> dict[str, Any]:
        q, pool, sch = self.queue, self.pool, self.scheduler
        counts = await self.repo.counts_by_state() if self.repo is not None else {}  # type: ignore[attr-defined]
        return {
            "started": self.started,
            "started_at": self.started_at,
            "queue": {
                "depth": q.qsize() if q else 0,
                "capacity": q.maxsize if q else self.settings.queue_max_size,
                "held": len(q.held_ids()) if q else 0,
            },
            "workers": {
                "busy": pool.busy if pool else 0,
                "configured": pool.size if pool else self.settings.worker_count,
            },
            "poller": {
                "last_poll_at": self.last_poll_at,
                "last_run_at": sch.last_run_at if sch else None,
                "last_finished_at": sch.last_finished_at if sch else None,
                "last_error": sch.last_error if sch else None,
                "interval_s": self.settings.poll_interval_seconds,
                "running": sch.running if sch else False,
                "last_result": self.poller.last_result if self.poller else None,
            },
            "portal": {"started": self.portal_started, "error": self.portal_error},
            "review_dispatcher": {
                "running": self.dispatcher.running if self.dispatcher else False,
                "last_result": self.dispatcher.last_result if self.dispatcher else None,
            },
            "recovery": self.recovery,
            "tickets_by_state": counts,
        }


# ---------------------------------------------------------------------- CLI entrypoint


async def run(settings: Settings) -> None:
    """`support-agent run`: start everything, wait for SIGINT/SIGTERM, shut down gracefully."""
    configure_logging(settings.log_level, settings.log_file)
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    installed: list[int] = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
            installed.append(sig)
        except (NotImplementedError, RuntimeError):  # non-main thread / unsupported platform
            pass
    rt = Runtime(settings)
    try:
        await rt.start()
        await stop.wait()
        log.info("runtime.signal_received")
    finally:
        if rt.started:
            await rt.stop()
        for sig in installed:
            loop.remove_signal_handler(sig)
