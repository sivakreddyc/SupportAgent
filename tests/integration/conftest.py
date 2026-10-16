"""Pipeline harness: real SqliteRepository + BoundedTicketQueue + DiscoveryPoller + WorkerPool (+ ReviewDispatcher),
with in-memory fakes for the portal, user service, LLM and (optionally) policy."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable
from typing import Any

import pytest

from support_agent.common.states import TicketState
from tests.conftest import require
from tests.fakes import (
    FakeLLMClient,
    FakePolicyGate,
    FakeTicketPortal,
    FakeUserInfoClient,
    make_ticket,
    make_user,
)

S = TicketState
SETTLED = {S.DONE, S.NEEDS_REVIEW, S.DEAD_LETTER}


class Harness:
    def __init__(self, settings, *, queue_size: int = 1000, workers: int = 2, policy=None) -> None:
        self.SqliteRepository = require("support_agent.storage.repository", "SqliteRepository")
        self.BoundedTicketQueue = require("support_agent.pipeline.queue", "BoundedTicketQueue")
        self.DiscoveryPoller = require("support_agent.pipeline.poller", "DiscoveryPoller")
        self.WorkerPool = require("support_agent.pipeline.worker", "WorkerPool")
        self.settings = settings
        self.repo = self.SqliteRepository(settings.db_path)
        self.queue = self.BoundedTicketQueue(queue_size)
        self.portal = FakeTicketPortal()
        self.users = FakeUserInfoClient()
        self.llm = FakeLLMClient()
        self.policy = policy or FakePolicyGate()
        self.n_workers = workers
        self.pool = None
        self.poller = None

    async def __aenter__(self) -> Harness:
        await self.repo.init()
        self.poller = self.DiscoveryPoller(self.portal, self.repo, self.queue)
        return self

    async def __aexit__(self, *exc) -> None:
        if self.pool is not None:
            await self.pool.stop()
        await self.repo.close()

    # ------------------------------------------------------------------ setup helpers

    def add_tickets(self, n: int = 1, *, known_user: bool = True, **ticket_kw) -> list[str]:
        ids = []
        for _ in range(n):
            t = make_ticket(**ticket_kw)
            self.portal.add(t)
            if known_user:
                self.users.add(make_user(t.customer_email))
            ids.append(t.ticket_id)
        return ids

    def add(self, ticket, user=None) -> str:
        self.portal.add(ticket)
        if user is not None:
            self.users.add(user)
        return ticket.ticket_id

    async def start_workers(self, n: int | None = None) -> None:
        self.pool = self.WorkerPool(
            n or self.n_workers,
            self.queue,
            self.repo,
            self.portal,
            self.users,
            self.llm,
            self.policy,
            self.settings,
        )
        await self.pool.start()

    async def run(self, within: float = 10.0) -> dict[str, str]:
        """Poll once, start workers, wait until every known ticket is settled."""
        await self.poller.run_once()
        if self.pool is None:
            await self.start_workers()
        return await self.wait_settled(self.portal.tickets, within=within)

    # ------------------------------------------------------------------ waiting / inspection

    async def state(self, tid: str) -> str | None:
        row = await self.repo.get_ticket_row(tid)
        return row["state"] if row else None

    async def wait_settled(
        self, ids: Iterable[str], *, states: set[TicketState] = SETTLED, within: float = 10.0
    ) -> dict[str, str]:
        ids = list(ids)
        want = {s.value for s in states}
        deadline = time.monotonic() + within
        while True:
            got = {tid: await self.state(tid) for tid in ids}
            if all(v in want for v in got.values()):
                return got
            if time.monotonic() > deadline:
                raise AssertionError(f"not settled after {within}s: {got}")
            await asyncio.sleep(0.02)

    async def path(self, tid: str) -> list[str]:
        return [t["to_state"] for t in await self.repo.transitions(tid)]

    def sql(self, query: str, *params) -> list[dict[str, Any]]:
        import sqlite3

        conn = sqlite3.connect(self.settings.db_path)
        conn.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in conn.execute(query, params).fetchall()]
        finally:
            conn.close()


@pytest.fixture
def harness(settings):
    def make(**kw) -> Harness:
        return Harness(settings, **kw)

    return make
