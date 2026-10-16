"""Load / backpressure (PLAN §8, D8): 2000 generated tickets, queue size 100, a slow LLM.

Nothing lost, the queue never exceeds its capacity, rejections are counted (metric == scheduler_runs), every
ticket eventually lands in its expected outcome, and each auto-approved ticket gets exactly one draft.

* `test_backpressure_2000_runtime_fake_portal`: the real Runtime (scheduler, poller, queue, workers, gate,
  MockLLM with latency) over an in-memory portal and user service. Isolates the queue mechanics; ~20 s.
* `test_backpressure_2000_full_stack`: nothing injected: mock UI (2000 tickets via /admin/generate) driven
  by Playwright, real user service over HTTP. Several minutes.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from collections import Counter

import httpx
import pytest

from support_agent.observability import metrics as m
from tests.conftest import require
from tests.fakes import POLICY_FILE, FakeTicketPortal, FakeUserInfoClient, seed_case_from_raw

pytestmark = [pytest.mark.load, pytest.mark.timeout(900)]

N = 2000
QUEUE = 100


def rows(db_path: str, sql: str, *params) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def generated_cases(n: int = N, seed: int = 7, start_number: int = 1054):
    generator = require("support_agent.mocks.ticket_ui.generator")
    return [seed_case_from_raw(raw) for raw in generator.generate(n, seed, start_number=start_number)]


def metric(name: str, **labels) -> float:
    return m.REGISTRY.get_sample_value(name, labels) or 0.0


class DepthProbe:
    """Samples the runtime queue depth every few ms while the run is in progress."""

    def __init__(self, queue) -> None:
        self.queue = queue
        self.peak = 0
        self.samples = 0
        self._task: asyncio.Task | None = None

    async def _run(self) -> None:
        while True:
            self.peak = max(self.peak, self.queue.qsize())
            self.samples += 1
            await asyncio.sleep(0.003)

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)


async def wait_all_settled(db: str, n: int, *, within: float) -> dict[str, str]:
    deadline = time.monotonic() + within
    while True:
        got = {
            r["ticket_id"]: r["state"]
            for r in await asyncio.to_thread(rows, db, "SELECT ticket_id, state FROM tickets")
        }
        if len(got) == n and set(got.values()) <= {"DONE", "NEEDS_REVIEW", "DEAD_LETTER"}:
            return got
        if time.monotonic() > deadline:
            raise AssertionError(f"not settled in {within}s: {Counter(got.values())} of {len(got)}/{n}")
        await asyncio.sleep(0.5)


def check_run(db: str, states: dict[str, str], cases, rej0: float, drafts_by_ticket: dict[str, int]) -> None:
    expected = {c.ticket.ticket_id: c.expected_outcome for c in cases}
    assert set(states) == set(expected), "tickets lost or invented"
    assert Counter(states.values())["DEAD_LETTER"] == 0
    wrong = [
        t for t, s in states.items() if s != ("DONE" if expected[t] == "auto_approve" else "NEEDS_REVIEW")
    ]
    assert not wrong, wrong[:10]
    runs = rows(db, "SELECT SUM(skipped_full) s, COUNT(*) n, SUM(error IS NOT NULL) e FROM scheduler_runs")[0]
    assert runs["s"] > 0, "queue never filled: backpressure was not exercised"
    assert metric("queue_rejections_total") - rej0 == runs["s"], (
        "rejections metric != scheduler_runs.skipped_full"
    )
    done = [t for t, s in states.items() if s == "DONE"]
    assert all(drafts_by_ticket.get(t, 0) == 1 for t in done), "a DONE ticket has != 1 draft"
    assert all(drafts_by_ticket.get(t, 0) == 0 for t, s in states.items() if s == "NEEDS_REVIEW")


async def test_backpressure_2000_runtime_fake_portal(settings):
    Runtime = require("support_agent.pipeline.runtime", "Runtime")
    gate = require("support_agent.policy.gate", "RuleBasedPolicyGate").from_file(POLICY_FILE)
    MockLLMClient = require("support_agent.llm.mock_client", "MockLLMClient")
    cases = generated_cases()
    portal = FakeTicketPortal([c.ticket for c in cases])
    users = FakeUserInfoClient([c.user for c in cases if c.user is not None])
    s = settings.model_copy(
        update={
            "queue_max_size": QUEUE,
            "worker_count": 8,
            "mock_llm_latency_ms": 20,  # slow LLM: workers, not the poller, are the bottleneck
            "poll_interval_seconds": 0.2,
        }
    )
    llm = MockLLMClient(s)
    rej0 = metric("queue_rejections_total")
    rt = Runtime(s, portal=portal, user_client=users, llm=llm, policy=gate)
    await rt.start()
    probe = DepthProbe(rt.queue)
    probe.start()
    try:
        assert rt.queue.maxsize == QUEUE
        states = await wait_all_settled(s.db_path, N, within=600)
    finally:
        await probe.stop()
        await rt.stop()
    assert probe.samples > 100
    assert 0 < probe.peak <= QUEUE
    assert probe.peak == QUEUE, f"queue never reached capacity (peak {probe.peak})"
    drafts = Counter(d.ticket_id for d in portal.drafts)
    check_run(s.db_path, states, cases, rej0, drafts)
    assert portal.draft_submissions == sum(drafts.values())


@pytest.mark.browser
@pytest.mark.e2e
async def test_backpressure_2000_full_stack(settings, start_mock_ui, start_user_service):
    Runtime = require("support_agent.pipeline.runtime", "Runtime")
    ui = start_mock_ui(seed=False, page_size=100)
    us = start_user_service()
    async with httpx.AsyncClient(base_url=ui.url) as c:
        r = await c.post("/admin/generate", data={"n": N, "seed": 7})
        assert r.status_code in (200, 303)
    store = ui.app.state.store
    in_ui = store.ordered("open")
    cases = generated_cases(start_number=int(in_ui[0]["ticket_id"].split("-")[1]))
    by_id = {t["ticket_id"]: t for t in in_ui}
    assert set(by_id) == {c.ticket.ticket_id for c in cases}
    assert all(by_id[c.ticket.ticket_id]["body"] == c.ticket.body for c in cases)  # same content -> same meta
    s = settings.model_copy(
        update={
            "ticket_ui_url": ui.url,
            "user_service_url": us.url,
            "queue_max_size": QUEUE,
            "worker_count": 8,
            "browser_page_pool": 4,
            "mock_llm_latency_ms": 50,
            "poll_interval_seconds": 1.0,
            "ticket_timeout_seconds": 60,
        }
    )
    rej0 = metric("queue_rejections_total")
    rt = Runtime(s)
    await rt.start()
    probe = DepthProbe(rt.queue)
    probe.start()
    try:
        states = await wait_all_settled(s.db_path, N, within=840)
    finally:
        await probe.stop()
        await rt.stop()
    assert 0 < probe.peak <= QUEUE
    drafts = {t: len(store.get(t)["drafts"]) for t in states}
    check_run(s.db_path, states, cases, rej0, drafts)
