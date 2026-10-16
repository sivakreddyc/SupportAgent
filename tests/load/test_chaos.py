"""Chaos (PLAN §8, D9): the full stack (Playwright portal, HTTP user client, MockLLM, real gate, Runtime)
under injected failures. Every seed ticket must still reach its expected outcome, with each auto-approved
ticket's draft posted exactly once. Internal notes have no idempotency key, so they are at-least-once (≥1).

1. `test_chaos_ui_errors_and_slow_user_service`: UI error_rate 0.2 (500 before the handler runs) and a
   user service with 1 s latency (the client timeout stays at the 2 s default).
2. `test_chaos_lost_responses_after_write`: the nastier case for idempotency: the UI applies a draft/note
   POST and then answers 500 (response lost after the write) 30% of the time, on top of error_rate 0.1.
"""

from __future__ import annotations

import asyncio
import json
import random
import sqlite3
import time
from collections import Counter

import pytest

from tests.conftest import require
from tests.fakes import load_seed_cases

pytestmark = [pytest.mark.load, pytest.mark.e2e, pytest.mark.browser, pytest.mark.timeout(600)]

SEEDS = {c.ticket.ticket_id: c for c in load_seed_cases()}


def rows(db_path: str, sql: str, *params) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


async def wait_settled(db: str, n: int, *, within: float) -> dict[str, str]:
    deadline = time.monotonic() + within
    while True:
        got = {
            r["ticket_id"]: r["state"]
            for r in await asyncio.to_thread(rows, db, "SELECT ticket_id, state FROM tickets")
        }
        if len(got) == n and set(got.values()) <= {"DONE", "NEEDS_REVIEW", "DEAD_LETTER"}:
            return got
        if time.monotonic() > deadline:
            raise AssertionError(f"not settled in {within}s: {Counter(got.values())}")
        await asyncio.sleep(0.25)


class LoseResponseAfterWrite:
    """ASGI wrapper: run POST /tickets/*/draft|note normally, then (with probability p) replace the
    response with a 500, as if the connection dropped after the server committed the write."""

    def __init__(self, app, p: float, seed: int = 1234) -> None:
        self.app = app
        self.state = app.state
        self.p = p
        self.rng = random.Random(seed)
        self.lost = Counter()

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        if not (
            scope["type"] == "http"
            and scope["method"] == "POST"
            and path.startswith("/tickets/")
            and path.endswith(("/draft", "/note"))
            and self.rng.random() < self.p
        ):
            return await self.app(scope, receive, send)

        async def swallow(message):  # let the handler run to completion, drop its response
            return None

        await self.app(scope, receive, swallow)
        self.lost[path.rsplit("/", 1)[-1]] += 1
        body = b'<html><body data-testid="error-page">connection lost after write</body></html>'
        await send(
            {"type": "http.response.start", "status": 500, "headers": [(b"content-type", b"text/html")]}
        )
        await send({"type": "http.response.body", "body": body})


def assert_outcomes(db: str, states: dict[str, str], store) -> None:
    assert set(states) == set(SEEDS)
    dlq = [t for t, s in states.items() if s == "DEAD_LETTER"]
    assert not dlq, [(t, rows(db, "SELECT last_error FROM tickets WHERE ticket_id = ?", t)) for t in dlq]
    wrong = [
        t
        for t, c in SEEDS.items()
        if states[t] != ("DONE" if c.expected_outcome == "auto_approve" else "NEEDS_REVIEW")
    ]
    assert not wrong, wrong
    for tid, st in states.items():
        t = store.get(tid)
        if st == "DONE":
            assert len(t["drafts"]) == 1, (
                tid,
                [d["idempotency_key"] for d in t["drafts"]],
            )  # NO double posts
        else:
            assert t["drafts"] == [], tid
            assert len(t["notes"]) >= 1, tid  # at-least-once
            assert all(n["body"].startswith("Escalated") for n in t["notes"])
    reasons = {
        r["ticket_id"]: set(json.loads(r["reasons_json"])) for r in rows(db, "SELECT * FROM policy_decisions")
    }
    assert all(reasons[t] == set(c.expected_rules) for t, c in SEEDS.items())


def chaos_settings(settings, ui_url: str, us_url: str):
    return settings.model_copy(
        update={
            "ticket_ui_url": ui_url,
            "user_service_url": us_url,
            "user_service_timeout_seconds": 2.0,  # PLAN default; the service is slowed to 1 s
            "worker_count": 6,
            "browser_page_pool": 3,
            "max_retries": 6,
            "poll_interval_seconds": 0.5,
            "review_dispatch_interval_seconds": 0.3,
            "ticket_timeout_seconds": 60,
        }
    )


async def test_chaos_ui_errors_and_slow_user_service(settings, start_mock_ui, start_user_service):
    Runtime = require("support_agent.pipeline.runtime", "Runtime")
    ui = start_mock_ui(error_rate=0.2)
    us = start_user_service(latency_ms=1000)
    s = chaos_settings(settings, ui.url, us.url)
    rt = Runtime(s)
    await rt.start()
    try:
        states = await wait_settled(s.db_path, len(SEEDS), within=420)
        retried = rows(s.db_path, "SELECT COUNT(*) n FROM ticket_transitions WHERE to_state = 'FAILED'")[0][
            "n"
        ]
        slow = rows(s.db_path, "SELECT COUNT(*) n FROM user_lookups WHERE latency_ms >= 900")[0]["n"]
    finally:
        await rt.stop()
    assert_outcomes(s.db_path, states, ui.app.state.store)
    assert slow >= 40  # the service really was slow (~50 distinct emails; repeats are cache hits)
    print(f"worker-level retries under chaos: {retried}")


async def test_chaos_lost_responses_after_write(settings, serve, start_user_service):
    Runtime = require("support_agent.pipeline.runtime", "Runtime")
    create_app = require("support_agent.mocks.ticket_ui.app", "create_app")
    wrapper = LoseResponseAfterWrite(create_app(error_rate=0.1), p=0.3)
    ui = serve(wrapper)
    us = start_user_service()
    s = chaos_settings(settings, ui.url, us.url)
    rt = Runtime(s)
    await rt.start()
    try:
        states = await wait_settled(s.db_path, len(SEEDS), within=420)
    finally:
        await rt.stop()
    assert wrapper.lost["draft"] > 0 and wrapper.lost["note"] > 0, wrapper.lost  # chaos actually happened
    assert_outcomes(s.db_path, states, wrapper.state.store)
