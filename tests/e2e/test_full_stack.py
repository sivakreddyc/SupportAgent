"""End to end (PLAN §8, D7): mock ticket UI + mock user service (ephemeral ports) + the real `Runtime`
with no injected components: Playwright portal, HTTP user client, MockLLM, rule-based policy gate,
scheduler, poller, workers, review dispatcher, metric sampler and the ops API.

Expectations from the seed data: 53 tickets -> 19 DONE (auto_approve) + 34 NEEDS_REVIEW, each with the
exact `meta.expected_rules`; drafts / notes / statuses visible in the UI; /metrics deltas == DB rows;
approve + edit from the dashboard helpers get posted to the UI.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time

import httpx
import pytest
from prometheus_client.parser import text_string_to_metric_families

from support_agent.observability import metrics as m
from tests.conftest import require
from tests.fakes import load_seed_cases

pytestmark = [pytest.mark.e2e, pytest.mark.browser, pytest.mark.timeout(300)]

SEEDS = {c.ticket.ticket_id: c for c in load_seed_cases()}


def rows(db_path: str, sql: str, *params) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


async def wait_for(pred, *, within: float, every: float = 0.25, what: str = "condition"):
    deadline = time.monotonic() + within
    while True:
        result = await asyncio.to_thread(pred)
        if result:
            return result
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out after {within}s waiting for {what}")
        await asyncio.sleep(every)


def snapshot_counters() -> dict[tuple[str, tuple], float]:
    return {(n, tuple(sorted(lb.items()))): v for n, lb, v in m.snapshot()}


def parse_metrics(text: str) -> dict[tuple[str, tuple], float]:
    out = {}
    for fam in text_string_to_metric_families(text):
        for s in fam.samples:
            out[(s.name, tuple(sorted(s.labels.items())))] = s.value
    return out


def lf(text: str) -> str:
    """Browsers submit textarea values with CRLF line breaks; the UI store keeps them as received."""
    return text.replace("\r\n", "\n")


def delta(after: dict, before: dict, name: str, **labels) -> float:
    k = (name, tuple(sorted(labels.items())))
    return after.get(k, 0.0) - before.get(k, 0.0)


@pytest.fixture
def stack_settings(settings, start_mock_ui, start_user_service):
    ui = start_mock_ui()
    us = start_user_service()
    s = settings.model_copy(
        update={
            "ticket_ui_url": ui.url,
            "user_service_url": us.url,
            "worker_count": 4,
            "browser_page_pool": 3,
            "poll_interval_seconds": 0.5,
            "review_dispatch_interval_seconds": 0.2,
            "metrics_sample_interval_seconds": 0.5,
            "ticket_timeout_seconds": 30,
        }
    )
    return s, ui, us


async def test_full_stack_seed_run(stack_settings):
    Runtime = require("support_agent.pipeline.runtime", "Runtime")
    review = require("support_agent.storage.review")
    settings, ui, _us = stack_settings
    db = settings.db_path
    before = snapshot_counters()

    rt = Runtime(settings)
    await rt.start()
    try:
        assert rt.ops_port and rt.ops_port not in (8001, 8002, 8003, 8501)
        ops = f"http://127.0.0.1:{rt.ops_port}"

        # ---- every seeded ticket settles in DONE or NEEDS_REVIEW
        def settled():
            got = {r["ticket_id"]: r["state"] for r in rows(db, "SELECT ticket_id, state FROM tickets")}
            return got if len(got) == 53 and set(got.values()) <= {"DONE", "NEEDS_REVIEW"} else None

        states = await wait_for(settled, within=180, what="53 tickets to settle")
        assert set(states) == set(SEEDS)
        assert sum(v == "DONE" for v in states.values()) == 19
        assert sum(v == "NEEDS_REVIEW" for v in states.values()) == 34

        # ---- each seeded policy case lands in the expected outcome with the expected rules
        decisions = {r["ticket_id"]: r for r in rows(db, "SELECT * FROM policy_decisions")}
        mismatches = []
        for tid, case in SEEDS.items():
            want_state = "DONE" if case.expected_outcome == "auto_approve" else "NEEDS_REVIEW"
            got_rules = set(json.loads(decisions[tid]["reasons_json"]))
            if states[tid] != want_state or got_rules != set(case.expected_rules):
                mismatches.append((tid, states[tid], sorted(got_rules), case.expected_rules))
        assert not mismatches, mismatches

        # ---- drafts / notes / statuses visible in the ticket UI
        store = ui.app.state.store
        posted = {r["ticket_id"]: r for r in rows(db, "SELECT * FROM drafts WHERE status = 'posted'")}
        for tid, st in states.items():
            t = store.get(tid)
            if st == "DONE":
                assert len(t["drafts"]) == 1, tid  # exactly once
                assert lf(t["drafts"][0]["body"]) == posted[tid]["body"]
                assert t["drafts"][0]["idempotency_key"] == posted[tid]["idempotency_key"]
                assert t["status"] == "pending"
            else:
                assert t["drafts"] == [], tid
                assert len(t["notes"]) == 1 and t["notes"][0]["body"].startswith("Escalated"), tid
                assert t["status"] == "escalated"
        async with httpx.AsyncClient(base_url=ui.url, follow_redirects=True) as c:
            await c.post("/login", data={"username": "agent", "password": "agent", "next": "/tickets"})
            done_tid = next(t for t, s in states.items() if s == "DONE")
            html = (await c.get(f"/tickets/{done_tid}")).text
            assert posted[done_tid]["idempotency_key"] in html

        # ---- ops API
        async with httpx.AsyncClient(base_url=ops, timeout=10) as c:
            assert (await c.get("/health")).status_code == 200
            ready = await c.get("/ready")
            assert ready.status_code == 200, ready.text
            status = (await c.get("/api/status")).json()
            assert status["ready"]["ready"] is True
            after = parse_metrics((await c.get("/metrics")).text)

        # ---- /metrics agree with the DB
        assert delta(after, before, "tickets_discovered_total") == 53
        assert delta(after, before, "tickets_processed_total", outcome="auto_approve") == 19
        assert delta(after, before, "tickets_processed_total", outcome="human_review") == 34
        assert delta(after, before, "policy_decisions_total", outcome="auto_approve") == 19
        assert delta(after, before, "policy_decisions_total", outcome="human_review") == 34
        assert delta(after, before, "drafts_posted_total", source="llm") == len(posted) == 19
        llm_model = rows(db, "SELECT DISTINCT model FROM llm_calls")[0]["model"]
        n_calls = rows(db, "SELECT COUNT(*) n FROM llm_calls")[0]["n"]
        llm_calls = sum(
            delta(after, before, "llm_calls_total", model=llm_model, result=r)
            for r in ("ok", "refusal", "parse_error", "error")
        )
        assert llm_calls == n_calls
        in_tokens = rows(db, "SELECT SUM(input_tokens) n FROM llm_calls")[0]["n"]
        assert delta(after, before, "llm_tokens_total", model=llm_model, direction="input") == in_tokens
        for rule in {r for c in SEEDS.values() for r in c.expected_rules}:
            expected_hits = sum(rule in c.expected_rules for c in SEEDS.values())
            assert delta(after, before, "policy_rule_hits_total", rule=rule) == expected_hits, rule
        found = rows(db, "SELECT SUM(found) f, COUNT(*) n FROM user_lookups")[0]
        assert found["n"] == 53 and found["f"] == 50  # 3 unknown customers
        assert after[("queue_capacity", ())] == settings.queue_max_size
        assert after[("workers_configured", ())] == 4

        # ---- observability rows
        assert rows(db, "SELECT COUNT(*) n FROM scheduler_runs WHERE error IS NULL")[0]["n"] >= 1
        assert rows(db, "SELECT COUNT(*) n FROM spans WHERE name LIKE 'worker.%'")[0]["n"] >= 53 * 5
        await wait_for(
            lambda: rows(db, "SELECT COUNT(*) n FROM metric_samples")[0]["n"] > 0,
            within=5,
            what="metric samples",
        )

        # ---- human review: approve one, edit another -> posted by the review dispatcher
        pending = rows(
            db,
            """SELECT t.ticket_id, d.body FROM tickets t JOIN drafts d ON d.ticket_id = t.ticket_id
               WHERE t.state = 'NEEDS_REVIEW' AND d.body != '' ORDER BY t.ticket_id""",
        )
        (approve_tid, approve_body), (edit_tid, _) = (
            (pending[0]["ticket_id"], pending[0]["body"]),
            (
                pending[1]["ticket_id"],
                None,
            ),
        )
        await asyncio.to_thread(review.apply_review, db, approve_tid, "approve", "e2e")
        await asyncio.to_thread(review.apply_review, db, edit_tid, "edit", "e2e", "Edited reply from e2e")

        def reviewed_done():
            got = {r["ticket_id"]: r["state"] for r in rows(db, "SELECT ticket_id, state FROM tickets")}
            return got[approve_tid] == got[edit_tid] == "DONE"

        await wait_for(reviewed_done, within=30, what="reviewed tickets to be posted")
        assert [lf(d["body"]) for d in store.get(approve_tid)["drafts"]] == [approve_body]
        assert [lf(d["body"]) for d in store.get(edit_tid)["drafts"]] == ["Edited reply from e2e"]
        assert store.get(approve_tid)["status"] == "pending"
        final = parse_metrics(m.render_latest().decode())
        assert delta(final, before, "reviews_total", action="approve") == 1
        assert delta(final, before, "reviews_total", action="edit") == 1
        assert delta(final, before, "drafts_posted_total", source="human_edit") == 1

        # ---- a second poll cycle re-discovers nothing and posts nothing new
        await asyncio.sleep(1.2)
        assert rows(db, "SELECT COUNT(*) n FROM tickets")[0]["n"] == 53
        assert sum(len(store.get(t)["drafts"]) for t in SEEDS) == 19 + 2
    finally:
        await rt.stop()
    assert not rt.started


async def test_ready_reports_unreachable_ui(settings, start_user_service):
    """The runtime starts even if the ticket UI is down; /ready says why it isn't ready."""
    Runtime = require("support_agent.pipeline.runtime", "Runtime")
    us = start_user_service()
    s = settings.model_copy(
        update={"user_service_url": us.url, "ticket_ui_url": "http://127.0.0.1:1", "browser_timeout_ms": 2000}
    )
    rt = Runtime(s)
    await rt.start()
    try:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{rt.ops_port}", timeout=10) as c:
            assert (await c.get("/health")).status_code == 200
            r = await c.get("/ready")
            assert r.status_code == 503
            assert r.json()["ready"] is False and r.json()["reasons"]
    finally:
        await rt.stop()
