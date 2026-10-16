"""Poller + workers with real storage/queue and fakes (PLAN §5.6–5.8, B5–B7, B9):
dedup, backpressure, happy path, every policy rule -> NEEDS_REVIEW + note, retries -> DLQ,
non-retryable -> DLQ, timeout, idempotent posting under retries, crash recovery."""

from __future__ import annotations

import asyncio
import itertools

import pytest

from support_agent.common.contracts import PortalError, UserServiceError
from support_agent.common.states import TicketState
from support_agent.observability import metrics as m
from support_agent.observability.tracing import set_span_sink
from tests.conftest import require
from tests.fakes import (
    POLICY_FILE,
    fatal_llm_error,
    make_analysis,
    make_result,
    make_ticket,
    make_user,
    retryable_llm_error,
)

S = TicketState
HAPPY = [
    "DISCOVERED",
    "QUEUED",
    "PROCESSING",
    "ENRICHED",
    "ANALYZED",
    "POLICY_CHECKED",
    "DRAFT_POSTED",
    "DONE",
]


def metric(name: str, **labels) -> float:
    return m.REGISTRY.get_sample_value(name, labels) or 0.0


# =========================================================================== poller


async def test_poller_discovers_and_enqueues(harness):
    async with harness() as h:
        ids = h.add_tickets(5)
        res = await h.poller.run_once()
        assert (res["found"], res["new"], res["enqueued"], res["skipped_full"], res["error"]) == (
            5,
            5,
            5,
            0,
            None,
        )
        assert h.queue.qsize() == 5
        for tid in ids:
            assert await h.state(tid) == "QUEUED"
            assert h.queue.contains(tid)
        run = await h.repo.last_scheduler_run()
        assert (run["found"], run["enqueued"], run["skipped_full"], run["error"]) == (5, 5, 0, None)


async def test_poller_dedups(harness):
    async with harness() as h:
        h.add_tickets(3)
        await h.poller.run_once()
        res = await h.poller.run_once()
        assert (res["found"], res["new"], res["enqueued"]) == (3, 0, 0)
        assert h.queue.qsize() == 3
        assert len(h.sql("SELECT * FROM tickets")) == 3


async def test_poller_never_reenqueues_processed_ticket(harness):
    async with harness() as h:
        (tid,) = h.add_tickets(1)
        await h.run()
        assert await h.state(tid) == "DONE"
        await h.pool.stop()
        h.pool = None
        res = await h.poller.run_once()  # ticket still "open" on the fake UI (status 'pending' set though)
        assert res["enqueued"] == 0 and not h.queue.contains(tid)
        assert await h.state(tid) == "DONE"


async def test_every_queued_item_is_already_queued_in_db(harness):
    """Transition-first: a worker can never dequeue a ticket that is still DISCOVERED."""
    async with harness() as h:
        h.add_tickets(20)
        await h.poller.run_once()
        for tid in h.queue.held_ids():
            assert await h.state(tid) == "QUEUED"


async def test_backpressure_when_queue_full(harness):
    async with harness(queue_size=3) as h:
        ids = h.add_tickets(10)
        rej0 = metric("queue_rejections_total")
        res = await h.poller.run_once()
        assert (res["enqueued"], res["skipped_full"]) == (3, 7)
        assert metric("queue_rejections_total") - rej0 == 7
        states = [await h.state(t) for t in ids]
        assert states.count("QUEUED") == 3 and states.count("DISCOVERED") == 7
        assert h.queue.qsize() == 3 == h.queue.maxsize
        assert (await h.repo.last_scheduler_run())["skipped_full"] == 7
        # drain, then the next poll picks up the leftovers (oldest first)
        for _ in range(3):  # three workers claim their tickets
            item = await h.queue.get()
            await h.repo.transition(item.ticket_id, S.PROCESSING)
            h.queue.task_done(item.ticket_id)
        res = await h.poller.run_once()
        assert (res["enqueued"], res["requeued"], res["new"]) == (3, 0, 0)
        assert [await h.state(t) for t in ids].count("DISCOVERED") == 4


async def test_backpressure_nothing_lost_end_to_end(harness):
    async with harness(queue_size=5, workers=3) as h:
        ids = h.add_tickets(40)
        h.llm.faults.on("analyze").delay(0.01)
        await h.start_workers()
        peak = 0
        for _ in range(200):
            await h.poller.run_once()
            peak = max(peak, h.queue.qsize())
            states = {await h.state(t) for t in ids}
            if states == {"DONE"}:
                break
            await asyncio.sleep(0.02)
        assert {await h.state(t) for t in ids} == {"DONE"}
        assert peak <= 5
        assert len(h.portal.drafts) == 40


async def test_poller_portal_error_is_recorded_not_raised(harness):
    async with harness() as h:
        h.portal.faults.on("list_open_tickets").fail(1, PortalError("UI down"))
        err0 = metric("poller_runs_total", result="error")
        res = await h.poller.run_once()
        assert res["error"] == "PortalError: UI down"
        assert metric("poller_runs_total", result="error") - err0 == 1
        assert (await h.repo.last_scheduler_run())["error"] == "PortalError: UI down"


async def test_poller_requeues_orphaned_queued(harness):
    async with harness() as h:
        (tid,) = h.add_tickets(1)
        await h.poller.run_once()
        item = await h.queue.get()  # simulate an item lost from memory while QUEUED in DB
        h.queue.task_done(item.ticket_id)
        assert not h.queue.contains(tid)
        res = await h.poller.run_once()
        assert res["requeued"] == 1 and h.queue.contains(tid)


# =========================================================================== workers: happy path


async def test_happy_path_done_with_one_draft(harness):
    async with harness() as h:
        spans: list[dict] = []

        async def sink(rec):
            spans.append(rec)
            await h.repo.record_span(rec)

        set_span_sink(sink)
        (tid,) = h.add_tickets(1)
        done0 = metric("tickets_processed_total", outcome="auto_approve")
        assert (await h.run())[tid] == "DONE"
        assert await h.path(tid) == HAPPY
        drafts = h.portal.drafts_for(tid)
        assert len(drafts) == 1 and h.portal.draft_submissions == 1
        assert drafts[0].body == h.llm.default.analysis.draft_reply if h.llm.default else True
        row = h.sql("SELECT * FROM drafts WHERE ticket_id = ?", tid)
        assert len(row) == 1 and row[0]["status"] == "posted" and row[0]["posted_at"]
        assert row[0]["idempotency_key"] == drafts[0].idempotency_key
        assert h.portal.statuses[tid] == ["pending"]
        assert h.portal.notes[tid] == []
        assert len(h.sql("SELECT * FROM ticket_snapshots WHERE ticket_id = ?", tid)) == 1
        assert h.sql("SELECT found FROM user_lookups WHERE ticket_id = ?", tid)[0]["found"] == 1
        assert h.sql("SELECT success FROM llm_calls WHERE ticket_id = ?", tid)[0]["success"] == 1
        assert (
            h.sql("SELECT outcome FROM policy_decisions WHERE ticket_id = ?", tid)[0]["outcome"]
            == "auto_approve"
        )
        assert metric("tickets_processed_total", outcome="auto_approve") - done0 == 1
        names = {s["name"] for s in spans if s["ticket_id"] == tid}
        assert {"worker.fetch", "worker.user_lookup", "worker.llm", "worker.policy", "worker.post"} <= names
        trace = (await h.repo.get_ticket_row(tid))["trace_id"]
        assert all(s["trace_id"] == trace for s in spans if s["name"].startswith("worker."))


async def test_many_tickets_parallel_workers(harness):
    async with harness(workers=4) as h:
        ids = h.add_tickets(30)
        h.llm.faults.on("analyze").delay(0.03)
        got = await h.run(within=20)
        assert set(got.values()) == {"DONE"}
        assert 2 <= h.llm.max_concurrent <= 4
        assert len(h.portal.drafts) == 30
        assert {d.ticket_id for d in h.portal.drafts} == set(ids)
        assert metric("workers_configured") == 4


async def test_duplicate_queue_item_processed_once(harness):
    async with harness() as h:
        (tid,) = h.add_tickets(1)
        await h.poller.run_once()
        from support_agent.common.models import WorkItem

        await h.queue.put(WorkItem(ticket_id=tid, trace_id="dup"), timeout=1)  # may be refused as held
        await h.start_workers()
        await h.wait_settled([tid])
        await asyncio.sleep(0.1)
        assert h.llm.calls.count(tid) == 1
        assert len(h.portal.drafts_for(tid)) == 1


# =========================================================================== policy routing


RULE_CASES = {
    "LOW_CONFIDENCE": ({"confidence": 0.4}, {}, None),
    "REFUND": ({"refund_requested": True}, {}, None),
    "REFUND_HIGH_VALUE": ({"refund_requested": True, "refund_amount": 500}, {}, None),
    "LEGAL": ({"legal_flag": True}, {}, None),
    "SECURITY": ({"security_flag": True}, {}, None),
    "ANGRY": ({"sentiment": "angry"}, {}, None),
    "CRITICAL_URGENCY": ({"urgency": "critical"}, {}, None),
    "VIP": ({}, {"vip": True}, None),
    "ACCOUNT_NOT_ACTIVE": ({}, {"account_status": "suspended"}, None),
    "UNKNOWN_USER": ({}, None, None),
    "UNSAFE_DRAFT": ({"draft_reply": "I guarantee a full refund."}, {}, None),
    "LLM_REFUSAL": (None, {}, "refusal"),
    "LLM_NO_ANALYSIS": (None, {}, "max_tokens"),
}


@pytest.mark.parametrize("rule", list(RULE_CASES))
async def test_each_rule_routes_to_review_with_note(harness, rule):
    gate = require("support_agent.policy.gate", "RuleBasedPolicyGate").from_file(POLICY_FILE)
    analysis_kw, user_kw, stop = RULE_CASES[rule]
    async with harness(policy=gate) as h:
        t = make_ticket()
        tid = h.add(t, None if user_kw is None else make_user(t.customer_email, **user_kw))
        if analysis_kw is None:
            h.llm.respond(tid, make_result(no_analysis=True, stop_reason=stop))
        else:
            h.llm.respond(tid, make_analysis(**analysis_kw))
        review0 = metric("tickets_processed_total", outcome="human_review")
        hits0 = metric("policy_rule_hits_total", rule=rule)
        assert (await h.run())[tid] == "NEEDS_REVIEW"
        assert h.portal.drafts_for(tid) == []  # nothing posted to the customer
        assert len(h.portal.notes[tid]) == 1 and rule in h.portal.notes[tid][0]
        assert h.portal.statuses[tid] == ["escalated"]
        d = h.sql("SELECT * FROM drafts WHERE ticket_id = ?", tid)
        assert len(d) == 1 and d[0]["status"] == "pending_review"
        pd = h.sql("SELECT * FROM policy_decisions WHERE ticket_id = ?", tid)[0]
        assert pd["outcome"] == "human_review" and rule in pd["reasons_json"]
        assert metric("tickets_processed_total", outcome="human_review") - review0 == 1
        assert metric("policy_rule_hits_total", rule=rule) - hits0 == 1
        assert (await h.repo.pending_reviews())[0]["ticket_id"] == tid


async def test_retried_review_ticket_posts_note_once(harness):
    async with harness() as h:
        (tid,) = h.add_tickets(1)
        h.policy.review(tid, "ANGRY")
        h.portal.faults.on("set_status").fail(1, PortalError("500"))  # fails after the note was posted
        assert (await h.run())[tid] == "NEEDS_REVIEW"
        assert h.portal.notes[tid] == ["Escalated for human review: ANGRY"]


# =========================================================================== failures, retries, DLQ


async def test_transient_failure_retried_to_done(harness):
    async with harness() as h:
        (tid,) = h.add_tickets(1)
        h.llm.faults.on("analyze").fail(2, retryable_llm_error())
        assert (await h.run())[tid] == "DONE"
        path = await h.path(tid)
        assert path.count("FAILED") == 2 and path[-1] == "DONE"
        row = await h.repo.get_ticket_row(tid)
        assert row["attempt"] == 2 and "overloaded" in row["last_error"]
        assert len(h.portal.drafts_for(tid)) == 1
        assert len(h.sql("SELECT * FROM llm_calls WHERE ticket_id = ? AND success = 0", tid)) == 2


@pytest.mark.parametrize(
    "op,exc",
    [
        ("analyze", retryable_llm_error()),
        ("get_ticket", PortalError("500")),
        ("get_user_by_email", UserServiceError("503")),
        ("post_draft", PortalError("500")),
    ],
)
async def test_retry_exhaustion_dead_letters(harness, settings, op, exc):
    async with harness() as h:
        (tid,) = h.add_tickets(1)
        target = {"analyze": h.llm, "get_ticket": h.portal, "post_draft": h.portal}.get(op, h.users)
        target.faults.on(op).fail(-1, exc)
        dl0 = metric("tickets_dead_letter_total")
        assert (await h.run())[tid] == "DEAD_LETTER"
        row = await h.repo.get_ticket_row(tid)
        assert row["attempt"] == settings.max_retries == 3
        assert (await h.path(tid)).count("FAILED") == 3
        assert metric("tickets_dead_letter_total") - dl0 == 1
        assert row["finished_at"] and row["last_error"]
        last = (await h.repo.transitions(tid))[-1]
        assert last["to_state"] == "DEAD_LETTER" and last["detail"]["reason"] == "max_retries"
        assert h.portal.drafts_for(tid) == []


async def test_non_retryable_llm_error_dead_letters_immediately(harness):
    async with harness() as h:
        (tid,) = h.add_tickets(1)
        h.llm.respond(tid, fatal_llm_error("BadRequestError (400)"))
        assert (await h.run())[tid] == "DEAD_LETTER"
        assert (await h.path(tid)).count("FAILED") == 1
        assert h.llm.calls.count(tid) == 1
        last = (await h.repo.transitions(tid))[-1]
        assert last["detail"]["reason"] == "non_retryable"


async def test_ticket_timeout(harness, settings):
    h_settings = settings.model_copy(update={"ticket_timeout_seconds": 0.2, "max_retries": 2})
    from tests.integration.conftest import Harness

    async with Harness(h_settings) as h:
        (tid,) = h.add_tickets(1)
        h.llm.faults.on("analyze").delay(1.0)
        assert (await h.run(within=10))[tid] == "DEAD_LETTER"
        assert "timed out" in (await h.repo.get_ticket_row(tid))["last_error"]


async def test_bad_ticket_does_not_block_others(harness):
    async with harness(workers=2) as h:
        ids = h.add_tickets(6)
        h.llm.respond(ids[0], fatal_llm_error())
        got = await h.run()
        assert got[ids[0]] == "DEAD_LETTER"
        assert all(got[t] == "DONE" for t in ids[1:])


async def test_failure_after_post_does_not_double_post_deterministic_llm(harness):
    async with harness() as h:
        (tid,) = h.add_tickets(1)
        h.portal.faults.on("set_status").fail(1, PortalError("500 after post"))
        assert (await h.run())[tid] == "DONE"
        assert len(h.portal.drafts_for(tid)) == 1


async def test_failure_after_post_does_not_double_post_nondeterministic_llm(harness):
    """BUG-2 regression. PLAN §5.3: 'a retried worker never double-posts'; a real LLM returns a different
    draft on retry, so the key must not depend on the new draft text."""
    async with harness() as h:
        (tid,) = h.add_tickets(1)
        n = itertools.count(1)

        async def varying(ticket, user):
            return make_result(make_analysis(draft_reply=f"Hello! Variant {next(n)}"))

        h.llm.analyze = varying
        h.portal.faults.on("set_status").fail(1, PortalError("500 after post"))
        assert (await h.run())[tid] == "DONE"
        assert len(h.portal.drafts_for(tid)) == 1, [d.body for d in h.portal.drafts_for(tid)]


# =========================================================================== crash recovery (Runtime, B9)


async def test_crash_mid_pipeline_then_restart_recovers(harness, settings):
    Runtime = require("support_agent.pipeline.runtime", "Runtime")
    async with harness() as h:
        ids = h.add_tickets(4)
        h.llm.faults.on("analyze").delay(5)  # workers get stuck in the LLM step
        await h.poller.run_once()
        await h.start_workers(2)
        await asyncio.sleep(0.3)
        await h.pool.stop()  # "crash": in-flight tickets are left mid-pipeline
        h.pool = None
        states = {await h.state(t) for t in ids}
        assert states & {"ENRICHED", "PROCESSING"} and "QUEUED" in states
        portal, users, llm = h.portal, h.users, h.llm
    llm.faults.clear()
    rt = Runtime(settings, portal=portal, user_client=users, llm=llm, policy=h.policy)
    await rt.start()
    try:
        async with harness() as h2:
            got = await h2.wait_settled(ids, within=15)
    finally:
        await rt.stop()
    assert set(got.values()) == {"DONE"}
    for t in ids:
        assert len(portal.drafts_for(t)) == 1


async def test_recovery_reenqueues_every_recoverable_state(harness, settings):
    Runtime = require("support_agent.pipeline.runtime", "Runtime")
    seq = {
        "QUEUED": [S.QUEUED],
        "PROCESSING": [S.QUEUED, S.PROCESSING],
        "ENRICHED": [S.QUEUED, S.PROCESSING, S.ENRICHED],
        "ANALYZED": [S.QUEUED, S.PROCESSING, S.ENRICHED, S.ANALYZED],
        "POLICY_CHECKED": [S.QUEUED, S.PROCESSING, S.ENRICHED, S.ANALYZED, S.POLICY_CHECKED],
        "FAILED": [S.QUEUED, S.PROCESSING, S.FAILED],
        "DISCOVERED": [],
    }
    async with harness() as h:
        ids = {}
        for state, steps in seq.items():
            (tid,) = h.add_tickets(1)
            ids[state] = tid
            summary = h.portal.tickets[tid]
            await h.repo.upsert_discovered(summary, trace_id=f"tr-{tid}")
            for s in steps:
                await h.repo.transition(tid, s)
        # a ticket waiting on a human must be left alone
        (rev,) = h.add_tickets(1)
        await h.repo.upsert_discovered(h.portal.tickets[rev], trace_id="tr-rev")
        for s in [S.QUEUED, S.PROCESSING, S.ENRICHED, S.ANALYZED, S.POLICY_CHECKED, S.NEEDS_REVIEW]:
            await h.repo.transition(rev, s)
        portal = h.portal
        rt = Runtime(settings, portal=portal, user_client=h.users, llm=h.llm, policy=h.policy)
        await rt.start()
        try:
            got = await h.wait_settled(ids.values(), within=15)
        finally:
            await rt.stop()
        assert set(got.values()) == {"DONE"}, got
        assert await h.state(rev) == "NEEDS_REVIEW"
        assert portal.drafts_for(rev) == []
