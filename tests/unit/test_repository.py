"""SqliteRepository (PLAN §5.5, B2): pragmas, state machine enforcement, atomic transitions log,
step records, idempotent drafts, review queries, extras, observability rows."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime

import pytest

from support_agent.common.contracts import Repository
from support_agent.common.models import PolicyDecision, TicketMessage, TicketSummary
from support_agent.common.states import InvalidTransition, TicketState
from tests.conftest import require
from tests.fakes import make_result, make_ticket, make_user

SqliteRepository = require("support_agent.storage.repository", "SqliteRepository")

S = TicketState
PIPELINE = [S.QUEUED, S.PROCESSING, S.ENRICHED, S.ANALYZED, S.POLICY_CHECKED]


@pytest.fixture
async def repo(db_path):
    r = SqliteRepository(db_path)
    await r.init()
    yield r
    await r.close()


def summary(tid: str = "T-1", **kw) -> TicketSummary:
    data = dict(
        ticket_id=tid,
        subject="Help",
        customer_email="alice@example.com",
        status="open",
        created_at=datetime(2026, 10, 1, tzinfo=UTC),
    )
    data.update(kw)
    return TicketSummary(**data)


async def discovered(repo, tid: str = "T-1") -> str:
    assert await repo.upsert_discovered(summary(tid), trace_id=f"tr-{tid}")
    return tid


async def advance(repo, tid: str, *states: TicketState) -> None:
    for s in states:
        await repo.transition(tid, s, worker_id="w1")


def sql(db_path: str, query: str, *params):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(query, params).fetchall()]
    finally:
        conn.close()


# --------------------------------------------------------------------------- lifecycle / pragmas


async def test_protocol_and_init_idempotent(repo):
    assert isinstance(repo, Repository)
    await repo.init()  # second init is a no-op
    assert await repo.ping()


async def test_pragmas(repo):
    async with repo.conn.execute("PRAGMA journal_mode") as cur:
        assert (await cur.fetchone())[0].lower() == "wal"
    async with repo.conn.execute("PRAGMA busy_timeout") as cur:
        assert (await cur.fetchone())[0] == 5000
    async with repo.conn.execute("PRAGMA foreign_keys") as cur:
        assert (await cur.fetchone())[0] == 1


async def test_schema_has_plan_tables(repo, db_path):
    names = {r["name"] for r in sql(db_path, "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {
        "tickets",
        "ticket_transitions",
        "ticket_snapshots",
        "user_lookups",
        "llm_calls",
        "policy_decisions",
        "drafts",
        "review_actions",
        "spans",
        "metric_samples",
        "scheduler_runs",
    } <= names


async def test_reinit_on_existing_db_keeps_data(db_path):
    r1 = SqliteRepository(db_path)
    await r1.init()
    await discovered(r1)
    await r1.close()
    r2 = SqliteRepository(db_path)
    await r2.init()
    try:
        assert (await r2.get_ticket_row("T-1"))["state"] == "DISCOVERED"
    finally:
        await r2.close()


async def test_uninitialised_raises(db_path):
    with pytest.raises(RuntimeError):
        _ = SqliteRepository(db_path).conn


async def test_close_twice_ok(db_path):
    r = SqliteRepository(db_path)
    await r.init()
    await r.close()
    await r.close()


# --------------------------------------------------------------------------- discovery


async def test_upsert_discovered_dedups(repo, db_path):
    assert await repo.upsert_discovered(summary(), "tr1") is True
    assert await repo.upsert_discovered(summary(subject="changed"), "tr2") is False
    row = await repo.get_ticket_row("T-1")
    assert row["state"] == "DISCOVERED"
    assert row["attempt"] == 0
    assert row["trace_id"] == "tr1"
    assert row["subject"] == "Help"
    assert row["discovered_at"] and row["updated_at"]
    log = await repo.transitions("T-1")
    assert [(t["from_state"], t["to_state"]) for t in log] == [(None, "DISCOVERED")]


async def test_upsert_does_not_reset_progressed_ticket(repo):
    await discovered(repo)
    await advance(repo, "T-1", S.QUEUED, S.PROCESSING)
    assert await repo.upsert_discovered(summary(), "tr-again") is False
    assert (await repo.get_ticket_row("T-1"))["state"] == "PROCESSING"


async def test_get_missing_row(repo):
    assert await repo.get_ticket_row("nope") is None


async def test_tickets_in_states(repo):
    for i in range(4):
        await discovered(repo, f"T-{i}")
    await advance(repo, "T-1", S.QUEUED)
    await advance(repo, "T-2", S.QUEUED, S.PROCESSING)
    assert [r["ticket_id"] for r in await repo.tickets_in_states({S.DISCOVERED})] == ["T-0", "T-3"]
    got = {r["ticket_id"] for r in await repo.tickets_in_states({S.QUEUED, S.PROCESSING})}
    assert got == {"T-1", "T-2"}
    assert await repo.tickets_in_states(set()) == []


# --------------------------------------------------------------------------- transitions


async def test_full_happy_chain_logged(repo):
    await discovered(repo)
    chain = [*PIPELINE, S.DRAFT_POSTED, S.DONE]
    await advance(repo, "T-1", *chain)
    log = await repo.transitions("T-1")
    assert [t["to_state"] for t in log] == ["DISCOVERED", *[s.value for s in chain]]
    # each from_state equals the previous to_state
    for prev, cur in zip(log, log[1:], strict=False):
        assert cur["from_state"] == prev["to_state"]
    assert all(t["worker_id"] == "w1" for t in log[1:])
    row = await repo.get_ticket_row("T-1")
    assert row["state"] == "DONE" and row["finished_at"]


async def test_illegal_transition_raises_and_writes_nothing(repo):
    await discovered(repo)
    before = await repo.transitions("T-1")
    with pytest.raises(InvalidTransition) as ei:
        await repo.transition("T-1", S.DONE)
    assert ei.value.from_state == S.DISCOVERED and ei.value.to_state == S.DONE
    assert (await repo.get_ticket_row("T-1"))["state"] == "DISCOVERED"
    assert await repo.transitions("T-1") == before


async def test_transition_missing_ticket(repo):
    with pytest.raises(InvalidTransition) as ei:
        await repo.transition("ghost", S.QUEUED)
    assert ei.value.from_state is None


async def test_transition_accepts_string_state(repo):
    await discovered(repo)
    await repo.transition("T-1", "QUEUED")
    assert (await repo.get_ticket_row("T-1"))["state"] == "QUEUED"


async def test_error_attempt_and_detail(repo):
    await discovered(repo)
    await advance(repo, "T-1", S.QUEUED, S.PROCESSING)
    await repo.transition(
        "T-1", S.FAILED, error="PortalError: 500", bump_attempt=True, detail={"step": "fetch"}
    )
    row = await repo.get_ticket_row("T-1")
    assert row["attempt"] == 1 and row["last_error"] == "PortalError: 500"
    last = (await repo.transitions("T-1"))[-1]
    assert last["detail"] == {"step": "fetch", "error": "PortalError: 500"}
    # a later transition without an error keeps last_error
    await repo.transition("T-1", S.QUEUED)
    assert (await repo.get_ticket_row("T-1"))["last_error"] == "PortalError: 500"
    assert (await repo.transitions("T-1"))[-1]["detail"] == {}


async def test_dead_letter_finished_then_requeue_clears(repo):
    await discovered(repo)
    await advance(repo, "T-1", S.QUEUED, S.FAILED, S.DEAD_LETTER)
    assert (await repo.get_ticket_row("T-1"))["finished_at"]
    await repo.transition("T-1", S.QUEUED)
    assert (await repo.get_ticket_row("T-1"))["finished_at"] is None


async def test_concurrent_transitions_only_one_wins(repo):
    await discovered(repo)
    await advance(repo, "T-1", S.QUEUED)
    results = await asyncio.gather(
        *(repo.transition("T-1", S.PROCESSING, worker_id=f"w{i}") for i in range(5)), return_exceptions=True
    )
    assert sum(r is None for r in results) == 1
    assert all(isinstance(r, InvalidTransition) for r in results if r is not None)
    assert [t["to_state"] for t in await repo.transitions("T-1")].count("PROCESSING") == 1


async def test_concurrent_upserts(repo, db_path):
    results = await asyncio.gather(
        *(repo.upsert_discovered(summary(f"T-{i % 50}"), "tr") for i in range(200))
    )
    assert sum(results) == 50
    assert sql(db_path, "SELECT COUNT(*) AS n FROM tickets")[0]["n"] == 50
    assert sql(db_path, "SELECT COUNT(*) AS n FROM ticket_transitions")[0]["n"] == 50


async def test_state_and_log_consistent_after_many_writes(repo, db_path):
    for i in range(30):
        await discovered(repo, f"T-{i}")

    async def run(tid):
        await advance(repo, tid, *PIPELINE, S.NEEDS_REVIEW)

    await asyncio.gather(*(run(f"T-{i}") for i in range(30)))
    rows = sql(
        db_path,
        """SELECT t.ticket_id, t.state, (SELECT to_state FROM ticket_transitions tt
           WHERE tt.ticket_id = t.ticket_id ORDER BY id DESC LIMIT 1) AS last FROM tickets t""",
    )
    assert all(r["state"] == r["last"] == "NEEDS_REVIEW" for r in rows)


async def test_counts_by_state(repo):
    for i in range(3):
        await discovered(repo, f"T-{i}")
    await advance(repo, "T-0", S.QUEUED)
    counts = await repo.counts_by_state()
    assert set(counts) == {s.value for s in S}
    assert counts["DISCOVERED"] == 2 and counts["QUEUED"] == 1 and counts["DONE"] == 0


# --------------------------------------------------------------------------- step records


async def test_save_snapshot_and_priority(repo, db_path):
    await discovered(repo)
    t = make_ticket("T-1", priority="high", body="v1")
    t.messages.append(TicketMessage(author="alice@example.com", body="hello", at=datetime.now(UTC)))
    await repo.save_snapshot(t)
    await repo.save_snapshot(t.model_copy(update={"body": "v2"}))
    snaps = sql(db_path, "SELECT * FROM ticket_snapshots")
    assert len(snaps) == 1 and snaps[0]["body"] == "v2"
    assert json.loads(snaps[0]["messages_json"])[0]["body"] == "hello"
    assert (await repo.get_ticket_row("T-1"))["priority"] == "high"


async def test_record_user_lookup(repo, db_path):
    await discovered(repo, "T-1")
    await discovered(repo, "T-2")
    await repo.record_user_lookup("T-1", make_user(), 12.5)
    await repo.record_user_lookup("T-2", None, 3.0)
    rows = {r["ticket_id"]: r for r in sql(db_path, "SELECT * FROM user_lookups")}
    assert rows["T-1"]["found"] == 1 and json.loads(rows["T-1"]["user_json"])["email"] == "alice@example.com"
    assert rows["T-2"]["found"] == 0 and rows["T-2"]["user_json"] is None
    assert rows["T-1"]["latency_ms"] == 12.5


async def test_record_llm_call_variants(repo, db_path):
    await discovered(repo)
    await repo.record_llm_call("T-1", make_result(input_tokens=11, output_tokens=7))
    await repo.record_llm_call("T-1", make_result(no_analysis=True, stop_reason="refusal"))
    await repo.record_llm_call("T-1", None, error="LLMError: overloaded")
    rows = sql(db_path, "SELECT * FROM llm_calls ORDER BY id")
    assert len(rows) == 3
    ok, refusal, err = rows
    assert ok["success"] == 1 and ok["input_tokens"] == 11 and ok["output_tokens"] == 7
    assert json.loads(ok["analysis_json"])["category"] == "account"
    assert refusal["stop_reason"] == "refusal" and refusal["analysis_json"] is None
    assert err["success"] == 0 and err["error"] == "LLMError: overloaded" and err["model"] is None


async def test_record_policy_decision_replaces(repo, db_path):
    await discovered(repo)
    await repo.record_policy_decision("T-1", PolicyDecision(outcome="auto_approve"))
    await repo.record_policy_decision(
        "T-1", PolicyDecision(outcome="human_review", reasons=["LEGAL", "ANGRY"], priority="high")
    )
    rows = sql(db_path, "SELECT * FROM policy_decisions")
    assert len(rows) == 1
    assert rows[0]["outcome"] == "human_review" and rows[0]["priority"] == "high"
    assert json.loads(rows[0]["reasons_json"]) == ["LEGAL", "ANGRY"]


async def test_records_require_existing_ticket(repo):
    """Foreign keys are on: step records for an unknown ticket fail loudly."""
    with pytest.raises(sqlite3.IntegrityError):
        await repo.record_user_lookup("ghost", None, 1.0)


# --------------------------------------------------------------------------- drafts & review queries


async def test_create_draft_idempotent(repo, db_path):
    await discovered(repo)
    a = await repo.create_draft("T-1", "hello", source="llm", status="posted", idempotency_key="T-1:abc")
    b = await repo.create_draft("T-1", "hello", source="llm", status="posted", idempotency_key="T-1:abc")
    assert a == b
    assert len(sql(db_path, "SELECT * FROM drafts")) == 1


async def test_mark_draft_posted_keeps_first_timestamp(repo):
    await discovered(repo)
    d = await repo.create_draft("T-1", "x", source="llm", status="pending_review", idempotency_key="k")
    await repo.mark_draft_posted(d)
    first = (await repo.latest_draft("T-1"))["posted_at"]
    await asyncio.sleep(0.01)
    await repo.mark_draft_posted(d)
    latest = await repo.latest_draft("T-1")
    assert latest["status"] == "posted" and latest["posted_at"] == first


async def test_latest_draft(repo):
    await discovered(repo)
    assert await repo.latest_draft("T-1") is None
    await repo.create_draft("T-1", "one", source="llm", status="pending_review", idempotency_key="k1")
    await repo.create_draft("T-1", "two", source="human_edit", status="approved", idempotency_key="k2")
    assert (await repo.latest_draft("T-1"))["body"] == "two"


async def test_approved_pending_post(repo):
    for tid in ("T-1", "T-2"):
        await discovered(repo, tid)
        await advance(repo, tid, *PIPELINE, S.NEEDS_REVIEW)
        await repo.create_draft(
            tid, f"{tid} llm", source="llm", status="pending_review", idempotency_key=f"{tid}:a"
        )
    await repo.create_draft(
        "T-1", "T-1 edited", source="human_edit", status="approved", idempotency_key="T-1:b"
    )
    await repo.transition("T-1", S.APPROVED)
    rows = await repo.approved_pending_post()
    assert len(rows) == 1
    r = rows[0]
    assert r["ticket_id"] == "T-1" and r["body"] == "T-1 edited" and r["source"] == "human_edit"
    assert r["idempotency_key"] == "T-1:b" and r["trace_id"] == "tr-T-1"


async def test_pending_reviews_priority_then_age(repo):
    for tid, prio in (("T-1", "normal"), ("T-2", "high"), ("T-3", "normal")):
        await discovered(repo, tid)
        await advance(repo, tid, *PIPELINE)
        await repo.record_policy_decision(
            tid, PolicyDecision(outcome="human_review", reasons=["ANGRY"], priority=prio)
        )
        await repo.create_draft(tid, "d", source="llm", status="pending_review", idempotency_key=f"{tid}:k")
        await repo.transition(tid, S.NEEDS_REVIEW)
        await asyncio.sleep(0.002)
    rows = await repo.pending_reviews()
    assert [r["ticket_id"] for r in rows] == ["T-2", "T-1", "T-3"]
    assert rows[0]["reasons"] == ["ANGRY"] and rows[0]["policy_priority"] == "high"
    assert rows[0]["draft_body"] == "d"


async def test_requeued_from_dead_letter_and_failed_review_posts(repo):
    for tid in ("T-1", "T-2", "T-3"):
        await discovered(repo, tid)
    await advance(repo, "T-1", S.QUEUED, S.FAILED, S.DEAD_LETTER, S.QUEUED)  # manual requeue
    await advance(repo, "T-2", S.QUEUED)  # normal queue
    await advance(repo, "T-3", *PIPELINE, S.NEEDS_REVIEW, S.APPROVED, S.FAILED)
    assert [r["ticket_id"] for r in await repo.requeued_from_dead_letter()] == ["T-1"]
    assert [r["ticket_id"] for r in await repo.failed_review_posts()] == ["T-3"]


# --------------------------------------------------------------------------- observability rows


async def test_record_span(repo, db_path):
    await repo.record_span(
        {
            "trace_id": "tr",
            "ticket_id": "T-1",
            "name": "llm.analyze",
            "start_ts": "2026-10-16T00:00:00+00:00",
            "duration_ms": 12.3,
            "status": "error",
            "attrs": {"model": "m"},
        }
    )
    await repo.record_span({"name": "minimal"})
    rows = sql(db_path, "SELECT * FROM spans ORDER BY id")
    assert rows[0]["status"] == "error" and json.loads(rows[0]["attrs_json"]) == {"model": "m"}
    assert rows[1]["status"] == "ok" and rows[1]["start_ts"]


async def test_span_sink_integration(repo, db_path):
    from support_agent.observability.tracing import set_span_sink, span

    set_span_sink(repo.record_span)
    async with span("poller.run", trace_id="t", found=3):
        pass
    rows = sql(db_path, "SELECT * FROM spans")
    assert rows[0]["name"] == "poller.run" and json.loads(rows[0]["attrs_json"]) == {"found": 3}


async def test_record_metric_samples(repo, db_path):
    await repo.record_metric_samples([])
    await repo.record_metric_samples([("queue_depth", {}, 3.0), ("llm_calls_total", {"model": "m"}, 2.0)])
    rows = sql(db_path, "SELECT * FROM metric_samples ORDER BY name")
    assert [(r["name"], json.loads(r["labels_json"]), r["value"]) for r in rows] == [
        ("llm_calls_total", {"model": "m"}, 2.0),
        ("queue_depth", {}, 3.0),
    ]
    assert rows[0]["ts"] == rows[1]["ts"]  # one batch, one timestamp


async def test_sampler_writes_through_repo(repo, db_path):
    from support_agent.observability.sampler import sample_once

    n = await sample_once(repo)
    assert sql(db_path, "SELECT COUNT(*) AS n FROM metric_samples")[0]["n"] == n > 0


async def test_scheduler_runs(repo):
    assert await repo.last_scheduler_run() is None
    rid = await repo.start_scheduler_run()
    row = await repo.last_scheduler_run()
    assert row["id"] == rid and row["finished_at"] is None
    await repo.finish_scheduler_run(rid, found=10, enqueued=7, skipped_full=3, error=None)
    row = await repo.last_scheduler_run()
    assert (row["found"], row["enqueued"], row["skipped_full"], row["error"]) == (10, 7, 3, None)
    assert row["finished_at"]
    rid2 = await repo.start_scheduler_run()
    await repo.finish_scheduler_run(rid2, found=0, enqueued=0, skipped_full=0, error="PortalError: down")
    assert (await repo.last_scheduler_run())["error"] == "PortalError: down"


async def test_reader_sees_writes_while_repo_open(repo, db_path):
    """WAL: the dashboard's separate sqlite3 connection reads while the runtime holds its connection."""
    await discovered(repo)
    assert sql(db_path, "SELECT state FROM tickets")[0]["state"] == "DISCOVERED"
    await advance(repo, "T-1", S.QUEUED)
    assert sql(db_path, "SELECT state FROM tickets")[0]["state"] == "QUEUED"
