"""storage.review (B2a): apply_review approve/edit/reject and requeue_dead_letter, all atomic."""

from __future__ import annotations

import sqlite3

import pytest

from support_agent.common.models import PolicyDecision
from support_agent.common.states import InvalidTransition, TicketState
from tests.conftest import require
from tests.unit.test_repository import PIPELINE, advance, discovered, sql

SqliteRepository = require("support_agent.storage.repository", "SqliteRepository")
review = require("support_agent.storage.review")
apply_review, requeue_dead_letter = review.apply_review, review.requeue_dead_letter
ReviewError, TicketNotFound = review.ReviewError, review.TicketNotFound

S = TicketState


@pytest.fixture
async def repo(db_path):
    r = SqliteRepository(db_path)
    await r.init()
    yield r
    await r.close()


async def in_review(repo, tid="T-1", draft: str | None = "LLM draft") -> str:
    await discovered(repo, tid)
    await advance(repo, tid, *PIPELINE)
    await repo.record_policy_decision(tid, PolicyDecision(outcome="human_review", reasons=["ANGRY"]))
    if draft is not None:
        await repo.create_draft(
            tid, draft, source="llm", status="pending_review", idempotency_key=f"{tid}:llm"
        )
    await repo.transition(tid, S.NEEDS_REVIEW)
    return tid


def review_rows(db_path):
    return sql(db_path, "SELECT * FROM review_actions ORDER BY id")


async def test_approve(repo, db_path):
    await in_review(repo)
    apply_review(db_path, "T-1", "approve", "sam")
    assert (await repo.get_ticket_row("T-1"))["state"] == "APPROVED"
    d = await repo.latest_draft("T-1")
    assert d["status"] == "approved" and d["reviewer"] == "sam" and d["source"] == "llm"
    rows = review_rows(db_path)
    assert [(r["action"], r["reviewer"], r["edited_body"]) for r in rows] == [("approve", "sam", None)]
    last = (await repo.transitions("T-1"))[-1]
    assert last["worker_id"] == "reviewer:sam" and last["detail"]["action"] == "approve"
    pending = await repo.approved_pending_post()
    assert pending[0]["body"] == "LLM draft"


async def test_edit_supersedes_and_gets_unique_key(repo, db_path):
    await in_review(repo)
    apply_review(db_path, "T-1", "edit", "sam", edited_body="Human reply")
    drafts = sql(db_path, "SELECT * FROM drafts ORDER BY id")
    assert [(d["source"], d["status"]) for d in drafts] == [("llm", "superseded"), ("human_edit", "approved")]
    assert drafts[1]["idempotency_key"] != drafts[0]["idempotency_key"]
    assert (await repo.get_ticket_row("T-1"))["state"] == "APPROVED"
    assert review_rows(db_path)[0]["edited_body"] == "Human reply"
    assert (await repo.approved_pending_post())[0]["body"] == "Human reply"


async def test_edit_without_llm_draft(repo, db_path):
    await in_review(repo, draft=None)  # e.g. LLM_NO_ANALYSIS: nothing to approve
    with pytest.raises(ReviewError):
        apply_review(db_path, "T-1", "approve", "sam")
    apply_review(db_path, "T-1", "edit", "sam", edited_body="Written from scratch")
    assert (await repo.latest_draft("T-1"))["body"] == "Written from scratch"


async def test_reject(repo, db_path):
    await in_review(repo)
    apply_review(db_path, "T-1", "reject", "sam")
    row = await repo.get_ticket_row("T-1")
    assert row["state"] == "DONE" and row["finished_at"]
    assert [t["to_state"] for t in await repo.transitions("T-1")][-2:] == ["REJECTED", "DONE"]
    assert (await repo.latest_draft("T-1"))["status"] == "rejected"
    assert await repo.approved_pending_post() == []


@pytest.mark.parametrize(
    "action,body",
    [("delete", None), ("edit", None), ("edit", "   "), ("APPROVE", None)],
)
async def test_bad_input_rejected_without_writes(repo, db_path, action, body):
    await in_review(repo)
    with pytest.raises(ReviewError):
        apply_review(db_path, "T-1", action, "sam", edited_body=body)
    assert review_rows(db_path) == []
    assert (await repo.get_ticket_row("T-1"))["state"] == "NEEDS_REVIEW"


async def test_unknown_ticket(repo, db_path):
    with pytest.raises(TicketNotFound):
        apply_review(db_path, "ghost", "approve", "sam")


async def test_wrong_state_is_atomic(repo, db_path):
    await discovered(repo)
    await advance(repo, "T-1", S.QUEUED)
    with pytest.raises(InvalidTransition):
        apply_review(db_path, "T-1", "approve", "sam")
    assert review_rows(db_path) == []  # review_actions row rolled back


async def test_double_review_rejected(repo, db_path):
    await in_review(repo)
    apply_review(db_path, "T-1", "approve", "sam")
    with pytest.raises(InvalidTransition):
        apply_review(db_path, "T-1", "reject", "kim")
    assert len(review_rows(db_path)) == 1


async def test_requeue_dead_letter(repo, db_path):
    await discovered(repo)
    for _ in range(3):
        await advance(repo, "T-1", S.QUEUED)
        await repo.transition("T-1", S.FAILED, bump_attempt=True, error="boom")
    await repo.transition("T-1", S.DEAD_LETTER)
    assert (await repo.get_ticket_row("T-1"))["attempt"] == 3
    requeue_dead_letter(db_path, "T-1")
    row = await repo.get_ticket_row("T-1")
    assert row["state"] == "QUEUED" and row["attempt"] == 0 and row["finished_at"] is None
    assert [r["ticket_id"] for r in await repo.requeued_from_dead_letter()] == ["T-1"]


async def test_requeue_wrong_state(repo, db_path):
    await discovered(repo)
    with pytest.raises(InvalidTransition):
        requeue_dead_letter(db_path, "T-1")
    with pytest.raises(TicketNotFound):
        requeue_dead_letter(db_path, "ghost")


async def test_review_from_other_connection_while_runtime_writes(repo, db_path):
    """Dashboard (sqlite3) and runtime (aiosqlite) share the DB; busy_timeout absorbs contention."""
    for i in range(10):
        await in_review(repo, f"T-{i}")
    for i in range(10):
        apply_review(db_path, f"T-{i}", "approve" if i % 2 else "reject", "sam")
        await repo.record_span({"name": f"s{i}"})
    counts = await repo.counts_by_state()
    assert counts["APPROVED"] == 5 and counts["DONE"] == 5


def test_review_on_missing_db_file(tmp_path):
    with pytest.raises((sqlite3.Error, ReviewError)):
        apply_review(str(tmp_path / "missing" / "x.db"), "T-1", "approve", "sam")
