"""Human review end to end (PLAN §5.12 item 3, B2a + B8): worker -> NEEDS_REVIEW -> apply_review
(approve / edit / reject) -> ReviewDispatcher posts via the portal -> DONE. Plus post failures, retry,
dead-lettering, and DLQ requeue -> worker -> DONE."""

from __future__ import annotations

import asyncio

import pytest

from support_agent.common.contracts import PortalError
from support_agent.observability import metrics as m
from tests.conftest import require

review = require("support_agent.storage.review")
ReviewDispatcher = require("support_agent.pipeline.review_dispatcher", "ReviewDispatcher")


def metric(name: str, **labels) -> float:
    return m.REGISTRY.get_sample_value(name, labels) or 0.0


def dispatcher(h, **kw) -> ReviewDispatcher:
    kw.setdefault("backoff_base_s", 0.01)
    kw.setdefault("backoff_max_s", 0.02)
    return ReviewDispatcher(h.repo, h.portal, 0.05, queue=h.queue, **kw)


async def to_review(h, n: int = 1, *reasons: str) -> list[str]:
    ids = h.add_tickets(n)
    for tid in ids:
        h.policy.review(tid, *(reasons or ("ANGRY",)))
    got = await h.run()
    assert set(got.values()) == {"NEEDS_REVIEW"}
    return ids


async def test_approve_posts_llm_draft_once(harness, settings):
    async with harness() as h:
        (tid,) = await to_review(h)
        llm_draft = (await h.repo.latest_draft(tid))["body"]
        review.apply_review(settings.db_path, tid, "approve", "sam")
        posted0 = metric("drafts_posted_total", source="llm")
        appr0 = metric("reviews_total", action="approve")
        d = dispatcher(h)
        res = await d.run_once()
        assert res["posted"] == 1
        assert await h.state(tid) == "DONE"
        assert [x.body for x in h.portal.drafts_for(tid)] == [llm_draft]
        assert h.portal.statuses[tid] == ["escalated", "pending"]
        row = await h.repo.latest_draft(tid)
        assert row["status"] == "posted" and row["posted_at"]
        assert h.portal.drafts_for(tid)[0].idempotency_key == row["idempotency_key"]
        assert metric("drafts_posted_total", source="llm") - posted0 == 1
        assert metric("reviews_total", action="approve") - appr0 == 1
        assert (await h.path(tid))[-4:] == ["NEEDS_REVIEW", "APPROVED", "DRAFT_POSTED", "DONE"]
        assert (await d.run_once())["posted"] == 0  # second pass is a no-op
        assert len(h.portal.drafts_for(tid)) == 1


async def test_edit_posts_human_text(harness, settings):
    async with harness() as h:
        (tid,) = await to_review(h, 1, "LEGAL")
        review.apply_review(settings.db_path, tid, "edit", "kim", edited_body="Hand-written reply")
        edit0 = metric("reviews_total", action="edit")
        await dispatcher(h).run_once()
        assert await h.state(tid) == "DONE"
        assert [x.body for x in h.portal.drafts_for(tid)] == ["Hand-written reply"]
        assert metric("reviews_total", action="edit") - edit0 == 1
        drafts = h.sql("SELECT source, status FROM drafts WHERE ticket_id = ? ORDER BY id", tid)
        assert drafts == [
            {"source": "llm", "status": "superseded"},
            {"source": "human_edit", "status": "posted"},
        ]


async def test_reject_posts_nothing(harness, settings):
    async with harness() as h:
        (tid,) = await to_review(h)
        review.apply_review(settings.db_path, tid, "reject", "sam")
        res = await dispatcher(h).run_once()
        assert res["posted"] == 0
        assert await h.state(tid) == "DONE"
        assert h.portal.drafts_for(tid) == []
        assert len(h.sql("SELECT * FROM review_actions WHERE ticket_id = ?", tid)) == 1


async def test_post_failure_retried_then_posted(harness, settings):
    async with harness() as h:
        (tid,) = await to_review(h)
        review.apply_review(settings.db_path, tid, "approve", "sam")
        h.portal.faults.on("post_draft").fail(1, PortalError("500"))
        d = dispatcher(h)
        res = await d.run_once()
        assert res["failed"] == 1 and await h.state(tid) == "FAILED"
        for _ in range(20):
            await asyncio.sleep(0.03)
            await d.run_once()
            if await h.state(tid) == "DONE":
                break
        assert await h.state(tid) == "DONE"
        assert len(h.portal.drafts_for(tid)) == 1
        path = await h.path(tid)
        assert path[-6:] == ["NEEDS_REVIEW", "APPROVED", "FAILED", "APPROVED", "DRAFT_POSTED", "DONE"]
        assert (await h.repo.get_ticket_row(tid))["attempt"] == 1


async def test_persistent_post_failure_dead_letters(harness, settings):
    async with harness() as h:
        (tid,) = await to_review(h)
        review.apply_review(settings.db_path, tid, "approve", "sam")
        h.portal.faults.on("post_draft").fail(-1, PortalError("500"))
        d = dispatcher(h, max_retries=2)
        for _ in range(40):
            await d.run_once()
            if await h.state(tid) == "DEAD_LETTER":
                break
            await asyncio.sleep(0.03)
        assert await h.state(tid) == "DEAD_LETTER"
        assert h.portal.drafts_for(tid) == []


async def test_failure_after_post_finishes_without_double_post(harness, settings):
    async with harness() as h:
        (tid,) = await to_review(h)
        review.apply_review(settings.db_path, tid, "approve", "sam")
        h.portal.faults.on("set_status").fail(1, PortalError("500 after post"))
        d = dispatcher(h)
        for _ in range(20):
            await d.run_once()
            if await h.state(tid) == "DONE":
                break
            await asyncio.sleep(0.03)
        assert await h.state(tid) == "DONE"
        assert len(h.portal.drafts_for(tid)) == 1  # stored key is reused: the portal skips the repost


async def test_dispatcher_loop_runs_in_background(harness, settings):
    async with harness() as h:
        ids = await to_review(h, 3)
        d = dispatcher(h)
        await d.start()
        try:
            for tid in ids:
                review.apply_review(settings.db_path, tid, "approve", "sam")
            got = await h.wait_settled(ids, within=5)
        finally:
            await d.stop()
        assert set(got.values()) == {"DONE"}
        assert not d.running


async def test_dead_letter_requeue_flows_back_through_workers(harness, settings):
    async with harness() as h:
        (tid,) = h.add_tickets(1)
        h.llm.faults.on("analyze").fail(-1, PortalError("LLM down"))
        assert (await h.run())[tid] == "DEAD_LETTER"
        h.llm.faults.clear()
        review.requeue_dead_letter(settings.db_path, tid)
        assert (await h.repo.get_ticket_row(tid))["attempt"] == 0
        res = await dispatcher(h).run_once()
        assert res["requeued"] == 1
        got = await h.wait_settled([tid], within=5)
        assert got[tid] == "DONE"
        assert len(h.portal.drafts_for(tid)) == 1


@pytest.mark.parametrize("action", ["approve", "edit", "reject"])
async def test_review_writes_review_actions(harness, settings, action):
    async with harness() as h:
        (tid,) = await to_review(h)
        review.apply_review(
            settings.db_path, tid, action, "sam", edited_body="x" if action == "edit" else None
        )
        rows = h.sql("SELECT action, reviewer FROM review_actions WHERE ticket_id = ?", tid)
        assert rows == [{"action": action, "reviewer": "sam"}]
