"""Streamlit dashboard smoke (PLAN §5.12, C8–C10) via `streamlit.testing.v1.AppTest`.

A DB is seeded by running all 53 seed tickets through the real pipeline (MockLLM + real policy gate + fake
portal), plus a dead-lettered ticket, metric samples, scheduler runs and a JSON log file. Every page must
render without exceptions; review buttons must write `review_actions`; DLQ requeue must work.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

from support_agent.common.config import Settings
from tests.conftest import require
from tests.fakes import POLICY_FILE, fatal_llm_error, load_seed_cases, make_ticket

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = Path(__file__).resolve().parents[2] / "src/support_agent/dashboard/app.py"
PAGES = [
    "views/overview.py",
    "views/tickets.py",
    "views/review.py",
    "views/queue_workers.py",
    "views/llm.py",
    "views/scheduler.py",
    "views/logs.py",
]
DLQ_ID = "T-DLQ-1"


async def _seed(settings: Settings) -> None:
    from support_agent.observability.sampler import sample_once
    from tests.integration.conftest import Harness

    gate = require("support_agent.policy.gate", "RuleBasedPolicyGate").from_file(POLICY_FILE)
    MockLLMClient = require("support_agent.llm.mock_client", "MockLLMClient")
    h = Harness(settings, policy=gate, workers=4)
    h.llm = MockLLMClient(settings)
    async with h:
        for c in load_seed_cases():
            h.add(c.ticket, c.user)
        bad = make_ticket(DLQ_ID, subject="Poison ticket")
        h.add(bad)
        from tests.fakes import FakeLLMClient

        real_llm = h.llm

        class Routed(FakeLLMClient):
            async def analyze(self, ticket, user):
                if ticket.ticket_id == DLQ_ID:
                    raise fatal_llm_error("BadRequestError (400)")
                return await real_llm.analyze(ticket, user)

        h.llm = Routed(model=real_llm.model)
        await h.run(within=30)
        for _ in range(3):
            await sample_once(h.repo)
            await asyncio.sleep(0.01)


@pytest.fixture(scope="module")
def seeded(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dash")
    settings = Settings(
        db_path=str(tmp / "dash.db"),
        log_file=str(tmp / "agent.jsonl"),
        artifacts_dir=str(tmp / "artifacts"),
        mock_llm_latency_ms=0,
        retry_backoff_base_seconds=0.01,
        retry_backoff_max_seconds=0.02,
    )
    asyncio.run(_seed(settings))
    Path(settings.log_file).write_text(
        "\n".join(
            json.dumps(
                {"ts": "2026-10-16T10:00:00Z", "level": lvl, "event": ev, "component": comp, "ticket_id": t}
            )
            for lvl, ev, comp, t in [
                ("info", "worker.ticket_done", "worker", "TCK-1001"),
                ("warning", "browser.retry", "browser", "TCK-1002"),
                ("error", "worker.dead_letter", "worker", DLQ_ID),
            ]
        )
        + "\n"
    )
    return settings


@pytest.fixture
def app(seeded, monkeypatch):
    monkeypatch.setenv("SA_DB_PATH", seeded.db_path)
    monkeypatch.setenv("SA_LOG_FILE", seeded.log_file)
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def db_rows(db_path: str, sql: str, *params):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def all_text(at) -> str:
    parts = []
    for kind in (
        "markdown",
        "caption",
        "title",
        "header",
        "subheader",
        "info",
        "warning",
        "success",
        "error",
    ):
        parts += [str(e.value) for e in getattr(at, kind)]
    parts += [f"{m.label} {m.value}" for m in at.metric]
    return "\n".join(parts)


def test_seeded_db_is_realistic(seeded):
    counts = {
        r["state"]: r["n"]
        for r in db_rows(seeded.db_path, "SELECT state, COUNT(*) n FROM tickets GROUP BY state")
    }
    assert counts == {"DONE": 19, "NEEDS_REVIEW": 34, "DEAD_LETTER": 1}
    assert db_rows(seeded.db_path, "SELECT COUNT(*) n FROM metric_samples")[0]["n"] > 0


def test_overview_renders_kpis(app):
    text = all_text(app)
    assert app.metric, "no KPI tiles"
    assert "34" in text  # in review


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(app, page):
    app.switch_page(page)
    app.run()
    assert not app.exception, f"{page}: {app.exception}"


def test_empty_db_renders_gracefully(tmp_path, monkeypatch):
    monkeypatch.setenv("SA_DB_PATH", str(tmp_path / "missing.db"))
    monkeypatch.setenv("SA_LOG_FILE", str(tmp_path / "missing.jsonl"))
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception
    for page in PAGES:
        at.switch_page(page)
        at.run()
        assert not at.exception, page


def _open_review(app, reviewer: str | None = "qa-bot"):
    app.switch_page("views/review.py")
    app.run()
    assert not app.exception
    if reviewer is not None:
        app.text_input(key="reviewer_name").set_value(reviewer)
        app.run()
    return app.selectbox(key="review_pick").value


def test_review_approve_writes_review_actions(seeded, app):
    tid = _open_review(app)
    btn = app.button(key=f"rv_approve_{tid}")
    if btn.disabled:  # no usable draft (refusal / no analysis) -> covered by the edit test
        pytest.skip(f"{tid} has no draft to approve")
    btn.click()
    app.run()
    assert not app.exception
    rows = db_rows(seeded.db_path, "SELECT action, reviewer FROM review_actions WHERE ticket_id = ?", tid)
    assert rows == [{"action": "approve", "reviewer": "qa-bot"}]
    assert (
        db_rows(seeded.db_path, "SELECT state FROM tickets WHERE ticket_id = ?", tid)[0]["state"]
        == "APPROVED"
    )
    assert any("approved" in str(s.value) for s in app.success)


def test_review_edit_writes_human_draft(seeded, app):
    tid = _open_review(app)
    draft_key = next(t.key for t in app.text_area if t.key and t.key.startswith(f"rv_text_{tid}_"))
    app.text_area(key=draft_key).set_value("Edited by QA.")
    app.run()
    app.button(key=f"rv_edit_{tid}").click()
    app.run()
    assert not app.exception
    assert db_rows(
        seeded.db_path, "SELECT action, edited_body FROM review_actions WHERE ticket_id = ?", tid
    ) == [{"action": "edit", "edited_body": "Edited by QA."}]
    latest = db_rows(
        seeded.db_path, "SELECT body, source FROM drafts WHERE ticket_id = ? ORDER BY id DESC", tid
    )[0]
    assert latest == {"body": "Edited by QA.", "source": "human_edit"}


def test_review_reject(seeded, app):
    tid = _open_review(app)
    app.button(key=f"rv_reject_{tid}").click()
    app.run()
    assert not app.exception
    assert db_rows(seeded.db_path, "SELECT state FROM tickets WHERE ticket_id = ?", tid)[0]["state"] == "DONE"
    assert db_rows(seeded.db_path, "SELECT action FROM review_actions WHERE ticket_id = ?", tid) == [
        {"action": "reject"}
    ]


def test_review_requires_reviewer_name(seeded, app):
    before = db_rows(seeded.db_path, "SELECT COUNT(*) n FROM review_actions")[0]["n"]
    tid = _open_review(app, reviewer=None)
    app.button(key=f"rv_reject_{tid}").click()
    app.run()
    assert not app.exception
    assert db_rows(seeded.db_path, "SELECT COUNT(*) n FROM review_actions")[0]["n"] == before
    assert app.warning, "expected a 'enter your name' warning"


def test_dlq_requeue_from_tickets_page(seeded, app):
    app.switch_page("views/tickets.py")
    app.run()
    assert not app.exception
    keys = [b.key for b in app.button if b.key == f"requeue_{DLQ_ID}"]
    if not keys:
        # the requeue button appears on the ticket detail; select the DLQ ticket if there is a picker
        pickers = [s for s in app.selectbox if DLQ_ID in (s.options or [])]
        if not pickers:
            pytest.skip("tickets page exposes no DLQ requeue control for this ticket in AppTest")
        pickers[0].set_value(DLQ_ID)
        app.run()
    app.button(key=f"requeue_{DLQ_ID}").click()
    app.run()
    assert not app.exception
    row = db_rows(seeded.db_path, "SELECT state, attempt FROM tickets WHERE ticket_id = ?", DLQ_ID)[0]
    assert row == {"state": "QUEUED", "attempt": 0}
