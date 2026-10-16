"""SupportAgent console (Phase 5, U2–U5): one Streamlit page with tabs Observability · Tickets · User Info · SQL,
driven with `streamlit.testing.v1.AppTest`.

A module-scoped DB is seeded by the real pipeline (53 seed tickets + 150 generated, MockLLM + real policy gate
+ fake portal) plus one dead-lettered ticket: 204 tickets, so lists exceed the 100-row default. Tabs are picked
through session state (`tab`, `obs_section`, `tickets_section`) before `run()`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from support_agent.common.config import Settings, get_settings
from tests.conftest import require
from tests.fakes import (
    POLICY_FILE,
    FakeLLMClient,
    fatal_llm_error,
    load_seed_cases,
    make_ticket,
    seed_case_from_raw,
)

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parents[2] / "src/support_agent/dashboard/app.py")
DLQ_ID = "T-DLQ-1"
N_TICKETS = 53 + 150 + 1
DOWN = "http://127.0.0.1:1"  # nothing listens here

VIEWS = [
    *[
        ("Observability", {"obs_section": s})
        for s in ("Overview", "Queue & Workers", "LLM", "Scheduler & Poller", "Logs")
    ],
    ("Tickets", {"tickets_section": "All tickets"}),
    ("Tickets", {"tickets_section": "Review queue"}),
    ("User Info", {}),
    ("SQL", {}),
]
VIEW_IDS = [f"{t}:{next(iter(s.values()), '-')}" for t, s in VIEWS]


# --------------------------------------------------------------------------- fixtures


async def _seed(settings: Settings) -> None:
    from support_agent.observability.sampler import sample_once
    from tests.integration.conftest import Harness

    gate = require("support_agent.policy.gate", "RuleBasedPolicyGate").from_file(POLICY_FILE)
    MockLLMClient = require("support_agent.llm.mock_client", "MockLLMClient")
    generator = require("support_agent.mocks.ticket_ui.generator")
    real_llm = MockLLMClient(settings)

    class Routed(FakeLLMClient):
        async def analyze(self, ticket, user):
            if ticket.ticket_id == DLQ_ID:
                raise fatal_llm_error("BadRequestError (400)")
            return await real_llm.analyze(ticket, user)

    h = Harness(settings, policy=gate, workers=4)
    h.llm = Routed(model=real_llm.model)
    async with h:
        cases = load_seed_cases() + [seed_case_from_raw(r) for r in generator.generate(150, 3)]
        for c in cases:
            h.add(c.ticket, c.user)
        h.add(make_ticket(DLQ_ID, subject="Poison ticket"))
        await h.run(within=60)
        for _ in range(3):
            await sample_once(h.repo)
            await asyncio.sleep(0.01)


@pytest.fixture(scope="module")
def seeded(tmp_path_factory) -> Settings:
    tmp = tmp_path_factory.mktemp("console")
    s = Settings(
        db_path=str(tmp / "console.db"),
        log_file=str(tmp / "agent.jsonl"),
        artifacts_dir=str(tmp / "artifacts"),
        mock_llm_latency_ms=0,
        retry_backoff_base_seconds=0.01,
        retry_backoff_max_seconds=0.02,
    )
    asyncio.run(_seed(s))
    lines = [
        {"ts": "2026-10-16T10:00:00Z", "level": lvl, "event": ev, "component": comp, "ticket_id": t}
        for lvl, ev, comp, t in [
            ("info", "worker.ticket_done", "worker", "TCK-1001"),
            ("warning", "browser.retry", "browser", "TCK-1002"),
            ("error", "worker.dead_letter", "worker", DLQ_ID),
        ]
    ]
    Path(s.log_file).write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return s


@pytest.fixture
def empty_db(tmp_path) -> str:
    """Schema applied, no rows (a runtime that started but found nothing yet)."""
    SqliteRepository = require("support_agent.storage.repository", "SqliteRepository")
    path = str(tmp_path / "empty.db")

    async def init():
        r = SqliteRepository(path)
        await r.init()
        await r.close()

    asyncio.run(init())
    return path


@pytest.fixture
def console(monkeypatch, tmp_path):
    """`console(db_path, log=None, users_url=DOWN, tab=..., **session_state)` -> AppTest after the first run."""

    def make(db_path: str, *, log: str | None = None, users_url: str = DOWN, **state) -> AppTest:
        monkeypatch.setenv("SA_DB_PATH", db_path)
        monkeypatch.setenv("SA_LOG_FILE", log or str(tmp_path / "missing.jsonl"))
        monkeypatch.setenv("SA_USER_SERVICE_URL", users_url)
        get_settings.cache_clear()  # app.py reads the cached Settings; AppTest runs in this process
        at = AppTest.from_file(APP, default_timeout=60)
        for k, v in state.items():
            at.session_state[k] = v
        at.run()
        assert not at.exception, at.exception
        return at

    yield make
    get_settings.cache_clear()


def rows(db_path: str, sql: str, *params):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def digest(path: str) -> str:
    h = hashlib.sha256()
    for p in (Path(path), Path(path + "-wal")):
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()


def captions(at) -> list[str]:
    return [str(c.value) for c in at.caption]


def caption_with(at, text: str) -> str | None:
    return next((c for c in captions(at) if text in c), None)


def button(at, key: str):
    return next((b for b in at.button if b.key == key), None)


def frame_lengths(at) -> list[int]:
    return [len(d.value) for d in at.dataframe]


# --------------------------------------------------------------------------- rendering matrix


def test_seeded_db_shape(seeded):
    counts = {
        r["state"]: r["n"]
        for r in rows(seeded.db_path, "SELECT state, COUNT(*) n FROM tickets GROUP BY state")
    }
    assert sum(counts.values()) == N_TICKETS
    assert counts["DEAD_LETTER"] == 1 and counts["NEEDS_REVIEW"] > 0 and counts["DONE"] > 0


@pytest.mark.parametrize("tab,state", VIEWS, ids=VIEW_IDS)
def test_renders_on_seeded_db(console, seeded, tab, state):
    at = console(seeded.db_path, log=seeded.log_file, tab=tab, **state)
    errors = [str(e.value) for e in at.error]
    assert not errors, errors
    if tab == "User Info":  # service is down in this matrix: browse falls back to the DB, quietly
        assert caption_with(at, "Showing customer emails from tickets"), captions(at)


@pytest.mark.parametrize("tab,state", VIEWS, ids=VIEW_IDS)
def test_renders_on_empty_db(console, empty_db, tab, state):
    console(empty_db, tab=tab, **state)


@pytest.mark.parametrize("tab,state", VIEWS, ids=VIEW_IDS)
def test_renders_on_missing_db(console, tmp_path, tab, state):
    missing = str(tmp_path / "nope" / "missing.db")
    at = console(missing, tab=tab, **state)
    if tab != "User Info" and state.get("obs_section") != "Logs":  # logs don't need a DB
        assert any("No database" in str(i.value) for i in at.info), [str(i.value) for i in at.info]
    assert not Path(missing).exists()  # the console never creates the DB


def test_default_tab_is_observability(console, seeded):
    at = console(seeded.db_path, log=seeded.log_file)
    assert at.session_state["tab"] == "Observability"
    assert at.metric, "overview KPI tiles"


def test_header_counts(console, seeded):
    at = console(seeded.db_path)
    head = caption_with(at, "tickets ·")
    assert head and f"{N_TICKETS} tickets" in head and "dead letter" in head


def test_logs_section_shows_log_lines(console, seeded):
    at = console(seeded.db_path, log=seeded.log_file, tab="Observability", obs_section="Logs")
    blob = json.dumps([d.value.to_dict() for d in at.dataframe], default=str) + " ".join(captions(at))
    assert "worker.dead_letter" in blob or "browser.retry" in blob


def test_time_range_control(console, seeded):
    at = console(seeded.db_path, tab="Observability", obs_section="LLM")
    rng = at.segmented_control(key="obs_range")
    for opt in rng.options:
        rng.set_value(opt)
        at.run()
        assert not at.exception, opt


# --------------------------------------------------------------------------- Tickets tab


def test_tickets_100_by_default_and_load_more(console, seeded):
    at = console(seeded.db_path, tab="Tickets", tickets_section="All tickets")
    assert caption_with(at, f"Showing 100 of {N_TICKETS}") and caption_with(at, "more available")
    assert 100 in frame_lengths(at)
    button(at, "tickets_more").click()
    at.run()
    assert caption_with(at, f"Showing 200 of {N_TICKETS}")
    button(at, "tickets_more").click()
    at.run()
    assert caption_with(at, f"Showing {N_TICKETS} of {N_TICKETS}")
    assert not caption_with(at, "more available") and button(at, "tickets_more") is None


def test_tickets_page_size(console, seeded):
    at = console(seeded.db_path, tab="Tickets", tickets_section="All tickets")
    at.selectbox(key="tickets_page_size").set_value(500)
    at.run()
    assert caption_with(at, f"Showing {N_TICKETS} of {N_TICKETS}")


def test_tickets_filters(console, seeded):
    at = console(seeded.db_path, tab="Tickets", tickets_section="All tickets")
    n_review = rows(seeded.db_path, "SELECT COUNT(*) n FROM tickets WHERE state = 'NEEDS_REVIEW'")[0]["n"]
    at.multiselect(key="tickets_states").set_value(["NEEDS_REVIEW"])
    at.run()
    assert caption_with(at, f"of {n_review} ticket(s)"), captions(at)
    at.multiselect(key="tickets_states").set_value([])
    at.text_input(key="tickets_search").set_value("TCK-1001")
    at.run()
    assert caption_with(at, "Showing 1 of 1"), captions(at)


def test_ticket_detail_and_dlq_requeue(console, seeded):
    at = console(seeded.db_path, tab="Tickets", tickets_section="All tickets", tickets_pick=DLQ_ID)
    btn = button(at, f"requeue_{DLQ_ID}")
    assert btn is not None, [b.key for b in at.button]
    btn.click()
    at.run()
    assert not at.exception
    assert rows(seeded.db_path, "SELECT state, attempt FROM tickets WHERE ticket_id = ?", DLQ_ID)[0] == {
        "state": "QUEUED",
        "attempt": 0,
    }


def _review(console, seeded, reviewer: str | None = "qa-bot"):
    at = console(seeded.db_path, tab="Tickets", tickets_section="Review queue")
    if reviewer is not None:
        at.text_input(key="reviewer_name").set_value(reviewer)
        at.run()
    return at, at.selectbox(key="review_pick").value


def test_review_approve(console, seeded):
    at, tid = _review(console, seeded)
    btn = button(at, f"rv_approve_{tid}")
    if btn.disabled:
        at.selectbox(key="review_pick").set_value(
            next(o for o in at.selectbox(key="review_pick").options if o != tid)
        )
        at.run()
        tid = at.selectbox(key="review_pick").value
        btn = button(at, f"rv_approve_{tid}")
    btn.click()
    at.run()
    assert not at.exception
    assert rows(seeded.db_path, "SELECT action, reviewer FROM review_actions WHERE ticket_id = ?", tid) == [
        {"action": "approve", "reviewer": "qa-bot"}
    ]
    assert (
        rows(seeded.db_path, "SELECT state FROM tickets WHERE ticket_id = ?", tid)[0]["state"] == "APPROVED"
    )
    assert any("approved" in str(s.value) for s in at.success)


def test_review_edit(console, seeded):
    at, tid = _review(console, seeded)
    key = next(t.key for t in at.text_area if t.key and t.key.startswith(f"rv_text_{tid}_"))
    at.text_area(key=key).set_value("Edited by QA.")
    at.run()
    button(at, f"rv_edit_{tid}").click()
    at.run()
    assert not at.exception
    assert rows(
        seeded.db_path, "SELECT action, edited_body FROM review_actions WHERE ticket_id = ?", tid
    ) == [{"action": "edit", "edited_body": "Edited by QA."}]
    latest = rows(
        seeded.db_path, "SELECT body, source, status FROM drafts WHERE ticket_id = ? ORDER BY id DESC", tid
    )
    assert latest[0] == {"body": "Edited by QA.", "source": "human_edit", "status": "approved"}


def test_review_reject(console, seeded):
    at, tid = _review(console, seeded)
    button(at, f"rv_reject_{tid}").click()
    at.run()
    assert not at.exception
    assert rows(seeded.db_path, "SELECT state FROM tickets WHERE ticket_id = ?", tid)[0]["state"] == "DONE"
    assert rows(seeded.db_path, "SELECT action FROM review_actions WHERE ticket_id = ?", tid) == [
        {"action": "reject"}
    ]


def test_review_requires_reviewer(console, seeded):
    before = rows(seeded.db_path, "SELECT COUNT(*) n FROM review_actions")[0]["n"]
    at, tid = _review(console, seeded, reviewer=None)
    button(at, f"rv_reject_{tid}").click()
    at.run()
    assert rows(seeded.db_path, "SELECT COUNT(*) n FROM review_actions")[0]["n"] == before
    assert at.warning


# --------------------------------------------------------------------------- SQL tab


def _sql(at, text: str) -> None:
    at.text_area(key="sql_text").set_value(text)
    at.run()
    button(at, "sql_run").click()
    at.run()
    assert not at.exception


def test_sql_100_rows_by_default_and_load_next(console, seeded):
    at = console(seeded.db_path, tab="SQL")
    _sql(at, "SELECT ticket_id, state FROM tickets ORDER BY ticket_id")
    res = at.session_state["sql_result"]
    assert len(res.rows) == 100 and res.has_more
    assert caption_with(at, "more available")
    button(at, "sql_more").click()
    at.run()
    res = at.session_state["sql_result"]
    assert len(res.rows) == 200 and res.has_more
    button(at, "sql_more").click()
    at.run()
    res = at.session_state["sql_result"]
    assert len(res.rows) == N_TICKETS and not res.has_more
    assert button(at, "sql_more") is None
    assert not caption_with(at, "more available")


def test_sql_default_example_runs(console, seeded):
    at = console(seeded.db_path, tab="SQL")
    assert at.text_area(key="sql_text").value.strip()
    button(at, "sql_run").click()
    at.run()
    assert at.session_state["sql_result"] is not None and at.session_state["sql_error"] is None


def test_sql_examples_all_run(console, seeded):
    sql_view = require("support_agent.dashboard.views.sql")
    at = console(seeded.db_path, tab="SQL")
    for name, text in sql_view.EXAMPLES.items():
        at.selectbox(key="sql_example").set_value(name)
        at.run()
        assert at.text_area(key="sql_text").value == text, name
        button(at, "sql_run").click()
        at.run()
        assert at.session_state["sql_error"] is None, (name, at.session_state["sql_error"])


def test_sql_page_size(console, seeded):
    at = console(seeded.db_path, tab="SQL")
    at.selectbox(key="sql_page_size").set_value(500)
    _sql(at, "SELECT * FROM tickets")
    assert len(at.session_state["sql_result"].rows) == N_TICKETS


@pytest.mark.parametrize(
    "stmt",
    [
        "DELETE FROM tickets",
        "UPDATE tickets SET state = 'DONE'",
        "DROP TABLE tickets",
        "WITH x AS (SELECT 1) DELETE FROM tickets",
        "SELECT 1; DELETE FROM tickets",
        "PRAGMA query_only=OFF",
        "ATTACH DATABASE 'evil.db' AS evil",
    ],
)
def test_sql_writes_rejected_db_unchanged(console, seeded, stmt):
    at = console(seeded.db_path, tab="SQL")
    before, n = digest(seeded.db_path), rows(seeded.db_path, "SELECT COUNT(*) n FROM tickets")[0]["n"]
    _sql(at, stmt)
    assert at.session_state["sql_result"] is None
    assert at.session_state["sql_error"][0] == "guard"
    assert at.warning
    assert digest(seeded.db_path) == before
    assert rows(seeded.db_path, "SELECT COUNT(*) n FROM tickets")[0]["n"] == n


def test_sql_bad_query_shows_error(console, seeded):
    at = console(seeded.db_path, tab="SQL")
    _sql(at, "SELECT no_such_column FROM tickets")
    assert at.session_state["sql_result"] is None
    assert at.session_state["sql_error"] is not None and at.session_state["sql_error"][0] not in (
        "guard",
        "timeout",
    )
    assert at.error


@pytest.mark.timeout(60)
def test_sql_timeout_friendly(console, seeded):
    at = console(seeded.db_path, tab="SQL")
    _sql(
        at, "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT COUNT(*) FROM n, tickets"
    )
    assert at.session_state["sql_error"][0] == "timeout"
    assert any("stopped" in str(w.value).lower() for w in at.warning)


# --------------------------------------------------------------------------- User Info tab


def _lookup(at, ident: str) -> None:
    at.text_input(key="users_lookup").set_value(ident)
    next(b for b in at.button if b.label == "Look up").click()
    at.run()
    assert not at.exception


@pytest.fixture
def users_up(start_user_service):
    return start_user_service()


def test_users_lookup_by_email_and_id(console, seeded, users_up):
    case = next(c for c in load_seed_cases() if c.user is not None)
    at = console(seeded.db_path, users_url=users_up.url, tab="User Info")
    _lookup(at, case.ticket.customer_email)
    names = [str(s.value) for s in at.subheader]
    assert any(case.ticket.customer_email.split("@")[0].split(".")[0].lower() in n.lower() for n in names), (
        names
    )
    uid = next(c for c in captions(at) if "`U-" in c).split("`")[1]
    _lookup(at, uid)
    assert any(uid in c for c in captions(at))
    assert at.metric  # plan / account / VIP / LTV tiles


def test_users_unknown_user(console, seeded, users_up):
    at = console(seeded.db_path, users_url=users_up.url, tab="User Info")
    _lookup(at, "nobody@nowhere.example")
    assert any("No user record" in str(w.value) for w in at.warning)


def test_users_browse_list(console, seeded, users_up):
    at = console(seeded.db_path, users_url=users_up.url, tab="User Info")
    assert caption_with(at, "Showing 50 of 50"), captions(at)
    assert 50 in frame_lengths(at)
    assert button(at, "users_more") is None  # 50 users < the 100-row default
    at.text_input(key="users_q").set_value("U-100")
    at.run()
    cap = caption_with(at, "Showing ")
    assert cap and "of 50" not in cap


def test_users_service_down_falls_back_to_db(console, seeded):
    at = console(seeded.db_path, users_url=DOWN, tab="User Info")
    assert caption_with(at, "the user service is not reachable"), captions(at)
    distinct = rows(seeded.db_path, "SELECT COUNT(DISTINCT lower(customer_email)) n FROM tickets")[0]["n"]
    assert caption_with(at, f"of {distinct}"), captions(at)
    email = load_seed_cases()[0].ticket.customer_email
    _lookup(at, email)
    assert any("not reachable" in str(e.value) for e in at.error)
    assert at.dataframe, "tickets for the email from the DB"


def test_users_service_down_and_no_db(console, tmp_path):
    at = console(str(tmp_path / "missing.db"), users_url=DOWN, tab="User Info")
    assert any("not reachable" in str(e.value) for e in at.error)
