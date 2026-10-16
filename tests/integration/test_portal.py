"""PlaywrightTicketPortal vs the real mock ticket UI (PLAN §5.1/§5.3, A5–A8), headless Chromium.

Login, pagination, get_ticket fidelity, idempotent post_draft, notes, status, re-login after the session
is invalidated (cookies cleared or server secret rotated), 5xx retry -> PortalError + artifacts, metrics.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import httpx
import pytest

from support_agent.common.contracts import PortalError, TicketPortal
from support_agent.observability import metrics as m
from support_agent.observability.tracing import set_span_sink
from tests.conftest import require
from tests.fakes import load_seed_cases

pytestmark = pytest.mark.browser

PlaywrightTicketPortal = require("support_agent.browser.ticket_portal", "PlaywrightTicketPortal")

SEEDS = {c.ticket.ticket_id: c.ticket for c in load_seed_cases()}


def artifacts(directory: str, pattern: str) -> dict[str, str]:
    """suffix -> text (html) or path (png) of failure artifacts."""
    out = {}
    for f in Path(directory).glob(pattern):
        out[f.suffix] = f.read_text() if f.suffix == ".html" else str(f)
    return out


def metric(name: str, **labels) -> float:
    return m.REGISTRY.get_sample_value(name, labels) or 0.0


@pytest.fixture
async def portals(settings, start_mock_ui):
    """`p, ui = await portals(ui=None, retry_attempts=3, **settings_overrides)`; all closed at teardown."""
    created: list = []

    async def make(ui=None, *, retry_attempts=3, **settings_overrides):
        ui = ui or start_mock_ui()
        s = settings.model_copy(update={"ticket_ui_url": ui.url, **settings_overrides})
        p = PlaywrightTicketPortal(s, retry_attempts=retry_attempts, retry_base_delay=0.01)
        created.append(p)
        await p.start()
        return p, ui

    yield make
    for p in created:
        await p.close()


# --------------------------------------------------------------------------- lifecycle


async def test_lifecycle(settings, start_mock_ui):
    ui = start_mock_ui()
    p = PlaywrightTicketPortal(settings.model_copy(update={"ticket_ui_url": ui.url}))
    assert isinstance(p, TicketPortal)
    with pytest.raises(RuntimeError):
        await p.list_open_tickets()
    launches0 = metric("browser_launches_total")
    await p.start()
    await p.start()  # idempotent
    assert metric("browser_launches_total") - launches0 == 1
    await p.close()
    await p.close()  # idempotent
    with pytest.raises(RuntimeError):
        await p.get_ticket("TCK-1001")


async def test_bad_credentials_fail_fast(portals):
    p, _ = await portals(ticket_ui_password="wrong")
    with pytest.raises(PortalError, match="login"):
        await p.get_ticket("TCK-1001")


async def test_unreachable_ui(portals, start_mock_ui, settings):
    p = PlaywrightTicketPortal(
        settings.model_copy(update={"ticket_ui_url": "http://127.0.0.1:1", "browser_timeout_ms": 2000}),
        retry_attempts=2,
        retry_base_delay=0.01,
    )
    await p.start()
    try:
        with pytest.raises(PortalError):
            await p.list_open_tickets()
    finally:
        await p.close()


# --------------------------------------------------------------------------- reads


async def test_list_all_seed_tickets_across_pages(portals):
    p, _ = await portals()  # 53 seed tickets, 20 per page -> 3 pages + empty page
    rows = await p.list_open_tickets()
    assert {r.ticket_id for r in rows} == set(SEEDS)
    assert len(rows) == 53
    for r in rows:
        seed = SEEDS[r.ticket_id]
        assert r.subject == seed.subject
        assert r.customer_email == seed.customer_email
        assert r.created_at == seed.created_at
        assert r.status == "open"


async def test_pagination_small_pages(portals, start_mock_ui):
    ui = start_mock_ui(page_size=7)
    p, _ = await portals(ui)
    assert len(await p.list_open_tickets()) == 53


async def test_empty_ui(portals, start_mock_ui):
    p, _ = await portals(start_mock_ui(seed=False))
    assert await p.list_open_tickets() == []


async def test_generated_tickets_listed(portals, start_mock_ui):
    ui = start_mock_ui(page_size=50)
    async with httpx.AsyncClient(base_url=ui.url) as c:
        await c.post("/admin/generate", data={"n": 120, "seed": 1})
    p, _ = await portals(ui)
    rows = await p.list_open_tickets()
    assert len(rows) == 173 and len({r.ticket_id for r in rows}) == 173


@pytest.mark.parametrize("tid", ["TCK-1001", "TCK-1003", "TCK-1010", "TCK-1045", "TCK-1052"])
async def test_get_ticket_matches_seed(portals, tid):
    p, _ = await portals()
    t = await p.get_ticket(tid)
    seed = SEEDS[tid]
    assert t.ticket_id == tid
    assert t.subject == seed.subject
    assert t.body == seed.body  # exact, including newlines
    assert t.customer_email == seed.customer_email
    assert t.created_at == seed.created_at
    assert t.priority == seed.priority
    assert [(mm.author, mm.body) for mm in t.messages] == [(mm.author, mm.body) for mm in seed.messages]
    assert t.drafts == seed.drafts


async def test_get_unknown_ticket_is_fatal(portals):
    p, _ = await portals()
    err0 = metric("browser_ops_total", op="get_ticket", result="error")
    with pytest.raises(PortalError, match="not found"):
        await p.get_ticket("TCK-9999")
    assert metric("browser_ops_total", op="get_ticket", result="error") - err0 == 1


# --------------------------------------------------------------------------- writes


async def test_post_draft_idempotent(portals):
    p, _ = await portals()
    key = f"TCK-1001:{uuid.uuid4().hex[:8]}"
    await p.post_draft("TCK-1001", "Hello Maria,\n\nHere is how to export.", idempotency_key=key)
    await p.post_draft("TCK-1001", "Hello Maria,\n\nHere is how to export.", idempotency_key=key)
    t = await p.get_ticket("TCK-1001")
    assert t.drafts == ["Hello Maria,\n\nHere is how to export."]
    await p.post_draft("TCK-1001", "Second draft", idempotency_key=key + "-2")
    assert (await p.get_ticket("TCK-1001")).drafts == [
        "Hello Maria,\n\nHere is how to export.",
        "Second draft",
    ]


async def test_post_draft_concurrent_same_key(portals):
    p, _ = await portals(browser_page_pool=3)
    await asyncio.gather(*(p.post_draft("TCK-1002", "only once", idempotency_key="k-same") for _ in range(6)))
    assert (await p.get_ticket("TCK-1002")).drafts == ["only once"]


async def test_post_draft_idempotent_across_portal_instances(portals, start_mock_ui):
    """A crashed/restarted worker with a fresh browser must not double-post."""
    ui = start_mock_ui()
    p1, _ = await portals(ui)
    await p1.post_draft("TCK-1003", "draft", idempotency_key="k-restart")
    await p1.close()
    p2, _ = await portals(ui)
    await p2.post_draft("TCK-1003", "draft", idempotency_key="k-restart")
    assert (await p2.get_ticket("TCK-1003")).drafts == ["draft"]


async def test_draft_visible_over_plain_http(portals):
    p, ui = await portals()
    await p.post_draft("TCK-1004", "visible via html", idempotency_key="k-html")
    async with httpx.AsyncClient(base_url=ui.url, follow_redirects=True) as c:
        await c.post("/login", data={"username": "agent", "password": "agent", "next": "/tickets"})
        html = (await c.get("/tickets/TCK-1004")).text
    assert "visible via html" in html and 'data-idempotency-key="k-html"' in html


@pytest.mark.parametrize("body,key", [("   ", "k"), ("body", "")])
async def test_post_draft_rejects_bad_input(portals, body, key):
    p, _ = await portals()
    with pytest.raises(PortalError):
        await p.post_draft("TCK-1001", body, idempotency_key=key)


async def test_post_internal_note(portals):
    p, _ = await portals()
    await p.post_internal_note("TCK-1005", "Escalated: LEGAL, ANGRY")
    assert (await p.get_ticket("TCK-1005")).notes[-1] == "Escalated: LEGAL, ANGRY"
    with pytest.raises(PortalError):
        await p.post_internal_note("TCK-1005", "  ")


async def test_set_status_removes_from_open_list(portals):
    p, _ = await portals()
    await p.set_status("TCK-1006", "pending")
    assert (await p.get_ticket("TCK-1006")).status == "pending"
    await p.set_status("TCK-1006", "pending")  # no-op
    ids = {r.ticket_id for r in await p.list_open_tickets()}
    assert "TCK-1006" not in ids and len(ids) == 52
    with pytest.raises(PortalError):
        await p.set_status("TCK-1006", "archived")


async def test_parallel_reads_with_small_pool(portals):
    p, _ = await portals(browser_page_pool=2)
    ids = list(SEEDS)[:10]
    tickets = await asyncio.gather(*(p.get_ticket(t) for t in ids))
    assert [t.ticket_id for t in tickets] == ids


# --------------------------------------------------------------------------- session expiry


async def test_relogin_after_cookies_cleared(portals):
    p, _ = await portals()
    await p.get_ticket("TCK-1001")
    await p.list_open_tickets()
    await p._writer_ctx.clear_cookies()
    await p._discovery_ctx.clear_cookies()
    assert (await p.get_ticket("TCK-1001")).ticket_id == "TCK-1001"
    assert len(await p.list_open_tickets()) == 53
    await p.post_draft("TCK-1001", "after relogin", idempotency_key="k-relogin")
    assert "after relogin" in (await p.get_ticket("TCK-1001")).drafts


async def test_relogin_after_server_invalidates_sessions(portals):
    p, ui = await portals()
    await p.get_ticket("TCK-1001")
    ui.app.state.secret = uuid.uuid4().hex  # every existing session cookie is now invalid
    assert (await p.get_ticket("TCK-1007")).ticket_id == "TCK-1007"
    await p.post_internal_note("TCK-1007", "note after expiry")
    assert (await p.get_ticket("TCK-1007")).notes[-1] == "note after expiry"


# --------------------------------------------------------------------------- 5xx retries


async def test_persistent_500_raises_portal_error_with_artifacts(portals, settings):
    p, ui = await portals(retry_attempts=3)
    await p.get_ticket("TCK-1001")  # logged in
    ui.app.state.error_rate = 1.0
    err0 = metric("browser_ops_total", op="get_ticket", result="error")
    with pytest.raises(PortalError, match="3 attempt"):
        await p.get_ticket("TCK-1001")
    assert metric("browser_ops_total", op="get_ticket", result="error") - err0 == 1
    files = artifacts(settings.artifacts_dir, "*get_ticket_TCK-1001*")
    assert set(files) == {".png", ".html"}
    assert "error-page" in files[".html"]


async def login_both_contexts(p) -> None:
    """Log both browser contexts in first, so these tests exercise op retries rather than login."""
    await p.get_ticket("TCK-1001")
    await p.list_open_tickets()


class FailNextLogins:
    """ASGI wrapper: the next `n` POST /login requests get an HTTP 500 page; everything else passes through."""

    def __init__(self, app) -> None:
        self.app = app
        self.state = app.state
        self.remaining = 0
        self.failed = 0

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] == "http"
            and scope["method"] == "POST"
            and scope["path"] == "/login"
            and self.remaining > 0
        ):
            self.remaining -= 1
            self.failed += 1
            body = b'<html><body data-testid="error-page">Injected 500</body></html>'
            await send(
                {"type": "http.response.start", "status": 500, "headers": [(b"content-type", b"text/html")]}
            )
            await send({"type": "http.response.body", "body": body})
            return
        await self.app(scope, receive, send)


@pytest.fixture
def flaky_login_ui(serve):
    create_app = require("support_agent.mocks.ticket_ui.app", "create_app")
    wrapper = FailNextLogins(create_app())
    srv = serve(wrapper)
    return srv, wrapper


@pytest.mark.parametrize("failures", [1, 2])
async def test_login_survives_transient_500s(portals, flaky_login_ui, failures):
    """BUG-1 regression: a 5xx on POST /login is retried, not treated as bad credentials."""
    srv, gate = flaky_login_ui
    p, _ = await portals(srv, retry_attempts=3)
    gate.remaining = failures  # first login
    assert (await p.get_ticket("TCK-1001")).ticket_id == "TCK-1001"
    assert gate.failed == failures
    gate.app.state.secret = uuid.uuid4().hex  # session expiry -> re-login, also hit by 500s
    gate.remaining = failures
    assert (await p.get_ticket("TCK-1002")).ticket_id == "TCK-1002"
    assert gate.failed == 2 * failures


async def test_login_500s_beyond_retry_budget_raise_portal_error(portals, flaky_login_ui):
    srv, gate = flaky_login_ui
    p, _ = await portals(srv, retry_attempts=3)
    gate.remaining = 100
    with pytest.raises(PortalError) as ei:
        await p.get_ticket("TCK-1001")
    assert "credentials" not in str(ei.value).lower()  # it's a server error, not a bad password
    gate.remaining = 0
    assert (await p.get_ticket("TCK-1001")).ticket_id == "TCK-1001"  # recovers once the UI is healthy


async def test_transient_500s_are_retried(portals):
    p, ui = await portals(retry_attempts=12)
    await login_both_contexts(p)
    ui.app.state.error_rate = 0.4
    ok0 = metric("browser_ops_total", op="get_ticket", result="ok")
    for tid in list(SEEDS)[:5]:
        assert (await p.get_ticket(tid)).ticket_id == tid
    assert len(await p.list_open_tickets()) == 53
    assert metric("browser_ops_total", op="get_ticket", result="ok") - ok0 == 5


async def test_drafts_under_chaos_are_not_duplicated(portals):
    p, ui = await portals(retry_attempts=12)
    await login_both_contexts(p)
    ui.app.state.error_rate = 0.3
    for i in range(8):
        tid = f"TCK-10{10 + i}"
        await p.post_draft(tid, f"draft {i}", idempotency_key=f"chaos-{i}")
        await p.post_draft(tid, f"draft {i}", idempotency_key=f"chaos-{i}")
    ui.app.state.error_rate = 0.0
    for i in range(8):
        assert (await p.get_ticket(f"TCK-10{10 + i}")).drafts == [f"draft {i}"]


async def test_latency_tolerated(portals, start_mock_ui):
    p, _ = await portals(start_mock_ui(latency_ms=50))
    assert (await p.get_ticket("TCK-1001")).ticket_id == "TCK-1001"


# --------------------------------------------------------------------------- instrumentation


async def test_spans_and_op_metrics(portals):
    spans: list[dict] = []

    async def sink(rec):
        spans.append(rec)

    p, _ = await portals()
    set_span_sink(sink)
    ok0 = metric("browser_ops_total", op="post_draft", result="ok")
    lat0 = metric("browser_op_seconds_count", op="post_draft")
    await p.post_draft("TCK-1008", "x", idempotency_key="k-span")
    await p.post_draft("TCK-1008", "x", idempotency_key="k-span")
    assert metric("browser_ops_total", op="post_draft", result="ok") - ok0 == 2
    assert metric("browser_op_seconds_count", op="post_draft") - lat0 == 2
    posts = [s for s in spans if s["name"] == "browser.post_draft"]
    assert len(posts) == 2 and all(s["ticket_id"] == "TCK-1008" for s in posts)
    assert posts[1]["attrs"].get("skipped_duplicate") is True
