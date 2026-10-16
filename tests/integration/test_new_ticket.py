"""Phase 6: create a ticket by hand and trigger a poll (mock UI route, ops API, console helpers)."""

from __future__ import annotations

import asyncio
import re
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from support_agent.dashboard.views import new_ticket
from support_agent.mocks.ticket_ui.app import create_app
from support_agent.ops_api.app import create_ops_app
from support_agent.pipeline.scheduler import IntervalScheduler

# ---------- mock ticket UI: GET/POST /tickets/new ----------


@pytest.fixture
def ui_client() -> TestClient:
    c = TestClient(create_app(seed=True), follow_redirects=False)
    assert c.post("/login", data={"username": "agent", "password": "agent"}).status_code == 303
    return c


def _form(**over: str) -> dict[str, str]:
    return {
        "customer_email": "maria.gonzalez@example.com",
        "subject": "How do I export to CSV?",
        "body": "Where is the export option?",
        "priority": "",
        **over,
    }


def test_new_ticket_requires_login() -> None:
    c = TestClient(create_app(seed=True), follow_redirects=False)
    for r in (c.get("/tickets/new"), c.post("/tickets/new", data=_form())):
        assert r.status_code == 303 and r.headers["location"].startswith("/login")


def test_new_ticket_form_renders_and_is_not_captured_as_ticket_id(ui_client: TestClient) -> None:
    r = ui_client.get("/tickets/new")
    assert r.status_code == 200
    assert 'data-testid="new-ticket-form"' in r.text
    assert 'data-testid="new-ticket-link"' in ui_client.get("/tickets").text


def test_create_ticket_assigns_next_id_and_is_open(ui_client: TestClient) -> None:
    r = ui_client.post(
        "/tickets/new", data=_form(customer_email="  Maria.Gonzalez@Example.com ", priority="high")
    )
    assert r.status_code == 303
    tid = re.search(r"/tickets/(TCK-\d+)", r.headers["location"]).group(1)
    assert tid == "TCK-1054"  # 53 seed tickets → next id
    page = ui_client.get(f"/tickets/{tid}").text
    assert "How do I export to CSV?" in page and "maria.gonzalez@example.com" in page
    listing = ui_client.get("/tickets?status=open&page=3").text
    assert f'data-ticket-id="{tid}"' in listing and 'data-status="open"' in listing
    r2 = ui_client.post("/tickets/new", data=_form())
    assert "TCK-1055" in r2.headers["location"]


@pytest.mark.parametrize(
    "over, msg",
    [
        ({"subject": "  "}, "Subject and message are required."),
        ({"body": ""}, "Subject and message are required."),
        ({"customer_email": "not-an-email"}, "Enter a valid customer email."),
        ({"priority": "whenever"}, "Priority must be one of"),
    ],
)
def test_create_ticket_validation(ui_client: TestClient, over: dict[str, str], msg: str) -> None:
    r = ui_client.post("/tickets/new", data=_form(**over))
    assert r.status_code == 400
    assert msg in r.text
    assert 'data-testid="new-ticket-error"' in r.text


# ---------- ops API: POST /api/poll ----------


class _Poller:
    def __init__(self) -> None:
        self.calls = 0
        self.last_result: dict[str, Any] | None = None

    async def run_once(self, delay: float = 0.0) -> dict[str, Any]:
        self.calls += 1
        await asyncio.sleep(delay)
        self.last_result = {
            "found": 1,
            "new": 1,
            "enqueued": 1,
            "requeued": 0,
            "skipped_full": 0,
            "error": None,
        }
        return self.last_result


class _Runtime:
    def __init__(self, scheduler: Any, poller: Any) -> None:
        self.scheduler = scheduler
        self.poller = poller


async def test_poll_now_runs_a_poll_and_returns_result() -> None:
    poller = _Poller()
    sched = IntervalScheduler(3600, 0.0, poller.run_once, name="test")  # never ticks on its own
    app = create_ops_app(_Runtime(sched, poller))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://ops") as c:
        r = await c.post("/api/poll")
    assert r.status_code == 200
    assert r.json()["result"]["new"] == 1
    assert poller.calls == 1


async def test_poll_now_conflicts_while_a_poll_is_running() -> None:
    poller = _Poller()
    sched = IntervalScheduler(3600, 0.0, lambda: poller.run_once(delay=0.5), name="test")
    app = create_ops_app(_Runtime(sched, poller))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://ops") as c:
        assert await sched.run_now()  # first poll in flight
        r = await c.post("/api/poll")
        assert r.status_code == 409
        await asyncio.sleep(0.6)
        assert (await c.post("/api/poll")).status_code == 200


async def test_poll_now_when_runtime_not_started() -> None:
    app = create_ops_app(_Runtime(None, None))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://ops") as c:
        r = await c.post("/api/poll")
    assert r.status_code == 503


# ---------- console helpers (real HTTP against the mock UI on an ephemeral port) ----------


def test_console_create_ticket_helper(settings, mock_ui) -> None:
    s = settings.model_copy(update={"ticket_ui_url": mock_ui.url})
    tid = new_ticket.create_ticket(
        s,
        customer_email="dev.patel@example.com",
        subject="Charged twice",
        body="Refund $12 please",
        priority="",
    )
    assert re.fullmatch(r"TCK-\d+", tid)


def test_console_create_ticket_helper_surfaces_validation_error(settings, mock_ui) -> None:
    s = settings.model_copy(update={"ticket_ui_url": mock_ui.url})
    with pytest.raises(new_ticket.CreateError, match="valid customer email"):
        new_ticket.create_ticket(s, customer_email="nope", subject="x", body="y", priority="")


def test_console_helpers_report_unreachable_services(settings) -> None:
    s = settings.model_copy(update={"ticket_ui_url": "http://127.0.0.1:9", "ops_api_port": 9})
    with pytest.raises(new_ticket.CreateError, match="not reachable"):
        new_ticket.create_ticket(s, customer_email="a@b.co", subject="x", body="y", priority="")
    with pytest.raises(new_ticket.CreateError, match="not reachable"):
        new_ticket.poll_now(s)


def test_templates_are_complete() -> None:
    for name, tpl in new_ticket.TEMPLATES.items():
        assert set(tpl) == {"customer_email", "subject", "body", "priority"}, name
        assert tpl["priority"] in new_ticket.PRIORITIES, name
