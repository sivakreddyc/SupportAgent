"""Self-tests for the D1 harness: fakes honour the Protocols, fault injection works, servers use ephemeral ports."""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest
from fastapi import FastAPI

from support_agent.common.contracts import (
    LLMClient,
    LLMError,
    PolicyGate,
    PortalError,
    TicketPortal,
    UserInfoClient,
)
from tests.conftest import FORBIDDEN_PORTS, require
from tests.fakes import (
    FakeLLMClient,
    FakePolicyGate,
    FakeTicketPortal,
    FakeUserInfoClient,
    fatal_llm_error,
    make_analysis,
    make_ticket,
    make_user,
)


def test_fakes_satisfy_protocols(fake_portal, fake_users, fake_llm, fake_policy):
    assert isinstance(fake_portal, TicketPortal)
    assert isinstance(fake_users, UserInfoClient)
    assert isinstance(fake_llm, LLMClient)
    assert isinstance(fake_policy, PolicyGate)


async def test_fake_portal_dedups_drafts_on_idempotency_key():
    t = make_ticket()
    portal = FakeTicketPortal([t])
    await portal.post_draft(t.ticket_id, "hello", idempotency_key="k1")
    await portal.post_draft(t.ticket_id, "hello", idempotency_key="k1")
    assert len(portal.drafts_for(t.ticket_id)) == 1
    assert portal.calls["post_draft"] == 2
    detail = await portal.get_ticket(t.ticket_id)
    assert detail.drafts == ["hello"]


async def test_fake_portal_lists_only_open():
    portal = FakeTicketPortal([make_ticket(status="open"), make_ticket(status="closed")])
    assert len(await portal.list_open_tickets()) == 1


async def test_fault_injection_fails_n_times_then_recovers():
    t = make_ticket()
    portal = FakeTicketPortal([t])
    portal.faults.on("get_ticket").fail(2, PortalError("500"))
    for _ in range(2):
        with pytest.raises(PortalError):
            await portal.get_ticket(t.ticket_id)
    assert (await portal.get_ticket(t.ticket_id)).ticket_id == t.ticket_id


async def test_fault_injection_forever_and_class_factory():
    llm = FakeLLMClient()
    llm.faults.on("analyze").fail(-1, lambda: LLMError("x", retryable=True))
    for _ in range(3):
        with pytest.raises(LLMError) as ei:
            await llm.analyze(make_ticket(), None)
        assert ei.value.retryable is True


async def test_fault_injection_delay():
    users = FakeUserInfoClient([make_user()])
    users.faults.on("get_user_by_email").delay(0.1)
    t0 = time.perf_counter()
    assert await users.get_user_by_email("ALICE@example.com") is not None
    assert time.perf_counter() - t0 >= 0.09
    assert await users.get_user_by_email("nobody@example.com") is None


async def test_fake_llm_per_ticket_responses_and_concurrency():
    llm = FakeLLMClient()
    t1, t2 = make_ticket(), make_ticket()
    llm.respond(t1.ticket_id, make_analysis(sentiment="angry"))
    llm.respond(t2.ticket_id, fatal_llm_error())
    assert (await llm.analyze(t1, None)).analysis.sentiment == "angry"
    with pytest.raises(LLMError) as ei:
        await llm.analyze(t2, None)
    assert ei.value.retryable is False
    llm.faults.on("analyze").delay(0.05)
    await asyncio.gather(*(llm.analyze(make_ticket(), None) for _ in range(5)))
    assert llm.max_concurrent == 5


def test_fake_policy_defaults_and_overrides(fake_llm):
    gate = FakePolicyGate()
    t = make_ticket()
    from tests.fakes import make_result

    assert gate.evaluate(t, make_result(), None).outcome == "auto_approve"
    assert gate.evaluate(t, make_result(no_analysis=True), None).reasons == ["LLM_NO_ANALYSIS"]
    gate.review(t.ticket_id, "ANGRY")
    d = gate.evaluate(t, make_result(), None)
    assert d.outcome == "human_review" and d.reasons == ["ANGRY"]


def test_settings_fixture_is_isolated(settings, tmp_path):
    assert settings.db_path.startswith(str(tmp_path))
    assert settings.llm_provider == "mock"
    assert settings.ops_api_port not in FORBIDDEN_PORTS


def test_settings_fixture_ignores_shell_env(monkeypatch):
    # _isolate_env runs before this test body, so set-then-check proves Settings honours env (sanity)
    monkeypatch.setenv("SA_WORKER_COUNT", "9")
    from support_agent.common.config import Settings

    assert Settings().worker_count == 9


def test_serve_fixture_uses_ephemeral_port(serve):
    app = FastAPI()

    @app.get("/ping")
    def ping():
        return {"ok": True}

    srv = serve(app)
    assert srv.port not in FORBIDDEN_PORTS
    assert httpx.get(f"{srv.url}/ping").json() == {"ok": True}


def test_require_skips_for_missing_attr():
    with pytest.raises(pytest.skip.Exception):
        require("support_agent.common.models", "DoesNotExist")
    assert require("support_agent.common.models", "Ticket").__name__ == "Ticket"
