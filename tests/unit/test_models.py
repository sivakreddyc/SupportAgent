"""Shared models (PLAN §4.1): validation, literals, defaults, round-trips."""

from __future__ import annotations

import re
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from support_agent.common.models import (
    LLMAnalysis,
    LLMResult,
    LLMUsage,
    Order,
    PolicyDecision,
    Ticket,
    TicketSummary,
    UserInfo,
    WorkItem,
    new_trace_id,
    utcnow,
)
from tests.fakes import make_analysis, make_result, make_ticket, make_user


@pytest.mark.parametrize("c", [0.0, 0.5, 0.75, 1.0])
def test_confidence_in_bounds(c):
    assert make_analysis(confidence=c).confidence == c


@pytest.mark.parametrize("c", [-0.01, 1.01, 2, -1])
def test_confidence_out_of_bounds(c):
    with pytest.raises(ValidationError):
        make_analysis(confidence=c)


@pytest.mark.parametrize(
    "field,value",
    [
        ("category", "shipping"),
        ("sentiment", "furious"),
        ("urgency", "urgent"),
        ("confidence", "high"),
        ("refund_requested", "maybe"),
    ],
)
def test_analysis_rejects_bad_values(field, value):
    with pytest.raises(ValidationError):
        make_analysis(**{field: value})


@pytest.mark.parametrize(
    "field",
    [
        "category",
        "intent",
        "sentiment",
        "urgency",
        "confidence",
        "refund_requested",
        "legal_flag",
        "security_flag",
        "draft_reply",
        "reasoning_summary",
    ],
)
def test_analysis_required_fields(field):
    data = make_analysis().model_dump()
    del data[field]
    with pytest.raises(ValidationError):
        LLMAnalysis(**data)


def test_analysis_optional_fields_default():
    data = make_analysis().model_dump()
    del data["refund_amount"], data["suggested_actions"]
    a = LLMAnalysis(**data)
    assert a.refund_amount is None and a.suggested_actions == []


@pytest.mark.parametrize(
    "cat", ["billing", "refund", "technical", "account", "legal", "security", "feedback", "other"]
)
def test_all_categories_accepted(cat):
    assert make_analysis(category=cat).category == cat


def test_analysis_json_round_trip():
    a = make_analysis(refund_requested=True, refund_amount=120.5, suggested_actions=["refund"])
    assert LLMAnalysis.model_validate_json(a.model_dump_json()) == a


def test_analysis_json_schema_lists_fields():
    schema = LLMAnalysis.model_json_schema()
    assert set(schema["properties"]) == set(LLMAnalysis.model_fields)
    assert schema["properties"]["sentiment"]["enum"] == ["positive", "neutral", "frustrated", "angry"]


@pytest.mark.parametrize(
    "field,value", [("plan", "gold"), ("account_status", "banned"), ("signup_date", "yesterday")]
)
def test_user_rejects_bad_values(field, value):
    with pytest.raises(ValidationError):
        make_user(**{field: value})


def test_user_defaults():
    u = UserInfo(
        user_id="U1",
        email="a@b.c",
        name="A",
        plan="free",
        account_status="active",
        signup_date="2024-01-01",
        lifetime_value=0,
    )
    assert u.open_orders == [] and u.recent_refunds == 0 and u.vip is False


def test_order_status_literal():
    Order(order_id="O1", amount=10, status="paid", created_at=utcnow())
    with pytest.raises(ValidationError):
        Order(order_id="O1", amount=10, status="lost", created_at=utcnow())
    assert Order(order_id="O1", amount=1, status="paid", created_at=utcnow()).currency == "USD"


def test_ticket_defaults_are_not_shared():
    a, b = make_ticket(), make_ticket()
    a.drafts.append("x")
    a.messages.clear()
    assert b.drafts == [] and b.notes == [] and b.priority is None


def test_ticket_is_a_summary():
    t = make_ticket()
    assert isinstance(t, TicketSummary)
    s = TicketSummary.model_validate(t.model_dump(include=set(TicketSummary.model_fields)))
    assert s.ticket_id == t.ticket_id


def test_ticket_requires_body():
    with pytest.raises(ValidationError):
        Ticket(ticket_id="T", subject="s", customer_email="e", status="open", created_at=utcnow())


def test_ticket_parses_iso_datetime():
    t = Ticket(
        ticket_id="T",
        subject="s",
        customer_email="e",
        status="open",
        created_at="2026-10-16T10:00:00Z",
        body="b",
    )
    assert t.created_at.tzinfo is not None


@pytest.mark.parametrize("outcome", ["auto_approve", "human_review", "reject"])
def test_policy_outcomes(outcome):
    d = PolicyDecision(outcome=outcome)
    assert d.reasons == [] and d.priority == "normal"


def test_policy_rejects_bad_values():
    with pytest.raises(ValidationError):
        PolicyDecision(outcome="escalate")
    with pytest.raises(ValidationError):
        PolicyDecision(outcome="human_review", priority="urgent")


def test_llm_result_allows_missing_analysis():
    r = make_result(no_analysis=True, stop_reason="refusal")
    assert r.analysis is None and r.stop_reason == "refusal"
    assert LLMResult(analysis=None, latency_ms=1, model="m").usage == LLMUsage()


def test_llm_result_requires_model_and_latency():
    with pytest.raises(ValidationError):
        LLMResult(analysis=None)


def test_work_item_defaults():
    w = WorkItem(ticket_id="T", trace_id="x")
    assert w.attempt == 0
    assert w.enqueued_at.tzinfo is not None
    assert (utcnow() - w.enqueued_at).total_seconds() < 5


def test_utcnow_and_trace_ids():
    assert utcnow().tzinfo == UTC
    assert abs((utcnow() - datetime.now(UTC)).total_seconds()) < 1
    ids = {new_trace_id() for _ in range(1000)}
    assert len(ids) == 1000
    assert all(re.fullmatch(r"[0-9a-f]{32}", i) for i in ids)
