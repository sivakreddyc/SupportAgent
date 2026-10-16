"""MockLLMClient (PLAN §5.9): deterministic, schema-valid, test markers, fake usage, shared telemetry."""

from __future__ import annotations

import re
import time
import uuid

import pytest
import yaml

from support_agent.common.config import Settings
from support_agent.common.contracts import LLMClient
from support_agent.common.models import LLMAnalysis
from support_agent.observability import metrics as m
from support_agent.observability.tracing import set_span_sink
from tests.conftest import require
from tests.fakes import POLICY_FILE, load_seed_cases, make_ticket, make_user

MockLLMClient = require("support_agent.llm.mock_client", "MockLLMClient")

SEEDS = load_seed_cases()
UNSAFE_PATTERNS = [re.compile(p) for p in yaml.safe_load(POLICY_FILE.read_text())["unsafe_draft_patterns"]]


@pytest.fixture
def llm(settings) -> MockLLMClient:
    return MockLLMClient(settings)  # settings fixture: mock_llm_latency_ms=0


def metric(name: str, **labels) -> float:
    return m.REGISTRY.get_sample_value(name, labels) or 0.0


def test_satisfies_protocol(llm):
    assert isinstance(llm, LLMClient)
    assert isinstance(llm.model, str) and llm.model


async def test_deterministic_same_instance(llm):
    t = make_ticket(body="I want a refund of $45 for last month, please.")
    a, b = await llm.analyze(t, make_user()), await llm.analyze(t, make_user())
    assert a.analysis == b.analysis
    assert a.usage == b.usage
    assert a.stop_reason == b.stop_reason
    assert a.raw_text == b.raw_text


async def test_deterministic_across_instances(settings):
    t = make_ticket(body="My account was hacked and someone changed my email!")
    r1 = await MockLLMClient(settings).analyze(t, None)
    r2 = await MockLLMClient(settings).analyze(t, None)
    assert r1.model_dump(exclude={"latency_ms"}) == r2.model_dump(exclude={"latency_ms"})


@pytest.mark.parametrize("case", SEEDS, ids=[c.ticket.ticket_id for c in SEEDS])
async def test_every_seed_gives_valid_or_marked_result(llm, case):
    r = await llm.analyze(case.ticket, case.user)
    marked = "[mock:refusal]" in case.ticket.body or "[mock:no-analysis]" in case.ticket.body
    if marked:
        assert r.analysis is None
    else:
        assert isinstance(r.analysis, LLMAnalysis)
        LLMAnalysis.model_validate(r.analysis.model_dump())  # round-trips through validation
        assert r.analysis.draft_reply.strip()
        assert "[mock:" not in r.analysis.draft_reply
    assert r.usage.input_tokens > 0
    assert r.model


async def test_fake_usage_scales_with_input(llm):
    short = await llm.analyze(make_ticket(body="Hi"), None)
    long = await llm.analyze(make_ticket(body="Please help me with exports. " * 200), None)
    assert long.usage.input_tokens > short.usage.input_tokens
    assert short.usage.output_tokens > 0


async def test_marker_refusal(llm):
    r = await llm.analyze(make_ticket(body="How do I move tasks?\n\n[mock:refusal]"), None)
    assert r.analysis is None
    assert r.stop_reason == "refusal"


async def test_marker_no_analysis(llm):
    r = await llm.analyze(make_ticket(body="Printer-friendly page?\n\n[mock:no-analysis]"), None)
    assert r.analysis is None
    assert r.stop_reason != "refusal"  # must hit LLM_NO_ANALYSIS, not LLM_REFUSAL


async def test_marker_unsafe_draft(llm):
    r = await llm.analyze(make_ticket(body="Default board view?\n\n[mock:unsafe-draft]"), make_user())
    a = r.analysis
    assert a is not None
    assert any(p.search(a.draft_reply) for p in UNSAFE_PATTERNS), a.draft_reply
    # nothing else should gate it: the draft alone must be the reason
    assert a.confidence >= 0.75 and not (a.legal_flag or a.security_flag or a.refund_requested)
    assert a.sentiment != "angry" and a.urgency != "critical"


async def test_marker_in_subject_also_works(llm):
    r = await llm.analyze(make_ticket(subject="[mock:refusal] moving tasks", body="hello"), None)
    assert r.analysis is None and r.stop_reason == "refusal"


async def test_clean_ticket_is_confident_and_unflagged(llm):
    t = make_ticket(
        subject="How do I export my projects to CSV?",
        body="Hi, is there a way to export all my projects to CSV? Thanks, Maria",
    )
    a = (await llm.analyze(t, make_user())).analysis
    assert a.confidence >= 0.75
    assert not (a.refund_requested or a.legal_flag or a.security_flag)
    assert a.sentiment in ("neutral", "positive") and a.urgency in ("low", "medium")


async def test_refund_amount_parsed(llm):
    a = (
        await llm.analyze(make_ticket(body="I was double charged. Please refund the $249.99 charge."), None)
    ).analysis
    assert a.refund_requested is True
    assert a.refund_amount == pytest.approx(249.99)


async def test_refund_amount_with_thousands_separator(llm):
    a = (await llm.analyze(make_ticket(body="Refund my $1,200 annual plan, I never used it."), None)).analysis
    assert a.refund_requested and a.refund_amount == pytest.approx(1200)


@pytest.mark.parametrize(
    "body,flag",
    [
        ("My lawyer will be in touch about this lawsuit.", "legal_flag"),
        ("I think my account was hacked last night.", "security_flag"),
        ("I want my money back now.", "refund_requested"),
    ],
)
async def test_keyword_flags(llm, body, flag):
    a = (await llm.analyze(make_ticket(body=body), None)).analysis
    assert getattr(a, flag) is True


async def test_draft_never_promises_refund(llm):
    a = (await llm.analyze(make_ticket(body="Refund me $500 or else."), None)).analysis
    assert not any(p.search(a.draft_reply) for p in UNSAFE_PATTERNS), a.draft_reply


async def test_latency_setting_honoured(settings):
    llm = MockLLMClient(settings.model_copy(update={"mock_llm_latency_ms": 120}))
    t0 = time.perf_counter()
    r = await llm.analyze(make_ticket(), None)
    assert time.perf_counter() - t0 >= 0.11
    assert r.latency_ms >= 100


async def test_zero_latency_is_fast(llm):
    t0 = time.perf_counter()
    for _ in range(20):
        await llm.analyze(make_ticket(), None)
    assert time.perf_counter() - t0 < 1.0


async def test_metrics_and_span(llm):
    model = llm.model
    ok0 = metric("llm_calls_total", model=model, result="ok")
    ref0 = metric("llm_refusals_total", model=model)
    in0 = metric("llm_tokens_total", model=model, direction="input")
    spans: list[dict] = []

    async def sink(rec):
        spans.append(rec)

    set_span_sink(sink)
    r = await llm.analyze(make_ticket(ticket_id=f"T-{uuid.uuid4().hex[:6]}"), None)
    await llm.analyze(make_ticket(body="x [mock:refusal]"), None)
    assert metric("llm_calls_total", model=model, result="ok") - ok0 == 1
    assert metric("llm_refusals_total", model=model) - ref0 == 1
    assert metric("llm_tokens_total", model=model, direction="input") - in0 >= r.usage.input_tokens
    assert [s["name"] for s in spans].count("llm.analyze") == 2


def test_constructible_from_default_settings():
    assert MockLLMClient(Settings(llm_provider="mock")).model
