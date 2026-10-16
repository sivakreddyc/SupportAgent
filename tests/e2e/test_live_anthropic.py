"""Optional live check (PLAN §8): one real Anthropic call through AnthropicLLMClient, validated against
the LLMAnalysis schema. Runs only with `-m live` and ANTHROPIC_API_KEY set; costs a few cents.

    ANTHROPIC_API_KEY=... .venv/bin/pytest -m live tests/e2e/test_live_anthropic.py
"""

from __future__ import annotations

import os

import pytest

from support_agent.common.config import Settings
from support_agent.common.models import LLMAnalysis
from tests.conftest import require
from tests.fakes import make_ticket, make_user

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"),
]


async def test_real_call_returns_schema_valid_analysis():
    AnthropicLLMClient = require("support_agent.llm.anthropic_client", "AnthropicLLMClient")
    settings = Settings(
        llm_provider="anthropic",
        llm_model=os.environ.get("SA_LLM_MODEL", "claude-opus-5-5"),
        llm_effort=os.environ.get("SA_LLM_EFFORT", "low"),
        ticket_timeout_seconds=120,
    )
    client = AnthropicLLMClient(settings)
    ticket = make_ticket(
        subject="How do I export my projects to CSV?",
        body="Hi, is there a way to export all my projects and tasks to CSV? Thanks, Maria",
        customer_email="maria@example.com",
    )
    try:
        r = await client.analyze(ticket, make_user("maria@example.com"))
    finally:
        await client.close()
    assert r.stop_reason in ("end_turn", "refusal", "max_tokens")
    if r.stop_reason == "end_turn":
        assert isinstance(r.analysis, LLMAnalysis)
        assert 0.0 <= r.analysis.confidence <= 1.0
        assert r.analysis.draft_reply.strip()
        assert not r.analysis.legal_flag and not r.analysis.security_flag
    assert r.usage.input_tokens > 0 and r.usage.output_tokens > 0
    assert r.latency_ms > 0 and r.model
