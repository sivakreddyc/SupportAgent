"""AnthropicLLMClient (PLAN §5.9) against a mocked HTTP transport. Never touches the real API.

The SDK (anthropic 1.15) is built on `httpx2`, so the transport is `httpx2.MockTransport`
injected through `AsyncAnthropic(http_client=...)` and the client's `client=` kwarg.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from typing import Any

import anthropic
import pytest

from support_agent.common.config import Settings
from support_agent.common.contracts import LLMClient, LLMError
from support_agent.common.models import LLMAnalysis
from support_agent.observability import metrics as m
from support_agent.observability.logging import bind_context, clear_context
from support_agent.observability.tracing import set_span_sink
from tests.conftest import require
from tests.fakes import make_analysis, make_ticket, make_user

httpx2 = pytest.importorskip("httpx2")
mod = require("support_agent.llm.anthropic_client", "AnthropicLLMClient", "map_error")
AnthropicLLMClient = mod.AnthropicLLMClient
map_error = mod.map_error
SERVER_FALLBACK_BETA = mod.SERVER_FALLBACK_BETA
prompts = require("support_agent.llm.prompts", "SYSTEM_PROMPT", "build_user_message")

Handler = Callable[[Any], Any]


# --------------------------------------------------------------------------- helpers


def message_json(
    content: list[dict] | None = None,
    *,
    stop_reason: str = "end_turn",
    model: str = "claude-opus-5-5",
    usage: dict | None = None,
) -> dict:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content if content is not None else [],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage
        or {
            "input_tokens": 1200,
            "output_tokens": 310,
            "cache_read_input_tokens": 1000,
            "cache_creation_input_tokens": 50,
        },
    }


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def analysis_block(**overrides) -> dict:
    return text_block(make_analysis(**overrides).model_dump_json())


class Recorder:
    """httpx2 handler that records requests and replies with `reply(request)`."""

    def __init__(self, reply: Callable[[Any], Any]) -> None:
        self.reply = reply
        self.requests: list[Any] = []

    def __call__(self, request):
        self.requests.append(request)
        return self.reply(request)

    @property
    def body(self) -> dict:
        return json.loads(self.requests[-1].content)

    @property
    def headers(self):
        return self.requests[-1].headers


def ok(content: list[dict] | None = None, **kw) -> Recorder:
    payload = message_json([analysis_block()] if content is None else content, **kw)
    return Recorder(lambda req: httpx2.Response(200, json=payload))


def status(code: int, err_type: str = "api_error") -> Recorder:
    body = {"type": "error", "error": {"type": err_type, "message": f"boom {code}"}}
    return Recorder(lambda req: httpx2.Response(code, json=body, headers={"request-id": "req_abc123"}))


def make_client(handler: Handler, settings: Settings | None = None, **sdk_kw) -> AnthropicLLMClient:
    sdk_kw.setdefault("api_key", "test-key")
    sdk = anthropic.AsyncAnthropic(
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        **sdk_kw,
    )
    return AnthropicLLMClient(settings or Settings(), client=sdk)


def unique_settings(**kw) -> Settings:
    """A unique model name isolates metric deltas between tests (global registry)."""
    kw.setdefault("llm_model", f"claude-test-{uuid.uuid4().hex[:8]}")
    return Settings(**kw)


def metric(name: str, **labels) -> float:
    return m.REGISTRY.get_sample_value(name, labels) or 0.0


@pytest.fixture
def ticket():
    return make_ticket(subject="Cannot export CSV", body="Export to CSV fails with a spinner forever.")


# --------------------------------------------------------------------------- request shape


async def test_satisfies_protocol():
    c = make_client(ok())
    assert isinstance(c, LLMClient)
    assert c.model == Settings().llm_model


async def test_request_core_fields(ticket):
    rec = ok()
    s = Settings(llm_model="claude-sonnet-5-5", llm_max_tokens=2048)
    user = make_user()
    await make_client(rec, s).analyze(ticket, user)
    assert rec.requests[-1].method == "POST"
    assert rec.requests[-1].url.path == "/v1/messages"
    assert rec.headers["x-api-key"] == "test-key"
    b = rec.body
    assert b["model"] == "claude-sonnet-5-5"
    assert b["max_tokens"] == 2048
    assert b["messages"] == [{"role": "user", "content": prompts.build_user_message(ticket, user)}]


async def test_system_prompt_is_frozen_cached_block(ticket):
    rec = ok()
    await make_client(rec).analyze(ticket, make_user())
    system = rec.body["system"]
    assert system == [{"type": "text", "text": prompts.SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
    # volatile data stays out of the cacheable prefix
    assert ticket.subject not in system[0]["text"]
    assert ticket.customer_email not in system[0]["text"]


async def test_system_prompt_identical_across_tickets():
    rec = ok()
    c = make_client(rec)
    await c.analyze(make_ticket(body="first"), None)
    await c.analyze(
        make_ticket(body="second", customer_email="bob@example.com"), make_user("bob@example.com")
    )
    s1, s2 = (json.loads(r.content)["system"] for r in rec.requests)
    assert s1 == s2


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
async def test_output_config_effort_and_json_schema(ticket, effort):
    rec = ok()
    await make_client(rec, Settings(llm_effort=effort)).analyze(ticket, None)
    oc = rec.body["output_config"]
    assert oc["effort"] == effort
    assert oc["format"]["type"] == "json_schema"
    assert set(oc["format"]["schema"]["properties"]) == set(LLMAnalysis.model_fields)


async def test_server_fallback_enabled(ticket):
    rec = ok()
    await make_client(rec, Settings(llm_server_fallback=True)).analyze(ticket, None)
    assert rec.body["fallbacks"] == "default"
    assert SERVER_FALLBACK_BETA in rec.headers.get("anthropic-beta", "")
    assert SERVER_FALLBACK_BETA == "server-side-fallback-2026-07-01"


async def test_server_fallback_disabled(ticket):
    rec = ok()
    await make_client(rec, Settings(llm_server_fallback=False)).analyze(ticket, None)
    assert "fallbacks" not in rec.body
    assert SERVER_FALLBACK_BETA not in rec.headers.get("anthropic-beta", "")


async def test_forbidden_params_not_sent(ticket):
    rec = ok()
    await make_client(rec).analyze(ticket, make_user())
    b = rec.body
    for key in ("thinking", "temperature", "budget_tokens", "top_p", "top_k", "stop_sequences"):
        assert key not in b, key
    assert "budget_tokens" not in json.dumps(b)
    # no assistant prefill
    assert b["messages"][-1]["role"] == "user"


def test_build_request_inspectable(ticket):
    c = make_client(ok(), Settings(llm_server_fallback=False))
    kw = c.build_request(ticket, None)
    assert kw["output_format"] is LLMAnalysis
    assert kw["output_config"] == {"effort": "medium"}
    assert "betas" not in kw and "fallbacks" not in kw
    kw2 = make_client(ok(), Settings(llm_server_fallback=True)).build_request(ticket, None)
    assert kw2["betas"] == [SERVER_FALLBACK_BETA] and kw2["fallbacks"] == "default"


async def test_unknown_user_message_still_sent(ticket):
    rec = ok()
    await make_client(rec).analyze(ticket, None)
    assert rec.body["messages"][0]["content"] == prompts.build_user_message(ticket, None)


# --------------------------------------------------------------------------- response handling


async def test_happy_path_parses_analysis_and_usage(ticket):
    expected = make_analysis(category="technical", confidence=0.88)
    rec = ok([text_block(expected.model_dump_json())])
    r = await make_client(rec).analyze(ticket, None)
    assert r.analysis == expected
    assert r.stop_reason == "end_turn"
    assert r.usage.input_tokens == 1200
    assert r.usage.output_tokens == 310
    assert r.usage.cache_read_input_tokens == 1000
    assert r.usage.cache_creation_input_tokens == 50
    assert r.latency_ms >= 0
    assert r.model == "claude-opus-5-5"
    assert json.loads(r.raw_text) == expected.model_dump(mode="json")


async def test_refusal_with_empty_content(ticket):
    r = await make_client(ok([], stop_reason="refusal")).analyze(ticket, None)
    assert r.analysis is None
    assert r.stop_reason == "refusal"
    assert r.raw_text is None
    assert r.usage.input_tokens == 1200


async def test_refusal_with_partial_content(ticket):
    r = await make_client(ok([text_block('{"category": "secu')], stop_reason="refusal")).analyze(ticket, None)
    assert r.analysis is None
    assert r.stop_reason == "refusal"
    assert r.raw_text == '{"category": "secu'
    assert r.usage.output_tokens == 310


async def test_refusal_wins_over_valid_json(ticket):
    """stop_reason is checked before content: a refusal never yields an analysis."""
    r = await make_client(ok([analysis_block()], stop_reason="refusal")).analyze(ticket, None)
    assert r.analysis is None and r.stop_reason == "refusal"


async def test_max_tokens_truncation(ticket):
    partial = make_analysis().model_dump_json()[:60]
    r = await make_client(ok([text_block(partial)], stop_reason="max_tokens")).analyze(ticket, None)
    assert r.analysis is None
    assert r.stop_reason == "max_tokens"
    assert r.usage.output_tokens == 310


async def test_schema_violation_gives_no_analysis(ticket):
    bad = make_analysis().model_dump(mode="json") | {"confidence": 1.7}
    r = await make_client(ok([text_block(json.dumps(bad))])).analyze(ticket, None)
    assert r.analysis is None
    assert r.stop_reason == "end_turn"
    assert "1.7" in r.raw_text
    assert r.usage.input_tokens == 1200


async def test_non_json_text_gives_no_analysis(ticket):
    r = await make_client(ok([text_block("Sorry, here is some prose instead.")])).analyze(ticket, None)
    assert r.analysis is None and r.stop_reason == "end_turn"


async def test_served_model_reported_on_fallback(ticket):
    s = unique_settings()
    rec = ok([analysis_block()], model="claude-sonnet-5-5")
    r = await make_client(rec, s).analyze(ticket, None)
    assert r.model == "claude-sonnet-5-5"  # what actually served the request
    assert r.analysis is not None
    # metrics stay keyed by the configured model
    assert metric("llm_calls_total", model=s.llm_model, result="ok") == 1


async def test_missing_usage_fields_default_to_zero(ticket):
    rec = ok([analysis_block()], usage={"input_tokens": 7, "output_tokens": 3})
    r = await make_client(rec).analyze(ticket, None)
    assert (r.usage.input_tokens, r.usage.output_tokens) == (7, 3)
    assert r.usage.cache_read_input_tokens == 0 and r.usage.cache_creation_input_tokens == 0


# --------------------------------------------------------------------------- error mapping (HTTP)


@pytest.mark.parametrize(
    "code,retryable",
    [
        (400, False),
        (401, False),
        (403, False),
        (404, False),
        (413, False),
        (422, False),
        (408, True),
        (409, True),
        (429, True),
        (500, True),
        (502, True),
        (503, True),
        (504, True),
        (529, True),
    ],
)
async def test_http_status_mapping(ticket, code, retryable):
    c = make_client(status(code))
    with pytest.raises(LLMError) as ei:
        await c.analyze(ticket, None)
    err = ei.value
    assert err.retryable is retryable
    assert str(code) in str(err)
    assert "req_abc123" in str(err)
    assert isinstance(err.__cause__, anthropic.APIStatusError)


async def test_connection_error_is_retryable(ticket):
    def boom(req):
        raise httpx2.ConnectError("connection refused", request=req)

    with pytest.raises(LLMError) as ei:
        await make_client(boom).analyze(ticket, None)
    assert ei.value.retryable is True
    assert isinstance(ei.value.__cause__, anthropic.APIConnectionError)


async def test_timeout_is_retryable(ticket):
    def slow(req):
        raise httpx2.ReadTimeout("read timed out", request=req)

    with pytest.raises(LLMError) as ei:
        await make_client(slow).analyze(ticket, None)
    assert ei.value.retryable is True
    assert isinstance(ei.value.__cause__, anthropic.APITimeoutError)


async def test_missing_api_key_is_not_retryable(ticket, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    rec = ok()
    c = make_client(rec, api_key=None)
    with pytest.raises(LLMError) as ei:
        await c.analyze(ticket, None)
    assert ei.value.retryable is False
    assert rec.requests == []  # never reached the network


def _resp(code: int):
    return httpx2.Response(code, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))


@pytest.mark.parametrize(
    "code,retryable", [(418, False), (402, False), (500, True), (599, True), (429, True)]
)
def test_map_error_generic_status(code, retryable):
    e = anthropic.APIStatusError(f"status {code}", response=_resp(code), body=None)
    assert map_error(e).retryable is retryable


def test_map_error_non_api_exceptions_not_retryable():
    assert map_error(ValueError("x")).retryable is False
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    assert map_error(anthropic.APIConnectionError(request=req)).retryable is True
    assert map_error(anthropic.APITimeoutError(request=req)).retryable is True


# --------------------------------------------------------------------------- metrics + tracing


async def test_metrics_ok_call(ticket):
    s = unique_settings()
    await make_client(ok(), s).analyze(ticket, None)
    model = s.llm_model
    assert metric("llm_calls_total", model=model, result="ok") == 1
    assert metric("llm_tokens_total", model=model, direction="input") == 1200
    assert metric("llm_tokens_total", model=model, direction="output") == 310
    assert metric("llm_tokens_total", model=model, direction="cache_read") == 1000
    assert metric("llm_latency_seconds_count", model=model) == 1
    assert metric("llm_refusals_total", model=model) == 0


async def test_metrics_refusal(ticket):
    s = unique_settings()
    await make_client(ok([], stop_reason="refusal"), s).analyze(ticket, None)
    assert metric("llm_calls_total", model=s.llm_model, result="refusal") == 1
    assert metric("llm_refusals_total", model=s.llm_model) == 1
    assert metric("llm_calls_total", model=s.llm_model, result="ok") == 0


@pytest.mark.parametrize("stop", ["end_turn", "max_tokens"])
async def test_metrics_parse_error(ticket, stop):
    s = unique_settings()
    await make_client(ok([text_block('{"bad": ')], stop_reason=stop), s).analyze(ticket, None)
    assert metric("llm_calls_total", model=s.llm_model, result="parse_error") == 1
    assert metric("llm_tokens_total", model=s.llm_model, direction="output") == 310


async def test_metrics_error(ticket):
    s = unique_settings()
    with pytest.raises(LLMError):
        await make_client(status(503), s).analyze(ticket, None)
    assert metric("llm_calls_total", model=s.llm_model, result="error") == 1
    assert metric("llm_latency_seconds_count", model=s.llm_model) == 1


async def test_span_recorded_with_usage_and_trace(ticket):
    spans: list[dict] = []

    async def sink(rec):
        spans.append(rec)

    set_span_sink(sink)
    bind_context(trace_id="trace-xyz")
    try:
        await make_client(ok()).analyze(ticket, None)
    finally:
        clear_context()
    sp = next(s for s in spans if s["name"] == "llm.analyze")
    assert sp["ticket_id"] == ticket.ticket_id
    assert sp["trace_id"] == "trace-xyz"
    assert sp["status"] == "ok"
    assert sp["attrs"]["result"] == "ok"
    assert sp["attrs"]["input_tokens"] == 1200
    assert sp["attrs"]["stop_reason"] == "end_turn"


async def test_error_span_marked(ticket):
    spans: list[dict] = []

    async def sink(rec):
        spans.append(rec)

    set_span_sink(sink)
    with pytest.raises(LLMError):
        await make_client(status(400)).analyze(ticket, None)
    sp = next(s for s in spans if s["name"] == "llm.analyze")
    assert sp["status"] == "error"
    assert sp["attrs"]["retryable"] is False
