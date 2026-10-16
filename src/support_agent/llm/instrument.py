"""Shared metrics helpers for LLM clients (real and mock emit identical telemetry)."""

from __future__ import annotations

from structlog.contextvars import get_contextvars

from support_agent.common.models import LLMUsage
from support_agent.observability import metrics as m

# llm_calls_total{result} values
RESULT_OK = "ok"
RESULT_ERROR = "error"
RESULT_REFUSAL = "refusal"
RESULT_PARSE_ERROR = "parse_error"


def current_trace_id() -> str | None:
    """trace_id bound by the worker via observability.logging.bind_context, if any."""
    value = get_contextvars().get("trace_id")
    return str(value) if value else None


def record_llm_metrics(model: str, result: str, latency_s: float, usage: LLMUsage | None = None) -> None:
    m.LLM_CALLS.labels(model=model, result=result).inc()
    m.LLM_LATENCY.labels(model=model).observe(max(0.0, latency_s))
    if result == RESULT_REFUSAL:
        m.LLM_REFUSALS.labels(model=model).inc()
    if usage is not None:
        for direction, n in (
            ("input", usage.input_tokens),
            ("output", usage.output_tokens),
            ("cache_read", usage.cache_read_input_tokens),
            ("cache_creation", usage.cache_creation_input_tokens),
        ):
            if n:
                m.LLM_TOKENS.labels(model=model, direction=direction).inc(n)


def usage_span_attrs(usage: LLMUsage) -> dict[str, int]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_tokens": usage.cache_read_input_tokens,
        "cache_creation_tokens": usage.cache_creation_input_tokens,
    }
