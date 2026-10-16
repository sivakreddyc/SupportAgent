"""Real LLM provider: Anthropic Messages API via the official `anthropic` SDK (docs/PLAN.md §5.9).

Request shape (anthropic SDK 1.15):
    client.beta.messages.with_raw_response.parse(
        model=settings.llm_model, max_tokens=settings.llm_max_tokens,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": build_user_message(ticket, user)}],
        output_format=LLMAnalysis,                      # SDK merges this into output_config.format
        output_config={"effort": settings.llm_effort},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",   # only if llm_server_fallback
    )
No `thinking`, `budget_tokens`, `temperature` or prefill: those 400 on Opus 5.5 (thinking is always on;
effort is the control). `with_raw_response` is used because the SDK validates the JSON eagerly inside
`parse()`: a truncated (`max_tokens`) or mid-output-refused response would raise before we could read
`stop_reason` and usage, so on a validation failure we rebuild the plain message from the raw body.
"""

from __future__ import annotations

import time
from typing import Any

import anthropic
import pydantic
from anthropic.types.beta import BetaMessage

from support_agent.common.config import Settings
from support_agent.common.contracts import LLMError
from support_agent.common.models import LLMAnalysis, LLMResult, LLMUsage, Ticket, UserInfo
from support_agent.observability.logging import get_logger
from support_agent.observability.tracing import span

from .instrument import (
    RESULT_ERROR,
    RESULT_OK,
    RESULT_PARSE_ERROR,
    RESULT_REFUSAL,
    current_trace_id,
    record_llm_metrics,
    usage_span_attrs,
)
from .prompts import SYSTEM_PROMPT, build_user_message

SERVER_FALLBACK_BETA = "server-side-fallback-2026-07-01"

_log = get_logger("llm")


def _cls(name: str) -> type[BaseException] | None:
    return getattr(anthropic, name, None)


# Most-specific first. Status-code classes are siblings under APIStatusError, so order among them
# only matters relative to the APIStatusError / APIConnectionError catch-alls below.
_RETRYABLE: tuple[type[BaseException], ...] = tuple(
    c
    for c in (
        _cls("APITimeoutError"),  # subclass of APIConnectionError
        _cls("RateLimitError"),  # 429
        _cls("OverloadedError"),  # 529
        _cls("ServiceUnavailableError"),  # 503
        _cls("DeadlineExceededError"),  # 504
        _cls("InternalServerError"),  # other 5xx
        _cls("ConflictError"),  # 409 (SDK treats as retryable)
        _cls("APIConnectionError"),  # network failure before a response
    )
    if c is not None
)
_NON_RETRYABLE: tuple[type[BaseException], ...] = tuple(
    c
    for c in (
        _cls("BadRequestError"),  # 400
        _cls("AuthenticationError"),  # 401
        _cls("PermissionDeniedError"),  # 403
        _cls("NotFoundError"),  # 404 (bad model id)
        _cls("RequestTooLargeError"),  # 413
        _cls("UnprocessableEntityError"),  # 422
    )
    if c is not None
)


def map_error(e: BaseException) -> LLMError:
    """Translate an SDK exception into LLMError(retryable=...)."""
    status = getattr(e, "status_code", None)
    request_id = getattr(e, "request_id", None)
    msg = f"{type(e).__name__}"
    if status is not None:
        msg += f" ({status})"
    msg += f": {getattr(e, 'message', None) or e}"
    if request_id:
        msg += f" [request_id={request_id}]"

    if isinstance(e, _RETRYABLE):
        retryable = True
    elif isinstance(e, _NON_RETRYABLE):
        retryable = False
    elif isinstance(e, anthropic.APIStatusError):
        retryable = status is not None and (status >= 500 or status in (408, 409, 429))
    else:  # APIResponseValidationError, missing credentials (TypeError), ...
        retryable = False
    return LLMError(msg, retryable=retryable)


def _is_missing_credentials(e: BaseException) -> bool:
    """The SDK raises a bare TypeError at request time when no API key/auth is configured."""
    return isinstance(e, TypeError) and "authentication method" in str(e)


def _usage(u: Any) -> LLMUsage:
    if u is None:
        return LLMUsage()
    return LLMUsage(
        input_tokens=getattr(u, "input_tokens", 0) or 0,
        output_tokens=getattr(u, "output_tokens", 0) or 0,
        cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
        cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
    )


def _text(msg: Any) -> str | None:
    parts = [b.text for b in (getattr(msg, "content", None) or []) if getattr(b, "type", None) == "text"]
    return "".join(parts) if parts else None


def _fallback_used(msg: Any) -> bool:
    if any(getattr(b, "type", None) == "fallback" for b in (getattr(msg, "content", None) or [])):
        return True
    iterations = getattr(getattr(msg, "usage", None), "iterations", None) or []
    return any(getattr(it, "type", None) == "fallback_message" for it in iterations)


class AnthropicLLMClient:
    """`LLMClient` backed by the Anthropic API. Reads ANTHROPIC_API_KEY from the environment."""

    def __init__(self, settings: Settings, *, client: anthropic.AsyncAnthropic | None = None) -> None:
        self.settings = settings
        self.model = settings.llm_model
        self._client = client or anthropic.AsyncAnthropic(
            max_retries=2, timeout=float(settings.ticket_timeout_seconds)
        )

    async def close(self) -> None:
        await self._client.close()

    def build_request(self, ticket: Ticket, user: UserInfo | None) -> dict[str, Any]:
        """Keyword args for `beta.messages.parse` (exposed for inspection/tests)."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.settings.llm_max_tokens,
            "system": [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": build_user_message(ticket, user)}],
            "output_format": LLMAnalysis,
            "output_config": {"effort": self.settings.llm_effort},
        }
        if self.settings.llm_server_fallback:
            kwargs["betas"] = [SERVER_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    async def analyze(self, ticket: Ticket, user: UserInfo | None) -> LLMResult:
        model = self.model
        async with span(
            "llm.analyze",
            trace_id=current_trace_id(),
            ticket_id=ticket.ticket_id,
            model=model,
            provider="anthropic",
            effort=self.settings.llm_effort,
        ) as sp:
            t0 = time.perf_counter()
            parse_error: str | None = None
            try:
                raw = await self._client.beta.messages.with_raw_response.parse(
                    **self.build_request(ticket, user)
                )
                try:
                    msg: Any = await raw.parse()
                    parsed = msg.parsed_output
                except pydantic.ValidationError as ve:
                    parse_error = f"ValidationError: {ve.error_count()} error(s)"
                    msg = BetaMessage.model_validate(await raw.json())
                    parsed = None
            except (anthropic.AnthropicError, TypeError) as e:
                if isinstance(e, TypeError) and not _is_missing_credentials(e):
                    raise  # a genuine bug, not an API failure
                latency_s = time.perf_counter() - t0
                err = map_error(e)
                record_llm_metrics(model, RESULT_ERROR, latency_s)
                sp.set(result=RESULT_ERROR, retryable=err.retryable, error=str(err))
                _log.warning(
                    "llm.error",
                    ticket_id=ticket.ticket_id,
                    model=model,
                    retryable=err.retryable,
                    error=str(err),
                )
                raise err from e

            latency_s = time.perf_counter() - t0
            usage = _usage(getattr(msg, "usage", None))
            stop_reason = getattr(msg, "stop_reason", None)
            served_model = getattr(msg, "model", None) or model
            raw_text = _text(msg)

            # stop_reason first, content second.
            analysis: LLMAnalysis | None = None
            if stop_reason == "refusal":
                result = RESULT_REFUSAL
            elif stop_reason == "max_tokens":
                result = RESULT_PARSE_ERROR
                parse_error = parse_error or "truncated at max_tokens"
            elif parse_error is not None or not isinstance(parsed, LLMAnalysis):
                result = RESULT_PARSE_ERROR
                parse_error = parse_error or "no structured output in response"
            else:
                result = RESULT_OK
                analysis = parsed

            record_llm_metrics(model, result, latency_s, usage)
            fallback = _fallback_used(msg)
            sp.set(
                result=result,
                stop_reason=stop_reason,
                served_model=served_model,
                fallback_used=fallback,
                request_id=getattr(msg, "_request_id", None),
                **usage_span_attrs(usage),
            )
            if result != RESULT_OK:
                stop_details = getattr(msg, "stop_details", None)
                _log.warning(
                    "llm.no_analysis",
                    ticket_id=ticket.ticket_id,
                    model=served_model,
                    result=result,
                    stop_reason=stop_reason,
                    detail=parse_error,
                    refusal_category=getattr(stop_details, "category", None),
                )

            return LLMResult(
                analysis=analysis,
                usage=usage,
                latency_ms=round(latency_s * 1000, 3),
                model=served_model,
                stop_reason=stop_reason,
                raw_text=raw_text,
            )
