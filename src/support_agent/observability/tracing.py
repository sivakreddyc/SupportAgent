"""Lightweight per-ticket tracing.

    async with span("llm.analyze", ticket_id=tid, trace_id=tr, model=m) as sp:
        ...
        sp.set(output_tokens=123)

Every finished span is logged (event="span.end") and handed to the registered sink
(the runtime registers Repository.record_span). Sink failures never break the caller.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from .logging import get_logger

SpanSink = Callable[[dict[str, Any]], Awaitable[None]]

_sink: SpanSink | None = None
_log = get_logger("tracing")


def set_span_sink(sink: SpanSink | None) -> None:
    global _sink
    _sink = sink


class Span:
    def __init__(self, name: str, trace_id: str | None, ticket_id: str | None, attrs: dict[str, Any]) -> None:
        self.name = name
        self.trace_id = trace_id
        self.ticket_id = ticket_id
        self.attrs = dict(attrs)
        self.status = "ok"
        self.start_ts = datetime.now(UTC)
        self._t0 = time.perf_counter()
        self.duration_ms = 0.0

    def set(self, **attrs: Any) -> None:
        self.attrs.update(attrs)

    def to_record(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "ticket_id": self.ticket_id,
            "name": self.name,
            "start_ts": self.start_ts.isoformat(),
            "duration_ms": round(self.duration_ms, 3),
            "status": self.status,
            "attrs": self.attrs,
        }


@asynccontextmanager
async def span(name: str, *, trace_id: str | None = None, ticket_id: str | None = None, **attrs: Any):
    sp = Span(name, trace_id, ticket_id, attrs)
    try:
        yield sp
    except BaseException as e:
        sp.status = "error"
        sp.attrs.setdefault("error", f"{type(e).__name__}: {e}")
        raise
    finally:
        sp.duration_ms = (time.perf_counter() - sp._t0) * 1000
        rec = sp.to_record()
        _log.info("span.end", **{k: v for k, v in rec.items() if k != "attrs"}, **sp.attrs)
        if _sink is not None:
            try:
                await _sink(rec)
            except Exception as e:  # never break the pipeline because of tracing
                _log.warning("span.sink_failed", error=str(e), span=name)
