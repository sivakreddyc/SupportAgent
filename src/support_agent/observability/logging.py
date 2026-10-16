"""Structured JSON logging (structlog).

Usage:
    from support_agent.observability.logging import get_logger, bind_context
    log = get_logger("worker")
    log.info("ticket.processing", ticket_id=t, trace_id=tr)
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars, unbind_contextvars

_configured = False


def configure_logging(level: str = "INFO", log_file: str | None = None) -> None:
    global _configured
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.handlers.RotatingFileHandler(log_file, maxBytes=20_000_000, backupCount=5))
    root = logging.getLogger()
    root.handlers.clear()
    for h in handlers:
        h.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(h)
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore", "uvicorn.access", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="ts"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(component: str) -> structlog.stdlib.BoundLogger:
    if not _configured:
        configure_logging()
    return structlog.get_logger().bind(component=component)


def bind_context(**kwargs) -> None:
    """Bind trace_id / ticket_id / worker_id for the current asyncio task."""
    bind_contextvars(**kwargs)


def unbind_context(*keys: str) -> None:
    unbind_contextvars(*keys)


def clear_context() -> None:
    clear_contextvars()
