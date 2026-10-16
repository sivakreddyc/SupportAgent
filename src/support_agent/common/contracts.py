"""Component interfaces. Frozen contract — every agent codes against these.

Implementations:
  TicketPortal     -> support_agent.browser.ticket_portal.PlaywrightTicketPortal   (Agent A)
  UserInfoClient   -> support_agent.clients.user_info.HttpUserInfoClient          (Agent A)
  LLMClient        -> support_agent.llm.anthropic_client.AnthropicLLMClient       (Agent C)
                      support_agent.llm.mock_client.MockLLMClient                 (Agent C)
  PolicyGate       -> support_agent.policy.gate.RuleBasedPolicyGate               (Agent C)
  Repository       -> support_agent.storage.repository.SqliteRepository           (Agent B)
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .models import (
    LLMResult,
    PolicyDecision,
    Ticket,
    TicketSummary,
    UserInfo,
)
from .states import TicketState


class PortalError(Exception):
    """Raised by TicketPortal on unrecoverable UI errors (after its own retries)."""


class UserServiceError(Exception):
    """Raised by UserInfoClient on non-404 failures (after its own retries)."""


class LLMError(Exception):
    """Raised by LLMClient. `retryable` tells the worker whether to retry the ticket."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


@runtime_checkable
class TicketPortal(Protocol):
    async def start(self) -> None: ...
    async def close(self) -> None: ...
    async def list_open_tickets(self) -> list[TicketSummary]: ...
    async def get_ticket(self, ticket_id: str) -> Ticket: ...
    async def post_draft(self, ticket_id: str, body: str, *, idempotency_key: str) -> None: ...
    async def post_internal_note(self, ticket_id: str, note: str) -> None: ...
    async def set_status(self, ticket_id: str, status: str) -> None: ...


@runtime_checkable
class UserInfoClient(Protocol):
    async def get_user_by_email(self, email: str) -> UserInfo | None: ...
    async def close(self) -> None: ...


@runtime_checkable
class LLMClient(Protocol):
    model: str

    async def analyze(self, ticket: Ticket, user: UserInfo | None) -> LLMResult: ...


@runtime_checkable
class PolicyGate(Protocol):
    def evaluate(self, ticket: Ticket, result: LLMResult, user: UserInfo | None) -> PolicyDecision: ...


@runtime_checkable
class Repository(Protocol):
    """Intent-level persistence API. No raw SQL outside support_agent.storage."""

    async def init(self) -> None: ...
    async def close(self) -> None: ...

    # tickets & state
    async def upsert_discovered(self, summary: TicketSummary, trace_id: str) -> bool:
        """Insert as DISCOVERED if unseen. Returns True if newly inserted."""
        ...

    async def get_ticket_row(self, ticket_id: str) -> dict[str, Any] | None: ...
    async def tickets_in_states(self, states: set[TicketState]) -> list[dict[str, Any]]: ...
    async def transition(
        self,
        ticket_id: str,
        to_state: TicketState,
        *,
        worker_id: str | None = None,
        detail: dict[str, Any] | None = None,
        error: str | None = None,
        bump_attempt: bool = False,
    ) -> None:
        """Validates against the state machine; raises InvalidTransition. Atomic with the transitions log."""
        ...

    # step records
    async def save_snapshot(self, ticket: Ticket) -> None: ...
    async def record_user_lookup(self, ticket_id: str, user: UserInfo | None, latency_ms: float) -> None: ...
    async def record_llm_call(
        self, ticket_id: str, result: LLMResult | None, *, error: str | None = None
    ) -> None: ...
    async def record_policy_decision(self, ticket_id: str, decision: PolicyDecision) -> None: ...

    # drafts & review
    async def create_draft(
        self, ticket_id: str, body: str, *, source: str, status: str, idempotency_key: str
    ) -> int: ...
    async def mark_draft_posted(self, draft_id: int) -> None: ...
    async def latest_draft(self, ticket_id: str) -> dict[str, Any] | None: ...
    async def approved_pending_post(self) -> list[dict[str, Any]]:
        """Tickets in APPROVED with their draft to post (used by the review dispatcher)."""
        ...

    # observability
    async def record_span(self, span: dict[str, Any]) -> None: ...
    async def record_metric_samples(self, samples: list[tuple[str, dict[str, str], float]]) -> None: ...
    async def start_scheduler_run(self) -> int: ...
    async def finish_scheduler_run(
        self, run_id: int, *, found: int, enqueued: int, skipped_full: int, error: str | None = None
    ) -> None: ...
