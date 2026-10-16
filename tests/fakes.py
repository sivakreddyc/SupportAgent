"""In-memory fakes for every component Protocol, built only on `support_agent.common`.

Each fake supports failure injection through a `Faults` object:

    portal.faults.on("post_draft").fail(2, PortalError("boom"))   # fail the next 2 calls
    llm.faults.on("analyze").fail(1, LLMError("bad", retryable=False))
    users.faults.on("get_user_by_email").delay(0.5)                # slow down every call
"""

from __future__ import annotations

import asyncio
import itertools
import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from support_agent.common.contracts import LLMError, PortalError
from support_agent.common.models import (
    LLMAnalysis,
    LLMResult,
    LLMUsage,
    PolicyDecision,
    Ticket,
    TicketSummary,
    UserInfo,
)

# --------------------------------------------------------------------------- failure injection


@dataclass
class _OpFaults:
    remaining_failures: int = 0
    exc_factory: Callable[[], BaseException] | None = None
    delay_s: float = 0.0
    fail_always: bool = False

    def fail(self, times: int, exc: BaseException | Callable[[], BaseException]) -> _OpFaults:
        """Raise `exc` on the next `times` calls (times < 0 means forever)."""
        self.exc_factory = exc if callable(exc) and not isinstance(exc, BaseException) else (lambda: exc)
        if times < 0:
            self.fail_always = True
        else:
            self.remaining_failures = times
        return self

    def delay(self, seconds: float) -> _OpFaults:
        self.delay_s = seconds
        return self

    def clear(self) -> None:
        self.remaining_failures = 0
        self.exc_factory = None
        self.delay_s = 0.0
        self.fail_always = False


class Faults:
    def __init__(self) -> None:
        self._ops: dict[str, _OpFaults] = defaultdict(_OpFaults)

    def on(self, op: str) -> _OpFaults:
        return self._ops[op]

    def clear(self) -> None:
        self._ops.clear()

    async def apply(self, op: str) -> None:
        f = self._ops.get(op)
        if f is None:
            return
        if f.delay_s:
            await asyncio.sleep(f.delay_s)
        if f.exc_factory is not None and (f.fail_always or f.remaining_failures > 0):
            if not f.fail_always:
                f.remaining_failures -= 1
            raise f.exc_factory()


# --------------------------------------------------------------------------- model factories

_ids = itertools.count(1)
_T0 = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)


def make_ticket(
    ticket_id: str | None = None,
    *,
    subject: str = "Question about my account",
    body: str = "Hi, how do I change my notification settings? Thanks!",
    customer_email: str = "alice@example.com",
    status: str = "open",
    priority: str | None = None,
) -> Ticket:
    n = next(_ids)
    return Ticket(
        ticket_id=ticket_id or f"T-{n:05d}",
        subject=subject,
        customer_email=customer_email,
        status=status,
        created_at=_T0 + timedelta(minutes=n),
        body=body,
        priority=priority,
    )


def make_user(email: str = "alice@example.com", **overrides: Any) -> UserInfo:
    data: dict[str, Any] = {
        "user_id": f"U-{email.split('@')[0]}",
        "email": email,
        "name": email.split("@")[0].title(),
        "plan": "pro",
        "account_status": "active",
        "signup_date": date(2024, 1, 15),
        "lifetime_value": 420.0,
        "open_orders": [],
        "recent_refunds": 0,
        "vip": False,
    }
    data.update(overrides)
    return UserInfo(**data)


def make_analysis(**overrides: Any) -> LLMAnalysis:
    """A clean analysis that should pass every policy rule (auto_approve)."""
    data: dict[str, Any] = {
        "category": "account",
        "intent": "change notification settings",
        "sentiment": "neutral",
        "urgency": "low",
        "confidence": 0.95,
        "refund_requested": False,
        "refund_amount": None,
        "legal_flag": False,
        "security_flag": False,
        "draft_reply": "Hi! You can change notifications under Settings > Notifications. Best, Support",
        "reasoning_summary": "Simple how-to question.",
        "suggested_actions": [],
    }
    data.update(overrides)
    return LLMAnalysis(**data)


def make_result(
    analysis: LLMAnalysis | None = None,
    *,
    stop_reason: str | None = "end_turn",
    model: str = "fake-llm",
    input_tokens: int = 100,
    output_tokens: int = 50,
    latency_ms: float = 1.0,
    no_analysis: bool = False,
) -> LLMResult:
    return LLMResult(
        analysis=None if no_analysis else (analysis or make_analysis()),
        usage=LLMUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        latency_ms=latency_ms,
        model=model,
        stop_reason=stop_reason,
    )


# --------------------------------------------------------------------------- TicketPortal


@dataclass
class PostedDraft:
    ticket_id: str
    body: str
    idempotency_key: str


class FakeTicketPortal:
    """In-memory TicketPortal. Drafts are deduplicated on idempotency_key like the real portal."""

    def __init__(self, tickets: list[Ticket] | None = None) -> None:
        self.tickets: dict[str, Ticket] = {t.ticket_id: t for t in (tickets or [])}
        self.drafts: list[PostedDraft] = []
        self.draft_submissions: int = 0  # submits that actually created a draft
        self.notes: dict[str, list[str]] = defaultdict(list)
        self.statuses: dict[str, list[str]] = defaultdict(list)
        self.calls: dict[str, int] = defaultdict(int)
        self.faults = Faults()
        self.started = False
        self.closed = False

    def add(self, *tickets: Ticket) -> None:
        for t in tickets:
            self.tickets[t.ticket_id] = t

    def drafts_for(self, ticket_id: str) -> list[PostedDraft]:
        return [d for d in self.drafts if d.ticket_id == ticket_id]

    async def start(self) -> None:
        self.calls["start"] += 1
        await self.faults.apply("start")
        self.started = True

    async def close(self) -> None:
        self.calls["close"] += 1
        self.closed = True

    async def list_open_tickets(self) -> list[TicketSummary]:
        self.calls["list_open_tickets"] += 1
        await self.faults.apply("list_open_tickets")
        return [
            TicketSummary(**t.model_dump(include=set(TicketSummary.model_fields)))
            for t in self.tickets.values()
            if t.status == "open"
        ]

    async def get_ticket(self, ticket_id: str) -> Ticket:
        self.calls["get_ticket"] += 1
        await self.faults.apply("get_ticket")
        if ticket_id not in self.tickets:
            raise PortalError(f"ticket {ticket_id} not found")
        t = self.tickets[ticket_id]
        return t.model_copy(
            update={
                "drafts": [d.body for d in self.drafts_for(ticket_id)],
                "notes": list(self.notes[ticket_id]),
            }
        )

    async def post_draft(self, ticket_id: str, body: str, *, idempotency_key: str) -> None:
        self.calls["post_draft"] += 1
        await self.faults.apply("post_draft")
        if ticket_id not in self.tickets:
            raise PortalError(f"ticket {ticket_id} not found")
        if any(d.idempotency_key == idempotency_key for d in self.drafts):
            return
        self.drafts.append(PostedDraft(ticket_id, body, idempotency_key))
        self.draft_submissions += 1

    async def post_internal_note(self, ticket_id: str, note: str) -> None:
        self.calls["post_internal_note"] += 1
        await self.faults.apply("post_internal_note")
        if ticket_id not in self.tickets:
            raise PortalError(f"ticket {ticket_id} not found")
        self.notes[ticket_id].append(note)

    async def set_status(self, ticket_id: str, status: str) -> None:
        self.calls["set_status"] += 1
        await self.faults.apply("set_status")
        if ticket_id not in self.tickets:
            raise PortalError(f"ticket {ticket_id} not found")
        self.statuses[ticket_id].append(status)
        self.tickets[ticket_id] = self.tickets[ticket_id].model_copy(update={"status": status})


# --------------------------------------------------------------------------- UserInfoClient


class FakeUserInfoClient:
    def __init__(self, users: list[UserInfo] | None = None) -> None:
        self.users: dict[str, UserInfo] = {u.email.lower(): u for u in (users or [])}
        self.calls: list[str] = []
        self.faults = Faults()
        self.closed = False

    def add(self, *users: UserInfo) -> None:
        for u in users:
            self.users[u.email.lower()] = u

    async def get_user_by_email(self, email: str) -> UserInfo | None:
        self.calls.append(email)
        await self.faults.apply("get_user_by_email")
        return self.users.get(email.lower())

    async def close(self) -> None:
        self.closed = True


# --------------------------------------------------------------------------- LLMClient


class FakeLLMClient:
    """Returns `default` (a clean auto-approvable result) unless overridden per ticket_id.

    `responses[ticket_id]` may be an LLMResult, an LLMAnalysis, or an exception to raise.
    """

    def __init__(self, default: LLMResult | None = None, *, model: str = "fake-llm") -> None:
        self.model = model
        self.default = default
        self.responses: dict[str, LLMResult | LLMAnalysis | BaseException] = {}
        self.calls: list[str] = []
        self.faults = Faults()
        self.concurrent = 0
        self.max_concurrent = 0

    def respond(self, ticket_id: str, response: LLMResult | LLMAnalysis | BaseException) -> None:
        self.responses[ticket_id] = response

    async def analyze(self, ticket: Ticket, user: UserInfo | None) -> LLMResult:
        self.calls.append(ticket.ticket_id)
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            await self.faults.apply("analyze")
            r = self.responses.get(ticket.ticket_id)
            if isinstance(r, BaseException):
                raise r
            if isinstance(r, LLMAnalysis):
                return make_result(r, model=self.model)
            if isinstance(r, LLMResult):
                return r
            return self.default or make_result(model=self.model)
        finally:
            self.concurrent -= 1


def retryable_llm_error(msg: str = "overloaded") -> LLMError:
    return LLMError(msg, retryable=True)


def fatal_llm_error(msg: str = "bad request") -> LLMError:
    return LLMError(msg, retryable=False)


# --------------------------------------------------------------------------- PolicyGate


@dataclass
class FakePolicyGate:
    """auto_approve by default; per-ticket decisions via `decisions`, or a custom `fn`."""

    decisions: dict[str, PolicyDecision] = field(default_factory=dict)
    fn: Callable[[Ticket, LLMResult, UserInfo | None], PolicyDecision] | None = None
    calls: list[str] = field(default_factory=list)

    def review(self, ticket_id: str, *reasons: str, priority: str = "normal") -> None:
        self.decisions[ticket_id] = PolicyDecision(
            outcome="human_review", reasons=list(reasons) or ["TEST"], priority=priority
        )

    def evaluate(self, ticket: Ticket, result: LLMResult, user: UserInfo | None) -> PolicyDecision:
        self.calls.append(ticket.ticket_id)
        if self.fn is not None:
            return self.fn(ticket, result, user)
        if ticket.ticket_id in self.decisions:
            return self.decisions[ticket.ticket_id]
        if result.analysis is None:
            return PolicyDecision(outcome="human_review", reasons=["LLM_NO_ANALYSIS"])
        return PolicyDecision(outcome="auto_approve", reasons=[])


# --------------------------------------------------------------------------- seed tickets (A2/A3)

SEED_FILE = Path(__file__).resolve().parents[1] / "src/support_agent/mocks/ticket_ui/seed_tickets.json"
POLICY_FILE = Path(__file__).resolve().parents[1] / "config/policy.yaml"


@dataclass
class SeedCase:
    ticket: Ticket
    user: UserInfo | None
    profile: str
    expected_rules: list[str]
    expected_outcome: str


def user_for_profile(email: str, profile: str) -> UserInfo | None:
    """The user record a seed ticket's `meta.customer_profile` implies (independent of the A4 service)."""
    if profile == "unknown":
        return None
    overrides: dict[str, Any] = {
        "standard": {},
        "enterprise": {"plan": "enterprise"},
        "vip": {"vip": True},
        "suspended": {"account_status": "suspended"},
        "closed": {"account_status": "closed"},
    }[profile]
    return make_user(email, **overrides)


def seed_case_from_raw(raw: dict[str, Any]) -> SeedCase:
    """A seed / generated ticket dict (with `meta`) -> SeedCase."""
    ticket = Ticket(
        ticket_id=raw["ticket_id"],
        subject=raw["subject"],
        customer_email=raw["customer_email"],
        status=raw.get("status", "open"),
        created_at=raw["created_at"],
        body=raw["body"],
        priority=raw.get("priority"),
        messages=raw.get("messages", []),
        drafts=[d["body"] for d in raw.get("drafts", [])],
        notes=[n["body"] for n in raw.get("notes", [])],
    )
    meta = raw["meta"]
    return SeedCase(
        ticket,
        user_for_profile(ticket.customer_email, meta["customer_profile"]),
        meta["customer_profile"],
        list(meta["expected_rules"]),
        meta["expected_outcome"],
    )


def load_seed_cases() -> list[SeedCase]:
    return [seed_case_from_raw(raw) for raw in json.loads(SEED_FILE.read_text())]
