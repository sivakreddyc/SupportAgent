"""Shared data models. Frozen contract — changes go through 'Contract change requests' in docs/TASKS.md."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from typing import Literal

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_trace_id() -> str:
    return uuid.uuid4().hex


# ---------- ticket UI ----------


class TicketSummary(BaseModel):
    """One row of the ticket list page."""

    ticket_id: str
    subject: str
    customer_email: str
    status: str  # as shown on the UI: open | pending | escalated | closed
    created_at: datetime


class TicketMessage(BaseModel):
    author: str  # customer email, or "agent"/"support"
    body: str
    at: datetime


class Ticket(TicketSummary):
    """Ticket detail page."""

    body: str
    messages: list[TicketMessage] = Field(default_factory=list)
    priority: str | None = None
    drafts: list[str] = Field(default_factory=list)  # draft bodies already on the ticket
    notes: list[str] = Field(default_factory=list)  # internal notes already on the ticket


# ---------- user info service ----------


class Order(BaseModel):
    order_id: str
    amount: float
    currency: str = "USD"
    status: Literal["pending", "paid", "shipped", "delivered", "refunded", "cancelled"]
    created_at: datetime


class UserInfo(BaseModel):
    user_id: str
    email: str
    name: str
    plan: Literal["free", "pro", "enterprise"]
    account_status: Literal["active", "suspended", "closed"]
    signup_date: date
    lifetime_value: float
    open_orders: list[Order] = Field(default_factory=list)
    recent_refunds: int = 0
    vip: bool = False


# ---------- LLM ----------

Category = Literal["billing", "refund", "technical", "account", "legal", "security", "feedback", "other"]
Sentiment = Literal["positive", "neutral", "frustrated", "angry"]
Urgency = Literal["low", "medium", "high", "critical"]


class LLMAnalysis(BaseModel):
    """The structured JSON the LLM must return."""

    category: Category
    intent: str
    sentiment: Sentiment
    urgency: Urgency
    confidence: float = Field(ge=0.0, le=1.0)
    refund_requested: bool
    refund_amount: float | None = None
    legal_flag: bool
    security_flag: bool
    draft_reply: str
    reasoning_summary: str
    suggested_actions: list[str] = Field(default_factory=list)


class LLMUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


class LLMResult(BaseModel):
    analysis: LLMAnalysis | None  # None when the model refused or output could not be parsed
    usage: LLMUsage = Field(default_factory=LLMUsage)
    latency_ms: float
    model: str
    stop_reason: str | None = None  # "end_turn" | "max_tokens" | "refusal" | ...
    raw_text: str | None = None


# ---------- policy ----------

PolicyOutcome = Literal["auto_approve", "human_review", "reject"]


class PolicyDecision(BaseModel):
    outcome: PolicyOutcome
    reasons: list[str] = Field(default_factory=list)  # rule ids that fired, e.g. ["LOW_CONFIDENCE", "ANGRY"]
    priority: Literal["normal", "high"] = "normal"


# ---------- queue ----------


class WorkItem(BaseModel):
    ticket_id: str
    trace_id: str
    enqueued_at: datetime = Field(default_factory=utcnow)
    attempt: int = 0
