"""Deterministic offline LLM provider (docs/PLAN.md §5.9, `LLM_PROVIDER=mock`).

Keyword rules over the ticket subject, body and customer-authored messages produce a schema-valid
`LLMAnalysis` with a polite templated draft. The same input always gives the same `LLMResult`
(analysis, usage, stop_reason, raw_text and latency_ms — latency_ms reports the configured mock latency;
the real sleep is what the latency histogram observes). Emits the same metrics and `llm.analyze` span
as the real client.

Test markers (anywhere in subject/body/messages):
  [mock:refusal]      -> analysis=None, stop_reason="refusal"             (LLM_CALLS{result=refusal}, LLM_REFUSALS)
  [mock:no-analysis]  -> analysis=None, stop_reason="max_tokens"          (LLM_CALLS{result=parse_error})
  [mock:unsafe-draft] -> confident, unflagged analysis whose draft makes a guarantee (trips UNSAFE_DRAFT)
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import zlib

from support_agent.common.config import Settings
from support_agent.common.models import LLMAnalysis, LLMResult, LLMUsage, Ticket, UserInfo
from support_agent.observability.tracing import span

from .instrument import (
    RESULT_OK,
    RESULT_PARSE_ERROR,
    RESULT_REFUSAL,
    current_trace_id,
    record_llm_metrics,
    usage_span_attrs,
)
from .prompts import COMPANY_NAME, SYSTEM_PROMPT, build_user_message

MOCK_MODEL = "mock-llm"

MARKER_REFUSAL = "[mock:refusal]"
MARKER_NO_ANALYSIS = "[mock:no-analysis]"
MARKER_UNSAFE_DRAFT = "[mock:unsafe-draft]"
_MARKER_RE = re.compile(r"\[mock:[a-z-]+\]", re.I)


def _kw(*words: str) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b", re.I)


SECURITY_RE = _kw(
    "breach", "breached", "hacked", "hack", "compromised", "unauthorized access", "unauthorised access",
    "password leak", "leaked", "leak", "phishing", "2fa bypass", "account takeover", "stolen",
)  # fmt: skip
LEGAL_RE = _kw(
    "lawyer", "lawyers", "attorney", "lawsuit", "sue", "suing", "legal action", "court", "gdpr",
    "data subject request", "subpoena", "small claims", "legal team",
)  # fmt: skip
REFUND_RE = _kw("refund", "refunded", "chargeback", "money back", "reimburse", "reimbursement")
CRITICAL_RE = _kw(
    "outage", "emergency", "down for everyone", "effectively down", "is down", "critical",
    "top priority", "nobody can log in", "none of the",
)  # fmt: skip
HIGH_URGENCY_RE = _kw("urgent", "asap", "immediately", "right away", "today", "deadline", "by friday")
ANGRY_RE = _kw(
    "worst", "ridiculous", "furious", "unacceptable", "disgraceful", "angry", "sick of", "kidding me",
    "unbelievable", "pathetic", "useless", "a joke", "scam", "incompetent",
)  # fmt: skip
FRUSTRATED_RE = _kw(
    "still", "again", "twice", "three times", "nobody", "never arrived", "no email has arrived",
    "not working", "doesn't work", "doesnt work", "greyed out", "broken", "by mistake", "accidentally",
)  # fmt: skip
POSITIVE_RE = _kw("love", "loving", "fantastic", "amazing", "awesome", "great work", "keep up")
BILLING_RE = _kw(
    "invoice", "invoices", "billing", "billed", "charge", "charged", "payment", "card", "subscription",
    "price", "pricing", "renewal", "discount", "vat", "eur", "upgrade", "upgrading",
)  # fmt: skip
ACCOUNT_RE = _kw(
    "password", "log in", "login", "email address", "suspended", "reopen", "sso", "saml", "closed my",
    "audit log", "invite", "members",
)  # fmt: skip
FEEDBACK_RE = _kw("suggestion", "idea", "feature request", "loving", "love", "fantastic")
VAGUE_RE = _kw(
    "the thing", "any update", "following up", "doesnt work", "doesn't work", "not working", "help pls", "help",
)  # fmt: skip
# Acronyms that are not shouting.
_ACRONYMS = {
    "CSV", "API", "REST", "SAML", "SSO", "ACS", "GDPR", "HTTP", "DPA", "EUR", "USD", "VAT", "URL", "ID",
    "PDF", "UTC", "IT", "MB", "OK", "UI", "JIT", "URGENT", "ASAP",
}  # fmt: skip
_SHOUT_RE = re.compile(r"\b[A-Z]{4,}\b")
_AMOUNT_RE = re.compile(r"\$\s?(\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)")


def customer_text(ticket: Ticket) -> str:
    """Subject + body + customer-authored messages (support replies excluded)."""
    parts = [ticket.subject, ticket.body]
    for msg in ticket.messages:
        if msg.author.lower() not in {"agent", "support"}:
            parts.append(msg.body)
    return "\n".join(parts)


def _amounts(text: str) -> list[float]:
    return [float(a.replace(",", "")) for a in _AMOUNT_RE.findall(text)]


def _first_name(user: UserInfo | None) -> str | None:
    if user is None or not user.name.strip():
        return None
    return user.name.split()[0]


def _jitter(ticket_id: str) -> float:
    """Deterministic 0.00-0.04 variation so confidences don't all look identical."""
    return (zlib.crc32(ticket_id.encode()) % 5) / 100


def _draft(category: str, sentiment: str, urgency: str, user: UserInfo | None, amount: float | None) -> str:
    name = _first_name(user)
    lines = [f"Hi {name}," if name else "Hi there,", ""]
    if sentiment in ("angry", "frustrated"):
        opener = "I'm sorry for the frustration this has caused, and thank you for your patience."
    else:
        opener = "Thanks for reaching out to us."
    body = {
        "refund": (
            "I've passed your refund request"
            + (f" for ${amount:,.2f}" if amount is not None else "")
            + " to our billing team, who review every refund and credit request. They will follow up"
            " with you directly once they have looked into the charge."
        ),
        "legal": (
            "I've shared your message with our legal and privacy team, who handle requests like this one."
            " They will follow up with you directly."
        ),
        "security": (
            "I've escalated this to our security team. In the meantime, please change your password and turn"
            " on two-factor authentication, and revoke any API tokens you don't recognise."
        ),
        "billing": (
            "Our billing team can help with this. I've added the details of your request to your ticket so"
            " they have everything they need, and they will follow up shortly."
        ),
        "account": (
            "I've noted the details of your account question, and our team will check your account settings"
            " and follow up with the next steps."
        ),
        "feedback": (
            "We really appreciate you taking the time to share this. I've passed your suggestion on to our"
            " product team."
        ),
        "technical": (
            "I've noted the details of what you're seeing, and our support team will look into it and follow"
            " up with step-by-step guidance."
        ),
        "other": (
            "Could you share a little more detail about what you're trying to do, and what you see when it"
            " doesn't work? That will help us point you in the right direction quickly."
        ),
    }[category]
    lines.append(f"{opener} {body}")
    if urgency == "critical":
        lines += ["", "For live updates on any incident, please check status.acme.example."]
    lines += ["", "Best regards,", f"The {COMPANY_NAME} Support Team"]
    return "\n".join(lines)


def analyze_text(ticket: Ticket, user: UserInfo | None) -> LLMAnalysis:
    """Pure keyword analysis (no markers handled here except UNSAFE_DRAFT, see `MockLLMClient`)."""
    raw = customer_text(ticket)
    text = _MARKER_RE.sub(" ", raw)
    signals: list[str] = []

    security = bool(SECURITY_RE.search(text))
    legal = bool(LEGAL_RE.search(text))
    refund = bool(REFUND_RE.search(text))
    critical = bool(CRITICAL_RE.search(text))
    shouting = [w for w in _SHOUT_RE.findall(text) if w not in _ACRONYMS]
    angry = bool(ANGRY_RE.search(text)) or "!!!" in text or len(shouting) >= 2
    for flag, name in ((security, "security"), (legal, "legal"), (refund, "refund"), (critical, "critical")):
        if flag:
            signals.append(name)

    if security:
        category = "security"
    elif legal:
        category = "legal"
    elif refund:
        category = "refund"
    elif critical:
        category = "technical"
    elif FEEDBACK_RE.search(text) and POSITIVE_RE.search(text):
        category = "feedback"
    elif BILLING_RE.search(text):
        category = "billing"
    elif ACCOUNT_RE.search(text):
        category = "account"
    else:
        category = "technical"

    if angry:
        sentiment = "angry"
        signals.append("angry")
    elif FRUSTRATED_RE.search(text):
        sentiment = "frustrated"
    elif POSITIVE_RE.search(text):
        sentiment = "positive"
    else:
        sentiment = "neutral"

    if critical:
        urgency = "critical"
    elif security or legal or angry or HIGH_URGENCY_RE.search(text):
        urgency = "high"
    elif refund or category == "billing" or sentiment == "frustrated":
        urgency = "medium"
    else:
        urgency = "low"

    amount: float | None = None
    if refund:
        found = _amounts(text)
        amount = max(found) if found else None

    words = len(re.findall(r"\w+", f"{ticket.subject} {text}"))
    vague = words < 14 or (words < 25 and bool(VAGUE_RE.search(text)))
    if vague:
        confidence = 0.4
        category = "other"
        signals.append("vague")
    elif category in ("refund", "legal", "security"):
        confidence = 0.84 + _jitter(ticket.ticket_id)
    else:
        confidence = 0.88 + _jitter(ticket.ticket_id)
    if user is None and not vague:
        confidence -= 0.05
        signals.append("no_user_record")
    confidence = round(min(1.0, max(0.0, confidence)), 2)

    actions: list[str] = []
    if refund:
        actions.append("Escalate to billing for refund review")
    if legal:
        actions.append("Forward to legal/privacy team")
    if security:
        actions.append("Escalate to security team; verify account ownership")
    if critical:
        actions.append("Check for an active incident and page on-call")
    if vague:
        actions.append("Ask the customer for more detail")
    if user is None:
        actions.append("Verify the customer's identity (no account record)")

    subject = ticket.subject.strip().rstrip("?.!") or "their request"
    return LLMAnalysis(
        category=category,
        intent=f"Customer needs help with: {subject}",
        sentiment=sentiment,
        urgency=urgency,
        confidence=confidence,
        refund_requested=refund,
        refund_amount=amount,
        legal_flag=legal,
        security_flag=security,
        draft_reply=_draft(category, sentiment, urgency, user, amount),
        reasoning_summary=(
            f"Mock keyword analysis. Signals: {', '.join(signals) or 'none'}. "
            f"Category {category}, sentiment {sentiment}, urgency {urgency}."
        ),
        suggested_actions=actions[:4],
    )


def _unsafe(analysis: LLMAnalysis, ticket: Ticket) -> LLMAnalysis:
    """A confident, unflagged analysis whose draft makes a forbidden promise."""
    return analysis.model_copy(
        update={
            "confidence": round(0.9 + _jitter(ticket.ticket_id), 2),
            "sentiment": "neutral",
            "urgency": "low",
            "refund_requested": False,
            "refund_amount": None,
            "legal_flag": False,
            "security_flag": False,
            "draft_reply": analysis.draft_reply.replace(
                "Best regards,",
                "We guarantee this will be sorted out today.\n\nBest regards,",
            ),
        }
    )


class MockLLMClient:
    """`LLMClient` that never leaves the process."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = MOCK_MODEL
        self._latency_s = max(0, settings.mock_llm_latency_ms) / 1000

    async def close(self) -> None:
        return None

    async def analyze(self, ticket: Ticket, user: UserInfo | None) -> LLMResult:
        async with span(
            "llm.analyze",
            trace_id=current_trace_id(),
            ticket_id=ticket.ticket_id,
            model=self.model,
            provider="mock",
        ) as sp:
            t0 = time.perf_counter()
            if self._latency_s:
                await asyncio.sleep(self._latency_s)
            text = customer_text(ticket).lower()
            input_tokens = (len(SYSTEM_PROMPT) + len(build_user_message(ticket, user))) // 4

            analysis: LLMAnalysis | None
            if MARKER_REFUSAL in text:
                analysis, stop_reason, result, raw_text = None, "refusal", RESULT_REFUSAL, None
                output_tokens = 0
            else:
                analysis = analyze_text(ticket, user)
                if MARKER_UNSAFE_DRAFT in text:
                    analysis = _unsafe(analysis, ticket)
                raw_text = json.dumps(analysis.model_dump(mode="json"), sort_keys=True)
                output_tokens = max(1, len(raw_text) // 4)
                stop_reason, result = "end_turn", RESULT_OK
                if MARKER_NO_ANALYSIS in text:
                    raw_text = raw_text[: len(raw_text) // 2]  # truncated JSON
                    output_tokens = max(1, len(raw_text) // 4)
                    analysis, stop_reason, result = None, "max_tokens", RESULT_PARSE_ERROR

            usage = LLMUsage(input_tokens=input_tokens, output_tokens=output_tokens)
            record_llm_metrics(self.model, result, time.perf_counter() - t0, usage)
            sp.set(result=result, stop_reason=stop_reason, served_model=self.model, **usage_span_attrs(usage))
            return LLMResult(
                analysis=analysis,
                usage=usage,
                latency_ms=float(self.settings.mock_llm_latency_ms),
                model=self.model,
                stop_reason=stop_reason,
                raw_text=raw_text,
            )
