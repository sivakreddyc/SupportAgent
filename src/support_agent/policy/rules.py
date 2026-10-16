"""Policy rules (docs/PLAN.md §5.10). Each rule is a pure predicate over a `RuleContext`.

Rule order in `RULES` is the order reasons are reported in.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from support_agent.common.models import LLMAnalysis, LLMResult, Ticket, UserInfo

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def keyword_pattern(words: list[str]) -> re.Pattern[str] | None:
    """Whole-word / whole-phrase, case-insensitive. 'sue' does not match 'issued' or 'pursue'."""
    words = [str(w).strip() for w in words if str(w).strip()]
    if not words:
        return None
    alts = "|".join(re.escape(w).replace(r"\ ", r"\s+") for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"(?<![\w])(?:{alts})(?![\w])", re.IGNORECASE)


@dataclass(frozen=True)
class PolicyConfig:
    confidence_threshold: float = 0.75
    high_value_refund_amount: float = 100.0
    keywords: dict[str, re.Pattern[str] | None] = field(default_factory=dict)
    unsafe_draft_patterns: tuple[re.Pattern[str], ...] = ()
    allowed_draft_email_domains: frozenset[str] = frozenset()


@dataclass(frozen=True)
class RuleContext:
    ticket: Ticket
    result: LLMResult
    user: UserInfo | None
    config: PolicyConfig
    raw_text: str  # subject + body + all messages

    @property
    def analysis(self) -> LLMAnalysis | None:
        return self.result.analysis

    def keyword_hit(self, group: str) -> bool:
        pat = self.config.keywords.get(group)
        return bool(pat and pat.search(self.raw_text))


def raw_ticket_text(ticket: Ticket) -> str:
    return "\n".join([ticket.subject, ticket.body, *(m.body for m in ticket.messages)])


# --------------------------------------------------------------------------- predicates


def low_confidence(c: RuleContext) -> bool:
    return c.analysis is not None and c.analysis.confidence < c.config.confidence_threshold


def refund(c: RuleContext) -> bool:
    a = c.analysis
    return (a is not None and (a.refund_requested or a.category == "refund")) or c.keyword_hit("refund")


def refund_high_value(c: RuleContext) -> bool:
    a = c.analysis
    return (
        a is not None and a.refund_amount is not None and a.refund_amount > c.config.high_value_refund_amount
    )


def legal(c: RuleContext) -> bool:
    a = c.analysis
    return (a is not None and (a.legal_flag or a.category == "legal")) or c.keyword_hit("legal")


def security(c: RuleContext) -> bool:
    a = c.analysis
    return (a is not None and (a.security_flag or a.category == "security")) or c.keyword_hit("security")


def angry(c: RuleContext) -> bool:
    return c.analysis is not None and c.analysis.sentiment == "angry"


def critical_urgency(c: RuleContext) -> bool:
    return c.analysis is not None and c.analysis.urgency == "critical"


def vip(c: RuleContext) -> bool:
    return c.user is not None and (c.user.vip or c.user.plan == "enterprise")


def unknown_user(c: RuleContext) -> bool:
    return c.user is None


def account_not_active(c: RuleContext) -> bool:
    return c.user is not None and c.user.account_status != "active"


def foreign_emails(draft: str, customer_email: str, allowed_domains: frozenset[str]) -> list[str]:
    own = customer_email.strip().lower()
    out = []
    for e in EMAIL_RE.findall(draft):
        e_l = e.lower().rstrip(".")
        if e_l == own or e_l.rsplit("@", 1)[-1] in allowed_domains:
            continue
        out.append(e_l)
    return out


def unsafe_draft(c: RuleContext) -> bool:
    if c.analysis is None:
        return False
    draft = c.analysis.draft_reply
    if any(p.search(draft) for p in c.config.unsafe_draft_patterns):
        return True
    return bool(foreign_emails(draft, c.ticket.customer_email, c.config.allowed_draft_email_domains))


def llm_refusal(c: RuleContext) -> bool:
    return c.result.stop_reason == "refusal"


def llm_no_analysis(c: RuleContext) -> bool:
    return c.analysis is None and c.result.stop_reason != "refusal"


RULES: dict[str, Callable[[RuleContext], bool]] = {
    "LLM_REFUSAL": llm_refusal,
    "LLM_NO_ANALYSIS": llm_no_analysis,
    "LOW_CONFIDENCE": low_confidence,
    "REFUND": refund,
    "REFUND_HIGH_VALUE": refund_high_value,
    "LEGAL": legal,
    "SECURITY": security,
    "ANGRY": angry,
    "CRITICAL_URGENCY": critical_urgency,
    "VIP": vip,
    "UNKNOWN_USER": unknown_user,
    "ACCOUNT_NOT_ACTIVE": account_not_active,
    "UNSAFE_DRAFT": unsafe_draft,
}
