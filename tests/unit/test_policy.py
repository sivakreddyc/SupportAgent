"""RuleBasedPolicyGate (PLAN §5.10 + LLM_NO_ANALYSIS): each rule, boundaries, keywords on raw text,
combinations, priority, YAML toggles/thresholds, clean path, and the seed-ticket cross-check.

Skips at module level until C7 lands.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from support_agent.common.contracts import PolicyGate
from support_agent.common.models import PolicyDecision, TicketMessage
from tests.conftest import require
from tests.fakes import POLICY_FILE, load_seed_cases, make_analysis, make_result, make_ticket, make_user

RuleBasedPolicyGate = require("support_agent.policy.gate", "RuleBasedPolicyGate")

ALL_RULES = {
    "LOW_CONFIDENCE",
    "REFUND",
    "REFUND_HIGH_VALUE",
    "LEGAL",
    "SECURITY",
    "ANGRY",
    "CRITICAL_URGENCY",
    "VIP",
    "UNKNOWN_USER",
    "ACCOUNT_NOT_ACTIVE",
    "UNSAFE_DRAFT",
    "LLM_REFUSAL",
    "LLM_NO_ANALYSIS",
}
HIGH_PRIORITY = {"REFUND_HIGH_VALUE", "LEGAL", "SECURITY", "CRITICAL_URGENCY", "UNSAFE_DRAFT"}  # policy.yaml


@pytest.fixture(scope="module")
def gate():
    return RuleBasedPolicyGate.from_file(POLICY_FILE)


def evaluate(gate, *, ticket=None, user="default", result=None, **analysis) -> PolicyDecision:
    ticket = ticket or make_ticket()
    if user == "default":
        user = make_user(ticket.customer_email)
    result = result or make_result(make_analysis(**analysis))
    d = gate.evaluate(ticket, result, user)
    assert isinstance(d, PolicyDecision)
    assert len(d.reasons) == len(set(d.reasons)), "duplicate rule ids"
    assert set(d.reasons) <= ALL_RULES
    return d


def gate_from(tmp_path: Path, mutate) -> object:
    cfg = yaml.safe_load(POLICY_FILE.read_text())
    mutate(cfg)
    p = tmp_path / "policy.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return RuleBasedPolicyGate.from_file(p)


# --------------------------------------------------------------------------- basics


def test_satisfies_protocol(gate):
    assert isinstance(gate, PolicyGate)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.update(confidence_threshold="high"),  # BUG-3 regression (fixed)
        lambda c: c["unsafe_draft_patterns"].append("(unclosed"),
        lambda c: c.update(rules="not-a-mapping"),
    ],
    ids=["bad-threshold", "bad-regex", "bad-rules"],
)
def test_bad_yaml_raises_policy_config_error(tmp_path, mutate):
    PolicyConfigError = require("support_agent.policy.gate", "PolicyConfigError")
    with pytest.raises(PolicyConfigError):
        gate_from(tmp_path, mutate)


def test_missing_policy_file_raises(tmp_path):
    with pytest.raises((FileNotFoundError, OSError, Exception)):
        RuleBasedPolicyGate.from_file(tmp_path / "nope.yaml")


def test_allowed_draft_email_domains(gate):
    """policy.yaml `allowed_draft_email_domains`: our own support addresses are not someone else's PII."""
    d = evaluate(gate, draft_reply="Reply to support@acme.example if you need anything else.")
    assert "UNSAFE_DRAFT" not in d.reasons


def test_from_file_accepts_str_path():
    assert RuleBasedPolicyGate.from_file(str(POLICY_FILE)) is not None


def test_clean_path_auto_approves(gate):
    d = evaluate(gate)
    assert d.outcome == "auto_approve"
    assert d.reasons == []
    assert d.priority == "normal"


def test_evaluate_is_deterministic(gate):
    t, u, r = make_ticket(body="I want a refund"), make_user(), make_result(make_analysis(sentiment="angry"))
    assert gate.evaluate(t, r, u) == gate.evaluate(t, r, u)


# --------------------------------------------------------------------------- each rule alone

SINGLE_RULE_CASES = {
    "LOW_CONFIDENCE": dict(confidence=0.5),
    "REFUND": dict(refund_requested=True),
    "LEGAL": dict(legal_flag=True),
    "SECURITY": dict(security_flag=True),
    "ANGRY": dict(sentiment="angry"),
    "CRITICAL_URGENCY": dict(urgency="critical"),
}


@pytest.mark.parametrize("rule,analysis", list(SINGLE_RULE_CASES.items()), ids=list(SINGLE_RULE_CASES))
def test_analysis_rule_fires_alone(gate, rule, analysis):
    d = evaluate(gate, **analysis)
    assert d.outcome == "human_review"
    assert d.reasons == [rule]
    assert d.priority == ("high" if rule in HIGH_PRIORITY else "normal")


@pytest.mark.parametrize(
    "category,rule", [("refund", "REFUND"), ("legal", "LEGAL"), ("security", "SECURITY")]
)
def test_category_alone_fires_rule(gate, category, rule):
    d = evaluate(gate, category=category)
    assert d.outcome == "human_review" and rule in d.reasons


@pytest.mark.parametrize(
    "c,fires", [(0.0, True), (0.5, True), (0.7499, True), (0.75, False), (0.76, False), (1.0, False)]
)
def test_low_confidence_boundary(gate, c, fires):
    assert ("LOW_CONFIDENCE" in evaluate(gate, confidence=c).reasons) is fires


@pytest.mark.parametrize(
    "amount,fires", [(None, False), (50, False), (100, False), (100.01, True), (5000, True)]
)
def test_refund_high_value_boundary(gate, amount, fires):
    d = evaluate(gate, refund_requested=True, refund_amount=amount)
    assert "REFUND" in d.reasons
    assert ("REFUND_HIGH_VALUE" in d.reasons) is fires
    assert d.priority == ("high" if fires else "normal")


def test_refund_high_value_without_flag(gate):
    """Table: `refund_amount > 100` alone is enough."""
    assert "REFUND_HIGH_VALUE" in evaluate(gate, refund_amount=250).reasons


@pytest.mark.parametrize("sentiment", ["positive", "neutral", "frustrated"])
def test_not_angry(gate, sentiment):
    assert "ANGRY" not in evaluate(gate, sentiment=sentiment).reasons


@pytest.mark.parametrize("urgency", ["low", "medium", "high"])
def test_not_critical(gate, urgency):
    assert "CRITICAL_URGENCY" not in evaluate(gate, urgency=urgency).reasons


@pytest.mark.parametrize(
    "user_kw,rule",
    [
        ({"vip": True}, "VIP"),
        ({"plan": "enterprise"}, "VIP"),
        ({"account_status": "suspended"}, "ACCOUNT_NOT_ACTIVE"),
        ({"account_status": "closed"}, "ACCOUNT_NOT_ACTIVE"),
    ],
)
def test_user_rules(gate, user_kw, rule):
    t = make_ticket()
    d = evaluate(gate, ticket=t, user=make_user(t.customer_email, **user_kw))
    assert d.outcome == "human_review" and d.reasons == [rule]
    assert d.priority == "normal"


@pytest.mark.parametrize("plan", ["free", "pro"])
def test_non_enterprise_non_vip_passes(gate, plan):
    t = make_ticket()
    assert evaluate(gate, ticket=t, user=make_user(t.customer_email, plan=plan)).outcome == "auto_approve"


def test_unknown_user(gate):
    d = evaluate(gate, user=None)
    assert d.outcome == "human_review" and d.reasons == ["UNKNOWN_USER"]


@pytest.mark.parametrize(
    "draft",
    [
        "Your card 4111 1111 1111 1111 is on file.",
        "Thanks! Card 4111-1111-1111-1111 was charged.",
        "We will refund you in full by Friday.",
        "We have refunded the charge.",
        "I guarantee this will never happen again.",
        "This email is legally binding.",
    ],
)
def test_unsafe_draft(gate, draft):
    d = evaluate(gate, draft_reply=draft)
    assert d.outcome == "human_review" and d.reasons == ["UNSAFE_DRAFT"]
    assert d.priority == "high"


def test_unsafe_draft_other_users_pii(gate):
    """PLAN §5.10: draft containing PII of *other* users (e.g. someone else's email) is unsafe."""
    t = make_ticket(customer_email="alice@example.com")
    d = evaluate(gate, ticket=t, draft_reply="Hi Alice, I checked with bob.smith@example.org and he says hi.")
    assert "UNSAFE_DRAFT" in d.reasons


def test_safe_draft_mentioning_own_email(gate):
    t = make_ticket(customer_email="alice@example.com")
    d = evaluate(gate, ticket=t, draft_reply="Hi Alice, we sent the export to alice@example.com.")
    assert "UNSAFE_DRAFT" not in d.reasons


def test_safe_draft_with_short_numbers(gate):
    d = evaluate(gate, draft_reply="Order 12345 shipped on 2026-10-12; call us at 555-0100.")
    assert d.outcome == "auto_approve"


def test_llm_refusal(gate):
    d = evaluate(gate, result=make_result(no_analysis=True, stop_reason="refusal"))
    assert d.outcome == "human_review"
    assert "LLM_REFUSAL" in d.reasons


def test_llm_refusal_with_analysis_still_gated(gate):
    d = evaluate(gate, result=make_result(make_analysis(), stop_reason="refusal"))
    assert "LLM_REFUSAL" in d.reasons


@pytest.mark.parametrize("stop", ["end_turn", "max_tokens", None])
def test_llm_no_analysis(gate, stop):
    d = evaluate(gate, result=make_result(no_analysis=True, stop_reason=stop))
    assert d.outcome == "human_review"
    assert "LLM_NO_ANALYSIS" in d.reasons
    assert "LLM_REFUSAL" not in d.reasons


def test_no_analysis_still_applies_user_and_keyword_rules(gate):
    t = make_ticket(body="I will sue you, my lawyer is ready.")
    d = evaluate(gate, ticket=t, user=None, result=make_result(no_analysis=True))
    assert {"LLM_NO_ANALYSIS", "UNKNOWN_USER", "LEGAL"} <= set(d.reasons)
    assert d.priority == "high"


# --------------------------------------------------------------------------- keywords on raw text


@pytest.mark.parametrize(
    "field,text,rule",
    [
        ("body", "If this isn't fixed I'm calling my lawyer.", "LEGAL"),
        ("body", "We are preparing a LAWSUIT.", "LEGAL"),
        ("body", "I will take legal action.", "LEGAL"),
        ("body", "This is a GDPR request for all my data.", "LEGAL"),
        ("subject", "Going to sue", "LEGAL"),
        ("body", "My account was hacked yesterday.", "SECURITY"),
        ("body", "Looks like a password leak on your side.", "SECURITY"),
        ("body", "Was there a data breach?", "SECURITY"),
        ("subject", "Phishing email pretending to be you", "SECURITY"),
        ("body", "I want my money back.", "REFUND"),
        ("body", "I'm filing a chargeback.", "REFUND"),
    ],
)
def test_keyword_on_raw_text_fires_even_if_llm_misses(gate, field, text, rule):
    t = make_ticket(**{field: text})
    d = evaluate(gate, ticket=t)  # analysis is clean: no flags, neutral, confident
    assert rule in d.reasons, d.reasons
    assert d.outcome == "human_review"


def test_keyword_in_customer_message_thread(gate):
    t = make_ticket(body="Follow-up on my ticket.")
    t = t.model_copy(
        update={
            "messages": [TicketMessage(author=t.customer_email, body="my lawyer says hi", at=t.created_at)]
        }
    )
    assert "LEGAL" in evaluate(gate, ticket=t).reasons


@pytest.mark.parametrize(
    "text",
    [
        "The invoice was issued yesterday, thanks.",  # 'sue' inside 'issued'
        "Thanks for the great hackathon swag!",  # 'hack' inside 'hackathon'
        "I'd like to pursue the premium plan.",  # 'sue' inside 'pursue'
        "The courtesy call was nice.",  # 'court' inside 'courtesy'
    ],
)
def test_keywords_respect_word_boundaries(gate, text):
    d = evaluate(gate, ticket=make_ticket(body=text))
    assert d.outcome == "auto_approve", d.reasons


# --------------------------------------------------------------------------- combinations + priority


def test_combo_all_reasons_recorded(gate):
    t = make_ticket(body="Refund me now or I sue!")
    d = evaluate(
        gate,
        ticket=t,
        user=make_user(t.customer_email, vip=True),
        sentiment="angry",
        refund_requested=True,
        refund_amount=300,
        confidence=0.4,
    )
    assert {"ANGRY", "REFUND", "REFUND_HIGH_VALUE", "LEGAL", "VIP", "LOW_CONFIDENCE"} <= set(d.reasons)
    assert d.outcome == "human_review" and d.priority == "high"


def test_combo_normal_priority_rules_stay_normal(gate):
    t = make_ticket()
    d = evaluate(
        gate, ticket=t, user=make_user(t.customer_email, plan="enterprise"), sentiment="angry", confidence=0.3
    )
    assert set(d.reasons) == {"VIP", "ANGRY", "LOW_CONFIDENCE"}
    assert d.priority == "normal"


def test_combo_unknown_user_and_security(gate):
    d = evaluate(gate, user=None, security_flag=True, urgency="critical")
    assert set(d.reasons) == {"UNKNOWN_USER", "SECURITY", "CRITICAL_URGENCY"}
    assert d.priority == "high"


def test_outcome_is_never_reject(gate):
    for kw in [{}, {"sentiment": "angry"}, {"legal_flag": True, "security_flag": True, "confidence": 0.0}]:
        assert evaluate(gate, **kw).outcome in ("auto_approve", "human_review")


# --------------------------------------------------------------------------- YAML configuration


@pytest.mark.parametrize("rule,analysis", list(SINGLE_RULE_CASES.items()), ids=list(SINGLE_RULE_CASES))
def test_disabled_rule_does_not_fire(tmp_path, rule, analysis):
    g = gate_from(tmp_path, lambda c: c["rules"][rule].update(enabled=False))
    d = evaluate(g, **analysis)
    assert rule not in d.reasons


def test_disabled_keyword_rule_ignores_raw_text(tmp_path):
    g = gate_from(tmp_path, lambda c: c["rules"]["LEGAL"].update(enabled=False))
    assert (
        evaluate(g, ticket=make_ticket(body="my lawyer will call"), legal_flag=True).outcome == "auto_approve"
    )


def test_disabled_user_rules(tmp_path):
    def off(c):
        for r in ("UNKNOWN_USER", "VIP"):
            c["rules"][r]["enabled"] = False

    g = gate_from(tmp_path, off)
    assert evaluate(g, user=None).outcome == "auto_approve"
    t = make_ticket()
    assert evaluate(g, ticket=t, user=make_user(t.customer_email, vip=True)).outcome == "auto_approve"


def test_disabled_llm_no_analysis(tmp_path):
    g = gate_from(tmp_path, lambda c: c["rules"]["LLM_NO_ANALYSIS"].update(enabled=False))
    d = evaluate(g, result=make_result(no_analysis=True))
    assert "LLM_NO_ANALYSIS" not in d.reasons


def test_confidence_threshold_from_yaml(tmp_path):
    g = gate_from(tmp_path, lambda c: c.update(confidence_threshold=0.9))
    assert "LOW_CONFIDENCE" in evaluate(g, confidence=0.85).reasons
    assert "LOW_CONFIDENCE" not in evaluate(g, confidence=0.95).reasons


def test_high_value_threshold_from_yaml(tmp_path):
    g = gate_from(tmp_path, lambda c: c.update(high_value_refund_amount=500))
    assert "REFUND_HIGH_VALUE" not in evaluate(g, refund_requested=True, refund_amount=150).reasons
    assert "REFUND_HIGH_VALUE" in evaluate(g, refund_requested=True, refund_amount=501).reasons


def test_keyword_list_from_yaml(tmp_path):
    g = gate_from(tmp_path, lambda c: c["keywords"]["legal"].append("ombudsman"))
    assert "LEGAL" in evaluate(g, ticket=make_ticket(body="I'll write to the ombudsman.")).reasons


def test_priority_from_yaml(tmp_path):
    def flip(c):
        c["rules"]["ANGRY"]["priority"] = "high"
        c["rules"]["LEGAL"].pop("priority", None)

    g = gate_from(tmp_path, flip)
    assert evaluate(g, sentiment="angry").priority == "high"
    assert evaluate(g, legal_flag=True).priority == "normal"


def test_unsafe_pattern_from_yaml(tmp_path):
    g = gate_from(tmp_path, lambda c: c["unsafe_draft_patterns"].append(r"(?i)\bfree upgrade\b"))
    assert "UNSAFE_DRAFT" in evaluate(g, draft_reply="Enjoy a free upgrade on us!").reasons


# --------------------------------------------------------------------------- seed cross-check (MockLLM + gate)

SEEDS = load_seed_cases()


@pytest.mark.parametrize("case", SEEDS, ids=[c.ticket.ticket_id for c in SEEDS])
async def test_seed_ticket_expected_rules(gate, settings, case):
    MockLLMClient = require("support_agent.llm.mock_client", "MockLLMClient")
    result = await MockLLMClient(settings).analyze(case.ticket, case.user)
    d = gate.evaluate(case.ticket, result, case.user)
    assert set(d.reasons) == set(case.expected_rules), (case.profile, d.reasons)
    assert d.outcome == case.expected_outcome
