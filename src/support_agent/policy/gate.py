"""Rule-based policy gate (docs/PLAN.md §5.10), configured by config/policy.yaml.

    gate = RuleBasedPolicyGate.from_file(settings.policy_file)
    decision = gate.evaluate(ticket, llm_result, user)

Pure and synchronous: no I/O after construction, no metrics (the worker records them).
Outcome: `human_review` if any enabled rule fires (all fired rule ids in `reasons`, in `RULES` order),
priority `high` if any fired rule has `priority: high`; otherwise `auto_approve`. Never `reject`.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import Any

import yaml

from support_agent.common.models import LLMResult, PolicyDecision, Ticket, UserInfo

from .rules import RULES, PolicyConfig, RuleContext, keyword_pattern, raw_ticket_text


class PolicyConfigError(ValueError):
    """policy.yaml is malformed."""


def _number(raw: dict[str, Any], key: str, default: float) -> float:
    value = raw.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise PolicyConfigError(f"{key} must be a number, got {value!r}") from e


class RuleBasedPolicyGate:
    def __init__(self, raw: dict[str, Any] | None = None) -> None:
        raw = raw or {}
        rules_cfg = raw.get("rules") or {}
        if not isinstance(rules_cfg, dict):
            raise PolicyConfigError("'rules' must be a mapping of rule id -> {enabled, priority}")
        unknown = set(rules_cfg) - set(RULES)
        if unknown:
            raise PolicyConfigError(f"unknown rule id(s) in policy: {sorted(unknown)}")

        try:
            patterns = tuple(re.compile(p) for p in raw.get("unsafe_draft_patterns") or [])
        except re.error as e:
            raise PolicyConfigError(f"bad unsafe_draft_patterns regex: {e}") from e

        self.config = PolicyConfig(
            confidence_threshold=_number(raw, "confidence_threshold", 0.75),
            high_value_refund_amount=_number(raw, "high_value_refund_amount", 100),
            keywords={k: keyword_pattern(list(v or [])) for k, v in (raw.get("keywords") or {}).items()},
            unsafe_draft_patterns=patterns,
            allowed_draft_email_domains=frozenset(
                str(d).lower() for d in raw.get("allowed_draft_email_domains") or []
            ),
        )
        # Rules absent from YAML are enabled with normal priority.
        self._enabled: list[tuple[str, Callable[[RuleContext], bool]]] = []
        self.priorities: dict[str, str] = {}
        for rule_id, fn in RULES.items():
            cfg = rules_cfg.get(rule_id) or {}
            prio = str(cfg.get("priority", "normal")).lower()
            if prio not in ("normal", "high"):
                raise PolicyConfigError(f"{rule_id}: priority must be normal|high, got {prio!r}")
            self.priorities[rule_id] = prio
            if cfg.get("enabled", True):
                self._enabled.append((rule_id, fn))

    @classmethod
    def from_file(cls, path: str | PathLike[str]) -> RuleBasedPolicyGate:
        data = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(data, dict):
            raise PolicyConfigError(f"{path}: top level must be a mapping")
        return cls(data)

    @property
    def enabled_rules(self) -> list[str]:
        return [r for r, _ in self._enabled]

    def evaluate(self, ticket: Ticket, result: LLMResult, user: UserInfo | None) -> PolicyDecision:
        ctx = RuleContext(
            ticket=ticket, result=result, user=user, config=self.config, raw_text=raw_ticket_text(ticket)
        )
        fired = [rule_id for rule_id, fn in self._enabled if fn(ctx)]
        if not fired:
            return PolicyDecision(outcome="auto_approve", reasons=[], priority="normal")
        high = any(self.priorities[r] == "high" for r in fired)
        return PolicyDecision(outcome="human_review", reasons=fired, priority="high" if high else "normal")
