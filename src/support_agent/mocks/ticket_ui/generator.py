"""Random ticket generator for ``POST /admin/generate`` (demos and load tests).

Generated tickets are built from the seed tickets used as templates, so their text has already passed
the keyword false-positive scan and each one inherits a known ``meta`` (customer profile, expected
rules, expected outcome). The customer email is the template's, so user lookups behave exactly as they
do for the seed ticket (known user, VIP, suspended, unknown, ...).

    generate(n, seed=None, policy_share=0.2, start_number=1054, now=None) -> list[dict]

The output is raw ticket dicts, the same shape as ``seed_tickets.json``. With a ``seed`` the result is
deterministic, including timestamps if ``now`` is also given.
"""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

SEED_FILE = Path(__file__).parent / "seed_tickets.json"
DEFAULT_POLICY_SHARE = 0.2
_MARKER = "[mock:"


def _templates() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(happy-path templates, policy-triggering templates). MockLLM marker tickets are excluded."""
    seed = json.loads(SEED_FILE.read_text())
    happy = [t for t in seed if t["meta"]["expected_outcome"] == "auto_approve"]
    gated = [t for t in seed if t["meta"]["expected_outcome"] != "auto_approve" and _MARKER not in t["body"]]
    return happy, gated


def generate(
    n: int,
    seed: int | None = None,
    *,
    policy_share: float = DEFAULT_POLICY_SHARE,
    start_number: int = 1054,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Return ``n`` new tickets ``TCK-<start_number>``, ``TCK-<start_number+1>``, ...

    ``policy_share`` (0..1) is the expected fraction of tickets that should trigger a policy rule;
    the rest are happy-path how-to questions.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if not 0.0 <= policy_share <= 1.0:
        raise ValueError("policy_share must be between 0 and 1")
    rng = random.Random(seed)
    happy, gated = _templates()
    base = now or datetime.now(UTC)
    out: list[dict[str, Any]] = []
    for i in range(n):
        tpl = rng.choice(gated if rng.random() < policy_share else happy)
        number = start_number + i
        out.append(
            {
                "ticket_id": f"TCK-{number}",
                "subject": tpl["subject"],
                "customer_email": tpl["customer_email"],
                # +1 ms per ticket keeps list ordering (created_at, id) stable and unique
                "created_at": (base + timedelta(milliseconds=i)).isoformat(),
                "priority": tpl["priority"],
                "status": "open",
                "body": tpl["body"],
                "messages": [],
                "drafts": [],
                "notes": [],
                "meta": {**tpl["meta"], "generated": True, "template": tpl["ticket_id"]},
            }
        )
    return out
