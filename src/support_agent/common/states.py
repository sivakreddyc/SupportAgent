"""Ticket state machine. Frozen contract (see docs/PLAN.md §4.3)."""

from __future__ import annotations

from enum import StrEnum


class TicketState(StrEnum):
    DISCOVERED = "DISCOVERED"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    ENRICHED = "ENRICHED"
    ANALYZED = "ANALYZED"
    POLICY_CHECKED = "POLICY_CHECKED"
    DRAFT_POSTED = "DRAFT_POSTED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    DONE = "DONE"
    FAILED = "FAILED"
    DEAD_LETTER = "DEAD_LETTER"


S = TicketState

_PIPELINE_STATES = {S.QUEUED, S.PROCESSING, S.ENRICHED, S.ANALYZED, S.POLICY_CHECKED, S.APPROVED}

ALLOWED_TRANSITIONS: dict[TicketState, frozenset[TicketState]] = {
    S.DISCOVERED: frozenset({S.QUEUED}),
    S.QUEUED: frozenset({S.PROCESSING, S.FAILED}),
    S.PROCESSING: frozenset({S.ENRICHED, S.FAILED}),
    S.ENRICHED: frozenset({S.ANALYZED, S.FAILED}),
    S.ANALYZED: frozenset({S.POLICY_CHECKED, S.FAILED}),
    S.POLICY_CHECKED: frozenset({S.DRAFT_POSTED, S.NEEDS_REVIEW, S.FAILED}),
    S.DRAFT_POSTED: frozenset({S.DONE}),
    S.NEEDS_REVIEW: frozenset({S.APPROVED, S.REJECTED}),
    S.APPROVED: frozenset({S.DRAFT_POSTED, S.FAILED}),
    S.REJECTED: frozenset({S.DONE}),
    S.FAILED: frozenset({S.QUEUED, S.DEAD_LETTER, S.APPROVED}),  # APPROVED: retry of a failed review post
    S.DONE: frozenset(),
    S.DEAD_LETTER: frozenset({S.QUEUED}),  # manual requeue from the dashboard
}

TERMINAL_STATES = frozenset({S.DONE, S.DEAD_LETTER})

# States a crashed runtime should re-enqueue on startup.
RECOVERABLE_STATES = frozenset(_PIPELINE_STATES - {S.APPROVED} | {S.FAILED})


class InvalidTransition(Exception):
    def __init__(self, ticket_id: str, from_state: TicketState | None, to_state: TicketState) -> None:
        super().__init__(f"ticket {ticket_id}: illegal transition {from_state} -> {to_state}")
        self.ticket_id = ticket_id
        self.from_state = from_state
        self.to_state = to_state


def can_transition(from_state: TicketState, to_state: TicketState) -> bool:
    return to_state in ALLOWED_TRANSITIONS[from_state]


def assert_transition(ticket_id: str, from_state: TicketState, to_state: TicketState) -> None:
    if not can_transition(from_state, to_state):
        raise InvalidTransition(ticket_id, from_state, to_state)
