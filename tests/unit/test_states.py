"""State machine (PLAN §4.3): every (from, to) pair, terminal and recoverable sets, reachability."""

from __future__ import annotations

import itertools
from collections import deque

import pytest

from support_agent.common.states import (
    ALLOWED_TRANSITIONS,
    RECOVERABLE_STATES,
    TERMINAL_STATES,
    InvalidTransition,
    TicketState,
    assert_transition,
    can_transition,
)

S = TicketState

# Hand-written from PLAN §4.3, plus the three documented extras in states.py:
#   APPROVED -> FAILED (review post failed), FAILED -> APPROVED (retry of that post),
#   DEAD_LETTER -> QUEUED (manual requeue from the dashboard).
EXPECTED: dict[TicketState, set[TicketState]] = {
    S.DISCOVERED: {S.QUEUED},
    S.QUEUED: {S.PROCESSING, S.FAILED},
    S.PROCESSING: {S.ENRICHED, S.FAILED},
    S.ENRICHED: {S.ANALYZED, S.FAILED},
    S.ANALYZED: {S.POLICY_CHECKED, S.FAILED},
    S.POLICY_CHECKED: {S.DRAFT_POSTED, S.NEEDS_REVIEW, S.FAILED},
    S.DRAFT_POSTED: {S.DONE},
    S.NEEDS_REVIEW: {S.APPROVED, S.REJECTED},
    S.APPROVED: {S.DRAFT_POSTED, S.FAILED},
    S.REJECTED: {S.DONE},
    S.FAILED: {S.QUEUED, S.DEAD_LETTER, S.APPROVED},
    S.DONE: set(),
    S.DEAD_LETTER: {S.QUEUED},
}

ALL_PAIRS = list(itertools.product(list(S), list(S)))
LEGAL = [(a, b) for a, b in ALL_PAIRS if b in EXPECTED[a]]
ILLEGAL = [(a, b) for a, b in ALL_PAIRS if b not in EXPECTED[a]]


def test_table_covers_every_state():
    assert set(ALLOWED_TRANSITIONS) == set(S)
    assert len(S) == 13


def test_table_matches_spec_exactly():
    assert {k: set(v) for k, v in ALLOWED_TRANSITIONS.items()} == EXPECTED


@pytest.mark.parametrize("src,dst", LEGAL, ids=[f"{a}->{b}" for a, b in LEGAL])
def test_legal_transition(src, dst):
    assert can_transition(src, dst)
    assert_transition("T-1", src, dst)  # does not raise


@pytest.mark.parametrize("src,dst", ILLEGAL, ids=[f"{a}->{b}" for a, b in ILLEGAL])
def test_illegal_transition(src, dst):
    assert not can_transition(src, dst)
    with pytest.raises(InvalidTransition) as ei:
        assert_transition("T-1", src, dst)
    assert ei.value.ticket_id == "T-1"
    assert ei.value.from_state == src
    assert ei.value.to_state == dst
    assert "T-1" in str(ei.value)


def test_no_self_loops():
    for s in S:
        assert not can_transition(s, s)


def test_nothing_returns_to_discovered():
    """Poller dedup relies on DISCOVERED being entry-only."""
    assert all(S.DISCOVERED not in v for v in ALLOWED_TRANSITIONS.values())


def test_every_pipeline_step_can_fail():
    for s in (S.QUEUED, S.PROCESSING, S.ENRICHED, S.ANALYZED, S.POLICY_CHECKED, S.APPROVED):
        assert can_transition(s, S.FAILED)


def test_human_states_cannot_fail_or_skip_review():
    assert not can_transition(S.NEEDS_REVIEW, S.FAILED)
    assert not can_transition(S.NEEDS_REVIEW, S.DRAFT_POSTED)  # must be APPROVED first
    assert not can_transition(S.NEEDS_REVIEW, S.DONE)


def test_terminal_states():
    assert TERMINAL_STATES == {S.DONE, S.DEAD_LETTER}
    assert ALLOWED_TRANSITIONS[S.DONE] == frozenset()
    # DEAD_LETTER is terminal for the pipeline; only a manual requeue leaves it
    assert ALLOWED_TRANSITIONS[S.DEAD_LETTER] == {S.QUEUED}


def test_recoverable_states():
    assert RECOVERABLE_STATES == {S.QUEUED, S.PROCESSING, S.ENRICHED, S.ANALYZED, S.POLICY_CHECKED, S.FAILED}
    # waiting on a human, handled by the review dispatcher, or finished: never re-enqueued
    for s in (S.DISCOVERED, S.NEEDS_REVIEW, S.APPROVED, S.DRAFT_POSTED, S.REJECTED, S.DONE, S.DEAD_LETTER):
        assert s not in RECOVERABLE_STATES
    assert not (RECOVERABLE_STATES & TERMINAL_STATES)


def test_recoverable_states_can_be_requeued():
    """Every recoverable state can get back to QUEUED (directly or via FAILED)."""
    for s in RECOVERABLE_STATES:
        assert s == S.QUEUED or can_transition(s, S.QUEUED) or can_transition(s, S.FAILED)


def _reachable(start: TicketState) -> set[TicketState]:
    seen, todo = {start}, deque([start])
    while todo:
        for n in ALLOWED_TRANSITIONS[todo.popleft()]:
            if n not in seen:
                seen.add(n)
                todo.append(n)
    return seen


def test_every_state_reachable_from_discovered():
    assert _reachable(S.DISCOVERED) == set(S)


def test_every_state_can_reach_a_terminal_state():
    for s in S:
        assert _reachable(s) & TERMINAL_STATES, s


def test_happy_and_review_paths():
    happy = [
        S.DISCOVERED,
        S.QUEUED,
        S.PROCESSING,
        S.ENRICHED,
        S.ANALYZED,
        S.POLICY_CHECKED,
        S.DRAFT_POSTED,
        S.DONE,
    ]
    review = [S.POLICY_CHECKED, S.NEEDS_REVIEW, S.APPROVED, S.DRAFT_POSTED, S.DONE]
    reject = [S.NEEDS_REVIEW, S.REJECTED, S.DONE]
    retry = [S.ENRICHED, S.FAILED, S.QUEUED, S.PROCESSING, S.FAILED, S.DEAD_LETTER, S.QUEUED]
    for path in (happy, review, reject, retry):
        for a, b in itertools.pairwise(path):
            assert can_transition(a, b), f"{a}->{b}"


def test_state_values_are_strings():
    for s in S:
        assert s == s.value and isinstance(s.value, str)
    assert TicketState("DONE") is S.DONE
