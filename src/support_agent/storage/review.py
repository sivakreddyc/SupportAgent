"""Synchronous write helpers for the Streamlit dashboard (plain sqlite3, separate process from the runtime).

    apply_review(db_path, ticket_id, "approve" | "edit" | "reject", reviewer, edited_body=None)
    requeue_dead_letter(db_path, ticket_id)

Each call is one `BEGIN IMMEDIATE` transaction: the state-machine check, the ticket update, the
`ticket_transitions` rows, the draft changes and the `review_actions` row commit together or not at all.
The runtime's review dispatcher posts APPROVED drafts and enqueues requeued tickets.

Errors:
  ReviewError         invalid action, missing/empty edited_body, no draft / empty draft to approve
  TicketNotFound      (subclass of ReviewError) unknown ticket_id
  InvalidTransition   ticket not in NEEDS_REVIEW (review) or DEAD_LETTER (requeue)
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any, Literal

from support_agent.common.states import TERMINAL_STATES, InvalidTransition, TicketState, assert_transition

from .db import connect_sync, utc_now_iso

ReviewAction = Literal["approve", "edit", "reject"]
REVIEW_ACTIONS: frozenset[str] = frozenset({"approve", "edit", "reject"})

S = TicketState


class ReviewError(ValueError):
    """A review/requeue request that can't be applied (bad input or missing data)."""


class TicketNotFound(ReviewError):
    def __init__(self, ticket_id: str) -> None:
        super().__init__(f"ticket {ticket_id} not found")
        self.ticket_id = ticket_id


def human_edit_idempotency_key(ticket_id: str, body: str, review_action_id: int) -> str:
    """Deterministic per review action, so it never collides with the LLM draft or an earlier review."""
    digest = hashlib.sha256(body.encode()).hexdigest()[:16]
    return f"{ticket_id}:human:{digest}:{review_action_id}"


def _current_state(conn: sqlite3.Connection, ticket_id: str) -> TicketState:
    row = conn.execute("SELECT state FROM tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
    if row is None:
        raise TicketNotFound(ticket_id)
    return TicketState(row["state"])


def _transition(
    conn: sqlite3.Connection,
    ticket_id: str,
    to_state: TicketState,
    *,
    worker_id: str | None,
    detail: dict[str, Any] | None = None,
    reset_attempt: bool = False,
) -> None:
    """Same semantics as SqliteRepository.transition(); must run inside an open transaction."""
    from_state = _current_state(conn, ticket_id)
    assert_transition(ticket_id, from_state, to_state)
    now = utc_now_iso()
    if to_state in TERMINAL_STATES:
        finished_sql, finished_val = "?", (now,)
    elif from_state in TERMINAL_STATES:
        finished_sql, finished_val = "NULL", ()
    else:
        finished_sql, finished_val = "finished_at", ()
    attempt_sql = "0" if reset_attempt else "attempt"
    conn.execute(
        f"""UPDATE tickets SET state = ?, updated_at = ?, attempt = {attempt_sql}, finished_at = {finished_sql}
            WHERE ticket_id = ?""",
        (to_state.value, now, *finished_val, ticket_id),
    )
    conn.execute(
        """INSERT INTO ticket_transitions (ticket_id, from_state, to_state, at, worker_id, detail_json)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (ticket_id, from_state.value, to_state.value, now, worker_id, json.dumps(detail) if detail else None),
    )


def _run_tx(db_path: str, fn) -> None:
    conn = connect_sync(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            fn(conn)
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
    finally:
        conn.close()


def apply_review(
    db_path: str,
    ticket_id: str,
    action: ReviewAction,
    reviewer: str,
    edited_body: str | None = None,
) -> None:
    """Apply a human review decision to a NEEDS_REVIEW ticket.

    approve: NEEDS_REVIEW -> APPROVED; latest draft -> status 'approved' (draft must be non-empty).
    edit:    new draft (source='human_edit', status='approved'); previous latest draft -> 'superseded';
             NEEDS_REVIEW -> APPROVED.
    reject:  NEEDS_REVIEW -> REJECTED -> DONE; latest draft -> 'rejected'.
    """
    if action not in REVIEW_ACTIONS:
        raise ReviewError(f"invalid review action {action!r}; expected one of {sorted(REVIEW_ACTIONS)}")
    if action == "edit" and (edited_body is None or not edited_body.strip()):
        raise ReviewError("edit requires a non-empty edited_body")
    worker_id = f"reviewer:{reviewer}"

    def body(conn: sqlite3.Connection) -> None:
        state = _current_state(conn, ticket_id)
        if state is not S.NEEDS_REVIEW:
            raise InvalidTransition(ticket_id, state, S.REJECTED if action == "reject" else S.APPROVED)
        draft = conn.execute(
            "SELECT id, body FROM drafts WHERE ticket_id = ? ORDER BY id DESC LIMIT 1", (ticket_id,)
        ).fetchone()
        if action == "approve" and (draft is None or not draft["body"].strip()):
            raise ReviewError(f"ticket {ticket_id} has no non-empty draft to approve; use 'edit'")

        now = utc_now_iso()
        cur = conn.execute(
            "INSERT INTO review_actions (ticket_id, action, reviewer, edited_body, at) VALUES (?, ?, ?, ?, ?)",
            (ticket_id, action, reviewer, edited_body if action == "edit" else None, now),
        )
        review_id = int(cur.lastrowid)
        detail: dict[str, Any] = {"action": action, "reviewer": reviewer, "review_action_id": review_id}

        if action == "approve":
            conn.execute(
                "UPDATE drafts SET status = 'approved', reviewer = ? WHERE id = ?", (reviewer, draft["id"])
            )
            detail["draft_id"] = draft["id"]
            _transition(conn, ticket_id, S.APPROVED, worker_id=worker_id, detail=detail)
        elif action == "edit":
            assert edited_body is not None
            if draft is not None:
                conn.execute(
                    "UPDATE drafts SET status = 'superseded', reviewer = ? WHERE id = ?",
                    (reviewer, draft["id"]),
                )
            cur = conn.execute(
                """INSERT INTO drafts (ticket_id, body, source, status, idempotency_key, reviewer, created_at)
                   VALUES (?, ?, 'human_edit', 'approved', ?, ?, ?)""",
                (
                    ticket_id,
                    edited_body,
                    human_edit_idempotency_key(ticket_id, edited_body, review_id),
                    reviewer,
                    now,
                ),
            )
            detail["draft_id"] = int(cur.lastrowid)
            if draft is not None:
                detail["superseded_draft_id"] = draft["id"]
            _transition(conn, ticket_id, S.APPROVED, worker_id=worker_id, detail=detail)
        else:  # reject
            if draft is not None:
                conn.execute(
                    "UPDATE drafts SET status = 'rejected', reviewer = ? WHERE id = ?",
                    (reviewer, draft["id"]),
                )
                detail["draft_id"] = draft["id"]
            _transition(conn, ticket_id, S.REJECTED, worker_id=worker_id, detail=detail)
            _transition(conn, ticket_id, S.DONE, worker_id=worker_id, detail={"action": "reject"})

    _run_tx(db_path, body)


def requeue_dead_letter(db_path: str, ticket_id: str) -> None:
    """DEAD_LETTER -> QUEUED with attempt reset to 0. The runtime's review dispatcher enqueues it."""

    def body(conn: sqlite3.Connection) -> None:
        state = _current_state(conn, ticket_id)
        if state is not S.DEAD_LETTER:
            raise InvalidTransition(ticket_id, state, S.QUEUED)
        _transition(
            conn,
            ticket_id,
            S.QUEUED,
            worker_id="dashboard",
            detail={"action": "requeue_dead_letter"},
            reset_attempt=True,
        )

    _run_tx(db_path, body)
