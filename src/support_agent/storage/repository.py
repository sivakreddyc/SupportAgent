"""SqliteRepository — the runtime's implementation of `common.contracts.Repository`.

One aiosqlite connection per process. All writes go through `self._lock` and an explicit
`BEGIN IMMEDIATE ... COMMIT`, so each write is short, atomic and serialised. Reads don't take the lock.
Timestamps are ISO-8601 UTC strings.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from contextlib import asynccontextmanager
from typing import Any

import aiosqlite

from support_agent.common.models import LLMResult, PolicyDecision, Ticket, TicketSummary, UserInfo
from support_agent.common.states import TERMINAL_STATES, InvalidTransition, TicketState, assert_transition

from .db import connect, to_iso, utc_now_iso


def _json(value: Any) -> str:
    return json.dumps(value, default=str)


class SqliteRepository:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ lifecycle

    async def init(self) -> None:
        if self._conn is None:
            self._conn = await connect(self.db_path)

    async def close(self) -> None:
        if self._conn is not None:
            async with self._lock:
                await self._conn.close()
                self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("SqliteRepository not initialised; call await repo.init()")
        return self._conn

    @asynccontextmanager
    async def _tx(self):
        """Serialised write transaction. Nothing may `await` before the lock (keeps FIFO ordering)."""
        async with self._lock:
            conn = self.conn
            await conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                await conn.execute("ROLLBACK")
                raise
            else:
                await conn.execute("COMMIT")

    async def _fetchone(self, sql: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
        async with self.conn.execute(sql, tuple(params)) as cur:
            row = await cur.fetchone()
        return dict(row) if row is not None else None

    async def _fetchall(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        async with self.conn.execute(sql, tuple(params)) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def ping(self) -> bool:
        try:
            await self._fetchone("SELECT 1 AS ok")
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ tickets & state

    async def upsert_discovered(self, summary: TicketSummary, trace_id: str) -> bool:
        now = utc_now_iso()
        async with self._tx() as conn:
            cur = await conn.execute(
                """INSERT OR IGNORE INTO tickets
                   (ticket_id, subject, customer_email, state, attempt, ui_created_at,
                    discovered_at, updated_at, trace_id)
                   VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?)""",
                (
                    summary.ticket_id,
                    summary.subject,
                    summary.customer_email,
                    TicketState.DISCOVERED.value,
                    to_iso(summary.created_at),
                    now,
                    now,
                    trace_id,
                ),
            )
            inserted = cur.rowcount == 1
            if inserted:
                await conn.execute(
                    """INSERT INTO ticket_transitions (ticket_id, from_state, to_state, at, worker_id, detail_json)
                       VALUES (?, NULL, ?, ?, NULL, ?)""",
                    (summary.ticket_id, TicketState.DISCOVERED.value, now, _json({"source": "poller"})),
                )
        return inserted

    async def get_ticket_row(self, ticket_id: str) -> dict[str, Any] | None:
        return await self._fetchone("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,))

    async def tickets_in_states(self, states: set[TicketState]) -> list[dict[str, Any]]:
        if not states:
            return []
        marks = ",".join("?" for _ in states)
        return await self._fetchall(
            f"SELECT * FROM tickets WHERE state IN ({marks}) ORDER BY discovered_at, ticket_id",
            [TicketState(s).value for s in states],
        )

    async def transition(
        self,
        ticket_id: str,
        to_state: TicketState,
        *,
        worker_id: str | None = None,
        detail: dict[str, Any] | None = None,
        error: str | None = None,
        bump_attempt: bool = False,
    ) -> None:
        to_state = TicketState(to_state)
        async with self._tx() as conn:
            async with conn.execute("SELECT state FROM tickets WHERE ticket_id = ?", (ticket_id,)) as cur:
                row = await cur.fetchone()
            if row is None:
                raise InvalidTransition(ticket_id, None, to_state)
            from_state = TicketState(row["state"])
            assert_transition(ticket_id, from_state, to_state)

            now = utc_now_iso()
            if to_state in TERMINAL_STATES:
                finished_sql = "?"
                finished_val: tuple[Any, ...] = (now,)
            elif from_state in TERMINAL_STATES:  # e.g. DEAD_LETTER -> QUEUED (manual requeue)
                finished_sql, finished_val = "NULL", ()
            else:
                finished_sql, finished_val = "finished_at", ()
            await conn.execute(
                f"""UPDATE tickets
                    SET state = ?, updated_at = ?,
                        last_error = COALESCE(?, last_error),
                        attempt = attempt + ?,
                        finished_at = {finished_sql}
                    WHERE ticket_id = ?""",
                (to_state.value, now, error, 1 if bump_attempt else 0, *finished_val, ticket_id),
            )
            full_detail = dict(detail or {})
            if error is not None:
                full_detail.setdefault("error", error)
            await conn.execute(
                """INSERT INTO ticket_transitions (ticket_id, from_state, to_state, at, worker_id, detail_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    ticket_id,
                    from_state.value,
                    to_state.value,
                    now,
                    worker_id,
                    _json(full_detail) if full_detail else None,
                ),
            )

    # ------------------------------------------------------------------ step records

    async def save_snapshot(self, ticket: Ticket) -> None:
        messages = [m.model_dump(mode="json") for m in ticket.messages]
        async with self._tx() as conn:
            await conn.execute(
                """INSERT OR REPLACE INTO ticket_snapshots (ticket_id, body, messages_json, fetched_at)
                   VALUES (?, ?, ?, ?)""",
                (ticket.ticket_id, ticket.body, _json(messages), utc_now_iso()),
            )
            if ticket.priority is not None:
                await conn.execute(
                    "UPDATE tickets SET priority = ? WHERE ticket_id = ?", (ticket.priority, ticket.ticket_id)
                )

    async def record_user_lookup(self, ticket_id: str, user: UserInfo | None, latency_ms: float) -> None:
        async with self._tx() as conn:
            await conn.execute(
                """INSERT OR REPLACE INTO user_lookups (ticket_id, found, user_json, latency_ms, at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    ticket_id,
                    1 if user is not None else 0,
                    user.model_dump_json() if user is not None else None,
                    latency_ms,
                    utc_now_iso(),
                ),
            )

    async def record_llm_call(
        self, ticket_id: str, result: LLMResult | None, *, error: str | None = None
    ) -> None:
        success = 1 if (error is None and result is not None) else 0
        usage = result.usage if result is not None else None
        async with self._tx() as conn:
            await conn.execute(
                """INSERT INTO llm_calls (ticket_id, model, input_tokens, output_tokens, cache_read_tokens,
                                          latency_ms, stop_reason, success, error, analysis_json, at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ticket_id,
                    result.model if result is not None else None,
                    usage.input_tokens if usage else 0,
                    usage.output_tokens if usage else 0,
                    usage.cache_read_input_tokens if usage else 0,
                    result.latency_ms if result is not None else None,
                    result.stop_reason if result is not None else None,
                    success,
                    error,
                    result.analysis.model_dump_json() if result is not None and result.analysis else None,
                    utc_now_iso(),
                ),
            )

    async def record_policy_decision(self, ticket_id: str, decision: PolicyDecision) -> None:
        async with self._tx() as conn:
            await conn.execute(
                """INSERT OR REPLACE INTO policy_decisions (ticket_id, outcome, reasons_json, priority, at)
                   VALUES (?, ?, ?, ?, ?)""",
                (ticket_id, decision.outcome, _json(decision.reasons), decision.priority, utc_now_iso()),
            )

    # ------------------------------------------------------------------ drafts & review

    async def create_draft(
        self, ticket_id: str, body: str, *, source: str, status: str, idempotency_key: str
    ) -> int:
        """Insert a draft. If `idempotency_key` already exists (a retried ticket), return the existing id."""
        async with self._tx() as conn:
            cur = await conn.execute(
                """INSERT OR IGNORE INTO drafts (ticket_id, body, source, status, idempotency_key, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (ticket_id, body, source, status, idempotency_key, utc_now_iso()),
            )
            if cur.rowcount == 1 and cur.lastrowid is not None:
                return int(cur.lastrowid)
            # Existing key (retried / re-run ticket): reuse the row, refresh its status unless already posted.
            await conn.execute(
                "UPDATE drafts SET status = ? WHERE idempotency_key = ? AND status != 'posted'",
                (status, idempotency_key),
            )
            async with conn.execute(
                "SELECT id FROM drafts WHERE idempotency_key = ?", (idempotency_key,)
            ) as cur2:
                row = await cur2.fetchone()
            return int(row["id"])

    async def mark_draft_posted(self, draft_id: int) -> None:
        async with self._tx() as conn:
            await conn.execute(
                "UPDATE drafts SET status = 'posted', posted_at = COALESCE(posted_at, ?) WHERE id = ?",
                (utc_now_iso(), draft_id),
            )

    async def latest_draft(self, ticket_id: str) -> dict[str, Any] | None:
        return await self._fetchone(
            "SELECT * FROM drafts WHERE ticket_id = ? ORDER BY id DESC LIMIT 1", (ticket_id,)
        )

    async def approved_pending_post(self) -> list[dict[str, Any]]:
        """APPROVED tickets with their latest draft. Keys: ticket_id, trace_id, attempt, discovered_at,
        draft_id, body, source, draft_status, idempotency_key."""
        return await self._fetchall(
            """SELECT t.ticket_id, t.trace_id, t.attempt, t.discovered_at,
                      d.id AS draft_id, d.body, d.source, d.status AS draft_status, d.idempotency_key
               FROM tickets t
               JOIN drafts d ON d.id = (SELECT MAX(id) FROM drafts WHERE ticket_id = t.ticket_id)
               WHERE t.state = ?
               ORDER BY t.updated_at""",
            (TicketState.APPROVED.value,),
        )

    async def pending_reviews(self) -> list[dict[str, Any]]:
        """NEEDS_REVIEW tickets with latest draft + policy decision (high priority first, then oldest)."""
        rows = await self._fetchall(
            """SELECT t.ticket_id, t.subject, t.customer_email, t.discovered_at, t.updated_at, t.trace_id,
                      d.id AS draft_id, d.body AS draft_body, d.status AS draft_status,
                      p.outcome, p.reasons_json, COALESCE(p.priority, 'normal') AS policy_priority
               FROM tickets t
               LEFT JOIN drafts d ON d.id = (SELECT MAX(id) FROM drafts WHERE ticket_id = t.ticket_id)
               LEFT JOIN policy_decisions p ON p.ticket_id = t.ticket_id
               WHERE t.state = ?
               ORDER BY CASE COALESCE(p.priority, 'normal') WHEN 'high' THEN 0 ELSE 1 END, t.updated_at""",
            (TicketState.NEEDS_REVIEW.value,),
        )
        for r in rows:
            r["reasons"] = json.loads(r.pop("reasons_json") or "[]")
        return rows

    # ------------------------------------------------------------------ runtime helpers (not in Protocol)

    async def _latest_from_state(self, state: TicketState, from_state: TicketState) -> list[dict[str, Any]]:
        return await self._fetchall(
            """SELECT t.* FROM tickets t
               WHERE t.state = ?
                 AND (SELECT tt.from_state FROM ticket_transitions tt
                      WHERE tt.ticket_id = t.ticket_id ORDER BY tt.id DESC LIMIT 1) = ?
               ORDER BY t.updated_at""",
            (state.value, from_state.value),
        )

    async def requeued_from_dead_letter(self) -> list[dict[str, Any]]:
        """Tickets in QUEUED whose latest transition was DEAD_LETTER -> QUEUED (manual dashboard requeue)."""
        return await self._latest_from_state(TicketState.QUEUED, TicketState.DEAD_LETTER)

    async def failed_review_posts(self) -> list[dict[str, Any]]:
        """Tickets in FAILED whose latest transition was APPROVED -> FAILED (review dispatcher post failure)."""
        return await self._latest_from_state(TicketState.FAILED, TicketState.APPROVED)

    async def counts_by_state(self) -> dict[str, int]:
        rows = await self._fetchall("SELECT state, COUNT(*) AS n FROM tickets GROUP BY state")
        counts = {s.value: 0 for s in TicketState}
        counts.update({r["state"]: int(r["n"]) for r in rows})
        return counts

    async def transitions(self, ticket_id: str) -> list[dict[str, Any]]:
        rows = await self._fetchall(
            "SELECT * FROM ticket_transitions WHERE ticket_id = ? ORDER BY id", (ticket_id,)
        )
        for r in rows:
            r["detail"] = json.loads(r["detail_json"]) if r["detail_json"] else {}
        return rows

    # ------------------------------------------------------------------ observability

    async def record_span(self, span: dict[str, Any]) -> None:
        async with self._tx() as conn:
            await conn.execute(
                """INSERT INTO spans (trace_id, ticket_id, name, start_ts, duration_ms, status, attrs_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    span.get("trace_id"),
                    span.get("ticket_id"),
                    span["name"],
                    span.get("start_ts") or utc_now_iso(),
                    float(span.get("duration_ms") or 0.0),
                    span.get("status", "ok"),
                    _json(span.get("attrs") or {}),
                ),
            )

    async def record_metric_samples(self, samples: list[tuple[str, dict[str, str], float]]) -> None:
        if not samples:
            return
        ts = utc_now_iso()
        async with self._tx() as conn:
            await conn.executemany(
                "INSERT INTO metric_samples (ts, name, labels_json, value) VALUES (?, ?, ?, ?)",
                [(ts, name, _json(labels), float(value)) for name, labels, value in samples],
            )

    async def start_scheduler_run(self) -> int:
        async with self._tx() as conn:
            cur = await conn.execute("INSERT INTO scheduler_runs (started_at) VALUES (?)", (utc_now_iso(),))
            return int(cur.lastrowid)

    async def finish_scheduler_run(
        self, run_id: int, *, found: int, enqueued: int, skipped_full: int, error: str | None = None
    ) -> None:
        async with self._tx() as conn:
            await conn.execute(
                """UPDATE scheduler_runs SET finished_at = ?, found = ?, enqueued = ?, skipped_full = ?, error = ?
                   WHERE id = ?""",
                (utc_now_iso(), found, enqueued, skipped_full, error, run_id),
            )

    async def last_scheduler_run(self) -> dict[str, Any] | None:
        return await self._fetchone("SELECT * FROM scheduler_runs ORDER BY id DESC LIMIT 1")
