"""Read-only data access for the Streamlit dashboard.

This is the one allowed exception to "no SQL outside storage/": SELECT-only queries over a sqlite
connection opened with `mode=ro`. Every write goes through `support_agent.storage.review`.
Functions are Streamlit-free (plain pandas) so they can be tested and reused.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterable
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from support_agent.common.config import PROJECT_ROOT, get_settings

DbStatus = Literal["missing", "uninitialized", "ok"]

FINAL_STATES = ("DONE", "DEAD_LETTER")
ACTIVE_STATES = ("QUEUED", "PROCESSING", "ENRICHED", "ANALYZED", "POLICY_CHECKED", "APPROVED")
ALL_STATES = (
    "DISCOVERED",
    "QUEUED",
    "PROCESSING",
    "ENRICHED",
    "ANALYZED",
    "POLICY_CHECKED",
    "DRAFT_POSTED",
    "NEEDS_REVIEW",
    "APPROVED",
    "REJECTED",
    "DONE",
    "FAILED",
    "DEAD_LETTER",
)


# --------------------------------------------------------------------------- connection


def resolve_db_path() -> str:
    """`SA_DB_PATH` env wins, else settings.db_path. Relative paths: cwd first, then the project root."""
    raw = os.environ.get("SA_DB_PATH") or get_settings().db_path
    p = Path(raw).expanduser()
    if p.is_absolute() or p.exists():
        return str(p.resolve())
    return str((PROJECT_ROOT / p).resolve())


def resolve_log_path() -> str:
    raw = os.environ.get("SA_LOG_FILE") or get_settings().log_file
    p = Path(raw).expanduser()
    if p.is_absolute() or p.exists():
        return str(p.resolve())
    return str((PROJECT_ROOT / p).resolve())


def connect_ro(db_path: str) -> sqlite3.Connection:
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def db_status(db_path: str) -> DbStatus:
    if not Path(db_path).is_file():
        return "missing"
    try:
        with closing(connect_ro(db_path)) as conn:
            row = conn.execute(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='tickets'"
            ).fetchone()
            return "ok" if row and row[0] else "uninitialized"
    except sqlite3.Error:
        return "uninitialized"


def query(db_path: str, sql: str, params: Iterable[Any] = ()) -> pd.DataFrame:
    with closing(connect_ro(db_path)) as conn:
        return pd.read_sql_query(sql, conn, params=list(params))


def _one(db_path: str, sql: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
    with closing(connect_ro(db_path)) as conn:
        row = conn.execute(sql, list(params)).fetchone()
        return dict(row) if row else None


def to_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, format="ISO8601", errors="coerce")


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def _loads(value: Any, default: Any = None) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- overview


def state_counts(db_path: str) -> dict[str, int]:
    df = query(db_path, "SELECT state, count(*) AS n FROM tickets GROUP BY state")
    return {r.state: int(r.n) for r in df.itertuples()}


def kpis(db_path: str) -> dict[str, Any]:
    counts = state_counts(db_path)
    dec = query(db_path, "SELECT outcome, count(*) AS n FROM policy_decisions GROUP BY outcome")
    by_outcome = {r.outcome: int(r.n) for r in dec.itertuples()}
    decided = sum(by_outcome.values())
    auto = by_outcome.get("auto_approve", 0)
    return {
        "discovered": sum(counts.values()),
        "processed": decided,
        "auto_approved": auto,
        "auto_approved_pct": (100.0 * auto / decided) if decided else None,
        "in_review": counts.get("NEEDS_REVIEW", 0),
        "in_flight": sum(counts.get(s, 0) for s in ACTIVE_STATES),
        "failed": counts.get("FAILED", 0),
        "dead_letter": counts.get("DEAD_LETTER", 0),
        "done": counts.get("DONE", 0),
        "states": counts,
    }


def finished_tickets(db_path: str, since: datetime | None = None) -> pd.DataFrame:
    """Tickets with a finished_at: e2e seconds and the final outcome bucket."""
    df = query(
        db_path,
        """SELECT t.ticket_id, t.state, t.discovered_at, t.finished_at, p.outcome,
                  EXISTS(SELECT 1 FROM review_actions r WHERE r.ticket_id = t.ticket_id) AS reviewed
           FROM tickets t LEFT JOIN policy_decisions p ON p.ticket_id = t.ticket_id
           WHERE t.finished_at IS NOT NULL AND (? IS NULL OR t.finished_at >= ?)""",
        (_iso(since), _iso(since)),
    )
    if df.empty:
        return df.assign(e2e_seconds=pd.Series(dtype=float), result=pd.Series(dtype=str))
    df["discovered_at"] = to_ts(df["discovered_at"])
    df["finished_at"] = to_ts(df["finished_at"])
    df["e2e_seconds"] = (df["finished_at"] - df["discovered_at"]).dt.total_seconds()

    def bucket(r) -> str:
        if r.state == "DEAD_LETTER":
            return "Dead letter"
        if r.reviewed:
            return "Human reviewed"
        if r.outcome == "auto_approve":
            return "Auto-approved"
        return "Other"

    df["result"] = [bucket(r) for r in df.itertuples()]
    return df


def discovered_times(db_path: str, since: datetime | None = None) -> pd.DataFrame:
    df = query(
        db_path,
        "SELECT ticket_id, discovered_at FROM tickets WHERE (? IS NULL OR discovered_at >= ?)",
        (_iso(since), _iso(since)),
    )
    df["discovered_at"] = to_ts(df["discovered_at"])
    return df


# --------------------------------------------------------------------------- tickets


def tickets(
    db_path: str, states: Iterable[str] | None = None, search: str | None = None, limit: int = 1000
) -> pd.DataFrame:
    states = list(states or [])
    clauses, params = [], []
    if states:
        clauses.append(f"t.state IN ({','.join('?' * len(states))})")
        params += states
    if search and search.strip():
        like = f"%{search.strip()}%"
        clauses.append("(t.ticket_id LIKE ? OR t.subject LIKE ? OR t.customer_email LIKE ?)")
        params += [like, like, like]
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    df = query(
        db_path,
        f"""SELECT t.ticket_id, t.subject, t.customer_email, t.state, t.attempt,
                   COALESCE(p.priority, t.priority) AS priority, p.outcome, p.reasons_json,
                   t.discovered_at, t.updated_at, t.finished_at, t.last_error
            FROM tickets t LEFT JOIN policy_decisions p ON p.ticket_id = t.ticket_id
            {where}
            ORDER BY t.updated_at DESC LIMIT ?""",
        (*params, int(limit)),
    )
    df["reasons"] = [", ".join(_loads(v, []) or []) for v in df["reasons_json"]]
    return df.drop(columns=["reasons_json"])


def ticket_detail(db_path: str, ticket_id: str) -> dict[str, Any] | None:
    row = _one(db_path, "SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,))
    if row is None:
        return None
    snap = _one(db_path, "SELECT * FROM ticket_snapshots WHERE ticket_id = ?", (ticket_id,))
    user = _one(db_path, "SELECT * FROM user_lookups WHERE ticket_id = ?", (ticket_id,))
    policy = _one(db_path, "SELECT * FROM policy_decisions WHERE ticket_id = ?", (ticket_id,))
    transitions = query(
        db_path,
        "SELECT id, from_state, to_state, at, worker_id, detail_json FROM ticket_transitions "
        "WHERE ticket_id = ? ORDER BY id",
        (ticket_id,),
    )
    spans = query(
        db_path,
        "SELECT id, trace_id, name, start_ts, duration_ms, status, attrs_json FROM spans "
        "WHERE ticket_id = ? OR (trace_id IS NOT NULL AND trace_id = ?) ORDER BY start_ts, id",
        (ticket_id, row.get("trace_id")),
    )
    llm = query(db_path, "SELECT * FROM llm_calls WHERE ticket_id = ? ORDER BY id", (ticket_id,))
    drafts = query(
        db_path,
        "SELECT id, source, status, body, reviewer, created_at, posted_at, idempotency_key FROM drafts "
        "WHERE ticket_id = ? ORDER BY id",
        (ticket_id,),
    )
    reviews = query(
        db_path,
        "SELECT action, reviewer, edited_body, at FROM review_actions WHERE ticket_id = ? ORDER BY id",
        (ticket_id,),
    )
    for df, col in ((transitions, "at"), (spans, "start_ts")):
        df[col] = to_ts(df[col])
    return {
        "ticket": row,
        "snapshot": (
            {**snap, "messages": _loads(snap.get("messages_json"), [])} if snap is not None else None
        ),
        "user_lookup": ({**user, "user": _loads(user.get("user_json"))} if user is not None else None),
        "policy": ({**policy, "reasons": _loads(policy.get("reasons_json"), [])} if policy else None),
        "transitions": transitions,
        "spans": spans,
        "llm_calls": llm,
        "drafts": drafts,
        "reviews": reviews,
    }


# --------------------------------------------------------------------------- review


def pending_reviews(db_path: str) -> pd.DataFrame:
    """NEEDS_REVIEW tickets with their latest draft + policy decision; high priority first, then oldest."""
    df = query(
        db_path,
        """SELECT t.ticket_id, t.subject, t.customer_email, t.updated_at, t.discovered_at,
                  COALESCE(p.priority, 'normal') AS priority, p.reasons_json,
                  d.id AS draft_id, d.body AS draft_body, d.source AS draft_source, d.status AS draft_status
           FROM tickets t
           LEFT JOIN policy_decisions p ON p.ticket_id = t.ticket_id
           LEFT JOIN drafts d ON d.id = (SELECT MAX(id) FROM drafts WHERE ticket_id = t.ticket_id)
           WHERE t.state = 'NEEDS_REVIEW'
           ORDER BY CASE COALESCE(p.priority, 'normal') WHEN 'high' THEN 0 ELSE 1 END, t.updated_at""",
    )
    df["reasons"] = [_loads(v, []) or [] for v in df["reasons_json"]]
    return df.drop(columns=["reasons_json"])


def recent_reviews(db_path: str, limit: int = 20) -> pd.DataFrame:
    return query(
        db_path,
        "SELECT r.at, r.ticket_id, r.action, r.reviewer, t.state FROM review_actions r "
        "LEFT JOIN tickets t ON t.ticket_id = r.ticket_id ORDER BY r.id DESC LIMIT ?",
        (int(limit),),
    )


# --------------------------------------------------------------------------- llm


def llm_calls(db_path: str, since: datetime | None = None) -> pd.DataFrame:
    df = query(
        db_path,
        """SELECT id, ticket_id, model, input_tokens, output_tokens, cache_read_tokens, latency_ms,
                  stop_reason, success, error, at,
                  CASE WHEN error IS NOT NULL THEN 'error'
                       WHEN stop_reason = 'refusal' THEN 'refusal'
                       WHEN analysis_json IS NULL THEN 'parse_error'
                       ELSE 'ok' END AS result
           FROM llm_calls WHERE (? IS NULL OR at >= ?) ORDER BY id""",
        (_iso(since), _iso(since)),
    )
    df["at"] = to_ts(df["at"])
    return df


# --------------------------------------------------------------------------- metrics & scheduler


def metric_samples(db_path: str, names: Iterable[str], since: datetime | None = None) -> pd.DataFrame:
    """Long frame: ts, name, labels (dict), label (sorted 'k=v' string), value."""
    names = list(names)
    if not names:
        return pd.DataFrame(columns=["ts", "name", "labels", "label", "value"])
    df = query(
        db_path,
        f"""SELECT ts, name, labels_json, value FROM metric_samples
            WHERE name IN ({",".join("?" * len(names))}) AND (? IS NULL OR ts >= ?)
            ORDER BY ts""",
        (*names, _iso(since), _iso(since)),
    )
    df["ts"] = to_ts(df["ts"])
    df["labels"] = [_loads(v, {}) or {} for v in df["labels_json"]]
    df["label"] = [",".join(f"{k}={v}" for k, v in sorted(d.items())) for d in df["labels"]]
    return df.drop(columns=["labels_json"])


def scheduler_runs(db_path: str, since: datetime | None = None, limit: int = 500) -> pd.DataFrame:
    df = query(
        db_path,
        """SELECT id, started_at, finished_at, found, enqueued, skipped_full, error FROM scheduler_runs
           WHERE (? IS NULL OR started_at >= ?) ORDER BY id DESC LIMIT ?""",
        (_iso(since), _iso(since), int(limit)),
    )
    df["started_at"] = to_ts(df["started_at"])
    df["finished_at"] = to_ts(df["finished_at"])
    df["duration_s"] = (df["finished_at"] - df["started_at"]).dt.total_seconds()
    return df.sort_values("id")


def last_scheduler_error(db_path: str) -> dict[str, Any] | None:
    return _one(
        db_path,
        "SELECT id, started_at, error FROM scheduler_runs WHERE error IS NOT NULL ORDER BY id DESC LIMIT 1",
    )


# --------------------------------------------------------------------------- logs


def read_log_tail(
    log_path: str, max_lines: int = 1000, max_bytes: int = 4_000_000
) -> tuple[pd.DataFrame, int]:
    """Last `max_lines` JSON log lines. Returns (frame, number of unparseable lines skipped)."""
    p = Path(log_path)
    if not p.is_file():
        return pd.DataFrame(), 0
    with p.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - max_bytes))
        chunk = f.read().decode("utf-8", errors="replace")
    lines = chunk.splitlines()
    if size > max_bytes and lines:
        lines = lines[1:]  # first line is probably partial
    lines = lines[-max_lines:]
    records, bad = [], 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            if not isinstance(rec, dict):
                raise ValueError
        except ValueError:
            bad += 1
            continue
        records.append(rec)
    df = pd.DataFrame.from_records(records)
    for col in ("ts", "level", "component", "event", "ticket_id", "trace_id", "worker_id"):
        if col not in df.columns:
            df[col] = None
    df["ts"] = to_ts(df["ts"])
    return df, bad
