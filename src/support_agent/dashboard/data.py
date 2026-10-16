"""Read-only data access for the Streamlit dashboard.

This is the one allowed exception to "no SQL outside storage/": SELECT-only queries over a sqlite
connection opened with `mode=ro`. Every write goes through `support_agent.storage.review`.
Functions are Streamlit-free (plain pandas) so they can be tested and reused.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass
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
    # Convert before the empty check: callers group by these columns, which needs datetime dtype
    # even when no rows matched.
    df["discovered_at"] = to_ts(df["discovered_at"])
    df["finished_at"] = to_ts(df["finished_at"])
    if df.empty:
        return df.assign(e2e_seconds=pd.Series(dtype=float), result=pd.Series(dtype=str))
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


def _ticket_filter(states: Iterable[str] | None, search: str | None) -> tuple[str, list[Any]]:
    states = list(states or [])
    clauses: list[str] = []
    params: list[Any] = []
    if states:
        clauses.append(f"t.state IN ({','.join('?' * len(states))})")
        params += states
    if search and search.strip():
        like = f"%{search.strip()}%"
        clauses.append("(t.ticket_id LIKE ? OR t.subject LIKE ? OR t.customer_email LIKE ?)")
        params += [like, like, like]
    return (("WHERE " + " AND ".join(clauses)) if clauses else ""), params


def tickets_count(db_path: str, states: Iterable[str] | None = None, search: str | None = None) -> int:
    where, params = _ticket_filter(states, search)
    row = _one(db_path, f"SELECT count(*) AS n FROM tickets t {where}", params)
    return int(row["n"]) if row else 0


def tickets(
    db_path: str, states: Iterable[str] | None = None, search: str | None = None, limit: int = 100
) -> pd.DataFrame:
    """Most recently updated first; `limit` rows (the console loads 100 by default)."""
    where, params = _ticket_filter(states, search)
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


def tickets_for_email(db_path: str, email: str, limit: int = 100) -> pd.DataFrame:
    df = query(
        db_path,
        """SELECT t.ticket_id, t.subject, t.state, COALESCE(p.priority, t.priority) AS priority, p.outcome,
                  p.reasons_json, t.discovered_at, t.updated_at
           FROM tickets t LEFT JOIN policy_decisions p ON p.ticket_id = t.ticket_id
           WHERE lower(t.customer_email) = lower(?) ORDER BY t.updated_at DESC LIMIT ?""",
        (email.strip(), int(limit)),
    )
    df["reasons"] = [", ".join(_loads(v, []) or []) for v in df["reasons_json"]]
    return df.drop(columns=["reasons_json"])


def customer_emails(
    db_path: str, q: str | None = None, limit: int = 100, offset: int = 0
) -> tuple[int, pd.DataFrame]:
    """Distinct customer emails seen on tickets (User Info fallback when the user service can't list)."""
    where, params = "", []
    if q and q.strip():
        where, params = "WHERE customer_email LIKE ?", [f"%{q.strip()}%"]
    total_row = _one(
        db_path, f"SELECT count(DISTINCT lower(customer_email)) AS n FROM tickets {where}", params
    )
    df = query(
        db_path,
        f"""SELECT lower(customer_email) AS email, count(*) AS tickets, max(updated_at) AS last_ticket_at
            FROM tickets {where} GROUP BY lower(customer_email) ORDER BY email LIMIT ? OFFSET ?""",
        (*params, int(limit), int(offset)),
    )
    return (int(total_row["n"]) if total_row else 0), df


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


# --------------------------------------------------------------------------- SQL console (read-only)

SQL_TIMEOUT_S = 5.0
READONLY_PRAGMAS = frozenset(
    {
        "table_info",
        "table_xinfo",
        "table_list",
        "index_list",
        "index_info",
        "index_xinfo",
        "foreign_key_list",
        "database_list",
        "collation_list",
        "function_list",
        "pragma_list",
        "compile_options",
        "user_version",
        "schema_version",
        "page_count",
        "page_size",
        "freelist_count",
        "journal_mode",
        "integrity_check",
        "quick_check",
        "foreign_key_check",
        "encoding",
    }
)
_DENY_WORDS = frozenset(
    {
        "INSERT",
        "UPDATE",
        "DELETE",
        "UPSERT",
        "DROP",
        "ALTER",
        "CREATE",
        "ATTACH",
        "DETACH",
        "VACUUM",
        "REINDEX",
        "ANALYZE",
        "BEGIN",
        "COMMIT",
        "ROLLBACK",
        "SAVEPOINT",
        "RELEASE",
        "LOAD_EXTENSION",
    }
)
_READ_STARTS = frozenset({"SELECT", "WITH", "VALUES"})


class SqlGuardError(ValueError):
    """The statement is not a single read-only query."""


class SqlTimeout(RuntimeError):
    """The query ran longer than the console's time limit."""


@dataclass
class QueryResult:
    statement: str
    columns: list[str]
    types: list[str]
    rows: list[tuple[Any, ...]]
    has_more: bool
    elapsed_ms: float

    def frame(self) -> pd.DataFrame:
        cols = _dedupe(self.columns)
        return pd.DataFrame.from_records(self.rows, columns=cols) if cols else pd.DataFrame()


def _dedupe(cols: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for c in cols:
        n = seen.get(c, 0)
        out.append(c if n == 0 else f"{c}_{n}")
        seen[c] = n + 1
    return out


def _scan_sql(sql: str) -> tuple[str, str]:
    """Return (sql without comments, same text with string/identifier contents blanked out).

    The blanked copy is what keyword and ';' checks look at, so words inside literals, quoted
    identifiers or comments can't fool the guard.
    """
    clean: list[str] = []
    code: list[str] = []
    i, n = 0, len(sql)
    closers = {"'": "'", '"': '"', "`": "`", "[": "]"}
    while i < n:
        ch = sql[i]
        if ch == "-" and sql.startswith("--", i):
            j = sql.find("\n", i)
            i = n if j < 0 else j
            continue
        if ch == "/" and sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            if j < 0:
                raise SqlGuardError("Unterminated /* comment.")
            clean.append(" ")
            code.append(" ")
            i = j + 2
            continue
        if ch in closers:
            close = closers[ch]
            j = i + 1
            while True:
                k = sql.find(close, j)
                if k < 0:
                    raise SqlGuardError("Unterminated quoted string or identifier.")
                if close in "'\"`" and sql.startswith(close * 2, k):  # doubled quote = escaped
                    j = k + 2
                    continue
                break
            clean.append(sql[i : k + 1])
            code.append(ch + " " * (k - i - 1) + close)
            i = k + 1
            continue
        clean.append(ch)
        code.append(ch)
        i += 1
    return "".join(clean), "".join(code)


def check_readonly_sql(sql: str) -> str:
    """Validate a console statement; return it normalised (comments and trailing ';' removed).

    Accepts exactly one SELECT / WITH / VALUES / EXPLAIN [QUERY PLAN] <select> / read-only PRAGMA.
    The read-only connection (`mode=ro` + `PRAGMA query_only`) is the real safety net; this guard
    gives clear messages and blocks things `query_only` doesn't (ATTACH, PRAGMA writes, extensions).
    """
    clean, code = _scan_sql(sql or "")
    clean, code = clean.strip(), code.strip()
    while code.endswith(";"):
        code, clean = code[:-1].rstrip(), clean[:-1].rstrip()
    if not code:
        raise SqlGuardError("Enter a query.")
    if ";" in code:
        raise SqlGuardError("Only one statement at a time; remove the extra `;`.")
    if not sqlite3.complete_statement(clean + ";") or code.count("(") != code.count(")"):
        raise SqlGuardError("The statement looks incomplete (unbalanced quotes or parentheses?).")
    words = [w.upper() for w in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", code)]
    first = words[0]
    if first == "PRAGMA":
        m = re.fullmatch(r"PRAGMA\s+(?:\w+\s*\.\s*)?(\w+)\s*(?:\(\s*[^()=;]*\))?", code, re.IGNORECASE)
        if not m or "=" in code:
            raise SqlGuardError("Only read-only PRAGMAs are allowed, e.g. `PRAGMA table_info(tickets)`.")
        if m.group(1).lower() not in READONLY_PRAGMAS:
            raise SqlGuardError(
                f"`PRAGMA {m.group(1)}` isn't allowed here. Allowed: {', '.join(sorted(READONLY_PRAGMAS))}."
            )
        return clean
    rest = words
    if first == "EXPLAIN":
        rest = words[1:]
        if rest[:2] == ["QUERY", "PLAN"]:
            rest = rest[2:]
        if not rest:
            raise SqlGuardError("EXPLAIN needs a query after it.")
    if rest[0] not in _READ_STARTS:
        raise SqlGuardError(
            "This console is read-only: only SELECT, WITH, VALUES, EXPLAIN and read-only PRAGMA "
            f"statements are allowed (got `{words[0]}`)."
        )
    banned = sorted(set(words) & _DENY_WORDS)
    if banned or re.search(r"\bREPLACE\s+INTO\b", code, re.IGNORECASE):
        raise SqlGuardError(
            f"This console is read-only: `{(banned or ['REPLACE INTO'])[0]}` is not allowed. "
            "Review actions live on the Tickets tab."
        )
    return clean


def _sql_type(v: Any) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bool | int):
        return "INTEGER"
    if isinstance(v, float):
        return "REAL"
    if isinstance(v, bytes | bytearray | memoryview):
        return "BLOB"
    return "TEXT"


def run_readonly_query(
    db_path: str, sql: str, limit: int = 100, timeout_s: float = SQL_TIMEOUT_S
) -> QueryResult:
    """Run one guarded statement on a read-only connection; fetch at most `limit` rows (+1 to detect more)."""
    stmt = check_readonly_sql(sql)
    conn = connect_ro(db_path)
    conn.row_factory = None
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 5_000)
    t0 = time.perf_counter()
    try:
        cur = conn.execute(stmt)
        rows = cur.fetchmany(int(limit) + 1)
        columns = [d[0] for d in cur.description] if cur.description else []
    except sqlite3.OperationalError as e:
        if "interrupted" in str(e).lower():
            raise SqlTimeout(f"Query stopped after {timeout_s:.0f} s. Add a WHERE clause or LIMIT.") from e
        raise
    finally:
        conn.close()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    has_more = len(rows) > limit
    rows = rows[: int(limit)]
    types = []
    for i in range(len(columns)):
        types.append(next((_sql_type(r[i]) for r in rows if r[i] is not None), "NULL"))
    return QueryResult(stmt, columns, types, rows, has_more, round(elapsed_ms, 2))


def schema_overview(db_path: str) -> list[dict[str, Any]]:
    """Tables (and views) with columns and row counts, for the SQL console's schema browser."""
    with closing(connect_ro(db_path)) as conn:
        objs = conn.execute(
            "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') "
            "AND name NOT LIKE 'sqlite_%' ORDER BY type, name"
        ).fetchall()
        out = []
        for o in objs:
            name = o["name"]
            quoted = '"' + name.replace('"', '""') + '"'
            cols = conn.execute(f"PRAGMA table_info({quoted})").fetchall()
            try:
                n = conn.execute(f"SELECT count(*) FROM {quoted}").fetchone()[0]
            except sqlite3.Error:
                n = None
            out.append(
                {
                    "name": name,
                    "type": o["type"],
                    "rows": n,
                    "columns": [
                        {
                            "name": c["name"],
                            "type": c["type"] or "",
                            "pk": bool(c["pk"]),
                            "notnull": bool(c["notnull"]),
                        }
                        for c in cols
                    ],
                }
            )
        return out
