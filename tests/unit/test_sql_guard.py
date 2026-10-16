"""Read-only SQL console data layer (Phase 5 U5, `dashboard/data.py`): the statement guard, the read-only
connection as a second line of defence, the 100-row default + has_more, the timeout, and the schema browser."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from tests.conftest import require

data = require("support_agent.dashboard.data", "check_readonly_sql", "run_readonly_query", "SqlGuardError")
SqlGuardError, SqlTimeout = data.SqlGuardError, data.SqlTimeout


@pytest.fixture
def db(tmp_path) -> str:
    """A real schema with 250 tickets."""
    path = str(tmp_path / "console.db")
    from support_agent.storage.db import schema_sql

    conn = sqlite3.connect(path)
    conn.executescript(schema_sql())
    conn.executemany(
        "INSERT INTO tickets (ticket_id, subject, customer_email, state, discovered_at, updated_at, trace_id) "
        "VALUES (?, ?, ?, 'DONE', '2026-10-16T00:00:00Z', '2026-10-16T00:00:00Z', 'tr')",
        [(f"T-{i:04d}", f"subject {i}", f"u{i}@example.com") for i in range(250)],
    )
    conn.commit()
    conn.close()
    return path


def digest(path: str) -> str:
    """Hash of the DB file plus its WAL (if any)."""
    h = hashlib.sha256()
    for p in (Path(path), Path(path + "-wal")):
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()


def count(path: str) -> int:
    conn = sqlite3.connect(path)
    try:
        return conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
    finally:
        conn.close()


# --------------------------------------------------------------------------- guard: rejected

REJECTED = [
    "UPDATE tickets SET state = 'DONE'",
    "DELETE FROM tickets",
    "DROP TABLE tickets",
    "ALTER TABLE tickets ADD COLUMN x",
    "CREATE TABLE x (a)",
    "CREATE TEMP TABLE x AS SELECT 1",
    "INSERT INTO tickets (ticket_id) VALUES ('x')",
    "REPLACE INTO tickets (ticket_id) VALUES ('x')",
    "ATTACH DATABASE '/tmp/evil.db' AS evil",
    "DETACH DATABASE main",
    "VACUUM",
    "VACUUM INTO '/tmp/copy.db'",
    "REINDEX",
    "BEGIN; DELETE FROM tickets; COMMIT",
    "SELECT 1; DROP TABLE tickets",
    "SELECT 1; SELECT 2",
    "WITH x AS (SELECT 1) DELETE FROM tickets",
    "WITH x AS (SELECT 1) UPDATE tickets SET state = 'DONE'",
    "WITH x AS (SELECT 1) INSERT INTO tickets (ticket_id) SELECT 'y'",
    "PRAGMA query_only=OFF",
    "PRAGMA query_only = 0",
    "PRAGMA journal_mode=DELETE",
    "PRAGMA writable_schema=ON",
    "PRAGMA foreign_keys",  # not on the allow-list
    "/* harmless */ DELETE FROM tickets",
    "-- comment\nDELETE FROM tickets",
    "SELECT 1 -- ; looks like a comment\n; DELETE FROM tickets",
    "SELECT 1 /* ; */ ; DELETE FROM tickets",
    "SELECT ';'; DELETE FROM tickets",
    "SELECT * FROM tickets WHERE subject = 'x'; DROP TABLE tickets; --'",
    "SELECT load_extension('evil')",
    "EXPLAIN DELETE FROM tickets",
    "EXPLAIN",
    "",
    "   ",
    "-- only a comment",
    "SELECT * FROM (SELECT 1",
    "SELECT 'unterminated",
]


@pytest.mark.parametrize("sql", REJECTED)
def test_guard_rejects(sql):
    with pytest.raises(SqlGuardError) as ei:
        data.check_readonly_sql(sql)
    assert str(ei.value)  # a human-readable message for the console


# --------------------------------------------------------------------------- guard: accepted

ACCEPTED = [
    "SELECT * FROM tickets",
    "select ticket_id from tickets limit 5;",
    "SELECT 1;;",
    "WITH s AS (SELECT state, COUNT(*) n FROM tickets GROUP BY state) SELECT * FROM s",
    "VALUES (1), (2)",
    "EXPLAIN SELECT * FROM tickets",
    "EXPLAIN QUERY PLAN SELECT * FROM tickets WHERE state = 'DONE'",
    "PRAGMA table_info(tickets)",
    "PRAGMA main.table_info(tickets)",
    "PRAGMA index_list(tickets)",
    "PRAGMA user_version",
    "PRAGMA integrity_check",
    "pragma journal_mode",
    "SELECT 'DELETE FROM tickets; DROP TABLE x' AS harmless_text",  # keywords inside a literal
    'SELECT "subject" FROM tickets',
    "SELECT * FROM tickets /* DROP TABLE tickets; */",
    "SELECT * FROM tickets -- ; DELETE",
    "SELECT replace(subject, 'a', 'b') FROM tickets",  # REPLACE() function, not REPLACE INTO
    "SELECT updated_at, created_at FROM drafts",  # column names containing UPDATE / CREATE
    "SELECT * FROM ticket_transitions WHERE to_state = 'DROP'",
]


@pytest.mark.parametrize("sql", ACCEPTED)
def test_guard_accepts(sql):
    out = data.check_readonly_sql(sql)
    assert out and not out.rstrip().endswith(";")


def test_guard_strips_trailing_semicolons_and_comments():
    assert data.check_readonly_sql("SELECT 1 ;  ").strip() == "SELECT 1"


def test_allow_listed_pragmas_are_read_only():
    writes = {
        "query_only",
        "writable_schema",
        "foreign_keys",
        "journal_size_limit",
        "locking_mode",
        "synchronous",
    }
    assert not (data.READONLY_PRAGMAS & writes)
    assert {"table_info", "index_list", "integrity_check"} <= data.READONLY_PRAGMAS


# --------------------------------------------------------------------------- execution


def test_default_100_rows_with_more(db):
    r = data.run_readonly_query(db, "SELECT * FROM tickets ORDER BY ticket_id")
    assert len(r.rows) == 100 and r.has_more is True
    assert r.columns[0] == "ticket_id"
    assert r.rows[0][0] == "T-0000"
    assert r.elapsed_ms >= 0


def test_larger_limit_and_exact_end(db):
    r = data.run_readonly_query(db, "SELECT ticket_id FROM tickets", limit=200)
    assert len(r.rows) == 200 and r.has_more
    r = data.run_readonly_query(db, "SELECT ticket_id FROM tickets", limit=250)
    assert len(r.rows) == 250 and r.has_more is False


def test_column_types_reported(db):
    r = data.run_readonly_query(db, "SELECT ticket_id, attempt, 1.5 AS f, NULL AS n, x'00' AS b FROM tickets")
    assert r.types == ["TEXT", "INTEGER", "REAL", "NULL", "BLOB"]


def test_pragma_and_explain_run(db):
    r = data.run_readonly_query(db, "PRAGMA table_info(tickets)")
    assert "ticket_id" in [row[1] for row in r.rows]
    assert data.run_readonly_query(db, "EXPLAIN QUERY PLAN SELECT * FROM tickets WHERE state='DONE'").rows


def test_rejected_write_leaves_db_untouched(db):
    before = digest(db)
    for sql in ("DELETE FROM tickets", "DROP TABLE tickets", "WITH x AS (SELECT 1) DELETE FROM tickets"):
        with pytest.raises(SqlGuardError):
            data.run_readonly_query(db, sql)
    assert digest(db) == before and count(db) == 250


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM tickets",
        "UPDATE tickets SET state = 'X'",
        "DROP TABLE tickets",
        "CREATE TABLE evil (a)",
        "INSERT INTO tickets (ticket_id, subject, customer_email, state, discovered_at, updated_at, trace_id) "
        "VALUES ('x','x','x','x','x','x','x')",
        "PRAGMA query_only=OFF",
    ],
)
def test_connection_is_read_only_even_if_guard_bypassed(db, monkeypatch, sql):
    """Defence in depth: with the guard disabled, the `mode=ro` + `query_only` connection still refuses."""
    before = digest(db)
    monkeypatch.setattr(data, "check_readonly_sql", lambda s: s)
    try:
        data.run_readonly_query(db, sql)
        # PRAGMA query_only=OFF itself may "succeed" on a mode=ro connection; then a write must still fail
        if sql.startswith("PRAGMA"):
            data.run_readonly_query(db, "DELETE FROM tickets")
    except sqlite3.Error as e:
        assert "readonly" in str(e).lower() or "read-only" in str(e).lower() or "query_only" in str(e).lower()
    assert digest(db) == before and count(db) == 250


def test_timeout_gives_friendly_error(db):
    heavy = "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT COUNT(*) FROM n, tickets"
    with pytest.raises(SqlTimeout) as ei:
        data.run_readonly_query(db, heavy, timeout_s=0.3)
    assert "LIMIT" in str(ei.value) or "WHERE" in str(ei.value)


def test_sql_error_propagates_as_sqlite_error(db):
    with pytest.raises(sqlite3.Error):
        data.run_readonly_query(db, "SELECT nope FROM tickets")


def test_missing_db(tmp_path):
    with pytest.raises(sqlite3.Error):
        data.run_readonly_query(str(tmp_path / "missing.db"), "SELECT 1")
    assert not (tmp_path / "missing.db").exists()  # read-only open never creates the file


def test_schema_overview(db):
    tables = {t["name"]: t for t in data.schema_overview(db)}
    assert tables["tickets"]["rows"] == 250
    cols = {c["name"]: c for c in tables["tickets"]["columns"]}
    assert cols["ticket_id"]["pk"] is True
    assert {"drafts", "ticket_transitions", "review_actions", "spans"} <= set(tables)
