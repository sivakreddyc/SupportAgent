"""SQLite connection helpers (async for the runtime, sync for the Streamlit dashboard).

Both paths apply the same pragmas (WAL, busy_timeout=5000, foreign_keys=ON) and the idempotent
schema in `schema.sql` (every statement is `CREATE ... IF NOT EXISTS`, so applying it is the migration).
Connections run in autocommit mode (`isolation_level=None`); callers open explicit
`BEGIN IMMEDIATE ... COMMIT` blocks so transactions stay short and visible.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
BUSY_TIMEOUT_MS = 5000

_PRAGMAS = (
    f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}",
    "PRAGMA journal_mode = WAL",
    "PRAGMA foreign_keys = ON",
    "PRAGMA synchronous = NORMAL",
)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def to_iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat()


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def schema_sql() -> str:
    return SCHEMA_PATH.read_text()


def _ensure_parent(db_path: str) -> None:
    if db_path != ":memory:" and not db_path.startswith("file:"):
        Path(db_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


async def connect(db_path: str) -> aiosqlite.Connection:
    """Open an aiosqlite connection with pragmas applied and the schema ensured."""
    _ensure_parent(db_path)
    conn = await aiosqlite.connect(db_path, isolation_level=None)
    conn.row_factory = aiosqlite.Row
    for pragma in _PRAGMAS:
        await conn.execute(pragma)
    await conn.executescript(schema_sql())
    return conn


def connect_sync(db_path: str) -> sqlite3.Connection:
    """Plain sqlite3 connection (dashboard / review helpers) with the same pragmas and schema."""
    _ensure_parent(db_path)
    conn = sqlite3.connect(db_path, isolation_level=None, timeout=BUSY_TIMEOUT_MS / 1000)
    conn.row_factory = sqlite3.Row
    for pragma in _PRAGMAS:
        conn.execute(pragma)
    conn.executescript(schema_sql())
    return conn
