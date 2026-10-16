-- SupportAgent SQLite schema (docs/PLAN.md §5.5). Timestamps are ISO-8601 UTC strings.
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS tickets (
    ticket_id       TEXT PRIMARY KEY,
    subject         TEXT NOT NULL,
    customer_email  TEXT NOT NULL,
    state           TEXT NOT NULL,
    attempt         INTEGER NOT NULL DEFAULT 0,
    priority        TEXT,
    ui_created_at   TEXT,
    discovered_at   TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    finished_at     TEXT,
    last_error      TEXT,
    trace_id        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tickets_state ON tickets(state);

CREATE TABLE IF NOT EXISTS ticket_transitions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id   TEXT NOT NULL REFERENCES tickets(ticket_id),
    from_state  TEXT,
    to_state    TEXT NOT NULL,
    at          TEXT NOT NULL,
    worker_id   TEXT,
    detail_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_transitions_ticket ON ticket_transitions(ticket_id);

CREATE TABLE IF NOT EXISTS ticket_snapshots (
    ticket_id     TEXT PRIMARY KEY REFERENCES tickets(ticket_id),
    body          TEXT NOT NULL,
    messages_json TEXT NOT NULL,
    fetched_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS user_lookups (
    ticket_id   TEXT PRIMARY KEY REFERENCES tickets(ticket_id),
    found       INTEGER NOT NULL,
    user_json   TEXT,
    latency_ms  REAL,
    at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id           TEXT NOT NULL REFERENCES tickets(ticket_id),
    model               TEXT,
    input_tokens        INTEGER NOT NULL DEFAULT 0,
    output_tokens       INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens   INTEGER NOT NULL DEFAULT 0,
    latency_ms          REAL,
    stop_reason         TEXT,
    success             INTEGER NOT NULL,
    error               TEXT,
    analysis_json       TEXT,
    at                  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_ticket ON llm_calls(ticket_id);

CREATE TABLE IF NOT EXISTS policy_decisions (
    ticket_id    TEXT PRIMARY KEY REFERENCES tickets(ticket_id),
    outcome      TEXT NOT NULL,
    reasons_json TEXT NOT NULL,
    priority     TEXT NOT NULL DEFAULT 'normal',
    at           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS drafts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id       TEXT NOT NULL REFERENCES tickets(ticket_id),
    body            TEXT NOT NULL,
    source          TEXT NOT NULL,          -- llm | human_edit
    status          TEXT NOT NULL,          -- pending_review | approved | posted | rejected | superseded (replaced by a human edit)
    idempotency_key TEXT NOT NULL UNIQUE,
    reviewer        TEXT,
    created_at      TEXT NOT NULL,
    posted_at       TEXT
);
CREATE INDEX IF NOT EXISTS idx_drafts_ticket ON drafts(ticket_id);

CREATE TABLE IF NOT EXISTS review_actions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id   TEXT NOT NULL REFERENCES tickets(ticket_id),
    action      TEXT NOT NULL,              -- approve | edit | reject
    reviewer    TEXT,
    edited_body TEXT,
    at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS spans (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id    TEXT,
    ticket_id   TEXT,
    name        TEXT NOT NULL,
    start_ts    TEXT NOT NULL,
    duration_ms REAL NOT NULL,
    status      TEXT NOT NULL,
    attrs_json  TEXT
);
CREATE INDEX IF NOT EXISTS idx_spans_ticket ON spans(ticket_id);
CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id);

CREATE TABLE IF NOT EXISTS metric_samples (
    ts          TEXT NOT NULL,
    name        TEXT NOT NULL,
    labels_json TEXT NOT NULL,
    value       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_metric_samples_name_ts ON metric_samples(name, ts);

CREATE TABLE IF NOT EXISTS scheduler_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    found         INTEGER,
    enqueued      INTEGER,
    skipped_full  INTEGER,
    error         TEXT
);
