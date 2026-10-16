"""SQL tab: read-only query console over the SupportAgent SQLite DB.

Guard + execution live in `dashboard.data` (`check_readonly_sql`, `run_readonly_query`): one SELECT /
WITH / VALUES / EXPLAIN / read-only PRAGMA statement, on a `mode=ro` + `query_only` connection, first
100 rows by default (`fetchmany`), 5 s timeout. Not auto-refreshed, so editing is never interrupted.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pandas as pd
import streamlit as st

from support_agent.dashboard import data, ui

PAGE_SIZES = [100, 500, 1000, 5000]
HISTORY_MAX = 25

_LATEST_TR = "(SELECT ticket_id FROM ticket_transitions ORDER BY id DESC LIMIT 1)"
_LATEST_SPAN = "(SELECT ticket_id FROM spans WHERE ticket_id IS NOT NULL ORDER BY id DESC LIMIT 1)"
EXAMPLES: dict[str, str] = {
    "Tickets by state": "SELECT state, count(*) AS tickets\nFROM tickets\nGROUP BY state\nORDER BY tickets DESC",
    "Latest tickets": (
        "SELECT ticket_id, state, subject, customer_email, updated_at\nFROM tickets\nORDER BY updated_at DESC"
    ),
    "One ticket's transitions": (
        "-- replace the subquery with a ticket id, e.g. 'TCK-1050'\n"
        "SELECT at, from_state, to_state, worker_id, detail_json\nFROM ticket_transitions\n"
        f"WHERE ticket_id = {_LATEST_TR}\nORDER BY id"
    ),
    "Escalation reasons": (
        "SELECT j.value AS rule, count(*) AS tickets\nFROM policy_decisions p, json_each(p.reasons_json) j\n"
        "GROUP BY j.value\nORDER BY tickets DESC"
    ),
    "Review queue (high priority first)": (
        "SELECT t.ticket_id, p.priority, p.reasons_json, t.subject, t.updated_at\nFROM tickets t\n"
        "JOIN policy_decisions p USING (ticket_id)\nWHERE t.state = 'NEEDS_REVIEW'\n"
        "ORDER BY p.priority = 'high' DESC, t.updated_at"
    ),
    "Drafts by status": (
        "SELECT source, status, count(*) AS drafts\nFROM drafts\nGROUP BY source, status\nORDER BY drafts DESC"
    ),
    "LLM token usage by model": (
        "SELECT model, count(*) AS calls,\n       sum(input_tokens) AS input_tokens,\n"
        "       sum(output_tokens) AS output_tokens,\n       sum(cache_read_tokens) AS cache_read_tokens,\n"
        "       round(avg(latency_ms)) AS avg_latency_ms\nFROM llm_calls\nGROUP BY model"
    ),
    "Spans for a ticket": (
        "-- replace the subquery with a ticket id\n"
        "SELECT name, start_ts, round(duration_ms, 1) AS ms, status, attrs_json\nFROM spans\n"
        f"WHERE ticket_id = {_LATEST_SPAN}\nORDER BY start_ts"
    ),
    "Slowest tickets (end-to-end)": (
        "SELECT ticket_id, state,\n"
        "       round((julianday(finished_at) - julianday(discovered_at)) * 86400, 2) AS e2e_seconds\n"
        "FROM tickets\nWHERE finished_at IS NOT NULL\nORDER BY e2e_seconds DESC"
    ),
    "Failures and dead letters": (
        "SELECT ticket_id, state, attempt, last_error, updated_at\nFROM tickets\n"
        "WHERE state IN ('FAILED', 'DEAD_LETTER')\nORDER BY updated_at DESC"
    ),
    "Recent review actions": "SELECT at, ticket_id, action, reviewer\nFROM review_actions\nORDER BY id DESC",
    "Table columns (PRAGMA)": "PRAGMA table_info(tickets)",
    "Query plan (EXPLAIN)": "EXPLAIN QUERY PLAN\nSELECT * FROM tickets WHERE state = 'NEEDS_REVIEW'",
}


def _use_example() -> None:
    name = st.session_state.get("sql_example")
    if name in EXAMPLES:
        st.session_state["sql_text"] = EXAMPLES[name]


def _use_history() -> None:
    sql = st.session_state.get("sql_history_pick")
    if sql:
        st.session_state["sql_text"] = sql


def _execute(db: str, sql: str, rows: int) -> None:
    """Run and store the outcome in session state (so plain reruns don't re-execute the query)."""
    hist = st.session_state.setdefault("sql_history", [])
    entry = {
        "at": datetime.now(UTC).strftime("%H:%M:%S"),
        "sql": sql.strip(),
        "rows": None,
        "ms": None,
        "status": "ok",
    }
    try:
        res = data.run_readonly_query(db, sql, limit=rows)
    except data.SqlGuardError as e:
        st.session_state["sql_result"], st.session_state["sql_error"] = None, ("guard", str(e))
        entry["status"] = "rejected"
    except data.SqlTimeout as e:
        st.session_state["sql_result"], st.session_state["sql_error"] = None, ("timeout", str(e))
        entry["status"] = "timeout"
    except sqlite3.Error as e:
        st.session_state["sql_result"], st.session_state["sql_error"] = (
            None,
            ("sqlite", f"{type(e).__name__}: {e}"),
        )
        entry["status"] = "error"
    else:
        st.session_state["sql_result"], st.session_state["sql_error"] = res, None
        st.session_state["sql_last"] = {"sql": sql, "rows": rows}
        entry["rows"], entry["ms"] = len(res.rows), res.elapsed_ms
    if not hist or hist[0]["sql"] != entry["sql"] or hist[0]["status"] != entry["status"]:
        hist.insert(0, entry)
        del hist[HISTORY_MAX:]


def _run(db: str) -> None:
    sql = st.session_state.get("sql_text") or ""
    _execute(db, sql, int(st.session_state.get("sql_page_size", PAGE_SIZES[0])))


def _load_more(db: str) -> None:
    last = st.session_state.get("sql_last")
    if last:
        step = int(st.session_state.get("sql_page_size", PAGE_SIZES[0]))
        _execute(db, last["sql"], int(last["rows"]) + step)


def _schema(db: str) -> None:
    st.markdown("**Schema**")
    try:
        tables = data.schema_overview(db)
    except sqlite3.Error as e:
        st.error(f"Could not read the schema: {e}")
        return
    for t in tables:
        rows = "?" if t["rows"] is None else f"{t['rows']:,}"
        with st.expander(f"`{t['name']}` · {rows} rows", expanded=False):
            st.dataframe(
                pd.DataFrame(t["columns"]),
                hide_index=True,
                column_config={"pk": st.column_config.CheckboxColumn("PK"), "notnull": "NOT NULL"},
            )


def _results(db: str) -> None:
    err = st.session_state.get("sql_error")
    if err:
        kind, msg = err
        if kind == "guard":
            st.warning(msg, icon=":material/block:")
        elif kind == "timeout":
            st.warning(msg, icon=":material/timer_off:")
        else:
            st.error(msg, icon=":material/error:")
        return
    res: data.QueryResult | None = st.session_state.get("sql_result")
    if res is None:
        ui.empty("Run a query to see results. Pick an example above to get started.")
        return
    status = st.columns([4, 1, 1], vertical_alignment="center")
    status[0].caption(
        f"Showing {len(res.rows):,} row(s)"
        + (" (more available)" if res.has_more else "")
        + f" · {len(res.columns)} column(s) · {res.elapsed_ms:,.1f} ms"
    )
    step = int(st.session_state.get("sql_page_size", PAGE_SIZES[0]))
    if res.has_more:
        status[1].button(
            f"Load next {step}",
            icon=":material/expand_more:",
            on_click=_load_more,
            args=(db,),
            key="sql_more",
        )
    df = res.frame()
    if not df.empty or res.columns:
        status[2].download_button(
            "CSV",
            df.to_csv(index=False).encode(),
            file_name="query.csv",
            mime="text/csv",
            icon=":material/download:",
            on_click="ignore",
            key="sql_csv",
        )
    if not res.columns:
        st.caption("The statement returned no columns.")
        return
    st.dataframe(df, hide_index=True, height=min(38 + 35 * max(len(df), 1), 520))
    st.caption(
        "Column types: " + " · ".join(f"`{c}` {t}" for c, t in zip(df.columns, res.types, strict=True))
    )


def _body(db: str) -> None:
    st.info(
        "Read-only console: one `SELECT` / `WITH` / `EXPLAIN` / read-only `PRAGMA` statement at a time, "
        f"first {PAGE_SIZES[0]} rows, {data.SQL_TIMEOUT_S:.0f} s limit. To approve, edit or reject drafts use "
        "the **Tickets** tab (Review queue).",
        icon=":material/lock:",
    )
    main, side = st.columns([3, 1], gap="large")
    with main:
        top = st.columns([3, 1], vertical_alignment="bottom")
        top[0].selectbox(
            "Examples",
            list(EXAMPLES),
            index=None,
            placeholder="Pick an example query…",
            key="sql_example",
            on_change=_use_example,
        )
        top[1].selectbox("Rows", PAGE_SIZES, key="sql_page_size")
        if "sql_text" not in st.session_state:
            st.session_state["sql_text"] = EXAMPLES["Tickets by state"]
        st.text_area(
            "SQL",
            key="sql_text",
            height=170,
            label_visibility="collapsed",
            placeholder="SELECT * FROM tickets",
        )
        st.button(
            "Run", icon=":material/play_arrow:", type="primary", on_click=_run, args=(db,), key="sql_run"
        )
        _results(db)
        hist = st.session_state.get("sql_history") or []
        if hist:
            with st.expander(f"History ({len(hist)})"):
                st.dataframe(pd.DataFrame(hist), hide_index=True)
                st.selectbox(
                    "Re-use a previous query",
                    [h["sql"] for h in hist],
                    index=None,
                    key="sql_history_pick",
                    on_change=_use_history,
                    format_func=lambda s: s.replace("\n", " ")[:120],
                )
    with side:
        _schema(db)


def render(db: str) -> None:
    """SQL tab (not auto-refreshed)."""
    st.fragment(_body)(db)
