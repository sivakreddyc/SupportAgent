"""Logs: tail of the JSON-lines log file with level / component / ticket filters."""

from __future__ import annotations

import json

import streamlit as st

from support_agent.dashboard import data, ui

st.title("Logs")
log_path = data.resolve_log_path()
LEVELS = ["debug", "info", "warning", "error", "critical"]
CORE = ["ts", "level", "component", "event", "ticket_id", "trace_id", "worker_id"]

f = st.columns([1, 2, 2, 2, 3])
max_lines = f[0].selectbox("Lines", [200, 1000, 5000], index=1, key="logs_lines")
levels = f[1].multiselect("Level", LEVELS, key="logs_levels", placeholder="All levels")
_initial, _ = data.read_log_tail(log_path, max_lines=5000)
_components = sorted(c for c in _initial.get("component", []).dropna().unique()) if not _initial.empty else []
chosen = f[2].multiselect("Component", _components, key="logs_components", placeholder="All components")
ticket = f[3].text_input("Ticket id", key="logs_ticket", placeholder="e.g. TCK-1001").strip()
text = f[4].text_input("Search", key="logs_text", placeholder="Text in event or fields").strip().lower()


@st.fragment(run_every=ui.refresh_every())
def body() -> None:
    df, bad = data.read_log_tail(log_path, max_lines=max_lines)
    if df.empty:
        st.info(
            f"No log lines at `{log_path}` yet. The runtime writes JSON lines there "
            "(override with `SA_LOG_FILE`).",
            icon=":material/receipt_long:",
        )
        return
    df["level"] = df["level"].fillna("").astype(str).str.lower()
    view = df
    if levels:
        view = view[view["level"].isin(levels)]
    if chosen:
        view = view[view["component"].isin(chosen)]
    if ticket:
        view = view[view["ticket_id"].fillna("").astype(str).str.contains(ticket, case=False, regex=False)]
    extra_cols = [c for c in view.columns if c not in CORE]
    view = view.assign(
        fields=[
            json.dumps({k: v for k, v in r.items() if v is not None and v == v}, default=str)
            for r in view[extra_cols].to_dict("records")
        ]
        if extra_cols
        else ""
    )
    if text:
        hay = (view["event"].fillna("").astype(str) + " " + view["fields"]).str.lower()
        view = view[hay.str.contains(text, regex=False)]
    counts = df["level"].value_counts()
    st.caption(
        f"Showing {len(view)} of the last {len(df)} lines · "
        + " · ".join(f"{lvl} {counts.get(lvl, 0)}" for lvl in ("error", "warning", "info") if counts.get(lvl))
        + (f" · {bad} unparseable line(s) skipped" if bad else "")
        + f" · `{log_path}`"
    )
    st.dataframe(
        view.sort_values("ts", ascending=False)[CORE + ["fields"]],
        hide_index=True,
        height=560,
        column_config={
            "ts": st.column_config.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm:ss.SSS"),
            "fields": st.column_config.TextColumn("Fields", width="large"),
        },
    )


body()
