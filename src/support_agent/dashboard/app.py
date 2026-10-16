"""SupportAgent console: one page, four tabs (docs/TASKS.md Phase 5).

    support-agent dashboard        (= streamlit run src/support_agent/dashboard/app.py)

Tabs: Observability · Tickets (incl. review queue) · User Info · SQL (read-only).
Tabs are lazy (`on_change="rerun"`): only the selected tab's code runs, and the selection is kept in the
URL (`?tab=SQL`). Inside a tab, each section is its own `st.fragment`; only Observability and the
ticket list auto-refresh. DB: settings.db_path, override with env SA_DB_PATH (opened read-only);
review / requeue writes go through support_agent.storage.review.
"""

from __future__ import annotations

from datetime import UTC, datetime

import streamlit as st

from support_agent.common.config import get_settings
from support_agent.dashboard import data, ui
from support_agent.dashboard.views import observability, sql, tickets, users

TABS = ["Observability", "Tickets", "User Info", "SQL"]

st.set_page_config(
    page_title="SupportAgent Console",
    page_icon=":material/support_agent:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

settings = get_settings()
db_path = data.resolve_db_path()
log_path = data.resolve_log_path()

with st.sidebar:
    st.subheader("Settings")
    st.toggle(
        "Auto-refresh every 5 s", value=True, key="auto_refresh", help="Observability and the ticket list"
    )
    st.caption(f":material/database: DB (read-only)  \n`{db_path}`")
    st.caption(f":material/person_search: User service  \n`{settings.user_service_url}`")
    st.caption(f":material/receipt_long: Log file  \n`{log_path}`")

head = st.columns([3, 2], vertical_alignment="bottom")
head[0].title("SupportAgent Console")
status = data.db_status(db_path)
if status == "ok":
    try:
        counts = data.state_counts(db_path)
        head[1].caption(
            f":material/confirmation_number: {sum(counts.values())} tickets · "
            f":material/rate_review: {counts.get('NEEDS_REVIEW', 0)} awaiting review · "
            f":material/dangerous: {counts.get('DEAD_LETTER', 0)} dead letter · "
            f"loaded {datetime.now(UTC).strftime('%H:%M:%S')} UTC"
        )
    except Exception:  # header stats must never break the page
        pass

obs_tab, tickets_tab, users_tab, sql_tab = st.tabs(TABS, key="tab", on_change="rerun", bind="query-params")

with obs_tab:
    if obs_tab.open:
        observability.render(db_path)
with tickets_tab:
    if tickets_tab.open and ui.db_ready(db_path):
        tickets.render(db_path)
with users_tab:
    if users_tab.open:
        users.render(db_path if status == "ok" else None, settings)
with sql_tab:
    if sql_tab.open and ui.db_ready(db_path):
        sql.render(db_path)
