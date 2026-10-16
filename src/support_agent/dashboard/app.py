"""SupportAgent ops dashboard (docs/PLAN.md §5.12).

Run:  support-agent dashboard            (= streamlit run src/support_agent/dashboard/app.py)
DB:   settings.db_path, override with env SA_DB_PATH. Opened read-only; review / requeue writes go
      through support_agent.storage.review.
"""

from __future__ import annotations

from datetime import UTC, datetime

import streamlit as st

from support_agent.dashboard import data

st.set_page_config(
    page_title="SupportAgent Ops",
    page_icon=":material/support_agent:",
    layout="wide",
    initial_sidebar_state="expanded",
)

db_path = data.resolve_db_path()
review_title = "Human Review"
if data.db_status(db_path) == "ok":
    try:
        n = data.state_counts(db_path).get("NEEDS_REVIEW", 0)
        if n:
            review_title = f"Human Review ({n})"
    except Exception:  # never block navigation on a stats query
        pass

pages = {
    "Monitor": [
        st.Page("views/overview.py", title="Overview", icon=":material/dashboard:", default=True),
        st.Page(
            "views/tickets.py", title="Tickets", icon=":material/confirmation_number:", url_path="tickets"
        ),
    ],
    "Act": [
        st.Page("views/review.py", title=review_title, icon=":material/rate_review:", url_path="review"),
    ],
    "System": [
        st.Page(
            "views/queue_workers.py", title="Queue & Workers", icon=":material/stacks:", url_path="queue"
        ),
        st.Page("views/llm.py", title="LLM", icon=":material/psychology:", url_path="llm"),
        st.Page(
            "views/scheduler.py", title="Scheduler & Poller", icon=":material/schedule:", url_path="scheduler"
        ),
        st.Page("views/logs.py", title="Logs", icon=":material/receipt_long:", url_path="logs"),
    ],
}
nav = st.navigation(pages)

with st.sidebar:
    st.toggle("Auto-refresh every 5 s", value=True, key="auto_refresh")
    st.caption(f":material/database: `{db_path}`")
    st.caption(f"Loaded {datetime.now(UTC).strftime('%H:%M:%S')} UTC")

nav.run()
