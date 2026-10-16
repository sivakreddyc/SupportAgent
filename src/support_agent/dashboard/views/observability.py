"""Observability tab: KPIs, throughput, latency, queue & workers, LLM, scheduler & poller, logs.

Reuses the section renderers in overview / queue_workers / llm / scheduler / logs. Sub-tabs are lazy
(only the selected one runs), and each section is its own auto-refreshing fragment.
"""

from __future__ import annotations

import streamlit as st

from support_agent.dashboard import data, ui
from support_agent.dashboard.views import llm, logs, overview, queue_workers, scheduler

SECTIONS = ["Overview", "Queue & Workers", "LLM", "Scheduler & Poller", "Logs"]


def render(db: str) -> None:
    top = st.columns([3, 2], vertical_alignment="center")
    with top[0]:
        since = ui.time_range("obs_range")
    top[1].caption(
        ":material/autorenew: refreshes every 5 s"
        if ui.refresh_every()
        else ":material/pause: auto-refresh off"
    )
    tabs = st.tabs(SECTIONS, key="obs_section", on_change="rerun")
    with tabs[4]:
        if tabs[4].open:
            logs.render(data.resolve_log_path())  # logs work even without a DB
    if not any(t.open for t in tabs[:4]):
        return
    if not ui.db_ready(db):
        return
    for tab, section in zip(tabs[:4], (overview, queue_workers, llm, scheduler), strict=True):
        with tab:
            if tab.open:
                section.render(db, since)
