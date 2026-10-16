"""Tickets tab: ticket list (100 rows by default) + per-ticket detail (timeline, spans, text, user,
LLM JSON, policy, drafts), DLQ requeue, and the Review queue sub-tab (views/review.py)."""

from __future__ import annotations

import json

import altair as alt
import pandas as pd
import streamlit as st

from support_agent.dashboard import data, ui

STATE_ICON = {
    "DONE": ":material/check_circle:",
    "DRAFT_POSTED": ":material/send:",
    "NEEDS_REVIEW": ":material/rate_review:",
    "APPROVED": ":material/thumb_up:",
    "REJECTED": ":material/block:",
    "FAILED": ":material/error:",
    "DEAD_LETTER": ":material/dangerous:",
}

PAGE_SIZES = [100, 500, 1000, 5000]


def _requeue(db: str, ticket_id: str) -> None:
    from support_agent.common.states import InvalidTransition
    from support_agent.storage.review import ReviewError, TicketNotFound, requeue_dead_letter

    try:
        requeue_dead_letter(db, ticket_id)
    except TicketNotFound:
        st.session_state["tickets_flash"] = ("error", f"{ticket_id} no longer exists.")
    except InvalidTransition:
        st.session_state["tickets_flash"] = ("warning", f"{ticket_id} is no longer in the dead-letter queue.")
    except ReviewError as e:
        st.session_state["tickets_flash"] = ("error", f"Could not requeue {ticket_id}: {e}")
    else:
        st.session_state["tickets_flash"] = (
            "success",
            f"{ticket_id} requeued. The runtime will pick it up shortly.",
        )
        ui.clear_cache()


def _detail(db: str, ticket_id: str) -> None:
    d = data.ticket_detail(db, ticket_id)
    if d is None:
        st.warning(f"Ticket `{ticket_id}` not found.")
        return
    t = d["ticket"]
    st.subheader(f"{STATE_ICON.get(t['state'], ':material/circle:')} {t['ticket_id']} · {t['subject']}")
    c = st.columns([2, 1, 1, 1, 1])
    c[0].metric("State", t["state"])
    c[1].metric("Attempts", t["attempt"])
    c[2].metric("Priority", (d["policy"] or {}).get("priority") or t.get("priority") or "—")
    disc, fin = data.to_ts(pd.Series([t["discovered_at"], t["finished_at"]]))
    c[3].metric("Discovered", disc.strftime("%H:%M:%S") if pd.notna(disc) else "—")
    c[4].metric("End-to-end", ui.fmt_num((fin - disc).total_seconds(), " s", 2) if pd.notna(fin) else "—")
    st.caption(f"Customer `{t['customer_email']}` · trace `{t['trace_id']}`")
    if t.get("last_error"):
        st.error(f"Last error: {t['last_error']}", icon=":material/error:")
    if t["state"] == "DEAD_LETTER":
        st.button(
            "Requeue",
            icon=":material/replay:",
            type="primary",
            key=f"requeue_{ticket_id}",
            on_click=_requeue,
            args=(db, ticket_id),
            help="Move back to QUEUED with attempts reset; the runtime re-processes it.",
        )

    tabs = st.tabs(
        ["Timeline", "Ticket text", "Customer", "LLM", "Policy & drafts"], key=f"detail_tabs_{ticket_id}"
    )
    with tabs[0]:
        tr = d["transitions"]
        if tr.empty:
            ui.empty("No transitions recorded.")
        else:
            view = tr.assign(
                transition=tr["from_state"].fillna("∅") + " → " + tr["to_state"],
                detail=[json.dumps(json.loads(x)) if isinstance(x, str) else "" for x in tr["detail_json"]],
            )[["at", "transition", "worker_id", "detail"]]
            st.dataframe(
                view,
                hide_index=True,
                column_config={"at": st.column_config.DatetimeColumn("At", format="HH:mm:ss.SSS")},
            )
        sp = d["spans"]
        st.markdown("**Spans**")
        if sp.empty:
            ui.empty("No spans recorded for this ticket.")
        else:
            sp = sp.assign(end_ts=sp["start_ts"] + pd.to_timedelta(sp["duration_ms"], unit="ms"))
            sp["label"] = [f"{i + 1:02d} {n}" for i, n in enumerate(sp["name"])]
            gantt = (
                alt.Chart(sp)
                .mark_bar(cornerRadius=3, height=14)
                .encode(
                    y=alt.Y("label:N", title=None, sort=None),
                    x=ui.time_x("start_ts"),
                    x2="end_ts:T",
                    color=alt.Color(
                        "status:N",
                        scale=alt.Scale(domain=["ok", "error"], range=[ui.SERIES[0], ui.STATUS["critical"]]),
                        title="Status",
                    ),
                    tooltip=[
                        "name",
                        "status",
                        alt.Tooltip("duration_ms:Q", format=",.1f", title="ms"),
                        "attrs_json",
                    ],
                )
                .properties(height=40 + 22 * len(sp))
            )
            ui.chart(gantt)
            ui.chart(
                ui.bars(
                    sp,
                    "label",
                    "duration_ms",
                    x_type="N",
                    horizontal=True,
                    y_title="Duration (ms)",
                    height=40 + 22 * len(sp),
                )
            )
    with tabs[1]:
        snap = d["snapshot"]
        if snap is None:
            ui.empty("The ticket body hasn't been fetched yet.")
        else:
            st.text_area("Body", snap["body"], height=180, disabled=True, key=f"body_{ticket_id}")
            for m in snap["messages"] or []:
                with st.chat_message("user" if m.get("author") not in ("agent", "support") else "assistant"):
                    st.caption(f"{m.get('author')} · {m.get('at')}")
                    st.write(m.get("body"))
            st.caption(f"Fetched {ui.fmt_ts(snap['fetched_at'])}")
    with tabs[2]:
        ul = d["user_lookup"]
        if ul is None:
            ui.empty("User lookup hasn't run yet.")
        elif not ul["found"]:
            st.warning("No user record found for this email (UNKNOWN_USER).", icon=":material/person_off:")
        else:
            st.json(ul["user"])
        if ul is not None:
            st.caption(f"Lookup latency {ui.fmt_num(ul['latency_ms'], ' ms', 1)}")
    with tabs[3]:
        llm = d["llm_calls"]
        if llm.empty:
            ui.empty("No LLM call yet.")
        for r in llm.itertuples():
            st.markdown(
                f"**Call {r.id}** · `{r.model}` · stop `{r.stop_reason}` · {ui.fmt_num(r.latency_ms, ' ms')} · "
                f"{r.input_tokens} in / {r.output_tokens} out / {r.cache_read_tokens} cache-read tokens"
            )
            if r.error:
                st.error(r.error)
            if r.analysis_json:
                st.json(json.loads(r.analysis_json))
            elif not r.error:
                st.warning(f"No structured analysis (stop_reason `{r.stop_reason}`).")
    with tabs[4]:
        pol = d["policy"]
        if pol is None:
            ui.empty("No policy decision yet.")
        else:
            st.markdown(f"**Outcome:** `{pol['outcome']}` · **priority** `{pol['priority']}`")
            if pol["reasons"]:
                st.pills("Rules fired", pol["reasons"], key=f"reasons_{ticket_id}", disabled=True)
            else:
                st.caption("No rule fired.")
        dr = d["drafts"]
        if dr.empty:
            ui.empty("No drafts.")
        for r in dr.itertuples():
            with st.container(border=True):
                st.caption(
                    f"Draft {r.id} · {r.source} · **{r.status}** · created {ui.fmt_ts(r.created_at)}"
                    + (f" · posted {ui.fmt_ts(r.posted_at)}" if r.posted_at else "")
                    + (f" · reviewer {r.reviewer}" if r.reviewer else "")
                )
                st.text(r.body or "(empty)")
        if not d["reviews"].empty:
            st.markdown("**Review actions**")
            st.dataframe(d["reviews"], hide_index=True)


def _reset_shown() -> None:
    st.session_state["tickets_shown"] = st.session_state.get("tickets_page_size", PAGE_SIZES[0])


def _load_more() -> None:
    size = st.session_state.get("tickets_page_size", PAGE_SIZES[0])
    st.session_state["tickets_shown"] = st.session_state.get("tickets_shown", size) + size


def _list_body(db: str) -> None:
    f = st.columns([2, 3, 1], vertical_alignment="bottom")
    states = f[0].multiselect(
        "State", data.ALL_STATES, key="tickets_states", placeholder="All states", on_change=_reset_shown
    )
    search = f[1].text_input(
        "Search",
        key="tickets_search",
        placeholder="Ticket id, subject or customer email",
        on_change=_reset_shown,
    )
    size = f[2].selectbox("Rows", PAGE_SIZES, key="tickets_page_size", on_change=_reset_shown)
    shown = max(int(st.session_state.get("tickets_shown", size)), size)

    flash = st.session_state.pop("tickets_flash", None)
    if flash:
        getattr(st, flash[0])(flash[1])
    total = data.tickets_count(db, states, search)
    df = data.tickets(db, states, search, limit=shown)
    if df.empty:
        ui.empty("No tickets match these filters." if (states or search) else "No tickets yet.")
        return
    more = total > len(df)
    cap = st.columns([4, 1], vertical_alignment="center")
    cap[0].caption(
        f"Showing {len(df)} of {total} ticket(s), most recently updated first"
        + (" · more available" if more else "")
    )
    if more:
        cap[1].button(
            f"Load {size} more", icon=":material/expand_more:", on_click=_load_more, key="tickets_more"
        )
    event = st.dataframe(
        df,
        hide_index=True,
        height=320,
        key="tickets_table",
        on_select="rerun",
        selection_mode="single-row",
        column_order=[
            "ticket_id",
            "state",
            "priority",
            "subject",
            "customer_email",
            "reasons",
            "attempt",
            "updated_at",
            "last_error",
        ],
        column_config={
            "ticket_id": "Ticket",
            "updated_at": st.column_config.DatetimeColumn("Updated", format="YYYY-MM-DD HH:mm:ss"),
            "reasons": "Policy reasons",
        },
    )
    ids = df["ticket_id"].tolist()
    # Apply a table click only when the selection itself changes, so an auto-refresh that shifts rows
    # never silently switches the open ticket. The picker is set before it renders.
    rows = tuple(getattr(getattr(event, "selection", None), "rows", None) or [])
    if rows != st.session_state.get("tickets_last_sel", ()):
        st.session_state["tickets_last_sel"] = rows
        if rows and rows[0] < len(ids):
            st.session_state["tickets_pick"] = ids[rows[0]]
    if "tickets_pick" not in st.session_state and st.query_params.get("ticket"):
        st.session_state["tickets_pick"] = st.query_params["ticket"]
    current = st.session_state.get("tickets_pick")
    options = ids if current is None or current in ids else [current, *ids]
    selected = st.selectbox(
        "Open ticket",
        options,
        index=None,
        placeholder="Select a row above or pick a ticket id",
        key="tickets_pick",
    )
    if selected:
        st.query_params["ticket"] = selected
        st.divider()
        _detail(db, selected)


def render(db: str) -> None:
    """Tickets tab: "All tickets" (auto-refreshing list + detail) and "Review queue" sub-tabs (lazy)."""
    from support_agent.common.config import get_settings
    from support_agent.dashboard.views import new_ticket, review

    all_tab, review_tab, new_tab = st.tabs(
        ["All tickets", "Review queue", "New ticket"], key="tickets_section", on_change="rerun"
    )
    with all_tab:
        if all_tab.open:
            st.fragment(_list_body, run_every=ui.refresh_every())(db)
    with review_tab:
        if review_tab.open:
            review.render(db)
    with new_tab:
        if new_tab.open:
            new_ticket.render(db, get_settings())
