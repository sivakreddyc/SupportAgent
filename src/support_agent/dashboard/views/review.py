"""Human Review: pending NEEDS_REVIEW tickets (high priority first) with Approve / Edit & Approve / Reject.

Writes go through support_agent.storage.review.apply_review (imported lazily in the handlers); the
runtime's review dispatcher posts approved drafts to the ticket UI.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from support_agent.dashboard import data, ui

st.title("Human Review")
db = ui.require_db()

ACTION_LABEL = {"approve": "approved", "edit": "edited and approved", "reject": "rejected"}


def _apply(ticket_id: str, action: str, text_key: str, draft_body: str) -> None:
    """Button callback. Reads reviewer + editor text from session state (callback args are bound at
    render time, so text typed just before the click would otherwise be stale)."""
    from support_agent.common.states import InvalidTransition
    from support_agent.storage.review import ReviewError, TicketNotFound, apply_review

    reviewer = (st.session_state.get("reviewer_name") or "").strip()
    text = st.session_state.get(text_key, draft_body) or ""
    edited_body = text if action == "edit" else None
    if not reviewer:
        st.session_state["review_flash"] = ("warning", "Enter your name in **Reviewer** before reviewing.")
        return
    if action == "edit" and not text.strip():
        st.session_state["review_flash"] = (
            "warning",
            "The reply is empty. Write a reply, or use **Reject**.",
        )
        return
    if action == "approve" and text.strip() != draft_body.strip():
        st.session_state["review_flash"] = (
            "warning",
            "You changed the draft. Use **Edit & Approve** to send your version.",
        )
        return
    try:
        apply_review(db, ticket_id, action, reviewer, edited_body=edited_body)
    except TicketNotFound:
        st.session_state["review_flash"] = ("error", f"{ticket_id} no longer exists.")
    except InvalidTransition:
        st.session_state["review_flash"] = (
            "warning",
            f"{ticket_id} is no longer waiting for review — someone else probably handled it.",
        )
    except ReviewError as e:
        st.session_state["review_flash"] = ("error", f"Could not apply this review: {e}")
    else:
        st.session_state["review_flash"] = (
            "success",
            f"{ticket_id} {ACTION_LABEL[action]} by {reviewer}."
            + (" The runtime will post the reply shortly." if action != "reject" else ""),
        )
        st.session_state.pop("review_pick", None)


@st.fragment(run_every=ui.refresh_every())
def queue_panel() -> None:
    q = data.pending_reviews(db)
    c = st.columns(3)
    c[0].metric("Pending", len(q), border=True)
    c[1].metric("High priority", int((q["priority"] == "high").sum()) if not q.empty else 0, border=True)
    if not q.empty:
        oldest = data.to_ts(q["updated_at"]).min()
        wait = (pd.Timestamp.now(tz="UTC") - oldest).total_seconds() / 60
        c[2].metric("Oldest waiting", ui.fmt_num(wait, " min", 0), border=True)
    else:
        c[2].metric("Oldest waiting", "—", border=True)
    if q.empty:
        ui.empty("Nothing waiting for review. Escalated tickets will appear here.")
        return
    view = q.assign(reasons=[", ".join(r) for r in q["reasons"]], waiting_since=data.to_ts(q["updated_at"]))
    st.dataframe(
        view,
        hide_index=True,
        height=min(36 + 35 * len(view), 280),
        column_order=["priority", "ticket_id", "subject", "reasons", "customer_email", "waiting_since"],
        column_config={
            "priority": "Priority",
            "ticket_id": "Ticket",
            "waiting_since": st.column_config.DatetimeColumn("Waiting since", format="HH:mm:ss"),
        },
    )
    if set(q["ticket_id"]) != set(st.session_state.get("review_known_ids", [])):
        st.caption(":material/sync: The queue changed. Use **Refresh** to update the ticket picker.")


flash = st.session_state.pop("review_flash", None)
if flash:
    getattr(st, flash[0])(flash[1])

pending = data.pending_reviews(db)
st.session_state["review_known_ids"] = pending["ticket_id"].tolist()
queue_panel()
if pending.empty:
    st.stop()

top = st.columns([2, 2, 1], vertical_alignment="bottom")
reviewer = top[0].text_input("Reviewer", key="reviewer_name", placeholder="Your name (required)").strip()
labels = {
    r.ticket_id: f"{'🔴 ' if r.priority == 'high' else ''}{r.ticket_id} · {r.subject[:60]}"
    for r in pending.itertuples()
}
ids = list(labels)
if st.session_state.get("review_pick") not in ids:
    st.session_state["review_pick"] = ids[0]
tid = top[1].selectbox("Ticket", ids, format_func=labels.get, key="review_pick")
top[2].button("Refresh", icon=":material/refresh:", width="stretch")

row = pending.set_index("ticket_id").loc[tid]
detail = data.ticket_detail(db, tid) or {}
draft_body = row["draft_body"] if isinstance(row["draft_body"], str) else ""
has_draft = bool(draft_body.strip())

st.divider()
left, right = st.columns([3, 2], gap="large")
with right:
    st.markdown("**Why this needs review**")
    st.pills(
        "Rules fired",
        row["reasons"] or ["(none recorded)"],
        key=f"rv_reasons_{tid}",
        disabled=True,
        label_visibility="collapsed",
    )
    if row["priority"] == "high":
        st.caption(":material/priority_high: High priority")
    snap = detail.get("snapshot")
    st.markdown(f"**Customer message** · `{row['customer_email']}`")
    with st.container(border=True, height=260):
        st.markdown(f"**{row['subject']}**")
        st.text(snap["body"] if snap else "(ticket body not fetched)")
        for m in (snap or {}).get("messages") or []:
            st.caption(f"{m.get('author')} · {m.get('at')}")
            st.text(m.get("body"))
    ul = detail.get("user_lookup")
    if ul and ul.get("found") and ul.get("user"):
        u = ul["user"]
        st.caption(
            f":material/person: {u.get('name')} · {u.get('plan')} · {u.get('account_status')}"
            + (" · VIP" if u.get("vip") else "")
            + f" · LTV ${u.get('lifetime_value', 0):,.0f} · {u.get('recent_refunds', 0)} recent refunds"
        )
    else:
        st.caption(":material/person_off: No user record for this email")
    llm = detail.get("llm_calls")
    if llm is not None and not llm.empty:
        last = llm.iloc[-1]
        if isinstance(last["analysis_json"], str):
            a = json.loads(last["analysis_json"])
            st.caption(
                f":material/psychology: {a['category']} · {a['sentiment']} · urgency {a['urgency']} · "
                f"confidence {a['confidence']:.2f}"
            )
            with st.expander("Model reasoning summary"):
                st.write(a.get("reasoning_summary"))
                if a.get("suggested_actions"):
                    st.markdown("\n".join(f"- {x}" for x in a["suggested_actions"]))
        else:
            st.caption(f":material/psychology: No usable analysis (stop_reason `{last['stop_reason']}`)")

with left:
    st.markdown("**Draft reply**")
    if not has_draft:
        st.info(
            "There is no usable AI draft for this ticket (the model refused or returned no analysis). "
            "Write a reply below and use **Edit & Approve**, or **Reject**.",
            icon=":material/edit_note:",
        )
    text_key = f"rv_text_{tid}_{row['draft_id']}"
    edited = st.text_area(
        "Draft reply",
        value=draft_body,
        height=320,
        key=text_key,
        label_visibility="collapsed",
    )
    changed = edited.strip() != draft_body.strip()
    if changed:
        st.caption(":material/edit: You changed the draft. Use **Edit & Approve** to send your version.")
    if not reviewer:
        st.caption(":material/badge: Enter your name in **Reviewer** above before reviewing.")
    b = st.columns(3)
    b[0].button(
        "Approve",
        icon=":material/check:",
        type="primary",
        disabled=not has_draft,
        help="Send the AI draft as is" if has_draft else "No draft to approve; use Edit & Approve",
        on_click=_apply,
        args=(tid, "approve", text_key, draft_body),
        key=f"rv_approve_{tid}",
        width="stretch",
    )
    b[1].button(
        "Edit & Approve",
        icon=":material/edit:",
        help="Send the text in the editor (saved as a human-edited draft)",
        on_click=_apply,
        args=(tid, "edit", text_key, draft_body),
        key=f"rv_edit_{tid}",
        width="stretch",
    )
    b[2].button(
        "Reject",
        icon=":material/close:",
        help="Close without sending a reply",
        on_click=_apply,
        args=(tid, "reject", text_key, draft_body),
        key=f"rv_reject_{tid}",
        width="stretch",
    )

recent = data.recent_reviews(db, 10)
if not recent.empty:
    with st.expander("Recent review actions"):
        st.dataframe(recent, hide_index=True)
