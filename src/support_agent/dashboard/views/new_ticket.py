"""Tickets → New ticket: create a ticket in the mock ticket UI and watch the agent process it.

The ticket is created the way a customer would create it: it's posted to the mock ticket UI
(`settings.ticket_ui_url`, logging in with the ticket-UI credentials). The agent then discovers it
through the browser like any other ticket. "Poll now" calls the runtime's ops API (`POST /api/poll`)
so you don't have to wait for the next scheduled poll. Nothing is pushed into the queue directly,
because that would skip the discovery step being tested.
"""

from __future__ import annotations

import re
from typing import Any

import httpx
import streamlit as st

from support_agent.common.config import Settings
from support_agent.common.states import TERMINAL_STATES, TicketState
from support_agent.dashboard import data, ui

# Quick-fill templates. Emails belong to seed users in the mock User Info Service, except "Unknown customer".
TEMPLATES: dict[str, dict[str, str]] = {
    "Custom (blank)": {"customer_email": "", "subject": "", "body": "", "priority": ""},
    "How-to question (auto-approve)": {
        "customer_email": "maria.gonzalez@example.com",
        "subject": "How do I export my board to CSV?",
        "body": "Hi, I'd like to export all the tasks on my Marketing board to a CSV file for a report. "
        "Where can I find that option? Thanks!",
        "priority": "",
    },
    "Small refund": {
        "customer_email": "dev.patel@example.com",
        "subject": "Charged twice this month",
        "body": "Hello, I was charged $12 twice for my subscription this month. Could I get a refund "
        "for the duplicate charge please?",
        "priority": "normal",
    },
    "Large refund + angry": {
        "customer_email": "maria.gonzalez@example.com",
        "subject": "WORST SERVICE EVER - refund my $480 NOW",
        "body": "This is ridiculous!!! You billed me $480 for the annual plan after I cancelled. "
        "I want my money back immediately. Unacceptable.",
        "priority": "high",
    },
    "Legal threat": {
        "customer_email": "dev.patel@example.com",
        "subject": "Formal GDPR data request",
        "body": "Under GDPR I am making a data subject request for all personal data you hold about me. "
        "If I don't hear back within 30 days my lawyer will be in touch.",
        "priority": "high",
    },
    "Security incident": {
        "customer_email": "maria.gonzalez@example.com",
        "subject": "I think my account was hacked",
        "body": "I got a login alert from a country I've never been to and my password no longer works. "
        "I think my account has been compromised. Please help.",
        "priority": "urgent",
    },
    "Enterprise (VIP) customer": {
        "customer_email": "it-admin@northwind-logistics.example",
        "subject": "Setting up SSO for our team",
        "body": "We'd like to enable single sign-on for our organisation. Can you point us to the setup "
        "guide for SAML?",
        "priority": "normal",
    },
    "Suspended account": {
        "customer_email": "brian.foster@example.net",
        "subject": "Can't log in to my workspace",
        "body": "Since yesterday I can't access my workspace at all. Can you tell me what's going on?",
        "priority": "",
    },
    "Unknown customer": {
        "customer_email": "new.prospect@unknown-company.example",
        "subject": "Question about team pricing",
        "body": "Hi, we're a team of 25 evaluating your product. Do you offer volume discounts on the pro plan?",
        "priority": "",
    },
    "LLM refusal (mock marker)": {
        "customer_email": "maria.gonzalez@example.com",
        "subject": "Question about my workspace",
        "body": "Can you help me with my workspace settings? [mock:refusal]",
        "priority": "",
    },
}
PRIORITIES = ["", "low", "normal", "high", "urgent"]
STAGES = [s.value for s in TicketState]
_TICKET_ID = re.compile(r"/tickets/(TCK-\d+)")


class CreateError(Exception):
    pass


def create_ticket(settings: Settings, *, customer_email: str, subject: str, body: str, priority: str) -> str:
    """Create a ticket in the mock ticket UI through its HTML form. Returns the new ticket id."""
    base = settings.ticket_ui_url.rstrip("/")
    try:
        with httpx.Client(base_url=base, timeout=10, follow_redirects=False) as c:
            login = c.post(
                "/login",
                data={
                    "username": settings.ticket_ui_user,
                    "password": settings.ticket_ui_password,
                    "next": "/tickets",
                },
            )
            if login.status_code != 303:
                raise CreateError(f"Login to the ticket UI failed (HTTP {login.status_code}).")
            r = c.post(
                "/tickets/new",
                data={
                    "customer_email": customer_email,
                    "subject": subject,
                    "body": body,
                    "priority": priority,
                },
            )
    except httpx.HTTPError as e:
        raise CreateError(
            f"The ticket UI at `{base}` is not reachable ({type(e).__name__}). Is the stack running?"
        ) from e
    if r.status_code == 303 and (m := _TICKET_ID.search(r.headers.get("location", ""))):
        return m.group(1)
    if r.status_code == 400:
        err = re.search(r'data-testid="new-ticket-error">([^<]+)<', r.text)
        raise CreateError(err.group(1) if err else "The ticket UI rejected the ticket.")
    raise CreateError(f"Unexpected response from the ticket UI (HTTP {r.status_code}).")


def poll_now(settings: Settings) -> dict[str, Any]:
    """Ask the runtime to run a discovery poll now. Returns the poll result."""
    host = "127.0.0.1" if settings.ops_api_host in ("0.0.0.0", "") else settings.ops_api_host
    url = f"http://{host}:{settings.ops_api_port}/api/poll"
    try:
        r = httpx.post(url, timeout=35)
    except httpx.HTTPError as e:
        raise CreateError(
            f"The agent runtime's ops API at `{url}` is not reachable ({type(e).__name__})."
        ) from e
    if r.status_code == 409:
        raise CreateError("A poll is already running; the ticket will be picked up by it or the next one.")
    if r.status_code != 200:
        raise CreateError(f"Poll request failed (HTTP {r.status_code}).")
    return r.json().get("result") or {}


def _apply_template() -> None:
    tpl = TEMPLATES[st.session_state["nt_template"]]
    st.session_state["nt_email"] = tpl["customer_email"]
    st.session_state["nt_subject"] = tpl["subject"]
    st.session_state["nt_body"] = tpl["body"]
    st.session_state["nt_priority"] = tpl["priority"]


def _open_ticket(ticket_id: str) -> None:
    st.session_state["tickets_pick"] = ticket_id
    st.session_state["tickets_section"] = "All tickets"
    st.query_params["ticket"] = ticket_id


def _progress(db: str, ticket_id: str) -> None:
    """Live view of one ticket moving through the pipeline (refreshes every second until it settles)."""
    detail = data.ticket_detail(db, ticket_id) if data.db_status(db) == "ok" else None
    if detail is None:
        st.info(
            f"**{ticket_id}** has not been discovered by the agent yet. "
            "It will appear here after the next poll.",
            icon=":material/hourglass_top:",
        )
        return
    t = detail["ticket"]
    state = t["state"]
    seen = list(detail["transitions"]["to_state"]) if not detail["transitions"].empty else [state]
    steps = [s for s in STAGES if s in seen]
    st.markdown(f"**{ticket_id}** · current state: `{state}`")
    st.markdown("  →  ".join(f"`{s}`" for s in steps))
    if state in {s.value for s in TERMINAL_STATES} or state == TicketState.NEEDS_REVIEW.value:
        policy = detail.get("policy") or {}
        reasons = ", ".join(policy.get("reasons") or []) or "none"
        if state == TicketState.NEEDS_REVIEW.value:
            st.warning(
                f"Escalated for human review (rules: {reasons}). Go to **Review queue** to approve, "
                "edit or reject the draft.",
                icon=":material/rate_review:",
            )
        elif state == TicketState.DONE.value:
            st.success(
                f"Done. Policy rules fired: {reasons}. The reply is posted on the ticket in the ticket UI.",
                icon=":material/check_circle:",
            )
        else:
            st.error(f"Ended in `{state}`: {t.get('last_error') or 'see the ticket detail'}.")
        st.button(
            "Open ticket detail", icon=":material/open_in_new:", on_click=_open_ticket, args=(ticket_id,)
        )


def render(db: str, settings: Settings) -> None:
    st.caption(
        "Create a ticket as a customer would. It's posted to the mock ticket UI "
        f"(`{settings.ticket_ui_url}`), then the agent discovers it through the browser and runs the full "
        "pipeline: queue → user lookup → LLM → policy → draft posted or sent to review."
    )
    st.selectbox("Start from a template", list(TEMPLATES), key="nt_template", on_change=_apply_template)
    with st.form("new_ticket_form", clear_on_submit=False, border=True):
        c1, c2 = st.columns([3, 1])
        email = c1.text_input("Customer email", key="nt_email", placeholder="name@example.com")
        priority = c2.selectbox(
            "Priority", PRIORITIES, key="nt_priority", format_func=lambda p: p or "(none)"
        )
        subject = st.text_input("Subject", key="nt_subject")
        body = st.text_area("Message", key="nt_body", height=140)
        poll = st.checkbox("Poll now after creating (don't wait for the next scheduled poll)", value=True)
        submitted = st.form_submit_button("Create ticket", type="primary", icon=":material/add:")

    if submitted:
        try:
            ticket_id = create_ticket(
                settings,
                customer_email=email.strip(),
                subject=subject.strip(),
                body=body.strip(),
                priority=priority,
            )
        except CreateError as e:
            st.error(str(e), icon=":material/error:")
        else:
            st.session_state["nt_created"] = ticket_id
            msg = f"Created **{ticket_id}** in the ticket UI."
            if poll:
                try:
                    res = poll_now(settings)
                    msg += (
                        f" Poll ran: found {res.get('found', '?')} open, {res.get('new', '?')} new, "
                        f"{res.get('enqueued', '?')} queued."
                    )
                except CreateError as e:
                    msg += f" (Poll now failed: {e})"
            ui.clear_cache()
            st.success(msg, icon=":material/check:")

    created = st.session_state.get("nt_created")
    if created:
        st.subheader("Pipeline progress")
        st.fragment(_progress, run_every=1)(db, created)
