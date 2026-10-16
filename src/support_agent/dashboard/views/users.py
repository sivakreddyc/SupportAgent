"""User Info tab: look up a customer in the User Info Service, browse users, see their tickets.

Talks to `settings.user_service_url` with `X-API-Key` (sync httpx, short timeout). Works without the
DB (tickets section hidden) and degrades gracefully when the service is down or lacks `GET /users`
(falls back to the distinct customer emails on tickets).
"""

from __future__ import annotations

from typing import Any

import httpx
import pandas as pd
import streamlit as st

from support_agent.common.config import Settings
from support_agent.dashboard import data, ui

PAGE_SIZES = [100, 500, 1000]
TIMEOUT_S = 3.0


class ServiceError(Exception):
    """User service unreachable or answered with an unexpected status."""


class ListNotSupported(ServiceError):
    """The running user service has no `GET /users` endpoint yet."""


def _client(settings: Settings) -> httpx.Client:
    return httpx.Client(
        base_url=settings.user_service_url.rstrip("/"),
        headers={"X-API-Key": settings.user_service_api_key},
        timeout=TIMEOUT_S,
    )


def _get(settings: Settings, path: str, params: dict[str, Any] | None = None) -> httpx.Response:
    try:
        with _client(settings) as c:
            r = c.get(path, params=params)
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        raise ServiceError(
            f"The User Info Service at `{settings.user_service_url}` is not reachable ({type(e).__name__}). "
            "Start it with `support-agent user-service`, or check `SA_USER_SERVICE_URL`."
        ) from e
    except httpx.HTTPError as e:
        raise ServiceError(f"User service request failed: {type(e).__name__}: {e}") from e
    if r.status_code == 401:
        raise ServiceError("The User Info Service rejected the API key (`SA_USER_SERVICE_API_KEY`).")
    return r


def fetch_user(settings: Settings, ident: str) -> dict[str, Any] | None:
    """By email if it contains '@', else by user id. None on 404."""
    ident = ident.strip()
    path = f"/users/by-email/{ident}" if "@" in ident else f"/users/{ident}"
    r = _get(settings, path)
    if r.status_code == 404:
        return None
    if r.status_code != 200:
        raise ServiceError(f"User service returned HTTP {r.status_code} for `{path}`.")
    return r.json()


def fetch_orders(settings: Settings, user_id: str) -> list[dict[str, Any]] | None:
    r = _get(settings, f"/users/{user_id}/orders")
    return r.json() if r.status_code == 200 else None


def list_users(settings: Settings, q: str, limit: int, offset: int = 0) -> tuple[int, list[dict[str, Any]]]:
    r = _get(settings, "/users", {"q": q or None, "limit": limit, "offset": offset})
    if r.status_code in (404, 405):
        raise ListNotSupported("This user service has no `GET /users` endpoint.")
    if r.status_code != 200:
        raise ServiceError(f"User service returned HTTP {r.status_code} for `/users`.")
    body = r.json()
    return int(body.get("total", 0)), list(body.get("items", []))


# --------------------------------------------------------------------------- rendering


def _select(ident: str) -> None:
    st.session_state["users_selected"] = ident.strip()


def _lookup_submitted() -> None:
    q = (st.session_state.get("users_lookup") or "").strip()
    if q:
        _select(q)


def _reset_shown() -> None:
    st.session_state["users_shown"] = st.session_state.get("users_page_size", PAGE_SIZES[0])


def _load_more() -> None:
    size = st.session_state.get("users_page_size", PAGE_SIZES[0])
    st.session_state["users_shown"] = st.session_state.get("users_shown", size) + size


def _profile(db: str | None, settings: Settings, ident: str) -> None:
    try:
        user = fetch_user(settings, ident)
    except ServiceError as e:
        st.error(str(e), icon=":material/cloud_off:")
        user = None
        if "@" not in ident:
            return
    else:
        if user is None:
            st.warning(
                f"No user record for `{ident}`. The agent treats this customer as **UNKNOWN_USER** "
                "and sends their tickets to human review.",
                icon=":material/person_off:",
            )
    email = user["email"] if user else (ident if "@" in ident else None)
    if user:
        st.subheader(f":material/person: {user['name']}")
        st.caption(f"`{user['user_id']}` · `{user['email']}` · customer since {user['signup_date']}")
        status_icon = {"active": ":material/check_circle:", "suspended": ":material/pause_circle:"}.get(
            user["account_status"], ":material/cancel:"
        )
        c = st.columns(5)
        c[0].metric("Plan", user["plan"].title(), border=True)
        c[1].metric("Account", f"{user['account_status'].title()}", border=True)
        c[2].metric("VIP", "Yes" if user.get("vip") else "No", border=True)
        c[3].metric("Lifetime value", f"${user['lifetime_value']:,.0f}", border=True)
        c[4].metric("Recent refunds", user.get("recent_refunds", 0), border=True)
        flags = []
        if user.get("vip") or user["plan"] == "enterprise":
            flags.append("VIP")
        if user["account_status"] != "active":
            flags.append("ACCOUNT_NOT_ACTIVE")
        st.caption(
            f"{status_icon} Policy rules this customer triggers: "
            + (", ".join(f"`{f}`" for f in flags) if flags else "none")
        )
        st.markdown("**Orders**")
        try:
            orders = fetch_orders(settings, user["user_id"])
        except ServiceError:
            orders = None
        if orders is None:
            orders = user.get("open_orders") or []
            st.caption("Showing open orders from the profile (order history unavailable).")
        if orders:
            st.dataframe(
                pd.DataFrame(orders),
                hide_index=True,
                column_config={"amount": st.column_config.NumberColumn("Amount", format="%.2f")},
            )
        else:
            ui.empty("No orders.")
    if email and db:
        st.markdown(f"**Tickets from `{email}`**")
        t = data.tickets_for_email(db, email)
        if t.empty:
            ui.empty("No tickets from this email in the database.")
        else:
            st.dataframe(
                t,
                hide_index=True,
                column_config={
                    "updated_at": st.column_config.DatetimeColumn("Updated", format="YYYY-MM-DD HH:mm")
                },
            )
            st.caption("Open a ticket in the **Tickets** tab for its full timeline and drafts.")


def _browse(db: str | None, settings: Settings) -> None:
    f = st.columns([3, 1], vertical_alignment="bottom")
    q = (
        f[0]
        .text_input(
            "Search users", key="users_q", placeholder="Email, name or user id", on_change=_reset_shown
        )
        .strip()
    )
    size = f[1].selectbox("Rows", PAGE_SIZES, key="users_page_size", on_change=_reset_shown)
    shown = max(int(st.session_state.get("users_shown", size)), size)
    source = "service"
    try:
        total, items = list_users(settings, q, min(shown, 1000))
        df = pd.DataFrame(items)
        if not df.empty:
            df = df[
                [
                    "user_id",
                    "email",
                    "name",
                    "plan",
                    "account_status",
                    "vip",
                    "lifetime_value",
                    "recent_refunds",
                ]
            ]
    except ServiceError as e:
        if db is None:
            st.error(str(e), icon=":material/cloud_off:")
            return
        source = "db"
        total, df = data.customer_emails(db, q, shown)
        why = (
            "the running user service has no `GET /users` endpoint"
            if isinstance(e, ListNotSupported)
            else "the user service is not reachable"
        )
        st.caption(f":material/info: Showing customer emails from tickets ({why}).")
    if df.empty:
        ui.empty("No users match." if q else "No users.")
        return
    more = total > len(df)
    cap = st.columns([4, 1], vertical_alignment="center")
    cap[0].caption(
        f"Showing {len(df)} of {total}" + (" · more available" if more else "") + " · select a row to open it"
    )
    if more and (source == "db" or shown < 1000):  # the service caps limit at 1000
        cap[1].button(
            f"Load {size} more", on_click=_load_more, key="users_more", icon=":material/expand_more:"
        )
    event = st.dataframe(
        df,
        hide_index=True,
        height=300,
        key=f"users_table_{source}",
        on_select="rerun",
        selection_mode="single-row",
        column_config={"lifetime_value": st.column_config.NumberColumn("LTV", format="$%.0f")},
    )
    rows = tuple(getattr(getattr(event, "selection", None), "rows", None) or [])
    if rows != st.session_state.get("users_last_sel", ()):
        st.session_state["users_last_sel"] = rows
        if rows and rows[0] < len(df):
            _select(str(df.iloc[rows[0]]["email"]))


def _body(db: str | None, settings: Settings) -> None:
    with st.form("users_lookup_form", border=False):
        f = st.columns([4, 1], vertical_alignment="bottom")
        f[0].text_input(
            "Look up a user", key="users_lookup", placeholder="Email address or user id (e.g. U-1001)"
        )
        f[1].form_submit_button(
            "Look up", icon=":material/search:", on_click=_lookup_submitted, width="stretch"
        )
    st.caption(f":material/link: `{settings.user_service_url}`")

    left, right = st.columns([2, 3], gap="large")
    with left:
        st.markdown("**Browse**")
        _browse(db, settings)
    with right:
        selected = st.session_state.get("users_selected")
        if selected:
            _profile(db, settings, selected)
        else:
            ui.empty("Look up a user by email or id, or select a row on the left.")


def render(db: str | None, settings: Settings) -> None:
    """User Info tab. Not auto-refreshed (it calls a live service)."""
    st.fragment(_body)(db, settings)
