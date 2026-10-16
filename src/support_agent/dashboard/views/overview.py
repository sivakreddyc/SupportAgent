"""Overview: KPI tiles, throughput over time, e2e latency."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from support_agent.dashboard import data, ui

st.title("Overview")
db = ui.require_db()
since = ui.time_range("overview_range")


@st.fragment(run_every=ui.refresh_every())
def body() -> None:
    k = data.kpis(db)
    c = st.columns(6)
    c[0].metric("Discovered", ui.fmt_num(k["discovered"]), border=True)
    c[1].metric(
        "Processed", ui.fmt_num(k["processed"]), help="Tickets that reached a policy decision", border=True
    )
    c[2].metric(
        "Auto-approved",
        ui.fmt_num(k["auto_approved_pct"], "%", 1),
        help=f"{k['auto_approved']} of {k['processed']} policy decisions",
        border=True,
    )
    c[3].metric("In review", ui.fmt_num(k["in_review"]), border=True)
    c[4].metric("Failed (retrying)", ui.fmt_num(k["failed"]), border=True)
    c[5].metric("Dead letter", ui.fmt_num(k["dead_letter"]), border=True)

    if k["discovered"] == 0:
        ui.empty("No tickets discovered yet. The poller writes tickets here as soon as it finds them.")
        return

    fin = data.finished_tickets(db, since)
    disc = data.discovered_times(db, since)
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Throughput")
        if fin.empty and disc.empty:
            ui.empty("No activity in this time range.")
        else:
            bucket = ui.bucket_for(pd.concat([fin["finished_at"], disc["discovered_at"]]))
            done = (
                fin.groupby([pd.Grouper(key="finished_at", freq=bucket), "result"])
                .size()
                .reset_index(name="tickets")
                .rename(columns={"finished_at": "time"})
            )
            domain = [
                r
                for r in ("Auto-approved", "Human reviewed", "Dead letter", "Other")
                if r in set(done["result"])
            ]
            if not done.empty:
                ui.chart(
                    ui.bars(
                        done,
                        "time",
                        "tickets",
                        "result",
                        y_title=f"Finished / {bucket}",
                        domain=domain,
                        bucket=bucket,
                    )
                )
            found = (
                disc.groupby(pd.Grouper(key="discovered_at", freq=bucket)).size().reset_index(name="tickets")
            ).rename(columns={"discovered_at": "time"})
            st.caption(f"Discovered per {bucket}")
            ui.chart(ui.line(found, "time", "tickets", y_title="Discovered", height=140, integer=True))
    with right:
        st.subheader("End-to-end latency")
        if fin.empty:
            ui.empty("No finished tickets in this time range yet.")
        else:
            s = fin["e2e_seconds"].dropna()
            m = st.columns(3)
            m[0].metric("p50", ui.fmt_num(s.quantile(0.5), " s", 2))
            m[1].metric("p95", ui.fmt_num(s.quantile(0.95), " s", 2))
            m[2].metric("Finished", ui.fmt_num(len(s)))
            ui.chart(ui.histogram(fin, "e2e_seconds", title="Discovered → finished (s)"))
            st.caption("Includes time spent waiting for human review.")

    st.subheader("Tickets by state")
    states = pd.DataFrame(
        [{"state": s, "tickets": n} for s, n in k["states"].items() if n], columns=["state", "tickets"]
    )
    ui.chart(
        ui.bars(
            states,
            "state",
            "tickets",
            x_type="N",
            horizontal=True,
            y_title="Tickets",
            height=40 + 34 * len(states),
        )
    )


body()
