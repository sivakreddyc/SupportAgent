"""Queue & Workers: depth vs capacity, rejections, queue wait, busy vs configured workers, saturation hint."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from support_agent.dashboard import data, ui

GAUGES = ["queue_depth", "queue_capacity", "workers_busy", "workers_configured"]
COUNTERS = [
    "queue_rejections_total",
    "queue_enqueued_total",
    "queue_wait_seconds_sum",
    "queue_wait_seconds_count",
]
SATURATION = 0.9
SATURATION_WINDOW = pd.Timedelta(minutes=2)


def _sustained_above(ts: pd.Series, ratio: pd.Series) -> bool:
    """True when every sample in the last SATURATION_WINDOW (at least 3) is above the threshold."""
    recent = ratio[ts >= ts.max() - SATURATION_WINDOW]
    return len(recent) >= 3 and bool((recent > SATURATION).all())


def _body(db: str, since: datetime | None) -> None:
    m = data.metric_samples(db, GAUGES + COUNTERS, since)
    if m.empty:
        ui.empty(
            "No metric samples in this time range. The runtime snapshots its metrics into `metric_samples` "
            "every few seconds while it is running."
        )
        return
    g = m[m["name"].isin(GAUGES)].pivot_table(index="ts", columns="name", values="value", aggfunc="last")
    g = g.reindex(columns=GAUGES).sort_index()
    last = g.ffill().iloc[-1]
    depth, cap = last.get("queue_depth"), last.get("queue_capacity")
    busy, conf = last.get("workers_busy"), last.get("workers_configured")

    def total(name: str) -> float:
        s = m[m["name"] == name]
        return float(ui.counter_rate(s)["value"].sum()) if not s.empty else 0.0

    rejections = total("queue_rejections_total")
    enqueued = total("queue_enqueued_total")
    wait_n, wait_sum = total("queue_wait_seconds_count"), total("queue_wait_seconds_sum")

    c = st.columns(5)
    qutil = (depth / cap) if cap else None
    wutil = (busy / conf) if conf else None
    c[0].metric(
        "Queue depth",
        f"{ui.fmt_num(depth)} / {ui.fmt_num(cap)}",
        ui.fmt_num(qutil and qutil * 100, "% full", 0),
        delta_color="off",
        border=True,
    )
    c[1].metric(
        "Workers busy",
        f"{ui.fmt_num(busy)} / {ui.fmt_num(conf)}",
        ui.fmt_num(wutil and wutil * 100, "% busy", 0),
        delta_color="off",
        border=True,
    )
    c[2].metric("Enqueued", ui.fmt_num(enqueued), help="In the selected time range", border=True)
    c[3].metric(
        "Rejected (queue full)",
        ui.fmt_num(rejections),
        help="Backpressure: ticket stays DISCOVERED and is retried next poll",
        border=True,
    )
    c[4].metric("Avg queue wait", ui.fmt_num(wait_sum / wait_n if wait_n else None, " s", 2), border=True)

    gf = g.ffill().reset_index()
    if (
        conf
        and "workers_busy" in gf
        and _sustained_above(gf["ts"], gf["workers_busy"] / gf["workers_configured"])
    ):
        st.warning(
            "**Workers saturated**: more than 90% of workers have been busy for the last 2 minutes. "
            "Consider raising `WORKER_COUNT`, or lowering LLM latency (model / effort).",
            icon=":material/warning:",
        )
    if cap and "queue_depth" in gf and _sustained_above(gf["ts"], gf["queue_depth"] / gf["queue_capacity"]):
        st.warning(
            "**Queue near capacity**: the queue has been over 90% full for the last 2 minutes; new tickets "
            "are being held back (backpressure).",
            icon=":material/warning:",
        )

    bucket = ui.bucket_for(m["ts"])
    long = g.reset_index().melt(id_vars="ts", var_name="series", value_name="value").dropna()
    long = long.groupby([pd.Grouper(key="ts", freq=bucket), "series"], as_index=False)["value"].mean()
    left, right = st.columns(2)
    with left:
        st.subheader("Queue depth vs capacity")
        q = long[long["series"].isin(["queue_depth", "queue_capacity"])].replace(
            {"queue_depth": "Depth", "queue_capacity": "Capacity"}
        )
        ui.chart(ui.line(q, "ts", "value", "series", y_title="Items", domain=["Depth", "Capacity"]))
    with right:
        st.subheader("Workers busy vs configured")
        w = long[long["series"].isin(["workers_busy", "workers_configured"])].replace(
            {"workers_busy": "Busy", "workers_configured": "Configured"}
        )
        ui.chart(ui.line(w, "ts", "value", "series", y_title="Workers", domain=["Busy", "Configured"]))

    left, right = st.columns(2)
    with left:
        st.subheader("Enqueued and rejected")
        rates = ui.counter_rate(
            m[m["name"].isin(["queue_enqueued_total", "queue_rejections_total"])].assign(
                label=lambda d: d["name"]
            ),
            bucket,
        )
        rates["label"] = rates["label"].replace(
            {"queue_enqueued_total": "Enqueued", "queue_rejections_total": "Rejected"}
        )
        if rates.empty:
            ui.empty("No enqueue activity.")
        else:
            ui.chart(
                ui.bars(
                    rates,
                    "ts",
                    "value",
                    "label",
                    y_title=f"Items / {bucket}",
                    domain=["Enqueued", "Rejected"],
                    bucket=bucket,
                )
            )
    with right:
        st.subheader("Average queue wait")
        ws = ui.counter_rate(m[m["name"] == "queue_wait_seconds_sum"].assign(label="sum"), bucket)
        wc = ui.counter_rate(m[m["name"] == "queue_wait_seconds_count"].assign(label="count"), bucket)
        if ws.empty or wc.empty:
            ui.empty("No dequeues yet.")
        else:
            j = ws.merge(wc, on="ts", suffixes=("_sum", "_count"))
            j = j[j["value_count"] > 0].assign(wait=lambda d: d["value_sum"] / d["value_count"])
            ui.chart(ui.line(j, "ts", "wait", y_title="Seconds (enqueue → dequeue)"))


def render(db: str, since: datetime | None) -> None:
    """Queue & Workers section (auto-refreshing fragment). Used by the Observability tab."""
    st.fragment(_body, run_every=ui.refresh_every())(db, since)
