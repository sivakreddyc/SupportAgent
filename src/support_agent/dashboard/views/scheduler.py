"""Scheduler & Poller: runs, duration, found/enqueued/skipped per run, browser op success rate, last error."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from support_agent.common.config import PROJECT_ROOT, get_settings
from support_agent.dashboard import data, ui

st.title("Scheduler & Poller")
db = ui.require_db()
since = ui.time_range("sched_range")


def _artifacts(limit: int = 5) -> list[Path]:
    d = Path(get_settings().artifacts_dir).expanduser()
    if not d.is_absolute() and not d.exists():
        d = PROJECT_ROOT / d
    if not d.is_dir():
        return []
    files = [p for p in d.rglob("*") if p.suffix.lower() in (".png", ".html")]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]


@st.fragment(run_every=ui.refresh_every())
def body() -> None:
    runs = data.scheduler_runs(db, since)
    ops = data.metric_samples(db, ["browser_ops_total", "scheduler_skipped_total"], since)

    c = st.columns(5)
    c[0].metric("Poll runs", ui.fmt_num(len(runs)), border=True)
    c[1].metric(
        "Last run", runs["started_at"].max().strftime("%H:%M:%S") if not runs.empty else "—", border=True
    )
    c[2].metric(
        "Avg duration",
        ui.fmt_num(runs["duration_s"].mean() if not runs.empty else None, " s", 2),
        border=True,
    )
    c[3].metric(
        "Runs with errors", ui.fmt_num(int(runs["error"].notna().sum()) if not runs.empty else 0), border=True
    )
    skipped = ops[ops["name"] == "scheduler_skipped_total"]
    c[4].metric(
        "Ticks skipped",
        ui.fmt_num(ui.counter_rate(skipped)["value"].sum() if not skipped.empty else 0),
        help="Skipped because the previous poll was still running (overlap guard)",
        border=True,
    )

    err = data.last_scheduler_error(db)
    if err:
        st.error(
            f"Last poll error (run {err['id']}, {ui.fmt_ts(err['started_at'])}): {err['error']}",
            icon=":material/error:",
        )
    arts = _artifacts()
    if arts:
        with st.expander(f"Browser failure artifacts ({len(arts)} most recent)"):
            for p in arts:
                st.caption(f"`{p}`")
            pngs = [p for p in arts if p.suffix.lower() == ".png"]
            if pngs:
                st.image(str(pngs[0]), caption=pngs[0].name)

    if runs.empty:
        ui.empty("No scheduler runs in this time range.")
    else:
        left, right = st.columns(2)
        with left:
            st.subheader("Poll duration")
            ui.chart(
                ui.line(runs.dropna(subset=["duration_s"]), "started_at", "duration_s", y_title="Seconds")
            )
        with right:
            st.subheader("Tickets per run")
            per = runs.melt(
                id_vars="started_at",
                value_vars=["found", "enqueued", "skipped_full"],
                var_name="what",
                value_name="tickets",
            ).dropna()
            per["what"] = per["what"].map(
                {"found": "Found", "enqueued": "Enqueued", "skipped_full": "Skipped (queue full)"}
            )
            ui.chart(
                ui.line(
                    per,
                    "started_at",
                    "tickets",
                    "what",
                    y_title="Tickets",
                    integer=True,
                    domain=["Found", "Enqueued", "Skipped (queue full)"],
                )
            )

    st.subheader("Browser operations")
    b = ops[ops["name"] == "browser_ops_total"]
    if b.empty:
        ui.empty("No browser operation samples in this time range.")
    else:
        b = b.assign(
            op=[lab.get("op", "?") for lab in b["labels"]],
            result=[lab.get("result", "?") for lab in b["labels"]],
        )
        bucket = ui.bucket_for(b["ts"])
        inc = ui.counter_rate(b, bucket)
        meta = b.drop_duplicates("label").set_index("label")[["op", "result"]]
        inc = inc.join(meta, on="label")
        per_bucket = inc.groupby(["ts", "result"])["value"].sum().unstack(fill_value=0)
        if "ok" not in per_bucket:
            per_bucket["ok"] = 0
        total = per_bucket.sum(axis=1)
        rate = (100 * per_bucket["ok"] / total.where(total > 0)).dropna().reset_index(name="success_pct")
        left, right = st.columns([3, 2])
        with left:
            st.caption(f"Success rate per {bucket}")
            if rate.empty:
                ui.empty("No browser operations in this window.")
            else:
                ui.chart(ui.line(rate, "ts", "success_pct", y_title="Success (%)", rule_y=100))
        with right:
            tbl = inc.groupby(["op", "result"])["value"].sum().unstack(fill_value=0)
            tbl["total"] = tbl.sum(axis=1)
            tbl["success_%"] = (100 * tbl.get("ok", 0) / tbl["total"].where(tbl["total"] > 0)).round(1)
            st.dataframe(tbl.sort_values("total", ascending=False))

    if not runs.empty:
        st.subheader("Recent runs")
        st.dataframe(
            runs.sort_values("id", ascending=False).head(50),
            hide_index=True,
            column_config={
                "started_at": st.column_config.DatetimeColumn("Started", format="YYYY-MM-DD HH:mm:ss"),
                "finished_at": st.column_config.DatetimeColumn("Finished", format="HH:mm:ss"),
                "duration_s": st.column_config.NumberColumn("Duration (s)", format="%.2f"),
            },
        )


body()
