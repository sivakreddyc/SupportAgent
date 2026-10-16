"""LLM: calls by result, refusals, tokens in/out over time, latency distribution, tokens per ticket."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from support_agent.dashboard import data, ui

RESULTS = ["ok", "refusal", "parse_error", "error"]


def _body(db: str, since: datetime | None) -> None:
    df = data.llm_calls(db, since)
    if df.empty:
        ui.empty("No LLM calls in this time range.")
        return
    n = len(df)
    by = df["result"].value_counts()
    lat = df["latency_ms"].dropna()
    c = st.columns(6)
    c[0].metric("Calls", ui.fmt_num(n), border=True)
    c[1].metric("Success", ui.fmt_num(100 * by.get("ok", 0) / n, "%", 1), border=True)
    c[2].metric("Refusals", ui.fmt_num(by.get("refusal", 0)), border=True)
    c[3].metric(
        "Parse errors",
        ui.fmt_num(by.get("parse_error", 0)),
        help="Truncated or schema-invalid output",
        border=True,
    )
    c[4].metric("API errors", ui.fmt_num(by.get("error", 0)), border=True)
    c[5].metric(
        "Latency p50",
        ui.fmt_num(lat.quantile(0.5) / 1000, " s", 2) if not lat.empty else "—",
        f"p95 {ui.fmt_num(lat.quantile(0.95) / 1000, ' s', 2)}" if not lat.empty else None,
        delta_color="off",
        delta_arrow="off",
        border=True,
    )
    tin, tout, tcache = df["input_tokens"].sum(), df["output_tokens"].sum(), df["cache_read_tokens"].sum()
    st.caption(
        f"Tokens: {tin:,} input · {tout:,} output · {tcache:,} cache-read "
        f"({100 * tcache / (tin + tcache):.0f}% of prompt tokens served from cache)"
        if (tin + tcache)
        else ""
    )

    bucket = ui.bucket_for(df["at"])
    left, right = st.columns(2)
    with left:
        st.subheader("Calls by result")
        calls = df.groupby([pd.Grouper(key="at", freq=bucket), "result"]).size().reset_index(name="calls")
        dom = [r for r in RESULTS if r in set(calls["result"])]
        ui.chart(
            ui.bars(calls, "at", "calls", "result", y_title=f"Calls / {bucket}", domain=dom, bucket=bucket)
        )
    with right:
        st.subheader("Tokens over time")
        tok = (
            df.groupby(pd.Grouper(key="at", freq=bucket))[
                ["input_tokens", "output_tokens", "cache_read_tokens"]
            ]
            .sum()
            .reset_index()
            .melt(id_vars="at", var_name="direction", value_name="tokens")
        )
        tok["direction"] = tok["direction"].map(
            {"input_tokens": "Input", "output_tokens": "Output", "cache_read_tokens": "Cache read"}
        )
        ui.chart(
            ui.line(
                tok,
                "at",
                "tokens",
                "direction",
                y_title=f"Tokens / {bucket}",
                domain=["Input", "Output", "Cache read"],
            )
        )

    left, right = st.columns(2)
    with left:
        st.subheader("Latency distribution")
        if lat.empty:
            ui.empty("No latency recorded.")
        else:
            ui.chart(
                ui.histogram(df.assign(latency_s=df["latency_ms"] / 1000), "latency_s", title="Latency (s)")
            )
    with right:
        st.subheader("Tokens per ticket")
        per = df.groupby("ticket_id")[["input_tokens", "output_tokens"]].sum()
        per["total"] = per.sum(axis=1)
        ui.chart(ui.histogram(per.reset_index(), "total", title="Tokens per ticket (in + out)"))
        st.caption(
            f"Mean {per['total'].mean():,.0f} · max {per['total'].max():,.0f} "
            f"({per['total'].idxmax()}) · {len(per)} tickets"
        )

    st.subheader("By model")
    models = df.groupby("model", dropna=False).agg(
        calls=("id", "count"),
        ok=("result", lambda s: int((s == "ok").sum())),
        refusals=("result", lambda s: int((s == "refusal").sum())),
        p50_ms=("latency_ms", "median"),
        input_tokens=("input_tokens", "sum"),
        output_tokens=("output_tokens", "sum"),
    )
    st.dataframe(models, column_config={"p50_ms": st.column_config.NumberColumn("p50 (ms)", format="%.0f")})
    bad = df[df["result"] != "ok"].sort_values("at", ascending=False).head(20)
    if not bad.empty:
        st.subheader("Recent non-ok calls")
        st.dataframe(bad[["at", "ticket_id", "model", "result", "stop_reason", "error"]], hide_index=True)


def render(db: str, since: datetime | None) -> None:
    """LLM section (auto-refreshing fragment). Used by the Observability tab."""
    st.fragment(_body, run_every=ui.refresh_every())(db, since)
