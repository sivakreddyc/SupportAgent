"""Shared Streamlit helpers: DB guard, refresh, time range, consistent altair charts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import altair as alt
import pandas as pd
import streamlit as st

from . import data

REFRESH_SECONDS = 5

# Categorical slots in fixed order (validated reference palette); status colours are reserved.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
NEUTRAL = "#8a8984"

# Fixed colour per entity so a filter never repaints survivors.
RESULT_COLORS = {
    "Auto-approved": STATUS["good"],
    "Human reviewed": SERIES[0],
    "Dead letter": STATUS["critical"],
    "Other": NEUTRAL,
    "ok": STATUS["good"],
    "refusal": STATUS["serious"],
    "parse_error": STATUS["warning"],
    "error": STATUS["critical"],
}

TIME_RANGES: dict[str, timedelta | None] = {
    "Last 15 min": timedelta(minutes=15),
    "Last hour": timedelta(hours=1),
    "Last 6 hours": timedelta(hours=6),
    "Last 24 hours": timedelta(hours=24),
    "All time": None,
}


def db_ready(path: str) -> bool:
    """Render a friendly empty state (without stopping the script) unless the DB is usable."""
    status = data.db_status(path)
    if status == "missing":
        st.info(
            f"No database yet at `{path}`.\n\n"
            "Start the agent runtime (`support-agent run`) and it will create it, "
            "or point the console at another file with `SA_DB_PATH`.",
            icon=":material/database:",
        )
        return False
    if status == "uninitialized":
        st.info(
            f"`{path}` exists but has no SupportAgent tables yet. Start the runtime once to initialise it.",
            icon=":material/database:",
        )
        return False
    return True


def clear_cache() -> None:
    """Call after any write so cached reads don't show stale state."""
    st.cache_data.clear()


def require_db() -> str:
    """Return the DB path, or render a friendly empty state and stop the page."""
    path = data.resolve_db_path()
    status = data.db_status(path)
    if status == "missing":
        st.info(
            f"No database yet at `{path}`.\n\n"
            "Start the agent runtime (`support-agent run`) and it will create it, "
            "or point the dashboard at another file with `SA_DB_PATH`.",
            icon=":material/database:",
        )
        st.stop()
    if status == "uninitialized":
        st.info(
            f"`{path}` exists but has no SupportAgent tables yet. Start the runtime once to initialise it.",
            icon=":material/database:",
        )
        st.stop()
    return path


def refresh_every() -> int | None:
    return REFRESH_SECONDS if st.session_state.get("auto_refresh", True) else None


def time_range(key: str, default: str = "Last hour") -> datetime | None:
    label = st.segmented_control(
        "Time range",
        list(TIME_RANGES),
        default=default,
        key=key,
        label_visibility="collapsed",
    )
    delta = TIME_RANGES.get(label or default)
    return datetime.now(UTC) - delta if delta else None


def bucket_for(ts: pd.Series) -> str:
    if ts.empty:
        return "1min"
    span = ts.max() - ts.min()
    if span <= pd.Timedelta(hours=2):
        return "1min"
    if span <= pd.Timedelta(hours=24):
        return "10min"
    return "1h"


def empty(msg: str) -> None:
    st.caption(f":material/info: {msg}")


def fmt_num(v: float | int | None, suffix: str = "", digits: int = 0) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{v:,.{digits}f}{suffix}"


def fmt_ts(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    ts = pd.to_datetime(v, utc=True, format="ISO8601", errors="coerce")
    return "—" if pd.isna(ts) else ts.strftime("%Y-%m-%d %H:%M:%S UTC")


def time_x(field: str, title: str | None = "Time (UTC)") -> alt.X:
    return alt.X(
        f"{field}:T",
        title=title,
        scale=alt.Scale(type="utc"),
        axis=alt.Axis(formatType="utc", format="%H:%M", labelOverlap=True),
    )


def time_tip(field: str, title: str = "Time (UTC)") -> alt.Tooltip:
    return alt.Tooltip(f"{field}:T", title=title, formatType="utc", format="%Y-%m-%d %H:%M:%S")


def _time_unit(bucket: str) -> alt.TimeUnitParams:
    """Vega-Lite UTC time unit matching a pandas bucket ("1min", "10min", "1h") so bars span the bucket."""
    td = pd.Timedelta(bucket)
    if td < pd.Timedelta(hours=1):
        return alt.TimeUnitParams(
            unit="utcyearmonthdatehoursminutes", step=max(1, int(td.total_seconds() // 60))
        )
    if td < pd.Timedelta(days=1):
        return alt.TimeUnitParams(unit="utcyearmonthdatehours", step=max(1, int(td.total_seconds() // 3600)))
    return alt.TimeUnitParams(unit="utcyearmonthdate")


def _y(field: str, title: str | None, integer: bool, **kw) -> alt.Y:
    axis = alt.Axis(tickMinStep=1, format="d") if integer else alt.Axis()
    return alt.Y(f"{field}:Q", title=title, axis=axis, **kw)


def _color(field: str, domain: list[str] | None = None, title: str | None = None) -> alt.Color:
    if domain is None:
        return alt.Color(f"{field}:N", title=title, scale=alt.Scale(range=SERIES))
    rng = [RESULT_COLORS.get(d, SERIES[i % len(SERIES)]) for i, d in enumerate(domain)]
    return alt.Color(f"{field}:N", title=title, scale=alt.Scale(domain=domain, range=rng))


def line(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    *,
    y_title: str | None = None,
    domain: list[str] | None = None,
    height: int = 240,
    rule_y: float | None = None,
    integer: bool = False,
) -> alt.Chart:
    base = alt.Chart(df)
    enc = {
        "x": time_x(x),
        "y": _y(y, y_title, integer),
        "tooltip": [time_tip(x)]
        + ([alt.Tooltip(f"{color}:N")] if color else [])
        + [alt.Tooltip(f"{y}:Q", format=",.2f")],
    }
    if color:
        enc["color"] = _color(color, domain)
    chart = base.mark_line(strokeWidth=2, interpolate="monotone", point=False).encode(**enc)
    hover = base.mark_point(size=80, opacity=0).encode(**enc)  # bigger hit target for the tooltip
    layers = [chart, hover]
    if rule_y is not None:
        layers.append(
            alt.Chart(pd.DataFrame({"y": [rule_y]}))
            .mark_rule(strokeDash=[4, 4], color=NEUTRAL)
            .encode(y="y:Q")
        )
    return alt.layer(*layers).properties(height=height)


def bars(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    *,
    x_type: str = "T",
    y_title: str | None = None,
    domain: list[str] | None = None,
    height: int = 240,
    horizontal: bool = False,
    bucket: str | None = None,
    integer: bool = True,
) -> alt.Chart:
    """Bar chart. Temporal bars need `bucket` (e.g. "1min") so each bar spans its whole bucket."""
    tip_x = time_tip(x) if x_type == "T" else alt.Tooltip(f"{x}:{x_type}")
    tooltip = (
        [tip_x] + ([alt.Tooltip(f"{color}:N")] if color else []) + [alt.Tooltip(f"{y}:Q", format=",.2f")]
    )
    if horizontal:
        mark = alt.Chart(df).mark_bar(cornerRadiusEnd=4, stroke="white", strokeWidth=1)
        enc = {"y": alt.Y(f"{x}:{x_type}", title=None, sort="-x"), "x": alt.X(f"{y}:Q", title=y_title)}
    elif x_type == "T":
        mark = alt.Chart(df).mark_bar(
            cornerRadiusTopLeft=2, cornerRadiusTopRight=2, stroke="white", strokeWidth=0.5
        )
        enc = {
            "x": alt.X(
                f"{x}:T",
                timeUnit=_time_unit(bucket or "1min"),
                title="Time (UTC)",
                axis=alt.Axis(formatType="utc", format="%H:%M", labelOverlap=True),
            ),
            "y": _y(y, y_title, integer, stack="zero" if color else None),
        }
    else:
        mark = alt.Chart(df).mark_bar(
            cornerRadiusTopLeft=3, cornerRadiusTopRight=3, stroke="white", strokeWidth=1
        )
        enc = {"x": alt.X(f"{x}:{x_type}", title=None), "y": _y(y, y_title, integer)}
    enc["tooltip"] = tooltip
    enc["color"] = _color(color, domain) if color else alt.value(SERIES[0])
    return mark.encode(**enc).properties(height=height)


def histogram(df: pd.DataFrame, field: str, *, title: str, bins: int = 30, height: int = 220) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_bar(
            color=SERIES[0], cornerRadiusTopLeft=3, cornerRadiusTopRight=3, stroke="white", strokeWidth=1
        )
        .encode(
            x=alt.X(f"{field}:Q", bin=alt.Bin(maxbins=bins), title=title),
            y=alt.Y("count():Q", title="Count"),
            tooltip=[alt.Tooltip(f"{field}:Q", bin=alt.Bin(maxbins=bins), title=title), "count():Q"],
        )
        .properties(height=height)
    )


def chart(c: alt.Chart) -> None:
    st.altair_chart(c, width="stretch")


def counter_rate(df: pd.DataFrame, bucket: str | None = None) -> pd.DataFrame:
    """Turn cumulative counter samples (ts, label, value) into per-bucket increments (reset-safe)."""
    if df.empty:
        return pd.DataFrame(columns=["ts", "label", "value"])
    out = []
    for label, g in df.sort_values("ts").groupby("label", sort=False):
        d = g["value"].diff()
        d = d.where(d >= 0, g["value"])  # counter reset (process restart)
        out.append(pd.DataFrame({"ts": g["ts"], "label": label, "value": d.fillna(0)}))
    res = pd.concat(out, ignore_index=True)
    if bucket:
        res = res.groupby([pd.Grouper(key="ts", freq=bucket), "label"], as_index=False)["value"].sum()
    return res


def gauge_series(df: pd.DataFrame, bucket: str | None = None) -> pd.DataFrame:
    if df.empty or not bucket:
        return df
    return df.groupby([pd.Grouper(key="ts", freq=bucket), "name", "label"], as_index=False)["value"].mean()
