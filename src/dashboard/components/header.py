import streamlit as st
from datetime import datetime
import requests

from utils.html_utils import render_html
from utils.loader import get_batch_info


def check_api_status(api_base):
    try:
        r = requests.get(f"{api_base}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _pipeline_status(seconds_since_update, idle_threshold=120):
    if seconds_since_update is None:
        return "offline", "No data loaded yet"
    if seconds_since_update <= idle_threshold:
        return "live", "Live"
    return "processing", "Idle"


def render_header(df, api_base):
    ai_online = check_api_status(api_base)
    batch = get_batch_info()

    total_products = len(df)
    departments = df["department"].nunique()
    methods = df["optimization_method"].nunique()

    seconds_since_update = None
    last_updated_str = "—"
    batch_id = batch.get("batch_id") or "—"
    processing_time = batch.get("processing_time")

    if batch.get("last_updated") is not None:
        seconds_since_update = (
            datetime.now() - batch["last_updated"].to_pydatetime()
        ).total_seconds()
        last_updated_str = batch["last_updated"].strftime("%H:%M:%S")

    pipe_state, pipe_label = _pipeline_status(seconds_since_update)
    pipe_dot = {"live": "&#128994;", "processing": "&#128993;", "offline": "&#128308;"}[pipe_state]

    ai_class = "live" if ai_online else "offline"
    ai_label = "AI Assistant Online" if ai_online else "AI Assistant Offline"
    ai_dot = "&#128994;" if ai_online else "&#128308;"

    processing_str = f"{processing_time:.1f}s" if processing_time is not None else "—"

    render_html(
        f'<div class="header-card">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">'

        f'<div>'
        f'<div style="display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;">'
        f'<h1 style="margin:0;color:var(--text-display);font-size:26px;font-weight:700;">Pricing Intelligence Dashboard</h1>'
        f'<span style="color:var(--text-muted);font-size:14px;">Enterprise Pricing Optimization</span>'
        f'</div>'
        f'<div style="margin-top:6px;color:var(--text-secondary);font-size:13px;">'
        f'Catalog: <b style="color:var(--text-primary);">{total_products:,}</b> products &nbsp;·&nbsp; '
        f'<b style="color:var(--text-primary);">{departments}</b> departments &nbsp;·&nbsp; '
        f'<b style="color:var(--text-primary);">{methods}</b> optimization methods'
        f'</div>'
        f'</div>'

        f'<div style="text-align:right;">'
        f'<div style="display:flex;gap:8px;justify-content:flex-end;flex-wrap:wrap;">'
        f'<span class="status-pill {pipe_state}">{pipe_dot} {pipe_label}</span>'
        f'<span class="status-pill {ai_class}">{ai_dot} {ai_label}</span>'
        f'</div>'
        f'<div style="margin-top:6px;color:var(--text-muted);font-size:12px;">'
        f'Last updated {last_updated_str} &nbsp;·&nbsp; Batch {batch_id} &nbsp;·&nbsp; {processing_str} to process'
        f'</div>'
        f'</div>'

        f'</div>'
        f'</div>'
    )

    if seconds_since_update is not None and seconds_since_update > 120:
        minutes = int(seconds_since_update // 60)
        render_html(
            f'<div class="warning-card" style="margin-bottom:16px;">'
            f'No new prediction batch received in the last {minutes} minute'
            f'{"s" if minutes != 1 else ""}. Showing the most recent available results.'
            f'</div>'
        )


def render_controls():
    col1, col2, col3, col4 = st.columns([1, 1, 1, 3])

    with col1:
        refresh = st.button("Refresh", use_container_width=True)
    with col2:
        export = st.button("Export", use_container_width=True)
    with col3:
        auto_refresh_on = st.toggle("Auto-refresh", value=False)
    with col4:
        interval = st.selectbox(
            "Interval",
            [5, 10, 30],
            index=1,
            format_func=lambda s: f"Every {s}s",
            disabled=not auto_refresh_on,
            label_visibility="collapsed"
        )

    if auto_refresh_on:
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=interval * 1000, key="dashboard_autorefresh")
        except ImportError:
            st.caption(
                "Auto-refresh requires the optional `streamlit-autorefresh` "
                "package (`pip install streamlit-autorefresh`)."
            )

    return refresh, export