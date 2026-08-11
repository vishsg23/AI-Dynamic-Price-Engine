import streamlit as st
import pandas as pd
import numpy as np
import json
import os

import plotly.express as px
import plotly.graph_objects as go

from utils.html_utils import render_html

BACKGROUND = "#1E1E22"
CARD = "#242428"
GRID = "#34343A"
TEXT = "#EDEDEF"
TEXT_LIGHT = "#B4B4BA"

GREEN = "#4FCB8A"
RED = "#E0654A"
BLUE = "#4FA8CB"
PURPLE = "#A379D9"
AMBER = "#E8A33D"

DEPARTMENT_COLORS_SEQ = [GREEN, AMBER, BLUE, PURPLE, RED, "#7BC96F", "#D98CD9", "#6FA8C9"]


def apply_layout(fig, height=500):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        height=height,
        title=dict(text=""),
        font=dict(family="Inter", color=TEXT, size=13),
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=12, color=TEXT_LIGHT)),
        margin=dict(l=50, r=30, t=60, b=50),
        hoverlabel=dict(bgcolor="#2A2A2F", bordercolor=GRID, font_size=13,
                         font_family="Inter", font_color=TEXT),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, gridwidth=1, griddash="dot",
                      zeroline=False, tickfont=dict(size=13, color=TEXT_LIGHT),
                      title_font=dict(size=13, color=TEXT_LIGHT), automargin=True)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, gridwidth=1, griddash="dot",
                      zeroline=False, tickfont=dict(size=13, color=TEXT_LIGHT),
                      title_font=dict(size=13, color=TEXT_LIGHT), automargin=True)
    return fig


def chart_header(title, subtitle=""):
    render_html(f'<div class="dashboard-card"><h3 style="margin-bottom:4px;">{title}</h3><p>{subtitle}</p></div>')


def insight_box(title, text, color=None):
    import re
    formatted = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    formatted = re.sub(r"\s+", " ", formatted).strip()
    render_html(
        f'<div class="note-box"><b>{title}</b><br>{formatted}</div>'
    )


def empty_state(message):
    render_html(f'<div class="note-box">{message}</div>')


def render_profit_waterfall(df):
    chart_header("Profit contribution waterfall", "How each pricing strategy builds total profit")

    comparable = df[df["is_profit_comparable"]]
    by_method = comparable.groupby("optimization_method", as_index=False)["profit_uplift"].sum()
    by_method = by_method.sort_values("profit_uplift", ascending=False)

    if by_method.empty:
        empty_state("No optimization data is available for the selected filters. Try another department or product.")
        return

    measures = ["relative"] * len(by_method) + ["total"]
    x = by_method["optimization_method"].tolist() + ["Total"]
    y = by_method["profit_uplift"].tolist() + [by_method["profit_uplift"].sum()]

    fig = go.Figure(go.Waterfall(
        orientation="v", measure=measures, x=x, y=y,
        connector=dict(line=dict(color="#5A5A62")),
        increasing=dict(marker=dict(color=GREEN)),
        decreasing=dict(marker=dict(color=RED)),
        totals=dict(marker=dict(color=BLUE)),
        text=[f"${v:,.0f}" for v in y], textposition="outside"
    ))
    apply_layout(fig, 480)
    fig.update_layout(yaxis_title="Profit uplift ($)", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def render_department_bar(df):
    chart_header("Profit by department", "Where the pricing engine is finding the most value")
    comparable = df[df["is_profit_comparable"]]
    dept = comparable.groupby("department", as_index=False)["profit_uplift"].sum().sort_values("profit_uplift", ascending=False)
    if dept.empty:
        empty_state("No optimization data is available for the selected filters. Try another department or product.")
        return
    fig = px.bar(dept, x="department", y="profit_uplift", color="department",
                 color_discrete_sequence=DEPARTMENT_COLORS_SEQ)
    apply_layout(fig, 480)
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Profit uplift ($)")
    st.plotly_chart(fig, use_container_width=True)


def render_scatter_explorer(df):
    chart_header("Price change vs profit uplift", "Bubble size = current profit, color = department")
    plot_df = df[df["is_profit_comparable"]].copy()
    if plot_df.empty:
        empty_state("No optimization data is available for the selected filters. Try another department or product.")
        return
    plot_df["bubble_size"] = plot_df["current_profit"].clip(lower=1)

    fig = px.scatter(
        plot_df, x="price_change_pct", y="profit_uplift", size="bubble_size",
        color="department", color_discrete_sequence=DEPARTMENT_COLORS_SEQ,
        hover_name="product_name",
        hover_data={"current_price": ":.2f", "recommended_price": ":.2f",
                    "price_change_pct": ":.1f", "profit_uplift": ":.2f", "bubble_size": False}
    )
    apply_layout(fig, 550)
    fig.add_hline(y=0, line_dash="dash", line_color="#5A5A62")
    fig.add_vline(x=0, line_dash="dash", line_color="#5A5A62")
    fig.update_layout(xaxis_title="Price change (%)", yaxis_title="Profit uplift ($)")
    st.plotly_chart(fig, use_container_width=True)


def render_product_table(df):
    chart_header("Product table", "Full catalog with active filters applied")
    if df.empty:
        empty_state("No products match the selected filters. Try another department or product.")
        return
    display_cols = [
        "product_id", "product_name", "department", "current_price", "recommended_price",
        "price_change_pct", "predicted_demand", "profit_uplift", "optimization_method",
        "stock_urgency_category"
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(
        df[display_cols].sort_values("profit_uplift", ascending=False, na_position="last"),
        use_container_width=True, hide_index=True, height=420,
        column_config={
            "current_price": st.column_config.NumberColumn(format="$%.2f"),
            "recommended_price": st.column_config.NumberColumn(format="$%.2f"),
            "price_change_pct": st.column_config.NumberColumn(format="%+.1f%%"),
            "profit_uplift": st.column_config.NumberColumn(format="$%.2f"),
        }
    )


def render_price_simulator(df, product_lookup, prediction_service):
    chart_header("Live price simulator", "Drag the slider — the actual model predicts demand in real time")

    product_ids = df["product_id"].unique().tolist()
    if not product_ids:
        empty_state("No products available to simulate for the current filters.")
        return

    labels = df.drop_duplicates("product_id").set_index("product_id")["product_name"]
    selected_id = st.selectbox(
        "Product", product_ids,
        format_func=lambda pid: f"{labels.get(pid, pid)} (#{pid})",
        key="simulator_product"
    )

    if selected_id not in product_lookup.index:
        empty_state("No pricing data available for this product yet. Try another product.")
        return

    try:
        row = product_lookup.loc[selected_id].to_dict()
        row["product_id"] = selected_id
        base_price = float(row.get("base_price", row.get("current_price", 10)))

        lo, hi = round(base_price * 0.6, 2), round(base_price * 1.6, 2)
        if lo >= hi:
            hi = lo + 1.0

        candidate_price = st.slider(
            "Candidate price", min_value=lo, max_value=hi, value=round(base_price, 2),
            step=0.01, key="simulator_price"
        )

        prices = np.linspace(lo, hi, 40)
        demands = [prediction_service.predict_demand(p, row) for p in prices]
        profits = [(p - row.get("cost_price", 0)) * d for p, d in zip(prices, demands)]

        live_demand = prediction_service.predict_demand(candidate_price, row)
        live_profit = (candidate_price - row.get("cost_price", 0)) * live_demand

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted demand at this price", f"{live_demand:.0f} units")
        c2.metric("Predicted profit at this price", f"${live_profit:,.2f}")
        c3.metric("Base price", f"${base_price:.2f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prices, y=demands, mode="lines", name="Predicted demand",
                                  line=dict(color=BLUE, width=2)))
        fig.add_vline(x=candidate_price, line_dash="dash", line_color=AMBER,
                      annotation_text="Selected", annotation_font_color=AMBER)
        fig.add_vline(x=base_price, line_dash="dot", line_color=TEXT_LIGHT,
                      annotation_text="Base", annotation_font_color=TEXT_LIGHT)
        apply_layout(fig, 380)
        fig.update_layout(xaxis_title="Price ($)", yaxis_title="Predicted demand (units)")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=prices, y=profits, mode="lines", name="Predicted profit",
                                   line=dict(color=GREEN, width=2)))
        fig2.add_vline(x=candidate_price, line_dash="dash", line_color=AMBER)
        apply_layout(fig2, 380)
        fig2.update_layout(xaxis_title="Price ($)", yaxis_title="Predicted profit ($)")
        st.plotly_chart(fig2, use_container_width=True)

    except Exception:
        empty_state("No pricing data available for this product yet. Try another product.")


def render_strategy_donut(df):
    chart_header("Optimization strategy distribution", "Which pricing logic each product used")
    if df.empty:
        empty_state("No products match the selected filters.")
        return
    method = df["optimization_method"].value_counts().reset_index()
    method.columns = ["Method", "Count"]
    fig = px.pie(method, names="Method", values="Count", hole=0.6,
                 color_discrete_sequence=[BLUE, GREEN, AMBER, PURPLE, RED])
    fig.update_traces(textinfo="percent+label")
    apply_layout(fig, 450)
    st.plotly_chart(fig, use_container_width=True)


def render_strategy_by_department(df):
    chart_header("Strategy mix by department", "Which departments lean on which pricing logic")
    comparable = df[df["is_profit_comparable"]]
    grouped = comparable.groupby(["department", "optimization_method"], as_index=False)["profit_uplift"].sum()
    if grouped.empty:
        empty_state("No optimization data is available for the selected filters. Try another department or product.")
        return
    fig = px.bar(grouped, x="department", y="profit_uplift", color="optimization_method",
                 barmode="group", color_discrete_sequence=[BLUE, GREEN, AMBER])
    apply_layout(fig, 480)
    fig.update_layout(xaxis_title="", yaxis_title="Profit uplift ($)")
    st.plotly_chart(fig, use_container_width=True)


def render_elasticity_scatter(df):
    chart_header("Price elasticity vs profit uplift", "Does more elastic demand mean more upside?")
    if "price_elasticity" not in df.columns:
        empty_state("price_elasticity column not present in this results file.")
        return
    plot_df = df[df["is_profit_comparable"]].dropna(subset=["price_elasticity", "profit_uplift"])
    if plot_df.empty:
        empty_state("No comparable products with elasticity data for the selected filters.")
        return
    fig = px.scatter(plot_df, x="price_elasticity", y="profit_uplift", color="department",
                      color_discrete_sequence=DEPARTMENT_COLORS_SEQ, hover_name="product_name")
    apply_layout(fig, 480)
    fig.update_layout(xaxis_title="Price elasticity", yaxis_title="Profit uplift ($)")
    st.plotly_chart(fig, use_container_width=True)
    corr = plot_df["price_elasticity"].corr(plot_df["profit_uplift"])
    insight_box(
        "Relationship strength",
        f"Correlation between elasticity and uplift: **{corr:.2f}**. Values near 0 mean elasticity alone doesn't predict which products benefit most."
    )


def render_spoilage_bar(df):
    chart_header("Spoilage savings rescued", "Revenue saved from near-expiry write-off, by product")
    expiry_df = df[~df["is_profit_comparable"]].copy()
    if expiry_df.empty:
        st.success("No products currently flagged for expiry markdown.")
        return
    expiry_df = expiry_df.nlargest(15, "spoilage_savings_estimate")
    fig = px.bar(expiry_df.sort_values("spoilage_savings_estimate"), x="spoilage_savings_estimate",
                 y="product_name", orientation="h", color_discrete_sequence=[AMBER])
    apply_layout(fig, max(380, 28 * len(expiry_df)))
    fig.update_layout(xaxis_title="Spoilage savings rescued ($)", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)


def render_stock_heatmap(df):
    chart_header("Stock urgency by department", "Where inventory risk is concentrated")
    if "stock_urgency_category" not in df.columns or df.empty:
        empty_state("No inventory data available for the selected filters.")
        return
    pivot = pd.crosstab(df["department"], df["stock_urgency_category"])
    if pivot.empty:
        empty_state("No inventory data available for the selected filters.")
        return
    fig = px.imshow(pivot, text_auto=True, color_continuous_scale="Oranges", aspect="auto")
    apply_layout(fig, 420)
    fig.update_layout(xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)


def render_expiry_table(df):
    chart_header("Expiry markdown products", "Honest metric: revenue rescued, not profit_uplift")
    expiry_df = df[~df["is_profit_comparable"]]
    if expiry_df.empty:
        st.success("No products currently flagged for expiry markdown.")
        return
    cols = ["product_id", "product_name", "current_price", "recommended_price", "spoilage_savings_estimate"]
    cols = [c for c in cols if c in expiry_df.columns]
    st.dataframe(
        expiry_df[cols].sort_values("spoilage_savings_estimate", ascending=False),
        use_container_width=True, hide_index=True, height=350,
        column_config={
            "current_price": st.column_config.NumberColumn(format="$%.2f"),
            "recommended_price": st.column_config.NumberColumn(format="$%.2f"),
            "spoilage_savings_estimate": st.column_config.NumberColumn(format="$%.2f"),
        }
    )


def render_model_health():
    chart_header("Model validation", "Same checks run during retrain_all.py, made visible here")

    val_path = os.path.join("models", "validation_metrics.json")
    if not os.path.exists(val_path):
        empty_state("No validation_metrics.json found yet — run `python src/ml/retrain_all.py` to populate this tab.")
        return

    with open(val_path) as f:
        v = json.load(f)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test MAPE", f"{v['test_mape']:.1f}%")
    c2.metric("Train MAPE", f"{v['train_mape']:.1f}%")
    c3.metric("Naive baseline MAPE", f"{v['naive_mape']:.1f}%")
    c4.metric("Beats naive by", f"{v['naive_mape'] - v['test_mape']:.1f} pts")

    imp = pd.Series(v["feature_importance"]).sort_values(ascending=True)
    fig = px.bar(x=imp.values, y=imp.index, orientation="h", color_discrete_sequence=[BLUE])
    apply_layout(fig, 380)
    fig.update_layout(xaxis_title="Importance", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    stats = v["per_product_mape_stats"]
    st.markdown("**Per-product MAPE distribution**")
    stat_cols = st.columns(5)
    for col, key in zip(stat_cols, ["min", "25%", "50%", "75%", "max"]):
        col.metric(key, f"{stats.get(key, 0):.1f}%")

    st.caption(f"Best iteration: {v['best_iteration']} / {v['n_estimators_max']} max — early stopping active if this is below the max.")

    insight_box(
        "Note on Ridge",
        "Ridge regression was evaluated during development and removed — it was training against a mismatched dataset and added no real price-response signal. XGBoost's own current_price feature carries price sensitivity instead."
    )