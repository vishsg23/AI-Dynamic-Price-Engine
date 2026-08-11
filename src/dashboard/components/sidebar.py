import streamlit as st

from utils.html_utils import render_html


def render_sidebar(df):

    render_html(
        "<h2 style='margin-bottom:0px;font-size:20px;'>Filters</h2>"
        "<p style='font-size:13px;color:var(--text-muted);'>Filter products and pricing recommendations</p>",
        container=st.sidebar
    )

    filtered = df.copy()

    departments = ["All"] + sorted(df["department"].unique().tolist())
    department = st.sidebar.selectbox("Department", departments)
    if department != "All":
        filtered = filtered[filtered["department"] == department]

    search = st.sidebar.text_input(
        "Search Product",
        placeholder="Type product name..."
    )
    if search:
        filtered = filtered[
            filtered["product_name"].str.contains(search, case=False, na=False)
        ]

    with st.sidebar.expander("Advanced Filters", expanded=False):

        if filtered.empty:
            st.caption("No products match the filters above.")
        else:
            methods = ["All"] + sorted(filtered["optimization_method"].unique().tolist())
            method = st.selectbox("Optimization Method", methods)
            if method != "All":
                filtered = filtered[filtered["optimization_method"] == method]

            min_profit = float(filtered["profit_uplift"].min()) if not filtered.empty else 0.0
            max_profit = float(filtered["profit_uplift"].max()) if not filtered.empty else 1.0
            if min_profit == max_profit:
                max_profit = min_profit + 1.0
            profit_range = st.slider(
                "Profit Uplift ($)",
                min_value=min_profit,
                max_value=max_profit,
                value=(min_profit, max_profit)
            )
            filtered = filtered[
                (filtered["profit_uplift"] >= profit_range[0]) &
                (filtered["profit_uplift"] <= profit_range[1])
            ]

            min_change = float(filtered["price_change_pct"].min()) if not filtered.empty else 0.0
            max_change = float(filtered["price_change_pct"].max()) if not filtered.empty else 0.0
            if min_change == max_change:
                max_change = min_change + 1.0
            price_change = st.slider(
                "Price Change %",
                min_value=min_change,
                max_value=max_change,
                value=(min_change, max_change)
            )
            filtered = filtered[
                (filtered["price_change_pct"] >= price_change[0]) &
                (filtered["price_change_pct"] <= price_change[1])
            ]

            inventory_options = ["All"] + sorted(
                filtered["stock_urgency_category"].dropna().unique().tolist()
            )
            inventory = st.selectbox("Inventory Status", inventory_options)
            if inventory != "All":
                filtered = filtered[filtered["stock_urgency_category"] == inventory]

    if filtered.empty:
        render_html(
            '<div class="note-box">No products match the current filters.</div>',
            container=st.sidebar
        )
    else:
        dept_label = department if department != "All" else "all departments"
        render_html(
            f'<div class="note-box">Showing <b>{len(filtered):,}</b> products in <b>{dept_label}</b>.</div>',
            container=st.sidebar
        )

    reset = st.sidebar.button("Reset Filters", use_container_width=True)
    if reset:
        st.rerun()

    return filtered