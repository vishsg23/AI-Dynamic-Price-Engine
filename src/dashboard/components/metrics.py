import streamlit as st
from utils.html_utils import render_html

COLOR_SUCCESS = "var(--success)"
COLOR_WARNING = "var(--warning)"
COLOR_DANGER = "var(--danger)"
COLOR_INFO = "var(--info)"


def hero_kpi(label, value, caption):
    render_html(
        f'<div class="hero-kpi">'
        f'<div class="hero-label">{label}</div>'
        f'<div class="hero-value">{value}</div>'
        f'<div class="hero-caption">{caption}</div>'
        f'</div>'
    )


def metric_card(label, value, icon, color, caption, suffix=""):
    render_html(
        f'<div class="kpi-card">'
        f'<div class="kpi-top">'
        f'<span class="kpi-label">{label}</span>'
        f'<div class="kpi-icon" style="background:{color};">{icon}</div>'
        f'</div>'
        f'<div>'
        f'<div class="kpi-value">{value}{suffix}</div>'
        f'<div class="kpi-caption">{caption}</div>'
        f'</div>'
        f'</div>'
    )


def render_metrics(filtered):
    comparable = filtered[filtered["is_profit_comparable"]]

    total_gain = comparable["profit_uplift"].sum()
    avg_profit_pct = comparable["profit_uplift_pct"].mean()
    optimized = len(filtered)
    avg_change = filtered["price_change_pct"].mean()
    spoilage_total = filtered.loc[~filtered["is_profit_comparable"], "spoilage_savings_estimate"].sum()

    hero_kpi(
        "Total profit uplift (this view)",
        f"${total_gain:,.0f}",
        f"Across {len(comparable):,} price-optimized products"
        + (f" · {avg_profit_pct:.1f}% average uplift" if avg_profit_pct == avg_profit_pct else "")
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Optimized products", f"{optimized:,}", "&#128722;", COLOR_INFO,
                     "Products with an active pricing recommendation")
    with c2:
        metric_card("Average price change", f"{avg_change:+.1f}", "&#128200;", COLOR_INFO,
                     "Mean recommended price adjustment", suffix="%")
    with c3:
        price_increases = len(filtered[filtered["price_change_pct"] > 0])
        metric_card("Price increases", f"{price_increases:,}", "&#128200;", COLOR_SUCCESS,
                     "Products where the model recommends raising price")

    c4, c5, c6 = st.columns(3)
    with c4:
        price_cuts = len(filtered[filtered["price_change_pct"] < 0])
        metric_card("Price cuts", f"{price_cuts:,}", "&#127991;", COLOR_WARNING,
                     "Products where the model recommends lowering price")
    with c5:
        metric_card("Spoilage savings rescued", f"${spoilage_total:,.0f}", "&#128176;", COLOR_SUCCESS,
                     "Revenue rescued from near-expiry write-off")
    with c6:
        critical_inventory = len(
            filtered[filtered["stock_urgency_category"] == "Critical"]
        ) if "stock_urgency_category" in filtered.columns else 0
        metric_card(
            "Critical inventory", f"{critical_inventory:,}", "&#9888;",
            COLOR_DANGER if critical_inventory > 0 else COLOR_SUCCESS,
            "Needs immediate stock attention" if critical_inventory > 0 else "All clear"
        )


def executive_summary_card(filtered):
    comparable = filtered[filtered["is_profit_comparable"]]

    total_gain = comparable["profit_uplift"].sum()
    optimized = len(filtered)
    avg_change = filtered["price_change_pct"].mean()
    critical = len(
        filtered[filtered["stock_urgency_category"] == "Critical"]
    ) if "stock_urgency_category" in filtered.columns else 0

    top_dept_line = ""
    if "department" in comparable.columns and not comparable.empty and total_gain > 0:
        by_dept = comparable.groupby("department")["profit_uplift"].sum().sort_values(ascending=False)
        if len(by_dept) and by_dept.iloc[0] > 0:
            top_dept = by_dept.index[0]
            top_share = (by_dept.iloc[0] / total_gain) * 100
            top_dept_line = f"<li>{top_dept} contributes {top_share:.0f}% of total gains</li>"

    critical_line = (
        f"<li>{critical} product(s) need immediate inventory attention</li>"
        if critical > 0 else "<li>No critical inventory alerts</li>"
    )

    render_html(
        f'<div class="note-box">'
        f'<b>Today\'s Summary</b>'
        f'<ul style="margin:6px 0 0 18px;padding:0;">'
        f'<li>Expected profit uplift: ${total_gain:,.0f}</li>'
        f'<li>{optimized:,} products optimized</li>'
        f'{top_dept_line}'
        f'{critical_line}'
        f'<li>Average recommended price change: {avg_change:+.1f}%</li>'
        f'</ul>'
        f'</div>'
    )