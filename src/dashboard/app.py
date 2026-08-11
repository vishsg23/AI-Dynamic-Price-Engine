import os
import sys
import streamlit as st
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from utils.loader import load_results, load_product_lookup
from utils.html_utils import render_html
from components.header import render_header, render_controls
from components.sidebar import render_sidebar
from components.metrics import render_metrics, executive_summary_card
import charts.dashboard_charts as charts

st.set_page_config(page_title="Pricing Intelligence Dashboard", page_icon="📊",
                    layout="wide", initial_sidebar_state="expanded")

API_BASE = "http://localhost:8001/api/v1"


def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "styles", "dark_theme.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_ai_tab(product_data):
    render_html('<span class="ai-badge">&#129302; Powered by Gemini</span>')
    st.subheader("AI Pricing Assistant")

    if product_data is None:
        st.info("Select a product from the sidebar to get an explanation.")
        return

    left, right = st.columns([1, 2])
    with left:
        st.metric("Current price", f"${product_data['current_price']:.2f}")
        st.metric("Recommended", f"${product_data['recommended_price']:.2f}")
        st.metric("Price change", f"{product_data['price_change_pct']:+.2f}%")
        uplift = product_data.get("profit_uplift")
        st.metric("Profit uplift", f"${uplift:.2f}" if uplift == uplift else "N/A (expiry markdown)")

    with right:
        st.info("Click **Get AI Explanation** to retrieve the Gemini-generated pricing rationale for this product.")
        if st.button("Get AI Explanation", use_container_width=True, key="get_ai_explanation_btn"):
            with st.spinner("Generating recommendation..."):
                try:
                    response = requests.get(
                        f"{API_BASE}/price/{int(product_data['product_id'])}", timeout=20
                    )
                    if response.status_code == 200:
                        data = response.json()
                        explanation = data.get("ai_explanation", "No explanation was returned for this product.")
                        confidence_label = data.get("ai_confidence_label")
                        confidence_pct = data.get("ai_confidence_pct")
                        reasoning = data.get("ai_reasoning")

                        confidence_line = ""
                        if confidence_label and confidence_label != "Unavailable":
                            pct_str = f" ({confidence_pct}%)" if confidence_pct is not None else ""
                            confidence_line = f'<div style="color:var(--accent);font-weight:700;margin-bottom:6px;">Confidence: {confidence_label}{pct_str}</div>'

                        reasoning_line = f'<div style="color:var(--text-secondary);font-size:13px;margin-top:8px;">{reasoning}</div>' if reasoning else ""

                        render_html(
                            f'<div class="ai-output-card">'
                            f'{confidence_line}'
                            f'<div>{explanation}</div>'
                            f'{reasoning_line}'
                            f'</div>'
                        )
                    else:
                        render_html(
                            f'<div class="warning-card">The AI service responded with an unexpected status '
                            f'({response.status_code}). Please try again shortly.</div>'
                        )
                except requests.exceptions.ConnectionError:
                    render_html(
                        '<div class="danger-card"><b>AI service unavailable.</b><br>'
                        'Start it with: <code>uvicorn src.api.main:app --port 8001</code></div>'
                    )
                except requests.exceptions.Timeout:
                    render_html(
                        '<div class="danger-card"><b>AI service timed out.</b><br>Please try again.</div>'
                    )
                except Exception:
                    render_html(
                        '<div class="danger-card"><b>No explanation is available for this product right now.</b><br>'
                        'Please try again, or select a different product.</div>'
                    )


def main():
    load_css()

    try:
        df = load_results()
    except FileNotFoundError as e:
        st.error("Could not find data/outputs/optimization_results.csv. Run the pipeline first.")
        st.exception(e)
        st.stop()

    if df is None or df.empty:
        st.error("The dataset loaded successfully but contains no rows.")
        st.stop()

    render_header(df, API_BASE)
    refresh, export = render_controls()
    if refresh:
        st.cache_data.clear()
        st.rerun()

    filtered = render_sidebar(df)
    if filtered.empty:
        st.warning("No products match the current filters. Adjust the filters in the sidebar.")
        st.stop()

    st.caption("This dashboard answers one question: how is your pricing strategy performing right now, and what should you act on?")
    executive_summary_card(filtered)

    render_metrics(filtered)
    st.markdown("---")

    product_data = None
    if len(filtered):
        selected_label = st.sidebar.selectbox(
            "Product for AI Assistant / simulator",
            filtered["display_label"]
        )
        row = filtered[filtered["display_label"] == selected_label]
        if not row.empty:
            product_data = row.iloc[0].to_dict()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Executive Overview", "Product Explorer", "Pricing Strategy",
        "Inventory & Expiry Risk", "Model Health", "AI Assistant"
    ])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            charts.render_profit_waterfall(filtered)
        with c2:
            charts.render_department_bar(filtered)

    with tab2:
        charts.render_scatter_explorer(filtered)
        charts.render_product_table(filtered)
        st.markdown("---")
        try:
            product_lookup, prediction_service = load_product_lookup()
            charts.render_price_simulator(filtered, product_lookup, prediction_service)
        except FileNotFoundError:
            charts.empty_state(
                "Live simulator needs data/processed/master_features.csv and a trained model — "
                "run the training pipeline first."
            )

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            charts.render_strategy_donut(filtered)
        with c2:
            charts.render_strategy_by_department(filtered)
        charts.render_elasticity_scatter(filtered)

    with tab4:
        c1, c2 = st.columns(2)
        with c1:
            charts.render_spoilage_bar(filtered)
        with c2:
            charts.render_stock_heatmap(filtered)
        charts.render_expiry_table(filtered)

    with tab5:
        charts.render_model_health()

    with tab6:
        render_ai_tab(product_data)

    if export:
        st.sidebar.download_button(
            "Download filtered data (CSV)",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="filtered_pricing_results.csv",
            mime="text/csv",
            use_container_width=True
        )


if __name__ == "__main__":
    main() 