"""
===========================================================
Safe HTML rendering helper
===========================================================
Streamlit's st.markdown(..., unsafe_allow_html=True) runs the text
through a CommonMark parser first. CommonMark treats any line that
starts with 4+ spaces of indentation as a literal, pre-formatted
CODE BLOCK — not HTML to render. Since our card components are
written inside indented Python functions, the HTML strings end up
indented too, so instead of a styled card you get the raw tags
printed on screen.

render_html() strips leading whitespace from every line before
handing it to st.markdown, so this class of bug can't happen no
matter how the calling code is indented.
"""

import streamlit as st


def render_html(html_str: str, container=None):
    """Render an HTML string safely. Pass container=st.sidebar to render
    into the sidebar instead of the main area."""
    cleaned = "\n".join(line.strip() for line in html_str.strip().split("\n"))
    target = container if container is not None else st
    target.markdown(cleaned, unsafe_allow_html=True)