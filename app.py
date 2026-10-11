"""StratLens entry point: streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="StratLens", page_icon="assets/favicon.png", layout="wide", initial_sidebar_state="expanded")

from stratlens import theme, ui  # noqa: E402  (after set_page_config)

ui.inject_css()
ui.apply_theme_switch()

pages = {
    "Analyse": [
        st.Page("views/home.py", title="Home", icon=":material/home:", default=True),
        st.Page("views/company.py", title="Company", icon=":material/apartment:", url_path="company"),
        st.Page("views/compare.py", title="Compare", icon=":material/compare_arrows:", url_path="compare"),
        st.Page("views/ask.py", title="Ask", icon=":material/forum:", url_path="ask"),
    ],
    "Data": [
        st.Page("views/add_report.py", title="Add a report", icon=":material/upload_file:", url_path="add-report"),
    ],
    "Model": [
        st.Page("views/how_it_works.py", title="How it works", icon=":material/account_tree:", url_path="how-it-works"),
    ],
}

logo = "logo_dark.svg" if theme.is_dark() else "logo.svg"
st.logo(str(ui.ROOT / "assets" / logo), size="large")
nav = st.navigation(pages, position="sidebar")
if st.query_params.get("demo") == "1" and "demo_started" not in st.session_state:  # ?demo=1: warm caches first
    st.session_state["demo_started"] = True
    ui.start_demo()
ui.demo_bar()
import os, time  # noqa: E402,E401
_t0 = time.perf_counter()
try:
    nav.run()
finally:
    if os.environ.get("STRATLENS_PROFILE"):
        print(f"PROFILE {nav.url_path or 'home'} {time.perf_counter() - _t0:.3f}s", flush=True)
