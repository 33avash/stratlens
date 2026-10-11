"""StratLens entry point: streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="StratLens", layout="wide", initial_sidebar_state="expanded")

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
st.navigation(pages, position="sidebar").run()
