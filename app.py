"""StratLens entry point: streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="StratLens", layout="wide", initial_sidebar_state="expanded")

from stratlens import theme, ui  # noqa: E402  (after set_page_config)

ui.inject_css()

pages = {
    "Analysis": [
        st.Page("views/benchmark.py", title="Overview", icon=":material/space_dashboard:", default=True),
        st.Page("views/reports.py", title="Briefs", icon=":material/description:", url_path="briefs"),
        st.Page("views/ask.py", title="Ask", icon=":material/forum:", url_path="ask"),
    ],
    "Data": [
        st.Page("views/analyse.py", title="Upload", icon=":material/upload_file:", url_path="upload"),
    ],
    "Model": [
        st.Page("views/how_it_works.py", title="How it works", icon=":material/account_tree:", url_path="how-it-works"),
        st.Page("views/audit.py", title="Evaluation", icon=":material/monitoring:", url_path="evaluation"),
        st.Page("views/model_card.py", title="Model card", icon=":material/badge:", url_path="model-card"),
    ],
}

logo = "logo_dark.svg" if theme.is_dark() else "logo.svg"
st.logo(str(ui.ROOT / "assets" / logo), size="large")
nav = st.navigation(pages, position="sidebar")

metrics = ui.load_metrics()
if metrics:
    with st.sidebar:
        st.markdown(f'<div class="sl-sbfoot">model · tf-idf + lr<br>data · {metrics["generated"]}</div>',
                    unsafe_allow_html=True)

nav.run()
