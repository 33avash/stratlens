"""StratLens entry point: streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="StratLens", layout="wide", initial_sidebar_state="expanded")

from stratlens import ui  # noqa: E402  (after set_page_config)

ui.inject_css()

pages = [
    st.Page("pages/benchmark.py", title="Benchmark", default=True),
    st.Page("pages/analyse.py", title="Analyse reports"),
    st.Page("pages/audit.py", title="Model audit"),
    st.Page("pages/model_card.py", title="Model card"),
]

st.logo(str(ui.ROOT / "assets" / "logo.svg"), size="large")
nav = st.navigation(pages, position="sidebar")

with st.sidebar:
    st.markdown('<p class="sl-brand-sub">FMCG strategy disclosure intelligence</p>', unsafe_allow_html=True)
    bench = ui.load_benchmark()
    if bench is not None:
        reports = ", ".join(f"{ui.company_name(c)} {y}" for c, y in
                            bench[["company", "year"]].drop_duplicates().sort_values(["company", "year"]).values)
        about = (f"Sample analysis of {bench[['company', 'year']].drop_duplicates().shape[0]} annual reports: "
                 f"{reports}. Only the Chairman/CEO letter and MD&A are analysed.")
    else:
        about = "No sample analysis built yet."
    st.markdown(f'<div class="sl-about"><b>About the data</b><br>{about}<br><br>Training labels were drafted by an '
                'AI assistant and are pending human verification. Theme shares measure disclosure emphasis, not '
                'investment or execution.</div>', unsafe_allow_html=True)

nav.run()
