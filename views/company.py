"""Company: brief, benchmark against peers, and the evidence behind every share (replaces Briefs)."""
import streamlit as st

from stratlens import company_view, ui
from stratlens.config import COMPANIES, STRATEGIC_THEMES

an, passages = ui.load_analytics(), ui.load_passages()
if an is None or passages is None:
    ui.header("Company")
    ui.empty_state("No reports have been processed yet.", "Run <code>python -m stratlens.build</code>.")
    st.stop()

qp = st.query_params
companies = [c for c in COMPANIES if c in an["companies"]] + [c for c in an["companies"] if c not in COMPANIES]
years = an["years"]
if qp.get("company") in companies:
    st.session_state.setdefault("co_company", qp["company"])
if qp.get("year") in years:
    st.session_state.setdefault("co_year", qp["year"])
st.session_state.setdefault("co_company", companies[0])
st.session_state.setdefault("co_year", years[-1])
focus = qp.get("theme") if qp.get("theme") in STRATEGIC_THEMES else None

ui.header("Company", "What one company's strategy talks about, how it compares with its peers, and the passages "
                     "behind every number.")
c1, c2, _ = st.columns([2, 1, 3], vertical_alignment="bottom")
with c1:
    company = st.selectbox("Company", companies, format_func=ui.company_name, key="co_company")
with c2:
    year = st.selectbox("Year", years, key="co_year")
if (qp.get("company"), qp.get("year")) != (company, year):
    st.query_params.update({"company": company, "year": year})
    if qp.get("company") not in (None, company):
        focus = None

key = f"{company}|{year}"
r = an["reports"].get(key)
if r is None:
    others = [y for y in years if f"{company}|{y}" in an["reports"]]
    ui.empty_state(f"The {year} report for {ui.company_name(company)} is missing.",
                   f"Available: {', '.join(others)}." if others else "")
    st.stop()


def _focus(theme_code: str) -> None:
    st.query_params["theme"] = theme_code
    st.rerun()


company_view.render(an, passages, company, year, focus=focus, narrative=ui.load_narratives()["brief"].get(key),
                    on_theme_click=_focus, feats=ui.load_features())
