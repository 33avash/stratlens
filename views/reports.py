"""Briefs: a short strategy brief for one report, or a comparison of 2-3 companies in the same year."""
import html

import streamlit as st

from stratlens import compare, report, ui
from stratlens.llm import client as llm

preds = ui.load_benchmark()
metrics = ui.load_metrics()
ui.header("Briefs", "A one-page brief per report, or a side-by-side.")
if preds is None:
    ui.empty_state("No analysis yet.", "Run <code>python train.py</code>.")
    st.stop()

client = llm.get_client()
reports = preds[["company", "year"]].drop_duplicates().sort_values(["company", "year"])
c1, c2, c3 = st.columns([1.2, 2.4, 1], vertical_alignment="bottom")
with c1:
    mode = st.segmented_control("Type", ["Single", "Compare"], default="Single") or "Single"
if mode == "Single":
    with c2:
        options = [f"{c}|{y}" for c, y in reports.values]
        pick = st.selectbox("Report", options, index=len(options) - 1,
                            format_func=lambda k: f"{ui.company_name(k.split('|')[0])} · {k.split('|')[1]}")
    companies, year = [pick.split("|")[0]], pick.split("|")[1]
else:
    years = sorted(reports.year.unique())
    year = years[-1]
    available = sorted(reports[reports.year == year].company)
    with c2:
        companies = st.multiselect(f"Companies · {year}", available, default=available[:3], max_selections=3,
                                   format_func=ui.company_name)
with c3:
    use_ai = st.toggle("AI writing", value=client is not None, disabled=client is None,
                       help="Rewrites the narrative from the computed figures; every number is checked against the "
                            "data.")
if mode == "Compare" and len(companies) < 2:
    ui.empty_state("Pick at least two companies.")
    st.stop()


@st.cache_data(show_spinner=False, max_entries=32)
def _brief(companies: tuple, year: str, ai: bool, data_version: str) -> dict:
    return report.build_brief(preds, metrics, list(companies), year, client=client if ai else None)


with st.spinner("Writing brief…"):
    brief = _brief(tuple(companies), year, bool(use_ai and client), metrics["generated"] if metrics else "")
n, f = brief["narrative"], brief["facts"]
for nt in brief["notes"]:
    ui.note(html.escape(nt), warn=True)
slug = "_".join(companies) + f"_{year}"

left, right = st.columns([7, 5], gap="medium")
with left:
    with ui.card("brief_main", brief["title"], meta="AI-assisted" if brief["narrative_source"] != "template"
                 else "template"):
        st.markdown(f'<p class="sl-lede">{html.escape(n["summary"])}</p>', unsafe_allow_html=True)
        ui.subhead("Theme mix")
        st.markdown(html.escape(n["theme_mix"]))
        if brief["kind"] == "single":
            ui.subhead("Change")
            st.markdown(html.escape(n["shifts"]))
        ui.subhead("Specificity")
        st.markdown(html.escape(n["specificity"]))
        d1, d2, _ = st.columns([1, 1, 2])
        with d1:
            st.download_button("HTML", report.render_html(brief).encode("utf-8"), icon=":material/download:",
                               file_name=f"stratlens_{slug}.html", mime="text/html", use_container_width=True)
        with d2:
            try:
                st.download_button("PDF", report.render_pdf(brief), icon=":material/download:",
                                   file_name=f"stratlens_{slug}.pdf", mime="application/pdf",
                                   use_container_width=True)
            except Exception:
                pass

with right:
    for pf in ([f] if brief["kind"] == "single" else f["per_company"]):
        with ui.card(f"mix_{pf['company']}", pf["company_name"],
                     meta=f"{pf['n_strategic']} of {pf['n_passages']} passages are strategy"):
            rows = "".join(
                f"<tr><td>{ui.theme_label(t['code'])}</td><td class='r'>{t['share_pct']}%</td>"
                + (f"<td class='r'>{t['diff_vs_peers_pp']:+.1f}</td>" if "peer_avg_pct" in t else "") + "</tr>"
                for t in sorted(pf["themes"], key=lambda r: -r["share_pct"]))
            head = "<th>Theme</th><th class='r'>Share</th>" + (
                f"<th class='r'>{ui.tip('vs peers', compare.VS_PEERS_HELP)}</th>" if pf["peers"] else "")
            st.markdown(f"<table class='sl-table'><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>",
                        unsafe_allow_html=True)

with ui.card("brief_ev", "Evidence"):
    for pf in ([f] if brief["kind"] == "single" else f["per_company"]):
        for theme_name, items in pf["evidence"].items():
            for e in items[:2]:
                st.markdown(f'<div class="sl-quote"><div class="q">{html.escape(e["text"])}</div><div class="m">'
                            f'{html.escape(theme_name)} · {html.escape(e["company_name"])} {e["year"]} · '
                            f'p.{e["page"]} · confidence {e["confidence"]:.2f} · {e["spec_tier"].lower()}'
                            + (f' ({html.escape(e["spec_signals"])})' if e.get("spec_signals") else "")
                            + '</div></div>', unsafe_allow_html=True)
st.markdown(f'<p class="sl-meta">{html.escape(report.CAVEAT)}</p>', unsafe_allow_html=True)
