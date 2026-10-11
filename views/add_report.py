"""Add a report: run a new annual report through the same pipeline and see it next to the six companies.

The result lives only in this browser session; nothing is written to disk."""
import hashlib
import json

import pandas as pd
import streamlit as st

from stratlens import analytics, company_view, explain, features, specificity, ui
from stratlens import model as mdl
from stratlens.config import COMPANIES, PIPELINE_STEPS
from stratlens.extract import ExtractionError, extract_bytes, parse_filename
from stratlens.preprocess import segment
from stratlens.sections import Section, resolve_sections, validate

YEARS = [f"FY{y}" for y in range(2020, 2028)]

ui.header("Add a report", "Run another annual report through the same pipeline and compare it with the six "
                          "companies. It stays in this browser session only.")
model, meta = ui.load_model()
base, an = ui.load_passages(), ui.load_analytics()
if model is None or base is None or an is None:
    ui.empty_state("The model or the processed reports are missing.",
                   "Run <code>python train.py</code> and <code>python -m stratlens.build</code>.")
    st.stop()


@st.cache_data(show_spinner=False, max_entries=4)
def _extract(data: bytes, name: str):
    return extract_bytes(data, name)


progress = st.empty()


def show_progress(done: int, active: int | None = None) -> None:
    with progress.container():
        ui.pipeline_strip(active=active, done=done, href=None)


with ui.card("up", "1 · Report"):
    up = st.file_uploader("Annual report (PDF or TXT)", type=["pdf", "txt"], label_visibility="collapsed")
    company, year = "", None
    if up is not None:
        g_company, g_year = parse_filename(up.name)
        c1, c2, _ = st.columns([1, 1, 2])
        with c1:
            company = st.text_input("Company code", value=g_company or "", max_chars=12,
                                    help=f"Known codes: {', '.join(COMPANIES)}. A known code gets its peer group."
                                    ).strip().upper()
        with c2:
            year = st.selectbox("Year", YEARS, index=YEARS.index(g_year) if g_year in YEARS else len(YEARS) - 2)
if up is None or not company:
    show_progress(0, 0)
    ui.empty_state("Upload a PDF named like <code>britannia_FY2026.pdf</code> (or set the company and year above).",
                   "Only the Chairman's letter and the MD&A are read.")
    st.stop()

data = up.getvalue()
file_key = hashlib.md5(data).hexdigest()
if st.session_state.get("up_file") != file_key:
    st.session_state["up_file"] = file_key
    st.session_state.pop("up_out", None)
try:
    show_progress(0, 0)
    with st.spinner(f"Reading {up.name}…"):
        doc = _extract(data, up.name)
except ExtractionError as exc:
    ui.note(str(exc), warn=True)
    st.stop()

# ---------- Read · Clean · Split ----------
detected = resolve_sections(doc, company, year)
problems = validate(doc, detected) if detected else ["nothing detected"]
with ui.card("pages", "2 · Pages to read", meta=f"{doc.page_count:,} pages in the PDF"):
    if doc.looks_scanned:
        ui.note("This looks like a scanned PDF without a text layer; run OCR first.", warn=True)
    if problems:
        ui.note("The Chairman's letter and MD&A could not be located reliably (" + "; ".join(problems) + "). "
                "Enter the page ranges below, using the PDF's own page numbers.", warn=True)
    else:
        ui.meta("Detected automatically. Check the ranges against the PDF and correct them if needed.")
    sec_df = pd.DataFrame([{"section": s.section, "start_page": s.start_page, "end_page": s.end_page}
                           for s in detected] if not problems else
                          [{"section": "LETTER", "start_page": None, "end_page": None},
                           {"section": "MDA", "start_page": None, "end_page": None}])
    edited = st.data_editor(
        sec_df, num_rows="dynamic", hide_index=True, key=f"sec_{file_key}", use_container_width=False,
        column_config={
            "section": st.column_config.SelectboxColumn("Section", options=["LETTER", "MDA"], required=True),
            "start_page": st.column_config.NumberColumn("From page", min_value=1, max_value=doc.page_count, step=1),
            "end_page": st.column_config.NumberColumn("To page", min_value=1, max_value=doc.page_count, step=1)})
    sections, bad = [], []
    for r in edited.dropna().itertuples():
        if int(r.start_page) > int(r.end_page):
            bad.append(f"{r.section}: from {int(r.start_page)} > to {int(r.end_page)}")
        else:
            sections.append(Section(r.section, int(r.start_page), int(r.end_page), "manual"))
    if bad or not sections:
        show_progress(1, 1)
        ui.empty_state("Enter at least one page range." if not bad else "Fix the page ranges: " + "; ".join(bad))
        st.stop()
    res = segment(doc, sections, company, year)
    passages = res.passages
    ui.meta(f"{sum(len(s.pages()) for s in sections)} pages kept · {len(res.drops):,} blocks removed while cleaning "
            f"(tables, headers, boilerplate) · <b>{len(passages):,} passages</b>")
    if passages.empty:
        ui.note("No passages survived cleaning. Point the ranges at narrative text.", warn=True)
        st.stop()

sig = (file_key, company, year, tuple((s.section, s.start_page, s.end_page) for s in sections))
if st.session_state.get("up_sig") != sig:
    st.session_state.pop("up_out", None)

# ---------- Classify theme · Score concreteness ----------
if "up_out" not in st.session_state:
    show_progress(3, 3)
    if not st.button(f"Classify {len(passages):,} passages and compare", type="primary",
                     icon=":material/play_arrow:"):
        st.stop()
    with st.status("Running the pipeline…", expanded=True) as status:
        st.write(f"{PIPELINE_STEPS[3]['step']}…")
        pf = mdl.predict_frame(model, passages["text"], meta["threshold"])
        out = pd.concat([passages.reset_index(drop=True), pf], axis=1)
        out["why"] = explain.why_terms(model, out["text"])
        st.write(f"{PIPELINE_STEPS[4]['step']}…")
        spec = specificity.score_texts(out["text"].tolist())
        out["specificity"] = [x["specificity"] for x in spec]
        out["spec_tier"] = [x["spec_tier"] for x in spec]
        out["spec_signals"] = [json.dumps(x["spec_signals"], ensure_ascii=False) for x in spec]
        st.write(f"{PIPELINE_STEPS[5]['step']}…")
        st.session_state["up_out"], st.session_state["up_sig"] = out, sig
        status.update(label="Done", state="complete", expanded=False)

out = st.session_state["up_out"]
show_progress(6)


@st.cache_data(show_spinner=False, max_entries=4)
def _combined(sig_, _out: pd.DataFrame):
    rest = base[~((base.company == company) & (base.year == year))]
    allp = pd.concat([rest, _out], ignore_index=True)
    return allp, analytics.compute(allp)


with st.spinner("Comparing with the six companies…"):
    allp, an2 = _combined(sig, out)
key = f"{company}|{year}"
r = an2["reports"][key]
if r.get("all_avg") and len(r["all_avg"]["companies"]) <= 1:  # no other report that year: use the latest year's
    latest = max(an["years"])
    any_key = next(k for k in an["reports"] if k.endswith("|" + latest))
    r["all_avg"] = an["reports"][any_key]["all_avg"]
    ui.note(f"No other company has a {year} report here, so the averages are from {latest}.")

@st.cache_data(show_spinner=False, max_entries=4)
def _feats(sig_, _allp: pd.DataFrame) -> dict:
    return {"novelty": {}, "follow_up": {}, "promises": {key: features.promises(_allp, key)},
            "signature": {year: features.signature(_allp, year)}}


company_view.render(an2, allp, company, year, uid="up", feats=_feats(sig, allp))

with st.expander("Inspect a passage step by step"):
    pick = st.selectbox("Passage", out.passage_id.tolist(),
                        format_func=lambda pid: out.set_index("passage_id").loc[pid, "text"][:100] + "…")
    ui.pipeline_inspector(out.set_index("passage_id").loc[pick, "text"])
cols = ["passage_id", "company", "year", "section", "page", "word_count", "pred", "confidence", "needs_review",
        "specificity", "spec_tier"] + [f"p_{c}" for c in model.classes_] + ["text"]
export = out[cols].rename(columns={"pred": "theme", "specificity": "concreteness", "spec_tier": "concreteness_tier"})
export["concreteness_tier"] = export["concreteness_tier"].map(ui.TIER_NAMES).fillna(export["concreteness_tier"])
st.download_button("Download passages (CSV)", export.to_csv(index=False).encode("utf-8"),
                   icon=":material/download:", file_name=f"stratlens_{company}_{year}.csv", mime="text/csv")
