"""Upload: run a new report through the pipeline (extract, select pages, segment, classify, score specificity)."""
import hashlib
import json

import pandas as pd
import streamlit as st

from stratlens import model as mdl
from stratlens import specificity, ui
from stratlens.config import COMPANY_NAMES, THEME_NAMES, THEMES
from stratlens.extract import ExtractionError, extract_bytes, parse_filename
from stratlens.preprocess import segment
from stratlens.sections import Section, resolve_sections

STEPS = ["Upload", "Pages", "Classify", "Review"]
YEARS = [f"FY{y}" for y in range(2020, 2028)]

ui.header("Upload", "Classify a new annual report with the trained model.")
model, meta = ui.load_model()
if model is None:
    ui.empty_state("No trained model.", "Run <code>python train.py</code>.")
    st.stop()


@st.cache_data(show_spinner=False, max_entries=4)
def _extract(data: bytes, name: str):
    return extract_bytes(data, name)


step_slot = st.empty()
with ui.card("up", "Report"):
    up = st.file_uploader("PDF or TXT", type=["pdf", "txt"], label_visibility="collapsed")
    company, year = "", None
    if up is not None:
        g_company, g_year = parse_filename(up.name)
        c1, c2, _ = st.columns([1, 1, 2])
        with c1:
            company = st.text_input("Company code", value=g_company or "", max_chars=12).strip().upper()
        with c2:
            year = st.selectbox("Year", YEARS, index=YEARS.index(g_year) if g_year in YEARS else len(YEARS) - 2)
if up is None or not company:
    with step_slot.container():
        ui.steps(STEPS, 0)
    st.stop()

data = up.getvalue()
file_key = hashlib.md5(data).hexdigest()
if st.session_state.get("an_file") != file_key:
    st.session_state["an_file"] = file_key
    st.session_state.pop("an_preds", None)
try:
    with st.spinner(f"Reading {up.name}…"):
        doc = _extract(data, up.name)
except ExtractionError as exc:
    with step_slot.container():
        ui.steps(STEPS, 0)
    ui.note(str(exc), warn=True)
    st.stop()
with step_slot.container():
    ui.steps(STEPS, 1 if "an_preds" not in st.session_state else 3)

# ---------- pages ----------
detected = resolve_sections(doc, company, year)
method = detected[0].method if detected else None
with ui.card("pages", "Pages", meta={"override": "confirmed ranges", "heading": "detected"}.get(method, "manual")):
    ui.kpi_strip([(f"{doc.page_count:,}", "pages"), (f"{doc.char_count / 1000:,.0f}k", "characters"),
                  (f"{doc.scanned_share:.0%}", "scanned")])
    if doc.looks_scanned:
        ui.note("Looks like a scanned PDF; text is incomplete. Run OCR first.", warn=True)
    if method is None:
        ui.note("No letter or MD&A heading found. Enter the pages below.", warn=True)
    sec_df = pd.DataFrame([{"section": s.section, "start_page": s.start_page, "end_page": s.end_page} for s in detected]
                          or [{"section": "MDA", "start_page": 1, "end_page": min(doc.page_count, 10)}])
    c1, c2 = st.columns([5, 7], gap="medium")
    with c1:
        edited = st.data_editor(
            sec_df, num_rows="dynamic", hide_index=True, key=f"sec_{file_key}", use_container_width=True,
            column_config={
                "section": st.column_config.SelectboxColumn("Section", options=["LETTER", "MDA"], required=True),
                "start_page": st.column_config.NumberColumn("Start", min_value=1, max_value=doc.page_count, step=1,
                                                            required=True),
                "end_page": st.column_config.NumberColumn("End", min_value=1, max_value=doc.page_count, step=1,
                                                          required=True)})
    sections, problems = [], []
    for r in edited.dropna().itertuples():
        if int(r.start_page) > int(r.end_page):
            problems.append(f"{r.section}: start {int(r.start_page)} > end {int(r.end_page)}")
        else:
            sections.append(Section(r.section, int(r.start_page), int(r.end_page), "manual"))
    if problems or not sections:
        ui.note("Fix the page ranges. " + "; ".join(problems), warn=True)
        st.stop()
    res = segment(doc, sections, company, year)
    passages, drops = res.passages, res.drops
    with c2:
        ui.kpi_strip([(f"{sum(len(s.pages()) for s in sections)}", "pages used"), (f"{len(passages):,}", "passages"),
                      (f"{len(drops):,}", "blocks dropped")])
    if passages.empty:
        ui.note("No passages survived cleaning. Point the ranges at narrative text.", warn=True)
        st.stop()

# ---------- classify ----------
sig = (file_key, company, year, tuple((s.section, s.start_page, s.end_page) for s in sections))
if st.button(f"Classify {len(passages):,} passages", type="primary", icon=":material/play_arrow:"):
    with st.spinner("Classifying…"):
        pf = mdl.predict_frame(model, passages["text"], meta["threshold"])
        out = pd.concat([passages.reset_index(drop=True), pf], axis=1)
        out["source"] = f"Uploaded: {up.name}"
        spec = specificity.score_texts(out["text"].tolist())
        out["specificity"] = [x["specificity"] for x in spec]
        out["spec_tier"] = [x["spec_tier"] for x in spec]
        out["spec_signals"] = [json.dumps(x["spec_signals"], ensure_ascii=False) for x in spec]
        st.session_state["an_preds"], st.session_state["an_sig"] = out, sig
    st.rerun()
out = st.session_state.get("an_preds")
if out is None:
    st.stop()
if st.session_state.get("an_sig") != sig:
    ui.note("Settings changed since the last run. Classify again.", warn=True)

# ---------- review ----------
strategic = out[out.pred != "OTHER"]
dist = strategic["pred"].value_counts(normalize=True)
with ui.card("rev", "Results", meta=f"{company} {year}"):
    ui.kpi_strip([(f"{len(out):,}", "passages"), (f"{len(strategic):,}", "strategy"),
                  (f"{int(out.needs_review.sum())}", "low confidence"),
                  (f"{dist.index[0]} {dist.iloc[0]:.0%}" if len(dist) else "–", "top theme")])
    f1, f2, f3 = st.columns([2, 1, 1], vertical_alignment="bottom")
    with f1:
        theme_filter = st.multiselect("Themes", THEMES, default=[], format_func=lambda t: THEME_NAMES[t],
                                      placeholder="All themes")
    with f2:
        sort = st.selectbox("Sort", ["Lowest confidence", "Highest confidence", "Page"])
    with f3:
        n_rows = st.selectbox("Rows", [15, 40, 100])
    view = out[out.pred.isin(theme_filter)] if theme_filter else out
    view = {"Lowest confidence": view.sort_values("confidence"),
            "Highest confidence": view.sort_values("confidence", ascending=False),
            "Page": view.sort_values(["page", "passage_id"])}[sort]
    if view.empty:
        ui.empty_state("No passages match.")
    else:
        ui.predictions_table(view, max_rows=n_rows)
    with st.expander("Inspect a passage step by step"):
        pick = st.selectbox("Passage", view.passage_id.tolist() if not view.empty else out.passage_id.tolist(),
                            format_func=lambda pid: out.set_index("passage_id").loc[pid, "text"][:100] + "…")
        ui.pipeline_inspector(out.set_index("passage_id").loc[pick, "text"])
    b1, b2, _ = st.columns([1, 1.2, 3])
    cols = ["passage_id", "company", "year", "section", "page", "word_count", "pred", "confidence", "needs_review",
            "specificity", "spec_tier"] + [f"p_{c}" for c in model.classes_] + ["text"]
    with b1:
        st.download_button("CSV", out[cols].to_csv(index=False).encode("utf-8"), icon=":material/download:",
                           file_name=f"stratlens_{company}_{year}.csv", mime="text/csv", use_container_width=True)
    with b2:
        session = st.session_state.setdefault("session_reports", {})
        if st.button("Add to Overview", icon=":material/add:", use_container_width=True):
            session[f"{company}_{year}"] = out.assign(in_training_set=False)
            st.toast(f"{COMPANY_NAMES.get(company, company)} {year} added to Overview.")
