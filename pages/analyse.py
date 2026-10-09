"""Analyse reports: upload -> check extraction -> classify -> review and export."""
import hashlib

import pandas as pd
import streamlit as st

from stratlens import model as mdl
from stratlens import ui
from stratlens.config import COMPANY_NAMES, THEME_NAMES, THEMES
from stratlens.extract import ExtractionError, extract_bytes, parse_filename
from stratlens.preprocess import segment
from stratlens.sections import Section, resolve_sections

STEPS = ["Upload", "Check extraction", "Classify", "Review & export"]
YEARS = [f"FY{y}" for y in range(2020, 2028)]

ui.page_header("Analyse reports",
               "Upload an annual report (PDF or TXT), confirm the company, year and pages to analyse, then classify "
               "its strategy passages with the trained model.")

model, meta = ui.load_model()
if model is None:
    ui.empty_state("No trained model is available.", "Run <code>python train.py</code>, then reload this page.")
    st.stop()


@st.cache_data(show_spinner=False, max_entries=4)
def _extract(data: bytes, name: str):
    return extract_bytes(data, name)


# ---------- 1. upload ----------
up = st.file_uploader("Annual report", type=["pdf", "txt"], label_visibility="collapsed",
                      help="Name files like HUL_FY2026.pdf to pre-fill company and year.")
if up is None:
    ui.steps(STEPS, 0)
    ui.empty_state("No report uploaded yet.", "Drop a PDF or TXT annual report above to begin.")
    st.stop()

data = up.getvalue()
file_key = hashlib.md5(data).hexdigest()
if st.session_state.get("an_file") != file_key:  # new file: reset downstream state
    st.session_state["an_file"] = file_key
    st.session_state.pop("an_preds", None)

g_company, g_year = parse_filename(up.name)
c1, c2, _ = st.columns([1, 1, 2])
with c1:
    company = st.text_input("Company code", value=g_company or "", max_chars=12,
                            help="Short code, e.g. HUL, DABUR, NESTLE. Used to label the report in the benchmark.")
    company = company.strip().upper()
with c2:
    year = st.selectbox("Financial year", YEARS, index=YEARS.index(g_year) if g_year in YEARS else len(YEARS) - 2)
if not company:
    ui.steps(STEPS, 0)
    ui.empty_state("Enter a company code to continue.", "For example HUL, DABUR or NESTLE.")
    st.stop()

# ---------- 2. check extraction ----------
try:
    size_mb = len(data) / 1e6
    with st.spinner(f"Extracting text from {up.name} ({size_mb:.1f} MB)…"):
        doc = _extract(data, up.name)
except ExtractionError as exc:
    ui.steps(STEPS, 0)
    ui.note(f"<b>Could not read this file.</b> {exc}", warn=True)
    st.stop()

ui.steps(STEPS, 1 if "an_preds" not in st.session_state else 3)
ui.section("Extraction check", "What was read from the file, before any filtering.", first=True)
ui.kpi_strip([
    (f"{doc.page_count:,}", "Pages"),
    (f"{doc.char_count:,}", "Characters extracted"),
    (f"{doc.scanned_share:.0%}", "Pages that look scanned"),
    (f"{len(doc.toc)}", "Bookmarks found"),
])
if doc.looks_scanned:
    ui.note("<b>This looks like a scanned PDF.</b> Most pages contain images but almost no text, so results will be "
            "incomplete. Use a text-based PDF or run OCR first.", warn=True)

detected = resolve_sections(doc, company, year)
method = detected[0].method if detected else None
ui.section("Pages to analyse",
           "Only the Chairman/CEO letter and the MD&A are analysed, so that financial statements, governance and "
           "BRSR text do not dominate. Edit the page ranges if detection is wrong.")
if method == "override":
    st.markdown('<p class="sl-meta">Page ranges come from the manually confirmed list in '
                '<code>config/section_overrides.csv</code>.</p>', unsafe_allow_html=True)
elif method == "heading":
    st.markdown('<p class="sl-meta">Page ranges were detected from headings. Check them against the PDF.</p>',
                unsafe_allow_html=True)
else:
    ui.note("No Chairman's letter or MD&A heading was detected. Enter the page ranges below.", warn=True)

sec_df = pd.DataFrame([{"section": s.section, "start_page": s.start_page, "end_page": s.end_page} for s in detected]
                      or [{"section": "MDA", "start_page": 1, "end_page": min(doc.page_count, 10)}])
edited = st.data_editor(
    sec_df, num_rows="dynamic", hide_index=True, key=f"sec_{file_key}", use_container_width=False,
    column_config={
        "section": st.column_config.SelectboxColumn("Section", options=["LETTER", "MDA"], required=True,
                                                    help="LETTER = Chairman/CEO letter; MDA = Management Discussion & Analysis"),
        "start_page": st.column_config.NumberColumn("Start page", min_value=1, max_value=doc.page_count, step=1, required=True),
        "end_page": st.column_config.NumberColumn("End page", min_value=1, max_value=doc.page_count, step=1, required=True),
    })
sections, problems = [], []
for r in edited.dropna().itertuples():
    if int(r.start_page) > int(r.end_page):
        problems.append(f"{r.section}: start page {int(r.start_page)} is after end page {int(r.end_page)}")
    else:
        sections.append(Section(r.section, int(r.start_page), int(r.end_page), "manual"))
if problems:
    ui.note("Fix the page ranges: " + "; ".join(problems) + ".", warn=True)
if not sections:
    ui.empty_state("No valid page ranges.", "Add at least one section with a start and end page.")
    st.stop()

res = segment(doc, sections, company, year)
passages, drops = res.passages, res.drops
drop_counts = drops["reason"].value_counts()
reason_names = {"caption": "headings / captions", "short": "under 8 words", "table": "tables / figures",
                "statutory": "statutory boilerplate", "numeric": "mostly numbers", "duplicate": "duplicates"}
ui.kpi_strip([
    (f"{sum(len(s.pages()) for s in sections)}", "Pages selected"),
    (f"{len(passages):,}", "Passages kept"),
    (f"{len(drops):,}", "Blocks dropped"),
    (f"{passages['word_count'].median():.0f}" if len(passages) else "–", "Median words per passage"),
])
if len(drop_counts):
    st.markdown('<p class="sl-meta">Dropped: ' + " · ".join(f"{reason_names.get(k, k)} {v}" for k, v in
                drop_counts.items()) + "</p>", unsafe_allow_html=True)
if passages.empty:
    ui.note("No passages survived cleaning on these pages. Check that the page ranges point at narrative text.",
            warn=True)
    st.stop()
with st.expander(f"Preview the first {min(8, len(passages))} kept passages"):
    for r in passages.head(8).itertuples():
        st.markdown(f'<p style="margin:0 0 10px 0">{r.text}<br><span class="sl-meta">{ui.section_name(r.section)} · '
                    f'p. {r.page} · {r.word_count} words</span></p>', unsafe_allow_html=True)
if len(drops):
    with st.expander("Preview dropped blocks"):
        st.dataframe(drops[["reason", "page", "text"]].head(40), hide_index=True, use_container_width=True)

# ---------- 3. classify ----------
ui.section("Classify", f"Predict a theme for each of the {len(passages):,} passages. Passages with confidence below "
                       f"{meta['threshold']:.2f} are flagged for human review.")
sig = (file_key, company, year, tuple((s.section, s.start_page, s.end_page) for s in sections))
if st.button("Classify passages", type="primary"):
    with st.spinner(f"Classifying {len(passages):,} passages…"):
        pf = mdl.predict_frame(model, passages["text"], meta["threshold"])
        out = pd.concat([passages.reset_index(drop=True), pf], axis=1)
        out["source"] = f"Uploaded: {up.name}"
        st.session_state["an_preds"] = out
        st.session_state["an_sig"] = sig
    st.rerun()

out = st.session_state.get("an_preds")
if out is None:
    st.stop()
if st.session_state.get("an_sig") != sig:
    ui.note("The company, year or page ranges changed since classification. Click <b>Classify passages</b> again "
            "to refresh the results.", warn=True)

# ---------- 4. review & export ----------
ui.section("Review and export", "Predictions for this report. Filter to the flagged passages to review them first.")
strategic = out[out.pred != "OTHER"]
dist = strategic["pred"].value_counts(normalize=True)
ui.kpi_strip([
    (f"{len(out):,}", "Passages classified"),
    (f"{len(strategic):,}", "Strategic (not Other)"),
    (f"{int(out.needs_review.sum()):,}", "Flagged for review"),
    (f"{THEME_NAMES[dist.index[0]].split(' ')[0] if len(dist) else '–'}", "Most emphasised theme"),
])
f1, f2, f3, f4 = st.columns([1.6, 1, 1, 0.8])
with f1:
    theme_filter = st.multiselect("Themes", THEMES, default=[], format_func=lambda t: THEME_NAMES[t],
                                  placeholder="All themes")
with f2:
    only_flagged = st.toggle("Only flagged passages", value=False)
with f3:
    sort = st.selectbox("Sort by", ["Lowest confidence first", "Highest confidence first", "Page order"])
with f4:
    n_rows = st.selectbox("Rows", [15, 40, 100], index=0)
view = out
if theme_filter:
    view = view[view.pred.isin(theme_filter)]
if only_flagged:
    view = view[view.needs_review]
view = {"Lowest confidence first": view.sort_values("confidence"),
        "Highest confidence first": view.sort_values("confidence", ascending=False),
        "Page order": view.sort_values(["page", "passage_id"])}[sort]
if view.empty:
    ui.empty_state("No passages match these filters.", "Clear a filter to see more.")
else:
    ui.predictions_table(view, max_rows=n_rows)

with st.expander("Why this label? Explain one passage"):
    pick = st.selectbox("Passage", view.passage_id.tolist() if not view.empty else out.passage_id.tolist(),
                        format_func=lambda pid: out.set_index("passage_id").loc[pid, "text"][:110] + "…")
    ui.why_label(out.set_index("passage_id").loc[pick, "text"])

b1, b2, _ = st.columns([1.3, 1.5, 2])
export_cols = ["passage_id", "company", "year", "section", "page", "word_count", "pred", "confidence",
               "needs_review"] + [f"p_{c}" for c in model.classes_] + ["text"]
with b1:
    st.download_button("Export predictions (CSV)", out[export_cols].to_csv(index=False).encode("utf-8"),
                       file_name=f"stratlens_{company}_{year}.csv", mime="text/csv")
with b2:
    session = st.session_state.setdefault("session_reports", {})
    key = f"{company}_{year}"
    label = "Update in benchmark session" if key in session else "Add to benchmark session"
    if st.button(label):
        session[key] = out.assign(in_training_set=False)
        st.toast(f"{COMPANY_NAMES.get(company, company)} {year} added. Open Benchmark to compare.")
if st.session_state.get("session_reports"):
    names = ", ".join(st.session_state["session_reports"].keys())
    st.markdown(f'<p class="sl-meta">In this session’s benchmark: {names}. Session uploads are kept until the '
                'browser tab is closed and replace the sample report with the same company and year.</p>',
                unsafe_allow_html=True)
