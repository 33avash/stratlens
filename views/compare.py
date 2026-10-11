"""Compare: any two reports (a company across years, or two companies) with a check for chance."""
import html

import streamlit as st

from stratlens import analytics, charts, feature_views, features, insights, ui
from stratlens.config import COMPARE_PRESETS, STRATEGIC_THEMES, THEME_NAMES
from stratlens.llm import narrate
from stratlens.llm.client import get_client
from stratlens.theme import PLOTLY_CONFIG

an, passages = ui.load_analytics(), ui.load_passages()
ui.header("Compare", "Two reports side by side: which differences are real and which are just noise.")
if an is None or passages is None:
    ui.empty_state("No reports have been processed yet.", "Run <code>python -m stratlens.build</code>.")
    st.stop()

keys = sorted(an["reports"], key=lambda k: (k.split("|")[0], k.split("|")[1]))
if len(keys) < 2:
    ui.empty_state("Comparing needs at least two reports.")
    st.stop()

qp = st.query_params
presets = [p for p in COMPARE_PRESETS if p["a"] in an["reports"] and p["b"] in an["reports"]]
default_a, default_b = (presets[0]["a"], presets[0]["b"]) if presets else (keys[0], keys[1])
st.session_state.setdefault("cmp_a", qp.get("a") if qp.get("a") in keys else default_a)
st.session_state.setdefault("cmp_b", qp.get("b") if qp.get("b") in keys else default_b)


def _preset(a, b):
    st.session_state["cmp_a"], st.session_state["cmp_b"] = a, b


if presets:
    cols = st.columns(len(presets) + 1)
    for col, p in zip(cols, presets):
        with col:
            st.button(p["label"], on_click=_preset, args=(p["a"], p["b"]), use_container_width=True,
                      type="primary" if (st.session_state["cmp_a"], st.session_state["cmp_b"]) == (p["a"], p["b"])
                      else "secondary")
c1, c2 = st.columns(2)
with c1:
    a = st.selectbox("Report A", keys, format_func=ui.report_label, key="cmp_a")
with c2:
    b = st.selectbox("Report B", keys, format_func=ui.report_label, key="cmp_b")
if (qp.get("a"), qp.get("b")) != (a, b):
    st.query_params.update({"a": a, "b": b})
if a == b:
    ui.empty_state("Pick two different reports.")
    st.stop()

la, lb = ui.report_label(a), ui.report_label(b)
diffs = analytics.pair(an, a, b)
ra, rb = an["reports"][a], an["reports"][b]
sig = sorted(((abs(d["pp"]), t) for t, d in diffs.items() if d["significant"]), reverse=True)

# ---------- summary ----------
nar = ui.load_narratives()["compare"].get(f"{a}~{b}")
if nar is None:
    tpl, tok = insights.compare_template(an, a, b, diffs)
    text, src = narrate.narrate(tpl, tok, get_client(), cache_only=True)
    nar = {"text": text, "source": src}
title = (f"Biggest real difference: {THEME_NAMES[sig[0][1]].lower()}" if sig
         else "No difference bigger than chance")
with ui.card("cmp_sum", title, meta=f"{la} → {lb}"):
    st.markdown(f'<p class="sl-para">{html.escape(nar["text"])}</p>', unsafe_allow_html=True)
    ui.meta("Wording polished by AI; every number is inserted from the data." if nar["source"] != "template"
            else "Written from the data.")

# ---------- diverging chart ----------
left, right = st.columns([7, 5], gap="medium")
with left:
    with ui.card("cmp_div", f"How {lb} differs from {la}",
                 meta="share of strategy passages, B minus A · click a bar for passages"):
        ev = st.plotly_chart(charts.diverging(diffs, la, lb), use_container_width=True, config=PLOTLY_CONFIG,
                             on_select="rerun", selection_mode="points", key=f"div_{a}_{b}")
        ui.meta("Coloured bars are bigger than chance; grey bars are within noise. The whisker is the likely "
                "range: if the report had happened to contain other passages on the same topics, the difference "
                "would fall inside it 95 times out of 100.")
with right:
    with ui.card("cmp_conc", "How specific each report is, by theme",
                 meta="average concreteness, 0–5"):
        va = {t: ra["themes"][t]["concreteness"] for t in STRATEGIC_THEMES}
        vb = {t: rb["themes"][t]["concreteness"] for t in STRATEGIC_THEMES}
        st.plotly_chart(charts.paired(va, vb, la, lb, rng=(0, 5)), use_container_width=True, config=PLOTLY_CONFIG)
        ui.meta("Concreteness counts figures, dates, named initiatives and commitments in a passage (0–5). "
                "A blank bar means the report has no passages on that theme.")

# ---------- only in A / only in B ----------
@st.cache_data(show_spinner=False, max_entries=32)
def _novelty(a_: str, b_: str) -> dict:
    return features.novelty(passages, ui.load_embeddings(), a_, b_)


nov = ui.load_features().get("novelty", {}).get(f"{a}~{b}")
if nov is None and ui.load_embeddings():
    with st.spinner("Matching passages by meaning…"):
        nov = _novelty(a, b)
if nov:
    with ui.card("cmp_only", f"Topics only in one report: {nov['only_in_a']['n']} only in {la}, "
                             f"{nov['only_in_b']['n']} only in {lb}",
                 meta="strategy passages with no close counterpart in the other report"):
        feature_views.novelty_columns(passages, nov, ("only_in_a", f"Only in {la}"), ("only_in_b", f"Only in {lb}"),
                                      "cmp")

# ---------- passages for a clicked theme ----------
pts = (ev.selection.points if ev and ev.selection else []) or []
pick = pts[0].get("customdata", [None])[0] if pts else None
if pick is None and sig:
    pick = sig[0][1]
if pick:
    d = diffs[pick]
    with ui.card("cmp_ev", f"Passages on {THEME_NAMES[pick].lower()}",
                 meta=f"{ra['themes'][pick]['share'] * 100:.0f}% in {la} · "
                      f"{rb['themes'][pick]['share'] * 100:.0f}% in {lb}"):
        p1, p2 = st.columns(2, gap="medium")
        for col, key, lab in [(p1, a, la), (p2, b, lb)]:
            with col:
                ui.subhead(lab)
                co, yr = key.split("|")
                df = ui.evidence_rows(passages, co, yr, pick).head(3)
                if df.empty:
                    ui.empty_state("No passages on this theme with enough model confidence.")
                for _, row in df.iterrows():
                    ui.passage(row)
                st.markdown(ui.link("company?" + ui.qs(company=co, year=yr, theme=pick), "All passages →"),
                            unsafe_allow_html=True)
