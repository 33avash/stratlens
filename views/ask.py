"""Ask: retrieve relevant passages, then (if AI writing is configured) answer from them with citations."""
import html
import re

import streamlit as st

from stratlens import ui
from stratlens.config import SECTION_NAMES
from stratlens.llm import client as llm
from stratlens.llm import prompts, retrieval
from stratlens.report import clean

preds = ui.load_benchmark()
ui.header("Ask", "Answers come only from the passages listed under them. Every passage is searched, "
                 "strategy and Other alike.")
if preds is None:
    ui.empty_state("No analysis yet.", "Run <code>python train.py</code>.")
    st.stop()
client = llm.get_client()

with st.form("ask_form", border=False):
    c0, c1, c2, c3 = st.columns([4.2, 1.3, 1, 0.7], vertical_alignment="bottom")
    with c0:
        q = st.text_input("Question", placeholder="How is Dabur talking about quick commerce?", key="ask_q")
    with c1:
        company = st.selectbox("Company", ["All"] + sorted(preds.company.unique()), index=0,
                               format_func=lambda c: "All companies" if c == "All" else ui.company_name(c))
    with c2:
        year = st.selectbox("Year", ["All"] + sorted(preds.year.unique()), index=0,
                            format_func=lambda y: "All years" if y == "All" else y)
    with c3:
        st.form_submit_button("Ask", type="primary", use_container_width=True)
if not q.strip():
    st.stop()


@st.cache_data(show_spinner=False, max_entries=64)
def _search(q, company, year):
    return retrieval.search(preds, q, k=8, company=None if company == "All" else company,
                            year=None if year == "All" else year)


with st.spinner("Searching…"):
    hits, method = _search(q, company, year)
if hits.empty:
    ui.note("Nothing relevant enough to answer. Try other wording or widen the filters.", warn=True)
    st.stop()

passages = [{"company_name": ui.company_name(r.company), "year": r.year,
             "section_name": SECTION_NAMES.get(r.section, r.section), "page": int(r.page), "text": clean(r.text)}
            for r in hits.itertuples()]

if client:
    with ui.card("ask_ans", "Answer", meta="AI-generated from cited passages"):
        try:
            with st.spinner("Writing…"):
                ans = client.generate(prompts.ASK_SYSTEM, prompts.ask_prompt(q, passages), max_tokens=700)
            if "NOT_IN_PASSAGES" in ans:
                ui.note("The passages don't answer this question.", warn=True)
            else:
                cite_re = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]")  # [2], [1, 3] or [6-8]
                source_text = " ".join(p["text"] for p in passages)
                nums = [n for n in re.findall(r"\d+(?:\.\d+)?%?", cite_re.sub("", ans))
                        if n.rstrip("%") not in source_text]
                st.markdown(f'<div class="sl-answer">{cite_re.sub(r"<sup>[\1]</sup>", html.escape(ans))}</div>',
                            unsafe_allow_html=True)
                if nums:
                    ui.note("Not found in the sources: " + ", ".join(html.escape(n) for n in nums), warn=True)
        except llm.LLMError as exc:
            ui.note(html.escape(str(exc)), warn=True)

scope = f"{ui.company_name(retrieval.company_in_query(q))} only · " if "limited to" in method else ""
with ui.card("ask_src", "Sources", meta=f"{scope}{len(hits)} results · {method.split(';')[0]}"):
    for i, (p, r) in enumerate(zip(passages, hits.itertuples()), start=1):
        st.markdown(f'<div class="sl-quote"><div class="q"><b class="mono">[{i}]</b> {html.escape(p["text"])}</div>'
                    f'<div class="m">{html.escape(p["company_name"])} {p["year"]} · {html.escape(p["section_name"])} · '
                    f'p.{p["page"]} · {html.escape(r.pred)} · '
                    + ui.tip(f"relevance {r.relevance:.2f}", "Reciprocal rank fusion of BM25 and embedding ranks, "
                             "scaled so 1.00 means ranked first by both.")
                    + (f' · BM25 #{r.bm25_rank} · semantic #{r.emb_rank}' if r.emb_rank else f' · BM25 #{r.bm25_rank}')
                    + '</div></div>', unsafe_allow_html=True)
