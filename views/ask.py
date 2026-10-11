"""Ask: find the passages that answer a question, then (if available) a short answer that cites them."""
import html

import streamlit as st

from stratlens import ui
from stratlens.config import ASK_EXAMPLES
from stratlens.llm import answer as ask
from stratlens.llm import retrieval
from stratlens.llm.client import get_client

passages = ui.load_passages()
ui.header("Ask", "Ask anything about the reports. Answers use only the passages listed under them.")
if passages is None:
    ui.empty_state("No reports have been processed yet.", "Run <code>python -m stratlens.build</code>.")
    st.stop()

examples = ui.load_ask_examples()


def _set(q):
    st.session_state["ask_q"] = q


st.markdown('<p class="sl-meta">Try one:</p>', unsafe_allow_html=True)
cols = st.columns(len(ASK_EXAMPLES))
for col, q in zip(cols, ASK_EXAMPLES):
    with col:
        st.button(q, on_click=_set, args=(q,), use_container_width=True, key=f"ex_{q}")

with st.form("ask_form", border=False):
    c0, c1 = st.columns([6, 1], vertical_alignment="bottom")
    with c0:
        st.text_input("Question", placeholder="e.g. What is HUL doing about rural distribution?", key="ask_q")
    with c1:
        st.form_submit_button("Ask", type="primary", use_container_width=True)
q = st.session_state.get("ask_q", "").strip()
if not q:
    ui.empty_state("Pick an example above or type a question.")
    st.stop()


@st.cache_data(show_spinner=False, max_entries=64)
def _live(question: str):
    hits, method = retrieval.search(passages, question, k=8)
    srcs = ask.sources(hits)
    return srcs, method, ask.answer(question, srcs, get_client())


@st.cache_data(show_spinner=False, max_entries=16)
def _answer(question: str, ids: tuple):
    hits = passages.set_index("passage_id").loc[list(ids)].reset_index()
    return ask.answer(question, ask.sources(hits), get_client())


cached = examples.get(q)
if cached and cached["passage_ids"] and set(cached["passage_ids"]) <= set(passages.passage_id):
    # example question: retrieval was done offline; the answer too, unless the AI service failed at build time
    hits = passages.set_index("passage_id").loc[cached["passage_ids"]].reset_index()
    srcs, method = ask.sources(hits), cached["method"]
    if cached.get("status") in ("ok", "not_in_passages"):
        ans = cached
    else:
        with st.spinner("Writing the answer…"):
            ans = _answer(q, tuple(s["passage_id"] for s in srcs))
else:
    with st.spinner("Searching the reports…"):
        srcs, method, ans = _live(q)

if not srcs:
    ui.note("Nothing in the reports is relevant enough to answer this. Try other wording.", warn=True)
    st.stop()

with ui.card("ask_ans", "Answer", meta="AI-written from the passages below" if ans["status"] == "ok" else ""):
    if ans["status"] == "ok":
        body = ask.CITE_RE.sub(lambda m: "".join(
            f'<a class="sl-cite" href="#src-{n.strip()}">[{n.strip()}]</a>'
            for n in m.group(1).replace("–", ",").replace("-", ",").split(",") if n.strip()), html.escape(ans["text"]))
        st.markdown(f'<div class="sl-answer">{body}</div>', unsafe_allow_html=True)
        if ans.get("unsupported"):
            ui.note("Numbers not found in the passages: " + ", ".join(html.escape(n) for n in ans["unsupported"]),
                    warn=True)
    elif ans["status"] == "not_in_passages":
        ui.note("The passages found don't answer this directly. Read them below.", warn=True)
    else:
        reason = ("AI answers are switched off (no API key)." if ans["status"] == "no_ai"
                  else "The AI service is unavailable right now.")
        ui.note(f"{reason} Here are the most relevant passages instead.")

named = retrieval.companies_in_query(q)
scope = f"only {' and '.join(ui.short(c) for c in named)}, named in the question" if named else ""
with ui.card("ask_src", "Passages", meta=f"{len(srcs)} found" + (f" · {scope}" if scope else "")):
    rows = passages.set_index("passage_id").loc[[s["passage_id"] for s in srcs]].reset_index()
    for i, (_, row) in enumerate(rows.iterrows(), start=1):
        ui.passage(row, anchor=f"src-{i}", label=f"[{i}]", show_theme=True)
    ui.meta("Found by matching words and meaning (keyword search plus sentence embeddings). "
            "Every passage is searched, strategy or not.")
