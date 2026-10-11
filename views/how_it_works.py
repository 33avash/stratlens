"""How it works: the NLP pipeline as a diagram, one line per stage, and an inspector that runs text through it."""
import html

import pandas as pd
import streamlit as st

from stratlens import ui
from stratlens.config import TEST_CSV

m = ui.load_metrics()
thr = m["threshold"]["value"] if m else 0.45
n_train, n_test = (m["dataset"]["n_train"], m["dataset"]["n_test"]) if m else ("the", "held-out")
ui.header("How it works", "From a 300-page PDF to comparable theme shares, one NLP step at a time.")

# ---------- diagram ----------
W, H, GAP = 128, 54, 150


def node(col, y, title, sub, kind=""):
    x = 4 + col * GAP
    return (f'<g class="node {kind}"><rect x="{x}" y="{y}" width="{W}" height="{H}" rx="9"/>'
            f'<text x="{x + 12}" y="{y + 23}">{html.escape(title)}</text>'
            f'<text class="s" x="{x + 12}" y="{y + 41}">{html.escape(sub)}</text></g>')


main = [("1 Read", "PDF text"), ("2 Clean", "regex"), ("3 Split", "sentences"),
        ("4 Hide names", "masking"), ("5 TF-IDF", "word weights"), ("6 Classify", "7 themes"),
        ("Confidence", f"flag < {thr:.2f}"), ("Theme shares", "per report")]
ya, yb, yc = 34, 150, 250
parts = [node(i, ya, t, s, "core") for i, (t, s) in enumerate(main)]
parts += [f'<path class="edge" d="M{4 + i * GAP + W},{ya + H / 2} H{4 + (i + 1) * GAP - 3}" marker-end="url(#ar)"/>'
          for i in range(len(main) - 1)]
parts += [node(3, yb, "7 spaCy NER", "+ regex signals"), node(4, yb, "Specificity", "0–5 score"),
          node(3, yc, "8 Retrieve", "BM25 + embeddings"), node(4, yc, "Retrieval", "rank fusion"),
          node(5, yc, "Gemini · Ask", "cited answer", "llm"), node(7, yb, "Gemini · Briefs", "prose only", "llm")]
sx = 4 + 2 * GAP + W / 2  # branch point below Segment: the branches read unmasked text
parts += [f'<path class="edge" d="M{sx},{ya + H} V{yc + H / 2} H{4 + 3 * GAP - 3}" marker-end="url(#ar)"/>',
          f'<path class="edge" d="M{sx},{yb + H / 2} H{4 + 3 * GAP - 3}" marker-end="url(#ar)"/>',
          f'<path class="edge" d="M{4 + 3 * GAP + W},{yb + H / 2} H{4 + 4 * GAP - 3}" marker-end="url(#ar)"/>',
          f'<path class="edge" d="M{4 + 4 * GAP + W},{yb + H / 2} H{4 + 7 * GAP - 3}" marker-end="url(#ar)"/>',
          f'<path class="edge" d="M{4 + 3 * GAP + W},{yc + H / 2} H{4 + 4 * GAP - 3}" marker-end="url(#ar)"/>',
          f'<path class="edge" d="M{4 + 4 * GAP + W},{yc + H / 2} H{4 + 5 * GAP - 3}" marker-end="url(#ar)"/>',
          f'<path class="edge" d="M{4 + 7 * GAP + W / 2},{ya + H} V{yb - 3}" marker-end="url(#ar)"/>',
          f'<text class="lane" x="4" y="{ya - 12}">CLASSIFIER</text>',
          f'<text class="lane" x="{4 + 3 * GAP}" y="{yb - 10}">SPECIFICITY (UNMASKED TEXT)</text>',
          f'<text class="lane" x="{4 + 3 * GAP}" y="{yc - 10}">ASK: RETRIEVAL + LLM</text>',
          f'<text class="lane" x="{4 + 7 * GAP}" y="{yb - 10}">BRIEFS</text>']
svg = (f'<svg class="sl-flow" viewBox="0 0 {4 + 7 * GAP + W + 4} {yc + H + 6}" role="img" '
       'aria-label="Pipeline: Extract, Clean, Segment, Mask, TF-IDF, Logistic regression, Confidence, Aggregate shares; '
       'branches from Segment to spaCy NER and Specificity, and to Embeddings, Retrieval and Gemini for Ask; '
       'Aggregate shares and Specificity feed Gemini for Briefs.">'
       '<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
       '<path d="M0,0 L10,5 L0,10 z"/></marker></defs>' + "".join(parts) + "</svg>")
with ui.card("flow", "Pipeline", meta="solid = trained or rule-based NLP · dashed = LLM"):
    st.markdown(svg, unsafe_allow_html=True)

# ---------- NLP at each step ----------
# (step, NLP technique, syllabus unit, what it does in plain words, where you see it)
STEPS = [
    ("1 · Read", "Text extraction", "Unit 1 · NLP pipeline",
     "Pulls text with page numbers from the PDF and keeps only the Chairman's letter and MD&A.", "Upload"),
    ("2 · Clean", "Text preprocessing with regular expressions", "Unit 2 · regex",
     "Strips headers, page numbers, tables and legal boilerplate; repairs broken words.", "Upload → dropped blocks"),
    ("3 · Split", "Sentence segmentation and tokenisation", "Unit 1–2",
     "Breaks text into sentences (knows 'Rs.' and 'Mr.' aren't sentence ends), then 8–70 word passages.",
     "every evidence passage"),
    ("4 · Hide names", "Entity masking (gazetteer + regex)", "Unit 2 · NER · Unit 5 · bias",
     "Swaps company names, brands, numbers and years for [COMPANY] [BRAND] [NUM] [YEAR], so the model learns "
     "topics, not who wrote it.", "inspector step 2"),
    ("5 · Words → numbers", "Bag-of-words → TF-IDF with 1–2 word n-grams", "Unit 3 · feature representation",
     "Scores each word and word pair by how distinctive it is for the passage, e.g. 'quick commerce'.",
     "inspector step 3"),
    ("6 · Classify", "Supervised text classification (logistic regression)", "Unit 4 · ML techniques",
     f"Learns from {n_train} labelled passages to assign one of six strategy themes or Other, with a confidence.",
     "Overview, Evaluation"),
    ("   compared with", "Keyword lexicon · MiniLM sentence embeddings", "Unit 4 · lexicons · Unit 3 · embeddings",
     "A simple keyword list and a meaning-based embedding model were tested on the same passages.",
     "Evaluation → Models"),
    ("7 · How concrete", "Named Entity Recognition (spaCy) + regex", "Unit 2 · NER",
     "Scores 0–5 per passage for figures, dates, named initiatives and commitments.", "Overview scatter"),
    ("8 · Ask and briefs", "Retrieval (BM25 + embeddings) and an LLM", "Unit 5 · LLMs",
     "Finds the passages for a question; Gemini writes answers and briefs that cite them. It never labels or "
     "computes numbers.", "Ask, Briefs"),
    ("9 · Check", "Evaluation: precision, recall, F1, subgroups, explainability", "Unit 5 · evaluation, bias",
     f"Scores the model on {n_test} unseen passages, by company and writing style, and shows which words drove "
     "each label.", "Evaluation, Model card"),
]
with ui.card("stages", "The NLP at each step", meta="syllabus unit in grey"):
    rows = "".join(f"<tr><td><b>{html.escape(s)}</b></td><td class='nlp'>{html.escape(t)}<br>"
                   f"<span class='unit'>{html.escape(u)}</span></td><td>{html.escape(w)}</td>"
                   f"<td class='sl-small'>{html.escape(a)}</td></tr>" for s, t, u, w, a in STEPS)
    st.markdown("<table class='sl-table'><thead><tr><th>Step</th><th>NLP technique</th><th>What it does</th>"
                f"<th>See it in</th></tr></thead><tbody>{rows}</tbody></table>", unsafe_allow_html=True)

# ---------- inspector ----------
EXAMPLES = [
    "We aim to double our direct reach to 3 million outlets by FY27, with a sharper focus on rural markets.",
    "Dabur launched a new Vatika shampoo range and extended its juice portfolio into new fruit variants.",
    "Quick commerce is now our fastest growing channel, contributing a rising share of e-commerce sales.",
    "We will reduce virgin plastic by 50% by 2030 through Project Circular Bharat across 10 cities.",
    "The Board met six times during the year under review.",
]
with ui.card("inspect", "Pipeline inspector", meta="runs the saved model, step by step"):
    mode = st.segmented_control("Input", ["Example", "Held-out passage", "Your text"], default="Example",
                                label_visibility="collapsed", key="insp_mode") or "Example"
    text = ""
    if mode == "Example":
        text = st.selectbox("Example", EXAMPLES, label_visibility="collapsed", key="insp_ex")
        st.markdown('<p class="sl-meta">Illustrative sentences, not quotes from the reports.</p>',
                    unsafe_allow_html=True)
    elif mode == "Held-out passage" and TEST_CSV.exists():
        test = pd.read_csv(TEST_CSV)
        i = st.selectbox("Passage", test.index, label_visibility="collapsed", key="insp_test",
                         format_func=lambda k: f"[{test.loc[k, 'label']}] {test.loc[k, 'text'][:110]}…")
        text = test.loc[i, "text"]
        st.markdown(f'<p class="sl-meta">AI-drafted label: {ui.theme_label(test.loc[i, "label"])}</p>',
                    unsafe_allow_html=True)
    else:
        text = st.text_area("Text", height=100, label_visibility="collapsed", key="insp_txt",
                            placeholder="Paste a sentence or paragraph from an annual report.")
    if len(text.split()) >= 3:
        ui.pipeline_inspector(text.strip())
    elif mode == "Your text":
        ui.empty_state("Paste at least a sentence.")

# ---------- LLM ----------
with ui.card("llm", "What the LLM does and doesn't do"):
    st.markdown(
        "<table class='sl-table'><tbody>"
        "<tr><td class='sl-small'>Does</td><td>Writes brief summaries from the classifier's computed shares, shifts "
        "and specificity, and answers Ask questions from retrieved passages, citing each one.</td></tr>"
        "<tr><td class='sl-small'>Doesn't</td><td>Classify passages, compute any number, or choose which passages "
        "count. Every label comes from the logistic regression above.</td></tr>"
        "<tr><td class='sl-small'>Checks</td><td>Numbers in brief text must match the computed figures, otherwise "
        "that section falls back to template text. Ask refuses when retrieval finds nothing relevant or the "
        "passages don't answer the question.</td></tr></tbody></table>", unsafe_allow_html=True)
