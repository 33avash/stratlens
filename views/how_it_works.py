"""How it works: the NLP pipeline as a diagram, one line per stage, and an inspector that runs text through it."""
import html

import pandas as pd
import streamlit as st

from stratlens import ui
from stratlens.config import TEST_CSV

m = ui.load_metrics()
thr = m["threshold"]["value"] if m else 0.45
ui.header("How it works", "From PDF to theme shares: what each stage does and where you see its output.")

# ---------- diagram ----------
W, H, GAP = 128, 54, 150


def node(col, y, title, sub, kind=""):
    x = 4 + col * GAP
    return (f'<g class="node {kind}"><rect x="{x}" y="{y}" width="{W}" height="{H}" rx="9"/>'
            f'<text x="{x + 12}" y="{y + 23}">{html.escape(title)}</text>'
            f'<text class="s" x="{x + 12}" y="{y + 41}">{html.escape(sub)}</text></g>')


main = [("Extract", "PyMuPDF"), ("Clean", "rules + regex"), ("Segment", "sentence split"),
        ("Mask", "gazetteer + regex"), ("TF-IDF", "1–2 grams"), ("Logistic reg.", "7 themes"),
        ("Confidence", f"flag < {thr:.2f}"), ("Aggregate", "theme shares")]
ya, yb, yc = 34, 150, 250
parts = [node(i, ya, t, s, "core") for i, (t, s) in enumerate(main)]
parts += [f'<path class="edge" d="M{4 + i * GAP + W},{ya + H / 2} H{4 + (i + 1) * GAP - 3}" marker-end="url(#ar)"/>'
          for i in range(len(main) - 1)]
parts += [node(3, yb, "spaCy NER", "+ regex signals"), node(4, yb, "Specificity", "0–5 score"),
          node(3, yc, "Embeddings", "MiniLM + BM25"), node(4, yc, "Retrieval", "rank fusion"),
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

# ---------- stages ----------
STAGES = [
    ("Extract", "Pulls text blocks with page numbers; keeps only the Chairman/CEO letter and MD&A pages.",
     "PyMuPDF; heading detection plus hand-confirmed page ranges", "Upload → Pages; page numbers on every passage"),
    ("Clean", "Removes running headers, page numbers, tables, captions and statutory boilerplate; fixes "
              "hyphenation.", "rules and regular expressions; every drop is logged",
     "Upload → blocks dropped; Model card → data"),
    ("Segment", "Splits sentences (abbreviation-aware) and packs them into 8–70 word passages; drops duplicates.",
     "regex sentence splitter", "every passage shown in Evidence, Briefs and Ask"),
    ("Mask", "Replaces company names, brand names, numbers and years with [COMPANY] [BRAND] [NUM] [YEAR].",
     "curated name lists + regex (a gazetteer, not statistical NER)", "Evaluation → Robustness: masking test"),
    ("TF-IDF", "Turns masked text into weighted word and two-word features.",
     "scikit-learn TfidfVectorizer, sublinear tf, min_df 2", "Evaluation → Explainability: top terms"),
    ("Logistic regression", "Assigns one of six strategy themes or Other.",
     "balanced class weights; C chosen by 5-fold CV", "Overview fingerprint; Evaluation → Performance"),
    ("Confidence", f"Top-class probability; passages below {thr:.2f} are flagged as low confidence.",
     "threshold checked on cross-validated predictions", "Overview → Low confidence and the filter"),
    ("Aggregate", "Theme passages ÷ strategic (non-Other) passages per report; YoY shifts with bootstrap CIs.",
     "counts + 1,000-resample bootstrap", "Overview fingerprint and YoY; Briefs"),
    ("Specificity", "Scores 0–5 for quantities, dates, named initiatives and commitments on the unmasked text.",
     "spaCy NER (en_core_web_sm) + regex", "Overview scatter; signals on each evidence passage; Briefs"),
    ("Retrieval", "Finds passages for a question, strategic or Other.",
     "BM25 + MiniLM embeddings, reciprocal rank fusion, relevance gate", "Ask → Sources"),
    ("Gemini", "Writes prose from computed figures and answers from retrieved passages.",
     "LLM, temperature 0.2, numbers checked against the data", "Briefs text; Ask answer"),
]
with ui.card("stages", "Stages"):
    rows = "".join(f"<tr><td><b>{html.escape(s)}</b></td><td>{html.escape(w)}</td><td class='sl-small'>"
                   f"{html.escape(t)}</td><td class='sl-small'>{html.escape(a)}</td></tr>" for s, w, t, a in STAGES)
    st.markdown("<table class='sl-table'><thead><tr><th>Stage</th><th>What it does</th><th>Technique</th>"
                f"<th>Where you see it</th></tr></thead><tbody>{rows}</tbody></table>", unsafe_allow_html=True)

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
