"""How it works: the pipeline (with the step-by-step inspector), how accurate it is, and its limits.

The only page that uses NLP terms; each one carries a hover definition."""
import html

import pandas as pd
import streamlit as st

from stratlens import audit, charts, ui
from stratlens.config import MODEL_CARD, TEST_CSV, THEME_NAMES
from stratlens.theme import PLOTLY_CONFIG

m, au, an = ui.load_metrics(), ui.load_audit(), ui.load_analytics()
ui.header("How it works", "From a 300-page PDF to comparable theme shares, one NLP step at a time.")

T = {  # hover definitions for the jargon used on this page
    "macro-F1": "The F1 score (balance of precision and recall) computed for each of the 7 themes, then averaged, "
                "so rare themes count as much as common ones. 1.0 is perfect.",
    "95% CI": "95% confidence interval: the range the score would fall in 95 times out of 100 if the test passages "
              "were redrawn (bootstrap resampling).",
    "pp": "Percentage points: the difference between two percentages (14% vs 6% is +8 pp).",
    "TF-IDF": "Term frequency × inverse document frequency: a word's weight is high when it is frequent in this "
              "passage but rare across passages.",
    "confidence": "The model's probability for its chosen theme (0–1). Passages below 0.45 are flagged and never "
                  "shown as evidence.",
    "kappa": "Cohen's kappa: agreement between two labellers after removing the agreement expected by chance. "
             "Above 0.6 is usually read as substantial, above 0.8 as almost perfect.",
}
tip = lambda label, key=None: ui.tip(label, T[key or label])  # noqa: E731

tab1, tab2, tab3 = st.tabs(["Pipeline", "Accuracy", "Limits"])

# ======================================================================== pipeline
with tab1:
    ui.pipeline_strip(href=None)
    ui.note("<b>Gemini only writes sentences and answers from retrieved passages. It never labels passages or "
            "computes numbers.</b> Every theme comes from the trained classifier below, and every number on screen "
            "is computed from its output.")

    EXAMPLES = [
        "We aim to double our direct reach to 3 million outlets by FY27, with a sharper focus on rural markets.",
        "Dabur launched a new Vatika shampoo range and extended its juice portfolio into new fruit variants.",
        "Quick commerce is now our fastest growing channel, contributing a rising share of e-commerce sales.",
        "We will reduce virgin plastic by 50% by 2030 through Project Circular Bharat across 10 cities.",
        "The Board met six times during the year under review.",
    ]
    with ui.card("inspect", "Pipeline inspector", meta="runs the saved model on any sentence, step by step"):
        mode = st.segmented_control("Input", ["Example", "Held-out passage", "Your text"], default="Example",
                                    label_visibility="collapsed", key="insp_mode") or "Example"
        text = ""
        if mode == "Example":
            text = st.selectbox("Example", EXAMPLES, label_visibility="collapsed", key="insp_ex")
            ui.meta("Illustrative sentences, not quotes from the reports.")
        elif mode == "Held-out passage" and TEST_CSV.exists():
            test = pd.read_csv(TEST_CSV)
            i = st.selectbox("Passage", test.index, label_visibility="collapsed", key="insp_test",
                             format_func=lambda k: f"[{THEME_NAMES[test.loc[k, 'label']]}] {test.loc[k, 'text'][:100]}…")
            text = test.loc[i, "text"]
            ui.meta(f"AI-drafted label: {ui.theme_label(test.loc[i, 'label'])}")
        else:
            text = st.text_area("Text", height=100, label_visibility="collapsed", key="insp_txt",
                                placeholder="Paste a sentence or paragraph from an annual report.")
        if len(text.split()) >= 3:
            ui.pipeline_inspector(text.strip())
        elif mode == "Your text":
            ui.empty_state("Paste at least a sentence.")

    n_train, n_test = (m["dataset"]["n_train"], m["dataset"]["n_test"]) if m else ("the", "held-out")
    STEPS = [
        ("Read", "Text extraction (PyMuPDF), section detection", "Unit 1 · NLP pipeline",
         "Pulls text with page numbers out of the PDF and keeps only the Chairman/CEO letter and MD&A."),
        ("Clean", "Regular expressions", "Unit 2 · regex",
         "Strips running headers, page numbers, tables and legal boilerplate; repairs hyphenated words."),
        ("Split", "Sentence segmentation and tokenisation", "Unit 1–2",
         "Breaks text into sentences (knows 'Rs.' and 'Mr.' aren't sentence ends), then 8–70 word passages."),
        ("Classify theme", "Name masking → TF-IDF word and word-pair features → logistic regression",
         "Units 2–4 · NER, features, ML",
         f"Hides company and brand names, numbers and years, then a classifier trained on {n_train} labelled "
         "passages picks one of six strategy themes or Other, with a confidence."),
        ("", "Compared with: keyword lexicon · MiniLM sentence embeddings", "Unit 3–4 · lexicons, embeddings",
         "A keyword list and a meaning-based embedding model were scored on the same test passages."),
        ("Score concreteness", "Named entity recognition (spaCy) + regex", "Unit 2 · NER",
         "0–5 per passage: figures, a date or deadline, a named initiative, commitment language."),
        ("Compare", "Share of passages per theme + bootstrap resampling", "Unit 5 · evaluation",
         "Theme shares per report; a difference counts as real only if its 95% interval excludes zero."),
        ("Ask", "Retrieval (BM25 + MiniLM embeddings, rank fusion) + an LLM", "Unit 5 · LLMs",
         "Finds the passages for a question; Gemini writes a short answer that cites them."),
    ]
    with st.expander("Show the NLP behind each step"):
        rows = "".join(f"<tr><td><b>{html.escape(s)}</b></td><td class='nlp'>{html.escape(t)}<br>"
                       f"<span class='unit'>{html.escape(u)}</span></td><td>{html.escape(w)}</td></tr>"
                       for s, t, u, w in STEPS)
        st.markdown("<table class='sl-table'><thead><tr><th>Step</th><th>NLP technique</th><th>What it does</th>"
                    f"</tr></thead><tbody>{rows}</tbody></table>", unsafe_allow_html=True)
        ui.meta(f"Words are turned into numbers with {tip('TF-IDF')}. The LLM's wording is checked: numbers are "
                "inserted from the data after it writes, and Ask refuses when nothing relevant is found.")

# ======================================================================== accuracy
with tab2:
    if m is None:
        ui.empty_state("No evaluation results yet.", "Run <code>python train.py</code>.")
    else:
        t, b, ci = m["test"]["model"], m["test"]["baseline"], m["test"]["model"]["macro_f1_ci"]
        n_test = m["dataset"]["n_test"]
        ui.kpi_cards([
            {"label": "Theme accuracy (macro-F1)", "value": f"{t['macro_f1']:.2f}", "help": T["macro-F1"],
             "sub": f"95% CI {ci[0]:.2f}–{ci[1]:.2f}"},
            {"label": "Keyword matching", "value": f"{b['macro_f1']:.2f}", "sub": "the simple baseline it beats"},
            {"label": "Passages labelled correctly", "value": f"{t['accuracy']:.0%}",
             "sub": f"of {n_test} passages never seen in training"},
        ])
        ui.meta(f"Scored against AI-drafted labels on {n_test} held-out passages from Dabur, HUL and Nestlé "
                "(the companies in training). Your hand labels, below, check both the model and those drafts.")

        c1, c2 = st.columns([7, 5], gap="medium")
        with c1:
            with ui.card("pc", "Per theme", meta="F1 on held-out passages (support in brackets)"):
                st.plotly_chart(charts.per_theme_f1(m["test"]["per_class"], m["test"]["baseline_per_class"]),
                                use_container_width=True, config=PLOTLY_CONFIG)
        with c2:
            with ui.card("cmp", "Models compared", meta="same test passages"):
                rows = "".join(
                    f"<tr><td>{html.escape(ui.system_name(r['system']))}"
                    f"{' ' + ui.pill('used', 'good') if r['note'] == 'main model' else ''}</td>"
                    f"<td class='r'><b>{r['macro_f1']:.2f}</b></td>"
                    f"<td class='r sl-small'>{r['macro_f1_ci'][0]:.2f}–{r['macro_f1_ci'][1]:.2f}</td></tr>"
                    for r in m["comparison"])
                st.markdown(f"<table class='sl-table'><thead><tr><th>System</th><th class='r'>{tip('macro-F1')}</th>"
                            f"<th class='r'>{tip('95% CI')}</th></tr></thead><tbody>{rows}</tbody></table>",
                            unsafe_allow_html=True)
                ui.callout("Meaning-based embeddings tie with word weights, so the model whose every decision can "
                           "be traced to words is the one used.")
            with ui.card("cm", "Where it goes wrong", meta="rows: correct theme · columns: model's theme"):
                st.plotly_chart(charts.confusion(m["test"]["confusion"], 420), use_container_width=True,
                                config=PLOTLY_CONFIG)

        with ui.card("human", "Checked by hand: unseen companies and the AI-drafted labels",
                     meta="from data/labels/verify_*.csv"):
            if not au:
                ui.empty_state("Waiting for hand labels.",
                               "Fill <code>human_label</code> in <code>data/labels/verify_new_companies.csv</code> "
                               "and <code>verify_test_set.csv</code>, then run <code>python -m stratlens.audit</code>.")
            else:
                un, ts, k = au.get("unseen_companies"), au.get("test_set"), au.get("ai_vs_human")
                items = []
                if un:
                    items.append({"label": "Unseen companies (macro-F1)", "value": f"{un['macro_f1']:.2f}",
                                  "help": T["macro-F1"], "sub": f"{', '.join(ui.short(c) for c in un['companies'])}"
                                  f" · n = {un['n']}"})
                if ts:
                    items.append({"label": "Original test set (macro-F1)", "value": f"{ts['macro_f1']:.2f}",
                                  "help": T["macro-F1"], "sub": f"against your labels · n = {ts['n']}"})
                if k:
                    items.append({"label": "AI drafts vs you (kappa)", "value": f"{k['kappa']:.2f}", "help": T["kappa"],
                                  "sub": f"{k['agreement']:.0%} identical"})
                ui.kpi_cards(items)
                if au.get("partial"):
                    ui.note("Partial: not every row is labelled yet.", warn=True)
                rows = "".join(
                    f"<tr><td>{html.escape(p['name'])}{' ' + ui.pill('small n', 'warn') if p['too_small'] else ''}"
                    f"</td><td class='sl-small'>{p['source']}</td><td class='r'>{p['n']}</td>"
                    f"<td class='r'><b>{p['macro_f1']:.2f}</b></td><td class='r sl-small'>"
                    + (f"{p['macro_f1_ci'][0]:.2f}–{p['macro_f1_ci'][1]:.2f}" if p.get("macro_f1_ci") else "–")
                    + "</td></tr>" for p in au["per_company"])
                st.markdown(f"<table class='sl-table'><thead><tr><th>Company</th><th>Sample</th><th class='r'>n</th>"
                            f"<th class='r'>{tip('macro-F1')}</th><th class='r'>{tip('95% CI')}</th></tr></thead>"
                            f"<tbody>{rows}</tbody></table>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3, gap="medium")
        mk, p, cy = m["masking"], m["masking"]["company_probe"], m["cross_year"]
        with c1:
            with ui.card("mask", "Hiding names doesn't hurt"):
                ui.kpi_strip([(f"{mk['masked_macro_f1']:.2f}", "names hidden"), (f"{mk['unmasked_macro_f1']:.2f}",
                                                                                 "names shown")])
        with c2:
            with ui.card("probe", "Writing style still shows", meta="company guessed from masked text"):
                ui.kpi_strip([(f"{p['masked']:.0%}", "guessed"), (f"{p['majority_baseline']:.0%}", "by chance")])
        with c3:
            with ui.card("cy", "Next year's reports"):
                if cy.get("available"):
                    ui.kpi_strip([(f"{cy['macro_f1']:.2f}", f"macro-F1, {cy['test_year']}"),
                                  (f"{cy['n']}", "passages")])
                    ui.meta(f"Trained on {cy['train_year']} only, tested on {cy['test_year']}.")
                else:
                    ui.empty_state("Not available.")
        if MODEL_CARD.exists():
            st.download_button("Model card (markdown)", MODEL_CARD.read_text(encoding="utf-8").encode("utf-8"),
                               file_name="StratLens_MODEL_CARD.md", mime="text/markdown", icon=":material/download:")

# ======================================================================== limits
with tab3:
    lims = [("It measures emphasis, not spending.",
             "A theme's share is how much of the strategy text talks about it. It says nothing about money spent, "
             "execution or results, and it is not a ranking.")]
    if m:
        p = m["masking"]["company_probe"]
        lims.append((f"Writing style reveals the company {p['masked']:.0%} of the time.",
                     f"Even with names, brands and numbers hidden, a classifier guesses the author {p['masked']:.0%} of "
                     f"the time (chance: {p['majority_baseline']:.0%}). Some differences between companies may be house "
                     "style rather than strategy."))
    weak = []
    if au:
        weak = [x for x in au["per_company"] if x["macro_f1"] < 0.6]
    elif m:
        weak = [{"name": ui.company_name(s["group"]), "macro_f1": s["macro_f1"], "n": s["n"]} for s in m["subgroups"]
                if s["dimension"] == "Company" and s["macro_f1"] < 0.6]
    if weak:
        lims.append(("It is weaker on some companies.",
                     "; ".join(f"{w['name']}: {w['macro_f1']:.2f} on {w['n']} passages" for w in weak)
                     + ". The Company page warns when this applies."))
    if an:
        unseen = [c for c in an["companies"] if ui.reliability(c)[1] == "not in training"]
        if unseen:
            lims.append((f"{len(unseen)} companies were never in training.",
                         f"{', '.join(ui.company_name(c) for c in unseen)} are classified by a model trained only on "
                         "Dabur, HUL and Nestlé India; the model is less confident on them."))
    lims.append(("Labels are partly AI-drafted.",
                 "The training labels were drafted by an AI assistant following written guidelines. A hand-labelled "
                 "sample checks them" + (f" (agreement kappa {au['ai_vs_human']['kappa']:.2f})."
                                         if au and au.get("ai_vs_human") else "; that check is in progress.")))
    lims.append(("Only the Chairman's letter and the MD&A are read.",
                 "Sustainability reports, notes to accounts and investor presentations are ignored. Britannia "
                 "publishes no letter, and Godrej's MD&A is its whole integrated report, so its sustainability share "
                 "is not like-for-like."))
    lims.append(("A bag of words misses meaning.",
                 "Negation, tense and intent are invisible to word counts: 'we did not expand' looks like expansion."))
    for h, d in lims:
        st.markdown(f'<div class="sl-lim"><b>{html.escape(h)}</b><span>{html.escape(d)}</span></div>',
                    unsafe_allow_html=True)
    if m:
        ui.meta("Key audit findings from the original evaluation: " + " ".join(
            f"{h}." for h, _ in audit.key_findings(m)))
