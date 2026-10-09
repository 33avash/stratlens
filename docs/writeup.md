# StratLens: what Indian FMCG companies emphasise in their strategy narratives

## 1. Problem
Annual reports run to 250–600 pages, and the strategy story is a small part of them. Our question is: **how do competing Indian FMCG companies differ in what they emphasise when they describe their strategy, how concretely do they say it, and how does that change year to year?**

StratLens classifies narrative passages into six strategy themes plus Other:
- Product innovation & portfolio (PROD)
- Market expansion & distribution (DIST)
- Digital & e-commerce (DIGI)
- Operations & supply chain (OPS)
- Sustainability & sourcing (SUST)
- Consumer & brand strategy (BRAND)

It then compares each theme's share of the strategy narrative across companies and years, scores how specific each passage is, and links every number to its source passages. The output measures **disclosure emphasis, not investment or execution**.

## 2. Dataset
| Report | Letter pages | MD&A pages | Passages kept | Labelled |
|---|---|---|---|---|
| HUL FY2025 | 24–27 | 108–119 | 254 | 192 |
| HUL FY2026 | 9–10 (spreads) | 43–47 (spreads) | 189 | 147 |
| Dabur FY2025 | 19–21 | 130–151 | 275 | 199 |
| Dabur FY2026 | 28–30 | 174–201 | 322 | 233 |
| Nestlé India FY2026 | 7–8 | 34–37, 42–44 | 144 | 114 |

- **Sources:** official annual report PDFs; page ranges confirmed by hand (`data/sections.csv`). Nestlé India FY2025 is not yet included, so Nestlé has no year-over-year comparison.
- **Labels:** 885 passages, drafted by an AI assistant (Claude) at build time under written guidelines with 12 tie-break rules (`docs/labelling_guidelines.md`).
- **Class mix:** PROD 122, DIST 70, DIGI 65, OPS 61, SUST 45, BRAND 129, OTHER 393.
- **Verification:** labels were AI-drafted following the labelling guidelines and have **not been human-verified**. Every score below measures agreement with those drafts, not with an expert; this is the first limitation listed in the model card.

## 3. Preprocessing
1. **Extraction and section restriction:** PyMuPDF extracts text blocks with page numbers. Scanned, corrupt and empty files raise plain-English errors. Only the letter and MD&A pages are kept, so BRSR, governance and accounts don't dominate.
2. **Cleaning:** removes repeated headers and footers, page numbers, table-like blocks (over 40% numbers), captions and headings, and statutory boilerplate (Companies Act, AGM and director text). It also repairs hyphenation and merges sentences split across blocks.
3. **Segmentation:** an abbreviation-aware regex sentence splitter (Rs., Cr., Mr., i.e.) packs sentences into passages of 8–70 words, then exact duplicates are removed. Result: 1,184 passages kept, 464 blocks dropped (each logged with a reason).
4. **Masking (new):** before modelling, company names, 166 curated brand names, numbers and years become `[COMPANY]`, `[BRAND]`, `[NUM]`, `[YEAR]`. Internally they are distinct tokens, so they never collide with real words like "brand". The aim is a model that learns themes rather than authorship.

## 4. Approach
- **Main model:** masked TF-IDF on unigrams and bigrams (sublinear tf), then logistic regression with balanced class weights.
  - Split: stratified 80/20 (708 train / 177 test, seed 42).
  - Tuning: C = 10, chosen by 5-fold CV on the training set only.
- **Comparisons on the same test set:**
  - `all-MiniLM-L6-v2` sentence embeddings + logistic regression (C tuned the same way).
  - Unmasked TF-IDF (the masking test).
  - A keyword-lexicon baseline.
- **Uncertainty:** 1,000-resample bootstrap 95% CIs on macro-F1 (overall and per subgroup), and a paired bootstrap for model-vs-model differences. For year-over-year shifts, passages are resampled within each report, and a shift counts only if its interval excludes zero.
- **Specificity:** a rule-based 0–5 score per passage, with spaCy NER cached to disk:
  - **Quantities:** +1, or +2 for three or more.
  - **Date or deadline:** +1.
  - **Named initiative or brand:** +1 (spaCy NER, "Project X", quoted names; the company's own name and governance bodies are excluded).
  - **Commitment language:** +1 ("will", "target", "aim to", "by FY27").
  - **Tiers:** 0–1 Vague, 2 Moderate, 3+ Specific.
  - **Spot-checks** (by the AI assistant, not a person; `docs/specificity_spotcheck.csv`): 26/30 tiers agreed on 30 random passages, and 22/30 on a second sample of 10 strategic passages per tier (48/60 overall). The misses in the second sample run both ways (4 too high: "targeted" and "commitment to…" fire the commitment rule, "2.0" counts as a quantity; 4 too low), and no recalibrated rule beat the current one on both samples, so the thresholds were kept. Each evidence passage now shows which signals fired (e.g. "2 numbers · date · named: Vatika").
- **LLM layer:** Gemini (`gemini-3.5-flash`, temperature 0.2), used for two things only:
  - **Report prose:** it rewrites the narrative of a strategy brief from a JSON of computed figures.
  - **Ask the reports:** it answers questions from retrieved passages. Retrieval searches the unmasked text of every passage (strategy and Other) across all years by default. It is hybrid: BM25 over words and two-word phrases (so "quick commerce" is matched exactly) and MiniLM embedding cosine, fused by reciprocal rank fusion into a 0–1 relevance score shown per source. A passage is eligible only if its embedding cosine clears 0.40 (on-topic queries scored 0.56–0.67, off-topic ones at most 0.36), it contains a two-word phrase from the question, or it contains most of the question's content words; if none is eligible the app refuses. A question that names a company is restricted to that company.
  - **Spelling:** prompts ask for Indian English (British) spelling. One-year reports always say "Only one year available" rather than describing a comparison that cannot exist.
  - **Guardrails:** prompts forbid numbers or claims not in the data, and the model must cite passages or reply "NOT_IN_PASSAGES". Numbers in generated report text are checked against the figures; any section that fails falls back to template text.
  - **Fallback:** with no key, or on an API error, everything still works from templates.
- **App:** Streamlit with seven pages. Analysis: Overview · Briefs · Ask. Data: Upload. Model: How it works · Evaluation · Model card. *How it works* draws the pipeline and has an inspector that runs any sentence through cleaning, masking, TF-IDF, the classifier, the confidence flag and the specificity rules. The app loads saved artefacts and never retrains.

## 5. Results
**Model comparison (held-out n = 177; macro-F1 with 95% CI):**

| System | Accuracy | Macro-F1 | 95% CI |
|---|---|---|---|
| TF-IDF + LR, masked (main) | 0.81 | **0.77** | 0.69–0.84 |
| MiniLM embeddings + LR, masked | 0.79 | 0.77 | 0.69–0.83 |
| TF-IDF + LR, unmasked | 0.81 | 0.78 | 0.69–0.84 |
| Keyword baseline | 0.49 | 0.49 | 0.41–0.55 |

- **Embeddings vs TF-IDF:** embeddings differ by −0.01 (paired CI −0.07 to +0.06). The scores are effectively equal, so TF-IDF stays the main model because its decisions can be explained term by term.
- **Masking test:**
  - **Accuracy:** masking changes macro-F1 by −0.00 (paired CI −0.05 to +0.04).
  - **Company probe:** a company-identification classifier still reaches 79% on masked text (82% unmasked; majority guess 49%). Names are not what makes a passage identifiable; house style and topics are.
- **Confidence flag:** flags 21% of test passages (37 of 177; 113 of 1,184 corpus passages). Accuracy is 89% above the 0.45 threshold and 49% below it.
- **Cross-year check:** trained on FY2025 and tested on FY2026 (including unseen Nestlé), macro-F1 is 0.71.

**Audit (key finding):**
- **Masking and house style:** masking names, brands and numbers does not change theme accuracy (paired CI −0.05 to +0.04), but company identity is still 79% predictable from masked text (majority guess 49%). House style may therefore confound cross-company comparisons.
- **Direction of errors:** the largest group of errors (15 of 34) is strategy passages predicted as Other; the reverse happens 7 times. Strategy passages are missed more often than invented, so theme shares are likely conservative.
- **Company:** the only subgroup gap that survives bootstrap intervals is Nestlé India, at macro-F1 0.44 (CI 0.23–0.58) against 0.82 (0.67–0.91) for HUL. This rests on n = 25 Nestlé test passages and one year of data, so it is indicative only.
- **Writing style:** the v1 finding that the model is "much weaker on Chairman's letters" (0.59 vs 0.79 then) does not hold up. With masking and CIs, letters score 0.75 (0.45–0.86) and MD&A 0.77 (0.67–0.86), which is within noise. Macro-F1 over seven classes on 46 passages is too unstable to support that claim.
- **Most common single confusion:** BRAND predicted as OTHER (4 of 26).

**Benchmark findings (disclosure emphasis):**
- **Dabur:** its narrative is brand-led (34–36% BRAND), and its writing is the most concrete of the three (39% of strategy passages carry concrete detail, vs 13% for HUL and 10% for Nestlé).
- **YoY shifts:** the only shift that clears the noise is Dabur's digital & e-commerce, 6% → 14% (+8.0 pp, CI +1.9 to +14.1). Every other shift, including HUL's +4.3 pp on brand, is within noise.
- **HUL:** its narrative is spread evenly across themes, and it gives sustainability about three times the space the others do.

**Limitations:**
- Labels were AI-drafted following the labelling guidelines and have not been human-verified.
- House style: company identity is 79% predictable after masking, so cross-company differences may partly reflect writing style.
- The data is small: 5 reports, with test n per class of 9–79.
- Bag of words misses intent and tense.
- Specificity counts brand mentions, which flatters brand-by-brand reporting.
- About 60% of benchmark passages were seen in training.
- Year-over-year intervals ignore labelling error.

## 6. LLM layer design
The classifier remains the NLP core: every label and every number comes from the trained model and the bootstrap. Gemini is an optional writing layer behind a small `LLMClient` interface (only Gemini is implemented), with all prompts in `stratlens/llm/prompts.py`.

- **Guardrails:**
  - Grounding instructions in every prompt.
  - JSON output for report sections.
  - Automatic number checking, with template fallback per section.
  - Mandatory citations, and an explicit "not in passages" refusal.
  - Visible "AI-generated from cited passages" labels.
- **Operations:** low temperature, a 45 s timeout, two retries on 429/503 with friendly messages, and an on-disk cache so demos are repeatable and free.
