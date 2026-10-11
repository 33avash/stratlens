# StratLens: reading FMCG strategy from annual reports with NLP

## 1. Problem
An FMCG annual report runs to 250–600 pages, and only about 20 of them tell the strategy story. An analyst who follows HUL, Dabur and Nestlé India cannot easily answer three questions:
1. **What** does each company emphasise when it describes its strategy?
2. **How concretely** does it say it (figures and deadlines, or general statements)?
3. **What changed** from one year to the next?

StratLens answers these with an NLP pipeline. It sorts every strategy passage into one of six themes, compares each theme's share across companies and years, and links every number back to the sentences behind it.

| Code | Theme |
|---|---|
| PROD | Product innovation & portfolio |
| DIST | Market expansion & distribution |
| DIGI | Digital & e-commerce |
| OPS | Operations & supply chain |
| SUST | Sustainability & sourcing |
| BRAND | Consumer & brand strategy |
| OTHER | Not strategy (financial, generic, governance) |

> The output measures **what companies choose to talk about**, not what they spend or achieve.

## 2. Dataset
- **Source:** five official annual reports: HUL FY2025 and FY2026, Dabur FY2025 and FY2026, and Nestlé India FY2026. Only the **Chairman's letter and the MD&A** are used, so legal and financial pages don't swamp the strategy text. The page ranges are recorded in `data/sections.csv`.
- **Size:** 1,184 passages after cleaning. 885 of them are labelled (708 for training, 177 held back for testing).
- **Class mix:** PROD 122 · DIST 70 · DIGI 65 · OPS 61 · SUST 45 · BRAND 129 · OTHER 393.
- **Labels:** drafted by an AI assistant following written guidelines with tie-break rules (`docs/labelling_guidelines.md`). **They have not been checked by a person**, so every score below measures agreement with those drafts.

## 3. Preprocessing and the NLP at each step

| # | Step | NLP technique | Syllabus unit | What it does |
|---|---|---|---|---|
| 1 | Read | Text extraction (PyMuPDF) | 1: NLP pipeline | Pulls text with page numbers; keeps the letter and MD&A pages |
| 2 | Clean | Text preprocessing with **regular expressions** | 2: regex | Removes headers, page numbers, tables and legal boilerplate; repairs hyphenated words. 464 blocks dropped, each with a logged reason |
| 3 | Split | **Sentence segmentation** and tokenisation | 1–2 | Splits sentences without breaking on "Rs." or "Mr.", packs them into 8–70 word passages and removes duplicates |
| 4 | Hide names | **Entity masking** (name lists + regex) | 2: NER; 5: bias | Replaces company names, 166 brands, numbers and years with `[COMPANY]`, `[BRAND]`, `[NUM]`, `[YEAR]`, so the model learns topics rather than authors |
| 5 | Words → numbers | **Bag-of-words → TF-IDF**, 1–2 word **n-grams** | 3: feature representation | Weights each word and word pair (e.g. "quick commerce") by how distinctive it is |
| 6 | Classify | **Supervised text classification** (logistic regression) | 4: ML techniques | Picks one of 7 classes, with a confidence score; passages below 0.45 are flagged |
| 7 | How concrete | **Named Entity Recognition** (spaCy) + regex | 2: NER | Scores each passage 0–5: figures (up to +2), a date (+1), a named initiative (+1), a commitment such as "will" or "target" (+1) |
| 8 | Ask and briefs | **Retrieval** (BM25 + embeddings) and an **LLM** (Gemini) | 5: LLMs | Finds passages for a question and writes cited answers and one-page briefs. The LLM never labels text or computes numbers |
| 9 | Check | Precision, recall, F1, subgroup tests, explainability | 5: evaluation, bias | See sections 5 and 6 |

## 4. Approach
- **Main model:** TF-IDF features and a logistic regression with balanced class weights. Regularisation (C = 10) was chosen by 5-fold cross-validation on the training set only. The test set was used once, at the end.
- **Comparisons on the same 177 test passages:**
  - a **keyword lexicon** baseline (a hand-written word list per theme);
  - **MiniLM sentence embeddings** with the same classifier (a meaning-based representation);
  - the main model **without masking**, to see whether hiding names matters.
- **Why the simpler model:** embeddings scored the same (difference −0.01, 95% CI −0.07 to +0.06). TF-IDF's decisions can be traced to individual words, so it is the one served.
- **Honest uncertainty:** every score carries a 95% bootstrap confidence interval. A year-on-year change counts only if its interval excludes zero.

## 5. Results

| Model (same 177 unseen passages) | Accuracy | Macro-F1 | 95% CI |
|---|---|---|---|
| **Word-weight classifier (TF-IDF)**, served | 0.81 | **0.77** | 0.69–0.84 |
| Meaning-based classifier (MiniLM embeddings) | 0.79 | 0.77 | 0.69–0.83 |
| Word-weight classifier, names not hidden | 0.81 | 0.78 | 0.69–0.84 |
| Keyword matching (lexicon) | 0.49 | 0.49 | 0.41–0.55 |

- **The model beats keywords by 0.28 macro-F1.** It is right 81% of the time on text it has never seen.
- **The confidence flag works:** 21% of test passages fall below 0.45. Accuracy is 89% above the threshold and 49% below it.
- **It holds across years:** trained on FY2025 and tested on FY2026, including the unseen Nestlé report, it scores 0.71.

**What the companies emphasise** (share of strategy passages):
- **Dabur** leads with brand-building (34–36%, about 16–22 points above peers), and its writing is the most concrete of the three.
- **HUL** spreads its attention evenly across themes and gives sustainability about three times the space the others do.
- **Nestlé India** leans on distribution (25%, 12 points above peers).
- **Only one year-on-year change clears the noise:** Dabur's digital & e-commerce share, from 6% to 14% (95% CI +1.9 to +14.1 points). Every other shift is within noise.

## 6. Audit (summary; full detail in `MODEL_CARD.md`)
- **Subgroups:** short vs long passages, Chairman's letter vs MD&A, and by company. Writing style (letter 0.75 vs MD&A 0.77) and length (0.75 vs 0.75) make no real difference. Nestlé India is weaker (0.44 vs 0.82 for HUL), but on only 25 test passages, so it is a flag rather than a result.
- **Explainability:** the app shows the top words per theme (model coefficients) and, for any passage, which words pushed it towards its label. These weights show association, not causation.
- **Key finding:** hiding names doesn't change accuracy, yet a separate classifier still identifies the company 79% of the time from masked text (chance is 49%). Cross-company differences may therefore partly reflect house writing style.
- **Error direction:** 15 of 34 errors are strategy passages called Other, and 7 go the other way, so theme shares are, if anything, understated.

## 7. Limitations
1. **Labels are AI-drafted and not human-verified.** This is the main limitation.
2. **Small data:** 5 reports, 3 companies, and only 9–79 test passages per theme, so intervals are wide.
3. **Bag-of-words** ignores negation, tense and intent ("we will not expand" looks like expansion).
4. **The specificity score** is rule-based and rewards naming brands, which favours Dabur's brand-by-brand style.
