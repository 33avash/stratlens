# Model card: StratLens theme classifier

*Generated from reports/metrics.json on 2026-10-09.*

### Purpose
Classifies passages from Indian FMCG annual reports (Chairman/CEO letter and MD&A) into six strategy themes plus Other, to compare disclosure emphasis across companies and years. Not a measure of spending, execution or performance.

### Data
- **Reports:** Dabur India FY2025, Dabur India FY2026, Hindustan Unilever FY2025, Hindustan Unilever FY2026, Nestlé India FY2026. One year only: Nestlé India.
- **Passages:** 1,184 kept after cleaning; 464 blocks dropped.
- **Labels:** 885 passages (PROD 122 · DIST 70 · DIGI 65 · OPS 61 · SUST 45 · BRAND 129 · OTHER 393).
- **Provenance:** labels were AI-drafted following the labelling guidelines (`docs/labelling_guidelines.md`) and have not been human-verified.

### Model
Masked TF-IDF (1–2 grams) + logistic regression with balanced class weights. C = 10 by 5-fold CV on 708 training passages; tested on 177 held-out passages. Company names, brands, numbers and years are masked before training.

| System | Accuracy | Macro-F1 | 95% CI |
|---|---|---|---|
| TF-IDF + LR (masked) | 0.81 | **0.77** | 0.69–0.84 |
| MiniLM embeddings + LR (masked) | 0.79 | **0.77** | 0.69–0.83 |
| TF-IDF + LR (unmasked) | 0.81 | **0.78** | 0.69–0.84 |
| Keyword baseline | 0.49 | **0.49** | 0.41–0.55 |

### Subgroups (held-out)
| Group | n | Macro-F1 | 95% CI |
|---|---|---|---|
| Long (>=25 words) | 134 | 0.75 | 0.64–0.83 |
| Short (<25 words) | 43 | 0.75 | 0.46–0.83 |
| Chairman/CEO letter | 46 | 0.75 | 0.45–0.86 |
| MD&A | 131 | 0.77 | 0.67–0.86 |
| Dabur India | 77 | 0.80 | 0.60–0.90 |
| Hindustan Unilever | 75 | 0.82 | 0.67–0.91 |
| Nestlé India *(small n)* | 25 | 0.44 | 0.23–0.58 |

### Key finding
Masking company names, brands and numbers does not change theme accuracy (paired CI -0.05 to +0.04), but company identity is still 79% predictable from masked text (majority guess 49%), so house style may confound cross-company comparisons. The largest group of errors is strategic passages predicted as Other (15 of 34; the reverse happens 7 times), so strategy passages are missed more often than invented and theme shares are likely conservative. By company, Nestlé India scores macro-F1 0.44 [0.23–0.58] against 0.82 [0.67–0.91] for Hindustan Unilever. Only n = 25 test passages come from Nestlé India, so treat this as indicative.

### Explainability
Top coefficients per theme and per-passage term contributions (tf-idf × coefficient) are shown in the app. They show association, not causation.

### Limitations
1. **Labels:** AI-drafted following the labelling guidelines and not human-verified. Every score here measures agreement with those drafts, not with an expert.
2. **Size:** 5 reports; 9–79 test passages per theme, so intervals are wide.
3. **Representation:** bag of words misses negation, tense and intent.
4. **Specificity:** rule-based; 26/30 and 22/30 tiers agreed with in two spot-checks (48/60 overall; misses run both ways, so the thresholds were kept). Spot-checks were done by the AI assistant, not a person. Brand mentions count as named initiatives (favours brand-by-brand writing); "targeted", "commitment to…" and version numbers such as "2.0" can over-score.
5. **Errors:** 15 of 34 test errors are strategic passages predicted as Other (7 the reverse), so theme shares are likely conservative.
6. **House style:** company identity is 79% predictable from masked text, so cross-company differences may partly reflect writing style.
7. **Benchmark:** shares include training passages; YoY intervals ignore labelling error.

### Text generation
An optional LLM (Gemini) writes brief prose from the classifier's computed figures and answers Ask questions from passages found by hybrid retrieval (BM25 + MiniLM embeddings, rank fusion). It does not classify passages or produce metrics; numbers it writes are checked against the data, and Ask refuses when nothing relevant is retrieved.

### App
Analysis: Overview · Briefs · Ask. Data: Upload. Model: How it works (pipeline diagram and inspector) · Evaluation · Model card.

### Not for
Investment decisions, valuation, ranking companies, inferring actual spend, or other industries without re-validation.
