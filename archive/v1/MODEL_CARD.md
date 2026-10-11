# Model card: StratLens theme classifier

*Generated from reports/metrics.json on 2026-10-10.*

### Purpose
Classifies passages from Indian FMCG annual reports (Chairman/CEO letter and MD&A) into six strategy themes plus Other, to compare disclosure emphasis across companies and years. Not a measure of spending, execution or performance.

### Data
- **Reports:** Dabur India FY2025, Dabur India FY2026, Hindustan Unilever FY2025, Hindustan Unilever FY2026, Nestlé India FY2026. One year only: Nestlé India.
- **Passages:** 1,184 kept after cleaning; 464 blocks dropped.
- **Labels:** 885 passages (PROD 122 · DIST 70 · DIGI 65 · OPS 61 · SUST 45 · BRAND 129 · OTHER 393).
- **Provenance:** labels were AI-drafted following the labelling guidelines (`docs/labelling_guidelines.md`) and have not been human-verified.

### Model
- **NLP pipeline:** PDF text extraction → regex cleaning → sentence segmentation → masking of names, brands and numbers → TF-IDF features on words and word pairs → logistic regression over 7 classes.
- **Training:** 708 labelled passages, regularisation C = 10 picked by 5-fold cross-validation; scored once on 177 held-out passages.

| System | Accuracy | Macro-F1 | 95% CI |
|---|---|---|---|
| Word-weight classifier (TF-IDF) | 0.81 | **0.77** | 0.69–0.84 |
| Meaning-based classifier (MiniLM embeddings) | 0.79 | **0.77** | 0.69–0.83 |
| Word-weight classifier, names not hidden | 0.81 | **0.78** | 0.69–0.84 |
| Keyword matching (lexicon) | 0.49 | **0.49** | 0.41–0.55 |

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

### Key audit findings
- **Writing style gives the company away (79%).** Hiding names, brands and numbers doesn't hurt theme accuracy, yet a classifier still guesses the company 79% of the time (chance 49%). Cross-company differences may partly be house style.
- **It misses strategy more than it invents it (15 vs 7).** Of 34 test errors, 15 are strategy passages called Other and 7 go the other way, so theme shares are, if anything, understated.
- **Weaker on Nestlé India (0.44 vs 0.82).** Nestlé India scores macro-F1 0.44 [0.23–0.58] against 0.82 [0.67–0.91] for Hindustan Unilever on only 25 test passages, so it's a flag, not a verdict.

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
