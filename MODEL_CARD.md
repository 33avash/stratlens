# Model card: StratLens theme classifier

**Purpose.** This model classifies passages from Indian FMCG annual reports (Chairman/CEO letter and MD&A) into six strategy themes plus Other. The output is used to compare how companies *distribute their disclosure emphasis* across themes and how that emphasis shifts year to year. **Intended users:** students and analysts exploring narrative strategy disclosure, as one input alongside reading the reports.

**Data and provenance.**
- **Reports:** five annual reports: HUL FY2025 and FY2026, Dabur FY2025 and FY2026, Nestlé India FY2026. Nestlé India FY2025 was not available, so there is no Nestlé year-over-year comparison.
- **Sections used:** only the confirmed letter and MD&A page ranges (`data/sections.csv`). BRSR, governance and financial statements are excluded.
- **Passages:** 1,184 kept after cleaning; 464 blocks were dropped as captions, short fragments, tables or statutory boilerplate.
- **Labelling sample:** 885 passages, made up of 70% of each report's MD&A plus all letter passages (letters are oversampled for subgroup power).
- **Class counts:** PROD 122 · DIST 70 · DIGI 65 · OPS 61 · SUST 45 · BRAND 129 · OTHER 393. No class is below 40; SUST is the thinnest.

**Annotation method and verification status.** All labels were **drafted by an AI assistant (Claude) during the build** following `docs/labelling_guidelines.md`, and are marked `label_source = ai_drafted`. No LLM is called at runtime. A blind, class-stratified review sample of 150 passages is provided for human verification.

<!-- AGREEMENT:START -->
**Not yet verified.** A blind review sample of 150 passages is in `data/labels/human_review_sample.csv`; agreement will appear here after it is filled in.
<!-- AGREEMENT:END -->

**Model.**
- **Pipeline:** TF-IDF on unigrams and bigrams (sublinear tf, min_df 2, English stop words), then logistic regression with balanced class weights.
- **Tuning:** C was chosen by 5-fold stratified CV on the training set only. Best C = 30, CV macro-F1 0.76.
- **Split:** stratified 80/20 (708 train / 177 test, seed 42), saved in `data/splits/`.
- **Confidence:** each prediction's confidence is its top-class probability. Passages below **0.45** are flagged for review.

**Overall results (held-out test set, n = 177).**

| System | Accuracy | Macro precision | Macro recall | Macro-F1 |
|---|---|---|---|---|
| TF-IDF + logistic regression | 0.81 | 0.82 | 0.75 | **0.78** |
| Keyword baseline | 0.49 | 0.52 | 0.58 | 0.49 |

- **Per-class F1:** PROD 0.76, DIST 0.69, DIGI 0.73, OPS 0.78, SUST 0.82, BRAND 0.78, OTHER 0.87. Test support per class is only 9–79.
- **Review flag:** flags 15% of test passages. Accuracy is 87% above the threshold and 52% below it.
- **Cross-year check:** trained on FY2025 only and tested on FY2026 (n = 494, including the unseen company Nestlé), macro-F1 is 0.71.

**Subgroup results (test set; n < 30 = too small to conclude).**

| Dimension | Group | n | Macro-F1 | Macro P | Macro R |
|---|---|---|---|---|---|
| Writing style | Chairman/CEO letter | 46 | **0.59** | 0.69 | 0.56 |
| Writing style | MD&A | 131 | 0.79 | 0.81 | 0.77 |
| Length | Short (<25 words) | 43 | 0.75 | 0.80 | 0.75 |
| Length | Long (≥25 words) | 134 | 0.75 | 0.79 | 0.73 |
| Company | Dabur / HUL | 77 / 75 | 0.78 / 0.83 | 0.82 / 0.88 | 0.80 / 0.83 |
| Company | Nestlé *(too small to conclude)* | 25 | 0.46 | 0.45 | 0.53 |

**Key audit finding.** The model is markedly weaker on **Chairman/CEO letters** than on MD&A text: macro-F1 0.59 against 0.79, a 0.20 gap. Letters mix several levers in one sentence and use rhetorical, anecdotal language ("I met Firubi, our first woman area sales manager…"), so fewer of the theme's tell-tale terms appear. The most frequent single error is BRAND passages predicted as OTHER (4 of 26). Consumer-trend sentences share vocabulary with macro-economic commentary.

**Explainability.** The top coefficient terms are intuitive: *campaign, brand, consumer* (BRAND); *digital, ai, commerce, influencer* (DIGI); *reach, trade, retail, outlets* (DIST); *cost, supply chain, manufacturing, quality* (OPS); *portfolio, launch, innovation, variants* (PROD); *sustainability, plastic, circular, packaging* (SUST). Some terms are company or year artefacts: *india* and *2025* lean toward OTHER, and *feel* toward BRAND. The app's "Why this label?" view shows per-passage term contributions (tf-idf × coefficient). These weights show **association, not causation**.

**Limitations.**
1. **Unverified labels.** Scores measure agreement with AI-drafted labels by one annotator, not with expert consensus. The guideline boundaries (digital medium vs message; consumer trend vs macro) are judgement calls that a second annotator may draw differently.
2. **Small, narrow data.** The data covers 3 companies and 5 reports, with test n per class of 9–79. Nestlé has 25 test passages and only one year, and per-class scores have wide uncertainty.
3. **Bag-of-words model.** It cannot read negation, tense or intent ("we will", "we did not"), and it misreads rhetorical letter prose.
4. **Benchmark overlap with training.** About 60% of corpus passages (708 of 1,184) were in the training set, so their benchmark labels largely echo the drafted labels.
5. **Shares measure emphasis, not substance.** With 61–195 strategic passages per report, one passage moves a share by 0.5–1.6 pp, so small year-over-year shifts are noise.
6. **Reporting differences.** Nestlé's MD&A is embedded in its Board's Report, and HUL FY2026 uses two-page spreads. Section length and format differ by company.

**Inappropriate uses.** Do not use this model for investment decisions, stock prediction or valuation; for ranking companies as better or worse; for inferring actual spend, execution or ESG performance; or for reports outside Indian FMCG, or report sections other than letters and MD&A, without re-validation.
