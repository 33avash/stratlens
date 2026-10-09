# StratLens: classifying strategic emphasis in FMCG annual reports

## 1. Problem
Annual reports are long, and the strategy narrative in them is buried among financial statements, governance and sustainability disclosures. Our question is: **how do competing Indian FMCG companies differ in their communicated strategic priorities, and how do those priorities change year to year?**

StratLens answers this by classifying narrative passages into six strategy themes plus Other:
- Product innovation & portfolio (PROD)
- Market expansion & distribution (DIST)
- Digital & e-commerce (DIGI)
- Operations & supply chain (OPS)
- Sustainability & sourcing (SUST)
- Consumer & brand strategy (BRAND)

It then compares each theme's normalised share of the strategic narrative across companies and years. Every number links back to the passages behind it. The output measures **disclosure emphasis, not investment or execution**.

## 2. Dataset
| Report | Letter pages | MD&A pages | Passages kept | Labelled |
|---|---|---|---|---|
| HUL FY2025 | 24–27 | 108–119 | 254 | 192 |
| HUL FY2026 | 9–10 (spreads) | 43–47 (spreads) | 189 | 147 |
| Dabur FY2025 | 19–21 | 130–151 | 275 | 199 |
| Dabur FY2026 | 28–30 | 174–201 | 322 | 233 |
| Nestlé India FY2026 | 7–8 | 34–37, 42–44 | 144 | 114 |

- **Sources:** official annual report PDFs (HUL, Dabur and Nestlé India for FY2025–FY2026), kept outside version control.
- **Missing report:** Nestlé India FY2025 was not available, so Nestlé has no year-over-year comparison.
- **Replaced company:** the brief suggested Marico; Nestlé India replaced it because those were the reports available. ITC was excluded because its report covers non-FMCG businesses.
- **Section ranges:** confirmed by inspecting each PDF and stored in `config/section_overrides.csv` and `data/sections.csv`.
- **Labels:** 885 passages, labelled by **AI drafting (Claude, during the build)** under written guidelines with 12 tie-break rules (`docs/labelling_guidelines.md`).
- **Class mix:** PROD 122, DIST 70, DIGI 65, OPS 61, SUST 45, BRAND 129, OTHER 393.
- **Verification:** a blind 150-passage human-review sample and an agreement/Cohen's kappa script are provided. Verification is pending.

## 3. Preprocessing
1. **Extraction:** text blocks are extracted with PyMuPDF, keeping page numbers. Scanned, corrupt, encrypted and empty files are detected and raise plain-English errors.
2. **Section restriction:** only the Chairman/CEO letter and MD&A are kept, so 300–600-page reports are not dominated by BRSR, governance and accounts.
3. **Header/footer removal:** running headers and footers are lines repeated on 3 or more pages; they are removed along with page numbers.
4. **Block filters:** table-like blocks (over 40% numeric tokens) and headings or captions (fewer than 6 words, or starting with Figure/Source/Note) are dropped.
5. **Text repair:** hyphenation is repaired, bullets are stripped, and blocks split mid-sentence are merged.
6. **Segmentation:** a regex sentence splitter that knows abbreviations (Rs., Cr., Mr., i.e.) splits the text, and sentences are packed into passages of up to 70 words.
7. **Passage filters:** passages under 8 words, statutory boilerplate (Companies Act, AGM, director appointments) and numeric-heavy passages are dropped, and the rest de-duplicated.
8. **Outcome:** 1,184 passages kept; 464 blocks dropped and logged with a reason.

## 4. Approach
- **Model:** TF-IDF on unigrams and bigrams (sublinear tf), then logistic regression with balanced class weights. The model is interpretable, fast and reproducible (seed 42).
- **Split and tuning:** stratified 80/20 split saved to disk; C tuned by 5-fold CV on the training set only (best C = 30).
- **Baseline:** a keyword-lexicon baseline written from the guidelines, used only for comparison.
- **Confidence:** each prediction's confidence is its top-class probability; below 0.45 it is flagged for review. CV showed this flags about 19% of predictions, which is a useful review rate.
- **Benchmark logic:** theme share = theme passages ÷ strategic (non-Other) passages per company-year. YoY change is in percentage points.
- **App:** a Streamlit app with four pages (Benchmark, Analyse reports, Model audit, Model card). It loads saved artefacts and never retrains.

## 5. Results
**Held-out test set (n = 177):** macro-F1 **0.78** (accuracy 0.81) versus **0.49** for the keyword baseline. Per-class F1 ranges from 0.69 (DIST) to 0.87 (OTHER). The review flag catches 15% of test passages, where accuracy drops to 52%, against 87% for unflagged passages. A stricter cross-year test (train FY2025, test FY2026 including unseen Nestlé) gives macro-F1 0.71.

**Audit.** Subgroup analysis shows the clearest weakness in **writing style**: macro-F1 is 0.59 on Chairman/CEO letters (n = 46) against 0.79 on MD&A (n = 131). Passage length makes no difference (0.75 vs 0.75). Nestlé (n = 25) is too small to conclude. The most common error is BRAND → OTHER. Error analysis attributes most mistakes to genuine overlap (influencer-led brand building, consumer trends vs macro commentary) and to rhetorical letter prose.

**Benchmark (disclosure emphasis).**
- **Dabur:** its narrative is the most brand- and product-led (BRAND 37%/35%, PROD 30%/27% in FY2025/FY2026). Digital is its largest shift, +7.6 pp (6% → 14%).
- **HUL:** spreads emphasis evenly across all six themes, and gives sustainability far more space (14–17%) than Dabur or Nestlé (4–5%).
- **Nestlé FY2026:** distribution and product account for 26% each.

These are differences in what the companies write about, not in what they do.

**Limitations.**
- The labels are AI-drafted and unverified, and a single annotator drew the theme boundaries.
- The data is small (5 reports, test n per class 9–79), and a bag-of-words model misses intent and tense.
- About 60% of benchmark passages were seen in training.
- Year-over-year shifts of a few points are within the noise of 61–195 strategic passages per report.
