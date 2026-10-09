# StratLens plan

## Section selection
- **Scope:** Chairman/CEO letter and MD&A only. Page ranges were confirmed by opening each PDF and are recorded in `config/section_overrides.csv`, with the output in `data/sections.csv`.
- **Fallback:** heading detection, used for uploads, covers reports that have no override.
- **HUL FY2025:** letters p24–27; MD&A narrative p108–119. From p120 the Board's Report turns to results and statutory content.
- **HUL FY2026:** two-page spreads. Letters p9–10; MD&A p43–47 (p48 onward is ratios and statutory text).
- **Dabur FY2025 / FY2026:** letter p19–21 / p28–30; MD&A p130–151 / p174–201.
- **Nestlé FY2026:** letter p7–8. Its MD&A is embedded in the Board's Report, so only the narrative pages are used (p34–37 and p42–44); director, dividend and audit pages are excluded.

## Dataset
- **Corpus:** 1,184 passages of 8–84 words after cleaning. Drops are logged with a reason in `data/processed/drop_log.csv`.
- **Labelling sample:** 70% of each report's MD&A plus 100% of letter passages, giving 885 labelled passages. Every class has at least 40; SUST is the thinnest at 45.
- **Labels:** AI-drafted (Claude, build time) and marked `ai_drafted`. A blind 150-passage review sample has near-equal class quotas.

## Model
- **Classifier:** TF-IDF (1–2 grams, sublinear tf, min_df 2) plus logistic regression with balanced class weights.
- **Tuning:** C chosen by 5-fold CV on the training set only, over a grid of 0.1–300; best C = 30.
- **Split:** stratified 80/20, seed 42.
- **Baseline:** a keyword lexicon.
- **Review threshold:** 0.45, kept because it flags 19% of CV predictions; accuracy is 86% above it versus 49% below.
- **Cross-year check:** train FY2025, test FY2026.

## Subgroups
Passage length (<25 vs ≥25 words), writing style (letter vs MD&A) and company. Any group with n < 30 is flagged as too small to conclude.

## Pages and visuals
- **Benchmark:** KPI strip; fingerprint heatmap; grouped theme bars (click to drill down); YoY diverging small multiples with a scale check; evidence quotes with highlighted terms; CSV export.
- **Analyse reports:** a 4-step indicator. Extraction KPIs and scanned warning; editable page ranges; kept and dropped counts by reason; classify; a filterable review table with confidence bars; "why this label"; CSV export; add to session benchmark.
- **Model audit:** everything is read from `reports/metrics.json`.
- **Model card:** rendered markdown with a download button.

## Deviations from the brief
| Brief default | What was done | Why |
|---|---|---|
| HUL, Dabur, Marico × 2 years | HUL and Dabur × 2 years, plus Nestlé India FY2026 | These were the reports provided. ITC was excluded because it is a conglomerate (cigarettes, hotels, paper) that would distort an FMCG comparison. The Nestlé file first supplied was Nestlé S.A.'s global review, not Nestlé India's, so Nestlé FY2025 is still missing. |
| Stratified sample of ~600–900 | 885, with letters oversampled (100%) | Letters are scarce. At 70% the letter test group would have been under 30. |
| Proportional review sample | Near-equal quotas per class (~21 each) | Proportional allocation gave only 8 SUST passages, too few for per-class agreement. |
| Theme palette as suggested | SUST `#6B7A3A` → `#5E7A28`, BRAND `#A65A44` → `#C0604A` | The suggested olive and terracotta were indistinguishable under deuteranopia (validator ΔE 0.6). They are now separated by lightness, and every theme mark also carries a text label. |
| Company colours | HUL `#24578A`, Dabur `#C27C2E`, Nestlé `#7D4E9E` | This set passes lightness, chroma, CVD and normal-vision checks on white. |
| NLTK sentence splitting | Regex splitter that handles abbreviations | Avoids a runtime data download and handles Indian report abbreviations (Rs., Cr.). |
| Heatmap drill-down by click | Click-through on the bar charts; evidence selectors for every cell | Streamlit's chart selection does not support heatmap cells. |
