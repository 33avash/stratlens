# StratLens

StratLens classifies strategy passages in Indian FMCG annual reports (Chairman/CEO letter and MD&A) into six themes plus Other. It then compares each company's **disclosure emphasis** across companies and years, with every number traceable to source passages.

Built for NLP CA3 (Build, Audit, Pitch). See `WRITEUP.md`, `MODEL_CARD.md` and `PITCH.md`.

## Setup
Requires Python 3.11+ (developed on 3.13, Windows).
```bash
cd stratlens
python -m venv .venv && .venv\Scripts\activate      # optional
pip install -r requirements.txt
```

## Data
Put the annual report PDFs in `data/raw/`, named `COMPANY_FYyyyy.pdf`. The repository does not include them; they come from each company's investor-relations page.

| File | Report |
|---|---|
| `HUL_FY2025.pdf` | Hindustan Unilever Integrated Annual Report 2024-25 |
| `HUL_FY2026.pdf` | Hindustan Unilever Integrated Annual Report 2025-26 |
| `DABUR_FY2025.pdf` | Dabur India Integrated Annual Report 2024-25 |
| `DABUR_FY2026.pdf` | Dabur India Integrated Annual Report 2025-26 |
| `NESTLE_FY2026.pdf` | Nestlé India Annual Report 2025-26 |
| `NESTLE_FY2025.pdf` | Nestlé India Annual Report for the period ended March 2025 *(not yet added)* |

The page ranges used for each report are confirmed by hand in `config/section_overrides.csv`.

## Pipeline commands
```bash
python scripts/build_corpus.py          # extract, section, clean, segment -> data/processed/corpus.csv, data/sections.csv
python scripts/sample_for_labelling.py  # draw labelling sample (70% of MD&A, 100% of letters per report)
python scripts/merge_labels.py          # merge drafted label batches -> data/labels/labels.csv
python train.py                         # split, tune, evaluate, audit -> models/, reports/metrics.json, benchmark predictions
python scripts/make_review_sample.py    # blind 150-passage human review sample
python scripts/compute_agreement.py     # after filling human_label: agreement + kappa, updates MODEL_CARD.md
```
`train.py` is deterministic (seed 42). The app only loads saved artefacts.

## Tests
```bash
python -m pytest -q
```

## Launch
```bash
streamlit run app.py
```
Pages:
- **Benchmark** (landing): fingerprint, theme comparison, YoY shift and evidence.
- **Analyse reports:** upload → check → classify → export, or add the report to the session benchmark.
- **Model audit:** results, subgroups, explainability and errors.
- **Model card.**

## Adding a report (e.g. Nestlé FY2025)
1. Save the PDF as `data/raw/NESTLE_FY2025.pdf`, then add its letter and MD&A page ranges to `config/section_overrides.csv`.
2. Run `python scripts/build_corpus.py`, then `python scripts/sample_for_labelling.py`. Existing samples are kept and only new strata are drawn.
3. Label the new rows: add a `data/labels/drafts/bNN.txt` file with lines of the form `<row> <LABEL> [note]`, then run `python scripts/merge_labels.py`.
4. Run `python train.py`, then `python -m pytest -q`.
5. Update the figures quoted in `MODEL_CARD.md`, `WRITEUP.md` and `PITCH.md`. The app reads its numbers live from `reports/metrics.json`.

You can also upload a report in the app's Analyse page to classify it with the current model, without retraining.

## Layout
```
app.py, pages/            Streamlit UI (Benchmark, Analyse, Audit, Model card)
stratlens/                library: extract, sections, preprocess, dataset, baseline, model, explain, audit, compare, theme, ui
scripts/                  corpus building, labelling, benchmark, review and agreement
config/                   confirmed section page ranges
data/                     raw (gitignored), processed, labels, splits, sections.csv
models/, reports/         trained model, metrics.json, error notes, test errors
docs/                     labelling guidelines, screenshots
tests/                    pytest suite (fixtures generated at test time)
```
