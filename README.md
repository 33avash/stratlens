# StratLens

StratLens classifies strategy passages in Indian FMCG annual reports (the Chairman/CEO letter and MD&A) into six themes plus Other. It then compares each company's **disclosure emphasis** (how much of the narrative each theme gets) and how **concretely** they write, across companies and years. Every number traces back to source passages.

Built for NLP CA3 (Build–Audit–Pitch). See `WRITEUP.md`, `MODEL_CARD.md`, `PITCH.md` and `PLAN_v2.md`.

## Setup
Requires Python 3.11+ (developed on 3.13, Windows).
```bash
cd stratlens
python -m venv .venv && .venv\Scripts\activate      # optional
pip install -r requirements.txt                      # CPU-only torch; includes the spaCy model
```
The first run downloads the sentence-transformers model `all-MiniLM-L6-v2` (about 90 MB) from Hugging Face.

### AI writing (optional)
```bash
copy .env.example .env      # then set GEMINI_API_KEY=...
```
| Variable | Meaning |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API key. Without it the app runs fully and uses template text. |
| `GEMINI_MODEL` | Optional; default `gemini-3.5-flash`. |

`.env` is git-ignored. On Streamlit Cloud, put the same keys under *App settings → Secrets*.

Gemini only writes prose from figures the pipeline has already computed, and answers questions from retrieved passages with citations. It never labels passages or computes metrics. Generated text is cached in `data/cache/llm.json`, so repeated demos don't call the API again.

## Data
Put the annual report PDFs in `data/raw/`, named `COMPANY_FYyyyy.pdf`. They are not in the repo. The page ranges used are confirmed by hand in `config/section_overrides.csv`.

| File | Report |
|---|---|
| `HUL_FY2025.pdf`, `HUL_FY2026.pdf` | Hindustan Unilever Integrated Annual Reports 2024-25, 2025-26 |
| `DABUR_FY2025.pdf`, `DABUR_FY2026.pdf` | Dabur India Integrated Annual Reports 2024-25, 2025-26 |
| `NESTLE_FY2026.pdf` | Nestlé India Annual Report 2025-26 (FY2025 not yet added) |

## Commands
```bash
python scripts/build_corpus.py          # extract, section, clean, segment -> data/processed/corpus.csv
python train.py                         # split, tune, compare models, masking test, CIs, audit,
                                        #   benchmark predictions + specificity, MODEL_CARD.md
python -m pytest -q                     # tests
streamlit run app.py                    # launch
```
Labelling pipeline (already done): `scripts/sample_for_labelling.py` draws the sample, the drafted batches are in `data/labels/drafts/`, `scripts/merge_labels.py` merges them. Labels were AI-drafted following `docs/labelling_guidelines.md` and have not been human-verified.

## App pages
| Page | What it does |
|---|---|
| Overview | Per-report summaries, strategy fingerprint, "how much vs how concretely", YoY shifts with 95% intervals, evidence with fired specificity signals |
| Briefs | Strategy brief for one report or 2–3 companies; HTML and PDF export |
| Ask | Question → hybrid retrieval over all passages (BM25 + embeddings, rank fusion) → cited answer (needs Gemini; without it, passages only) |
| Upload | Upload PDF/TXT → check → classify → export, or add the report to the session benchmark |
| How it works | Pipeline diagram, one line per stage, and a pipeline inspector for any sentence |
| Evaluation | Model comparison, masking test, CIs, subgroups, explainability, errors |
| Model card | Generated `MODEL_CARD.md` |

## Layout
```
app.py, views/            Streamlit UI (named views/ so Streamlit's legacy pages/ auto-routing stays off)
stratlens/                extract, sections, preprocess, masking, dataset, baseline, model, embeddings, explain,
                          audit, uncertainty, specificity, compare, trace, report, theme, ui
stratlens/llm/            client (Gemini), prompts (all LLM prompts), retrieval
scripts/                  corpus, labelling, benchmark, agreement, model card
data/                     raw (git-ignored), processed, labels, splits, cache (embeddings, NER, LLM text)
models/, reports/         trained model, metrics.json, error notes
assets/                   stylesheet, logo, OFL fonts for PDF export
docs/                     labelling guidelines, specificity spot-check, screenshots
tests/                    pytest suite
```
