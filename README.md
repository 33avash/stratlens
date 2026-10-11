# StratLens

**An NLP tool that reads FMCG annual reports and shows what each company emphasises in its strategy, how concretely, and what changed year to year.**

Annual reports run to 250–600 pages; the strategy story is about 20 of them. StratLens extracts the Chairman's letter and MD&A, sorts every passage into six strategy themes with a trained text classifier, and compares the themes' shares across HUL, Dabur and Nestlé India. Every number links back to the sentences behind it.

![Overview](docs/screenshots/overview.png)

> Shares measure what companies choose to talk about, not what they spend or achieve.

## The NLP at each step

| # | Step | NLP technique |
|---|---|---|
| 1 | Read | Text extraction from PDF, with page numbers |
| 2 | Clean | Text preprocessing with regular expressions |
| 3 | Split | Sentence segmentation and tokenisation into 8–70 word passages |
| 4 | Hide names | Entity masking: company, brand, number and year become placeholders |
| 5 | Words → numbers | Bag-of-words → TF-IDF on words and word pairs (n-grams) |
| 6 | Classify | Supervised text classification (logistic regression), with a confidence score |
| 7 | How concrete | Named Entity Recognition (spaCy) + regex scoring of figures, dates and commitments |
| 8 | Ask and briefs | Retrieval (BM25 + sentence embeddings) and an LLM that writes cited answers |
| 9 | Check | Precision, recall, F1, subgroup audit, word-level explainability |

The app's **How it works** page shows this table and lets you run any sentence through every step.

## Results

5 reports → 1,184 passages → 885 labelled (708 train / 177 unseen test).

| Model (same 177 unseen passages) | Accuracy | Macro-F1 | 95% CI |
|---|---|---|---|
| **Word-weight classifier (TF-IDF)**, served | 0.81 | **0.77** | 0.69–0.84 |
| Meaning-based classifier (MiniLM embeddings) | 0.79 | 0.77 | 0.69–0.83 |
| Keyword matching (lexicon) | 0.49 | 0.49 | 0.41–0.55 |

Embeddings tie with TF-IDF, so the model whose decisions trace back to words is served.

**What we found**
- Dabur leads with brand-building (about a third of its strategy talk); HUL gives sustainability about three times the space of its peers; Nestlé India leans on distribution.
- The only year-on-year shift beyond noise is Dabur's digital & e-commerce share, 6% → 14%.

**Key audit findings**
- **House style leaks:** hiding names doesn't change accuracy, but the company is still 79% identifiable from masked text (chance 49%).
- **Errors are conservative:** 15 of 34 errors miss a strategy passage, and 7 invent one.
- **Nestlé India is weaker** (0.44 vs 0.82 for HUL) but rests on only 25 test passages.

## Quick start

Python 3.11+.

```bash
pip install -r requirements.txt     # CPU-only torch; includes the spaCy model
streamlit run app.py
```

Trained artefacts are committed, so the app runs without retraining or the source PDFs. The first run downloads `all-MiniLM-L6-v2` (~90 MB).

**Optional AI writing:** copy `.env.example` to `.env` and set `GEMINI_API_KEY` (optional `GEMINI_MODEL`, default `gemini-3.5-flash`). Without a key, everything works with template text.

## App

| Section | Page | What it shows |
|---|---|---|
| Analysis | Overview | What we found, strategy fingerprint, how much vs how concretely, evidence passages, YoY shifts with 95% CIs |
| | Briefs | One-page brief per report or a 2–3 company comparison; HTML/PDF export |
| | Ask | Question → hybrid retrieval over all passages → cited answer |
| Data | Upload | Classify a new report (PDF/TXT) and add it to the Overview |
| Model | How it works | Pipeline diagram, stage table, step-by-step inspector |
| | Evaluation | Model comparison, robustness (subgroups, masking, author probe, cross-year), explainability, errors |
| | Model card | Generated `MODEL_CARD.md` |

![Evaluation](docs/screenshots/evaluation.png)

## Reproduce

Put the reports in `data/raw/` as `COMPANY_FYyyyy.pdf` (not in the repo; page ranges are in `config/section_overrides.csv`), then:

```bash
python scripts/build_corpus.py   # extract → clean → segment → data/processed/corpus.csv
python train.py                  # tune, compare, audit → models/, reports/metrics.json, MODEL_CARD.md
python -m pytest -q              # 58 tests
```

## Repository

```
app.py, views/        Streamlit app (one file per page)
stratlens/            pipeline: extract, preprocess, masking, model, explain, audit, uncertainty,
                      specificity, compare, report, trace; llm/ (Gemini client, prompts, retrieval)
train.py, scripts/    training, corpus build, labelling, benchmark, model card
data/                 corpus, labels, splits, predictions, caches (raw PDFs git-ignored)
models/, reports/     trained classifier, metrics.json, error analysis
docs/                 write-up, pitch, labelling guidelines, specificity spot-check
tests/                pytest suite
```

## Limitations

- **Labels** were AI-drafted following [`docs/labelling_guidelines.md`](docs/labelling_guidelines.md) and have **not been human-verified**. Scores measure agreement with those drafts.
- **Small data:** 5 reports, 3 companies, 9–79 test passages per theme; Nestlé has one year only.
- **Bag of words** misses negation, tense and intent.
- **Specificity** is rule-based (48/60 tiers agreed in two spot-checks) and favours brand-by-brand writing.

More: [model card](MODEL_CARD.md) · [write-up](docs/writeup.md) · [pitch](docs/pitch.md)
