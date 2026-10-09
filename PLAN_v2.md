# StratLens v2 plan

**Starting point (v1, from `reports/metrics.json`):**
- **Data:** 885 AI-drafted labels across 5 reports.
- **Model:** TF-IDF + LR scores macro-F1 0.78 on the held-out test set (n = 177), against 0.49 for the keyword baseline.
- **Key audit finding:** the model is weaker on Chairman/CEO letters (macro-F1 0.59) than on MD&A text (0.79).

v2 extends this without rewriting it. The existing pages, the design system and the 80/20 split all stay.

## What gets built
| Item | Module / file | Notes |
|---|---|---|
| Masking of company, brand, number and year | `stratlens/masking.py` | Uses curated per-company brand lists plus regex. Placeholders are distinct tokens (`__brand__`), so they never collide with real words such as "brand". |
| Masking test | `train.py` → `metrics.json` | Test macro-F1 with vs without masking on the same split. Also a company-identification probe (can text predict the company?) with vs without masking. |
| Embedding model | `stratlens/embeddings.py` | `all-MiniLM-L6-v2` + LR, with C tuned by CV on train only. Embeddings are cached to `data/cache/`. |
| Bootstrap CIs | `stratlens/uncertainty.py` | 95% CIs for test macro-F1 (overall and per subgroup, 1,000 resamples), and for YoY share changes (resampling passages within each report). |
| Specificity score | `stratlens/specificity.py` | Concrete signals: numbers/₹/%, dates/deadlines, named initiatives (spaCy NER + "Project X"), commitment language. Tiers: Vague / Moderate / Specific. NER is cached to disk. A 30-passage spot-check goes in `docs/specificity_spotcheck.csv`. |
| Human review | `stratlens/review.py`, page `Review labels` | Shows the blind sample first, then low-confidence passages. Saves to `data/human_labels.csv`. Shows live agreement and kappa. `train.py` prefers human labels and counts them. |
| Report briefs + per-report summaries | `stratlens/report.py`, page `Reports` | Template text built from computed figures, with Gemini rewriting it when a key is configured. Number check on generated text. HTML and PDF export (fpdf2 with bundled OFL fonts). A short summary per report also appears on the Benchmark page. |
| LLM layer (Gemini) | `stratlens/llm/{client,prompts,retrieval}.py` | `google-genai`, model from `GEMINI_MODEL` (default `gemini-3.5-flash`, thinking low, temperature 0.2). Retries on 429/503 and falls back to templates. Generated text is cached on disk. |
| Ask the reports | page `Ask` | Embedding retrieval with a relevance floor. Gemini answers from the cited passages only; with no key, the page shows the retrieved passages without an answer. |
| Model card auto-generated | `scripts/write_model_card.py` | Called by `train.py`, so every number in the card comes from `metrics.json`. |

## Deviations
- The brief names `src/llm/prompts.py`. The package here is `stratlens/`, so prompts live in `stratlens/llm/prompts.py`.
- The YoY bootstrap resamples passages, not labels or model weights. It captures how much a share depends on which passages a report happens to contain, not model error.
- Deployment: `torch` is pinned to the CPU build via an extra index URL, so Streamlit Cloud doesn't pull CUDA. If `sentence-transformers` is unavailable at runtime, Ask falls back to TF-IDF retrieval and says so.
- Human labels saved on Streamlit Cloud are lost when the app restarts; label locally and commit `data/human_labels.csv`.
