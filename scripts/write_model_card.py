"""Generate MODEL_CARD.md from reports/metrics.json, the corpus files and the specificity spot-check.

Every number in the card comes from those files; only the explanatory prose is fixed text.
Run automatically by train.py.
Usage: python scripts/write_model_card.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from stratlens import audit
from stratlens.config import (COMPANY_NAMES, CORPUS_CSV, DROPS_CSV, METRICS_JSON, MODEL_CARD, ROOT, SECTIONS_CSV,
                              THEMES)

SPOTCHECK_CSV = ROOT / "docs" / "specificity_spotcheck.csv"


def f2(x):
    return f"{x:.2f}"


def ci(c):
    return f"{c[0]:.2f}–{c[1]:.2f}" if c else "–"


def main() -> None:
    m = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    corpus = pd.read_csv(CORPUS_CSV)
    drops = pd.read_csv(DROPS_CSV)
    secs = pd.read_csv(SECTIONS_CSV)
    reports = secs[["company", "year"]].drop_duplicates().sort_values(["company", "year"])
    report_list = ", ".join(f"{COMPANY_NAMES.get(c, c)} {y}" for c, y in reports.values)
    years_per = reports.groupby("company").size()
    one_year = [COMPANY_NAMES.get(c, c) for c, n in years_per.items() if n < 2]

    d = m["dataset"]
    t = m["test"]["model"]
    th = m["threshold"]
    mk = m["masking"]
    probe = mk["company_probe"]
    cc = d["class_counts"]

    spot = pd.read_csv(SPOTCHECK_CSV) if SPOTCHECK_CSV.exists() else None
    if spot is not None and len(spot):
        by = spot.groupby("batch", sort=False)["agree"].agg(["sum", "count"])
        spot_txt = (" and ".join(f"{int(r['sum'])}/{int(r['count'])}" for _, r in by.iterrows())
                    + f" tiers agreed with in two spot-checks ({int(by['sum'].sum())}/{int(by['count'].sum())} overall; "
                    "misses run both ways, so the thresholds were kept)")
    else:
        spot_txt = "not spot-checked"
    ed = m["test"].get("error_direction") or audit.error_direction(m["test"]["confusion"])
    comp_rows = "\n".join(
        f"| {r['system']} | {f2(r['accuracy'])} | {f2(r['macro_precision'])} | {f2(r['macro_recall'])} | "
        f"**{f2(r['macro_f1'])}** | {ci(r['macro_f1_ci'])} |" for r in m["comparison"])
    sub_rows = "\n".join(
        f"| {s['dimension'].split(' (')[0]} | {s['group']}{' *(n < ' + str(m['small_n']) + ', too small)*' if s['too_small'] else ''} "
        f"| {s['n']} | {f2(s['macro_f1'])} | {ci(s.get('macro_f1_ci'))} | {f2(s['macro_precision'])} | "
        f"{f2(s['macro_recall'])} |" for s in m["subgroups"])
    terms = "; ".join(f"*{', '.join(x['term'].replace('__', '') for x in m['global_terms'][c][:4])}* ({c})"
                      for c in THEMES if c != "OTHER")
    thin = d["thin_classes"]

    sup = [v_["support"] for v_ in m["test"]["per_class"].values()]
    sub_rows = "\n".join(
        f"| {COMPANY_NAMES.get(s['group'], s['group'])}{' *(small n)*' if s['too_small'] else ''} | {s['n']} | "
        f"{f2(s['macro_f1'])} | {ci(s.get('macro_f1_ci'))} |" for s in m["subgroups"])
    comp_rows = "\n".join(f"| {r['system']} | {f2(r['accuracy'])} | **{f2(r['macro_f1'])}** | {ci(r['macro_f1_ci'])} |"
                          for r in m["comparison"])
    card = f"""# Model card: StratLens theme classifier

*Generated from reports/metrics.json on {m['generated']}.*

### Purpose
Classifies passages from Indian FMCG annual reports (Chairman/CEO letter and MD&A) into six strategy themes plus Other, to compare disclosure emphasis across companies and years. Not a measure of spending, execution or performance.

### Data
- **Reports:** {report_list}.{(' One year only: ' + ', '.join(one_year) + '.') if one_year else ''}
- **Passages:** {len(corpus):,} kept after cleaning; {len(drops):,} blocks dropped.
- **Labels:** {d['n_labelled']} passages ({' · '.join(f'{k} {v}' for k, v in cc.items())}).{(' Thin: ' + ', '.join(thin) + '.') if thin else ''}
- **Provenance:** labels were AI-drafted following the labelling guidelines (`docs/labelling_guidelines.md`) and have not been human-verified.

### Model
Masked TF-IDF (1–2 grams) + logistic regression with balanced class weights. C = {m['model']['best_C']} by 5-fold CV on {d['n_train']} training passages; tested on {t['n']} held-out passages. Company names, brands, numbers and years are masked before training.

| System | Accuracy | Macro-F1 | 95% CI |
|---|---|---|---|
{comp_rows}

### Subgroups (held-out)
| Group | n | Macro-F1 | 95% CI |
|---|---|---|---|
{sub_rows}

### Key finding
{m['key_finding']}

### Explainability
Top coefficients per theme and per-passage term contributions (tf-idf × coefficient) are shown in the app. They show association, not causation.

### Limitations
1. **Labels:** AI-drafted following the labelling guidelines and not human-verified. Every score here measures agreement with those drafts, not with an expert.
2. **Size:** {len(reports)} reports; {min(sup)}–{max(sup)} test passages per theme, so intervals are wide.
3. **Representation:** bag of words misses negation, tense and intent.
4. **Specificity:** rule-based; {spot_txt}. Spot-checks were done by the AI assistant, not a person. Brand mentions count as named initiatives (favours brand-by-brand writing); "targeted", "commitment to…" and version numbers such as "2.0" can over-score.
5. **Errors:** {ed['strategic_to_other']} of {ed['total']} test errors are strategic passages predicted as Other ({ed['other_to_strategic']} the reverse), so theme shares are likely conservative.
6. **House style:** company identity is {probe['masked']:.0%} predictable from masked text, so cross-company differences may partly reflect writing style.
7. **Benchmark:** shares include training passages; YoY intervals ignore labelling error.

### Text generation
An optional LLM (Gemini) writes brief prose from the classifier's computed figures and answers Ask questions from passages found by hybrid retrieval (BM25 + MiniLM embeddings, rank fusion). It does not classify passages or produce metrics; numbers it writes are checked against the data, and Ask refuses when nothing relevant is retrieved.

### App
Analysis: Overview · Briefs · Ask. Data: Upload. Model: How it works (pipeline diagram and inspector) · Evaluation · Model card.

### Not for
Investment decisions, valuation, ranking companies, inferring actual spend, or other industries without re-validation.
"""
    MODEL_CARD.write_text(card, encoding="utf-8")
    print(f"wrote {MODEL_CARD.name} ({len(card.split())} words)")


if __name__ == "__main__":
    main()
