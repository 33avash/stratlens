"""Human audit: export passages for hand-labelling, then score the model and the AI-drafted labels against them.

  python -m stratlens.audit --export     write the two CSVs below (never overwrites a file that has labels in it)
  python -m stratlens.audit              compute data/processed/audit.json once every human_label is filled
  python -m stratlens.audit --partial    same, on the rows filled so far (marked partial; for progress checks)

Files (UTF-8 with BOM so Excel shows ₹ and curly quotes):
  data/labels/verify_new_companies.csv  30 passages per company that was never in training, stratified by the
                                        model's predicted theme (equal allocation, so every theme gets checked)
  data/labels/verify_test_set.csv       100 passages from the original held-out test set, stratified by the
                                        AI-drafted label, with that label and the model's prediction

human_label accepts a theme code (PROD, DIST, DIGI, OPS, SUST, BRAND, OTHER) or full theme name, any case.

Because the new-company sample is stratified by *predicted* theme, plain accuracy over-weights rare predictions.
`weighted_accuracy` re-weights each row by (passages with that prediction in the company) / (sampled rows with that
prediction), estimating accuracy over all of the company's passages. Macro-F1 on a subset averages only the themes
that occur in it (in the human or the model labels), so a theme absent from a small sample does not count as 0.
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from .config import (COMPANY_NAMES, LABELS, LABELS_CSV, PASSAGES_CSV, PROCESSED, SEED, SMALL_N, TEST_CSV,
                     THEME_NAMES, THEMES)

VERIFY_NEW_CSV = LABELS / "verify_new_companies.csv"
VERIFY_TEST_CSV = LABELS / "verify_test_set.csv"
AUDIT_JSON = PROCESSED / "audit.json"
PER_NEW_COMPANY = 30
N_TEST = 100
N_BOOT = 1000

NEW_COLS = ["passage_id", "company", "year", "page", "text", "model_label", "model_confidence", "human_label", "notes"]
TEST_COLS = ["passage_id", "company", "year", "page", "text", "ai_drafted_label", "model_label", "model_confidence",
             "human_label", "notes"]


# ---------------------------------------------------------------- export
def unseen_companies(passages: pd.DataFrame, labels: pd.DataFrame) -> list[str]:
    """Companies with passages in the app but none in the training labels."""
    return sorted(set(passages.company) - set(labels.company))


def equal_allocation(sizes: pd.Series, n: int) -> pd.Series:
    """Spread n as evenly as possible over strata, never taking more than a stratum has."""
    alloc = pd.Series(0, index=sizes.index)
    left = min(n, int(sizes.sum()))
    while left > 0:
        open_ = sizes[alloc < sizes].index
        share = max(1, left // len(open_))
        for k in sorted(open_, key=lambda k: -(sizes[k] - alloc[k])):
            take = min(share, sizes[k] - alloc[k], left)
            alloc[k] += take
            left -= take
            if left == 0:
                break
    return alloc


def sample_new(passages: pd.DataFrame, companies: list[str], n: int = PER_NEW_COMPANY, seed: int = SEED) -> pd.DataFrame:
    parts = []
    for c in companies:
        g = passages[passages.company == c]
        alloc = equal_allocation(g.groupby("pred").size(), n)
        for theme, k in alloc.items():
            if k:
                parts.append(g[g.pred == theme].sample(int(k), random_state=seed))
    out = pd.concat(parts).sample(frac=1, random_state=seed)  # shuffle so rows are not grouped by prediction
    return out.assign(model_label=out.pred, model_confidence=out.confidence.round(2), human_label="", notes="")[NEW_COLS]


def sample_test(test: pd.DataFrame, passages: pd.DataFrame, n: int = N_TEST, seed: int = SEED) -> pd.DataFrame:
    from .dataset import stratified_sample
    t = test.merge(passages[["passage_id", "pred", "confidence"]], on="passage_id", how="left")
    if t.pred.isna().any():
        raise RuntimeError("Some test passages are missing from passages.csv; run python -m stratlens.build first.")
    s = stratified_sample(t, n, by=["label"], seed=seed).sample(frac=1, random_state=seed)
    return s.assign(ai_drafted_label=s.label, model_label=s.pred, model_confidence=s.confidence.round(2),
                    human_label="", notes="")[TEST_COLS]


def _has_labels(path) -> bool:
    if not path.exists():
        return False
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    return (df.get("human_label", pd.Series(dtype=str)).str.strip() != "").any()


def export(force: bool = False) -> int:
    passages = pd.read_csv(PASSAGES_CSV)
    labels = pd.read_csv(LABELS_CSV)
    new = unseen_companies(passages, labels)
    for path, make in [(VERIFY_NEW_CSV, lambda: sample_new(passages, new)),
                       (VERIFY_TEST_CSV, lambda: sample_test(pd.read_csv(TEST_CSV), passages))]:
        if _has_labels(path):
            print(f"KEPT {path.relative_to(LABELS.parent.parent)}: it already has human labels; not overwriting.")
            continue
        if path.exists() and not force:
            print(f"KEPT {path.relative_to(LABELS.parent.parent)}: exists (empty). Use --force to redraw.")
            continue
        df = make()
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"wrote {path.relative_to(LABELS.parent.parent)}: {len(df)} rows")
        by = "model_label" if "ai_drafted_label" not in df else "ai_drafted_label"
        print(df.groupby(["company", by]).size().unstack(fill_value=0).to_string(), "\n")
    print(f"Unseen companies: {', '.join(new)}. Fill human_label (theme code or name), then run "
          "python -m stratlens.audit")
    return 0


# ---------------------------------------------------------------- scoring
_NAME_TO_CODE = {v.lower(): k for k, v in THEME_NAMES.items()} | {k.lower(): k for k in THEMES}


def normalise_label(x) -> str | None:
    if not isinstance(x, str) or not x.strip():
        return None
    key = x.strip().lower()
    if key in _NAME_TO_CODE:
        return _NAME_TO_CODE[key]
    raise ValueError(f"unknown label {x!r} (use one of {', '.join(THEMES)} or a full theme name)")


def macro_f1(y, p) -> float:
    labs = sorted(set(y) | set(p))
    return float(f1_score(y, p, labels=labs, average="macro", zero_division=0))


def boot_ci(y, p, fn, n: int = N_BOOT, seed: int = SEED, w=None) -> list[float] | None:
    y, p = np.asarray(y), np.asarray(p)
    if len(y) < 2:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        vals.append(fn(y[i], p[i]) if w is None else fn(y[i], p[i], np.asarray(w)[i]))
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return [round(float(lo), 4), round(float(hi), 4)]


def _kappa(a, b) -> float:
    return 0.0 if len(set(a) | set(b)) < 2 else float(cohen_kappa_score(a, b))


def _wacc(y, p, w) -> float:
    return float(np.sum(w * (y == p)) / np.sum(w))


def score(y, p, w=None) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    out = {"n": int(len(y)), "accuracy": round(float(accuracy_score(y, p)), 4), "macro_f1": round(macro_f1(y, p), 4),
           "macro_f1_ci": boot_ci(y, p, macro_f1), "accuracy_ci": boot_ci(y, p, lambda a, b: float((a == b).mean()))}
    if w is not None:
        out["weighted_accuracy"] = round(_wacc(y, p, np.asarray(w)), 4)
        out["weighted_accuracy_ci"] = boot_ci(y, p, _wacc, w=w)
    out["too_small"] = out["n"] < SMALL_N
    return out


def per_class_f1(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    labs = [t for t in THEMES if t in set(y) | set(p)]
    f = f1_score(y, p, labels=labs, average=None, zero_division=0)
    return {t: {"f1": round(float(v), 4), "support": int((y == t).sum())} for t, v in zip(labs, f)}


def _confusion(y, p) -> dict:
    return {"labels": THEMES, "matrix": confusion_matrix(y, p, labels=THEMES).tolist()}


def _load_filled(path, partial: bool) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run python -m stratlens.audit --export first.")
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"human_label": str, "notes": str}, keep_default_na=False)
    bad = []
    for i, x in df.human_label.items():
        try:
            df.at[i, "human_label"] = normalise_label(x) or ""
        except ValueError as exc:
            bad.append(f"row {i + 2}: {exc}")
    if bad:
        raise ValueError(f"{path.name}:\n  " + "\n  ".join(bad[:10]))
    blank = int((df.human_label == "").sum())
    if blank and not partial:
        raise ValueError(f"{path.name}: {blank} of {len(df)} rows have no human_label yet. Fill them, or run with "
                         "--partial to check progress.")
    return df[df.human_label != ""].reset_index(drop=True)


def stratum_weights(sample: pd.DataFrame, passages: pd.DataFrame) -> np.ndarray:
    """Inverse sampling probability: passages with that prediction in the company / sampled rows with it."""
    pop = passages.groupby(["company", "pred"]).size()
    drawn = sample.groupby(["company", "model_label"]).size()
    return np.array([pop[(c, m)] / drawn[(c, m)] for c, m in zip(sample.company, sample.model_label)], dtype=float)


def compute(new: pd.DataFrame, test: pd.DataFrame, passages: pd.DataFrame, partial: bool = False) -> dict:
    out = {"generated": date.today().isoformat(), "partial": bool(partial),
           "label_source": "human labels by the project author (data/labels/verify_*.csv)",
           "n_labelled": {"new_companies": int(len(new)), "test_set": int(len(test))}}

    if len(new):
        w = stratum_weights(new, passages)
        un = score(new.human_label, new.model_label, w)
        un["per_class"] = per_class_f1(new.human_label, new.model_label)
        un["confusion"] = _confusion(new.human_label, new.model_label)
        un["companies"] = sorted(new.company.unique())
        out["unseen_companies"] = un

    if len(test):
        t = score(test.human_label, test.model_label)
        t["per_class"] = per_class_f1(test.human_label, test.model_label)
        t["confusion"] = _confusion(test.human_label, test.model_label)
        out["test_set"] = t
        y, a = test.human_label.to_numpy(), test.ai_drafted_label.to_numpy()
        out["ai_vs_human"] = {"n": int(len(y)), "agreement": round(float((y == a).mean()), 4),
                              "kappa": round(_kappa(y, a), 4), "kappa_ci": boot_ci(y, a, _kappa),
                              "confusion": _confusion(y, a)}
        # The original evaluation scored the model against AI drafts; on the same rows, for comparison.
        out["test_set_vs_ai_drafts"] = score(test.ai_drafted_label, test.model_label)

    per = []
    for df, src in [(test, "test set"), (new, "unseen company sample")]:
        for c, g in df.groupby("company"):
            s = score(g.human_label, g.model_label)
            per.append({"company": c, "name": COMPANY_NAMES.get(c, c), "source": src, **s})
    out["per_company"] = sorted(per, key=lambda r: r["company"])
    return out


def run(partial: bool = False) -> int:
    try:
        new = _load_filled(VERIFY_NEW_CSV, partial)
        test = _load_filled(VERIFY_TEST_CSV, partial)
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        return 1
    if new.empty and test.empty:
        print("No human labels yet.")
        return 1
    res = compute(new, test, pd.read_csv(PASSAGES_CSV), partial)
    AUDIT_JSON.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print_summary(res)
    print(f"\nwrote {AUDIT_JSON}" + (" (PARTIAL: not every row is labelled)" if partial else ""))
    return 0


def _fmt(s: dict | None, key: str) -> str:
    if not s or s.get(key) is None:
        return "–"
    ci = s.get(f"{key}_ci")
    return f"{s[key]:.2f}" + (f" [{ci[0]:.2f}–{ci[1]:.2f}]" if ci else "")


def print_summary(r: dict) -> None:
    un, t, k = r.get("unseen_companies"), r.get("test_set"), r.get("ai_vs_human")
    rows = [("Unseen companies (model vs you)", un), ("Original test set (model vs you)", t),
            ("Original test set (model vs AI drafts)", r.get("test_set_vs_ai_drafts"))]
    print(f"{'':42}{'n':>5}  {'accuracy':<20}{'macro-F1':<20}")
    for name, s in rows:
        if s:
            print(f"{name:42}{s['n']:>5}  {_fmt(s, 'accuracy'):<20}{_fmt(s, 'macro_f1'):<20}")
    if un and "weighted_accuracy" in un:
        print(f"{'  unseen, re-weighted to all passages':42}{'':>5}  {_fmt(un, 'weighted_accuracy')}")
    if k:
        print(f"\nAI drafts vs you: {k['agreement']:.0%} agree, Cohen's kappa {_fmt(k, 'kappa')}")
    print("\nPer company (model vs you):")
    for p in r["per_company"]:
        print(f"  {p['name']:<28}{p['source']:<24}n={p['n']:<4} macro-F1 {_fmt(p, 'macro_f1')}"
              + ("  (small n)" if p["too_small"] else ""))


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="python -m stratlens.audit", description=__doc__.split("\n")[0])
    ap.add_argument("--export", action="store_true", help="write the label files for hand-labelling")
    ap.add_argument("--force", action="store_true", help="with --export: redraw files that have no labels yet")
    ap.add_argument("--partial", action="store_true", help="score the rows labelled so far")
    a = ap.parse_args(argv)
    return export(a.force) if a.export else run(a.partial)
