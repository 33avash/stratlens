"""Train, evaluate and save the StratLens theme classifier.

Steps
  1. labels: AI-drafted following docs/labelling_guidelines.md (not human-verified)
  2. stratified 80/20 split (saved to data/splits); all tuning uses the training set only
  3. main model: masked TF-IDF + logistic regression (C by 5-fold CV)
  4. comparisons on the same held-out test set: unmasked TF-IDF (masking test), sentence-embedding + LR,
     keyword baseline; bootstrap 95% CIs and a paired bootstrap for model-vs-model differences
  5. audit: per-class, confusion, subgroups (with CIs), threshold analysis, company-identification probe,
     cross-year check, explainability, misclassified examples, key finding
  6. save model + reports/metrics.json, rebuild benchmark predictions and MODEL_CARD.md
Usage: python train.py
"""
import json
import sys
from datetime import date

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

from stratlens import audit, baseline, embeddings, explain, uncertainty
from stratlens import model as mdl
from stratlens.config import (ERROR_NOTES_CSV, LABELS_CSV, METRICS_JSON, REPORTS, REVIEW_THRESHOLD, SEED,
                              SMALL_N, SPLITS, TEST_CSV, THEMES, TRAIN_CSV)
from stratlens.dataset import load_labels, make_split
from stratlens.masking import mask

MIN_FLAG, MAX_FLAG, TARGET_FLAG = 0.05, 0.50, 0.20
N_ERROR_EXAMPLES = 8
EMB_C_GRID = [0.1, 0.3, 1, 3, 10, 30]
SWITCH_MARGIN = 0.03  # embeddings replace TF-IDF only if better by this much AND the paired CI excludes zero


def choose_threshold(train: pd.DataFrame, C: float) -> tuple[float, dict]:
    proba = cross_val_predict(mdl.build_pipeline(C), train["text"], train["label"],
                              cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")
    conf = proba.max(axis=1)
    classes = np.array(sorted(train["label"].unique()))
    correct = pd.Series(classes[proba.argmax(axis=1)] == train["label"].values)
    flagged = float((conf < REVIEW_THRESHOLD).mean())
    info = {"default": REVIEW_THRESHOLD, "cv_share_flagged_at_default": round(flagged, 4)}
    if MIN_FLAG <= flagged <= MAX_FLAG:
        info["rule"] = "default kept: flags a useful share of cross-validated training predictions"
        thr = REVIEW_THRESHOLD
    else:
        thr = round(float(np.quantile(conf, TARGET_FLAG)), 2)
        info["rule"] = (f"default flagged {flagged:.0%} of CV predictions (outside {MIN_FLAG:.0%}-{MAX_FLAG:.0%}); "
                        f"using the CV {TARGET_FLAG:.0%} confidence quantile instead")
    info["cv_analysis"] = audit.threshold_analysis(pd.Series(conf), correct, thr)
    return thr, info


def cross_year(labels: pd.DataFrame, C: float) -> dict:
    years = sorted(labels["year"].unique())
    if len(years) < 2:
        return {"available": False, "reason": "only one financial year labelled"}
    tr, te = labels[labels.year == years[0]], labels[labels.year == years[-1]]
    if tr["label"].nunique() < len(THEMES):
        return {"available": False, "reason": f"not all classes present in {years[0]}"}
    m = mdl.build_pipeline(C).fit(tr["text"], tr["label"])
    res = audit.overall(te["label"], m.predict(te["text"]))
    res.update({"available": True, "train_year": years[0], "test_year": years[-1], "n_train": int(len(tr)),
                "companies_train": sorted(tr.company.unique()), "companies_test": sorted(te.company.unique())})
    return res


def company_probe(labels: pd.DataFrame) -> dict:
    """How well can the text identify its author? 5-fold CV accuracy of a company classifier."""
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    res = {"majority_baseline": round(float(labels["company"].value_counts(normalize=True).iloc[0]), 4)}
    for name, masked in [("unmasked", False), ("masked", True)]:
        p = mdl.build_pipeline(1.0, masked=masked)
        res[name] = round(float(cross_val_score(p, labels["text"], labels["company"], cv=cv,
                                                scoring="accuracy").mean()), 4)
    return res


def score(name, y, pred, note=""):
    m = audit.overall(y, pred)
    lo, hi = uncertainty.metric_ci(y, pred)
    return {"system": name, **m, "macro_f1_ci": [lo, hi], "note": note}


def main() -> None:
    labels = load_labels(LABELS_CSV)
    counts = labels["label"].value_counts().reindex(THEMES, fill_value=0)
    print(f"{len(labels)} labelled passages (AI-drafted)\n{counts.to_string()}")

    train, test = make_split(labels)
    SPLITS.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_CSV, index=False)
    test.to_csv(TEST_CSV, index=False)

    # ---- main model: masked TF-IDF + LR
    search = mdl.tune(train["text"], train["label"], masked=True)
    best_C = search.best_params_["clf__C"]
    cv_table = [{"C": p["clf__C"], "cv_macro_f1": round(float(s), 4), "cv_std": round(float(sd), 4)}
                for p, s, sd in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"],
                                    search.cv_results_["std_test_score"])]
    model = search.best_estimator_
    print(f"masked TF-IDF: best C = {best_C} (CV macro-F1 {search.best_score_:.3f})")
    threshold, thr_info = choose_threshold(train, best_C)

    # ---- comparison systems (each tuned on train only)
    unmasked = mdl.tune(train["text"], train["label"], masked=False)
    emb_info = {"available": embeddings.available(), "model": embeddings.MODEL_NAME}
    emb_pred = None
    if emb_info["available"]:
        emb_search = mdl.tune(train["text"], train["label"], estimator=embeddings.EmbeddingLR(preprocess=mask),
                              grid={"C": EMB_C_GRID})
        emb_pred = emb_search.best_estimator_.predict(test["text"])
        emb_info.update({"best_C": emb_search.best_params_["C"],
                         "cv_macro_f1": round(float(emb_search.best_score_), 4)})
        print(f"embeddings: best C = {emb_info['best_C']} (CV macro-F1 {emb_search.best_score_:.3f})")

    # ---- held-out evaluation
    pf = mdl.predict_frame(model, test["text"], threshold)
    test = pd.concat([test.reset_index(drop=True), pf], axis=1)
    test["baseline"] = baseline.predict(test["text"])
    test["unmasked_pred"] = unmasked.best_estimator_.predict(test["text"])
    test["correct"] = test["pred"] == test["label"]
    m_model = audit.overall(test["label"], test["pred"])
    m_base = audit.overall(test["label"], test["baseline"])

    comparison = [score("TF-IDF + LR (masked)", test["label"], test["pred"], "main model")]
    if emb_pred is not None:
        test["emb_pred"] = emb_pred
        comparison.append(score("MiniLM embeddings + LR (masked)", test["label"], emb_pred))
    comparison.append(score("TF-IDF + LR (unmasked)", test["label"], test["unmasked_pred"], "masking test"))
    comparison.append(score("Keyword baseline", test["label"], test["baseline"]))

    choice = {"main": "TF-IDF + LR (masked)"}
    if emb_pred is not None:
        d, lo, hi = uncertainty.paired_diff_ci(test["label"], emb_pred, test["pred"])
        choice["embedding_minus_tfidf"] = {"diff": d, "ci": [lo, hi]}
        if d > SWITCH_MARGIN and lo > 0:
            choice["reason"] = (f"Embeddings beat TF-IDF by {d:.2f} macro-F1 (paired 95% CI {lo:+.2f} to {hi:+.2f}). "
                                "TF-IDF is still served in the app because its decisions can be explained term by term.")
        else:
            choice["reason"] = (f"Embeddings differ from TF-IDF by {d:+.2f} macro-F1 (paired 95% CI {lo:+.2f} to "
                                f"{hi:+.2f}), so the scores are close; TF-IDF stays the main model because its "
                                "decisions can be explained term by term.")
    else:
        choice["reason"] = "Embedding model unavailable in this environment; TF-IDF is the main model."

    dm, dlo, dhi = uncertainty.paired_diff_ci(test["label"], test["pred"], test["unmasked_pred"])
    masking = {"masked_macro_f1": comparison[0]["macro_f1"],
               "unmasked_macro_f1": audit.overall(test["label"], test["unmasked_pred"])["macro_f1"],
               "unmasked_best_C": unmasked.best_params_["clf__C"],
               "diff": dm, "diff_ci": [dlo, dhi], "company_probe": company_probe(labels)}
    print(f"masking: masked {masking['masked_macro_f1']:.3f} vs unmasked {masking['unmasked_macro_f1']:.3f}; "
          f"probe {masking['company_probe']}")

    cm = audit.confusion(test["label"], test["pred"])
    test = audit.add_subgroup_columns(test)
    subs = audit.subgroups(test)
    for s in subs:
        col = {v: k for k, v in audit.SUBGROUP_DIMENSIONS.items()}[s["dimension"]]
        part = test[test[col] == s["group"]]
        s["macro_f1_ci"] = list(uncertainty.metric_ci(part["label"], part["pred"]))
    confused = audit.most_confused(cm)

    # ---- misclassified examples (author notes joined from reports/error_notes.csv)
    errors = test[~test["correct"]].sort_values("confidence", ascending=False)
    notes = pd.read_csv(ERROR_NOTES_CSV) if ERROR_NOTES_CSV.exists() else pd.DataFrame(columns=["passage_id", "likely_reason"])
    errors = errors.merge(notes[["passage_id", "likely_reason"]], on="passage_id", how="left")
    noted = errors[errors["likely_reason"].notna()]
    shown = noted if len(noted) >= N_ERROR_EXAMPLES else pd.concat([noted, errors[errors["likely_reason"].isna()]])
    err_examples = [{
        "passage_id": r.passage_id, "company": r.company, "year": r.year, "section": r.section, "page": int(r.page),
        "text": r.text, "true": r.label, "pred": r.pred, "confidence": float(r.confidence),
        "likely_reason": r.likely_reason if isinstance(r.likely_reason, str) else None,
    } for r in shown.head(N_ERROR_EXAMPLES).itertuples()]
    REPORTS.mkdir(parents=True, exist_ok=True)
    errors[["passage_id", "company", "year", "section", "page", "label", "pred", "confidence", "text"]].to_csv(
        REPORTS / "test_errors.csv", index=False)

    m_model_ci = uncertainty.metric_ci(test["label"], test["pred"])
    metrics = {
        "generated": date.today().isoformat(),
        "label_source": ("AI-drafted (Claude, build time) following docs/labelling_guidelines.md; "
                         "not human-verified"),
        "dataset": {
            "n_labelled": int(len(labels)), "n_train": int(len(train)), "n_test": int(len(test)),
            "class_counts": {k: int(v) for k, v in counts.items()},
            "thin_classes": [k for k, v in counts.items() if v < 40],
            "by_company_year": {f"{c} {y}": int(n) for (c, y), n in labels.groupby(["company", "year"]).size().items()},
        },
        "model": {"type": "Masked TF-IDF (1-2 grams, sublinear tf, min_df=2) + LogisticRegression (balanced)",
                  "best_C": best_C, "cv_folds": mdl.CV_FOLDS, "cv_results": cv_table,
                  "cv_best_macro_f1": round(float(search.best_score_), 4), "seed": SEED},
        "comparison": comparison, "model_choice": choice, "embedding": emb_info, "masking": masking,
        "threshold": {"value": threshold, **thr_info,
                      "test": audit.threshold_analysis(test["confidence"], test["correct"], threshold)},
        "test": {"model": {**m_model, "macro_f1_ci": list(m_model_ci)}, "baseline": m_base,
                 "per_class": audit.per_class(test["label"], test["pred"]),
                 "baseline_per_class": audit.per_class(test["label"], test["baseline"]), "confusion": cm,
                 "most_confused": confused, "error_direction": audit.error_direction(cm)},
        "subgroups": subs, "small_n": SMALL_N,
        "cross_year": cross_year(labels, best_C),
        "global_terms": explain.global_terms(model, 15),
        "misclassified_examples": err_examples,
        "n_test_errors": int((~test["correct"]).sum()),
    }
    metrics["key_finding"] = audit.key_finding(metrics)
    METRICS_JSON.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    mdl.save(model, {"threshold": threshold, "best_C": best_C, "classes": list(model.classes_), "masked": True,
                     "trained": date.today().isoformat(), "n_train": int(len(train))})
    print(f"key finding: {metrics['key_finding']}")
    print(f"saved model and {METRICS_JSON}")

    sys.path.insert(0, "scripts")
    import build_benchmark
    import write_model_card
    build_benchmark.main()
    write_model_card.main()


if __name__ == "__main__":
    main()
