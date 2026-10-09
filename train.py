"""Train, evaluate and save the StratLens theme classifier.

Steps: stratified 80/20 split (saved to data/splits) -> tune C by 5-fold CV on train only ->
evaluate on the held-out test set -> compare with keyword baseline -> subgroup audit ->
explainability -> cross-year robustness check -> save model, reports/metrics.json and benchmark predictions.
All reported metrics come from the held-out test set unless labelled otherwise.
Usage: python train.py
"""
import json
import sys
from datetime import date

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from stratlens import audit, baseline, explain
from stratlens import model as mdl
from stratlens.config import (ERROR_NOTES_CSV, LABELS_CSV, METRICS_JSON, REPORTS, REVIEW_THRESHOLD, SEED, SMALL_N,
                              SPLITS, TEST_CSV, THEMES, TRAIN_CSV)
from stratlens.dataset import load_labels, make_split

# If the default threshold flags fewer than MIN_FLAG or more than MAX_FLAG of CV predictions,
# it is not a useful review rule; fall back to the CV quantile that flags TARGET_FLAG.
MIN_FLAG, MAX_FLAG, TARGET_FLAG = 0.05, 0.50, 0.20
N_ERROR_EXAMPLES = 8


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
    pred = m.predict(te["text"])
    res = audit.overall(te["label"], pred)
    res.update({"available": True, "train_year": years[0], "test_year": years[-1], "n_train": int(len(tr)),
                "companies_train": sorted(tr.company.unique()), "companies_test": sorted(te.company.unique())})
    return res


def main() -> None:
    labels = load_labels(LABELS_CSV)
    counts = labels["label"].value_counts().reindex(THEMES, fill_value=0)
    print(f"{len(labels)} labelled passages\n{counts.to_string()}")

    train, test = make_split(labels)
    SPLITS.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_CSV, index=False)
    test.to_csv(TEST_CSV, index=False)

    search = mdl.tune(train["text"], train["label"])
    best_C = search.best_params_["clf__C"]
    cv_table = [{"C": p["clf__C"], "cv_macro_f1": round(float(s), 4), "cv_std": round(float(sd), 4)}
                for p, s, sd in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"],
                                    search.cv_results_["std_test_score"])]
    print(f"best C = {best_C} (CV macro-F1 {search.best_score_:.3f})")
    model = search.best_estimator_

    threshold, thr_info = choose_threshold(train, best_C)

    # ---- held-out evaluation
    pf = mdl.predict_frame(model, test["text"], threshold)
    test = test.reset_index(drop=True)
    test = pd.concat([test, pf], axis=1)
    test["baseline"] = baseline.predict(test["text"])
    test["correct"] = test["pred"] == test["label"]
    m_model = audit.overall(test["label"], test["pred"])
    m_base = audit.overall(test["label"], test["baseline"])
    cm = audit.confusion(test["label"], test["pred"])
    test = audit.add_subgroup_columns(test)
    subs = audit.subgroups(test)
    confused = audit.most_confused(cm)
    print(f"test macro-F1 model {m_model['macro_f1']:.3f} vs baseline {m_base['macro_f1']:.3f}")

    # ---- misclassified examples (author notes, if written, are joined from reports/error_notes.csv)
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

    metrics = {
        "generated": date.today().isoformat(),
        "label_source": "ai_drafted (Claude, build-time); human verification pending unless reports/agreement.json exists",
        "dataset": {
            "n_labelled": int(len(labels)), "n_train": int(len(train)), "n_test": int(len(test)),
            "class_counts": {k: int(v) for k, v in counts.items()},
            "thin_classes": [k for k, v in counts.items() if v < 40],
            "by_company_year": {f"{c} {y}": int(n) for (c, y), n in labels.groupby(["company", "year"]).size().items()},
        },
        "model": {"type": "TF-IDF (1-2 grams, sublinear tf, min_df=2) + LogisticRegression (balanced)",
                  "best_C": best_C, "cv_folds": mdl.CV_FOLDS, "cv_results": cv_table,
                  "cv_best_macro_f1": round(float(search.best_score_), 4), "seed": SEED},
        "threshold": {"value": threshold, **thr_info,
                      "test": audit.threshold_analysis(test["confidence"], test["correct"], threshold)},
        "test": {"model": m_model, "baseline": m_base, "per_class": audit.per_class(test["label"], test["pred"]),
                 "baseline_per_class": audit.per_class(test["label"], test["baseline"]), "confusion": cm,
                 "most_confused": confused},
        "subgroups": subs, "small_n": SMALL_N,
        "cross_year": cross_year(labels, best_C),
        "global_terms": explain.global_terms(model, 15),
        "misclassified_examples": err_examples,
        "n_test_errors": int((~test["correct"]).sum()),
        "key_finding": audit.key_finding(m_model, m_base, subs, confused),
    }
    METRICS_JSON.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    mdl.save(model, {"threshold": threshold, "best_C": best_C, "classes": list(model.classes_),
                     "trained": date.today().isoformat(), "n_train": int(len(train))})
    print(f"key finding: {metrics['key_finding']}")
    print(f"saved model and {METRICS_JSON}")

    sys.path.insert(0, "scripts")
    import build_benchmark
    build_benchmark.main()


if __name__ == "__main__":
    main()
