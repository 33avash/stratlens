"""Audit metrics: overall, per-class, subgroup, threshold analysis and the key finding."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from .config import SMALL_N, THEMES

SHORT_WORDS = 25


def overall(y_true, y_pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=THEMES, average="macro", zero_division=0)
    return {"n": int(len(y_true)), "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "macro_precision": round(float(p), 4), "macro_recall": round(float(r), 4), "macro_f1": round(float(f), 4)}


def per_class(y_true, y_pred) -> dict:
    rep = classification_report(y_true, y_pred, labels=THEMES, output_dict=True, zero_division=0)
    return {t: {"precision": round(rep[t]["precision"], 4), "recall": round(rep[t]["recall"], 4),
                "f1": round(rep[t]["f1-score"], 4), "support": int(rep[t]["support"])} for t in THEMES}


def confusion(y_true, y_pred) -> dict:
    return {"labels": THEMES, "matrix": confusion_matrix(y_true, y_pred, labels=THEMES).tolist()}


def add_subgroup_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["length_group"] = np.where(df["word_count"] < SHORT_WORDS, f"Short (<{SHORT_WORDS} words)",
                                  f"Long (>={SHORT_WORDS} words)")
    df["style_group"] = df["section"].map({"LETTER": "Chairman/CEO letter", "MDA": "MD&A"}).fillna(df["section"])
    df["company_group"] = df["company"]
    return df


SUBGROUP_DIMENSIONS = {
    "length_group": "Passage length",
    "style_group": "Writing style (section)",
    "company_group": "Company",
}


def subgroups(df: pd.DataFrame, true_col: str = "label", pred_col: str = "pred") -> list[dict]:
    rows = []
    for col, dim in SUBGROUP_DIMENSIONS.items():
        for g, part in df.groupby(col):
            m = overall(part[true_col], part[pred_col])
            rows.append({"dimension": dim, "group": str(g), "n": m["n"], "accuracy": m["accuracy"],
                         "macro_f1": m["macro_f1"], "macro_precision": m["macro_precision"],
                         "macro_recall": m["macro_recall"], "too_small": m["n"] < SMALL_N})
    return rows


def threshold_analysis(conf: pd.Series, correct: pd.Series, threshold: float) -> dict:
    above = conf >= threshold
    return {
        "threshold": threshold,
        "share_flagged": round(float((~above).mean()), 4),
        "n_flagged": int((~above).sum()),
        "accuracy_above": round(float(correct[above].mean()), 4) if above.any() else None,
        "accuracy_below": round(float(correct[~above].mean()), 4) if (~above).any() else None,
    }


def most_confused(cm: dict) -> dict | None:
    m = np.array(cm["matrix"])
    np.fill_diagonal(m, 0)
    if m.sum() == 0:
        return None
    i, j = np.unravel_index(m.argmax(), m.shape)
    labs = cm["labels"]
    support = int(np.array(cm["matrix"])[i].sum())
    return {"true": labs[i], "pred": labs[j], "count": int(m[i, j]), "true_support": support}


def key_finding(model_m: dict, base_m: dict, subs: list[dict], confused: dict | None) -> str:
    """Plain-English finding assembled from the numbers (no hand-written values)."""
    parts = [f"On {model_m['n']} held-out passages the model reaches macro-F1 {model_m['macro_f1']:.2f} "
             f"versus {base_m['macro_f1']:.2f} for the keyword baseline."]
    best_gap = None
    for dim in SUBGROUP_DIMENSIONS.values():
        eligible = [s for s in subs if s["dimension"] == dim and not s["too_small"]]
        if len(eligible) >= 2:
            lo = min(eligible, key=lambda s: s["macro_f1"])
            hi = max(eligible, key=lambda s: s["macro_f1"])
            gap = hi["macro_f1"] - lo["macro_f1"]
            if best_gap is None or gap > best_gap[0]:
                best_gap = (gap, dim, lo, hi)
    if best_gap:
        gap, dim, lo, hi = best_gap
        parts.append(f"The largest subgroup gap is by {dim.lower()}: macro-F1 {lo['macro_f1']:.2f} for "
                     f"{lo['group']} (n={lo['n']}) against {hi['macro_f1']:.2f} for {hi['group']} (n={hi['n']}), "
                     f"a gap of {gap:.2f}.")
    small = [f"{s['group']} (n={s['n']})" for s in subs if s["too_small"]]
    if small:
        parts.append("Groups too small to conclude: " + ", ".join(small) + ".")
    if confused:
        parts.append(f"The most frequent error is {confused['true']} passages predicted as {confused['pred']} "
                     f"({confused['count']} of {confused['true_support']}).")
    return " ".join(parts)
