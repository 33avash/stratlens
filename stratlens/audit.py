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


def largest_gap(subs: list[dict]) -> tuple | None:
    """(gap, dimension, weaker group, stronger group) among subgroups with n >= SMALL_N."""
    best = None
    for dim in SUBGROUP_DIMENSIONS.values():
        eligible = [s for s in subs if s["dimension"] == dim and not s["too_small"]]
        if len(eligible) >= 2:
            lo = min(eligible, key=lambda s: s["macro_f1"])
            hi = max(eligible, key=lambda s: s["macro_f1"])
            gap = hi["macro_f1"] - lo["macro_f1"]
            if best is None or gap > best[0]:
                best = (gap, dim, lo, hi)
    return best


def _ci(s: dict) -> str:
    ci = s.get("macro_f1_ci")
    return f" [{ci[0]:.2f}–{ci[1]:.2f}]" if ci else ""


def _name(group: str) -> str:
    from .config import COMPANY_NAMES
    return COMPANY_NAMES.get(group, group)


def _overlap(a: dict, b: dict) -> bool:
    ca, cb = a.get("macro_f1_ci"), b.get("macro_f1_ci")
    if not ca or not cb:
        return True
    lo, hi = (a, b) if a["macro_f1"] <= b["macro_f1"] else (b, a)
    return lo["macro_f1_ci"][1] >= hi["macro_f1_ci"][0]


def clearest_gap(subs: list[dict]) -> tuple | None:
    """Largest within-dimension gap whose 95% CIs do not overlap (any n), else None."""
    best = None
    for dim in SUBGROUP_DIMENSIONS.values():
        groups = [s for s in subs if s["dimension"] == dim]
        for i, a in enumerate(groups):
            for b in groups[i + 1:]:
                if not _overlap(a, b):
                    lo, hi = (a, b) if a["macro_f1"] < b["macro_f1"] else (b, a)
                    gap = hi["macro_f1"] - lo["macro_f1"]
                    if best is None or gap > best[0]:
                        best = (gap, dim, lo, hi)
    return best


def error_direction(cm: dict) -> dict:
    """Split test errors into strategic->OTHER (missed), OTHER->strategic (invented) and strategic->strategic."""
    m, labs = np.array(cm["matrix"]), cm["labels"]
    o = labs.index("OTHER")
    total = int(m.sum() - np.trace(m))
    missed = int(m[:, o].sum() - m[o, o])
    invented = int(m[o].sum() - m[o, o])
    per_theme = {t: {"to_other": int(m[i, o]), "support": int(m[i].sum())} for i, t in enumerate(labs) if t != "OTHER"}
    return {"total": total, "strategic_to_other": missed, "other_to_strategic": invented,
            "between_themes": total - missed - invented, "per_theme": per_theme}


def key_finding(m: dict) -> str:
    """Plain-English key audit finding assembled from metrics.json values (no hand-written numbers).

    Order: (a) masking and house style, (b) direction of errors, (c) the subgroup gap, flagged when n is small."""
    parts = []
    mk = m.get("masking")
    if mk:
        p = mk["company_probe"]
        change = ("does not change theme accuracy" if abs(mk["diff"]) < 0.005 else
                  f"changes theme macro-F1 by {mk['diff']:+.2f}")
        parts.append(f"Masking company names, brands and numbers {change} (paired CI {mk['diff_ci'][0]:+.2f} to "
                     f"{mk['diff_ci'][1]:+.2f}), but company identity is still {p['masked']:.0%} predictable from "
                     f"masked text (majority guess {p['majority_baseline']:.0%}), so house style may confound "
                     "cross-company comparisons.")
    ed = error_direction(m["test"]["confusion"])
    if ed["total"]:
        lead = ("Most errors are" if ed["strategic_to_other"] * 2 > ed["total"] else "The largest group of errors is")
        parts.append(f"{lead} strategic passages predicted as Other ({ed['strategic_to_other']} of {ed['total']}; "
                     f"the reverse happens {ed['other_to_strategic']} times), so strategy passages are missed more "
                     "often than invented and theme shares are likely conservative.")
    clear = clearest_gap(m["subgroups"])
    if clear:
        g, dim, lo, hi = clear
        small = (f" Only n = {lo['n']} test passages come from {_name(lo['group'])}, so treat this as indicative."
                 if lo["too_small"] else "")
        parts.append(f"By {dim.lower()}, {_name(lo['group'])} scores macro-F1 {lo['macro_f1']:.2f}{_ci(lo)} against "
                     f"{hi['macro_f1']:.2f}{_ci(hi)} for {_name(hi['group'])}.{small}")
    return " ".join(parts)
