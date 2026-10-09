"""Explainability for the linear model.

Global: largest positive coefficients per class.
Local: per-term contribution to a class score = tf-idf weight x class coefficient.
These show statistical association with a label, not causation.
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.pipeline import Pipeline


def _parts(model: Pipeline):
    return model.named_steps["tfidf"], model.named_steps["clf"]


def global_terms(model: Pipeline, k: int = 15) -> dict[str, list[dict]]:
    vec, clf = _parts(model)
    vocab = vec.get_feature_names_out()
    out = {}
    for i, c in enumerate(clf.classes_):
        idx = np.argsort(clf.coef_[i])[::-1][:k]
        out[str(c)] = [{"term": str(vocab[j]), "weight": round(float(clf.coef_[i, j]), 3)} for j in idx]
    return out


def explain(model: Pipeline, text: str, k: int = 10) -> dict:
    """Class probabilities plus the terms pushing toward / away from the predicted class."""
    vec, clf = _parts(model)
    pre = model.named_steps.get("mask")
    x = vec.transform(pre.transform([text]) if pre is not None else [text])
    proba = model.predict_proba([text])[0]
    order = np.argsort(proba)[::-1]
    pred_i = int(order[0])
    vocab = vec.get_feature_names_out()
    cols = x.indices
    contrib = x.data * clf.coef_[pred_i, cols]
    terms = sorted(
        ({"term": str(vocab[j]), "contribution": round(float(v), 4)} for j, v in zip(cols, contrib)),
        key=lambda d: d["contribution"], reverse=True,
    )
    return {
        "pred": str(clf.classes_[pred_i]),
        "confidence": round(float(proba[pred_i]), 4),
        "top_classes": [{"label": str(clf.classes_[i]), "prob": round(float(proba[i]), 4)} for i in order[:3]],
        "positive": [t for t in terms if t["contribution"] > 0][:k],
        "negative": sorted([t for t in terms if t["contribution"] < 0], key=lambda d: d["contribution"])[:k],
    }


def highlight_spans(text: str, terms: list[str]) -> list[tuple[int, int]]:
    """Character spans in `text` matching the given (possibly bigram) terms, case-insensitive, non-overlapping."""
    spans = []
    for term in sorted(set(terms), key=len, reverse=True):
        if "__" in term:  # masking placeholders (__brand__, __num__) have no literal span in the original text
            continue
        words = [re.escape(w) for w in term.split()]
        pat = re.compile(r"\b" + r"[\s\W]+".join(words) + r"\b", re.I)
        for m in pat.finditer(text):
            if not any(s < m.end() and m.start() < e for s, e in spans):
                spans.append((m.start(), m.end()))
    return sorted(spans)
