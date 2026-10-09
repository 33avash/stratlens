"""Pipeline trace for one piece of text: every intermediate the app's classifier and specificity score produce.

Used by the How it works page (pipeline inspector). It calls the same functions as the corpus build and the
trained model, so what it shows is what the pipeline does, not a re-implementation.
"""
from __future__ import annotations

import numpy as np

from . import specificity
from .config import MIN_WORDS, REVIEW_THRESHOLD
from .masking import display, mask
from .preprocess import (NUMERIC_PASSAGE_SHARE, STATUTORY_RE, block_drop_reason, clean_block, numeric_share,
                         pack_sentences, split_sentences)


def passage_filter(text: str) -> str | None:
    """Why the corpus build would drop this passage, or None if it would be kept."""
    if len(text.split()) < MIN_WORDS:
        return f"short (< {MIN_WORDS} words)"
    if STATUTORY_RE.search(text):
        return "statutory boilerplate"
    if numeric_share(text) > NUMERIC_PASSAGE_SHARE:
        return "mostly numbers (table text)"
    return None


def segment_text(raw: str) -> tuple[str, list[str], str | None]:
    """Clean a pasted block like a PDF block, then split it into passages. Returns (cleaned, passages, block drop)."""
    cleaned = clean_block(raw, set())
    reason = block_drop_reason(cleaned)
    if reason in ("table", "empty"):
        return cleaned, [], reason
    return cleaned, pack_sentences(split_sentences(cleaned)), None


def trace(model, text: str, threshold: float = REVIEW_THRESHOLD, k: int = 10, use_cache: bool = True) -> dict:
    """All steps for one passage. `model` is the saved mask -> TF-IDF -> logistic regression pipeline."""
    masked = mask(text)
    vec, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    x = vec.transform([masked])
    proba = clf.predict_proba(x)[0]
    order = np.argsort(proba)[::-1]
    pred_i = int(order[0])
    vocab = vec.get_feature_names_out()
    feats = sorted(({"term": display(str(vocab[j])), "tfidf": round(float(w), 4),
                     "coef": round(float(clf.coef_[pred_i, j]), 3),
                     "contribution": round(float(w * clf.coef_[pred_i, j]), 4)}
                    for j, w in zip(x.indices, x.data)), key=lambda d: -d["tfidf"])
    spec = specificity.score_texts([text], use_cache=use_cache)[0]
    conf = float(proba[pred_i])
    return {
        "masked": display(masked),
        "n_features": int(x.nnz),
        "features": feats[:k],
        "contributions": sorted(feats, key=lambda d: -abs(d["contribution"]))[:k],
        "probs": [{"label": str(clf.classes_[i]), "prob": round(float(proba[i]), 4)} for i in order],
        "pred": str(clf.classes_[pred_i]),
        "confidence": round(conf, 4),
        "low_confidence": conf < threshold,
        "threshold": threshold,
        "filter": passage_filter(text),
        "specificity": {"score": spec["specificity"], "tier": spec["spec_tier"], "signals": spec["spec_signals"],
                        "summary": specificity.describe(spec["spec_signals"])},
    }
