"""TF-IDF + logistic regression classifier: training, persistence and inference."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from .config import MODEL_PATH, REVIEW_THRESHOLD, SEED

C_GRID = [0.1, 0.3, 1, 3, 10, 30, 100, 300]
CV_FOLDS = 5


def build_pipeline(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2,
                                  stop_words="english", strip_accents="unicode")),
        ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=SEED)),
    ])


def tune(texts: pd.Series, labels: pd.Series) -> GridSearchCV:
    """Choose C by stratified k-fold CV on the training set only (macro-F1)."""
    search = GridSearchCV(
        build_pipeline(), {"clf__C": C_GRID}, scoring="f1_macro",
        cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED), n_jobs=1,
    )
    search.fit(texts, labels)
    return search


def save(model: Pipeline, meta: dict, path=MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "meta": meta}, path)


def load(path=MODEL_PATH) -> tuple[Pipeline, dict]:
    bundle = joblib.load(path)
    return bundle["model"], bundle["meta"]


def predict_frame(model: Pipeline, texts: pd.Series | list[str], threshold: float = REVIEW_THRESHOLD) -> pd.DataFrame:
    """One row per text: predicted label, confidence (top-class probability), review flag and all class probs."""
    texts = list(texts)
    if not texts:
        cols = ["pred", "confidence", "needs_review"] + [f"p_{c}" for c in model.classes_]
        return pd.DataFrame(columns=cols)
    proba = model.predict_proba(texts)
    classes = model.classes_
    top = proba.argmax(axis=1)
    out = pd.DataFrame({
        "pred": classes[top],
        "confidence": proba[np.arange(len(texts)), top].round(4),
    })
    out["needs_review"] = out["confidence"] < threshold
    for j, c in enumerate(classes):
        out[f"p_{c}"] = proba[:, j].round(4)
    return out
