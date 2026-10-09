"""Labelling sample, train/test split, human-review sample and agreement statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split

from .config import REVIEW_SAMPLE_SIZE, SEED, TEST_SIZE, THEMES


def stratified_sample(df: pd.DataFrame, n: int, by: list[str], seed: int = SEED) -> pd.DataFrame:
    """Proportional allocation across strata (at least one row per non-empty stratum)."""
    if len(df) <= n:
        return df.copy()
    sizes = df.groupby(by).size()
    alloc = np.maximum(1, np.floor(sizes / sizes.sum() * n)).astype(int)
    # distribute remainder to the largest strata
    for key in (sizes - alloc).sort_values(ascending=False).index[: max(0, n - alloc.sum())]:
        alloc[key] += 1
    multi = isinstance(alloc.index, pd.MultiIndex)
    parts = [g.sample(min(len(g), alloc[k if multi else k[0]]), random_state=seed) for k, g in df.groupby(by)]
    return pd.concat(parts).sort_index()


def load_labels(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    bad = set(df["label"]) - set(THEMES)
    if bad:
        raise ValueError(f"Unknown labels in {path}: {sorted(bad)}")
    return df


def make_split(labelled: pd.DataFrame, seed: int = SEED, test_size: float = TEST_SIZE):
    train, test = train_test_split(labelled, test_size=test_size, stratify=labelled["label"], random_state=seed)
    return train.sort_index(), test.sort_index()


def review_sample(labelled: pd.DataFrame, n: int = REVIEW_SAMPLE_SIZE, seed: int = SEED) -> pd.DataFrame:
    """Blind review sample stratified by AI label x company. The AI label is NOT included.

    Classes get near-equal quotas (n / number of classes) so that per-class agreement is estimable
    for thin classes; within a class, rows are drawn proportionally by company. Unused quota is
    filled at random from the remaining rows.
    """
    quota = n // labelled["label"].nunique()
    parts = [stratified_sample(g, min(quota, len(g)), ["company"], seed) for _, g in labelled.groupby("label")]
    s = pd.concat(parts)
    if len(s) < n:
        rest = labelled.drop(s.index)
        s = pd.concat([s, rest.sample(min(n - len(s), len(rest)), random_state=seed)])
    out = s[["passage_id", "company", "year", "section", "page", "text"]].copy()
    out["human_label"] = ""
    out["reviewer_note"] = ""
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle so labels are not grouped


def agreement(human: pd.Series, ai: pd.Series) -> dict:
    """Percent agreement, Cohen's kappa and per-class agreement on rows where a human label exists."""
    mask = human.fillna("").astype(str).str.strip().ne("")
    h = human[mask].str.strip().str.upper()
    a = ai[mask]
    n = int(mask.sum())
    if n == 0:
        return {"n": 0}
    per_class = {}
    for t in THEMES:
        rows = a == t
        if rows.any():
            per_class[t] = {"n_ai": int(rows.sum()), "agree": round(float((h[rows] == t).mean()), 3)}
    return {
        "n": n,
        "percent_agreement": round(float((h == a).mean()), 4),
        "cohen_kappa": round(float(cohen_kappa_score(h, a, labels=THEMES)), 4),
        "per_class": per_class,
        "invalid_labels": sorted(set(h) - set(THEMES)),
    }
