"""Labelling sample and train/test split."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import SEED, TEST_SIZE, THEMES


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
