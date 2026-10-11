"""Bootstrap confidence intervals.

- Metric CIs resample held-out test passages (with replacement) and recompute the metric.
- YoY CIs resample passages within each report and recompute each theme's share of strategic passages.
  They capture how much a share depends on which passages a report happens to contain; they do not
  capture labelling or model error.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .config import SEED, STRATEGIC_THEMES, THEMES

N_BOOT = 1000


def metric_ci(y_true, y_pred, n: int = N_BOOT, seed: int = SEED, alpha: float = 0.05) -> tuple[float, float]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y_true), size=(n, len(y_true)))
    vals = [f1_score(y_true[i], y_pred[i], labels=THEMES, average="macro", zero_division=0) for i in idx]
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return round(float(lo), 4), round(float(hi), 4)


def paired_diff_ci(y_true, pred_a, pred_b, n: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    """Macro-F1(a) - macro-F1(b) on the same resampled test passages: (point, lo, hi)."""
    y, a, b = map(np.asarray, (y_true, pred_a, pred_b))
    f = lambda yy, pp: f1_score(yy, pp, labels=THEMES, average="macro", zero_division=0)  # noqa: E731
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    diffs = [f(y[i], a[i]) - f(y[i], b[i]) for i in idx]
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return round(f(y, a) - f(y, b), 4), round(float(lo), 4), round(float(hi), 4)


def _share_samples(labels: np.ndarray, n: int, rng) -> np.ndarray:
    """(n, len(STRATEGIC_THEMES)) bootstrap shares of strategic passages."""
    codes = np.array([THEMES.index(x) for x in labels])
    idx = rng.integers(0, len(codes), size=(n, len(codes)))
    counts = np.stack([np.bincount(codes[i], minlength=len(THEMES)) for i in idx])
    strat = counts[:, [THEMES.index(t) for t in STRATEGIC_THEMES]]
    tot = strat.sum(axis=1, keepdims=True)
    return np.divide(strat, tot, out=np.zeros_like(strat, dtype=float), where=tot > 0)


def yoy_ci(preds: pd.DataFrame, yoy: pd.DataFrame, label_col: str = "pred", n: int = N_BOOT,
           seed: int = SEED, exclude_review: bool = False) -> pd.DataFrame:
    """Add ci_lo / ci_hi (pp) and `significant` (CI excludes zero) to a compare.yoy_change() frame."""
    if yoy.empty:
        return yoy.assign(ci_lo=[], ci_hi=[], significant=[])
    df = preds[~preds["needs_review"].astype(bool)] if exclude_review and "needs_review" in preds else preds
    rng = np.random.default_rng(seed)
    out = []
    for co, g in yoy.groupby("company", sort=False):
        y0, y1 = g["year_from"].iloc[0], g["year_to"].iloc[0]
        s0 = _share_samples(df[(df.company == co) & (df.year == y0)][label_col].to_numpy(), n, rng)
        s1 = _share_samples(df[(df.company == co) & (df.year == y1)][label_col].to_numpy(), n, rng)
        diff = (s1 - s0) * 100
        lo, hi = np.quantile(diff, [0.025, 0.975], axis=0)
        ci = pd.DataFrame({"theme": STRATEGIC_THEMES, "ci_lo": lo, "ci_hi": hi})
        out.append(g.merge(ci, on="theme", how="left"))
    res = pd.concat(out, ignore_index=True)
    res["significant"] = (res["ci_lo"] > 0) | (res["ci_hi"] < 0)
    return res


def report_seed(company: str, year: str, seed: int = SEED) -> int:
    """Stable per-report seed, so a report's bootstrap samples are the same wherever they are used."""
    return seed + sum(ord(ch) * (i + 1) for i, ch in enumerate(f"{company}|{year}"))


def report_share_samples(labels, company: str, year: str, n: int = N_BOOT) -> np.ndarray:
    """(n, 6) bootstrap shares of one report's strategic passages, resampling its passages."""
    return _share_samples(np.asarray(labels), n, np.random.default_rng(report_seed(company, year)))


def diff_ci(samples_a: np.ndarray, samples_b: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """95% interval of share(B) - share(A) in percentage points, per strategic theme."""
    lo, hi = np.quantile((samples_b - samples_a) * 100, [alpha / 2, 1 - alpha / 2], axis=0)
    return lo, hi
