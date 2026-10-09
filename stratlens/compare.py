"""Competitive benchmark: normalised theme emphasis, year-over-year change and evidence lookup.

Shares measure disclosure emphasis (how much of a report's strategic narrative is spent on a theme),
not investment or execution.

Definitions used on every page:
  passages            all passages in scope (low-confidence ones removed when that filter is on)
  strategic passages  passages in scope whose predicted theme is not OTHER
  share               theme passages / strategic passages, per company-year
  vs peers            share minus the simple average share of the other companies in the same year (pp)
"""
from __future__ import annotations

import pandas as pd

from .config import STRATEGIC_THEMES

LABEL_COL = "pred"
VS_PEERS_HELP = ("Percentage points above or below the simple average share of the other companies in the same "
                 "year. Shares are theme passages divided by strategic (non-Other) passages.")


def in_scope(preds: pd.DataFrame, exclude_review: bool = False) -> pd.DataFrame:
    """All passages that count, strategic or not: the denominator for 'share of passages'."""
    if exclude_review and "needs_review" in preds:
        return preds[~preds["needs_review"].astype(bool)]
    return preds


def strategic(preds: pd.DataFrame, exclude_review: bool = False) -> pd.DataFrame:
    df = in_scope(preds, exclude_review)
    return df[df[LABEL_COL].isin(STRATEGIC_THEMES)]


def theme_shares(preds: pd.DataFrame, exclude_review: bool = False) -> pd.DataFrame:
    """One row per company x year x theme: count, strategic total, share (0-1) and the passage ids behind it."""
    df = strategic(preds, exclude_review)
    cols = ["company", "year", "theme", "count", "strategic_total", "share", "passage_ids"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for (co, yr), g in df.groupby(["company", "year"]):
        total = len(g)
        for t in STRATEGIC_THEMES:
            ids = g.loc[g[LABEL_COL] == t, "passage_id"].tolist()
            rows.append({"company": co, "year": yr, "theme": t, "count": len(ids),
                         "strategic_total": total, "share": len(ids) / total, "passage_ids": ids})
    return pd.DataFrame(rows, columns=cols)


def yoy_change(shares: pd.DataFrame) -> pd.DataFrame:
    """Latest year minus previous year, in percentage points, for companies with >= 2 years."""
    cols = ["company", "theme", "year_from", "year_to", "share_from", "share_to", "pp_change", "count_from", "count_to"]
    rows = []
    for co, g in shares.groupby("company"):
        years = sorted(g["year"].unique())
        if len(years) < 2:
            continue
        y0, y1 = years[-2], years[-1]
        a = g[g.year == y0].set_index("theme")
        b = g[g.year == y1].set_index("theme")
        for t in STRATEGIC_THEMES:
            s0, s1 = float(a.loc[t, "share"]), float(b.loc[t, "share"])
            rows.append({"company": co, "theme": t, "year_from": y0, "year_to": y1, "share_from": s0, "share_to": s1,
                         "pp_change": (s1 - s0) * 100, "count_from": int(a.loc[t, "count"]),
                         "count_to": int(b.loc[t, "count"])})
    return pd.DataFrame(rows, columns=cols)


def companies_missing_year(shares: pd.DataFrame) -> list[str]:
    n_years = shares.groupby("company")["year"].nunique()
    return sorted(n_years[n_years < 2].index)


def evidence(preds: pd.DataFrame, company: str, theme: str, year: str | None = None,
             exclude_review: bool = False) -> pd.DataFrame:
    """Passages behind a (company, theme[, year]) number, most confident first."""
    df = strategic(preds, exclude_review)
    df = df[(df.company == company) & (df[LABEL_COL] == theme)]
    if year:
        df = df[df.year == year]
    return df.sort_values("confidence", ascending=False)
