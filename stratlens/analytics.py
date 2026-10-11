"""Precomputed analytics for the app (written to data/processed/analytics.json by `python -m stratlens.build`).

Every number the app shows about reports comes from here or from passages.csv, so Home, Company and Compare
agree for the same report. Definitions (same as compare.py):
  share         theme passages / strategic (non-Other) passages of one report
  peer average  simple mean share of the other companies in the same peer group, same year (config/companies.yaml)
  all average   simple mean share of every company with a report that year (including this one)
  differences   B minus A in percentage points (pp), with a 95% bootstrap interval from resampling each report's
                passages; `significant` means the interval excludes zero. One bootstrap per report is shared by
                every comparison, so the YoY figures and the A-vs-B figures for the same pair are identical.
"""
from __future__ import annotations

from datetime import date
from itertools import combinations

import numpy as np
import pandas as pd

from . import uncertainty
from .config import EXPECTED_YEARS, STRATEGIC_THEMES, peers_of

LABEL_COL = "pred"


def report_key(company: str, year: str) -> str:
    return f"{company}|{year}"


def _report_stats(g: pd.DataFrame) -> dict:
    strat = g[g[LABEL_COL].isin(STRATEGIC_THEMES)]
    total = len(strat)
    themes = {}
    for t in STRATEGIC_THEMES:
        part = strat[strat[LABEL_COL] == t]
        themes[t] = {"count": int(len(part)), "share": len(part) / total if total else 0.0,
                     "concreteness": round(float(part["specificity"].mean()), 3) if len(part) else None,
                     "concrete_pct": round(float((part["specificity"] >= 2).mean()), 3) if len(part) else None}
    return {"n_passages": int(len(g)), "n_strategic": int(total),
            "n_low_confidence": int(g["needs_review"].astype(bool).sum()),
            "concreteness": round(float(strat["specificity"].mean()), 3) if total else None,
            "concrete_pct": round(float((strat["specificity"] >= 2).mean()), 3) if total else None,
            "themes": themes}


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def _avg_block(stats: dict, keys: list[str]) -> dict | None:
    if not keys:
        return None
    return {"companies": [k.split("|")[0] for k in keys],
            "concreteness": _mean([stats[k]["concreteness"] for k in keys]),
            "themes": {t: {"share": _mean([stats[k]["themes"][t]["share"] for k in keys]),
                           "concreteness": _mean([stats[k]["themes"][t]["concreteness"] for k in keys])}
                       for t in STRATEGIC_THEMES}}


def _diff_block(stats, samples, a: str, b: str) -> dict:
    lo, hi = uncertainty.diff_ci(samples[a], samples[b])
    out = {}
    for j, t in enumerate(STRATEGIC_THEMES):
        pp = (stats[b]["themes"][t]["share"] - stats[a]["themes"][t]["share"]) * 100
        out[t] = {"pp": round(pp, 2), "ci": [round(float(lo[j]), 2), round(float(hi[j]), 2)],
                  "significant": bool(lo[j] > 0 or hi[j] < 0)}
    return out


def compute(passages: pd.DataFrame) -> dict:
    reps = passages[["company", "year"]].drop_duplicates().sort_values(["company", "year"]).values.tolist()
    stats, samples = {}, {}
    for c, y in reps:
        g = passages[(passages.company == c) & (passages.year == y)]
        k = report_key(c, y)
        stats[k] = _report_stats(g)
        samples[k] = uncertainty.report_share_samples(g[LABEL_COL].to_numpy(), c, y)

    for c, y in reps:
        k = report_key(c, y)
        same_year = [report_key(cc, yy) for cc, yy in reps if yy == y]
        peers = [report_key(p, y) for p in peers_of(c) if report_key(p, y) in stats]
        stats[k]["peer_avg"] = _avg_block(stats, peers)
        stats[k]["all_avg"] = _avg_block(stats, same_year)

    yoy = {}
    for c in sorted({c for c, _ in reps}):
        years = sorted(y for cc, y in reps if cc == c)
        if len(years) >= 2:
            a, b = report_key(c, years[-2]), report_key(c, years[-1])
            yoy[c] = {"from": years[-2], "to": years[-1], "themes": _diff_block(stats, samples, a, b)}

    pairs = {f"{a}~{b}": _diff_block(stats, samples, a, b) for a, b in combinations(sorted(stats), 2)}

    companies = sorted({c for c, _ in reps})
    years = sorted(set(EXPECTED_YEARS) | {y for _, y in reps})
    missing = [report_key(c, y) for c in companies for y in years if report_key(c, y) not in stats]
    return {"generated": date.today().isoformat(), "n_boot": uncertainty.N_BOOT, "reports": stats, "yoy": yoy,
            "pairs": pairs, "missing": missing, "years": years, "companies": companies}


def pair(analytics: dict, a: str, b: str) -> dict:
    """Differences B - A for two report keys, whichever order they were stored in."""
    if f"{a}~{b}" in analytics["pairs"]:
        return analytics["pairs"][f"{a}~{b}"]
    rev = analytics["pairs"][f"{b}~{a}"]
    return {t: {"pp": -d["pp"], "ci": [-d["ci"][1], -d["ci"][0]], "significant": d["significant"]}
            for t, d in rev.items()}
