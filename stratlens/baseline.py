"""Keyword-lexicon baseline. Used only for audit comparison, never for app predictions.

The lexicon was written from the labelling guidelines before any model results were seen.
"""
from __future__ import annotations

import re

import pandas as pd

LEXICON = {
    "PROD": ["launch", "launched", "innovation", "portfolio", "premium", "premiumisation", "variant",
             "range", "new product", "formulation", "reformulat", "extension", "relaunch", "product"],
    "DIST": ["distribution", "rural", "outlets", "reach", "retail", "distributor", "general trade",
             "modern trade", "international", "market", "geograph", "expansion", "export", "penetration", "channel"],
    "DIGI": ["digital", "e-commerce", "ecommerce", "quick commerce", "d2c", "online", "app", "data",
             "analytics", "artificial intelligence", " ai ", "technology", "platform"],
    "OPS": ["manufacturing", "supply chain", "procurement", "logistics", "cost", "efficiency",
            "factory", "factories", "productivity", "capacity", "plant", "inflation", "commodity"],
    "SUST": ["sustainab", "plastic", "packaging", "water", "emission", "carbon", "net zero", "renewable",
             "climate", "sourcing", "regenerative", "waste", "circular"],
    "BRAND": ["brand", "consumer", "campaign", "advertis", "media", "ambassador", "engagement",
              "positioning", "awareness", "influencer", "communication", "equity"],
}
def score(text: str) -> dict[str, int]:
    """Substring hit counts per theme (keywords with surrounding spaces match whole words only)."""
    low = " " + re.sub(r"[^\w\s-]", " ", text.lower()) + " "
    return {t: sum(low.count(k) for k in kws) for t, kws in LEXICON.items()}


def predict_one(text: str) -> str:
    s = score(text)
    best = max(s.values())
    if best == 0:
        return "OTHER"
    # deterministic tie-break: lexicon order
    return next(t for t in LEXICON if s[t] == best)


def predict(texts: pd.Series | list[str]) -> list[str]:
    return [predict_one(t) for t in texts]
