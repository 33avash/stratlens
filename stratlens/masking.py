"""Mask company names, brand names, numbers and years so the model learns themes, not authorship.

Placeholders are single word-tokens that cannot collide with ordinary vocabulary (e.g. the theme word
"brand"): __company__, __brand__, __num__, __year__. `display()` turns them into [COMPANY] etc. for the UI.
Company terms and brand lists live in config/gazetteer.yaml; ambiguous everyday words (Real, Red, Simple,
Comfort, Glucose, Amla, Treat, Magic) are only masked as part of a multi-word brand (e.g. "Dabur Red Paste").
"""
from __future__ import annotations

import re

import yaml

from .config import GAZETTEER_YAML

_gaz = yaml.safe_load(GAZETTEER_YAML.read_text(encoding="utf-8"))
COMPANY_TERMS_BY_COMPANY: dict[str, list[str]] = _gaz.get("company_terms", {})
BRANDS: dict[str, list[str]] = _gaz.get("brands", {})
COMPANY_TERMS = [t for v in COMPANY_TERMS_BY_COMPANY.values() for t in v]

_ALL_BRANDS = sorted({b for v in BRANDS.values() for b in v}, key=len, reverse=True)
_COMPANY_SORTED = sorted(set(COMPANY_TERMS), key=len, reverse=True)


def _alt(terms):
    # word-boundary-ish: not preceded/followed by a letter or digit (handles ®, ’s, punctuation)
    return re.compile(r"(?<![\w])(?:" + "|".join(re.escape(t) for t in terms) + r")(?:®|™)?(?![\w])")


COMPANY_RE = _alt(_COMPANY_SORTED)
BRAND_RE = _alt(_ALL_BRANDS)
YEAR_RE = re.compile(
    r"(?<![\w])(?:FY\s?'?\d{2,4}(?:\s?[-–]\s?\d{2,4})?|(?:19|20)\d{2}(?:\s?[-–]\s?\d{2,4})?)(?![\w])", re.I)
NUM_RE = re.compile(
    r"(?<![\w])(?:[₹$`H]\s?)?\d[\d,]*(?:\.\d+)?\s?(?:%|bps|bn|mn|crores?|cr|lakhs?|million|billion|x)?(?![\w])", re.I)

TOKENS = {"__company__": "[COMPANY]", "__brand__": "[BRAND]", "__num__": "[NUM]", "__year__": "[YEAR]"}


def mask(text: str) -> str:
    text = COMPANY_RE.sub(" __company__ ", text)
    text = BRAND_RE.sub(" __brand__ ", text)
    text = YEAR_RE.sub(" __year__ ", text)
    text = NUM_RE.sub(" __num__ ", text)
    return re.sub(r"\s+", " ", text).strip()


def display(text_or_term: str) -> str:
    for tok, shown in TOKENS.items():
        text_or_term = text_or_term.replace(tok, shown)
    return text_or_term
