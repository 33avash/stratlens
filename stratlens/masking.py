"""Mask company names, brand names, numbers and years so the model learns themes, not authorship.

Placeholders are single word-tokens that cannot collide with ordinary vocabulary (e.g. the theme word
"brand"): __company__, __brand__, __num__, __year__. `display()` turns them into [COMPANY] etc. for the UI.
Brand lists are curated per company from the corpus; ambiguous everyday words (Real, Red, Simple, Comfort,
Boost, Glucose, Amla, Everyday) are only masked as part of a multi-word brand (e.g. "Dabur Red Paste").
"""
from __future__ import annotations

import re

COMPANY_TERMS = [
    "Hindustan Unilever Limited", "Hindustan Unilever", "Unilever", "HUL",
    "Dabur India Limited", "Dabur India", "Dabur", "H & B Stores Limited",
    "Nestlé India Limited", "Nestle India Limited", "Nestlé India", "Nestle India", "Nestlé S.A.", "Nestle S.A.",
    "Nestlé", "Nestle", "Hobi Group", "Hobi Kozmetik", "Namaste Laboratories LLC", "Namaste Laboratories",
]

BRANDS = {
    "HUL": ["Surf excel", "Surf Excel", "Rin", "Vim", "Wheel", "Comfort fabric", "Domex", "Lux", "Lifebuoy", "Dove",
            "Pears", "Closeup", "Pepsodent", "Vaseline", "Pond's", "Ponds", "Lakmē", "Lakme", "Glow & Lovely",
            "Glow and Lovely", "TRESemmé", "TRESemme", "Sunsilk", "Clinic Plus", "Nexxus", "Minimalist", "OZiva",
            "Liquid I.V.", "Liquid IV", "Pureit", "Pukka", "Brooke Bond", "Red Label", "Taj Mahal", "Bru", "Knorr",
            "Kissan", "Hellmann's", "Hellman's", "Horlicks", "Boost", "Kwality Wall's", "Magnum", "Cornetto",
            "Simple skincare", "Shakti", "Prabhat", "Shikhar", "Samadhan", "Sangam", "Suvidha", "Axe", "Rexona",
            "Ponds", "Indulekha", "Sunlight", "Cif", "Stratos"],
    "DABUR": ["Dabur Red Paste", "Dabur Red", "Dabur Amla", "Dabur Honey", "Dabur Glucose", "Dabur Chyawanprash",
              "Chyawanprakash", "Chyawanshakti", "ChyawanShakti", "Ratnaprash", "Vatika", "Meswak", "Miswak",
              "Babool", "Odonil", "Odomos", "Sanifresh", "Hajmola", "Pudin Hara", "Honitus", "Réal Activ",
              "Real Activ", "Réal", "Real Fizzin", "Real Bites", "Réal Bites", "Real Cheers", "Hommade", "Badshah",
              "Gulabari", "Fem", "FEM", "OxyLife", "Siens", "Lal Tail", "Lal Gold", "Janma Ghunti", "Shilajit",
              "Hobby", "Hobi", "Namaste", "ORS", "NewU", "Long & Lasting", "Herb'l", "Herb’l", "Herbolene",
              "Sesa", "Cool King", "Hingoli", "Nature Care", "Sat Isabgol", "Khajurprash", "Jaquline", "Botanica",
              "London Notes", "Dant Snan", "Samriddhi", "Saksham", "Yoddha", "Gurukool", "Drishti", "RAS Beauty",
              "Dabur Ventures", "Dermazone", "Red Paste", "Red Toothpaste", "Hammam Zaith", "Chyawanprash",
              "Herb’l Kids", "Swarnaprashan"],
    "NESTLE": ["MAGGI", "Maggi", "KITKAT", "KitKat", "Kitkat", "NESCAFÉ", "NESCAFE", "Nescafé", "Nescafe",
               "MILKMAID", "Milkmaid", "MILKYBAR", "Milkybar", "MUNCH", "Munch", "CERELAC", "Cerelac", "NAN",
               "Lactogen", "Purina", "NESPRESSO", "Nespresso", "Everyday Dairy Whitener", "a+", "Polo", "BAR-ONE",
               "Bar-One", "Nestlé Professional", "Nestle Professional", "NesMitra", "AskNestlé", "AskNestle",
               "Retail ONE", "Jigyasa", "Duo Gusto", "ExcellaPro", "Masala-Ae-Magic", "Masala‑Ae‑Magic"],
}

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
