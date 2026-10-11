"""Specificity: how concretely a passage states its strategy.

Rule (0-5 points, computed on the original, unmasked text):
  numbers      +1 if any quantity (₹ amount, %, count, crore, bps; years excluded), +2 if three or more
  time         +1 if a date or deadline (a year, FY27, 'by 2030', a month, a quarter)
  named        +1 if a named initiative, brand or programme (spaCy ORG/PRODUCT/EVENT/WORK_OF_ART/FAC entities,
               'Project X', or a quoted name)
  commitment   +1 if commitment language (will, target, aim to, commit, plan to, goal, by FY..)
Tiers: 0-1 Vague · 2 Moderate · 3-5 Specific.
spaCy is optional: without it, named initiatives fall back to the 'Project X' / quoted-name / brand-list rules.
NER output is cached on disk (data/cache/ner.json) keyed by a hash of the text.
"""
from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache

from .config import DATA
from .masking import BRAND_RE, COMPANY_RE, YEAR_RE

CACHE = DATA / "cache" / "ner.json"
NER_LABELS = {"ORG", "PRODUCT", "EVENT", "WORK_OF_ART", "FAC", "LAW"}
TIERS = ["Vague", "Moderate", "Specific"]

QTY_RE = re.compile(r"(?:[₹$`]\s?\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?\s?(?:%|bps|crores?|cr\b|lakhs?|million|billion|"
                    r"mn\b|bn\b|x\b|MT\b|tonnes?|hectares?|acres?|outlets|villages|cities|countries|stores|SKUs|"
                    r"factories|sites|households|people|farmers|units)|\b\d{1,3}(?:,\d{2,3})+\b|\b\d+(?:\.\d+)?\b)", re.I)
TIME_RE = re.compile(r"\b(?:by|until|before|from)\s+(?:FY\s?'?\d{2,4}|20\d{2})\b|\bFY\s?'?\d{2,4}\b|\b(?:19|20)\d{2}\b|"
                     r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b|"
                     r"\bQ[1-4]\b|\b(?:first|second|third|fourth) quarter\b", re.I)
COMMIT_RE = re.compile(r"\b(?:will|shall|target(?:s|ed|ing)?|aim(?:s|ed)? to|commit(?:s|ted|ment|ments)?|plan(?:s|ned)? to|"
                       r"goal|ambition|pledge[sd]?|intend(?:s|ed)? to|by (?:FY\s?'?\d{2,4}|20\d{2}))\b", re.I)
LIST_MARKER_RE = re.compile(r"(?:^|(?<=[.:;]\s))\s*\d{1,2}[.)]\s")  # '2. Manufacturing…' is a list item, not a quantity
NAMED_RE = re.compile(r"\b(?:Project|Programme|Program|Mission|Initiative|Campaign)\s+[A-Z][\w’']+|"
                      r"[‘“\"][A-Z][^’”\"]{2,40}[’”\"]")


# The company itself, governance bodies and generic labels are not "named initiatives".
GENERIC_RE = re.compile(r"\b(?:company|board|committee|directors?|shareholders?|management|government|esg|"
                        r"framework|policy|report|council)\b", re.I)


def _generic(ent: str) -> bool:
    rest = COMPANY_RE.sub("", ent).strip(" ’'s")
    return bool(GENERIC_RE.search(ent)) or rest == ""


@lru_cache(maxsize=1)
def _nlp():
    try:
        import spacy
        return spacy.load("en_core_web_sm", disable=["lemmatizer", "tagger", "parser", "attribute_ruler"])
    except Exception:
        return None


def nlp_available() -> bool:
    return _nlp() is not None


def _key(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _load_cache() -> dict:
    try:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def entities(texts: list[str], use_cache: bool = True) -> list[list[list[str]]]:
    """[[label, text], ...] per text. Cached on disk; computes missing ones in batch."""
    cache = _load_cache() if use_cache else {}
    missing = [t for t in dict.fromkeys(texts) if _key(t) not in cache]
    nlp = _nlp()
    if missing and nlp is not None:
        for t, doc in zip(missing, nlp.pipe(missing, batch_size=64)):
            cache[_key(t)] = [[e.label_, e.text] for e in doc.ents if e.label_ in NER_LABELS]
        if use_cache:
            try:
                CACHE.parent.mkdir(parents=True, exist_ok=True)
                CACHE.write_text(json.dumps(cache), encoding="utf-8")
            except OSError:
                pass  # read-only deployments still work, just without a warm cache
    return [cache.get(_key(t), []) for t in texts]


def signals(text: str, ents: list[list[str]] | None = None) -> dict:
    without_years = YEAR_RE.sub(" ", LIST_MARKER_RE.sub(" ", text))
    n_qty = len(QTY_RE.findall(without_years))
    named = [e[1] for e in (ents or []) if not _generic(e[1])] + NAMED_RE.findall(text) + BRAND_RE.findall(text)
    return {
        "numbers": n_qty,
        "time": bool(TIME_RE.search(text)),
        "named": sorted(set(n.strip() for n in named))[:6],
        "commitment": bool(COMMIT_RE.search(text)),
    }


def score_from_signals(s: dict) -> int:
    return (2 if s["numbers"] >= 3 else 1 if s["numbers"] >= 1 else 0) + int(s["time"]) + int(bool(s["named"])) \
        + int(s["commitment"])


def describe(s: dict | str | None) -> str:
    """Which signals fired, e.g. '2 numbers · date · named: Vatika · commitment'. Accepts the stored JSON string."""
    if isinstance(s, str):
        try:
            s = json.loads(s)
        except ValueError:
            return ""
    if not isinstance(s, dict):
        return ""
    parts = []
    if s.get("numbers"):
        parts.append(f"{s['numbers']} number{'s' if s['numbers'] != 1 else ''}")
    if s.get("time"):
        parts.append("date")
    if s.get("named"):
        names = s["named"]
        parts.append("named: " + ", ".join(names[:2]) + (f" +{len(names) - 2}" if len(names) > 2 else ""))
    if s.get("commitment"):
        parts.append("commitment")
    return " · ".join(parts) or "no concrete signals"


def tier(score: int) -> str:
    return "Vague" if score <= 1 else "Moderate" if score == 2 else "Specific"


def score_texts(texts: list[str], use_cache: bool = True) -> list[dict]:
    ents = entities(list(texts), use_cache)
    out = []
    for t, e in zip(texts, ents):
        s = signals(t, e)
        sc = score_from_signals(s)
        out.append({"specificity": sc, "spec_tier": tier(sc), "spec_signals": s})
    return out


FULL_CACHE = DATA / "cache" / "ner_full.json"
NUM_LABELS = {"DATE", "CARDINAL", "PERCENT", "MONEY", "QUANTITY"}


def number_entities(texts: list[str]) -> list[list[list[str]]]:
    """[[label, text], ...] of spaCy DATE / CARDINAL / PERCENT / MONEY / QUANTITY entities per text (for promises).
    Cached on disk separately from the concreteness cache; empty lists when spaCy is unavailable."""
    try:
        cache = json.loads(FULL_CACHE.read_text(encoding="utf-8"))
    except Exception:
        cache = {}
    missing = [t for t in dict.fromkeys(texts) if _key(t) not in cache]
    nlp = _nlp()
    if missing and nlp is not None:
        for t, doc in zip(missing, nlp.pipe(missing, batch_size=64)):
            cache[_key(t)] = [[e.label_, e.text] for e in doc.ents if e.label_ in NUM_LABELS]
        try:
            FULL_CACHE.write_text(json.dumps(cache), encoding="utf-8")
        except OSError:
            pass
    return [cache.get(_key(t), []) for t in texts]
