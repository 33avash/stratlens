"""Ask: answer a question from retrieved passages (shared by the Ask page and the offline precompute)."""
from __future__ import annotations

import re

import pandas as pd

from ..config import COMPANY_NAMES, SECTION_NAMES
from ..report import clean
from . import prompts
from .client import LLMError

CITE_RE = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]")  # [2], [1, 3] or [6-8]


def sources(hits: pd.DataFrame) -> list[dict]:
    return [{"passage_id": r.passage_id, "company": r.company, "company_name": COMPANY_NAMES.get(r.company, r.company),
             "year": r.year, "section": r.section, "section_name": SECTION_NAMES.get(r.section, r.section),
             "page": int(r.page), "text": clean(r.text), "theme": r.pred} for r in hits.itertuples()]


def answer(question: str, srcs: list[dict], client) -> dict:
    """{'status': ok|not_in_passages|no_ai|error, 'text': str, 'unsupported': [numbers not found in the sources]}"""
    if client is None:
        return {"status": "no_ai", "text": "", "unsupported": []}
    try:
        text = client.generate(prompts.ASK_SYSTEM, prompts.ask_prompt(question, srcs), max_tokens=700)
    except LLMError as exc:
        return {"status": "error", "text": str(exc), "unsupported": []}
    except Exception:
        return {"status": "error", "text": "The AI service failed unexpectedly.", "unsupported": []}
    if "NOT_IN_PASSAGES" in text:
        return {"status": "not_in_passages", "text": "", "unsupported": []}
    body = " ".join(s["text"] for s in srcs)
    nums = [n for n in re.findall(r"\d+(?:\.\d+)?%?", CITE_RE.sub("", text)) if n.rstrip("%") not in body]
    return {"status": "ok", "text": text, "unsupported": nums}
