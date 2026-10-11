"""LLM rewrite of template text with the numbers locked.

The template carries every number as a token ([[n1]], [[n2]] …). The LLM may only rephrase the words around them.
The rewrite is rejected (and the template kept) if it contains any digit of its own, drops or invents a token,
or runs too long. Tokens are then replaced by the computed values, so no number on screen comes from the LLM.
Outputs are cached on disk by the client (keyed by model + prompt), so the app works offline once built.
"""
from __future__ import annotations

import re

from ..insights import TOKEN_RE, fill

SYSTEM = (
    "You polish short analytic text about Indian FMCG annual reports for a non-specialist reader. "
    "Rules: keep every token written like [[n1]] exactly once and unchanged; never write a digit, number, "
    "percentage or year yourself (years in the input are words you may keep only by copying them exactly); do not add "
    "facts, opinions or causes; keep the same meaning and the same number of sentences; plain English, sentence case, "
    "British spelling, no hype words, no bullet points. Theme shares describe how much a report talks about a theme, "
    "not spending or performance. Return only the rewritten text.")


def _ok(original: str, out: str) -> bool:
    want = TOKEN_RE.findall(original)
    got = TOKEN_RE.findall(out)
    if sorted(want) != sorted(got):
        return False
    years = set(re.findall(r"FY\d{4}", original))
    rest = TOKEN_RE.sub("", out)
    for y in years:
        rest = rest.replace(y, "")
    return not re.search(r"\d", rest) and len(out) <= 1.6 * len(original) + 40


def narrate(template: str, tokens: dict, client=None, cache_only: bool = False) -> tuple[str, str]:
    """(text, source) where source is 'AI-polished' or 'template'. Never raises.
    cache_only: use a stored rewrite if one exists but never call the network (for page loads)."""
    if client is not None:
        try:
            out = (client.cached(SYSTEM, template) if cache_only else client.generate(SYSTEM, template, max_tokens=400))
            out = (out or "").strip()
            if out and _ok(template, out):
                return fill(out, tokens), "AI-polished"
        except Exception:
            pass
    return fill(template, tokens), "template"
