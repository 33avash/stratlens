"""Locate the narrative strategy sections (Chairman/CEO letter, MD&A).

Manual overrides (confirmed by inspecting each PDF) take precedence over
heading detection. Detection is the fallback for uploaded reports.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from .config import OVERRIDES_CSV
from .extract import Document

HEAD_CHARS = 300
MAX_LETTER_PAGES = 5
MAX_MDA_PAGES = 40

LETTER_RE = re.compile(
    r"(chairman['’]?s?\s*(message|statement|letter)|chairperson['’]?s? (message|statement)|"
    r"(ceo|md)\s*(&|and)?\s*(md|ceo)?['’]?s? (message|statement|letter)|message from the (chairman|ceo|managing director))",
    re.I,
)
MDA_RE = re.compile(r"(management\s+discussion\s*(&|and)\s*analysis|&\s*analysis\s+discussion\s+management)", re.I)
STOP_RE = re.compile(
    r"(report on corporate governance|corporate governance report|directors['’]? report|board['’]?s report|"
    r"business responsibility and sustainability|independent auditor['’]?s? report|balance sheet|notice of)",
    re.I,
)
CONTENTS_RE = re.compile(r"\b(contents|index)\b", re.I)
DEAR_RE = re.compile(r"\bdear (shareholders|members|stakeholders|shareowners)\b", re.I)
MIN_PAGE_CHARS = 800

# Sanity limits for detected (not manually confirmed) sections.
LETTER_PAGES = (1, 6)
MDA_PAGES = (3, 40)
MIN_SECTION_CHARS = 2000


@dataclass
class Section:
    section: str  # LETTER | MDA
    start_page: int
    end_page: int
    method: str  # heading | override | manual

    def pages(self) -> range:
        return range(self.start_page, self.end_page + 1)


def _head(doc: Document, i: int) -> str:
    return " ".join(doc.pages[i].text.split())[:HEAD_CHARS]


def _is_contents(doc: Document, i: int) -> bool:
    """A contents/index page: says 'contents', or its opening names several sections, or it is mostly page numbers."""
    head = _head(doc, i)
    if CONTENTS_RE.search(head):
        return True
    hits = sum(bool(r.search(head)) for r in (LETTER_RE, MDA_RE, STOP_RE))
    toks = doc.pages[i].text.split()
    nums = sum(t.isdigit() and len(t) <= 3 for t in toks)
    return hits >= 2 or (len(toks) > 0 and nums / len(toks) > 0.12)


def _min_chars(doc: Document) -> float:
    """Running-text threshold: MIN_PAGE_CHARS, or half the report's median page length if its pages are short."""
    lengths = sorted(len(p.text) for p in doc.pages if p.text.strip())
    return min(MIN_PAGE_CHARS, 0.5 * lengths[len(lengths) // 2]) if lengths else MIN_PAGE_CHARS


def _narrative(doc: Document, i: int, threshold: float) -> bool:
    """Enough running text to be a section page rather than a divider or teaser."""
    return len(doc.pages[i].text) >= threshold


def detect_sections(doc: Document) -> list[Section]:
    """Heading-based detection. Returns at most one LETTER and one MDA span.

    Letter: the first non-contents page with a chairman/CEO heading (or, failing that, 'Dear Shareholders') that has
    running text on it or on the next page (a divider page is skipped). MD&A: the first non-contents page with an
    MD&A heading, then forward until a stop heading. Results must still pass validate()."""
    found: list[Section] = []
    n = doc.page_count
    min_chars = _min_chars(doc)

    def first(rx) -> int | None:
        for i in range(n):
            if rx.search(_head(doc, i)) and not _is_contents(doc, i):
                if _narrative(doc, i, min_chars):
                    return i
                if i + 1 < n and _narrative(doc, i + 1, min_chars):
                    return i + 1
        return None

    letter = first(LETTER_RE)
    if letter is None:
        letter = first(DEAR_RE)
    if letter is not None:
        end = letter
        while end + 1 < n and end + 1 - letter < MAX_LETTER_PAGES:
            h = _head(doc, end + 1)
            if MDA_RE.search(h) or STOP_RE.search(h) or not _narrative(doc, end + 1, min_chars):
                break
            end += 1
        found.append(Section("LETTER", letter + 1, end + 1, "heading"))
    mda = first(MDA_RE)
    if mda is not None:
        end = mda
        while end + 1 < n and end + 1 - mda < MAX_MDA_PAGES and not STOP_RE.search(_head(doc, end + 1)):
            end += 1
        found.append(Section("MDA", mda + 1, end + 1, "heading"))
    return found


def load_overrides(company: str, year: str, path=OVERRIDES_CSV) -> list[Section]:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return []
    rows = df[(df.company == company) & (df.year == year)]
    return [Section(r.section, int(r.start_page), int(r.end_page), "override") for r in rows.itertuples()]


def resolve_sections(doc: Document, company: str | None, year: str | None, overrides_path=OVERRIDES_CSV) -> list[Section]:
    if company and year:
        ov = load_overrides(company, year, overrides_path)
        if ov:
            return ov
    return detect_sections(doc)


def heading(doc: Document, page: int, chars: int = 80) -> str:
    """First words of a page, to eyeball whether a range starts where it should."""
    if not 1 <= page <= doc.page_count:
        return ""
    return " ".join(doc.pages[page - 1].text.split())[:chars]


def validate(doc: Document, sections: list[Section]) -> list[str]:
    """Problems that make a section split untrustworthy. Overrides are trusted; detected ranges must pass
    every check. An empty list means the sections can be used."""
    if sections and all(s.method == "override" for s in sections):
        return [f"{s.section} p{s.start_page}-{s.end_page} is outside the {doc.page_count}-page PDF"
                for s in sections if s.start_page < 1 or s.end_page > doc.page_count]
    problems = []
    kinds = {s.section for s in sections}
    for need, label in (("LETTER", "Chairman/CEO letter"), ("MDA", "MD&A")):
        if need not in kinds:
            problems.append(f"no {label} heading found")
    for s in sections:
        lo, hi = LETTER_PAGES if s.section == "LETTER" else MDA_PAGES
        n = len(s.pages())
        if not lo <= n <= hi:
            problems.append(f"{s.section} spans {n} pages (expected {lo}-{hi})")
        chars = sum(len(doc.pages[p - 1].text) for p in s.pages() if 1 <= p <= doc.page_count)
        if chars < MIN_SECTION_CHARS:
            problems.append(f"{s.section} has only {chars} characters of text")
    let = [s for s in sections if s.section == "LETTER"]
    mda = [s for s in sections if s.section == "MDA"]
    if let and mda and set(let[0].pages()) & set(mda[0].pages()):
        problems.append("letter and MD&A ranges overlap")
    return problems
