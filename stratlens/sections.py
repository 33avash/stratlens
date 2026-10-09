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
    r"(chairman'?s?\s*(message|statement|letter)|chairperson'?s? (message|statement)|"
    r"(ceo|md)\s*(&|and)?\s*(md|ceo)?'?s? (message|statement|letter)|message from the (chairman|ceo|managing director))",
    re.I,
)
MDA_RE = re.compile(r"(management\s+discussion\s*(&|and)\s*analysis|&\s*analysis\s+discussion\s+management)", re.I)
STOP_RE = re.compile(
    r"(report on corporate governance|corporate governance report|directors'? report|board'?s report|"
    r"business responsibility and sustainability|independent auditor'?s? report|balance sheet|notice of)",
    re.I,
)


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


def detect_sections(doc: Document) -> list[Section]:
    """Heading-based detection. Returns at most one LETTER and one MDA span."""
    found: list[Section] = []
    n = doc.page_count
    letter = next((i for i in range(n) if LETTER_RE.search(_head(doc, i))), None)
    if letter is not None:
        end = letter
        while end + 1 < n and end + 1 - letter < MAX_LETTER_PAGES:
            h = _head(doc, end + 1)
            if MDA_RE.search(h) or STOP_RE.search(h) or len(doc.pages[end + 1].text) < 40:
                break
            end += 1
        found.append(Section("LETTER", letter + 1, end + 1, "heading"))
    # Skip the contents page: an MD&A heading on a page that also lists other sections is a TOC.
    mda = next(
        (i for i in range(n) if MDA_RE.search(_head(doc, i)) and not re.search(r"contents", _head(doc, i), re.I)),
        None,
    )
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
