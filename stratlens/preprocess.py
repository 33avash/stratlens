"""Cleaning and segmentation of section text into passages.

Pipeline per report:
  1. find running headers/footers (short lines repeated on >= 3 pages) across the whole report
  2. for pages inside the selected sections: drop header lines, page numbers, table-like and caption-like blocks
  3. merge blocks split mid-sentence, fix hyphenation, strip bullet glyphs
  4. sentence-split and pack sentences into passages of <= MAX_WORDS words
  5. drop short, statutory, numeric-heavy and duplicate passages (every drop is logged with a reason)
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass

import pandas as pd

from .config import MAX_WORDS, MIN_WORDS
from .extract import Document
from .sections import Section

REPEAT_MIN_PAGES = 3
HEADER_MAX_WORDS = 14
NUMERIC_BLOCK_SHARE = 0.40
NUMERIC_PASSAGE_SHARE = 0.30
CAPTION_MAX_WORDS = 6

BULLETS = "•●▪■◆►➢✓•●▪"
CAPTION_RE = re.compile(r"^(figure|fig\.|source|note|chart|table|graph|exhibit|\*)\b", re.I)
PAGE_NO_RE = re.compile(r"^\s*(page\s*)?\d{1,3}\s*$", re.I)
STATUTORY_RE = re.compile(
    r"(companies act|section \d+|regulation \d+|listing regulations|sebi \(|\bDIN\b|pursuant to|"
    r"rule \d+|schedule v|annexure|e-voting|record date|book closure|investor education and protection|"
    r"annual general meeting|ordinary resolution|special resolution|postal ballot|key managerial personnel|"
    r"independent (non-executive )?director|retire[sd]? by rotation|nomination and remuneration committee)",
    re.I,
)
NUM_TOKEN_RE = re.compile(r"^[\(\-–₹`H$]*[\d,.]+%?[\)x]*$")

ABBREVIATIONS = {
    "rs", "cr", "no", "nos", "ltd", "pvt", "mr", "mrs", "ms", "dr", "e.g", "i.e", "vs", "etc",
    "inc", "co", "approx", "st", "viz", "fy", "u.s", "u.k", "sr", "jr", "dept", "govt", "mn", "bn",
}
SENT_BOUNDARY_RE = re.compile(r"(?<=[.!?])[\"”’)]?\s+(?=[\"“‘(A-Z0-9])")


@dataclass
class PassageResult:
    passages: pd.DataFrame
    drops: pd.DataFrame


def _norm_line(line: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", "", line.lower())).strip(" |-–")


def find_repeated_lines(doc: Document) -> set[str]:
    """Short lines that appear on >= REPEAT_MIN_PAGES distinct pages: running headers/footers."""
    counts: Counter[str] = Counter()
    for page in doc.pages:
        seen = set()
        for block in page.blocks:
            for line in block.splitlines():
                n = _norm_line(line)
                if n and len(n.split()) <= HEADER_MAX_WORDS:
                    seen.add(n)
        counts.update(seen)
    return {line for line, c in counts.items() if c >= REPEAT_MIN_PAGES}


def numeric_share(text: str) -> float:
    toks = text.split()
    if not toks:
        return 0.0
    return sum(bool(NUM_TOKEN_RE.match(t)) for t in toks) / len(toks)


def clean_block(block: str, repeated: set[str]) -> str:
    lines = []
    for line in block.splitlines():
        if _norm_line(line) in repeated or PAGE_NO_RE.match(line):
            continue
        lines.append(line.strip())
    text = "\n".join(lines)
    text = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)  # hyphenation across lines
    text = re.sub(f"[{BULLETS}]", " ", text)
    text = text.replace("­", "")  # soft hyphen
    return re.sub(r"\s+", " ", text).strip()


def block_drop_reason(text: str) -> str | None:
    words = text.split()
    if not words:
        return "empty"
    if numeric_share(text) > NUMERIC_BLOCK_SHARE:
        return "table"
    if len(words) < CAPTION_MAX_WORDS or CAPTION_RE.match(text):
        return "caption"
    return None


def split_sentences(text: str) -> list[str]:
    parts, start = [], 0
    for m in SENT_BOUNDARY_RE.finditer(text):
        before = text[start:m.start()].rstrip("\"”’)")
        last = before.split()[-1].lower().rstrip(".") if before.split() else ""
        if last in ABBREVIATIONS or (len(last) == 1 and last.isalpha()):
            continue
        parts.append(text[start:m.start()].strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return [p for p in parts if p]


def pack_sentences(sentences: list[str], max_words: int = MAX_WORDS) -> list[str]:
    chunks, cur = [], []
    for s in sentences:
        n = len(s.split())
        if cur and sum(len(x.split()) for x in cur) + n > max_words:
            chunks.append(" ".join(cur))
            cur = []
        cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def _ends_sentence(text: str) -> bool:
    return bool(re.search(r"[.!?:][\"”’)]?$", text))


def dedupe_key(text: str) -> str:
    return hashlib.md5(re.sub(r"[^a-z]", "", text.lower()).encode()).hexdigest()


def segment(doc: Document, sections: list[Section], company: str, year: str) -> PassageResult:
    repeated = find_repeated_lines(doc)
    rows, drops = [], []

    def drop(section, page, reason, text):
        drops.append({"company": company, "year": year, "section": section, "page": page,
                      "reason": reason, "text": text[:300]})

    for sec in sections:
        # collect cleaned blocks as (page, text), merging mid-sentence continuations
        units: list[list] = []
        for pno in sec.pages():
            if pno < 1 or pno > doc.page_count:
                continue
            for raw in doc.pages[pno - 1].blocks:
                text = clean_block(raw, repeated)
                reason = block_drop_reason(text)
                if reason == "empty":
                    continue
                if reason:
                    drop(sec.section, pno, reason, text)
                    continue
                if units and not _ends_sentence(units[-1][1]) and text[:1].islower():
                    units[-1][1] += " " + text
                else:
                    units.append([pno, text])
        for pno, text in units:
            for k, chunk in enumerate(pack_sentences(split_sentences(text))):
                rows.append({"company": company, "year": year, "section": sec.section, "page": pno, "text": chunk})

    seen: set[str] = set()
    kept = []
    for r in rows:
        wc = len(r["text"].split())
        if wc < MIN_WORDS:
            drop(r["section"], r["page"], "short", r["text"]); continue
        if STATUTORY_RE.search(r["text"]):
            drop(r["section"], r["page"], "statutory", r["text"]); continue
        if numeric_share(r["text"]) > NUMERIC_PASSAGE_SHARE:
            drop(r["section"], r["page"], "numeric", r["text"]); continue
        key = dedupe_key(r["text"])
        if key in seen:
            drop(r["section"], r["page"], "duplicate", r["text"]); continue
        seen.add(key)
        r["word_count"] = wc
        kept.append(r)

    df = pd.DataFrame(kept, columns=["company", "year", "section", "page", "text", "word_count"])
    if not df.empty:
        df.insert(0, "passage_id", [
            f"{company}_{year}_{r.section}_p{r.page}_{i:03d}" for i, r in enumerate(df.itertuples())
        ])
    else:
        df.insert(0, "passage_id", [])
    return PassageResult(df, pd.DataFrame(drops, columns=["company", "year", "section", "page", "reason", "text"]))
