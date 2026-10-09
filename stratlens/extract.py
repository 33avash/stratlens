"""Text extraction from PDF and TXT with page numbers preserved."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

SCANNED_PAGE_CHARS = 50
SCANNED_REPORT_SHARE = 0.30
FILENAME_RE = re.compile(r"^([A-Za-z]+)_FY(\d{4})", re.I)


class ExtractionError(Exception):
    """Raised with a plain-English message the UI can show directly."""


@dataclass
class Page:
    number: int  # 1-based
    blocks: list[str]
    has_images: bool = False

    @property
    def text(self) -> str:
        return "\n".join(self.blocks)

    @property
    def scanned(self) -> bool:
        return len(self.text.strip()) < SCANNED_PAGE_CHARS and self.has_images


@dataclass
class Document:
    name: str
    pages: list[Page] = field(default_factory=list)
    toc: list[tuple[int, str, int]] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def char_count(self) -> int:
        return sum(len(p.text) for p in self.pages)

    @property
    def scanned_share(self) -> float:
        if not self.pages:
            return 0.0
        return sum(p.scanned for p in self.pages) / len(self.pages)

    @property
    def looks_scanned(self) -> bool:
        return self.scanned_share > SCANNED_REPORT_SHARE


def parse_filename(name: str) -> tuple[str | None, str | None]:
    """'HUL_FY2025.pdf' -> ('HUL', 'FY2025')."""
    m = FILENAME_RE.match(Path(name).name)
    if not m:
        return None, None
    return m.group(1).upper(), f"FY{m.group(2)}"


def _from_pdf_bytes(data: bytes, name: str) -> Document:
    try:
        pdf = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # pymupdf raises several types for bad files
        raise ExtractionError(
            f"'{name}' could not be opened as a PDF. The file may be corrupt or not a PDF; "
            "try re-downloading it."
        ) from exc
    if pdf.needs_pass:
        raise ExtractionError(f"'{name}' is password-protected. Remove the password and upload it again.")
    if pdf.page_count == 0:
        raise ExtractionError(f"'{name}' has no pages.")
    doc = Document(name=name, toc=[tuple(t[:3]) for t in pdf.get_toc()])
    for i, pg in enumerate(pdf):
        blocks = [b[4] for b in pg.get_text("blocks") if b[6] == 0 and b[4].strip()]
        doc.pages.append(Page(number=i + 1, blocks=blocks, has_images=bool(pg.get_images())))
    pdf.close()
    return doc


def _from_txt_bytes(data: bytes, name: str) -> Document:
    text = data.decode("utf-8", errors="replace")
    chunks = text.split("\f") if "\f" in text else [text]
    doc = Document(name=name)
    for i, chunk in enumerate(chunks):
        blocks = [b.strip() for b in re.split(r"\n\s*\n", chunk) if b.strip()]
        doc.pages.append(Page(number=i + 1, blocks=blocks))
    return doc


def extract_bytes(data: bytes, name: str) -> Document:
    if not data:
        raise ExtractionError(f"'{name}' is empty (0 bytes).")
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        doc = _from_pdf_bytes(data, name)
    elif suffix == ".txt":
        doc = _from_txt_bytes(data, name)
    else:
        raise ExtractionError(f"'{name}' is not a PDF or TXT file.")
    if doc.char_count == 0 and not doc.looks_scanned:
        raise ExtractionError(f"No text could be extracted from '{name}'.")
    return doc


def extract_file(path: str | Path) -> Document:
    path = Path(path)
    if not path.exists():
        raise ExtractionError(f"File not found: {path}")
    return extract_bytes(path.read_bytes(), path.name)
