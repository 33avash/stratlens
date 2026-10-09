"""Test fixtures. Files are generated with PyMuPDF at test time and are never used as training data."""
import sys
from pathlib import Path

import pymupdf
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HEADER = "Acme Consumer Ltd | Integrated Annual Report 2025"

LETTER = ("Chairman's Message\n\nDear Shareholders, this year we expanded our rural distribution network to "
          "reach forty thousand new villages across the country.\n\nWe also launched a premium range of herbal "
          "shampoos to meet evolving consumer needs in urban markets.")
MDA_1 = ("Management Discussion and Analysis\n\nOur e-commerce and quick commerce channels grew strongly during "
         "the year as consumers moved online for everyday purchases.\n\n12,563 2,317 1,768 18.4 19.4 48.0")
MDA_2 = ("We commissioned a new manufacturing facility in Tamil Nadu to add capacity and reduce logistics costs "
         "for southern markets.\n\nFigure 3: Revenue mix")
GOV = "Report on Corporate Governance\n\nThe Board met six times during the year under review."


def _page(doc, body, number):
    page = doc.new_page()
    page.insert_text((40, 30), HEADER, fontsize=8)
    y = 70
    for para in body.split("\n\n"):
        page.insert_textbox(pymupdf.Rect(40, y, 560, y + 90), para, fontsize=10)
        y += 100
    page.insert_text((300, 820), str(number), fontsize=8)


@pytest.fixture
def report_pdf(tmp_path):
    doc = pymupdf.open()
    for i, body in enumerate(["Contents\n\nAbout us 2", LETTER, MDA_1, MDA_2, GOV], start=1):
        _page(doc, body, i)
    path = tmp_path / "ACME_FY2025.pdf"
    doc.save(path)
    return path


@pytest.fixture
def scanned_pdf(tmp_path):
    doc = pymupdf.open()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 60, 60), False)
    pix.clear_with(200)
    for _ in range(3):
        page = doc.new_page()
        page.insert_image(pymupdf.Rect(50, 50, 500, 700), pixmap=pix)
    path = tmp_path / "SCAN_FY2025.pdf"
    doc.save(path)
    return path


@pytest.fixture
def txt_report(tmp_path):
    path = tmp_path / "ACME_FY2026.txt"
    path.write_text(LETTER + "\f" + MDA_1, encoding="utf-8")
    return path
