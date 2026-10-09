import pytest

from stratlens.extract import ExtractionError, extract_bytes, extract_file, parse_filename


def test_pdf_pages_preserved(report_pdf):
    doc = extract_file(report_pdf)
    assert doc.page_count == 5
    assert [p.number for p in doc.pages] == [1, 2, 3, 4, 5]
    assert "Chairman's Message" in doc.pages[1].text
    assert "Corporate Governance" in doc.pages[4].text
    assert not doc.looks_scanned


def test_txt_splits_on_form_feed(txt_report):
    doc = extract_file(txt_report)
    assert doc.page_count == 2
    assert "rural distribution" in doc.pages[0].text
    assert "quick commerce" in doc.pages[1].text


def test_empty_file_rejected():
    with pytest.raises(ExtractionError, match="empty"):
        extract_bytes(b"", "x.pdf")


def test_corrupt_pdf_rejected():
    with pytest.raises(ExtractionError, match="could not be opened"):
        extract_bytes(b"%PDF-1.4 this is not really a pdf", "broken.pdf")


def test_unsupported_type_rejected():
    with pytest.raises(ExtractionError, match="not a PDF or TXT"):
        extract_bytes(b"hello", "report.docx")


def test_scanned_pdf_detected(scanned_pdf):
    doc = extract_file(scanned_pdf)
    assert doc.scanned_share == 1.0
    assert doc.looks_scanned


def test_missing_file():
    with pytest.raises(ExtractionError, match="not found"):
        extract_file("does/not/exist.pdf")


@pytest.mark.parametrize("name,expected", [
    ("HUL_FY2025.pdf", ("HUL", "FY2025")),
    ("dabur_fy2026.txt", ("DABUR", "FY2026")),
    ("annual-report.pdf", (None, None)),
])
def test_parse_filename(name, expected):
    assert parse_filename(name) == expected
