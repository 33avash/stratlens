import pandas as pd

from stratlens.config import MIN_WORDS
from stratlens.extract import Document, Page, extract_file
from stratlens.preprocess import (block_drop_reason, clean_block, dedupe_key, find_repeated_lines, pack_sentences,
                                  segment, split_sentences)
from stratlens.sections import Section, detect_sections, resolve_sections


def test_repeated_header_and_page_numbers_removed(report_pdf):
    doc = extract_file(report_pdf)
    repeated = find_repeated_lines(doc)
    assert any("acme consumer ltd" in r for r in repeated)
    res = segment(doc, [Section("MDA", 3, 4, "manual")], "ACME", "FY2025")
    assert not res.passages["text"].str.contains("Acme Consumer Ltd").any()
    assert not res.passages["text"].str.fullmatch(r"\d+").any()


def test_short_table_and_caption_blocks_dropped(report_pdf):
    doc = extract_file(report_pdf)
    res = segment(doc, [Section("MDA", 3, 4, "manual")], "ACME", "FY2025")
    assert (res.passages["word_count"] >= MIN_WORDS).all()
    reasons = set(res.drops["reason"])
    assert "table" in reasons          # the row of figures
    assert "caption" in reasons        # 'Figure 3: Revenue mix' and the section heading
    assert res.passages["text"].str.contains("quick commerce").any()
    assert set(res.passages.columns) >= {"passage_id", "company", "year", "section", "page", "word_count", "text"}


def test_block_rules():
    assert block_drop_reason("12,563 2,317 1,768 18.4 19.4") == "table"
    assert block_drop_reason("Figure 2: Sales by region in the year") == "caption"
    assert block_drop_reason("Short heading") == "caption"
    assert block_drop_reason("We expanded our rural reach to forty thousand villages this year.") is None


def test_hyphenation_and_bullets():
    assert clean_block("e-com-\nmerce growth • strong", set()) == "e-commerce growth strong"


def test_sentence_split_respects_abbreviations():
    s = split_sentences("Revenue rose to Rs. 12,563 cr. in FY 2024-25. Margins were 18.4% as per Mr. Burman. "
                        "Growth i.e. volume led.")
    assert s == ["Revenue rose to Rs. 12,563 cr. in FY 2024-25.", "Margins were 18.4% as per Mr. Burman.",
                 "Growth i.e. volume led."]


def test_pack_sentences_caps_length():
    sents = [" ".join(["word"] * 30) + "."] * 3
    chunks = pack_sentences(sents, max_words=70)
    assert len(chunks) == 2
    assert all(len(c.split()) <= 70 for c in chunks)


def test_duplicates_removed():
    text = "We launched a premium herbal shampoo range for urban consumers this year."
    doc = Document("d.txt", [Page(1, [text]), Page(2, [text.upper().replace(".", "!")])])
    res = segment(doc, [Section("MDA", 1, 2, "manual")], "X", "FY2025")
    assert len(res.passages) == 1
    assert (res.drops["reason"] == "duplicate").sum() == 1
    assert dedupe_key("A b, c.") == dedupe_key("ab c")


def test_statutory_boilerplate_dropped():
    text = ("Pursuant to Section 134 of the Companies Act, the Board confirms the annual accounts were prepared "
            "on a going concern basis.")
    doc = Document("d.txt", [Page(1, [text])])
    res = segment(doc, [Section("MDA", 1, 1, "manual")], "X", "FY2025")
    assert res.passages.empty and res.drops["reason"].tolist() == ["statutory"]


def test_heading_detection(report_pdf):
    doc = extract_file(report_pdf)
    secs = {s.section: (s.start_page, s.end_page) for s in detect_sections(doc)}
    assert secs["LETTER"] == (2, 2)
    assert secs["MDA"] == (3, 4)  # stops before the governance report


def test_overrides_take_precedence(report_pdf, tmp_path):
    ov = tmp_path / "ov.csv"
    pd.DataFrame([{"company": "ACME", "year": "FY2025", "section": "MDA", "start_page": 4, "end_page": 4,
                   "note": "x"}]).to_csv(ov, index=False)
    doc = extract_file(report_pdf)
    secs = resolve_sections(doc, "ACME", "FY2025", overrides_path=ov)
    assert [(s.section, s.start_page, s.end_page, s.method) for s in secs] == [("MDA", 4, 4, "override")]
    secs = resolve_sections(doc, "OTHERCO", "FY2025", overrides_path=ov)
    assert all(s.method == "heading" for s in secs)
