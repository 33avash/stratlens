"""Phase 1: config files, gazetteer masking, section validation and precomputed analytics."""
import numpy as np
import pandas as pd

from stratlens import analytics, config
from stratlens.extract import extract_file
from stratlens.masking import mask
from stratlens.sections import Section, detect_sections, validate


def test_config_has_six_companies_and_peer_groups():
    assert set(config.COMPANIES) == {"DABUR", "MARICO", "HUL", "GODREJ", "NESTLE", "BRITANNIA"}
    assert config.peers_of("DABUR") == ["MARICO"]
    assert config.peers_of("NESTLE") == ["BRITANNIA"]
    assert config.peers_of("UNKNOWN") == []
    assert config.THEMES[-1] == "OTHER" and len(config.STRATEGIC_THEMES) == 6
    assert all(c.startswith("#") for c in config.THEME_COLORS.values())


def test_gazetteer_masks_new_companies_but_not_everyday_words():
    assert mask("Marico grew Parachute and Saffola") == "__company__ grew __brand__ and __brand__"
    assert mask("Britannia launched Good Day cookies") == "__company__ launched __brand__ cookies"
    assert mask("GCPL relaunched HIT and Good Knight") == "__company__ relaunched __brand__ and __brand__"
    assert mask("a big hit with consumers on a good day") == "a big hit with consumers on a good day"


def test_validate_rejects_untrustworthy_detection(report_pdf):
    doc = extract_file(report_pdf)
    secs = detect_sections(doc)
    assert any("characters" in p for p in validate(doc, secs))  # tiny synthetic sections
    assert "no MD&A heading found" in validate(doc, [s for s in secs if s.section == "LETTER"])
    overlap = [Section("LETTER", 2, 3, "heading"), Section("MDA", 3, 4, "heading")]
    assert "letter and MD&A ranges overlap" in validate(doc, overlap)
    # confirmed overrides are trusted, as long as they fit inside the PDF
    assert validate(doc, [Section("MDA", 2, 3, "override")]) == []
    assert validate(doc, [Section("MDA", 2, 99, "override")])


def _fake(company, year, labels):
    return pd.DataFrame({"company": company, "year": year, "pred": labels, "specificity": 2,
                         "needs_review": False, "passage_id": [f"{company}{year}{i}" for i in range(len(labels))]})


def test_analytics_yoy_matches_pairs_and_flips_sign():
    a = _fake("DABUR", "FY2025", ["PROD"] * 30 + ["DIGI"] * 5 + ["OTHER"] * 10)
    b = _fake("DABUR", "FY2026", ["PROD"] * 15 + ["DIGI"] * 20 + ["OTHER"] * 10)
    c = _fake("MARICO", "FY2026", ["BRAND"] * 20 + ["PROD"] * 20)
    an = analytics.compute(pd.concat([a, b, c], ignore_index=True))
    y = an["yoy"]["DABUR"]["themes"]["DIGI"]
    p = analytics.pair(an, "DABUR|FY2025", "DABUR|FY2026")["DIGI"]
    assert y == p and y["significant"] and y["pp"] > 0
    q = analytics.pair(an, "DABUR|FY2026", "DABUR|FY2025")["DIGI"]
    assert q["pp"] == -p["pp"] and q["ci"] == [-p["ci"][1], -p["ci"][0]]
    assert an["reports"]["DABUR|FY2026"]["peer_avg"]["companies"] == ["MARICO"]
    assert np.isclose(an["reports"]["DABUR|FY2026"]["all_avg"]["themes"]["BRAND"]["share"], 0.25)
    assert "MARICO|FY2025" in an["missing"]
