"""Tests for v2 modules: masking, specificity, bootstrap, report generation, LLM fallback, retrieval, trace."""
import json

import numpy as np
import pandas as pd
import pytest

from stratlens import compare, masking, report, specificity, uncertainty
from stratlens.config import STRATEGIC_THEMES, THEMES


# ---------- masking ----------
def test_mask_company_brand_number_year():
    t = "Dabur India launched Vatika Bio-Infusions in FY 2025-26, adding ₹135 crore and 12.5% growth."
    m = masking.mask(t)
    assert "Dabur" not in m and "Vatika" not in m
    assert "__company__" in m and "__brand__" in m and "__year__" in m and "__num__" in m
    assert "135" not in m and "12.5" not in m and "2025" not in m


def test_mask_keeps_theme_words_and_ambiguous_words():
    m = masking.mask("Our brand campaign reached real consumers and red-hot simple demand.")
    assert "brand" in m.split() and "real" in m and "simple" in m  # theme word and everyday words survive


def test_mask_placeholders_display():
    assert masking.display("__brand__ launch") == "[BRAND] launch"


def test_masked_pipeline_ignores_names():
    from stratlens import model as mdl
    texts = ["HUL launched a new premium shampoo range", "Dabur launched a new premium shampoo range"] * 4 + \
            ["HUL expanded rural distribution reach", "Dabur expanded rural distribution reach"] * 4
    labels = ["PROD", "PROD"] * 4 + ["DIST", "DIST"] * 4
    m = mdl.build_pipeline(1.0).fit(texts, labels)
    vocab = set(m.named_steps["tfidf"].get_feature_names_out())
    assert not {"hul", "dabur"} & vocab and "__company__" in vocab


# ---------- specificity ----------
@pytest.mark.parametrize("text,tier", [
    ("We remain committed to growth and value creation for all stakeholders.", "Vague"),
    ("We will reduce virgin plastic by 50% by 2030 through Project Circular Bharat across 10 cities.", "Specific"),
    ("Our direct reach expanded to 1.5 million outlets during the year.", "Vague"),
    ("We aim to double our direct reach to 3 million outlets by FY27.", "Specific"),
])
def test_specificity_tiers_hand_written(text, tier):
    r = specificity.score_texts([text], use_cache=False)[0]
    assert r["spec_tier"] == tier, r


def test_specificity_list_markers_and_company_names_do_not_count():
    s = specificity.signals("2. Governance: Dabur reviews risks with the Board.",
                            [["ORG", "Dabur"], ["ORG", "the Board"]])
    assert s["numbers"] == 0 and s["named"] == []


# ---------- bootstrap ----------
def test_metric_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y = np.array(THEMES * 20)
    pred = np.where(rng.random(len(y)) < 0.8, y, "OTHER")
    lo, hi = uncertainty.metric_ci(y, pred, n=300)
    from sklearn.metrics import f1_score
    point = f1_score(y, pred, labels=THEMES, average="macro")
    assert lo <= point <= hi and 0 <= lo < hi <= 1


def test_yoy_ci_flags_clear_and_noisy_shifts():
    from stratlens import compare
    rows = []
    for yr, digi in [("FY2025", 5), ("FY2026", 60)]:  # DIGI jumps; BRAND stays put
        labs = ["DIGI"] * digi + ["BRAND"] * 40 + ["PROD"] * (100 - digi)
        rows += [{"passage_id": f"{yr}{i}", "company": "A", "year": yr, "pred": l, "needs_review": False}
                 for i, l in enumerate(labs)]
    preds = pd.DataFrame(rows)
    yoy = uncertainty.yoy_ci(preds, compare.yoy_change(compare.theme_shares(preds)), n=400)
    r = yoy.set_index("theme")
    assert r.loc["DIGI", "significant"] and r.loc["DIGI", "ci_lo"] > 0
    assert set(yoy.theme) == set(STRATEGIC_THEMES)
    assert (yoy.ci_lo <= yoy.pp_change + 1e-9).all() and (yoy.pp_change <= yoy.ci_hi + 1e-9).all()


# ---------- report generation (no LLM) ----------
def _preds():
    rows = []
    spec = {("A", "FY2025"): {"PROD": 20, "DIST": 10, "BRAND": 10, "OTHER": 10},
            ("A", "FY2026"): {"PROD": 10, "DIST": 10, "DIGI": 20, "BRAND": 10, "OTHER": 5},
            ("B", "FY2026"): {"SUST": 15, "OPS": 15, "BRAND": 10, "OTHER": 5}}
    for (c, y), counts in spec.items():
        for lab, n in counts.items():
            for i in range(n):
                rows.append({"passage_id": f"{c}{y}{lab}{i}", "company": c, "year": y, "section": "MDA",
                             "page": i + 1, "text": f"{lab} passage {i} with 12% growth by 2030",
                             "pred": lab, "confidence": 0.8, "needs_review": False,
                             "specificity": i % 4, "spec_tier": specificity.tier(i % 4)})
    return pd.DataFrame(rows)


def test_brief_without_llm_uses_real_figures():
    b = report.build_brief(_preds(), None, ["A"], "FY2026", client=None)
    assert b["narrative_source"] == "template"
    f = b["facts"]
    assert sum(t["count"] for t in f["themes"]) == f["n_strategic"] == 50
    assert f["top_themes"][0] == "Digital & e-commerce"
    assert any(s["theme"] == "Digital & e-commerce" for s in f["significant_shifts"])
    for key, text in b["narrative"].items():
        assert report.unsupported_numbers(text, f) == [], (key, text)
    html_out = report.render_html(b)
    assert "<html" in html_out and "Digital &amp; e-commerce" in html_out
    pdf = report.render_pdf(b)
    assert pdf[:4] == b"%PDF" and len(pdf) > 2000


def test_comparison_brief_and_summaries():
    b = report.build_brief(_preds(), None, ["A", "B"], "FY2026")
    assert b["kind"] == "comparison" and set(b["facts"]["companies"]) == {"A", "B"}
    s = report.report_summaries(_preds())
    assert len(s) == 3 and all(x["summary"] for x in s)


def test_number_check_flags_invented_figures():
    f = {"themes": [{"share_pct": 34, "diff_vs_peers_pp": 15.6}]}
    assert report.unsupported_numbers("Brand takes 34% (+15.6 pp).", f) == []
    assert report.unsupported_numbers("Brand takes 47% of the narrative.", f) == ["47"]


class _FakeClient:
    name, model = "Fake", "fake-1"

    def __init__(self, payload):
        self.payload = payload

    def generate(self, *a, **k):
        return self.payload


def test_llm_sections_with_bad_numbers_fall_back_to_template():
    preds = _preds()
    good = report.build_brief(preds, None, ["A"], "FY2026")["narrative"]
    payload = json.dumps({"summary": "A spends 99% on digital.", "headline": good["headline"],
                          "theme_mix": good["theme_mix"], "shifts": good["shifts"],
                          "specificity": good["specificity"]})
    b = report.build_brief(preds, None, ["A"], "FY2026", client=_FakeClient(payload))
    assert b["narrative"]["summary"] == good["summary"]  # invented 99% rejected
    assert any("summary" in n and "99" in n for n in b["notes"])


def test_llm_fallback_when_no_provider(monkeypatch):
    from stratlens.llm import client
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(client, "_load_env", lambda: None)
    client.get_client.cache_clear()
    try:
        assert client.get_client() is None
        assert "off" in client.status()
        b = report.build_brief(_preds(), None, ["A"], "FY2026", client=client.get_client())
        assert b["narrative_source"] == "template" and b["narrative"]["summary"]
    finally:
        client.get_client.cache_clear()


def test_llm_error_degrades_gracefully():
    from stratlens.llm.client import LLMError

    class Broken:
        name, model = "Broken", "x"

        def generate(self, *a, **k):
            raise LLMError("Gemini is temporarily overloaded. Try again shortly.")

    b = report.build_brief(_preds(), None, ["A"], "FY2026", client=Broken())
    assert b["narrative_source"] == "template" and "overloaded" in b["notes"][0]


def test_clean_display_fixes():
    assert report.clean(chr(96) + '60 crore') == chr(8377) + '60 crore'
    assert report.clean('3. ' + chr(0xF0B7) + 'Cost') == '3. Cost'
    assert report.clean('3. ' + chr(7) + 'Cost') == '3. Cost'


# ---------- one-year briefs ----------
def test_one_year_brief_says_only_one_year():
    b = report.build_brief(_preds(), None, ["B"], "FY2026")
    assert not b["facts"]["has_prior_year"]
    assert b["narrative"]["shifts"].startswith("Only one year available")
    assert "noise" not in b["narrative"]["shifts"]


def test_one_year_brief_overrides_llm_shift_text():
    good = report.build_brief(_preds(), None, ["B"], "FY2026")["narrative"]
    payload = json.dumps({**good, "shifts": "No shift is distinguishable from noise."})
    b = report.build_brief(_preds(), None, ["B"], "FY2026", client=_FakeClient(payload))
    assert b["narrative_source"] == "AI-assisted"
    assert b["narrative"]["shifts"] == report.ONE_YEAR


def test_method_note_discloses_ai_drafted_labels():
    m = {"test": {"model": {"macro_f1": 0.77, "macro_f1_ci": [0.69, 0.84]}}, "dataset": {"n_labelled": 885}}
    b = report.build_brief(_preds(), m, ["A"], "FY2026")
    assert "AI-drafted" in b["method_note"] and "not been human-verified" in b["method_note"]


# ---------- consistent counts ----------
def test_counts_consistent_between_overview_and_briefs():
    preds = _preds()
    preds.loc[preds.index[:7], "needs_review"] = True
    for excl in (False, True):
        sh = compare.theme_shares(preds, excl)
        strat = compare.strategic(preds, excl)
        for c, y in [("A", "FY2025"), ("A", "FY2026"), ("B", "FY2026")]:
            f = report.report_facts(preds, c, y, exclude_review=excl)
            cell = sh[(sh.company == c) & (sh.year == y)]
            assert f["n_strategic"] == cell["strategic_total"].iloc[0] == len(strat[(strat.company == c) & (strat.year == y)])
            assert sum(t["count"] for t in f["themes"]) == f["n_strategic"]
            scope = compare.in_scope(preds, excl)
            assert f["n_passages"] == len(scope[(scope.company == c) & (scope.year == y)])


# ---------- hybrid retrieval ----------
def _corpus():
    texts = ["Quick commerce grew rapidly and now drives our e-commerce growth.",
             "Commerce in rural markets was quick to recover after the monsoon.",
             "The Board declared an interim dividend per share.",
             "We expanded direct reach to more villages through our distribution network.",
             "Our sustainability programme reduced plastic packaging across sites."]
    return pd.DataFrame({"passage_id": [f"p{i}" for i in range(len(texts))], "company": ["A", "A", "A", "B", "B"],
                         "year": ["FY2026"] * 3 + ["FY2025"] * 2, "section": ["MDA"] * 5, "page": range(1, 6),
                         "text": texts, "pred": ["DIGI", "DIST", "OTHER", "DIST", "SUST"]})


def test_bm25_ranks_exact_phrase_first():
    from stratlens.llm import retrieval
    docs = [sum(retrieval.terms(t), []) for t in _corpus().text]
    q = sum(retrieval.terms("quick commerce", query=True), [])
    scores = retrieval.bm25(docs, q)
    assert scores.argmax() == 0 and scores[0] > scores[1] > 0  # the phrase beats the same words apart


def test_hybrid_search_keyword_path_finds_other_passages(monkeypatch):
    from stratlens import embeddings
    from stratlens.llm import retrieval
    monkeypatch.setattr(embeddings, "available", lambda: False)
    hits, method = retrieval.search(_corpus(), "What did they say about quick commerce?")
    assert method.startswith("BM25") and hits.iloc[0]["passage_id"] == "p0"
    assert hits["relevance"].between(0, 1).all() and hits.iloc[0]["relevance"] == 1.0
    hits, _ = retrieval.search(_corpus(), "interim dividend")
    assert hits.iloc[0]["pred"] == "OTHER"  # Other passages are searchable
    hits, _ = retrieval.search(_corpus(), "What is the capital of France?")
    assert hits.empty  # nothing relevant -> caller refuses
    hits, _ = retrieval.search(_corpus(), "plastic packaging", year="FY2026")
    assert hits.empty  # year filter applies; no year filter is the default
    hits, _ = retrieval.search(_corpus(), "")
    assert hits.empty


def test_hybrid_search_fuses_both_rankings(monkeypatch):
    import numpy as np
    from stratlens import embeddings
    from stratlens.llm import retrieval
    corpus = _corpus()
    vecs = {t: np.eye(8)[i] for i, t in enumerate(corpus.text)}
    vecs["villages reach"] = np.eye(8)[3]
    monkeypatch.setattr(embeddings, "available", lambda: True)
    monkeypatch.setattr(embeddings, "encode", lambda texts, cache=True: np.stack([vecs[t] for t in texts]))
    hits, method = retrieval.search(corpus, "villages reach")
    assert method.startswith("hybrid") and hits.iloc[0]["passage_id"] == "p3"
    assert hits.iloc[0]["bm25_rank"] == 1 and hits.iloc[0]["emb_rank"] == 1 and hits.iloc[0]["relevance"] == 1.0
    assert retrieval.company_in_query("How is Dabur using quick commerce?") == "DABUR"
    assert retrieval.company_in_query("Compare HUL and Dabur") is None


# ---------- inspector trace + signals ----------
def test_specificity_describe_lists_fired_signals():
    s = {"numbers": 2, "time": True, "named": ["Vatika", "Real", "Hajmola"], "commitment": False}
    assert specificity.describe(s) == "2 numbers · date · named: Vatika, Real +1"
    assert specificity.describe(json.dumps(s)).startswith("2 numbers")
    assert specificity.describe({"numbers": 0, "time": False, "named": [], "commitment": False}) == "no concrete signals"


def test_trace_matches_model_prediction():
    from stratlens import model as mdl
    from stratlens import trace
    texts = ["We launched a new premium shampoo range for urban consumers"] * 4 + \
            ["We expanded rural distribution reach to more villages"] * 4
    labels = ["PROD"] * 4 + ["DIST"] * 4
    m = mdl.build_pipeline(1.0).fit(texts, labels)
    t = trace.trace(m, "Dabur expanded rural distribution reach to 50,000 villages by FY27.", 0.45, use_cache=False)
    assert t["pred"] == m.predict(["Dabur expanded rural distribution reach to 50,000 villages by FY27."])[0] == "DIST"
    assert "[COMPANY]" in t["masked"] and "[NUM]" in t["masked"] and "Dabur" not in t["masked"]
    assert abs(sum(p["prob"] for p in t["probs"]) - 1) < 1e-3 and t["confidence"] == t["probs"][0]["prob"]
    assert t["specificity"]["signals"]["time"] and t["specificity"]["score"] >= 2
    cleaned, passages, dropped = trace.segment_text("Our reach grew.\n\nPage 3\n\nWe also launched new SKUs in Q2.")
    assert dropped is None and passages
