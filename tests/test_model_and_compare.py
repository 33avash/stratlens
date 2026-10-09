import numpy as np
import pandas as pd
import pytest

from stratlens import baseline, compare, explain
from stratlens import model as mdl
from stratlens.config import MODEL_PATH, STRATEGIC_THEMES, THEMES

TOY = {
    "PROD": "we launched a new premium variant and extended the product portfolio with innovation",
    "DIST": "we expanded rural distribution reach to more retail outlets and new villages",
    "DIGI": "e-commerce and quick commerce sales grew with digital marketing and data analytics",
    "OPS": "a new manufacturing plant improved supply chain efficiency and reduced procurement cost",
    "SUST": "we reduced plastic packaging, water use and carbon emissions with renewable energy",
    "BRAND": "a new advertising campaign with a brand ambassador strengthened consumer engagement",
    "OTHER": "the board thanks shareholders and the economy grew while revenue rose this year",
}


@pytest.fixture(scope="module")
def toy_model():
    texts, labels = [], []
    for lab, base in TOY.items():
        for i in range(6):
            texts.append(f"{base} item{i} {base.split()[i]}")
            labels.append(lab)
    return mdl.build_pipeline(C=10).fit(texts, labels)


def test_predict_frame_valid(toy_model):
    out = mdl.predict_frame(toy_model, [TOY["DIST"], TOY["SUST"], "completely unrelated words here"], threshold=0.45)
    probs = out[[f"p_{c}" for c in toy_model.classes_]].to_numpy()
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-3)
    assert set(out["pred"]) <= set(THEMES)
    assert ((out["confidence"] >= 0) & (out["confidence"] <= 1)).all()
    assert (out["needs_review"] == (out["confidence"] < 0.45)).all()
    assert out.loc[0, "pred"] == "DIST" and out.loc[1, "pred"] == "SUST"


def test_predict_frame_empty(toy_model):
    assert mdl.predict_frame(toy_model, []).empty


def test_explain_consistent_with_prediction(toy_model):
    ex = explain.explain(toy_model, TOY["OPS"])
    assert ex["pred"] == "OPS"
    assert abs(sum(c["prob"] for c in ex["top_classes"]) - 1) < 0.5  # top-3 only
    assert all(t["contribution"] > 0 for t in ex["positive"])
    gt = explain.global_terms(toy_model, 5)
    assert set(gt) == set(THEMES) and all(len(v) == 5 for v in gt.values())


def test_highlight_spans_bigrams():
    spans = explain.highlight_spans("Quick commerce and quick commerce growth", ["quick commerce", "growth"])
    assert spans == [(0, 14), (19, 33), (34, 40)]


def test_baseline_predicts_valid_labels():
    preds = baseline.predict(list(TOY.values()) + ["nothing matches here"])
    assert set(preds) <= set(THEMES)
    assert preds[-1] == "OTHER"


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="trained model not built yet")
def test_saved_model_outputs_valid():
    model, meta = mdl.load()
    out = mdl.predict_frame(model, ["We expanded our direct reach to 1.5 million outlets."], meta["threshold"])
    assert out.loc[0, "pred"] in THEMES and 0 <= out.loc[0, "confidence"] <= 1


# ---------- comparison / YoY ----------
def _preds():
    rows = []
    spec = {("A", "FY2025"): ["PROD", "PROD", "DIST", "OTHER"],
            ("A", "FY2026"): ["PROD", "DIST", "DIST", "DIGI", "OTHER", "OTHER"],
            ("B", "FY2026"): ["SUST", "BRAND"]}
    for (c, y), labs in spec.items():
        for i, lab in enumerate(labs):
            rows.append({"passage_id": f"{c}{y}{i}", "company": c, "year": y, "section": "MDA", "page": i,
                         "text": "t", "pred": lab, "confidence": 0.9 if i else 0.3, "needs_review": i == 0})
    return pd.DataFrame(rows)


def test_shares_sum_to_one_and_exclude_other():
    sh = compare.theme_shares(_preds())
    assert set(sh["theme"]) == set(STRATEGIC_THEMES)
    for _, g in sh.groupby(["company", "year"]):
        assert abs(g["share"].sum() - 1) < 1e-9
    a25 = sh[(sh.company == "A") & (sh.year == "FY2025")].set_index("theme")
    assert a25.loc["PROD", "share"] == pytest.approx(2 / 3)  # OTHER not in denominator
    assert a25.loc["PROD", "strategic_total"] == 3


def test_drilldown_ids_match_counts():
    sh = compare.theme_shares(_preds())
    assert (sh["count"] == sh["passage_ids"].apply(len)).all()
    ev = compare.evidence(_preds(), "A", "DIST", "FY2026")
    row = sh[(sh.company == "A") & (sh.year == "FY2026") & (sh.theme == "DIST")].iloc[0]
    assert sorted(ev["passage_id"]) == sorted(row["passage_ids"])
    assert ev["confidence"].is_monotonic_decreasing


def test_yoy_percentage_points_and_missing_year():
    sh = compare.theme_shares(_preds())
    yoy = compare.yoy_change(sh).set_index(["company", "theme"])
    assert set(yoy.index.get_level_values(0)) == {"A"}  # B has one year only
    assert yoy.loc[("A", "PROD"), "pp_change"] == pytest.approx((1 / 4 - 2 / 3) * 100)
    assert yoy.loc[("A", "DIGI"), "pp_change"] == pytest.approx(25.0)
    assert compare.companies_missing_year(sh) == ["B"]


def test_exclude_review_filter():
    sh = compare.theme_shares(_preds(), exclude_review=True)
    a25 = sh[(sh.company == "A") & (sh.year == "FY2025")].set_index("theme")
    assert a25["strategic_total"].iloc[0] == 2  # first passage (needs_review) dropped


def test_empty_predictions():
    empty = _preds().iloc[0:0]
    assert compare.theme_shares(empty).empty
    assert compare.yoy_change(compare.theme_shares(empty)).empty
