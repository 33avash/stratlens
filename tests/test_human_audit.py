"""Phase 2: human-label audit (synthetic labels only; the real verify_*.csv files are never touched)."""
import numpy as np
import pandas as pd
import pytest

from stratlens import human_audit as ha


def test_equal_allocation_spreads_and_respects_stratum_size():
    a = ha.equal_allocation(pd.Series({"A": 100, "B": 2, "C": 50}), 30)
    assert a.sum() == 30 and a["B"] == 2 and abs(a["A"] - a["C"]) <= 1


def test_normalise_label_accepts_codes_and_names():
    assert ha.normalise_label(" prod ") == "PROD"
    assert ha.normalise_label("Digital & e-commerce") == "DIGI"
    assert ha.normalise_label("") is None
    with pytest.raises(ValueError):
        ha.normalise_label("marketing")


def test_macro_f1_ignores_themes_absent_from_subset():
    assert ha.macro_f1(["PROD", "DIST"], ["PROD", "DIST"]) == 1.0


def _passages():
    return pd.DataFrame({"company": ["MARICO"] * 100, "pred": ["PROD"] * 90 + ["OTHER"] * 10})


def test_weighted_accuracy_reweights_stratified_sample():
    # 5 PROD rows (all right) stand for 90 passages; 5 OTHER rows (all wrong) stand for 10
    new = pd.DataFrame({"company": "MARICO", "model_label": ["PROD"] * 5 + ["OTHER"] * 5,
                        "human_label": ["PROD"] * 5 + ["DIST"] * 5})
    w = ha.stratum_weights(new, _passages())
    s = ha.score(new.human_label, new.model_label, w)
    assert s["accuracy"] == 0.5 and np.isclose(s["weighted_accuracy"], 0.9)


def test_compute_reports_kappa_and_per_company():
    test = pd.DataFrame({"company": ["DABUR"] * 6, "human_label": ["PROD", "PROD", "DIST", "DIST", "OTHER", "OTHER"],
                         "ai_drafted_label": ["PROD", "PROD", "DIST", "DIST", "OTHER", "PROD"],
                         "model_label": ["PROD", "DIST", "DIST", "DIST", "OTHER", "OTHER"]})
    new = pd.DataFrame({"company": ["MARICO"] * 4, "human_label": ["PROD", "PROD", "OTHER", "OTHER"],
                        "model_label": ["PROD", "PROD", "OTHER", "PROD"]})
    p = pd.DataFrame({"company": ["MARICO"] * 4, "pred": ["PROD", "PROD", "OTHER", "PROD"]})
    r = ha.compute(new, test, p)
    assert r["ai_vs_human"]["agreement"] == round(5 / 6, 4) and 0 < r["ai_vs_human"]["kappa"] < 1
    assert {x["company"] for x in r["per_company"]} == {"DABUR", "MARICO"}
    assert r["unseen_companies"]["n"] == 4 and r["test_set"]["n"] == 6


def test_export_never_overwrites_labels(tmp_path, monkeypatch):
    f = tmp_path / "verify.csv"
    pd.DataFrame({"passage_id": ["a"], "human_label": ["PROD"]}).to_csv(f, index=False, encoding="utf-8-sig")
    assert ha._has_labels(f)
    pd.DataFrame({"passage_id": ["a"], "human_label": [""]}).to_csv(f, index=False, encoding="utf-8-sig")
    assert not ha._has_labels(f)


def test_run_refuses_until_filled(tmp_path, monkeypatch):
    f = tmp_path / "v.csv"
    pd.DataFrame({"company": ["X", "X"], "human_label": ["PROD", ""], "notes": ["", ""]}).to_csv(
        f, index=False, encoding="utf-8-sig")
    with pytest.raises(ValueError, match="1 of 2 rows"):
        ha._load_filled(f, partial=False)
    assert len(ha._load_filled(f, partial=True)) == 1
