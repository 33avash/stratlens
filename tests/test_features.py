"""Phase 4: promises, follow-up wording, novelty and signature phrases (no network, no embedding model)."""
import numpy as np
import pandas as pd

from stratlens import features


def _p(company, year, texts, pred="SUST"):
    return pd.DataFrame({"passage_id": [f"{company}{year}{i}" for i in range(len(texts))], "company": company,
                         "year": year, "page": 10, "text": texts, "pred": pred, "confidence": 0.9,
                         "word_count": [len(t.split()) for t in texts]})


def test_promises_need_a_figure_and_a_deadline_or_commitment():
    df = _p("ACME", "FY2026", [
        "We aim to make 80% of our plastic packaging recyclable by 2030 across all markets.",
        "Project SETU aims to expand our direct reach from 1 million outlets to 1.5 million by FY28.",
        "Sales grew 12% during the year on the back of strong rural demand across our core brands.",
        "As per Nielsen, online FMCG sales could reach 11% by 2030 across the country.",
        "We beat our 2026 goal of 22% plastic reduction two years early this year.",
        "We will double down on the early success of our new insecticide launches this year.",
    ])
    pr = features.promises(df, "ACME|FY2026")
    got = {p["sentence"][:12]: (p["target"], p["deadline"]) for p in pr}
    assert got == {"We aim to ma": ("80%", "by 2030"), "Project SETU": ("1.5 million", "by FY28")}


def test_deadline_must_be_after_the_report_year():
    assert features.deadline("We will double capacity by FY26.", "FY2026", []) is None
    assert features.deadline("We will double capacity by FY27.", "FY2026", []) == "by FY27"
    assert features.deadline("We aim for net zero by 2040.", "FY2026", [["DATE", "2040"]]) == "by 2040"


def test_follow_up_never_claims_achievement(monkeypatch):
    a = _p("ACME", "FY2025", ["We will reach 1.5 million outlets by FY27 through Project SETU direct reach."])
    b = _p("ACME", "FY2026", ["Project SETU extended our direct reach to more outlets this year.",
                              "Our digital marketing capability grew strongly this year."])
    emb = {"ACMEFY20260": np.array([1.0, 0]), "ACMEFY20261": np.array([0.9, 0.44])}
    from stratlens import embeddings
    monkeypatch.setattr(embeddings, "encode", lambda texts, cache=True: np.array([[1.0, 0]] * len(texts)))
    fu = features.follow_up(pd.concat([a, b]), emb, features.promises(a, "ACME|FY2025"), "ACME|FY2026")
    assert fu[0]["status"] == "mentioned again" and fu[0]["match"] == "ACMEFY20260"
    assert {"setu", "reach"} <= set(fu[0]["shared"])
    assert all(x["status"] in ("mentioned again", "not mentioned") for x in fu)


def test_novelty_flags_passages_without_a_close_counterpart():
    a = _p("ACME", "FY2025", ["Old topic about plastic packaging waste and recycling in our plants."] * 3)
    b = _p("ACME", "FY2026", ["Old topic about plastic packaging waste and recycling in our plants this year.",
                              "A brand new quick commerce partnership reaching dark stores in many cities."])
    emb = {**{f"ACMEFY2025{i}": np.array([1.0, 0]) for i in range(3)},
           "ACMEFY20260": np.array([0.95, 0.31]), "ACMEFY20261": np.array([0.1, 0.99])}
    nov = features.novelty(pd.concat([a, b]), emb, "ACME|FY2025", "ACME|FY2026")
    assert nov["only_in_b"]["n"] == 1 and nov["only_in_a"]["n"] == 0
    assert nov["only_in_b"]["groups"][0]["items"][0]["passage_id"] == "ACMEFY20261"


def test_signature_phrases_are_distinctive_and_recur():
    a = _p("ACME", "FY2026", ["Project Setu widened direct reach in rural markets.",
                              "Direct reach through Project Setu doubled in rural markets."], "DIST")
    b = _p("BETA", "FY2026", ["Modern trade and quick commerce grew in metro markets.",
                              "Quick commerce partnerships expanded in metro markets."], "DIST")
    sig = features.signature(pd.concat([a, b]), "FY2026")
    assert "direct reach" in [x["phrase"] for x in sig["ACME"]["DIST"]]
    assert "quick commerce" in [x["phrase"] for x in sig["BETA"]["DIST"]]
    assert all(x["n"] >= 2 for v in sig.values() for xs in v.values() for x in xs)
