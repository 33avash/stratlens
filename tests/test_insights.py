"""Phase 3: insight text is built only from analytics numbers; the LLM rewrite can never add a number."""
import re

import pandas as pd

from stratlens import analytics, insights
from stratlens.llm import narrate


def _an():
    def rep(c, y, labels, spec=2):
        return pd.DataFrame({"company": c, "year": y, "pred": labels, "specificity": spec, "needs_review": False,
                             "passage_id": [f"{c}{y}{i}" for i in range(len(labels))]})
    parts = [rep("DABUR", "FY2025", ["BRAND"] * 20 + ["DIGI"] * 4 + ["PROD"] * 16),
             rep("DABUR", "FY2026", ["BRAND"] * 20 + ["DIGI"] * 16 + ["PROD"] * 4, spec=3),
             rep("MARICO", "FY2026", ["BRAND"] * 6 + ["PROD"] * 24 + ["DIGI"] * 10, spec=1)]
    return analytics.compute(pd.concat(parts, ignore_index=True))


def test_findings_are_insights_with_links():
    f = insights.findings(_an())
    kinds = [x["kind"] for x in f]
    assert kinds == ["lean", "shift", "concrete"]
    # Marico: 60% product vs Dabur's 10% is the largest lean over a peer
    assert f[0]["title"] == "Marico talks about product innovation & portfolio 6 times as much as Dabur"
    assert f[0]["link"] == "company?company=MARICO&year=FY2026&theme=PROD"
    assert f[1]["title"] == "Dabur cut back on product innovation & portfolio"  # 40% -> 10%, the largest shift
    assert f[1]["link"].startswith("compare?a=DABUR|FY2025")


def test_brief_numbers_come_from_tokens():
    an = _an()
    tpl, tok = insights.brief_template(an, "DABUR", "FY2026")
    text = insights.fill(tpl, tok)
    assert "50%" in text and "Since FY2025" in text
    assert not re.search(r"\d", re.sub(r"\[\[n\d+\]\]|FY\d{4}", "", tpl))  # every number is a token


class FakeLLM:
    def __init__(self, out):
        self.out = out

    def generate(self, system, prompt, max_tokens=400):
        return self.out

    def cached(self, system, prompt):
        return self.out


def test_narrate_rejects_rewrites_that_add_or_drop_numbers():
    tpl, tok = "It rose from [[n1]] to [[n2]] in FY2026.", {"n1": "6%", "n2": "14%"}
    ok, src = narrate.narrate(tpl, tok, FakeLLM("Its share climbed from [[n1]] to [[n2]] in FY2026."))
    assert src == "AI-polished" and ok == "Its share climbed from 6% to 14% in FY2026."
    for bad in ["It rose from [[n1]] to 15% in FY2026.", "It rose to [[n2]].", "It rose 8 points from [[n1]] to [[n2]]."]:
        text, src = narrate.narrate(tpl, tok, FakeLLM(bad))
        assert src == "template" and text == "It rose from 6% to 14% in FY2026."


def test_compare_summary_flags_noise():
    an = _an()
    tpl, tok = insights.compare_template(an, "DABUR|FY2025", "DABUR|FY2026",
                                         analytics.pair(an, "DABUR|FY2025", "DABUR|FY2026"))
    text = insights.fill(tpl, tok)
    assert text.startswith("The biggest real difference is")
