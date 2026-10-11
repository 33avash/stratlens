"""Offline text generation for the app (run by `python -m stratlens.build`).

  narratives.json   brief per report (3 sentences) and summaries for the Compare presets and every company's
                    year-on-year pair; each {"text", "source": "AI-polished" | "template"}
  ask_examples.json the Ask page's example questions: retrieved passage ids + answer

All LLM calls go through the on-disk cache (data/cache/llm.json), so a rebuild with unchanged inputs makes no
network calls and the app needs none for these texts.
"""
from __future__ import annotations

import json

import pandas as pd

from . import analytics, insights
from .config import ASK_EXAMPLES, ASK_EXAMPLES_JSON, COMPARE_PRESETS, NARRATIVES_JSON
from .llm import answer as ask
from .llm import narrate, retrieval
from .llm.client import get_client


def compare_pairs(an: dict) -> list[tuple[str, str]]:
    pairs = [(p["a"], p["b"]) for p in COMPARE_PRESETS if p["a"] in an["reports"] and p["b"] in an["reports"]]
    pairs += [(f"{c}|{y['from']}", f"{c}|{y['to']}") for c, y in an["yoy"].items()]
    return list(dict.fromkeys(pairs))


def compare_text(an: dict, a: str, b: str, client) -> dict:
    tpl, tok = insights.compare_template(an, a, b, analytics.pair(an, a, b))
    text, source = narrate.narrate(tpl, tok, client)
    return {"text": text, "source": source}


def write_all(an: dict, passages: pd.DataFrame) -> dict:
    client = get_client()
    nar = {"brief": {}, "compare": {}}
    for key in an["reports"]:
        tpl, tok = insights.brief_template(an, *key.split("|"))
        text, source = narrate.narrate(tpl, tok, client)
        nar["brief"][key] = {"text": text, "source": source}
    for a, b in compare_pairs(an):
        nar["compare"][f"{a}~{b}"] = compare_text(an, a, b, client)
    NARRATIVES_JSON.write_text(json.dumps(nar, ensure_ascii=False, indent=1), encoding="utf-8")

    examples = {}
    for q in ASK_EXAMPLES:
        hits, method = retrieval.search(passages, q, k=8)
        srcs = ask.sources(hits)
        examples[q] = {"passage_ids": [s["passage_id"] for s in srcs], "method": method,
                       "relevance": hits["relevance"].round(3).tolist(), **ask.answer(q, srcs, client)}
    ASK_EXAMPLES_JSON.write_text(json.dumps(examples, ensure_ascii=False, indent=1), encoding="utf-8")
    n_ai = sum(v["source"] != "template" for d in nar.values() for v in d.values())
    n_all = sum(len(d) for d in nar.values())
    return {"narratives": f"{n_ai}/{n_all} AI-polished", "example answers":
            f"{sum(v['status'] == 'ok' for v in examples.values())}/{len(examples)} answered"}


def complete() -> bool:
    return NARRATIVES_JSON.exists() and ASK_EXAMPLES_JSON.exists()
