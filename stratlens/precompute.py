"""Offline text generation for the app (run by `python -m stratlens.build`).

  narratives.json   brief per report (3 sentences) and summaries for the Compare presets and every company's
                    year-on-year pair; each {"text", "source": "AI-polished" | "template"}
  ask_examples.json the Ask page's example questions: retrieved passage ids + answer
  features.json     what's new / dropped (year-on-year and Compare presets), promises and their follow-up,
                    signature phrases (see features.py)

All LLM calls go through the on-disk cache (data/cache/llm.json), so a rebuild with unchanged inputs makes no
network calls and the app needs none for these texts.
"""
from __future__ import annotations

import json

import pandas as pd

import numpy as np

from . import analytics, features, insights
from .config import ASK_EXAMPLES, ASK_EXAMPLES_JSON, COMPARE_PRESETS, EMBEDDINGS_NPZ, FEATURES_JSON, NARRATIVES_JSON
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


def load_embeddings() -> dict:
    z = np.load(EMBEDDINGS_NPZ, allow_pickle=False)
    return dict(zip(z["passage_id"].tolist(), z["vecs"]))


def write_features(an: dict, passages: pd.DataFrame) -> dict:
    f = features.compute_all(passages, load_embeddings(), an, compare_pairs(an))
    FEATURES_JSON.write_text(json.dumps(f, ensure_ascii=False), encoding="utf-8")
    return {"promises": sum(len(v) for v in f["promises"].values()),
            "new/dropped pairs": len(f["novelty"])}


def write_all(an: dict, passages: pd.DataFrame) -> dict:
    fstats = write_features(an, passages)
    client = get_client()
    nar = {"brief": {}, "compare": {}}
    examples = {}
    # Order = demo path first, so a daily API quota that runs out mid-build leaves only the least-seen texts as
    # templates (they are filled in on a later `python -m stratlens.build --narratives`; cached ones cost nothing).
    presets = [(p["a"], p["b"]) for p in COMPARE_PRESETS if p["a"] in an["reports"] and p["b"] in an["reports"]]
    latest = max(an["years"])
    briefs = sorted(an["reports"], key=lambda k: (k.split("|")[1] != latest, k))
    tasks = ([("brief", presets[0][1])] if presets else []) + [("compare", presets[0])] * bool(presets) \
        + [("ask", q) for q in ASK_EXAMPLES] + [("compare", p) for p in presets[1:]] \
        + [("brief", k) for k in briefs] + [("compare", p) for p in compare_pairs(an)]
    for kind, item in tasks:
        if kind == "brief" and item not in nar["brief"]:
            tpl, tok = insights.brief_template(an, *item.split("|"))
            text, source = narrate.narrate(tpl, tok, client)
            nar["brief"][item] = {"text": text, "source": source}
        elif kind == "compare" and f"{item[0]}~{item[1]}" not in nar["compare"]:
            nar["compare"][f"{item[0]}~{item[1]}"] = compare_text(an, item[0], item[1], client)
        elif kind == "ask" and item not in examples:
            hits, method = retrieval.search(passages, item, k=8)
            srcs = ask.sources(hits)
            examples[item] = {"passage_ids": [s["passage_id"] for s in srcs], "method": method,
                              "relevance": hits["relevance"].round(3).tolist(), **ask.answer(item, srcs, client)}
    NARRATIVES_JSON.write_text(json.dumps(nar, ensure_ascii=False, indent=1), encoding="utf-8")
    ASK_EXAMPLES_JSON.write_text(json.dumps(examples, ensure_ascii=False, indent=1), encoding="utf-8")
    n_ai = sum(v["source"] != "template" for d in nar.values() for v in d.values())
    n_all = sum(len(d) for d in nar.values())
    return {**fstats, "narratives": f"{n_ai}/{n_all} AI-polished", "example answers":
            f"{sum(v['status'] == 'ok' for v in examples.values())}/{len(examples)} answered"}


def complete() -> bool:
    return NARRATIVES_JSON.exists() and ASK_EXAMPLES_JSON.exists() and FEATURES_JSON.exists()
