"""Passage retrieval for Ask.

Searches the original (unmasked) text of every passage, strategic and Other alike.

Hybrid ranking:
  - BM25 over unigrams and bigrams, so exact phrases such as "quick commerce" rank highly
  - MiniLM embedding cosine, so paraphrases are found too
  - the two rankings are fused by reciprocal rank fusion: 1/(60 + bm25 rank) + 1/(60 + embedding rank),
    reported as `relevance` = fused score / best possible fused score (1.0 = ranked first by both)
Relevance gate: a passage can be returned only if its embedding cosine clears EMB_FLOOR, or it contains a
two-word phrase from the question, or it contains most of the question's content words. If nothing clears the
gate the caller refuses instead of answering.
If the question names one company and no company filter is set, retrieval is limited to that company and the
name is removed from the query, so passages are matched on topic rather than on the company name.
Without the embedding model, ranking and the gate use BM25 and keyword matches only.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .. import embeddings
from ..masking import COMPANY_RE

EMB_FLOOR = 0.40     # MiniLM cosine. Calibrated: on-topic queries score 0.56-0.67, off-topic ones at most 0.36
LEX_COVERAGE = 0.6   # share of the question's content words a passage must contain to pass on keywords alone
RRF_K = 60
BM25_K1, BM25_B = 1.5, 0.75
COMPANY_ALIASES = {"HUL": ["hul", "hindustan unilever", "unilever"], "DABUR": ["dabur"],
                   "NESTLE": ["nestlé", "nestle"]}
QUESTION_WORDS = {"how", "what", "why", "which", "who", "does", "did", "do", "say", "says", "talk", "talking",
                  "about", "company", "companies", "report", "reports", "mention", "mentions", "describe"}
TOKEN_RE = re.compile(r"[a-z0-9]+")


def company_in_query(query: str) -> str | None:
    q = query.lower()
    hits = [c for c, names in COMPANY_ALIASES.items() if any(re.search(rf"\b{re.escape(n)}\b", q) for n in names)]
    return hits[0] if len(hits) == 1 else None


def tokens(text: str, query: bool = False) -> list[str]:
    """Lower-cased content words (stop words removed; question words too when `query`)."""
    drop = ENGLISH_STOP_WORDS | QUESTION_WORDS if query else ENGLISH_STOP_WORDS
    return [t for t in TOKEN_RE.findall(text.lower().replace("’", "'")) if t not in drop]


def terms(text: str, query: bool = False) -> tuple[list[str], list[str]]:
    """(unigrams, bigrams). Bigrams join adjacent content words, e.g. 'quick commerce' -> 'quick_commerce'."""
    uni = tokens(text, query)
    return uni, [f"{a}_{b}" for a, b in zip(uni, uni[1:])]


def bm25(docs: list[list[str]], query: list[str]) -> np.ndarray:
    n = len(docs)
    if n == 0 or not query:
        return np.zeros(n)
    avgdl = sum(map(len, docs)) / n or 1.0
    df = Counter(t for d in docs for t in set(d))
    tfs = [Counter(d) for d in docs]
    out = np.zeros(n)
    for t in set(query):
        if not df[t]:
            continue
        idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
        for i, tf in enumerate(tfs):
            f = tf.get(t, 0)
            if f:
                out[i] += idf * f * (BM25_K1 + 1) / (f + BM25_K1 * (1 - BM25_B + BM25_B * len(docs[i]) / avgdl))
    return out


def _ranks(scores: np.ndarray) -> np.ndarray:
    """1 = best. Ties share the better rank."""
    order = (-scores).argsort(kind="stable")
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def search(passages: pd.DataFrame, query: str, k: int = 8, company: str | None = None,
           year: str | None = None) -> tuple[pd.DataFrame, str]:
    """Top-k relevant passages with `relevance` (0-1), `bm25_rank`, `emb_rank` and `emb_cosine`, plus a method note."""
    note = ""
    if not company:
        company = company_in_query(query)
        if company:
            note = f"; limited to {company} because the question names it"
    topic = re.sub(r"\s+", " ", COMPANY_RE.sub(" ", query)).strip() or query
    df = passages
    if company:
        df = df[df.company == company]
    if year:
        df = df[df.year == year]
    q_uni, q_bi = terms(topic, query=True)
    empty = df.head(0).assign(relevance=[], bm25_rank=[], emb_rank=[], emb_cosine=[])
    if df.empty or not q_uni:
        return empty, "none"

    texts = df["text"].tolist()
    doc_terms = [terms(t) for t in texts]
    lex = bm25([u + b for u, b in doc_terms], q_uni + q_bi)
    phrase = np.array([bool(set(q_bi) & set(b)) for _, b in doc_terms])
    coverage = np.array([len(set(q_uni) & set(u)) / len(set(q_uni)) for u, _ in doc_terms])
    keyword_ok = phrase | (coverage >= LEX_COVERAGE)
    r_lex = _ranks(lex)

    if embeddings.available():
        emb = embeddings.encode(texts) @ embeddings.encode([topic], cache=False)[0]
        r_emb = _ranks(emb)
        fused = 1 / (RRF_K + r_lex) + 1 / (RRF_K + r_emb)
        best = 2 / (RRF_K + 1)
        keep = (emb >= EMB_FLOOR) | keyword_ok
        method = "hybrid: BM25 + embeddings, rank fusion" + note
    else:
        emb, r_emb = np.full(len(texts), np.nan), np.zeros(len(texts), dtype=int)
        fused, best = 1 / (RRF_K + r_lex), 1 / (RRF_K + 1)
        keep = keyword_ok
        method = "BM25 only (embedding model unavailable)" + note
    out = df.assign(relevance=np.round(fused / best, 3), bm25_rank=r_lex, emb_rank=r_emb,
                    emb_cosine=np.round(emb, 3))[keep]
    return out.sort_values("relevance", ascending=False).head(k), method
