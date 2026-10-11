"""Phase 4 features, computed offline from passages + MiniLM embeddings (no new models).

novelty(a, b)      passages of B with no counterpart in A (max cosine < NOVELTY_THRESHOLD) and vice versa,
                   grouped by theme; each group labelled by its top TF-IDF bigram. Used for "What's new this year"
                   (A = same company last year) and Compare's "Only in A / Only in B".
promises(report)   sentences with a figure or target AND a future deadline or commitment language; the figure and
                   deadline are extracted with spaCy DATE / CARDINAL / PERCENT / MONEY entities plus regex.
follow_up(a, b)    each promise in A -> most similar passage in B: "mentioned again" or "not mentioned" (never
                   "achieved": a mention says nothing about delivery).
signature(...)     per company and theme, the bigrams/trigrams most distinctive against the other companies' passages
                   on the same theme (TF-IDF over one document per company); brand and company names removed.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from . import specificity
from .config import EVIDENCE_MIN_CONFIDENCE, STRATEGIC_THEMES
from .masking import BRAND_RE, COMPANY_RE, YEAR_RE
from .preprocess import split_sentences
from .report import clean

NOVELTY_THRESHOLD = 0.5   # chosen by the project author from examples at 0.4 / 0.5 / 0.6 (2026-10-11)
MIN_WORDS_SHOWN = 12      # very short passages are usually layout fragments; never shown as "new"
TOP_PER_GROUP = 5
SIGNATURE_N = 5
SENTINEL = " zzz "        # replaces removed names/numbers so n-grams never bridge across them


# ---------------------------------------------------------------- helpers
def _report(passages: pd.DataFrame, key: str) -> pd.DataFrame:
    c, y = key.split("|")
    return passages[(passages.company == c) & (passages.year == y)]


def _vecs(df: pd.DataFrame, emb: dict) -> np.ndarray:
    return np.stack([emb[i] for i in df.passage_id]) if len(df) else np.zeros((0, 384), dtype=np.float32)


def _strategic(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.pred.isin(STRATEGIC_THEMES) & (df.confidence >= EVIDENCE_MIN_CONFIDENCE)]


def scrub(text: str) -> str:
    """Lower-cased text with company/brand names, years and numbers replaced by a sentinel."""
    t = COMPANY_RE.sub(SENTINEL, clean(text))
    t = BRAND_RE.sub(SENTINEL, t)
    t = YEAR_RE.sub(SENTINEL, t)
    t = re.sub(r"\d[\d,.]*", SENTINEL, t)
    return t.lower()


GENERIC = {"key", "strong", "remain", "remained", "focused", "focus", "years", "ahead", "new", "led", "received",
           "positive", "delivered", "mega", "growth", "value", "enabled", "top", "continued", "significant", "year",
           "company", "business", "businesses", "well", "also", "including", "across", "various", "overall", "good",
           "high", "higher", "further", "like", "able", "help", "helped", "products", "product", "brands", "brand"}


def _generic(term: str) -> bool:
    """Filler phrases ('key growth', 'remain focused'): at least half their words are generic."""
    words = term.split()
    return sum(w in GENERIC for w in words) * 2 >= len(words)


def _keep_term(term: str) -> bool:
    return "zzz" not in term and not re.search(r"\d", term) and all(len(w) > 2 for w in term.split())


def group_label(texts: list[str], background: list[str]) -> str | None:
    """The bigram with the highest summed TF-IDF weight in `texts`, IDF learned on `background`."""
    if len(texts) < 2:
        return None  # a label drawn from a single passage is just two of its words
    vec = TfidfVectorizer(ngram_range=(2, 2), stop_words="english", sublinear_tf=True, min_df=1)
    try:
        vec.fit([scrub(t) for t in background])
    except ValueError:
        return None
    X = vec.transform([scrub(t) for t in texts])
    vocab = vec.get_feature_names_out()
    score = np.asarray(X.sum(axis=0)).ravel()
    df_in = np.asarray((X > 0).sum(axis=0)).ravel()
    need = 2 if len(texts) >= 3 else 1  # in larger groups the label must recur, not come from one passage
    for j in np.argsort(score)[::-1]:
        if score[j] <= 0:
            break
        if df_in[j] >= need and _keep_term(vocab[j]) and not _generic(vocab[j]):
            return str(vocab[j])
    return None


def group_label_any(texts: list[str], background: list[str]) -> str | None:
    """Bigram label, else the most characteristic single word (groups of two or more passages only)."""
    if len(texts) < 2:
        return None
    return group_label(texts, background) or _unigram_label(texts, background)


def _unigram_label(texts: list[str], background: list[str]) -> str | None:
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, token_pattern=r"(?u)\b[a-z][a-z\-]{3,}\b")
    try:
        vec.fit([scrub(t) for t in background])
    except ValueError:
        return None
    X = vec.transform([scrub(t) for t in texts])
    vocab = vec.get_feature_names_out()
    score = np.asarray(X.sum(axis=0)).ravel()
    df_in = np.asarray((X > 0).sum(axis=0)).ravel()
    need = 2 if len(texts) >= 3 else 1
    for j in np.argsort(score)[::-1]:
        if score[j] > 0 and df_in[j] >= need and "zzz" not in vocab[j] and vocab[j] not in GENERIC:
            return str(vocab[j])
    return None


# ---------------------------------------------------------------- novelty
def _only_in(src: pd.DataFrame, other: pd.DataFrame, emb: dict, background: list[str]) -> dict:
    """Strategy passages of `src` whose best match among all passages of `other` is below the threshold."""
    cand = _strategic(src)
    if cand.empty or other.empty:
        return {"n": 0, "n_strategic": int(len(cand)), "groups": []}
    sims = _vecs(cand, emb) @ _vecs(other, emb).T
    best = sims.max(axis=1)
    arg = sims.argmax(axis=1)
    cand = cand.assign(sim=best, match=other.passage_id.to_numpy()[arg])
    new = cand[cand.sim < NOVELTY_THRESHOLD]
    groups = []
    for t in STRATEGIC_THEMES:
        g = new[new.pred == t]
        if g.empty:
            continue
        shown = g[g.word_count >= MIN_WORDS_SHOWN].sort_values(["confidence", "sim"], ascending=[False, True])
        groups.append({"theme": t, "n": int(len(g)), "label": group_label_any(g.text.tolist(), background),
                       "items": [{"passage_id": r.passage_id, "page": int(r.page), "sim": round(float(r.sim), 3),
                                  "closest": r.match} for r in shown.head(TOP_PER_GROUP).itertuples()]})
    groups.sort(key=lambda x: -x["n"])
    return {"n": int(len(new)), "n_strategic": int(len(cand)), "groups": groups}


def novelty(passages: pd.DataFrame, emb: dict, a: str, b: str) -> dict:
    """{'only_in_b': ..., 'only_in_a': ...} for report keys a, b ('COMPANY|FYyyyy')."""
    ra, rb = _report(passages, a), _report(passages, b)
    background = pd.concat([ra, rb]).text.tolist()
    return {"a": a, "b": b, "threshold": NOVELTY_THRESHOLD,
            "only_in_b": _only_in(rb, ra, emb, background), "only_in_a": _only_in(ra, rb, emb, background)}


# ---------------------------------------------------------------- promises
PROMISE_RE = re.compile(  # commitment language with the company as subject, or an explicit commitment phrase
    r"\b(?:we|our company|the company|your company)\b[^.;]{0,50}?\b(?:will|shall|aims?|plans?|intends?|"
    r"(?:are|is|remain) committed|commits?|targets?|pledges?|endeavours?|expects? to|are on track)\b"
    r"|\b(?:our|the|its)\s+(?:target|goal|ambition|aim|commitment|aspiration)\s+(?:is|of|to|was set)\b"
    r"|\b(?:committed to|aims? to|aiming to|on track to|targets? to|targeting)\b", re.I)
SELF_RE = re.compile(r"\b(?:we|our|us|your company|the company)\b", re.I)
DONE_RE = re.compile(r"\b(?:beat|exceeded|surpassed|achieved|accomplished|met our)\b", re.I)
PAST_RE = re.compile(r"\b(?:was|were|grew|delivered|recorded|reported|stood at|remained|"
                     r"(?:has|have)\s+(?:\w+ly\s+)?(?!committed|pledged)\w+ed)\b", re.I)
DEADLINE_RE = re.compile(
    r"\b(?:by|until|before|in|within|over)\s+(?:the\s+)?(?:end\s+of\s+)?(?:(?:calendar\s+)?year\s+)?"
    r"(FY\s?'?(\d{2,4})|(20\d{2}))(?:[-–]\d{2,4})?\b"
    r"|\b(?:over|within|in)\s+the\s+next\s+(?:\w+|\d+)\s+(?:years?|quarters?|months?)\b"
    r"|\b(?:medium|long)[- ]term\b", re.I)
QTY_IN_SENT = re.compile(r"(?:[₹$`]\s?\d|\d[\d,.]*\s?(?:%|per\s?cent|bps|crores?|cr\b|lakhs?|million|billion|mn\b|bn\b|"
                         r"x\b|tonnes?|MT\b|outlets|villages|cities|districts|stores|plants?|hectares?|farmers)|"
                         r"\b\d+(?:\.\d+)?\s?(?:%|per cent)|\bdouble\b|\btriple\b|\bnet[- ]zero\b|\bhalve\b)", re.I)


def _fy_end(year: str) -> int:
    return int(year[2:])


def _year_of(m: re.Match) -> int | None:
    fy_num, cal = m.group(2), m.group(3)
    if cal:
        return int(cal)
    if fy_num:
        n = int(fy_num)
        return n + 2000 if n < 100 else n
    return None


def deadline(sentence: str, report_year: str, ents: list[list[str]]) -> str | None:
    """The future deadline in a sentence ('by FY27', 'by 2030', 'over the next three years'), or None.
    Dates at or before the report's financial-year end are past, not deadlines."""
    end = _fy_end(report_year)
    for m in DEADLINE_RE.finditer(sentence):
        y = _year_of(m)
        if y is None or y > end:
            return m.group(0).strip()
    for label, text in ents:  # spaCy DATE entities such as "2030" or "the next five years"
        if label == "DATE":
            ys = [int(x) for x in re.findall(r"20\d{2}", text)]
            if ys and min(ys) > end:
                return text.strip()
    return None


FULL_QTY_RE = re.compile(r"[\d,.]*\s?(?:crores?|cr\b|lakhs?|million|billion|%)?", re.I)


def target(sentence: str, ents: list[list[str]], dl: str | None = None) -> str | None:
    """The figure being promised, with its unit ('₹500 Crore', '1.5 million', '80%', 'net zero').
    With a deadline, the quantity closest before it ('from 1 million to 1.5 million by FY27' -> '1.5 million');
    otherwise the first quantity after the commitment word. Falls back to spaCy PERCENT / MONEY / QUANTITY /
    CARDINAL entities (years excluded)."""
    body = YEAR_RE.sub(lambda m: " " * len(m.group(0)), sentence)  # blank years but keep character positions
    found = []
    for m in QTY_IN_SENT.finditer(body):
        if m.group(0).lower() == "double" and body[m.end():m.end() + 5].lower() == " down":
            continue  # 'double down' is an idiom, not a target
        tail = FULL_QTY_RE.match(sentence, m.end())
        found.append((m.start(), (sentence[m.start():m.end()] + (tail.group(0) if tail else "")).strip(" .,;")))
    if found:
        pos = sentence.find(dl) if dl else -1
        if pos > 0:
            before = [f for f in found if f[0] < pos]
            if before:
                return before[-1][1]
        c = PROMISE_RE.search(sentence)
        after = [f for f in found if c and f[0] >= c.start()]
        return (after or found)[0][1]
    for label in ("PERCENT", "MONEY", "QUANTITY", "CARDINAL"):
        for lab, text in ents:
            if lab == label and not re.fullmatch(r"(?:FY)?\s?'?\d{2,4}", text.strip()):
                return text.strip()
    return None


def promises(passages: pd.DataFrame, key: str) -> list[dict]:
    rep = passages[passages.pred.isin(STRATEGIC_THEMES)]
    rep = rep[(rep.company == key.split("|")[0]) & (rep.year == key.split("|")[1])]
    cands = []
    for r in rep.itertuples():
        for s in split_sentences(clean(r.text)):
            if len(s.split()) >= 6 and QTY_IN_SENT.search(YEAR_RE.sub(" ", s)):
                cands.append((r, s))
    ents = specificity.number_entities([s for _, s in cands])
    out, seen = [], set()
    for (r, s), e in zip(cands, ents):
        # a promise = a figure or target AND (a future deadline OR commitment language), stated by the company
        dl = deadline(s, r.year, e)
        commit = PROMISE_RE.search(s)
        if not (dl or commit):
            continue
        if not commit and not SELF_RE.search(s):
            continue  # a dated forecast by someone else ('Nielsen expects ... by 2030') is not the company's promise
        vague = dl is not None and re.fullmatch(r"(?:medium|long)[- ]term", dl, re.I)
        if DONE_RE.search(s) or (PAST_RE.search(s) and (not dl or vague)):
            continue  # results already delivered are not promises
        if not target(s, e, dl):
            continue  # 'double down' and similar: no measurable figure after all
        k = re.sub(r"\W", "", s.lower())[:120]
        if k in seen:
            continue
        seen.add(k)
        out.append({"passage_id": r.passage_id, "page": int(r.page), "theme": r.pred, "sentence": s,
                    "target": target(s, e, dl), "deadline": dl, "confidence": round(float(r.confidence), 3)})
    out.sort(key=lambda x: (x["deadline"] is None, -x["confidence"]))
    return out


MIN_SHARED_WORDS = 2


def content_words(text: str) -> set[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return {w for w in re.findall(r"[a-z][a-z\-]{3,}", scrub(text))
            if w not in ENGLISH_STOP_WORDS and w not in GENERIC and w != "zzz"}


def follow_up(passages: pd.DataFrame, emb: dict, prom: list[dict], b: str) -> list[dict]:
    """For each promise (from an earlier report), the closest passage of report b that is similar in meaning
    (cosine >= NOVELTY_THRESHOLD) AND shares at least MIN_SHARED_WORDS content words with the promise, so a
    generally similar passage on another topic does not count as a mention."""
    from . import embeddings
    rb = _report(passages, b)
    if not prom or rb.empty:
        return []
    S = embeddings.encode([p["sentence"] for p in prom]) @ _vecs(rb, emb).T
    words_b = [content_words(t) for t in rb.text]
    out = []
    for p, row in zip(prom, S):
        want = content_words(p["sentence"])
        match, sim = None, float(row.max())
        for j in np.argsort(row)[::-1]:
            if row[j] < NOVELTY_THRESHOLD:
                break
            if len(want & words_b[j]) >= MIN_SHARED_WORDS:
                match, sim = rb.passage_id.iloc[j], float(row[j])
                break
        out.append({**p, "status": "mentioned again" if match else "not mentioned", "match": match,
                    "match_sim": round(sim, 3),
                    "shared": sorted(want & words_b[rb.passage_id.tolist().index(match)])[:5] if match else []})
    return out


# ---------------------------------------------------------------- signature phrases
def signature(passages: pd.DataFrame, year: str) -> dict:
    """{company: {theme: [{"phrase", "n", "passage_ids"}]}} for one year."""
    df = passages[(passages.year == year) & passages.pred.isin(STRATEGIC_THEMES)]
    out: dict = {}
    for t in STRATEGIC_THEMES:
        part = df[df.pred == t]
        cos = sorted(part.company.unique())
        if len(cos) < 2:
            continue
        docs = [" . ".join(scrub(x) for x in part[part.company == c].text) for c in cos]
        vec = TfidfVectorizer(ngram_range=(2, 3), stop_words="english", sublinear_tf=True, min_df=1,
                              token_pattern=r"(?u)\b[a-z][a-z\-]+\b")
        X = vec.fit_transform(docs).toarray()
        vocab = vec.get_feature_names_out()
        for i, c in enumerate(cos):
            texts = part[part.company == c]
            low = [scrub(x) for x in texts.text]
            picked = []
            for j in np.argsort(X[i])[::-1]:
                term = str(vocab[j])
                if X[i, j] <= 0 or len(picked) >= SIGNATURE_N:
                    break
                if not _keep_term(term) or _generic(term) or any(set(term.split()) & set(p["phrase"].split()) for p in picked):
                    continue
                ids = [pid for pid, lt in zip(texts.passage_id, low) if re.search(rf"\b{re.escape(term)}\b", lt)]
                if len(ids) < 2:  # a signature phrase must recur, not appear once
                    continue
                picked.append({"phrase": term, "n": len(ids), "passage_ids": ids[:5]})
            if picked:
                out.setdefault(c, {})[t] = picked
    return out


# ---------------------------------------------------------------- all
def compute_all(passages: pd.DataFrame, emb: dict, an: dict, pairs: list[tuple[str, str]]) -> dict:
    keys = sorted(an["reports"])
    prom = {k: promises(passages, k) for k in keys}
    fu = {}
    for c, y in an["yoy"].items():
        a, b = f"{c}|{y['from']}", f"{c}|{y['to']}"
        fu[b] = follow_up(passages, emb, prom.get(a, []), b)
    return {"threshold": NOVELTY_THRESHOLD,
            "novelty": {f"{a}~{b}": novelty(passages, emb, a, b) for a, b in pairs},
            "promises": prom, "follow_up": fu,
            "signature": {y: signature(passages, y) for y in an["years"]}}
