"""Plain-English insights built from analytics.json: Home finding cards, Company briefs and Compare summaries.

Every number in the text comes from analytics; the text itself is a template. Templates mark each number as a
token like [[n1]] so an optional LLM rewrite (llm/narrate.py) can rephrase the words but never the numbers.
`fill()` turns a (template, tokens) pair into the final sentence.
"""
from __future__ import annotations

import re

from .analytics import report_key
from .config import COMPANY_NAMES, COMPANY_SHORT, STRATEGIC_THEMES, THEME_NAMES, peers_of

TOKEN_RE = re.compile(r"\[\[(\w+)\]\]")


def short(code: str) -> str:
    return COMPANY_SHORT.get(code, code.title())  # unknown (uploaded) companies: 'TESTCO' -> 'Testco'



def tname(theme: str) -> str:
    return THEME_NAMES[theme].lower()


def pct(x: float | None) -> str:
    return "–" if x is None else f"{round(x * 100)}%"


def fill(template: str, tokens: dict) -> str:
    return TOKEN_RE.sub(lambda m: str(tokens[m.group(1)]), template)


class _T:
    """Collects tokens while a template is written: t.v('34%') -> '[[n1]]'."""

    def __init__(self):
        self.tokens: dict[str, str] = {}

    def v(self, value) -> str:
        k = f"n{len(self.tokens) + 1}"
        self.tokens[k] = str(value)
        return f"[[{k}]]"


def latest_year(an: dict) -> str:
    return max(y for y in an["years"] if any(k.endswith("|" + y) for k in an["reports"]))


def prev_year(an: dict, year: str) -> str | None:
    earlier = [y for y in an["years"] if y < year]
    return earlier[-1] if earlier else None


def _ratio_words(r: float) -> str | None:
    if r >= 2.75:
        return f"{round(r)} times as much as"
    if r >= 1.75:
        return "about twice as much as"
    return None


def _peer_label(an: dict, code: str, year: str) -> str | None:
    peers = [p for p in peers_of(code) if report_key(p, year) in an["reports"]]
    return " and ".join(short(p) for p in peers) if peers else None


# ---------------------------------------------------------------- Home findings
def findings(an: dict) -> list[dict]:
    """Up to three cards: biggest lean vs peers, biggest real year-on-year shift, most concrete company.
    Each: title (the insight), detail, link (query string for the evidence page)."""
    year = latest_year(an)
    out = []

    best = None
    for key, r in an["reports"].items():
        code, y = key.split("|")
        if y != year or not r.get("peer_avg"):
            continue
        for t in STRATEGIC_THEMES:
            s, p = r["themes"][t]["share"], r["peer_avg"]["themes"][t]["share"]
            if r["themes"][t]["count"] >= 5 and (best is None or s - p > best[0]):
                best = (s - p, code, t, s, p)
    if best and best[0] > 0:
        _, code, t, s, p = best
        peer = _peer_label(an, code, year)
        rw = _ratio_words(s / p) if p > 0 else None
        title = (f"{short(code)} talks about {tname(t)} {rw} {peer}" if rw else
                 f"{short(code)} gives {tname(t)} more space than {peer}")
        out.append({"kind": "lean", "title": title, "theme": t, "company": code, "year": year,
                    "detail": f"{pct(s)} of its {year} strategy passages, against {pct(p)} for {peer}.",
                    "link": f"company?company={code}&year={year}&theme={t}"})

    shifts = [(abs(d["pp"]), c, t, d, y) for c, y in an["yoy"].items() for t, d in y["themes"].items()
              if d["significant"]]
    if shifts:
        _, code, t, d, y = max(shifts)
        r0 = an["reports"][report_key(code, y["from"])]["themes"][t]["share"]
        r1 = an["reports"][report_key(code, y["to"])]["themes"][t]["share"]
        if r0 > 0 and r1 / r0 >= 1.75:
            title = f"{short(code)} {'more than doubled' if r1 / r0 >= 2.2 else 'doubled'} its focus on {tname(t)}"
        elif r1 < r0:
            title = f"{short(code)} cut back on {tname(t)}"
        else:
            title = f"{short(code)} turned towards {tname(t)}"
        out.append({"kind": "shift", "title": title, "theme": t, "company": code, "year": y["to"],
                    "detail": f"{pct(r0)} of its strategy passages in {y['from']}, {pct(r1)} in {y['to']}. "
                              "Bigger than chance.",
                    "link": f"compare?a={code}|{y['from']}&b={code}|{y['to']}"})
    elif an["yoy"]:
        out.append({"kind": "shift", "title": "No company's emphasis moved beyond chance", "theme": None,
                    "company": None, "year": year, "detail": "Every year-on-year change is within noise.",
                    "link": "compare"})

    conc = [(r["concrete_pct"], k.split("|")[0]) for k, r in an["reports"].items()
            if k.endswith("|" + year) and r.get("concrete_pct") is not None]
    if conc:
        top, code = max(conc)
        avg = sum(c for c, _ in conc) / len(conc)
        out.append({"kind": "concrete", "title": f"{short(code)} backs its strategy with the most specifics",
                    "theme": None, "company": code, "year": year,
                    "detail": f"{pct(top)} of its {year} strategy passages name figures, dates, initiatives or "
                              f"commitments, against {pct(avg)} across all {len(conc)} companies.",
                    "link": f"company?company={code}&year={year}"})
    return out


# ---------------------------------------------------------------- Company brief
def brief_template(an: dict, code: str, year: str) -> tuple[str, dict]:
    """Three sentences: what it emphasises, how it differs from peers, what changed."""
    r = an["reports"][report_key(code, year)]
    T = _T()
    ranked = sorted(STRATEGIC_THEMES, key=lambda t: -r["themes"][t]["share"])
    a, b = ranked[0], ranked[1]
    s1 = (f"In {year}, {COMPANY_NAMES.get(code, code.title())} gives the most space to {tname(a)} "
          f"({T.v(pct(r['themes'][a]['share']))} of its strategy passages), followed by {tname(b)} "
          f"({T.v(pct(r['themes'][b]['share']))}).")

    peer = _peer_label(an, code, year)
    if r.get("peer_avg") and peer:
        t = max(STRATEGIC_THEMES, key=lambda t: r["themes"][t]["share"] - r["peer_avg"]["themes"][t]["share"])
        s2 = (f"Compared with {peer}, it leans most towards {tname(t)} ({T.v(pct(r['themes'][t]['share']))} "
              f"against {T.v(pct(r['peer_avg']['themes'][t]['share']))}).")
    else:
        names = " and ".join(short(p) for p in peers_of(code))
        s2 = (f"There is no {names} report for {year} to compare it with." if names else
              "It has no peer group, so it is compared with the average of all companies.")

    prev = prev_year(an, year)
    y = an["yoy"].get(code)
    if y and y["to"] == year:
        sig = [(abs(d["pp"]), t, d) for t, d in y["themes"].items() if d["significant"]]
        if sig:
            _, t, d = max(sig)
            r0 = an["reports"][report_key(code, y["from"])]["themes"][t]["share"]
            word = "rose" if d["pp"] > 0 else "fell"
            s3 = (f"Since {y['from']}, its emphasis on {tname(t)} {word} from {T.v(pct(r0))} to "
                  f"{T.v(pct(r['themes'][t]['share']))}, a change bigger than chance.")
        else:
            s3 = f"Since {y['from']}, no theme's share has changed by more than chance."
    elif prev and report_key(code, prev) not in an["reports"]:
        s3 = f"The {prev} report is missing, so changes over time can't be shown."
    else:
        s3 = "This is the earliest report, so changes over time can't be shown yet."
    return " ".join([s1, s2, s3]), T.tokens


# ---------------------------------------------------------------- Compare summary
def report_label(key: str) -> str:
    code, year = key.split("|")
    return f"{short(code)} {year}"


def compare_template(an: dict, a: str, b: str, diffs: dict) -> tuple[str, dict]:
    """One paragraph on how report B differs from report A (diffs = analytics.pair(an, a, b))."""
    T = _T()
    ra, rb = an["reports"][a], an["reports"][b]
    la, lb = report_label(a), report_label(b)
    sig = sorted(((abs(d["pp"]), t) for t, d in diffs.items() if d["significant"]), reverse=True)
    if sig:
        t = sig[0][1]
        s = [f"The biggest real difference is {tname(t)}: {T.v(pct(ra['themes'][t]['share']))} of {la}'s strategy "
             f"passages against {T.v(pct(rb['themes'][t]['share']))} for {lb}."]
        if len(sig) > 1:
            others = ", ".join(tname(x) for _, x in sig[1:3])
            s.append(f"{others[0].upper() + others[1:]} also differ{'s' if len(sig) == 2 else ''} by more than "
                     f"chance; the other themes are within noise.")
        else:
            s.append("Every other theme is within noise.")
    else:
        s = [f"No theme's share differs between {la} and {lb} by more than chance."]
    ca, cb = ra.get("concrete_pct"), rb.get("concrete_pct")
    if ca is not None and cb is not None and round(ca * 100) != round(cb * 100):
        more, less, hi, lo = (la, lb, ca, cb) if ca > cb else (lb, la, cb, ca)
        s.append(f"{more} is more specific: {T.v(pct(hi))} of its strategy passages name figures, dates or "
                 f"initiatives, against {T.v(pct(lo))} for {less}.")
    return " ".join(s), T.tokens
