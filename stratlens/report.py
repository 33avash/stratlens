"""Strategy briefs: computed facts -> narrative (template, or Gemini with a number check) -> HTML / PDF.

All figures come from benchmark predictions and metrics.json. The LLM, when configured, only rewrites the
narrative from the facts JSON; sections whose numbers cannot be matched to the facts fall back to template text.
"""
from __future__ import annotations

import html
import json
import re
from datetime import date

import pandas as pd

from . import compare, specificity, uncertainty
from .config import COMPANY_NAMES, ROOT, SECTION_NAMES, STRATEGIC_THEMES, THEME_NAMES
from .llm import prompts
from .llm.client import LLMError

FONT_DIR = ROOT / "assets" / "fonts"
CAVEAT = ("Theme shares measure disclosure emphasis: how much of each report's strategy narrative covers a theme. "
          "They say nothing about spending, execution or results, and are not a ranking.")


RUPEE_RE = re.compile(r"`(?=\s?\d)")


def clean(text: str) -> str:
    """Display fix: some PDFs extract the rupee sign as a backtick ('`60 crore')."""
    text = RUPEE_RE.sub("₹", text)
    return re.sub("[\x00-\x08\x0b\x0c\x0e-\x1f\ue000-\uf8ff]", "", text).strip()  # control chars and private-use glyphs (PDF bullet icons)


def cname(code: str) -> str:
    return COMPANY_NAMES.get(code, code.title())


def tname(code: str) -> str:
    return THEME_NAMES[code].lower()


def pct(x: float) -> int:
    return int(round(x * 100))


# ---------------------------------------------------------------- facts
def _spec_profile(df: pd.DataFrame) -> dict:
    s = df[df.pred.isin(STRATEGIC_THEMES)]
    by = []
    for t in STRATEGIC_THEMES:
        part = s[s.pred == t]
        if len(part) >= 3:
            by.append({"theme": THEME_NAMES[t], "code": t, "n": int(len(part)),
                       "mean_score": round(float(part.specificity.mean()), 2),
                       "concrete_pct": pct((part.specificity >= 2).mean())})
    ranked = sorted(by, key=lambda r: r["mean_score"])
    return {"mean_score": round(float(s.specificity.mean()), 2) if len(s) else None,
            "concrete_pct": pct((s.specificity >= 2).mean()) if len(s) else None,
            "specific_pct": pct((s.spec_tier == "Specific").mean()) if len(s) else None,
            "by_theme": by,
            "most_concrete": ranked[-1]["theme"] if ranked else None,
            "least_concrete": ranked[0]["theme"] if ranked else None}


def _evidence(df: pd.DataFrame, theme: str, n: int = 3) -> list[dict]:
    part = df[df.pred == theme].sort_values(["specificity", "confidence"], ascending=False).head(n)
    return [{"text": clean(r.text), "company": r.company, "company_name": cname(r.company), "year": r.year,
             "section": SECTION_NAMES.get(r.section, r.section), "page": int(r.page),
             "confidence": round(float(r.confidence), 2), "spec_tier": r.spec_tier,
             "spec_signals": specificity.describe(getattr(r, "spec_signals", None))} for r in part.itertuples()]


ONE_YEAR = "Only one year available, so there is no year-over-year comparison."
LABELS_NOTE = ("Training labels were AI-drafted following the labelling guidelines and have not been "
               "human-verified.")


def report_facts(preds: pd.DataFrame, company: str, year: str, metrics: dict | None = None,
                 n_evidence: int = 3, exclude_review: bool = False) -> dict:
    """Facts for one report. Counts use compare's definitions, so they match the Overview for the same filter."""
    preds = compare.in_scope(preds, exclude_review)
    shares = compare.theme_shares(preds)
    me = shares[(shares.company == company) & (shares.year == year)].set_index("theme")
    if me.empty:
        raise ValueError(f"No strategic passages for {company} {year}.")
    peers = shares[(shares.year == year) & (shares.company != company)]
    peer_names = sorted(peers.company.unique())
    peer_avg = peers.groupby("theme")["share"].mean() if not peers.empty else None
    themes = []
    for t in STRATEGIC_THEMES:
        row = {"theme": THEME_NAMES[t], "code": t, "share_pct": pct(me.loc[t, "share"]), "count": int(me.loc[t, "count"])}
        if peer_avg is not None:
            row["peer_avg_pct"] = pct(peer_avg[t])
            row["diff_vs_peers_pp"] = round((me.loc[t, "share"] - peer_avg[t]) * 100, 1)
        themes.append(row)
    by_share = sorted(themes, key=lambda r: -r["share_pct"])
    distinctive = max(themes, key=lambda r: r.get("diff_vs_peers_pp", -999)) if peer_avg is not None else None

    rep = preds[(preds.company == company) & (preds.year == year)]
    yoy = compare.yoy_change(shares[shares.company == company])
    has_prior = not yoy.empty and yoy["year_to"].iloc[0] == year
    sig = []
    if has_prior:
        y = uncertainty.yoy_ci(preds[preds.company == company], yoy)
        for r in y[y.significant].sort_values("pp_change", key=abs, ascending=False).itertuples():
            sig.append({"theme": THEME_NAMES[r.theme], "from_year": r.year_from, "from_pct": pct(r.share_from),
                        "to_pct": pct(r.share_to), "change_pp": round(r.pp_change, 1),
                        "ci_pp": [round(r.ci_lo, 1), round(r.ci_hi, 1)]})
    facts = {
        "company": company, "company_name": cname(company), "year": year,
        "n_passages": int(len(rep)), "n_strategic": int(me["strategic_total"].iloc[0]),
        "peers": [cname(p) for p in peer_names], "themes": themes,
        "top_themes": [r["theme"] for r in by_share[:2]],
        "least_theme": by_share[-1]["theme"],
        "distinctive_theme": ({"theme": distinctive["theme"], "diff_vs_peers_pp": distinctive["diff_vs_peers_pp"]}
                              if distinctive else None),
        "has_prior_year": bool(has_prior),
        "prior_year": yoy["year_from"].iloc[0] if has_prior else None,
        "significant_shifts": sig,
        "specificity": _spec_profile(rep),
        "evidence": {THEME_NAMES[t["code"]]: _evidence(rep, t["code"], n_evidence) for t in by_share[:3]},
    }
    if metrics:
        facts["model"] = {"test_macro_f1": metrics["test"]["model"]["macro_f1"],
                          "test_macro_f1_ci": metrics["test"]["model"].get("macro_f1_ci"),
                          "labels_total": metrics["dataset"]["n_labelled"],
                          "labels": "AI-drafted following the labelling guidelines; not human-verified"}
    return facts


def comparison_facts(preds: pd.DataFrame, companies: list[str], year: str, metrics: dict | None = None) -> dict:
    per = [report_facts(preds[preds.company.isin(companies)], c, year, metrics, n_evidence=2) for c in companies]
    leaders = {}
    for t in STRATEGIC_THEMES:
        best = max(per, key=lambda f: next(x["share_pct"] for x in f["themes"] if x["code"] == t))
        leaders[THEME_NAMES[t]] = {"company": best["company_name"],
                                   "share_pct": next(x["share_pct"] for x in best["themes"] if x["code"] == t)}
    return {"year": year, "companies": [f["company_name"] for f in per], "per_company": per,
            "theme_leaders": leaders, "model": per[0].get("model")}


# ---------------------------------------------------------------- template narrative
def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def template_narrative(f: dict) -> dict:
    name, top = f["company_name"], [t for t in sorted(f["themes"], key=lambda r: -r["share_pct"])]
    t1, t2 = top[0], top[1]
    sp = f["specificity"]
    d = f.get("distinctive_theme")
    peers = _join(f["peers"]) if f["peers"] else ""

    tied = [t["theme"].lower() for t in top if t["share_pct"] == t1["share_pct"]]
    if len(tied) > 1:
        summary = [f"{name} spreads its {f['year']} strategy narrative evenly: {_join(tied)} each take "
                   f"{t1['share_pct']}%."]
    else:
        summary = [f"{name} spends {t1['share_pct']}% of its {f['year']} strategy narrative on "
                   f"{t1['theme'].lower()}, more than on any other theme."]
    if d and d["diff_vs_peers_pp"] > 0:
        if d["theme"] == t1["theme"] and len(tied) == 1:
            summary.append(f"That is also where it differs most from {peers} "
                           f"({d['diff_vs_peers_pp']:+.1f} pp above the peer average).")
        else:
            summary.append(f"Against {peers} it leans furthest toward {d['theme'].lower()} "
                           f"({d['diff_vs_peers_pp']:+.1f} pp above the peer average).")
    if f["significant_shifts"]:
        s = f["significant_shifts"][0]
        summary.append(f"The clearest change since {s['from_year']} is {s['theme'].lower()}, "
                       f"{s['from_pct']}% → {s['to_pct']}%.")
    elif sp["most_concrete"]:
        summary.append(f"Its most concrete writing is on {sp['most_concrete'].lower()}.")

    headline = [f"Of {f['n_passages']} passages in the letter and MD&A, {f['n_strategic']} discuss strategy. "
                f"{t1['theme']} ({t1['share_pct']}%) and {t2['theme'].lower()} ({t2['share_pct']}%) lead; "
                f"{f['least_theme'].lower()} gets the least space."]
    if d:
        headline.append(f"Relative to {peers}, the most distinctive emphasis is {d['theme'].lower()} "
                        f"({d['diff_vs_peers_pp']:+.1f} pp).")
    headline.append(f"Overall, {sp['concrete_pct']}% of its strategy passages carry concrete detail (figures, dates, "
                    f"named initiatives or commitments).")
    if f["has_prior_year"]:
        k = len(f["significant_shifts"])
        headline.append("No shift since {p} stands out from noise." if k == 0 else
                        ("One theme shift since {p} stands out from noise." if k == 1 else
                         f"{k} theme shifts since {{p}} stand out from noise."))
        headline[-1] = headline[-1].format(p=f["prior_year"])

    if f["peers"]:
        ordered = sorted(f["themes"], key=lambda r: -r.get("diff_vs_peers_pp", 0))
        above = [f"{r['theme'].lower()} ({r['share_pct']}% vs {r['peer_avg_pct']}%)" for r in ordered[:2]
                 if r["diff_vs_peers_pp"] > 0]
        below = [f"{r['theme'].lower()} ({r['share_pct']}% vs {r['peer_avg_pct']}%)" for r in ordered[::-1][:2]
                 if r["diff_vs_peers_pp"] < 0]
        mix = []
        if above:
            mix.append(f"Compared with the {peers} average, {name} writes more about {_join(above)}.")
        if below:
            mix.append(f"It writes less about {_join(below)}.")
        theme_mix = " ".join(mix)
    else:
        theme_mix = f"No peer report is available for {f['year']}, so the mix cannot be compared."

    if not f["has_prior_year"]:
        shifts = ONE_YEAR
    elif f["significant_shifts"]:
        shifts = "Shifts whose 95% interval excludes zero: " + "; ".join(
            f"{s['theme'].lower()} {s['from_pct']}% → {s['to_pct']}% ({s['change_pp']:+.1f} pp)"
            for s in f["significant_shifts"]) + ". Smaller moves are within noise."
    else:
        shifts = f"No theme's share moved by more than resampling noise between {f['prior_year']} and {f['year']}."

    if sp["by_theme"]:
        hi = max(sp["by_theme"], key=lambda r: r["mean_score"])
        lo = min(sp["by_theme"], key=lambda r: r["mean_score"])
        specificity = (f"{name} is most concrete on {hi['theme'].lower()} ({hi['concrete_pct']}% of passages carry "
                       f"figures, dates or named initiatives) and vaguest on {lo['theme'].lower()} "
                       f"({lo['concrete_pct']}%).")
    else:
        specificity = "Too few strategic passages to profile specificity."
    return {"summary": " ".join(summary), "headline": " ".join(headline), "theme_mix": theme_mix,
            "shifts": shifts, "specificity": specificity}


def template_comparison(c: dict) -> dict:
    per = c["per_company"]
    tops = [f"{f['company_name']} on {f['top_themes'][0].lower()} ({max(t['share_pct'] for t in f['themes'])}%)"
            for f in per]
    summary = f"In {c['year']}, each company's narrative centres on a different theme: " + "; ".join(tops) + "."
    leaders = c["theme_leaders"]
    lead_lines = [f"{th.lower()} ({v['company']}, {v['share_pct']}%)" for th, v in leaders.items()]
    concrete = sorted(per, key=lambda f: -(f["specificity"]["concrete_pct"] or 0))
    spec = (f"{concrete[0]['company_name']} writes most concretely overall ({concrete[0]['specificity']['concrete_pct']}% "
            f"of strategy passages carry concrete detail) and {concrete[-1]['company_name']} least "
            f"({concrete[-1]['specificity']['concrete_pct']}%).")
    return {"summary": summary,
            "headline": summary + " Theme by theme, the largest share belongs to: " + "; ".join(lead_lines) + ".",
            "theme_mix": " ".join(f"{f['company_name']}: {template_narrative(f)['theme_mix']}" for f in per),
            "specificity": spec}


# ---------------------------------------------------------------- number check + LLM
NUM_IN_TEXT = re.compile(r"(?<![\w.])[+−-]?\d+(?:\.\d+)?")


def _allowed_numbers(obj) -> set[float]:
    vals = set()
    if isinstance(obj, dict):
        for v in obj.values():
            vals |= _allowed_numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            vals |= _allowed_numbers(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        for v in (obj, abs(obj)):
            vals |= {float(v), float(round(v)), float(round(v, 1))}
    elif isinstance(obj, str):
        for n in NUM_IN_TEXT.findall(obj):
            vals |= _allowed_numbers(float(n.replace("−", "-")))
    return vals


def unsupported_numbers(text: str, facts: dict) -> list[str]:
    # small counts plus method constants: the 95% interval and the 0-5 specificity scale
    allowed = _allowed_numbers(facts) | {0.0, 1.0, 2.0, 3.0, 5.0, 95.0}
    bad = []
    for n in NUM_IN_TEXT.findall(text):
        v = abs(float(n.replace("−", "-")))
        if not any(abs(v - a) < 0.051 for a in allowed):
            bad.append(n)
    return bad


def llm_narrative(facts: dict, client, comparison: bool = False) -> tuple[dict, list[str]]:
    """Returns (narrative, notes). Sections that fail the number check fall back to template text."""
    slim = json.loads(json.dumps(facts))  # evidence texts are long; the narrative only needs a hint of them
    for f in ([slim] if not comparison else slim["per_company"]):
        f["evidence"] = {k: [e["text"][:220] for e in v] for k, v in f["evidence"].items()}
    prompt = prompts.comparison_prompt(slim) if comparison else prompts.report_prompt(slim)
    raw = client.generate(prompts.REPORT_SYSTEM, prompt, max_tokens=1400, json_mode=True)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise LLMError("AI writing returned unusable output; using template text.")
    base = template_comparison(facts) if comparison else template_narrative(facts)
    out, notes = {}, []
    for key in base:
        text = str(data.get(key, "")).strip()
        bad = unsupported_numbers(text, facts) if text else ["missing"]
        if not text or bad:
            out[key] = base[key]
            notes.append(f"'{key}' fell back to template text"
                         + (f" (unsupported numbers: {', '.join(bad)})" if text else ""))
        else:
            out[key] = text
    if not comparison and not facts["has_prior_year"]:
        out["shifts"] = ONE_YEAR  # never let the model describe a comparison that cannot exist
    return out, notes


# ---------------------------------------------------------------- brief
def _method_note(model: dict | None) -> str:
    parts = [CAVEAT, "Passages come only from the Chairman/CEO letter and MD&A. A TF-IDF + logistic regression "
                     "classifier assigned each passage a theme; shares are theme passages divided by all strategic "
                     "(non-Other) passages. Year-over-year shifts count only when their 95% bootstrap interval "
                     "excludes zero. Specificity counts concrete signals (figures, dates, named initiatives, "
                     "commitments)."]
    if model:
        ci = model.get("test_macro_f1_ci")
        parts.append(f"The classifier scores macro-F1 {model['test_macro_f1']:.2f}"
                     + (f" (95% CI {ci[0]:.2f}–{ci[1]:.2f})" if ci else "") + " on held-out passages. "
                     f"All {model['labels_total']} training labels were AI-drafted following the labelling "
                     "guidelines and have not been human-verified.")
    return " ".join(parts)


def build_brief(preds: pd.DataFrame, metrics: dict | None, companies: list[str], year: str,
                client=None) -> dict:
    comparison = len(companies) > 1
    facts = comparison_facts(preds, companies, year, metrics) if comparison else \
        report_facts(preds, companies[0], year, metrics)
    narrative = template_comparison(facts) if comparison else template_narrative(facts)
    source, notes = "template", []
    if client is not None:
        try:
            narrative, notes = llm_narrative(facts, client, comparison)
            source = "AI-assisted"
        except LLMError as exc:
            notes = [str(exc)]
        except Exception:
            notes = ["AI writing failed unexpectedly; using template text."]
    title = (f"{' vs '.join(facts['companies'])} · {year}" if comparison else f"{facts['company_name']} · {year}")
    return {"kind": "comparison" if comparison else "single", "title": title, "year": year, "facts": facts,
            "narrative": narrative, "narrative_source": source, "notes": notes,
            "method_note": _method_note(facts.get("model")), "generated": date.today().isoformat()}


def short_summary(f: dict) -> str:
    """Two short sentences for the benchmark's 'At a glance' list (the company name is shown separately)."""
    top = sorted(f["themes"], key=lambda r: -r["share_pct"])
    tied = [t["theme"].lower() for t in top if t["share_pct"] == top[0]["share_pct"]]
    first = (f"Even spread: {_join(tied)} at {top[0]['share_pct']}% each" if len(tied) > 1
             else f"Leads on {top[0]['theme'].lower()} ({top[0]['share_pct']}% of the strategy narrative)")
    d = f.get("distinctive_theme")
    if d and d["diff_vs_peers_pp"] > 0:
        first += ("; that is also where it stands furthest from peers" if d["theme"] == top[0]["theme"]
                  else f"; stands out from peers on {d['theme'].lower()}") + f" ({d['diff_vs_peers_pp']:+.1f} pp)."
    else:
        first += "."
    if f["significant_shifts"]:
        s = f["significant_shifts"][0]
        second = f"Clearest change since {s['from_year']}: {s['theme'].lower()}, {s['from_pct']}% → {s['to_pct']}%."
    elif f["has_prior_year"]:
        second = f"No theme moved beyond noise since {f['prior_year']}."
    elif f["specificity"]["most_concrete"]:
        second = f"Most concrete on {f['specificity']['most_concrete'].lower()}."
    else:
        second = ""
    return f"{first} {second}".strip()


def report_summaries(preds: pd.DataFrame) -> list[dict]:
    """One short summary per report (company-year) for the benchmark page."""
    out = []
    for c, y in preds[["company", "year"]].drop_duplicates().sort_values(["company", "year"]).values:
        try:
            f = report_facts(preds, c, y, n_evidence=0)
            out.append({"company": c, "year": y, "summary": short_summary(f)})
        except ValueError:
            continue
    return out


# ---------------------------------------------------------------- rendering
CSS = """
body{font-family:Inter,'Segoe UI',system-ui,sans-serif;color:#2B2B2B;background:#FAF8F4;margin:0}
main{max-width:780px;margin:0 auto;padding:40px 28px 60px;background:#fff}
h1,h2{font-family:'Source Serif 4',Georgia,serif;font-weight:600;letter-spacing:-.01em}
h1{font-size:28px;margin:0 0 4px} h2{font-size:18px;margin:28px 0 6px;padding-top:14px;border-top:1px solid #E6E1D8}
p{font-size:14px;line-height:1.6;margin:6px 0} .meta{font-size:12px;color:#6B6660}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums;margin-top:8px}
th{text-align:left;font-weight:500;color:#6B6660;font-size:12px;border-bottom:1px solid #E6E1D8;padding:5px 8px 5px 0}
td{border-bottom:1px solid #F0ECE5;padding:5px 8px 5px 0} td.r,th.r{text-align:right}
blockquote{border-left:3px solid #1F3A5F;margin:10px 0;padding:2px 0 2px 12px;font-family:'Source Serif 4',Georgia,serif;font-size:14.5px;line-height:1.5}
blockquote .meta{font-family:Inter,sans-serif;display:block;margin-top:3px}
.lede{font-family:'Source Serif 4',Georgia,serif;font-size:17px;line-height:1.5;margin:14px 0 4px}
.note{font-size:12px;color:#4A4640;background:#F2EEE7;padding:8px 12px;border-left:2px solid #1F3A5F}
"""


def _theme_table_html(f: dict) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(t['theme'])}</td><td class='r'>{t['share_pct']}%</td><td class='r'>{t['count']}</td>"
        + (f"<td class='r'>{t['peer_avg_pct']}%</td><td class='r'>{t['diff_vs_peers_pp']:+.1f}</td>"
           if "peer_avg_pct" in t else "") + "</tr>"
        for t in sorted(f["themes"], key=lambda r: -r["share_pct"]))
    head = ("<tr><th>Theme</th><th class='r'>Share</th><th class='r'>Passages</th>"
            + ("<th class='r'>Peer avg</th><th class='r'>vs peers (pp)</th>" if f["peers"] else "") + "</tr>")
    return f"<table>{head}{rows}</table>"


def _evidence_html(f: dict) -> str:
    out = []
    for theme, items in f["evidence"].items():
        if not items:
            continue
        out.append(f"<p><b>{html.escape(theme)}</b></p>")
        for e in items:
            out.append(f"<blockquote>{html.escape(e['text'])}<span class='meta'>{html.escape(e['company_name'])} · "
                       f"{e['year']} · {html.escape(e['section'])} · p. {e['page']} · {e['spec_tier'].lower()}"
                       + (f" ({html.escape(e['spec_signals'])})" if e.get("spec_signals") else "") +
                       "</span></blockquote>")
    return "".join(out)


def render_html(b: dict) -> str:
    n, f = b["narrative"], b["facts"]
    parts = [f"<h1>{html.escape(b['title'])}</h1>",
             f"<p class='meta'>StratLens strategy brief · generated {b['generated']} · narrative: "
             f"{html.escape(b['narrative_source'])}</p>",
             f"<p class='lede'>{html.escape(n['summary'])}</p>",
             "<h2>Overview</h2>", f"<p>{html.escape(n['headline'])}</p>"]
    if b["kind"] == "single":
        parts += ["<h2>Theme mix</h2>", f"<p>{html.escape(n['theme_mix'])}</p>", _theme_table_html(f),
                  "<h2>Year-over-year shifts</h2>", f"<p>{html.escape(n['shifts'])}</p>",
                  "<h2>Specificity</h2>", f"<p>{html.escape(n['specificity'])}</p>",
                  "<h2>Evidence</h2>", _evidence_html(f)]
    else:
        parts += ["<h2>Theme mix</h2>", f"<p>{html.escape(n['theme_mix'])}</p>"]
        for pf in f["per_company"]:
            parts += [f"<p><b>{html.escape(pf['company_name'])}</b></p>", _theme_table_html(pf)]
        parts += ["<h2>Specificity</h2>", f"<p>{html.escape(n['specificity'])}</p>", "<h2>Evidence</h2>"]
        parts += [_evidence_html(pf) for pf in f["per_company"]]
    parts += ["<h2>Method and caveats</h2>", f"<p class='note'>{html.escape(b['method_note'])}</p>"]
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{html.escape(b['title'])}</title>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<link href='https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Source+Serif+4:opsz,"
            "wght@8..60,400;8..60,600&display=swap' rel='stylesheet'>"
            f"<style>{CSS}</style></head><body><main>{''.join(parts)}</main></body></html>")


def render_pdf(b: dict) -> bytes:
    from fpdf import FPDF

    pdf = FPDF(format="A4", unit="mm")
    pdf.set_margins(18, 16, 18)
    pdf.set_auto_page_break(True, 16)
    pdf.add_font("Serif", "", str(FONT_DIR / "SourceSerif4-Regular.ttf"))
    pdf.add_font("Serif", "B", str(FONT_DIR / "SourceSerif4-Semibold.ttf"))
    pdf.add_page()
    navy, muted, ink = (31, 58, 95), (107, 102, 96), (43, 43, 43)
    w = pdf.w - 36

    def h1(t):
        pdf.set_font("Serif", "B", 20); pdf.set_text_color(*ink); pdf.multi_cell(w, 9, t, new_x="LMARGIN", new_y="NEXT")

    def h2(t):
        pdf.ln(3); pdf.set_draw_color(230, 225, 216); pdf.line(18, pdf.get_y(), 18 + w, pdf.get_y()); pdf.ln(3)
        pdf.set_font("Serif", "B", 13); pdf.set_text_color(*ink); pdf.multi_cell(w, 6.5, t, new_x="LMARGIN", new_y="NEXT")

    def para(t, size=10.5, color=ink):
        pdf.set_font("Serif", "", size); pdf.set_text_color(*color); pdf.multi_cell(w, 5.2, t, new_x="LMARGIN", new_y="NEXT"); pdf.ln(1.5)

    def table(f):
        pdf.set_font("Serif", "", 9.5)
        cols = [("Theme", 70), ("Share", 20), ("Passages", 22)] + ([("Peer avg", 22), ("vs peers (pp)", 30)]
                                                                   if f["peers"] else [])
        pdf.set_text_color(*muted)
        for name, cw in cols:
            pdf.cell(cw, 6, name, align="L" if name == "Theme" else "R")
        pdf.ln(6); pdf.set_text_color(*ink)
        for t in sorted(f["themes"], key=lambda r: -r["share_pct"]):
            vals = [t["theme"], f"{t['share_pct']}%", str(t["count"])]
            if f["peers"]:
                vals += [f"{t['peer_avg_pct']}%", f"{t['diff_vs_peers_pp']:+.1f}"]
            for (name, cw), v in zip(cols, vals):
                pdf.cell(cw, 5.6, v, align="L" if name == "Theme" else "R")
            pdf.ln(5.6)
        pdf.ln(2)

    def evidence(f):
        for theme, items in f["evidence"].items():
            if not items:
                continue
            pdf.set_font("Serif", "B", 10.5); pdf.set_text_color(*ink); pdf.multi_cell(w, 5.5, theme, new_x="LMARGIN", new_y="NEXT")
            for e in items:
                y0 = pdf.get_y()
                pdf.set_x(22); pdf.set_font("Serif", "", 10); pdf.set_text_color(*ink)
                pdf.multi_cell(w - 4, 5, e["text"], new_x="LMARGIN", new_y="NEXT")
                pdf.set_x(22); pdf.set_font("Serif", "", 8.5); pdf.set_text_color(*muted)
                pdf.multi_cell(w - 4, 4.5, f"{e['company_name']} · {e['year']} · {e['section']} · p. {e['page']} · "
                                           f"{e['spec_tier'].lower()}" + (f" ({e['spec_signals']})" if e.get("spec_signals") else ""),
                               new_x="LMARGIN", new_y="NEXT")
                pdf.set_draw_color(*navy); pdf.set_line_width(0.6); pdf.line(19, y0, 19, pdf.get_y())
                pdf.set_line_width(0.2); pdf.ln(2)

    n, f = b["narrative"], b["facts"]
    h1(b["title"])
    para(f"StratLens strategy brief · generated {b['generated']} · narrative: {b['narrative_source']}", 8.5, muted)
    pdf.ln(1); para(n["summary"], 12.5); h2("Overview"); para(n["headline"])
    if b["kind"] == "single":
        h2("Theme mix"); para(n["theme_mix"]); table(f)
        h2("Year-over-year shifts"); para(n["shifts"])
        h2("Specificity"); para(n["specificity"])
        h2("Evidence"); evidence(f)
    else:
        h2("Theme mix"); para(n["theme_mix"])
        for pf in f["per_company"]:
            pdf.set_font("Serif", "B", 10.5); pdf.multi_cell(w, 5.5, pf["company_name"], new_x="LMARGIN", new_y="NEXT"); table(pf)
        h2("Specificity"); para(n["specificity"])
        h2("Evidence")
        for pf in f["per_company"]:
            evidence(pf)
    h2("Method and caveats"); para(b["method_note"], 9, muted)
    return bytes(pdf.output())
