"""Shared Streamlit components and cached artefact loaders. Styling lives in assets/style.css.

The app only reads files written offline by `python -m stratlens.build`, `train.py` and `python -m stratlens.audit`.
"""
from __future__ import annotations

import html
import json
from contextlib import contextmanager
from urllib.parse import quote as urlquote

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from . import specificity, theme, trace
from . import model as mdl
from .config import (ANALYTICS_JSON, ASK_EXAMPLES_JSON, AUDIT_JSON, COMPANIES, COMPANY_NAMES, COMPANY_SHORT,
                     EVIDENCE_MIN_CONFIDENCE, MAX_WORDS, METRICS_JSON, MODEL_PATH, NARRATIVES_JSON, PASSAGES_CSV,
                     PIPELINE_STEPS, RELIABILITY_F1, REVIEW_THRESHOLD, ROOT, SECTION_NAMES, SECTIONS_CSV,
                     STRATEGIC_THEMES, SYSTEM_NAMES, THEME_COLORS, THEME_NAMES)
from .report import clean
from .theme import tint

CSS_PATH = ROOT / "assets" / "style.css"

LIGHT = {"--card": "#FFFFFF", "--border": "#E6E9E5", "--text": "#141A17", "--muted": "#6B756F",
         "--subtle": "#F3F5F2", "--bar": "#D3D9D4", "--accent": "#1E8E4E", "--accent-soft": "#E4F3EA",
         "--warn": "#9A6512", "--warn-soft": "#FBF1DF", "--neg-soft": "#FBE9E5"}
DARK = {"--card": "#141A17", "--border": "#232B27", "--text": "#E4E9E5", "--muted": "#8E9A93",
        "--subtle": "#1C2420", "--bar": "#34403A", "--accent": "#4CC983", "--accent-soft": "#16291F",
        "--warn": "#E3AE52", "--warn-soft": "#2A2314", "--neg-soft": "#2C1A16"}


def system_name(s: str) -> str:
    return SYSTEM_NAMES.get(s, s)


# ---------- loaders (artefacts are built offline; the app never retrains or recomputes) ----------
def _json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        return None, None
    return mdl.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict | None:
    return _json(METRICS_JSON)


@st.cache_data(show_spinner=False)
def load_passages() -> pd.DataFrame | None:
    if not PASSAGES_CSV.exists():
        return None
    df = pd.read_csv(PASSAGES_CSV)
    df["why"] = [json.loads(x) if isinstance(x, str) else [] for x in df.get("why_terms", pd.Series([None] * len(df)))]
    return df


load_benchmark = load_passages  # older name, used by the upload flow


@st.cache_data(show_spinner=False)
def load_analytics() -> dict | None:
    return _json(ANALYTICS_JSON)


@st.cache_data(show_spinner=False)
def load_narratives() -> dict:
    return _json(NARRATIVES_JSON) or {"brief": {}, "compare": {}}


@st.cache_data(show_spinner=False)
def load_ask_examples() -> dict:
    return _json(ASK_EXAMPLES_JSON) or {}


@st.cache_data(show_spinner=False)
def load_audit() -> dict | None:
    return _json(AUDIT_JSON)


@st.cache_data(show_spinner=False)
def load_sections() -> pd.DataFrame | None:
    return pd.read_csv(SECTIONS_CSV) if SECTIONS_CSV.exists() else None


# ---------- names ----------
def company_name(code: str) -> str:
    return COMPANY_NAMES.get(code, code.title())


def short(code: str) -> str:
    return COMPANY_SHORT.get(code, code.title())


def report_label(key: str) -> str:
    c, y = key.split("|")
    return f"{short(c)} {y}"


def section_name(code: str) -> str:
    return SECTION_NAMES.get(code, code)


# ---------- scaffolding ----------
def _vars(d: dict) -> str:
    return ";".join(f"{k}:{v}" for k, v in d.items())


def inject_css() -> None:
    """Stylesheet plus colour variables for the active Streamlit theme (OS preference until it is known)."""
    try:
        mode = st.context.theme.type
    except Exception:
        mode = None
    if mode == "dark":
        vars_css = f":root{{{_vars(DARK)}}}"
    elif mode == "light":
        vars_css = f":root{{{_vars(LIGHT)}}}"
    else:
        vars_css = f":root{{{_vars(LIGHT)}}} @media (prefers-color-scheme: dark){{:root{{{_vars(DARK)}}}}}"
    st.markdown(f"<style>{vars_css}\n{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def apply_theme_switch() -> None:
    """Streamlit keeps the chosen theme in the browser, so a switch is stored there and the page reloads."""
    pending = st.session_state.pop("_theme_switch", None)
    if pending:
        components.html(f"<script>window.parent.localStorage.setItem('stActiveTheme-/-v2', '\"{pending}\"');"
                        "window.parent.location.reload();</script>", height=0)


def _about() -> str:
    an, m = load_analytics(), load_metrics()
    if not an:
        return "No data built yet."
    reps = sorted(an["reports"])
    missing = an.get("missing", [])
    lines = [f"**{len(reps)} annual reports · {len(an['companies'])} companies** "
             f"({', '.join(company_name(c) for c in an['companies'])}), {', '.join(an['years'])}."]
    if missing:
        lines.append("Missing: " + ", ".join(report_label(k) for k in missing) + ".")
    lines.append("Only the Chairman/CEO letter and the MD&A are read. Sources: NSE annual-report filings "
                 "(see docs/data_sources.md).")
    lines.append(f"Themes come from a trained text classifier (7 classes). Data built {an['generated']}"
                 + (f"; model trained {m['generated']}." if m else "."))
    lines.append("Shares show how much a report talks about a theme, not what the company spends or achieves.")
    return "\n\n".join(lines)


def menu() -> None:
    """The ⋮ menu: appearance and About this data."""
    with st.popover("", icon=":material/more_vert:", help="Menu"):
        dark = theme.is_dark()
        pick = st.segmented_control("Appearance", ["Light", "Dark"], default="Dark" if dark else "Light",
                                    key="appearance")
        if pick and pick != ("Dark" if dark else "Light"):
            st.session_state["_theme_switch"] = pick
            st.rerun()
        st.markdown("**About this data**")
        st.markdown(f'<div class="sl-about">{_md(_about())}</div>', unsafe_allow_html=True)


def _md(text: str) -> str:
    """Tiny markdown for the About box: **bold** and paragraphs only."""
    import re
    esc = html.escape(text)
    esc = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", esc)
    return "".join(f"<p>{p}</p>" for p in esc.split("\n\n"))


def header(title: str, subtitle: str = "") -> None:
    c1, c2 = st.columns([20, 1], vertical_alignment="top")
    with c1:
        st.markdown(f'<div class="sl-head"><div><div class="sl-title">{html.escape(title)}</div>'
                    + (f'<p class="sl-sub">{subtitle}</p>' if subtitle else "") + "</div></div>",
                    unsafe_allow_html=True)
    with c2:
        menu()


page_header = header


@contextmanager
def card(key: str, title: str | None = None, caption: str | None = None, meta: str | None = None):
    with st.container(key=f"card_{key}"):
        if title:
            st.markdown(f'<div class="sl-card-h"><span class="sl-card-t">{html.escape(title)}</span>'
                        f'<span class="sl-card-m">{meta or ""}</span></div>'
                        + (f'<p class="sl-card-c">{caption}</p>' if caption else ""), unsafe_allow_html=True)
        yield


def subhead(text: str) -> None:
    st.markdown(f'<p class="sl-sub-h">{text}</p>', unsafe_allow_html=True)


def callout(text: str) -> None:
    st.markdown(f'<p class="sl-callout">{text}</p>', unsafe_allow_html=True)


def note(text: str, warn: bool = False) -> None:
    st.markdown(f'<div class="sl-note{" sl-warn" if warn else ""}">{text}</div>', unsafe_allow_html=True)


def empty_state(sentence: str, action: str = "") -> None:
    st.markdown(f'<div class="sl-empty">{sentence} {action}</div>', unsafe_allow_html=True)


def meta(text: str) -> None:
    st.markdown(f'<p class="sl-meta">{text}</p>', unsafe_allow_html=True)


def link(href: str, label: str, cls: str = "sl-link") -> str:
    return f'<a class="{cls}" href="{html.escape(href, quote=True)}" target="_self">{label}</a>'


def qs(**params) -> str:
    return "&".join(f"{k}={urlquote(str(v), safe='|')}" for k, v in params.items() if v not in (None, ""))


def kpi_cards(items: list[dict], inset: bool = False) -> None:
    """items: label, value, unit?, sub?, help? (hover definition on the label)"""
    cells = []
    for it in items:
        label = tip(it["label"], it["help"]) if it.get("help") else html.escape(it["label"])
        unit = f'<span class="unit">{it["unit"]}</span>' if it.get("unit") else ""
        sub = f'<div class="sub">{it["sub"]}</div>' if it.get("sub") else ""
        cells.append(f'<div class="sl-kpi{" inset" if inset else ""}"><div style="min-width:0"><div class="lab">'
                     f'{label}</div><div class="val">{it["value"]}{unit}</div>{sub}</div></div>')
    st.markdown(f'<div class="sl-kpis" style="--n:{len(items)}">{"".join(cells)}</div>', unsafe_allow_html=True)


def kpi_strip(items: list[tuple[str, str]]) -> None:
    kpi_cards([{"label": lab, "value": v} for v, lab in items], inset=True)


def pipeline_strip(active: int | None = None, href: str | None = "how-it-works", done: int | None = None) -> None:
    """The six plain-English steps. `active` highlights one (Add a report progress); `done` ticks earlier ones."""
    parts = []
    for i, s in enumerate(PIPELINE_STEPS):
        cls = "active" if i == active else ("done" if done is not None and i < done else "")
        mark = "✓" if cls == "done" else str(i + 1)
        parts.append(f'<span class="sl-ps {cls}" title="{html.escape(s["what"])}"><b>{mark}</b>{html.escape(s["step"])}'
                     "</span>")
    arrow = '<span class="sl-ps-arrow">→</span>'
    body = arrow.join(parts)
    tail = f' {link(href, "How it works →", "sl-ps-more")}' if href else ""
    st.markdown(f'<div class="sl-pstrip">{body}{tail}</div>', unsafe_allow_html=True)


def steps(labels: list[str], active: int) -> None:
    parts = [f'<div class="sl-step {"active" if i == active else ""}"><b>{i + 1}</b>{lab}</div>'
             for i, lab in enumerate(labels)]
    st.markdown(f'<div class="sl-steps">{"".join(parts)}</div>', unsafe_allow_html=True)


def pill(text: str, tone: str = "", help: str = "") -> str:
    t = f' title="{html.escape(help)}"' if help else ""
    return f'<span class="sl-pill {tone}"{t}>{html.escape(text)}</span>'


def tip(label: str, text: str) -> str:
    """Inline label with a hover definition (dotted underline, help cursor)."""
    return f'<span class="sl-tip" title="{html.escape(text)}">{html.escape(label)}</span>'


def theme_label(code: str, name: str | None = None) -> str:
    return (f'<span class="sl-theme"><span class="sl-dot" style="background:{THEME_COLORS.get(code, "#999")}"></span>'
            f'{html.escape(name or THEME_NAMES.get(code, code))}</span>')


def conf_bar(c: float, threshold: float = REVIEW_THRESHOLD) -> str:
    style = "" if c >= threshold else f"background:{theme.WARN}"
    return (f'<span class="sl-bar"><span style="width:{c * 100:.0f}%;{style}"></span></span>'
            f'<span class="num">{c:.2f}</span>')


def finding_cards(items: list[dict]) -> None:
    """Insight cards: the title is the finding; the whole card links to its evidence."""
    cells = []
    for it in items:
        dot = (f'<span class="sl-dot" style="background:{THEME_COLORS[it["theme"]]}"></span>'
               if it.get("theme") else "")
        cells.append(f'<a class="sl-fcard" href="{html.escape(it["link"], quote=True)}" target="_self">'
                     f'<div class="h">{dot}{html.escape(it["title"])}</div>'
                     f'<div class="d">{html.escape(it["detail"])}</div><div class="go">See evidence →</div></a>')
    st.markdown(f'<div class="sl-fcards">{"".join(cells)}</div>', unsafe_allow_html=True)


# ---------- fingerprint ----------
def fingerprint_grid(an: dict, mode: str) -> None:
    """Companies × themes. mode: a year (share of strategy passages) or 'Change' (latest minus previous year).
    Every cell links to the Company page filtered to that theme; missing reports say so."""
    themes = STRATEGIC_THEMES
    head = "".join(f'<th><span class="sl-dot" style="background:{THEME_COLORS[t]}"></span>'
                   f'{html.escape(THEME_NAMES[t])}</th>' for t in themes)
    rows = []
    years = an["years"]
    shares = [r["themes"][t]["share"] for r in an["reports"].values() for t in themes]
    smax = max(shares) if shares else 1
    for c in an["companies"]:
        cells = []
        if mode == "Change":
            y = an["yoy"].get(c)
            for t in themes:
                if not y:
                    missing = [k for k in an["missing"] if k.startswith(c + "|")]
                    cells.append(f'<td class="miss">{"missing " + missing[0].split("|")[1] if missing else "–"}</td>')
                    continue
                d = y["themes"][t]
                v = round(d["pp"])
                txt = "0" if v == 0 else f"{v:+d}"
                href = "compare?" + qs(a=f"{c}|{y['from']}", b=f"{c}|{y['to']}")
                if d["significant"]:
                    col = theme.POSITIVE if d["pp"] > 0 else theme.NEGATIVE
                    bg = tint(col, min(0.15 + abs(d["pp"]) / 40, 0.55))
                    cells.append(f'<td style="background:{bg}"><a href="{href}" target="_self" title="Bigger than '
                                 f'chance">{txt}</a></td>')
                else:
                    cells.append(f'<td class="noise"><a href="{href}" target="_self" title="Within noise">'
                                 f'{txt}</a></td>')
        else:
            r = an["reports"].get(f"{c}|{mode}")
            for t in themes:
                if r is None:
                    cells.append('<td class="miss">missing</td>')
                    continue
                s = r["themes"][t]["share"]
                a = 0.08 + 0.82 * (s / smax)
                strong = a > 0.55
                href = "company?" + qs(company=c, year=mode, theme=t)
                cells.append(f'<td style="background:{tint(theme.ACCENT, a)}" class="{"strong" if strong else ""}">'
                             f'<a href="{href}" target="_self" title="{r["themes"][t]["count"]} of '
                             f'{r["n_strategic"]} strategy passages">{s * 100:.0f}%</a></td>')
        name = link("company?" + qs(company=c, year=years[-1]), html.escape(short(c)))
        rows.append(f'<tr><th class="co">{name}</th>{"".join(cells)}</tr>')
    st.markdown(f'<div class="sl-fpwrap"><table class="sl-fp"><thead><tr><th></th>{head}</tr></thead>'
                f'<tbody>{"".join(rows)}</tbody></table></div>', unsafe_allow_html=True)


# ---------- passages ----------
def passage(row, show_why: bool = True, anchor: str | None = None, label: str | None = None) -> None:
    """One evidence passage: text, page, 'why this theme' chips and a details toggle with the raw metadata."""
    color = THEME_COLORS.get(row["pred"], "#999")
    text = html.escape(clean(row["text"]))
    why = row.get("why") if isinstance(row.get("why"), list) else []
    chips = ("".join(f"<span>{html.escape(w)}</span>" for w in why[:3]) if show_why and why else "")
    signals = specificity.describe(row.get("spec_signals"))
    details = (f'{html.escape(section_name(row["section"]))} · model confidence {row["confidence"]:.2f} · '
               f'concreteness {int(row["specificity"])}/5' + (f' ({html.escape(signals)})' if signals else "")
               + f' · {html.escape(row["passage_id"])}')
    idattr = f' id="{anchor}"' if anchor else ""
    lab = f'<b class="mono">{html.escape(label)}</b> ' if label else ""
    st.markdown(
        f'<div class="sl-quote"{idattr} style="border-left-color:{color}"><div class="q">{lab}{text}</div>'
        f'<div class="m"><span>{html.escape(short(row["company"]))} {row["year"]} · p.{int(row["page"])}</span>'
        + (f'<span class="sl-why" title="Words that pushed the model towards this theme">why this theme: '
           f'{chips}</span>' if chips else "")
        + f'<details><summary>details</summary>{details}</details></div></div>', unsafe_allow_html=True)


def evidence_rows(passages: pd.DataFrame, company: str, year: str, theme_code: str) -> pd.DataFrame:
    df = passages[(passages.company == company) & (passages.year == year) & (passages.pred == theme_code)
                  & (passages.confidence >= EVIDENCE_MIN_CONFIDENCE)]
    return df.sort_values(["confidence", "specificity"], ascending=False)


def reliability(company: str) -> tuple[float | None, str]:
    """Per-company theme accuracy (macro-F1) from the human audit if available, else the AI-drafted test set.
    Returns (f1 or None, source)."""
    au = load_audit()
    if au:
        for p in au.get("per_company", []):
            if p["company"] == company:
                return p["macro_f1"], "your labels"
    m = load_metrics()
    if m:
        for s in m.get("subgroups", []):
            if s["dimension"] == "Company" and s["group"] == company:
                return s["macro_f1"], "AI-drafted test labels"
    return None, "not in training"


def reliability_note(company: str) -> None:
    f1, src = reliability(company)
    if f1 is not None and f1 < RELIABILITY_F1:
        note("The model is less reliable on this company's writing; read the evidence.", warn=True)
    elif f1 is None:
        note(f"The model never saw {html.escape(short(company))}'s reports in training and its accuracy here "
             "has not been checked yet; read the evidence.", warn=True)


# ---------- pipeline inspector (How it works, Add a report) ----------
def highlight(text: str, terms: list[str], color: str) -> str:
    from . import explain
    spans = explain.highlight_spans(text, terms)
    out, last = [], 0
    for s, e in spans:
        out.append(html.escape(text[last:s]))
        out.append(f'<mark style="background:{tint(color, 0.22)}">{html.escape(text[s:e])}</mark>')
        last = e
    out.append(html.escape(text[last:]))
    return "".join(out)


def _step(n: int, title: str, how: str, body: str) -> None:
    st.markdown(f'<div class="sl-stp"><div class="sl-stp-n">{n}</div><div class="sl-stp-b"><div class="sl-stp-t">'
                f'{title}<span>{how}</span></div>{body}</div></div>', unsafe_allow_html=True)


def pipeline_inspector(text: str) -> None:
    """Every step for one piece of text: cleaned, masked, TF-IDF features, probabilities, theme, concreteness."""
    model, meta_ = load_model()
    if model is None:
        empty_state("No trained model.", "Run <code>python train.py</code>.")
        return
    cleaned, passages, dropped = trace.segment_text(text)
    if dropped:
        note(f"The build would drop this block as <b>{dropped}</b>. Inspecting it anyway.", warn=True)
    passage_ = passages[0] if passages else cleaned
    if len(passages) > 1:
        i = st.selectbox(f"Split into {len(passages)} passages of up to {MAX_WORDS} words. Inspect:",
                         range(len(passages)), format_func=lambda k: f"{k + 1}. {passages[k][:90]}…")
        passage_ = passages[i]
    t = trace.trace(model, passage_, meta_["threshold"])
    color = THEME_COLORS[t["pred"]]
    sp, sig = t["specificity"], t["specificity"]["signals"]

    left, right = st.columns([1, 1], gap="large")
    with left:
        drop = (f'<p class="sl-meta">The build would drop this passage: {html.escape(t["filter"])}.</p>'
                if t["filter"] else "")
        _step(1, "Clean and split", "header, bullet and hyphenation fixes · sentence split",
              f'<div class="sl-code">{html.escape(clean(passage_))}</div>{drop}')
        _step(2, "Hide names", "company, brand, number, year → placeholders",
              f'<div class="sl-code">{html.escape(t["masked"])}</div>')
        flag = (pill(f"low confidence · {t['confidence']:.2f} < {t['threshold']:.2f}", "warn") if t["low_confidence"]
                else pill(f"confidence {t['confidence']:.2f}", "good"))
        _step(5, "Theme", "the most probable of the 7 classes",
              f'<p style="margin:4px 0">{theme_label(t["pred"])} &nbsp; {flag}</p>')
        rows = [("Figures", str(sig["numbers"]), 2 if sig["numbers"] >= 3 else 1 if sig["numbers"] else 0),
                ("Date or deadline", "yes" if sig["time"] else "no", int(sig["time"])),
                ("Named initiative or brand", ", ".join(sig["named"]) or "none", int(bool(sig["named"]))),
                ("Commitment language", "yes" if sig["commitment"] else "no", int(sig["commitment"]))]
        body = "".join(f"<tr><td>{a}</td><td>{html.escape(b)}</td><td class='r'>+{c}</td></tr>" for a, b, c in rows)
        _step(6, "Concreteness", "regex + spaCy named entities on the original text",
              f'<table class="sl-table"><tbody>{body}<tr><td><b>Score</b></td><td>{sp["tier"]}</td>'
              f'<td class="r"><b>{sp["score"]}/5</b></td></tr></tbody></table>')
    with right:
        feats = "".join(f"<tr><td>{html.escape(f['term'])}</td><td class='r'>{f['tfidf']:.3f}</td>"
                        f"<td class='r'>{f['coef']:+.2f}</td><td class='r'>{f['contribution']:+.3f}</td></tr>"
                        for f in t["contributions"]) or "<tr><td class='sl-small' colspan='4'>no known terms</td></tr>"
        _step(3, "Words → numbers (TF-IDF)", "words and word pairs, weighted by how distinctive they are",
              f'<p class="sl-meta">{t["n_features"]} known terms present. Largest pushes towards '
              f'{html.escape(THEME_NAMES[t["pred"]])}: weight × learned coefficient.</p>'
              f'<table class="sl-table"><thead><tr><th>Term</th><th class="r">weight</th><th class="r">coef</th>'
              f'<th class="r">effect</th></tr></thead><tbody>{feats}</tbody></table>')
        probs = "".join(f"<tr><td>{theme_label(p['label'])}</td><td class='r'>{conf_bar(p['prob'], 0)}</td></tr>"
                        for p in t["probs"])
        _step(4, "Classify (logistic regression)", "one probability per class",
              f'<table class="sl-table"><tbody>{probs}</tbody></table>')
    st.markdown(f'<div class="sl-quote" style="border-left-color:{color}"><div class="q">'
                f'{highlight(clean(passage_), [f["term"] for f in t["contributions"] if f["contribution"] > 0], color)}'
                f'</div><div class="m">highlighted: words pushing towards this theme (association, not cause)'
                f'</div></div>', unsafe_allow_html=True)


def company_codes() -> list[str]:
    an = load_analytics()
    return an["companies"] if an else sorted(COMPANIES)
