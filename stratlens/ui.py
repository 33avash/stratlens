"""Shared Streamlit components and cached artefact loaders. Styling lives in assets/style.css."""
from __future__ import annotations

import html
import json
from contextlib import contextmanager

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from . import explain
from . import model as mdl
from . import specificity, theme, trace
from .config import (BENCHMARK_CSV, COMPANY_NAMES, MAX_WORDS, METRICS_JSON, MODEL_PATH, REVIEW_THRESHOLD, ROOT,
                     SECTION_NAMES, SECTIONS_CSV, SYSTEM_NAMES, THEME_NAMES)
from .report import clean
from .theme import THEME_COLORS, tint

CSS_PATH = ROOT / "assets" / "style.css"

def system_name(s: str) -> str:
    return SYSTEM_NAMES.get(s, s)

LIGHT = {"--card": "#FFFFFF", "--border": "#E6E9E5", "--text": "#141A17", "--muted": "#6B756F",
         "--subtle": "#F3F5F2", "--bar": "#D3D9D4", "--accent": "#1E8E4E", "--accent-soft": "#E4F3EA",
         "--warn": "#9A6512", "--warn-soft": "#FBF1DF", "--neg-soft": "#FBE9E5"}
DARK = {"--card": "#141A17", "--border": "#232B27", "--text": "#E4E9E5", "--muted": "#8E9A93",
        "--subtle": "#1C2420", "--bar": "#34403A", "--accent": "#4CC983", "--accent-soft": "#16291F",
        "--warn": "#E3AE52", "--warn-soft": "#2A2314", "--neg-soft": "#2C1A16"}


# ---------- loaders (artefacts come from train.py; the app never retrains) ----------
@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        return None, None
    return mdl.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict | None:
    return json.loads(METRICS_JSON.read_text(encoding="utf-8")) if METRICS_JSON.exists() else None


@st.cache_data(show_spinner=False)
def load_benchmark() -> pd.DataFrame | None:
    if not BENCHMARK_CSV.exists():
        return None
    df = pd.read_csv(BENCHMARK_CSV)
    df["source"] = "Sample analysis"
    return df


@st.cache_data(show_spinner=False)
def load_sections() -> pd.DataFrame | None:
    return pd.read_csv(SECTIONS_CSV) if SECTIONS_CSV.exists() else None


@st.cache_data(show_spinner=False, max_entries=2000)
def explain_text(text: str) -> dict | None:
    model, _ = load_model()
    return None if model is None else explain.explain(model, text, k=8)


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


def _queue_theme_switch() -> None:
    st.session_state["_theme_switch"] = "Light" if st.session_state["light_mode"] else "Dark"


def theme_toggle() -> None:
    """Light/dark switch. Streamlit keeps the chosen theme in the browser, so it is stored there and the page reloads."""
    pending = st.session_state.pop("_theme_switch", None)
    if pending:
        components.html(f"<script>window.parent.localStorage.setItem('stActiveTheme-/-v2', '\"{pending}\"');"
                        "window.parent.location.reload();</script>", height=0)
    st.session_state["light_mode"] = (pending == "Light") if pending else not theme.is_dark()
    st.toggle("Light mode", key="light_mode", on_change=_queue_theme_switch)


def header(title: str, subtitle: str = "", stamp: str = "") -> None:
    st.markdown(f'<div class="sl-head"><div><div class="sl-title">{title}</div>'
                + (f'<p class="sl-sub">{subtitle}</p>' if subtitle else "") + "</div>"
                + (f'<div class="sl-stamp">{stamp}</div>' if stamp else "") + "</div>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    header(title, subtitle)


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
    st.markdown(f'<div class="sl-note">{sentence} {action}</div>', unsafe_allow_html=True)


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


def findings(items: list[tuple[str, str]]) -> None:
    """Numbered short findings: a bold headline and one muted line each."""
    st.markdown('<div class="sl-finds">' + "".join(
        f'<div class="sl-find"><b>{i}</b><div><div class="h">{html.escape(h)}</div><div class="d">{html.escape(d)}'
        f'</div></div></div>' for i, (h, d) in enumerate(items, 1)) + "</div>", unsafe_allow_html=True)


def kpi_strip(items: list[tuple[str, str]]) -> None:
    kpi_cards([{"label": l, "value": v} for v, l in items], inset=True)


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


def company_badge(code: str) -> str:
    """Company initials on a neutral chip: companies never borrow a theme colour."""
    return (f'<span class="sl-co" style="border-color:{theme.company_color(code)}">'
            f'{html.escape(theme.company_initials(code))}</span>')


# ---------- formatters ----------
def theme_label(code: str, full: bool = True) -> str:
    name = THEME_NAMES.get(code, code) if full else code
    return (f'<span class="sl-theme"><span class="sl-dot" style="background:{THEME_COLORS.get(code, "#999")}"></span>'
            f'{html.escape(name)}</span>')


def conf_bar(c: float, threshold: float = REVIEW_THRESHOLD) -> str:
    style = "" if c >= threshold else f"background:{theme.WARN}"
    return (f'<span class="sl-bar"><span style="width:{c * 100:.0f}%;{style}"></span></span>'
            f'<span class="num">{c:.2f}</span>')


def company_name(code: str) -> str:
    return COMPANY_NAMES.get(code, code.title())


def section_name(code: str) -> str:
    return SECTION_NAMES.get(code, code)


def highlight(text: str, terms: list[str], color: str) -> str:
    spans = explain.highlight_spans(text, terms)
    out, last = [], 0
    for s, e in spans:
        out.append(html.escape(text[last:s]))
        out.append(f'<mark style="background:{tint(color, 0.22)}">{html.escape(text[s:e])}</mark>')
        last = e
    out.append(html.escape(text[last:]))
    return "".join(out)


def quote(row, show_terms: bool = True) -> None:
    theme_code = row["pred"]
    color = THEME_COLORS.get(theme_code, "#999")
    text = clean(row["text"])
    body = html.escape(text)
    if show_terms:
        ex = explain_text(row["text"])
        if ex and ex["pred"] == theme_code:
            body = highlight(text, [t["term"] for t in ex["positive"][:6]], color)
    signals = specificity.describe(row.get("spec_signals"))
    meta = (f'{company_name(row["company"])} · {row["year"]} · {section_name(row["section"])} · p.{int(row["page"])} · '
            f'confidence {row["confidence"]:.2f}'
            + (f' · {str(row["spec_tier"]).lower()}' if isinstance(row.get("spec_tier"), str) else "")
            + (f' ({html.escape(signals)})' if signals else ""))
    st.markdown(f'<div class="sl-quote" style="border-left-color:{color}"><div class="q">{body}</div>'
                f'<div class="m">{meta}</div></div>', unsafe_allow_html=True)


def predictions_table(df: pd.DataFrame, max_rows: int = 50) -> None:
    rows = []
    for r in df.head(max_rows).itertuples():
        spec = pill(r.spec_tier, "good" if r.spec_tier == "Specific" else "") \
            if isinstance(getattr(r, "spec_tier", None), str) else ""
        rows.append(f"<tr><td class='t'>{html.escape(clean(r.text))}</td><td>{theme_label(r.pred)}</td>"
                    f"<td>{conf_bar(r.confidence)}</td><td>{spec}</td><td class='r'>{int(r.page)}</td></tr>")
    st.markdown("<table class='sl-table'><thead><tr><th>Passage</th><th>Theme</th><th>Confidence</th>"
                "<th>Specificity</th><th class='r'>Page</th></tr></thead>"
                f"<tbody>{''.join(rows)}</tbody></table>", unsafe_allow_html=True)
    if len(df) > max_rows:
        st.markdown(f'<p class="sl-meta">{max_rows} of {len(df)} shown</p>', unsafe_allow_html=True)


def _step(n: int, title: str, how: str, body: str) -> None:
    st.markdown(f'<div class="sl-stp"><div class="sl-stp-n">{n}</div><div class="sl-stp-b"><div class="sl-stp-t">'
                f'{title}<span>{how}</span></div>{body}</div></div>', unsafe_allow_html=True)


def pipeline_inspector(text: str) -> None:
    """Every step for one piece of text: cleaned, masked, TF-IDF features, probabilities, theme, flag, specificity."""
    model, meta = load_model()
    if model is None:
        empty_state("No trained model.", "Run <code>python train.py</code>.")
        return
    cleaned, passages, dropped = trace.segment_text(text)
    if dropped:
        note(f"The corpus build would drop this block as <b>{dropped}</b>. Inspecting it anyway.", warn=True)
    passage = passages[0] if passages else cleaned
    if len(passages) > 1:
        i = st.selectbox(f"Segmented into {len(passages)} passages of up to {MAX_WORDS} words. Inspect:",
                         range(len(passages)), format_func=lambda k: f"{k + 1}. {passages[k][:90]}…")
        passage = passages[i]
    t = trace.trace(model, passage, meta["threshold"])
    color = THEME_COLORS[t["pred"]]
    sp, sig = t["specificity"], t["specificity"]["signals"]

    left, right = st.columns([1, 1], gap="large")
    with left:
        drop = (f'<p class="sl-meta">The corpus filters would drop this passage: {html.escape(t["filter"])}.</p>'
                if t["filter"] else "")
        _step(1, "Clean and segment", "header, bullet and hyphenation fixes · sentence split",
              f'<div class="sl-code">{html.escape(clean(passage))}</div>{drop}')
        _step(2, "Mask", "company, brand (curated lists), number, year → placeholders",
              f'<div class="sl-code">{html.escape(t["masked"])}</div>')
        flag = (pill(f"low confidence · {t['confidence']:.2f} < {t['threshold']:.2f}", "warn") if t["low_confidence"]
                else pill(f"confidence {t['confidence']:.2f} ≥ {t['threshold']:.2f}", "good"))
        _step(5, "Predicted theme", "confidence = top-class probability",
              f'<p style="margin:4px 0">{theme_label(t["pred"])} &nbsp; {flag}</p>')
        rows = [("Quantities", str(sig["numbers"]), 2 if sig["numbers"] >= 3 else 1 if sig["numbers"] else 0),
                ("Date or deadline", "yes" if sig["time"] else "no", int(sig["time"])),
                ("Named initiative or brand", ", ".join(sig["named"]) or "none", int(bool(sig["named"]))),
                ("Commitment language", "yes" if sig["commitment"] else "no", int(sig["commitment"]))]
        body = "".join(f"<tr><td>{a}</td><td>{html.escape(b)}</td><td class='r'>+{c}</td></tr>" for a, b, c in rows)
        _step(6, "Specificity", "regex + spaCy NER on unmasked text · 0–1 Vague · 2 Moderate · 3–5 Specific",
              f'<table class="sl-table"><tbody>{body}<tr><td><b>Score</b></td><td>{sp["tier"]}</td>'
              f'<td class="r"><b>{sp["score"]}/5</b></td></tr></tbody></table>')
    with right:
        feats = "".join(f"<tr><td>{html.escape(f['term'])}</td><td class='r'>{f['tfidf']:.3f}</td>"
                        f"<td class='r'>{f['coef']:+.2f}</td><td class='r'>{f['contribution']:+.3f}</td></tr>"
                        for f in t["contributions"]) or "<tr><td class='sl-small' colspan='4'>no known terms</td></tr>"
        _step(3, "TF-IDF features", "1–2 grams · sublinear tf · stop words removed",
              f'<p class="sl-meta">{t["n_features"]} vocabulary terms present. Largest effects on '
              f'{html.escape(THEME_NAMES[t["pred"]])}: tf-idf weight × class coefficient.</p>'
              f'<table class="sl-table"><thead><tr><th>Term</th><th class="r">tf-idf</th><th class="r">coef</th>'
              f'<th class="r">effect</th></tr></thead><tbody>{feats}</tbody></table>')
        probs = "".join(f"<tr><td>{theme_label(p['label'])}</td><td class='r'>{conf_bar(p['prob'], 0)}</td></tr>"
                        for p in t["probs"])
        _step(4, "Logistic regression", "balanced class weights · probabilities over 7 classes",
              f'<table class="sl-table"><tbody>{probs}</tbody></table>')
    st.markdown(f'<div class="sl-quote" style="border-left-color:{color}"><div class="q">'
                f'{highlight(clean(passage), [f["term"] for f in t["contributions"] if f["contribution"] > 0], color)}'
                f'</div><div class="m">highlighted: terms pushing toward {t["pred"]} (association, not causation)'
                f'</div></div>', unsafe_allow_html=True)
