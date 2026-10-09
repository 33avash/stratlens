"""Shared Streamlit components and cached artefact loaders. All styling lives in assets/style.css."""
from __future__ import annotations

import html
import json

import pandas as pd
import streamlit as st

from . import explain
from . import model as mdl
from .config import (BENCHMARK_CSV, COMPANY_NAMES, METRICS_JSON, MODEL_PATH, ROOT, SECTION_NAMES, THEME_NAMES)
from .theme import THEME_COLORS, tint

CSS_PATH = ROOT / "assets" / "style.css"

DISCLOSURE_NOTE = ("Shares measure <b>disclosure emphasis</b>: how much of each report's strategic narrative is "
                   "spent on a theme. They do not measure investment, execution or performance, and are not a ranking.")


# ---------- loaders (artefacts are produced by train.py; the app never retrains) ----------
@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        return None, None
    return mdl.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict | None:
    if not METRICS_JSON.exists():
        return None
    return json.loads(METRICS_JSON.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_benchmark() -> pd.DataFrame | None:
    if not BENCHMARK_CSV.exists():
        return None
    df = pd.read_csv(BENCHMARK_CSV)
    df["source"] = "Sample analysis"
    return df


@st.cache_data(show_spinner=False, max_entries=2000)
def explain_text(text: str) -> dict | None:
    model, _ = load_model()
    return None if model is None else explain.explain(model, text, k=8)


# ---------- page scaffolding ----------
def inject_css() -> None:
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="sl-title">{title}</div><p class="sl-sub">{subtitle}</p>', unsafe_allow_html=True)


def section(title: str, caption: str, first: bool = False) -> None:
    cls = "" if first else "sl-section"
    st.markdown(f'<div class="{cls}"><div class="sl-h2">{title}</div><p class="sl-cap">{caption}</p></div>',
                unsafe_allow_html=True)


def note(text: str, warn: bool = False) -> None:
    st.markdown(f'<div class="sl-note{" sl-warn" if warn else ""}">{text}</div>', unsafe_allow_html=True)


def kpi_strip(items: list[tuple[str, str]]) -> None:
    cells = "".join(f'<div class="sl-kpi"><div class="v">{v}</div><div class="l">{l}</div></div>' for v, l in items)
    st.markdown(f'<div class="sl-kpis">{cells}</div>', unsafe_allow_html=True)


def steps(labels: list[str], active: int) -> None:
    parts = []
    for i, lab in enumerate(labels):
        cls = "active" if i == active else ("done" if i < active else "")
        parts.append(f'<div class="sl-step {cls}"><b>{i + 1}</b>{lab}</div>')
    st.markdown(f'<div class="sl-steps">{"".join(parts)}</div>', unsafe_allow_html=True)


def empty_state(sentence: str, action: str) -> None:
    st.markdown(f'<div class="sl-note">{sentence} {action}</div>', unsafe_allow_html=True)


# ---------- small formatters ----------
def theme_label(code: str, full: bool = True) -> str:
    name = THEME_NAMES.get(code, code) if full else code
    return (f'<span class="sl-theme"><span class="sl-dot" style="background:{THEME_COLORS.get(code, "#999")}"></span>'
            f'{html.escape(name)}</span>')


def conf_bar(c: float) -> str:
    return f'<span class="sl-bar"><span style="width:{c * 100:.0f}%"></span></span><span class="num">{c:.2f}</span>'


def company_name(code: str) -> str:
    return COMPANY_NAMES.get(code, code.title())


def section_name(code: str) -> str:
    return SECTION_NAMES.get(code, code)


def highlight(text: str, terms: list[str], color: str) -> str:
    """HTML-escape text and wrap matched terms in a soft tint of the theme colour."""
    spans = explain.highlight_spans(text, terms)
    out, last = [], 0
    for s, e in spans:
        out.append(html.escape(text[last:s]))
        out.append(f'<mark style="background:{tint(color, 0.18)}">{html.escape(text[s:e])}</mark>')
        last = e
    out.append(html.escape(text[last:]))
    return "".join(out)


def quote(row, show_terms: bool = True) -> None:
    """Evidence passage styled as a quotation with meta line and highlighted contributing terms."""
    theme = row["pred"]
    color = THEME_COLORS.get(theme, "#999")
    body = html.escape(row["text"])
    if show_terms:
        ex = explain_text(row["text"])
        if ex and ex["pred"] == theme:
            body = highlight(row["text"], [t["term"] for t in ex["positive"][:6]], color)
    flag = ' · <span class="sl-flag">needs review</span>' if row.get("needs_review") else ""
    meta = (f'{company_name(row["company"])} · {row["year"]} · {section_name(row["section"])} · p. {int(row["page"])} · '
            f'confidence {row["confidence"]:.2f}{flag}')
    st.markdown(f'<div class="sl-quote" style="border-left-color:{color}"><div class="q">{body}</div>'
                f'<div class="m">{meta}</div></div>', unsafe_allow_html=True)


def predictions_table(df: pd.DataFrame, max_rows: int = 50) -> None:
    rows = []
    for r in df.head(max_rows).itertuples():
        flag = '<span class="sl-flag">review</span>' if r.needs_review else ""
        rows.append(f"<tr><td class='t'>{html.escape(r.text)}</td><td>{theme_label(r.pred)}</td>"
                    f"<td>{conf_bar(r.confidence)}</td><td>{flag}</td><td class='r'>{int(r.page)}</td></tr>")
    st.markdown("<table class='sl-table'><thead><tr><th>Passage</th><th>Theme</th><th>Confidence</th><th></th>"
                f"<th class='r'>Page</th></tr></thead><tbody>{''.join(rows)}</tbody></table>", unsafe_allow_html=True)
    if len(df) > max_rows:
        st.markdown(f'<p class="sl-meta">Showing {max_rows} of {len(df)} passages. Download the CSV for all rows.</p>',
                    unsafe_allow_html=True)


def why_label(text: str) -> None:
    """Local explanation: class probabilities and contributing terms for one passage."""
    ex = explain_text(text)
    if ex is None:
        empty_state("No trained model found.", "Run <code>python train.py</code> first.")
        return
    color = THEME_COLORS[ex["pred"]]
    st.markdown(f'<div class="sl-quote" style="border-left-color:{color}"><div class="q">'
                f'{highlight(text, [t["term"] for t in ex["positive"]], color)}</div></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.markdown('<p class="sl-cap">Top class probabilities</p>', unsafe_allow_html=True)
        rows = "".join(f"<tr><td>{theme_label(t['label'])}</td><td>{conf_bar(t['prob'])}</td></tr>"
                       for t in ex["top_classes"])
        st.markdown(f"<table class='sl-table'><tbody>{rows}</tbody></table>", unsafe_allow_html=True)
    with c2:
        st.markdown(f'<p class="sl-cap">Terms pushing toward {ex["pred"]}</p>', unsafe_allow_html=True)
        rows = "".join(f"<tr><td>{html.escape(t['term'])}</td><td class='r'>+{t['contribution']:.3f}</td></tr>"
                       for t in ex["positive"]) or "<tr><td class='sl-small'>No known terms</td></tr>"
        st.markdown(f"<table class='sl-table'><tbody>{rows}</tbody></table>", unsafe_allow_html=True)
    with c3:
        st.markdown(f'<p class="sl-cap">Terms pulling away from {ex["pred"]}</p>', unsafe_allow_html=True)
        rows = "".join(f"<tr><td>{html.escape(t['term'])}</td><td class='r'>{t['contribution']:.3f}</td></tr>"
                       for t in ex["negative"]) or "<tr><td class='sl-small'>None</td></tr>"
        st.markdown(f"<table class='sl-table'><tbody>{rows}</tbody></table>", unsafe_allow_html=True)
    st.markdown('<p class="sl-meta">Contribution = tf-idf weight × class coefficient. Weights show statistical '
                'association with a label in the training data, not causation.</p>', unsafe_allow_html=True)
