"""Renderers for the Phase 4 sections: what's new / only in A or B, promises made, signature phrases."""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from . import ui
from .config import THEME_COLORS, THEME_NAMES
from .report import clean


def _rows(passages: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    idx = passages.set_index("passage_id")
    ids = [i for i in ids if i in idx.index]
    return idx.loc[ids].reset_index()


def _group_title(g: dict, verb: str) -> str:
    label = f" · “{g['label']}”" if g.get("label") else ""
    return f"{THEME_NAMES[g['theme']]} · {g['n']} {verb}{label}"


def novelty_columns(passages: pd.DataFrame, nov: dict, left: tuple[str, str], right: tuple[str, str],
                    uid: str) -> None:
    """Two columns of theme groups. left/right: (side key in nov, heading)."""
    c1, c2 = st.columns(2, gap="medium")
    for col, (side, heading) in zip((c1, c2), (left, right)):
        with col:
            part = nov[side]
            ui.subhead(f"{heading} <span class='sl-small'>· {part['n']} of {part['n_strategic']} strategy "
                       "passages</span>")
            if not part["groups"]:
                ui.empty_state("Nothing here: every strategy passage has a close counterpart.")
            for i, g in enumerate(part["groups"]):
                verb = "passages" if g["n"] != 1 else "passage"
                with st.expander(_group_title(g, verb), expanded=(i == 0)):
                    rows = _rows(passages, [x["passage_id"] for x in g["items"]])
                    if rows.empty:
                        ui.empty_state("Only very short fragments; nothing worth showing.")
                    for _, row in rows.iterrows():
                        ui.passage(row)
                    if g["n"] > len(rows):
                        ui.meta(f"Showing the {len(rows)} clearest of {g['n']}.")


def whats_new(passages: pd.DataFrame, feats: dict, an: dict, company: str, year: str, uid: str) -> None:
    y = an["yoy"].get(company)
    prev = [yy for yy in an["years"] if yy < year]
    if not y or y["to"] != year:
        with ui.card(f"{uid}_new", "What's new this year"):
            if prev and f"{company}|{prev[-1]}" not in an["reports"]:
                ui.empty_state(f"The {prev[-1]} report is missing, so there is nothing to compare against.")
            else:
                ui.empty_state("This is the earliest report here; pick a later year to see what changed.")
        return
    nov = feats.get("novelty", {}).get(f"{company}|{y['from']}~{company}|{y['to']}")
    if not nov:
        return
    new = nov["only_in_b"]
    top = new["groups"][0] if new["groups"] else None
    title = (f"{new['n']} new strategy passages in {year}, most on {THEME_NAMES[top['theme']].lower()}" if top
             else f"Nothing substantially new in {year}")
    with ui.card(f"{uid}_new", title, meta=f"compared with {y['from']} · by meaning, not wording"):
        novelty_columns(passages, nov, ("only_in_b", f"New in {y['to']}"), ("only_in_a", f"Dropped since {y['from']}"),
                        uid)
        ui.meta(f"A passage counts as new when nothing in the {y['from']} report is close to it in meaning; "
                "dropped is the reverse. The quoted words are the group's most characteristic phrase. "
                "How it works explains the method.")


def _promise_row(p: dict, extra: str = "") -> str:
    s = html.escape(clean(p["sentence"]))
    if p.get("target"):
        t = html.escape(p["target"])
        s = s.replace(t, f"<b>{t}</b>", 1)
    dl = html.escape(p["deadline"]) if p.get("deadline") else "<span class='sl-small'>no date</span>"
    return (f"<tr><td class='t'>{s}</td><td>{ui.theme_label(p['theme'])}</td><td>{dl}</td>"
            f"<td class='r'>p.{p['page']}</td>{extra}</tr>")


def promises(passages: pd.DataFrame, feats: dict, an: dict, company: str, year: str, uid: str) -> None:
    key = f"{company}|{year}"
    prom = feats.get("promises", {}).get(key, [])
    fu = feats.get("follow_up", {}).get(key, [])
    y = an["yoy"].get(company)
    title = (f"{len(prom)} measurable promise{'s' if len(prom) != 1 else ''}"
             + (f", {sum(p['deadline'] is not None for p in prom)} with a deadline" if prom else "")
             if prom else "No measurable promises found")
    with ui.card(f"{uid}_prom", title, meta="a figure or target + a deadline or commitment"):
        if prom:
            rows = "".join(_promise_row(p) for p in prom[:12])
            st.markdown("<table class='sl-table sl-prom'><thead><tr><th>Promise</th><th>Theme</th><th>Deadline</th>"
                        f"<th class='r'>Page</th></tr></thead><tbody>{rows}</tbody></table>", unsafe_allow_html=True)
            if len(prom) > 12:
                ui.meta(f"12 of {len(prom)} shown, dated ones first.")
        else:
            ui.empty_state("The letter and MD&A contain no sentence that pairs a figure with a deadline or a "
                           "commitment such as \"we will\" or \"we aim to\".")
        if y and y["to"] == year:
            ui.subhead(f"Last year's promises: were they mentioned again in {year}?")
            if not fu:
                ui.empty_state(f"The {y['from']} report made no measurable promises.")
            else:
                idx = passages.set_index("passage_id")
                rows = []
                for p in fu:
                    hit = p["status"] == "mentioned again"
                    pill = ui.pill(p["status"], "good" if hit else "")
                    if hit and p.get("match") in idx.index:
                        m = idx.loc[p["match"]]
                        shared = ", ".join(p.get("shared", [])[:4])
                        pill += (f"<div class='sl-small' title='{html.escape(clean(m['text']))}'>{year} p."
                                 f"{int(m['page'])}" + (f" · shares: {html.escape(shared)}" if shared else "")
                                 + "</div>")
                    rows.append(_promise_row(p, f"<td>{pill}</td>"))
                st.markdown("<table class='sl-table sl-prom'><thead><tr><th>Promise in " + y["from"] + "</th>"
                            "<th>Theme</th><th>Deadline</th><th class='r'>Page</th><th>Follow-up</th></tr></thead>"
                            f"<tbody>{''.join(rows)}</tbody></table>", unsafe_allow_html=True)
                ui.meta("\"Mentioned again\" means a passage this year is close in meaning. It does not mean the "
                        "promise was achieved.")


def signature(passages: pd.DataFrame, feats: dict, company: str, year: str, uid: str) -> None:
    sig = feats.get("signature", {}).get(year, {}).get(company, {})
    first = next(iter(sorted(sig.items(), key=lambda kv: -max(x["n"] for x in kv[1]))), None)
    title = (f"Signature phrase: “{first[1][0]['phrase']}” ({THEME_NAMES[first[0]].lower()})" if first
             else "Signature phrases")
    with ui.card(f"{uid}_sig", title, meta=f"words this company uses on a theme that its peers don't · {year}"):
        if not sig:
            ui.empty_state("Not enough text to find distinctive phrases.")
            return
        idx = passages.set_index("passage_id")
        cells = []
        for t, items in sig.items():
            chips = []
            for x in items:
                pages = sorted({int(idx.loc[i, "page"]) for i in x["passage_ids"] if i in idx.index})
                chips.append(f"<span title='In {x['n']} passages, e.g. p. {', '.join(map(str, pages[:3]))}'>"
                             f"{html.escape(x['phrase'])}<em>{x['n']}</em></span>")
            cells.append(f"<div class='sl-sig'><div class='h'><span class='sl-dot' style='background:"
                         f"{THEME_COLORS[t]}'></span>{html.escape(THEME_NAMES[t])}</div><div class='sl-terms'>"
                         f"{''.join(chips)}</div></div>")
        st.markdown(f"<div class='sl-sigs'>{''.join(cells)}</div>", unsafe_allow_html=True)
        ui.meta("Two- and three-word phrases that are common in this company's passages on a theme and rare in "
                "the other companies'. Brand and company names are removed. The number is how many passages use "
                "it; hover for pages.")
