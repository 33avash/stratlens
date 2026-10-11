"""The Company view (brief → benchmark → evidence), shared by the Company page and Add a report."""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from . import charts, feature_views, insights, report, ui
from .config import COMPANIES, PEER_GROUPS, STRATEGIC_THEMES, THEME_NAMES, peers_of
from .llm import narrate
from .llm.client import get_client
from .theme import PLOTLY_CONFIG


def render(an: dict, passages: pd.DataFrame, company: str, year: str, focus: str | None = None,
           narrative: dict | None = None, uid: str = "co", on_theme_click=None, feats: dict | None = None) -> None:
    key = f"{company}|{year}"
    r = an["reports"][key]
    group = COMPANIES.get(company, {}).get("peer_group")
    peer_label = " and ".join(ui.short(p) for p in peers_of(company) if f"{p}|{year}" in an["reports"]) or None
    seg = COMPANIES.get(company, {}).get("segment", "")

    # ---------- 1. brief ----------
    nar = narrative
    if nar is None:  # not precomputed (e.g. an uploaded report): write it from the data (stored rewrite only)
        tpl, tok = insights.brief_template(an, company, year)
        text, src = narrate.narrate(tpl, tok, get_client(), cache_only=True)
        nar = {"text": text, "source": src}

    rows = [{"theme": THEME_NAMES[t], "code": t, "share": r["themes"][t]["share"],
             "peer": r["peer_avg"]["themes"][t]["share"] if r.get("peer_avg") else None,
             "all": r["all_avg"]["themes"][t]["share"] if r.get("all_avg") else None} for t in STRATEGIC_THEMES]
    rows_sorted = sorted(rows, key=lambda x: -x["share"])

    def _pdf() -> bytes:
        ev = {}
        for x in rows_sorted[:3]:
            df = ui.evidence_rows(passages, company, year, x["code"]).head(2)
            ev[x["theme"]] = [{"text": report.clean(t), "page": int(p)} for t, p in zip(df.text, df.page)]
        return report.company_pdf(f"{ui.company_name(company)} · {year}", nar["text"], rows_sorted, peer_label, ev)

    with ui.card(f"{uid}_brief", f"{ui.company_name(company)} · {year}",
                 meta=f"{seg}{' · peers: ' + peer_label if peer_label else ''}"):
        st.markdown(f'<p class="sl-lede">{html.escape(nar["text"])}</p>', unsafe_allow_html=True)
        b1, b2 = st.columns([4, 1], vertical_alignment="center")
        with b1:
            ui.meta(("Wording polished by AI; every number is inserted from the data." if nar["source"] != "template"
                     else "Written from the data.")
                    + f" {r['n_strategic']} of {r['n_passages']} passages read are about strategy.")
        with b2:
            try:
                st.download_button("PDF brief", _cached_pdf(key, nar["text"], _pdf), icon=":material/download:",
                                   file_name=f"stratlens_{company}_{year}.pdf", mime="application/pdf",
                                   width="stretch", key=f"{uid}_pdf")
            except Exception:
                pass

    def evidence_block(code: str, n: int = 3, more: int = 7) -> None:
        df = ui.evidence_rows(passages, company, year, code)
        if df.empty:
            ui.empty_state("No passages on this theme with enough model confidence.")
            return
        ui.passages_block(df.head(n))
        if len(df) > n:
            with st.expander(f"{min(len(df) - n, more)} more of {len(df)}"):
                ui.passages_block(df.iloc[n:n + more])

    if focus:
        t = r["themes"][focus]
        shown = len(ui.evidence_rows(passages, company, year, focus))
        with ui.card(f"{uid}_focus", f"{THEME_NAMES[focus]}: {t['share'] * 100:.0f}% of strategy passages",
                     meta=f"{t['count']} passages · {shown} confident enough to show · most typical first"):
            ui.reliability_note(company)
            evidence_block(focus)

    # ---------- 2. benchmark ----------
    if r.get("peer_avg") and peer_label:
        lean = max(rows, key=lambda x: x["share"] - x["peer"])
        title = (f"Leans furthest from {peer_label} on {lean['theme'].lower()}" if lean["share"] > lean["peer"]
                 else "Close to its peers on every theme")
    else:
        lean = max(rows, key=lambda x: x["share"] - (x["all"] or 0))
        title = f"Stands out from the average on {lean['theme'].lower()}"
    with ui.card(f"{uid}_bench", title, meta="share of strategy passages · click a bar for its passages"):
        fig = charts.benchmark(rows_sorted, ui.short(company), peer_label)
        ev = st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, on_select="rerun",
                             selection_mode="points", key=f"{uid}_bench_{key}")
        pts = (ev.selection.points if ev and ev.selection else []) or []
        if pts and pts[0].get("customdata") and on_theme_click:
            on_theme_click(pts[0]["customdata"])
        group_name = PEER_GROUPS.get(group, "")
        n_all = len(r["all_avg"]["companies"]) if r.get("all_avg") else 0
        ui.meta(f"Peers: {peer_label or 'no peer report this year'}"
                + (f" ({group_name.lower()})" if group_name and peer_label else "")
                + f". All companies: the average of the {n_all} companies with a {year} report.")
        conc = [f"{ui.short(company)} {r['concreteness']:.1f}"] if r.get("concreteness") is not None else []
        if r.get("peer_avg") and r["peer_avg"].get("concreteness") is not None and peer_label:
            conc.append(f"{peer_label} {r['peer_avg']['concreteness']:.1f}")
        if r.get("all_avg") and r["all_avg"].get("concreteness") is not None:
            conc.append(f"all companies {r['all_avg']['concreteness']:.1f}")
        st.markdown(ui.link("#evidence", "See evidence ↓"), unsafe_allow_html=True)
        if conc:
            ui.meta(ui.tip("Concreteness", ui.TIPS["concreteness"]) + " of the strategy passages, 0–5: "
                    + " · ".join(conc) + ".")

    # ---------- 3-5. what's new, promises, signature phrases ----------
    if feats is not None:
        feature_views.whats_new(passages, feats, an, company, year, uid)
        feature_views.promises(passages, feats, an, company, year, uid)
        feature_views.signature(passages, feats, company, year, uid)

    # ---------- 6. evidence ----------
    st.markdown('<div id="evidence"></div>', unsafe_allow_html=True)
    with ui.card(f"{uid}_evidence", "Evidence", meta="the most typical passages per theme · " + ui.tip("low-confidence hidden", ui.TIPS["confidence"])):
        ui.reliability_note(company)
        for x in rows_sorted:
            t = r["themes"][x["code"]]
            with st.expander(f"{x['theme']} · {t['share'] * 100:.0f}% · {t['count']} passages"):
                evidence_block(x["code"])


@st.cache_data(show_spinner=False, max_entries=24)
def _cached_pdf(key: str, text: str, _make) -> bytes:
    return _make()
