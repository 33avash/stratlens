"""Benchmark (landing): theme emphasis per company and year, YoY shift and evidence drill-down."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stratlens import compare, ui
from stratlens.config import STRATEGIC_THEMES, THEME_NAMES
from stratlens.theme import (BORDER, MUTED, NEGATIVE, PLOTLY_CONFIG, POSITIVE, SEQUENTIAL, TEXT, company_color)

ui.page_header("Strategy benchmark",
               "How competing FMCG companies divide their strategic narrative across six themes, and how that "
               "emphasis shifted between financial years. Every number links to the passages behind it.")

base = ui.load_benchmark()
metrics = ui.load_metrics()
if base is None:
    ui.empty_state("No sample analysis is available yet.",
                   "Add reports to <code>data/raw/</code>, then run <code>python scripts/build_corpus.py</code> and "
                   "<code>python train.py</code>.")
    st.stop()

session = st.session_state.get("session_reports", {})
include_session = False

# ---------- controls ----------
c1, c2, _ = st.columns([2, 2, 1])
with c1:
    exclude_review = st.toggle("Exclude low-confidence passages", value=False,
                               help="Drops passages whose top-class probability is below the review threshold.")
with c2:
    if session:
        include_session = st.toggle(f"Include {len(session)} uploaded report(s)", value=True)
    else:
        st.markdown('<p class="sl-meta" style="padding-top:8px">Upload more reports on the Analyse reports page.</p>',
                    unsafe_allow_html=True)

preds = base
if include_session:
    extra = pd.concat(session.values(), ignore_index=True)
    keys = set(map(tuple, extra[["company", "year"]].drop_duplicates().values))
    preds = pd.concat([base[~base[["company", "year"]].apply(tuple, axis=1).isin(keys)], extra], ignore_index=True)

shares = compare.theme_shares(preds, exclude_review)
yoy = compare.yoy_change(shares)
companies = sorted(shares["company"].unique())
years = sorted(shares["year"].unique())
strategic = compare.strategic(preds, exclude_review)

thr = metrics["threshold"]["value"] if metrics else None
ui.kpi_strip([
    (f"{preds[['company', 'year']].drop_duplicates().shape[0]}", "Reports analysed"),
    (f"{len(strategic):,}", f"Strategic passages (of {len(preds):,})"),
    (f"{metrics['test']['model']['macro_f1']:.2f}" if metrics else "–", "Test macro-F1 (held-out)"),
    (f"{int(preds['needs_review'].sum()):,}", f"Passages needing review (conf. < {thr:.2f})" if thr else "Needs review"),
])
ui.note(ui.DISCLOSURE_NOTE)

# ---------- fingerprint heatmap ----------
ui.section("Strategy fingerprint",
           "Share of each report's strategic passages assigned to each theme. Rows sum to 100%. "
           "Darker cells mean more of the narrative is spent on that theme.")
rows = [(c, y) for c in companies for y in years if ((shares.company == c) & (shares.year == y)).any()]
z, txt, hover = [], [], []
for c, y in rows:
    part = shares[(shares.company == c) & (shares.year == y)].set_index("theme")
    z.append([part.loc[t, "share"] * 100 for t in STRATEGIC_THEMES])
    txt.append([f"{part.loc[t, 'share'] * 100:.0f}%" for t in STRATEGIC_THEMES])
    hover.append([f"{ui.company_name(c)} {y}<br>{THEME_NAMES[t]}<br>{part.loc[t, 'share'] * 100:.1f}% "
                  f"({int(part.loc[t, 'count'])} of {int(part.loc[t, 'strategic_total'])} passages)"
                  for t in STRATEGIC_THEMES])
zmax = max(max(r) for r in z) if z else 1
xs = [THEME_NAMES[t].replace(" & ", " &<br>", 1) for t in STRATEGIC_THEMES]
ys = [f"{ui.company_name(c)} · {y}" for c, y in rows]
heat = go.Figure(go.Heatmap(z=z, x=xs, y=ys, hovertext=hover, hoverinfo="text", colorscale=SEQUENTIAL,
                            zmin=0, zmax=zmax, showscale=False, xgap=2, ygap=2))
# cell labels as annotations so dark cells get white text and light cells dark text
for i, r in enumerate(z):
    for j, v in enumerate(r):
        heat.add_annotation(x=xs[j], y=ys[i], text=txt[i][j], showarrow=False,
                            font=dict(size=12, color="#FFFFFF" if v > zmax * 0.62 else TEXT))
heat.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=TEXT, size=12))
heat.update_xaxes(side="top", tickangle=0, tickfont=dict(color=TEXT, size=12))
heat.update_layout(height=90 + 46 * len(rows), margin=dict(l=8, r=8, t=56, b=8))
st.plotly_chart(heat, use_container_width=True, config=PLOTLY_CONFIG)

# ---------- theme comparison ----------
ui.section("Theme comparison",
           "One bar per company for the selected year. Click a bar to load its passages in the evidence panel below.")
yr = st.segmented_control("Year", years, default=years[-1], key="cmp_year", label_visibility="collapsed") or years[-1]
sub = shares[shares.year == yr]
bars = go.Figure()
order = list(reversed(STRATEGIC_THEMES))
for i, c in enumerate(companies):
    part = sub[sub.company == c].set_index("theme")
    if part.empty:
        continue
    vals = [part.loc[t, "share"] * 100 for t in order]
    bars.add_bar(
        y=[THEME_NAMES[t] for t in order], x=vals, orientation="h", name=ui.company_name(c),
        marker=dict(color=company_color(c, i), line=dict(color="#FFFFFF", width=1)),
        text=[f"{v:.0f}%" for v in vals], textposition="outside", textfont=dict(size=11, color=MUTED),
        customdata=[[c, yr, t, int(part.loc[t, "count"]), int(part.loc[t, "strategic_total"])] for t in order],
        hovertemplate="%{fullData.name} " + yr + "<br>%{y}: %{x:.1f}%<br>%{customdata[3]} of %{customdata[4]} "
                      "strategic passages<extra></extra>")
bars.update_layout(barmode="group", height=120 + 62 * len(order), bargap=0.3, bargroupgap=0.06,
                   legend=dict(traceorder="normal"))
bars.update_xaxes(showgrid=True, gridcolor="#EEEAE3", ticksuffix="%", rangemode="tozero",
                  range=[0, sub["share"].max() * 118 if not sub.empty else 1])
bars.update_yaxes(showgrid=False, tickfont=dict(color=TEXT, size=12))
ev_bar = st.plotly_chart(bars, use_container_width=True, config=PLOTLY_CONFIG, on_select="rerun",
                         selection_mode="points", key="cmp_chart")

# ---------- YoY shift ----------
ui.section("Year-over-year shift",
           "Change in theme share between the two most recent years, in percentage points. Navy = more emphasis, "
           "terracotta = less. Click a bar to see the latest year's passages.")
missing = compare.companies_missing_year(shares)
ev_yoy = None
if yoy.empty:
    ui.empty_state("Year-over-year change needs two years of reports for at least one company.",
                   "Add a second year on the Analyse reports page.")
else:
    yoy_companies = sorted(yoy.company.unique())
    cols = st.columns(len(yoy_companies))
    lim = max(abs(yoy["pp_change"]).max() * 1.35, 2)
    for col, c in zip(cols, yoy_companies):
        part = yoy[yoy.company == c].set_index("theme").loc[order]
        fig = go.Figure(go.Bar(
            y=[THEME_NAMES[t] for t in order], x=part["pp_change"], orientation="h",
            marker=dict(color=[POSITIVE if v >= 0 else NEGATIVE for v in part["pp_change"]]),
            text=[f"{v:+.1f}" for v in part["pp_change"]], textposition="outside",
            textfont=dict(size=11, color=MUTED), cliponaxis=False,
            customdata=[[c, r.year_to, t, r.share_from * 100, r.share_to * 100, r.count_from, r.count_to]
                        for t, r in part.iterrows()],
            hovertemplate="%{y}<br>%{customdata[3]:.1f}% → %{customdata[4]:.1f}% (%{x:+.1f} pp)<br>"
                          "passages %{customdata[5]} → %{customdata[6]}<extra></extra>"))
        fig.update_xaxes(range=[-lim, lim], zeroline=True, zerolinecolor=BORDER, zerolinewidth=1, showgrid=False,
                         ticksuffix=" pp", tickfont=dict(size=10))
        fig.update_yaxes(showgrid=False, showticklabels=(c == yoy_companies[0]), tickfont=dict(color=TEXT, size=12))
        fig.update_layout(height=330, margin=dict(l=8, r=8, t=28, b=8),
                          title=dict(text=f"{ui.company_name(c)} · {part['year_from'].iloc[0]} → "
                                          f"{part['year_to'].iloc[0]}", font=dict(size=13, color=TEXT), x=0, y=0.98))
        with col:
            ev = st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, on_select="rerun",
                                 selection_mode="points", key=f"yoy_{c}")
            if ev and ev.selection and ev.selection.points:
                ev_yoy = ev
if not yoy.empty:
    totals = shares[shares.company.isin(yoy.company)].groupby(["company", "year"])["strategic_total"].first()
    st.markdown(f'<p class="sl-meta">Scale check: these reports have {totals.min()}–{totals.max()} strategic passages, '
                f'so a single passage moves a share by {100 / totals.max():.1f}–{100 / totals.min():.1f} pp. Treat '
                f'shifts of a few points as within noise; hover a bar for passage counts.</p>', unsafe_allow_html=True)
if missing:
    st.markdown('<p class="sl-meta">Not shown (one year only): ' + ", ".join(
        f"{ui.company_name(c)} ({', '.join(sorted(shares[shares.company == c].year.unique()))})" for c in missing)
        + ".</p>", unsafe_allow_html=True)

# ---------- evidence ----------
ui.section("Evidence",
           "The passages behind a number, most confident first. Highlighted words are the terms that contributed "
           "most to the predicted theme.")

# apply a chart click to the evidence selectors (only when the click is new)
for ev in (ev_bar, ev_yoy):
    if ev and ev.selection and ev.selection.points:
        cd = ev.selection.points[0].get("customdata")
        if cd:
            sig = (cd[0], cd[1], cd[2])
            if st.session_state.get("_ev_sig") != sig:
                st.session_state["_ev_sig"] = sig
                st.session_state["ev_company"], st.session_state["ev_year"], st.session_state["ev_theme"] = sig

e1, e2, e3, e4 = st.columns([1.2, 0.8, 1.6, 1])
with e1:
    company = st.selectbox("Company", companies, format_func=ui.company_name, key="ev_company")
with e2:
    cyears = sorted(shares[shares.company == company].year.unique())
    if st.session_state.get("ev_year") not in cyears:
        st.session_state["ev_year"] = cyears[-1]
    year = st.selectbox("Year", cyears, key="ev_year")
with e3:
    theme = st.selectbox("Theme", STRATEGIC_THEMES, format_func=lambda t: THEME_NAMES[t], key="ev_theme")
with e4:
    n_show = st.number_input("Passages to show", min_value=3, max_value=30, value=6, step=3)

cell = shares[(shares.company == company) & (shares.year == year) & (shares.theme == theme)]
ev_df = compare.evidence(preds, company, theme, year, exclude_review)
if cell.empty or ev_df.empty:
    ui.empty_state(f"No {THEME_NAMES[theme].lower()} passages for {ui.company_name(company)} {year}.",
                   "Choose another theme or year.")
else:
    r = cell.iloc[0]
    st.markdown(f'<p class="sl-meta">{ui.theme_label(theme)} &nbsp;·&nbsp; <span class="num">{int(r["count"])} of '
                f'{int(r["strategic_total"])}</span> strategic passages = <b class="num">{r["share"] * 100:.1f}%</b> '
                f'of {ui.company_name(company)} {year}</p>', unsafe_allow_html=True)
    for _, row in ev_df.head(int(n_show)).iterrows():
        ui.quote(row)
    export = ev_df[["passage_id", "company", "year", "section", "page", "pred", "confidence", "needs_review", "text"]]
    st.download_button("Download these passages (CSV)", export.to_csv(index=False).encode("utf-8"),
                       file_name=f"stratlens_{company}_{year}_{theme}.csv", mime="text/csv")
