"""Overview: theme emphasis per report, year-over-year shifts, specificity and evidence."""
import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from stratlens import compare, report, theme, ui, uncertainty
from stratlens.config import STRATEGIC_THEMES, THEME_NAMES
from stratlens.theme import MUTED, NEGATIVE, PLOTLY_CONFIG, POSITIVE, THEME_COLORS

SHORT = {"PROD": "Product", "DIST": "Distribution", "DIGI": "Digital", "OPS": "Operations",
         "SUST": "Sustainability", "BRAND": "Brand"}

base = ui.load_benchmark()
metrics = ui.load_metrics()
if base is None:
    ui.header("Overview")
    ui.empty_state("No analysis yet.", "Run <code>python train.py</code>.")
    st.stop()

session = st.session_state.get("session_reports", {})
include_session = False
if session:
    h1, h2 = st.columns([4, 1], vertical_alignment="bottom")
    with h1:
        ui.header("Overview", "What HUL, Dabur and Nestlé India emphasise in their strategy, read by an NLP model.")
    with h2:
        include_session = st.toggle("Include uploaded reports", value=True)
else:
    ui.header("Overview", "What HUL, Dabur and Nestlé India emphasise in their strategy, read by an NLP model.")

preds = base
if session and include_session:
    extra = pd.concat(session.values(), ignore_index=True)
    keys = set(map(tuple, extra[["company", "year"]].drop_duplicates().values))
    preds = pd.concat([base[~base[["company", "year"]].apply(tuple, axis=1).isin(keys)], extra], ignore_index=True)

shares = compare.theme_shares(preds)
yoy = compare.yoy_change(shares)
companies = sorted(shares["company"].unique())
years = sorted(shares["year"].unique())
scope = compare.in_scope(preds)
strategic = compare.strategic(preds)
per_rep = preds[["company", "year"]].drop_duplicates().sort_values(["company", "year"])
facts = {}
for c, y in per_rep[["company", "year"]].values:
    try:
        facts[(c, y)] = report.report_facts(preds, c, y, n_evidence=0)
    except ValueError:
        pass

# ---------- KPIs ----------
k = [{"label": "Annual reports", "value": f"{len(per_rep)}", "sub": f"{preds.company.nunique()} companies · "
      f"{years[0]}–{years[-1][-2:]}"},
     {"label": "Passages read", "value": f"{len(scope):,}", "sub": "Chairman's letter + MD&A"},
     {"label": "Strategy passages", "value": f"{len(strategic):,}", "sub": f"<b>{len(strategic) / len(scope):.0%}</b> "
      "classified into 6 themes", "help": "Passages whose predicted theme is not Other. Shares on every page use "
                                          "this as the denominator."}]
if metrics:
    t = metrics["test"]["model"]
    k.append({"label": "Model score (macro-F1)", "value": f"{t['macro_f1']:.2f}",
              "sub": f"unseen text · keywords {metrics['test']['baseline']['macro_f1']:.2f}",
              "help": "Average F1 across the seven themes on held-out passages the model never saw in training."})
ui.kpi_cards(k)

# ---------- what we found (all from the classifier's shares and bootstrap shifts) ----------
found = []
for c in companies:
    latest = [f for (cc, _), f in sorted(facts.items()) if cc == c]
    if not latest:
        continue
    f = latest[-1]
    d = f.get("distinctive_theme")
    if d and d["diff_vs_peers_pp"] > 0:
        row = next(r for r in f["themes"] if r["theme"] == d["theme"])
        found.append((f"{f['company_name']} leans on {SHORT[row['code']].lower()}",
                      f"{row['share_pct']}% of its {f['year']} strategy talk, {d['diff_vs_peers_pp']:+.0f} pp above peers."))
shifts = [(f["company_name"], s_) for f in facts.values() for s_ in f["significant_shifts"]]
if shifts:
    name, s_ = max(shifts, key=lambda x: abs(x[1]["change_pp"]))
    found.append((f"{len(shifts)} year-on-year shift{'s' if len(shifts) > 1 else ''} beyond noise",
                  f"{name}, {s_['theme'].lower()}: {s_['from_pct']}% → {s_['to_pct']}% (95% CI {s_['ci_pp'][0]:+.1f} to "
                  f"{s_['ci_pp'][1]:+.1f} pp). The rest is noise."))
elif len(years) > 1:
    found.append(("No year-on-year shift beyond noise", "Every change sits inside its 95% bootstrap interval."))
if found:
    with ui.card("found", "What we found", meta="disclosure emphasis, not spending"):
        ui.findings(found)

# ---------- fingerprint ----------
with ui.card("fingerprint", "Strategy fingerprint", meta="% of each report's strategy passages · hover for counts"):
    rows_ = [(c, y) for c in companies for y in years if ((shares.company == c) & (shares.year == y)).any()]
    z, txt, hover = [], [], []
    for c, y in rows_:
        part = shares[(shares.company == c) & (shares.year == y)].set_index("theme")
        z.append([part.loc[t, "share"] * 100 for t in STRATEGIC_THEMES])
        txt.append([f"{part.loc[t, 'share'] * 100:.0f}" for t in STRATEGIC_THEMES])
        hover.append([f"{ui.company_name(c)} {y} · {THEME_NAMES[t]}<br>{part.loc[t, 'share'] * 100:.0f}% "
                      f"({int(part.loc[t, 'count'])}/{int(part.loc[t, 'strategic_total'])})" for t in STRATEGIC_THEMES])
    zmax = max(max(r) for r in z) if z else 1
    xs = [SHORT[t] for t in STRATEGIC_THEMES]
    ys = [f"{ui.company_name(c).replace('Hindustan Unilever', 'HUL').replace(' India', '')} {y[-2:]}" for c, y in rows_]
    heat = go.Figure(go.Heatmap(z=z, x=xs, y=ys, hovertext=hover, hoverinfo="text", colorscale=theme.sequential(),
                                zmin=0, zmax=zmax, showscale=False, xgap=4, ygap=4))
    dark = theme.is_dark()
    for i, r in enumerate(z):
        for j, v in enumerate(r):
            strong = v > zmax * 0.55
            heat.add_annotation(x=xs[j], y=ys[i], text=txt[i][j], showarrow=False,
                                font=dict(size=14, family=theme.MONO,
                                          color=("#0C100E" if strong else "#E4E9E5") if dark else
                                          ("#FFFFFF" if strong else "#141A17")))
    heat.update_yaxes(autorange="reversed", showgrid=False)
    heat.update_xaxes(side="top", tickangle=0, showgrid=False, tickfont=dict(family=theme.SANS, size=13))
    theme.style(heat, 70 + 54 * len(rows_), margin=dict(l=4, r=4, t=28, b=4), legend=False)
    st.plotly_chart(heat, use_container_width=True, config=PLOTLY_CONFIG)
    st.markdown('<p class="sl-meta">Stronger colour = more of the strategy narrative. Measures emphasis, not spending.</p>', unsafe_allow_html=True)

# ---------- specificity ----------
def place_labels(xs, ys, labels, xr, yr):
    """Greedy placement: each label takes the first position that overlaps no other label or point."""
    cands = [("top center", 0, 1), ("bottom center", 0, -1), ("middle right", 1, 0), ("middle left", -1, 0),
             ("top right", 1, 1), ("bottom left", -1, -1), ("top left", -1, 1), ("bottom right", 1, -1)]
    pts = [((x - xr[0]) / (xr[1] - xr[0]), (y - yr[0]) / (yr[1] - yr[0])) for x, y in zip(xs, ys)]
    boxes, pos = [], [None] * len(pts)
    for i in sorted(range(len(pts)), key=lambda i: -pts[i][1]):
        px, py = pts[i]
        w, h = 0.0135 * len(labels[i]) + 0.02, 0.09
        chosen = None
        for name, dx, dy in cands:
            cx, cy = px + dx * (w / 2 + 0.025), py + dy * (h / 2 + 0.035)
            box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            inside = 0 <= box[0] and box[2] <= 1.02 and -0.02 <= box[1] and box[3] <= 1.02
            hit = any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]) for b in boxes) or \
                any(box[0] < qx < box[2] and box[1] < qy < box[3] for j, (qx, qy) in enumerate(pts) if j != i)
            if inside and not hit:
                chosen = (name, box)
                break
        chosen = chosen or ("top center", (px - w / 2, py, px + w / 2, py + h))
        pos[i] = chosen[0]
        boxes.append(chosen[1])
    return pos


ev_spec = None
if "specificity" in preds:
    with ui.card("spec", "How much vs how concretely",
                 meta="across: how much it is discussed · up: how concretely (0–5)"):
        yr2 = st.segmented_control("Year ", years, default=years[-1], key="spec_year",
                                   label_visibility="collapsed") or years[-1]
        sp = strategic[strategic.year == yr2].groupby(["company", "pred"]).agg(
            mean_spec=("specificity", "mean"), n=("specificity", "size")).reset_index()
        sp = sp.merge(shares[shares.year == yr2][["company", "theme", "share"]],
                      left_on=["company", "pred"], right_on=["company", "theme"])
        sp = sp[sp.n >= 3]
        cos = [c for c in companies if c in set(sp.company)]
        fig = make_subplots(rows=1, cols=len(cos), shared_yaxes=True, horizontal_spacing=0.04,
                            subplot_titles=[ui.company_name(c) for c in cos])
        xr = [0, max(sp["share"].max() * 100 * 1.12, 10)]
        yr_ = [0, round(max(2.6, sp["mean_spec"].max() * 1.15), 1)]
        for kk, c in enumerate(cos, start=1):
            part = sp[sp.company == c]
            labels = [SHORT[t] for t in part["pred"]]
            fig.add_scatter(
                x=part["share"] * 100, y=part["mean_spec"], mode="markers+text", name=ui.company_name(c),
                text=labels, textposition=place_labels((part["share"] * 100).tolist(), part["mean_spec"].tolist(),
                                                       labels, xr, yr_),
                textfont=dict(size=10.5, color=MUTED), showlegend=False, cliponaxis=False,
                marker=dict(size=[9 + min(n, 70) / 4 for n in part["n"]], color=[THEME_COLORS[t] for t in part["pred"]],
                            opacity=0.9, line=dict(width=0)),
                customdata=[[c, yr2, r.pred, THEME_NAMES[r.pred], int(r.n)] for r in part.itertuples()],
                hovertemplate="%{customdata[3]}<br>%{x:.0f}% of strategy passages · mean specificity %{y:.2f} of 5"
                              " · %{customdata[4]} passages<extra></extra>", row=1, col=kk)
            fig.update_xaxes(range=xr, ticksuffix="%", row=1, col=kk)
        fig.update_yaxes(range=yr_, row=1, col=1, title_text="concreteness", title_font=dict(size=11))
        theme.style(fig, 340, margin=dict(l=4, r=8, t=30, b=4), legend=False)
        fig.update_annotations(font=dict(size=12.5))
        ev_spec = st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, on_select="rerun",
                                  selection_mode="points", key="spec_chart")
        st.markdown('<p class="sl-meta">Concreteness comes from spaCy NER and rules: figures, dates, named initiatives '
                    'and commitments. Click a point to read its passages below.</p>', unsafe_allow_html=True)

# ---------- evidence + YoY ----------
c1, c2 = st.columns([7, 5], gap="medium")
with c1:
    with ui.card("evidence", "Evidence", meta="the passages behind every number"):
        if ev_spec and ev_spec.selection and ev_spec.selection.points:
            cd = ev_spec.selection.points[0].get("customdata")
            if cd:
                sig_key = (cd[0], cd[1], cd[2])
                if st.session_state.get("_ev_sig") != sig_key:
                    st.session_state["_ev_sig"] = sig_key
                    st.session_state["ev_company"], st.session_state["ev_year"], st.session_state["ev_theme"] = sig_key
        e1, e2 = st.columns([1.4, 1], vertical_alignment="bottom")
        with e1:
            company = st.selectbox("Company", companies, format_func=ui.company_name, key="ev_company")
        with e2:
            cyears = sorted(shares[shares.company == company].year.unique())
            if st.session_state.get("ev_year") not in cyears:
                st.session_state["ev_year"] = cyears[-1]
            year = st.selectbox("Year", cyears, key="ev_year")
        e3, e4 = st.columns([1.4, 1], vertical_alignment="bottom")
        with e3:
            theme_code = st.selectbox("Theme", STRATEGIC_THEMES, format_func=lambda t: THEME_NAMES[t], key="ev_theme")
        with e4:
            tier = st.selectbox("Concreteness", ["All", "Specific", "Moderate", "Vague"], key="ev_tier")
        cell = shares[(shares.company == company) & (shares.year == year) & (shares.theme == theme_code)]
        ev_all = compare.evidence(preds, company, theme_code, year)
        ev_df = ev_all[ev_all.spec_tier == tier] if tier != "All" and "spec_tier" in ev_all else ev_all
        if cell.empty or ev_df.empty:
            ui.empty_state("No matching passages.")
        else:
            r = cell.iloc[0]
            shown = f" · {len(ev_df)} {tier.lower()}" if tier != "All" else ""
            st.markdown(f'<p class="sl-meta"><span class="num">{int(r["count"])}</span> of '
                        f'<span class="num">{int(r["strategic_total"])}</span> strategy passages · '
                        f'<span class="num">{r["share"] * 100:.0f}%</span>{shown}</p>', unsafe_allow_html=True)
            for _, row in ev_df.head(3).iterrows():
                ui.quote(row)
            if len(ev_df) > 3:
                with st.expander(f"{min(len(ev_df), 13) - 3} more passages"):
                    for _, row in ev_df.iloc[3:13].iterrows():
                        ui.quote(row)
            cols_out = [c for c in ["passage_id", "company", "year", "section", "page", "pred", "confidence",
                                    "specificity", "spec_tier", "spec_signals", "text"] if c in ev_df]
            st.download_button("CSV", ev_df[cols_out].to_csv(index=False).encode("utf-8"), icon=":material/download:",
                               file_name=f"stratlens_{company}_{year}_{theme_code}.csv", mime="text/csv")

with c2:
    with ui.card("yoy", "Year-over-year", meta="95% bootstrap CI"):
        if yoy.empty:
            ui.empty_state("Needs two years for at least one company.")
        else:
            yci = uncertainty.yoy_ci(preds, yoy)
            lim = max(abs(yci[["ci_lo", "ci_hi"]]).max().max(), 1)

            def ci_svg(r, w=72, h=14):
                x = lambda v: (v + lim) / (2 * lim) * w  # noqa: E731
                col = (POSITIVE if r.pp_change >= 0 else NEGATIVE) if r.significant else MUTED
                return (f'<svg width="{w}" height="{h}" aria-hidden="true"><line x1="{w / 2}" y1="0" x2="{w / 2}" '
                        f'y2="{h}" stroke="{MUTED}" stroke-opacity=".35"/><line x1="{x(r.ci_lo):.1f}" y1="{h / 2}" '
                        f'x2="{x(r.ci_hi):.1f}" y2="{h / 2}" stroke="{col}" stroke-width="2" stroke-opacity=".5" '
                        f'stroke-linecap="round"/><circle cx="{x(r.pp_change):.1f}" cy="{h / 2}" r="3.5" fill="{col}"/></svg>')

            rows = []
            for c in sorted(yci.company.unique()):
                part = yci[yci.company == c].set_index("theme").loc[STRATEGIC_THEMES]
                rows.append(f"<tr class='grp'><td colspan='4'>{html.escape(ui.company_name(c))} · "
                            f"{part.year_from.iloc[0]} → {part.year_to.iloc[0]}</td></tr>")
                for t, r in part.iterrows():
                    val = (f"<b style='color:{POSITIVE if r.pp_change >= 0 else NEGATIVE}'>{r.pp_change:+.1f}</b>"
                           if r.significant else f"<span class='sl-small'>{r.pp_change:+.1f}</span>")
                    rows.append(f"<tr><td title='{THEME_NAMES[t]}'><span class='sl-theme'><span class='sl-dot' style='background:{THEME_COLORS[t]}'></span>{SHORT[t]}</span></td><td class='r' style='white-space:nowrap'>"
                                f"{r.share_from * 100:.0f} → {r.share_to * 100:.0f}%</td><td class='r'>{val}</td>"
                                f"<td>{ci_svg(r)}</td></tr>")
            st.markdown("<table class='sl-table'><thead><tr><th>Theme</th><th class='r'>Share</th><th class='r'>Δ pp</th>"
                        f"<th>CI</th></tr></thead><tbody>{''.join(rows)}</tbody></table>", unsafe_allow_html=True)
            st.markdown('<p class="sl-meta">Coloured = a real shift (95% interval excludes zero). Grey = within noise.</p>',
                        unsafe_allow_html=True)
