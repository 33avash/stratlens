"""Evaluation: held-out performance, robustness, explainability and errors (from reports/metrics.json)."""
import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stratlens import audit, theme, ui
from stratlens.config import ROOT, THEME_NAMES, THEMES
from stratlens.masking import display as show_masked
from stratlens.theme import ACCENT, MUTED, PLOTLY_CONFIG, THEME_COLORS

m = ui.load_metrics()
ui.header("Evaluation", "Held-out results for the theme classifier.", stamp=f"n = {m['dataset']['n_test']}" if m else "")
st.markdown('<p class="sl-meta" style="margin:-12px 0 16px 0">Labels were AI-drafted following the labelling '
            'guidelines and have not been human-verified.</p>', unsafe_allow_html=True)
if m is None:
    ui.empty_state("No results yet.", "Run <code>python train.py</code>.")
    st.stop()

t, b, ci = m["test"]["model"], m["test"]["baseline"], m["test"]["model"]["macro_f1_ci"]
cmp, thr, n_test = m["comparison"], m["threshold"]["value"], m["dataset"]["n_test"]
ui.kpi_cards([
    {"label": "Macro-F1", "value": f"{t['macro_f1']:.2f}", "sub": f"95% CI {ci[0]:.2f}–{ci[1]:.2f}"},
    {"label": "Accuracy", "value": f"{t['accuracy']:.0%}", "sub": f"keywords {b['accuracy']:.0%}"},
    {"label": "vs keyword baseline", "value": f"+{t['macro_f1'] - b['macro_f1']:.2f}", "sub": f"baseline {b['macro_f1']:.2f}"},
])
with ui.card("kf", "Key finding"):
    st.markdown(f'<p style="font-size:14px;line-height:1.65;margin:0">{html.escape(m["key_finding"])}</p>',
                unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Robustness", "Explainability", "Errors"])

with tab1:
    c1, c2 = st.columns([7, 5], gap="medium")
    with c1:
        with ui.card("cmp", "Models", meta="same test set"):
            rows = "".join(
                f"<tr><td>{html.escape(r['system'])} {ui.pill('main', 'good') if r['note'] == 'main model' else ''}</td>"
                f"<td class='r'>{r['accuracy']:.2f}</td><td class='r'><b>{r['macro_f1']:.2f}</b></td>"
                f"<td class='r sl-small'>{r['macro_f1_ci'][0]:.2f}–{r['macro_f1_ci'][1]:.2f}</td></tr>" for r in cmp)
            st.markdown("<table class='sl-table'><thead><tr><th>System</th><th class='r'>Acc</th><th class='r'>Macro-F1</th>"
                        f"<th class='r'>95% CI</th></tr></thead><tbody>{rows}</tbody></table>", unsafe_allow_html=True)
            d = m["model_choice"].get("embedding_minus_tfidf")
            if d:
                ui.callout(f"Embeddings vs TF-IDF: <b>{d['diff']:+.2f}</b> (CI {d['ci'][0]:+.2f} to {d['ci'][1]:+.2f}). "
                           "Tied, so the explainable model is kept.")
    with c2:
        with ui.card("setup", "Setup"):
            model, _ = ui.load_model()
            vocab = len(model.named_steps["tfidf"].get_feature_names_out()) if model is not None else 0
            spec = [("Input", "masked text"), ("Features", f"TF-IDF 1–2 grams · {vocab:,} terms"),
                    ("Classifier", "logistic regression, balanced"), ("C", f"{m['model']['best_C']} (5-fold CV)"),
                    ("Split", f"{m['dataset']['n_train']} / {m['dataset']['n_test']} stratified"),
                    ("Flag", f"confidence < {thr:.2f}")]
            st.markdown("<table class='sl-table'><tbody>" + "".join(
                f"<tr><td class='sl-small'>{k}</td><td class='r'>{html.escape(val)}</td></tr>" for k, val in spec)
                + "</tbody></table>", unsafe_allow_html=True)

    c1, c2 = st.columns([7, 5], gap="medium")
    with c1:
        with ui.card("pc", "Per theme", meta="F1"):
            pc, bpc = m["test"]["per_class"], m["test"]["baseline_per_class"]
            order = list(reversed(THEMES))
            labels = [f"{THEME_NAMES[c].split(' &')[0].split(' /')[0]} ({pc[c]['support']})" for c in order]
            f = go.Figure()
            f.add_bar(y=labels, x=[bpc[c]["f1"] for c in order], orientation="h", name="Keywords",
                      marker=dict(color="rgba(124,135,129,0.35)", cornerradius=3))
            f.add_bar(y=labels, x=[pc[c]["f1"] for c in order], orientation="h", name="Model",
                      marker=dict(color=ACCENT, cornerradius=3), text=[f"{pc[c]['f1']:.2f}" for c in order],
                      textposition="outside", textfont=dict(size=10.5, family=theme.MONO, color=MUTED),
                      cliponaxis=False)
            theme.style(f, 300, margin=dict(l=4, r=30, t=26, b=4))
            f.update_layout(barmode="group", legend=dict(traceorder="reversed"))
            f.update_xaxes(range=[0, 1.1])
            f.update_yaxes(showgrid=False)
            st.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)

    with c2:
        with ui.card("thr", "Confidence flag", meta=f"confidence < {thr:.2f} · {n_test} held-out passages"):
            tt = m["threshold"]["test"]
            ui.kpi_strip([(f"{tt['share_flagged']:.0%}", "flagged"), (f"{tt['accuracy_above']:.0%}", "acc. above"),
                          (f"{tt['accuracy_below']:.0%}", "acc. below")])
            mc = m["test"]["most_confused"]
            if mc:
                ui.callout(f"Most common error: <b>{mc['true']} → {mc['pred']}</b> ({mc['count']}/{mc['true_support']}).")


    c1, c2 = st.columns([7, 5], gap="medium")
    with c1:
        with ui.card("cm", "Confusion", meta="row = label · col = prediction"):
            cm = m["test"]["confusion"]
            labs, mat = cm["labels"], cm["matrix"]
            off = [[0 if i == j else val for j, val in enumerate(r)] for i, r in enumerate(mat)]
            zmax = max(max(r) for r in off) or 1
            cf = go.Figure(go.Heatmap(z=off, x=labs, y=labs, text=mat, colorscale=theme.sequential(), zmin=0,
                                      zmax=zmax, showscale=False, xgap=3, ygap=3,
                                      hovertemplate="%{y} → %{x}: %{text}<extra></extra>"))
            ink = theme.ink()
            for i, r in enumerate(mat):
                for j, val in enumerate(r):
                    cf.add_annotation(x=labs[j], y=labs[i], text=str(val), showarrow=False,
                                      font=dict(size=11.5, family=theme.MONO,
                                                color=ACCENT if i == j else (MUTED if val == 0 else ink)))
            theme.style(cf, 340, margin=dict(l=4, r=4, t=28, b=4), legend=False)
            cf.update_yaxes(autorange="reversed", showgrid=False)
            cf.update_xaxes(side="top", showgrid=False)
            st.plotly_chart(cf, use_container_width=True, config=PLOTLY_CONFIG)
    with c2:
        with ui.card("dir", "Where errors go", meta=f"{m['n_test_errors']} errors"):
            ed = m["test"].get("error_direction") or audit.error_direction(m["test"]["confusion"])
            ui.kpi_strip([(f"{ed['strategic_to_other']}", "theme → Other"), (f"{ed['other_to_strategic']}", "Other → theme"),
                          (f"{ed['between_themes']}", "theme → theme")])
            rows = "".join(f"<tr><td>{ui.theme_label(c)}</td><td class='r'>{v['to_other']}/{v['support']}</td></tr>"
                           for c, v in ed["per_theme"].items())
            st.markdown("<table class='sl-table'><thead><tr><th>Missed as Other</th><th class='r'>n</th></tr></thead>"
                        f"<tbody>{rows}</tbody></table>", unsafe_allow_html=True)
            st.markdown('<p class="sl-meta">Strategy passages are missed more often than invented, so theme shares '
                        'are likely conservative.</p>', unsafe_allow_html=True)

with tab2:
    c1, c2 = st.columns([7, 5], gap="medium")
    with c1:
        with ui.card("sub", "Subgroups", meta=f"95% CI · n < {m['small_n']} flagged"):
            sg = pd.DataFrame(m["subgroups"])

            def ci_svg(lo, hi, pt, w=150, h=14):
                return (f'<svg width="{w}" height="{h}" aria-hidden="true"><line x1="0" y1="{h / 2}" x2="{w}" y2="{h / 2}" '
                        f'stroke="{MUTED}" stroke-opacity=".2"/><line x1="{lo * w:.1f}" y1="{h / 2}" x2="{hi * w:.1f}" '
                        f'y2="{h / 2}" stroke="{ACCENT}" stroke-width="2" stroke-opacity=".5" stroke-linecap="round"/>'
                        f'<circle cx="{pt * w:.1f}" cy="{h / 2}" r="3.5" fill="{ACCENT}"/></svg>')

            rows = []
            for dim, part in sg.groupby("dimension", sort=False):
                rows.append(f"<tr class='grp'><td colspan='5'>{html.escape(dim.split(' (')[0])}</td></tr>")
                for r in part.itertuples():
                    c_ = r.macro_f1_ci if isinstance(r.macro_f1_ci, list) else [r.macro_f1, r.macro_f1]
                    rows.append(f"<tr><td>{html.escape(ui.company_name(r.group))} "
                                f"{ui.pill('small n', 'warn') if r.too_small else ''}</td><td class='r'>{r.n}</td>"
                                f"<td class='r'><b>{r.macro_f1:.2f}</b></td><td class='r sl-small'>{c_[0]:.2f}–{c_[1]:.2f}</td>"
                                f"<td>{ci_svg(c_[0], c_[1], r.macro_f1)}</td></tr>")
            st.markdown("<table class='sl-table'><thead><tr><th>Group</th><th class='r'>n</th><th class='r'>Macro-F1</th>"
                        f"<th class='r'>95% CI</th><th>0 — 1</th></tr></thead><tbody>{''.join(rows)}</tbody></table>",
                        unsafe_allow_html=True)
    with c2:
        with ui.card("cv", "Regularisation", meta="CV macro-F1 ± 1 sd"):
            cv = pd.DataFrame(m["model"]["cv_results"])
            f = go.Figure()
            f.add_scatter(x=cv.C, y=cv.cv_macro_f1 + cv.cv_std, mode="lines", line=dict(width=0), hoverinfo="skip")
            f.add_scatter(x=cv.C, y=cv.cv_macro_f1 - cv.cv_std, mode="lines", line=dict(width=0), fill="tonexty",
                          fillcolor="rgba(30,158,85,0.14)", hoverinfo="skip")
            f.add_scatter(x=cv.C, y=cv.cv_macro_f1, mode="lines+markers", line=dict(color=ACCENT, width=2),
                          marker=dict(size=7), hovertemplate="C=%{x}<br>%{y:.3f}<extra></extra>")
            theme.style(f, 280, legend=False)
            f.update_xaxes(type="log", title_text="C")
            st.plotly_chart(f, use_container_width=True, config=PLOTLY_CONFIG)
    c1, c2, c3 = st.columns(3, gap="medium")
    mk, p, cy = m["masking"], m["masking"]["company_probe"], m["cross_year"]
    with c1:
        with ui.card("mask", "Masking", meta="names, brands, numbers"):
            ui.kpi_strip([(f"{mk['masked_macro_f1']:.2f}", "masked"), (f"{mk['unmasked_macro_f1']:.2f}", "raw")])
            st.markdown(f'<p class="sl-meta">Δ {mk["diff"]:+.2f}, CI {mk["diff_ci"][0]:+.2f} to {mk["diff_ci"][1]:+.2f}</p>',
                        unsafe_allow_html=True)
    with c2:
        with ui.card("probe", "Author probe", meta="guess the company"):
            ui.kpi_strip([(f"{p['masked']:.0%}", "masked"), (f"{p['majority_baseline']:.0%}", "chance")])
            st.markdown(f'<p class="sl-meta">{p["unmasked"]:.0%} on raw text: style, not names, gives it away.</p>',
                        unsafe_allow_html=True)
    with c3:
        with ui.card("cy", "Cross-year", meta=f"{cy.get('train_year', '')} → {cy.get('test_year', '')}"):
            if cy.get("available"):
                ui.kpi_strip([(f"{cy['macro_f1']:.2f}", "macro-F1"), (f"{cy['n']}", "test n")])
                st.markdown('<p class="sl-meta">Includes a company unseen in training.</p>', unsafe_allow_html=True)
            else:
                ui.empty_state("Not available.")

with tab3:
    with ui.card("gt", "Top terms per theme", meta="coefficients · association, not causation"):
        gt = m["global_terms"]
        cols = st.columns(2, gap="large")
        for i, c in enumerate(THEMES):
            chips = "".join(f"<span>{html.escape(show_masked(x['term']))}<em>{x['weight']:.1f}</em></span>"
                            for x in gt[c][:8])
            with cols[i % 2]:
                st.markdown(f"<div style='margin-bottom:10px'>{ui.theme_label(c)}<div class='sl-terms'>{chips}</div>"
                            "</div>", unsafe_allow_html=True)
    with ui.card("why", "Why this label?"):
        st.markdown('<p class="sl-callout">Trace any passage through every step (cleaning, masking, TF-IDF '
                    'features, class probabilities, confidence flag and specificity signals) in the pipeline '
                    'inspector.</p>', unsafe_allow_html=True)
        st.page_link("views/how_it_works.py", label="Open the pipeline inspector", icon=":material/arrow_forward:")

with tab4:
    c1, c2 = st.columns([7, 5], gap="medium")
    with c1:
        with ui.card("err", "Misclassified", meta=f"{len(m['misclassified_examples'])} of {m['n_test_errors']} errors"):
            for e in m["misclassified_examples"]:
                reason = (f"<div class='m' style='font-family:inherit;font-size:12.5px'>{html.escape(e['likely_reason'])}"
                          "</div>" if e.get("likely_reason") else "")
                st.markdown(f'<div class="sl-quote" style="border-left-color:{THEME_COLORS[e["pred"]]}"><div class="q">'
                            f'{html.escape(e["text"])}</div><div class="m">{e["true"]} → {e["pred"]} · '
                            f'confidence {e["confidence"]:.2f} · {ui.company_name(e["company"])} {e["year"]}</div>{reason}</div>',
                            unsafe_allow_html=True)
    with c2:
        with ui.card("lim", "Limitations"):
            sup = [m["test"]["per_class"][c]["support"] for c in THEMES]
            spot_csv = ROOT / "docs" / "specificity_spotcheck.csv"
            spot = pd.read_csv(spot_csv)["agree"] if spot_csv.exists() else pd.Series(dtype=bool)
            spot_txt = f"{int(spot.sum())}/{len(spot)} spot-check tiers agreed" if len(spot) else "not spot-checked"
            lims = [("Labels", "AI-drafted following the labelling guidelines; not human-verified"),
                    ("Test size", f"{min(sup)}–{max(sup)} per theme"),
                    ("Model", "bag of words; no negation or tense"),
                    ("Coverage", f"{len(m['dataset']['by_company_year'])} reports, 3 companies"),
                    ("Specificity", f"rule-based; {spot_txt}; favours brand-by-brand text"),
                    ("House style", f"company {m['masking']['company_probe']['masked']:.0%} predictable after masking")]
            st.markdown("<table class='sl-table'><tbody>" + "".join(
                f"<tr><td class='sl-small'>{k}</td><td>{html.escape(val)}</td></tr>" for k, val in lims)
                + "</tbody></table>", unsafe_allow_html=True)
