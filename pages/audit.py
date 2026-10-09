"""Model audit: held-out performance, baseline comparison, subgroups, explainability and errors (from metrics.json)."""
import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stratlens import ui
from stratlens.config import TEST_CSV, THEME_NAMES, THEMES
from stratlens.theme import ACCENT, MUTED, PLOTLY_CONFIG, SEQUENTIAL, TEXT, THEME_COLORS

ui.page_header("Model audit",
               "How well the classifier works on passages it never saw in training, where it is weaker, and what "
               "drives its decisions. All figures are read from reports/metrics.json.")

m = ui.load_metrics()
if m is None:
    ui.empty_state("No audit results yet.", "Run <code>python train.py</code> to train and evaluate the model.")
    st.stop()

t, b = m["test"]["model"], m["test"]["baseline"]
ui.kpi_strip([
    (f"{t['macro_f1']:.2f}", f"Macro-F1 on {t['n']} held-out passages"),
    (f"{b['macro_f1']:.2f}", "Keyword baseline macro-F1"),
    (f"{t['accuracy']:.0%}", "Accuracy"),
    (f"{m['threshold']['test']['share_flagged']:.0%}", f"Flagged for review (conf. < {m['threshold']['value']:.2f})"),
])
ui.note(f"<b>Key audit finding.</b> {html.escape(m['key_finding'])}")
ui.note("<b>Label provenance.</b> Training and test labels were drafted by an AI assistant (Claude) during the build, "
        "following docs/labelling_guidelines.md, and are pending human verification. Scores measure agreement with "
        "those drafted labels, not with an expert gold standard.", warn=True)

# ---------- overall ----------
ui.section("Model versus baseline",
           "Held-out test set. The keyword baseline assigns the theme whose keyword list matches most often; it is "
           "never used for app predictions.")
ov = pd.DataFrame([
    {"System": "TF-IDF + logistic regression", **{k: t[k] for k in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]}},
    {"System": "Keyword baseline", **{k: b[k] for k in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]}},
])
st.dataframe(ov, hide_index=True, use_container_width=False, column_config={
    "accuracy": st.column_config.NumberColumn("Accuracy", format="%.2f"),
    "macro_precision": st.column_config.NumberColumn("Macro precision", format="%.2f"),
    "macro_recall": st.column_config.NumberColumn("Macro recall", format="%.2f"),
    "macro_f1": st.column_config.NumberColumn("Macro-F1", format="%.2f")})

pc, bpc = m["test"]["per_class"], m["test"]["baseline_per_class"]
order = list(reversed(THEMES))
fig = go.Figure()
fig.add_bar(y=[THEME_NAMES[c] for c in order], x=[bpc[c]["f1"] for c in order], orientation="h",
            name="Keyword baseline", marker_color="#C9C3B8",
            text=[f"{bpc[c]['f1']:.2f}" for c in order], textposition="outside", textfont=dict(size=11, color=MUTED),
            hovertemplate="%{y}<br>baseline F1 %{x:.2f}<extra></extra>")
fig.add_bar(y=[THEME_NAMES[c] for c in order], x=[pc[c]["f1"] for c in order], orientation="h", name="Model",
            marker_color=ACCENT, text=[f"{pc[c]['f1']:.2f}" for c in order], textposition="outside",
            textfont=dict(size=11, color=MUTED),
            customdata=[[pc[c]["precision"], pc[c]["recall"], pc[c]["support"]] for c in order],
            hovertemplate="%{y}<br>F1 %{x:.2f} · precision %{customdata[0]:.2f} · recall %{customdata[1]:.2f}"
                          "<br>n = %{customdata[2]}<extra></extra>")
fig.update_layout(barmode="group", height=470, legend=dict(traceorder="reversed"))
fig.update_xaxes(range=[0, 1.12], showgrid=True, gridcolor="#EEEAE3", title="F1 score")
fig.update_yaxes(showgrid=False, tickfont=dict(color=TEXT, size=12))
c1, c2 = st.columns([1.15, 1])
with c1:
    st.markdown('<p class="sl-cap">Per-class F1, model versus baseline</p>', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
with c2:
    st.markdown('<p class="sl-cap">Per-class report (model)</p>', unsafe_allow_html=True)
    thin = set(m["dataset"]["thin_classes"])
    rows = "".join(
        f"<tr><td>{ui.theme_label(c)}</td><td class='r'>{pc[c]['precision']:.2f}</td><td class='r'>{pc[c]['recall']:.2f}"
        f"</td><td class='r'>{pc[c]['f1']:.2f}</td><td class='r'>{pc[c]['support']}"
        f"{' <span class=sl-flag>small</span>' if pc[c]['support'] < 15 else ''}</td></tr>" for c in THEMES)
    st.markdown("<table class='sl-table'><thead><tr><th>Theme</th><th class='r'>Precision</th><th class='r'>Recall</th>"
                f"<th class='r'>F1</th><th class='r'>Test n</th></tr></thead><tbody>{rows}</tbody></table>",
                unsafe_allow_html=True)
    st.markdown(f'<p class="sl-meta">Test n per class is small (9–79), so per-class scores carry wide uncertainty. '
                f'Training set: {m["dataset"]["n_train"]} passages; best C = {m["model"]["best_C"]} '
                f'(5-fold CV macro-F1 {m["model"]["cv_best_macro_f1"]:.2f}).'
                + (f' Thin classes (&lt;40 labels): {", ".join(sorted(thin))}.' if thin else "") + "</p>",
                unsafe_allow_html=True)

# ---------- confusion ----------
ui.section("Confusion matrix", "Rows are the drafted label, columns the model's prediction. Off-diagonal cells are "
                               "errors; the diagonal is shaded lightly so the errors stand out.")
cm = m["test"]["confusion"]
labs = cm["labels"]
mat = cm["matrix"]
offdiag = [[0 if i == j else v for j, v in enumerate(r)] for i, r in enumerate(mat)]
zmax = max(max(r) for r in offdiag) or 1
cfig = go.Figure(go.Heatmap(
    z=offdiag, x=labs, y=labs, text=mat, colorscale=SEQUENTIAL, zmin=0, zmax=zmax,
    showscale=False, xgap=2, ygap=2, hovertemplate="label %{y} → predicted %{x}: %{text}<extra></extra>"))
for i, r in enumerate(mat):
    for j, v in enumerate(r):
        cfig.add_annotation(x=labs[j], y=labs[i], text=str(v), showarrow=False,
                            font=dict(size=12, color="#FFFFFF" if (i != j and v > zmax * 0.6) else
                                      (ACCENT if i == j else TEXT), weight=600 if i == j else 400))
cfig.update_yaxes(autorange="reversed", title="Drafted label", showgrid=False, tickfont=dict(color=TEXT))
cfig.update_xaxes(side="top", title="Predicted", tickfont=dict(color=TEXT))
cfig.update_layout(height=400, width=520, margin=dict(l=8, r=8, t=56, b=8))
c1, c2 = st.columns([1, 1])
with c1:
    st.plotly_chart(cfig, use_container_width=True, config=PLOTLY_CONFIG)
with c2:
    mc = m["test"]["most_confused"]
    th = m["threshold"]
    if mc:
        st.markdown(f"**Most frequent error.** {THEME_NAMES[mc['true']]} passages predicted as "
                    f"{THEME_NAMES[mc['pred']].lower()}: {mc['count']} of {mc['true_support']}.")
    st.markdown(f"**Review threshold.** Passages with top-class probability below {th['value']:.2f} are flagged. "
                f"On the test set this flags {th['test']['share_flagged']:.0%} of passages; accuracy is "
                f"{th['test']['accuracy_above']:.0%} above the threshold and {th['test']['accuracy_below']:.0%} below it. "
                f"Rule: {th['rule']} (cross-validated share flagged at the default: "
                f"{th['cv_share_flagged_at_default']:.0%}).")
    cy = m["cross_year"]
    if cy.get("available"):
        st.markdown(f"**Cross-year check.** Trained only on {cy['train_year']} ({cy['n_train']} passages; "
                    f"{', '.join(cy['companies_train'])}) and tested on all {cy['test_year']} labels "
                    f"(n = {cy['n']}): macro-F1 {cy['macro_f1']:.2f}, accuracy {cy['accuracy']:.0%}. "
                    "The test year includes companies absent from the training year, so this is a stricter test "
                    "than the random split.")
    else:
        st.markdown(f"**Cross-year check.** Not available: {cy.get('reason')}.")

# ---------- subgroups ----------
ui.section("Performance by text subgroup",
           f"Held-out test set split by passage length, writing style (section) and company. Groups with fewer than "
           f"{m['small_n']} passages are flagged as too small to conclude.")
sg = pd.DataFrame(m["subgroups"])
rows = []
for dim, part in sg.groupby("dimension", sort=False):
    for r in part.itertuples():
        flag = '<span class="sl-flag">too small to conclude</span>' if r.too_small else ""
        rows.append(f"<tr><td class='sl-small'>{dim}</td><td>{html.escape(r.group)}</td><td class='r'>{r.n}</td>"
                    f"<td class='r'>{r.macro_f1:.2f}</td><td class='r'>{r.macro_precision:.2f}</td>"
                    f"<td class='r'>{r.macro_recall:.2f}</td><td class='r'>{r.accuracy:.2f}</td><td>{flag}</td></tr>")
st.markdown("<table class='sl-table'><thead><tr><th>Dimension</th><th>Group</th><th class='r'>n</th>"
            "<th class='r'>Macro-F1</th><th class='r'>Macro precision</th><th class='r'>Macro recall</th>"
            f"<th class='r'>Accuracy</th><th></th></tr></thead><tbody>{''.join(rows)}</tbody></table>",
            unsafe_allow_html=True)
st.markdown('<p class="sl-meta">Macro scores average over the themes present in each group, so a group where a theme '
            'has only one or two examples can swing sharply. Accuracy and macro-F1 can disagree for that reason.</p>',
            unsafe_allow_html=True)

# ---------- global explainability ----------
ui.section("What the model associates with each theme",
           "Terms with the largest positive coefficients per theme (global explainability). Weights show "
           "association with a label in the training data, not causation.")
gt = m["global_terms"]
cols = st.columns(2)
for i, c in enumerate(THEMES):
    terms = gt[c][:12]
    chips = "".join(f"<span>{html.escape(x['term'])}<em>{x['weight']:.1f}</em></span>" for x in terms)
    with cols[i % 2]:
        st.markdown(f"<div style='margin-bottom:14px'>{ui.theme_label(c)}<div class='sl-terms'>{chips}</div></div>",
                    unsafe_allow_html=True)

# ---------- local explainability ----------
ui.section("Why this label?", "Pick a held-out test passage, or paste your own text, to see the class probabilities "
                              "and the terms that pushed the decision.")
mode = st.segmented_control("Input", ["Test passage", "Own text"], default="Test passage", label_visibility="collapsed")
if mode == "Own text":
    txt = st.text_area("Passage text", height=110,
                       placeholder="e.g. We expanded our direct reach to 1.5 million outlets, adding 4,000 villages.")
    if txt.strip():
        if len(txt.split()) < 4:
            ui.empty_state("That text is very short.", "Paste at least a sentence for a meaningful explanation.")
        else:
            ui.why_label(txt.strip())
    else:
        ui.empty_state("No text entered.", "Paste a passage above.")
elif TEST_CSV.exists():
    test = pd.read_csv(TEST_CSV)
    pick = st.selectbox("Test passage", test.index, format_func=lambda i: f"[{test.loc[i, 'label']}] "
                        f"{test.loc[i, 'company']} {test.loc[i, 'year']} p.{test.loc[i, 'page']} — "
                        f"{test.loc[i, 'text'][:95]}…")
    st.markdown(f'<p class="sl-meta">Drafted label: {ui.theme_label(test.loc[pick, "label"])}</p>',
                unsafe_allow_html=True)
    ui.why_label(test.loc[pick, "text"])
else:
    ui.empty_state("The test split file is missing.", "Run <code>python train.py</code>.")

# ---------- errors ----------
ui.section("Misclassified examples",
           f"Real held-out errors ({m['n_test_errors']} in total), most confident first. The likely reasons are the "
           "author's reading of each case, not model output.")
for e in m["misclassified_examples"]:
    reason = (f"<br><span style='color:#4A4640'><b>Likely reason:</b> {html.escape(e['likely_reason'])}</span>"
              if e.get("likely_reason") else "")
    st.markdown(
        f'<div class="sl-quote" style="border-left-color:{THEME_COLORS[e["pred"]]}"><div class="q">'
        f'{html.escape(e["text"])}</div><div class="m">Drafted label {ui.theme_label(e["true"])} · predicted '
        f'{ui.theme_label(e["pred"])} · confidence {e["confidence"]:.2f} · {ui.company_name(e["company"])} '
        f'{e["year"]} · {ui.section_name(e["section"])} · p. {e["page"]}{reason}</div></div>', unsafe_allow_html=True)
