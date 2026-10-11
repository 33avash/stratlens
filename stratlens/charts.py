"""Plotly figure builders. Every theme is drawn in its config colour and labelled with its full name."""
from __future__ import annotations

import plotly.graph_objects as go

from . import theme
from .config import STRATEGIC_THEMES, THEME_COLORS, THEME_NAMES, THEMES
from .theme import ACCENT, MUTED

PEER_GREY = "#8E9893"
ALL_GREY = "rgba(142,152,147,0.35)"


def _wrap(name: str, width: int = 18) -> str:
    words, lines, cur = name.split(), [], ""
    for w in words:
        if cur and len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return "<br>".join(lines + [cur])


def benchmark(rows: list[dict], company_label: str, peer_label: str | None) -> go.Figure:
    """rows: code, share, peer, all (0-1). Horizontal grouped bars: company / peers / all six."""
    order = list(reversed([r["code"] for r in rows]))
    by = {r["code"]: r for r in rows}
    y = [_wrap(THEME_NAMES[t]) for t in order]
    f = go.Figure()
    f.add_bar(y=y, x=[by[t]["all"] * 100 if by[t]["all"] is not None else None for t in order], orientation="h",
              name="All companies", marker=dict(color=ALL_GREY, cornerradius=3),
              hovertemplate="All companies: %{x:.0f}%<extra></extra>")
    if peer_label:
        f.add_bar(y=y, x=[by[t]["peer"] * 100 if by[t]["peer"] is not None else None for t in order],
                  orientation="h", name=peer_label, marker=dict(color=PEER_GREY, cornerradius=3),
                  hovertemplate=f"{peer_label}: %{{x:.0f}}%<extra></extra>")
    f.add_bar(y=[None], x=[None], orientation="h", name=f"{company_label} (theme colour)",
              marker=dict(color=theme.ink()), hoverinfo="skip")
    f.add_bar(y=y, x=[by[t]["share"] * 100 for t in order], orientation="h", name=company_label, showlegend=False,
              marker=dict(color=[THEME_COLORS[t] for t in order], cornerradius=3),
              text=[f"{by[t]['share'] * 100:.0f}%" for t in order], textposition="outside", cliponaxis=False,
              textfont=dict(family=theme.MONO, size=11), customdata=order,
              hovertemplate=f"{company_label}: %{{x:.0f}}%<extra></extra>")
    theme.style(f, 70 + 62 * len(order), margin=dict(l=4, r=36, t=30, b=4))
    f.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08, legend=dict(traceorder="reversed"))
    f.update_xaxes(ticksuffix="%", rangemode="tozero")
    f.update_yaxes(showgrid=False, tickfont=dict(size=12), type="category", categoryorder="array", categoryarray=y)
    return f


def diverging(diffs: dict, label_a: str, label_b: str) -> go.Figure:
    """B minus A in points per theme with the 95% range; grey when the range includes zero (within noise)."""
    order = list(reversed(STRATEGIC_THEMES))
    y = [_wrap(THEME_NAMES[t], 22) for t in order]
    d = [diffs[t] for t in order]
    colors = [THEME_COLORS[t] if x["significant"] else "rgba(142,152,147,0.45)" for t, x in zip(order, d)]
    f = go.Figure(go.Bar(
        y=y, x=[x["pp"] for x in d], orientation="h", marker=dict(color=colors, cornerradius=3),
        error_x=dict(type="data", symmetric=False, array=[x["ci"][1] - x["pp"] for x in d],
                     arrayminus=[x["pp"] - x["ci"][0] for x in d], color=MUTED, thickness=1.2, width=4),
        customdata=[[t, x["ci"][0], x["ci"][1], "bigger than chance" if x["significant"] else "within noise"]
                    for t, x in zip(order, d)],
        hovertemplate="%{x:+.1f} points (likely range %{customdata[1]:+.1f} to %{customdata[2]:+.1f}): "
                      "%{customdata[3]}<extra></extra>"))
    lim = max(max(abs(x["ci"][0]), abs(x["ci"][1])) for x in d) * 1.1 + 1
    theme.style(f, 70 + 58 * len(order), margin=dict(l=4, r=12, t=30, b=28), legend=False)
    f.update_xaxes(range=[-lim, lim], ticksuffix="", zeroline=True, zerolinecolor=MUTED, zerolinewidth=1,
                   title_text=f"← more in {label_a}      points      more in {label_b} →",
                   title_font=dict(size=11.5, color=MUTED))
    f.update_yaxes(showgrid=False, tickfont=dict(size=12), type="category", categoryorder="array", categoryarray=y)
    return f


def paired(values_a: dict, values_b: dict, label_a: str, label_b: str, suffix: str = "",
           rng: tuple | None = None) -> go.Figure:
    """Two bars per theme (e.g. concreteness 0-5). Missing values are left blank, never filled."""
    order = list(reversed(STRATEGIC_THEMES))
    y = [_wrap(THEME_NAMES[t], 22) for t in order]
    f = go.Figure()
    f.add_bar(y=[None], x=[None], orientation="h", name=f"{label_b} (darker)", marker=dict(color=theme.ink()),
              hoverinfo="skip")
    f.add_bar(y=[None], x=[None], orientation="h", name=f"{label_a} (lighter)",
              marker=dict(color=theme.tint(theme.ink(), 0.35)), hoverinfo="skip")
    f.add_bar(y=y, x=[values_b.get(t) for t in order], orientation="h", name=label_b, showlegend=False,
              marker=dict(color=[THEME_COLORS[t] for t in order], cornerradius=3),
              hovertemplate=f"{label_b}: %{{x:.1f}}{suffix}<extra></extra>")
    f.add_bar(y=y, x=[values_a.get(t) for t in order], orientation="h", name=label_a, showlegend=False,
              marker=dict(color=[theme.tint(THEME_COLORS[t], 0.35) for t in order], cornerradius=3),
              hovertemplate=f"{label_a}: %{{x:.1f}}{suffix}<extra></extra>")
    theme.style(f, 70 + 56 * len(order), margin=dict(l=4, r=12, t=30, b=4))
    f.update_layout(barmode="group", bargap=0.3, legend=dict(traceorder="reversed"))
    if rng:
        f.update_xaxes(range=list(rng))
    f.update_yaxes(showgrid=False, tickfont=dict(size=12), type="category", categoryorder="array", categoryarray=y)
    return f


def per_theme_f1(model: dict, baseline: dict) -> go.Figure:
    order = list(reversed(THEMES))
    y = [f"{_wrap(THEME_NAMES[c], 24)} ({model[c]['support']})" for c in order]
    f = go.Figure()
    f.add_bar(y=y, x=[baseline[c]["f1"] for c in order], orientation="h", name="Keyword matching",
              marker=dict(color="rgba(124,135,129,0.35)", cornerradius=3))
    f.add_bar(y=y, x=[model[c]["f1"] for c in order], orientation="h", name="Trained model",
              marker=dict(color=ACCENT, cornerradius=3), text=[f"{model[c]['f1']:.2f}" for c in order],
              textposition="outside", textfont=dict(size=10.5, family=theme.MONO, color=MUTED), cliponaxis=False)
    theme.style(f, 380, margin=dict(l=4, r=30, t=26, b=4))
    f.update_layout(barmode="group", legend=dict(traceorder="reversed"))
    f.update_xaxes(range=[0, 1.1])
    f.update_yaxes(showgrid=False)
    return f


def confusion(cm: dict, height: int = 340) -> go.Figure:
    labs = [_wrap(THEME_NAMES[c], 14) for c in cm["labels"]]
    mat = cm["matrix"]
    off = [[0 if i == j else v for j, v in enumerate(r)] for i, r in enumerate(mat)]
    zmax = max(max(r) for r in off) or 1
    f = go.Figure(go.Heatmap(z=off, x=labs, y=labs, text=mat, colorscale=theme.sequential(), zmin=0, zmax=zmax,
                             showscale=False, xgap=3, ygap=3, hovertemplate="%{y} → %{x}: %{text}<extra></extra>"))
    ink = theme.ink()
    for i, r in enumerate(mat):
        for j, v in enumerate(r):
            f.add_annotation(x=labs[j], y=labs[i], text=str(v), showarrow=False,
                             font=dict(size=11.5, family=theme.MONO,
                                       color=ACCENT if i == j else (MUTED if v == 0 else ink)))
    theme.style(f, height, margin=dict(l=4, r=4, t=28, b=4), legend=False)
    f.update_yaxes(autorange="reversed", showgrid=False, title_text="correct theme", title_font=dict(size=11))
    f.update_xaxes(side="top", showgrid=False)
    return f
