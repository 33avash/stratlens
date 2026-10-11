"""Design tokens shared by every page and chart. Mark colours are chosen to read on both light and dark."""
import streamlit as st

# ---- marks (mode-neutral)
ACCENT = "#1E9E55"
POSITIVE = "#1E9E55"
NEGATIVE = "#D2553B"
NOISE_POS = "rgba(30,158,85,0.28)"
NOISE_NEG = "rgba(210,85,59,0.28)"
WARN = "#C88A1E"
MUTED = "#7C8781"   # mid-grey: readable on white and on near-black
FAINT = "#7C8781"
GRID = "rgba(124,135,129,0.18)"
SANS = "Geist, Inter, 'Segoe UI', system-ui, sans-serif"
MONO = "'Geist Mono', ui-monospace, SFMono-Regular, Menlo, monospace"

# Theme hues (always shown next to their name) and company neutrals come from config/*.yaml.
# Companies never reuse a theme hue: colour always means theme. Companies get low-chroma neutrals,
# shown together with their short name, so they read on white and on near-black.
from .config import COMPANY_COLORS, COMPANY_SHORT, THEME_COLORS  # noqa: E402
FALLBACK_COMPANY_COLORS = ["#C9C4BA", "#8F8A82", "#B4B9BD"]

SEQ_LIGHT = [[0.0, "#F1F4F1"], [0.3, "#CDE7D7"], [0.6, "#76C495"], [0.85, "#2C9D5B"], [1.0, "#146B3A"]]
SEQ_DARK = [[0.0, "#18201C"], [0.3, "#1D3A2A"], [0.6, "#24683F"], [0.85, "#2F9A5A"], [1.0, "#5BD08A"]]

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


def is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def sequential():
    return SEQ_DARK if is_dark() else SEQ_LIGHT


def ink() -> str:
    """Primary text colour for the current mode (for chart annotations that need explicit colour)."""
    return "#E4E9E5" if is_dark() else "#141A17"


def company_color(code: str, i: int = 0) -> str:
    return COMPANY_COLORS.get(code, FALLBACK_COMPANY_COLORS[i % len(FALLBACK_COMPANY_COLORS)])


def company_initials(code: str) -> str:
    return COMPANY_SHORT.get(code, code[:3].upper())


def tint(hex_color: str, alpha: float = 0.15) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def style(fig, height: int, margin=None, legend: bool = True):
    """Shared chart styling. Background is transparent so the card shows through in either mode;
    tick and legend colours come from Streamlit's adaptive chart theme."""
    fig.update_layout(
        height=height, margin=margin or dict(l=4, r=12, t=8, b=4),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=SANS, size=12),
        hoverlabel=dict(font=dict(family=SANS, size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11),
                    title_text="") if legend else None,
        showlegend=legend,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, tickfont=dict(family=MONO, size=10.5))
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig
