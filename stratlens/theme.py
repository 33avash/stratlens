"""Single source of truth for colours, typography and the shared Plotly template."""
import plotly.graph_objects as go
import plotly.io as pio

BG = "#FAF8F4"
SURFACE = "#FFFFFF"
BORDER = "#E6E1D8"
TEXT = "#2B2B2B"
MUTED = "#6B6660"
ACCENT = "#1F3A5F"
POSITIVE = "#1F3A5F"   # YoY increase (navy)
NEGATIVE = "#A65A44"   # YoY decrease (terracotta)
GRID = "#EEEAE3"

SERIF = "'Source Serif 4', Georgia, serif"
SANS = "Inter, 'Segoe UI', system-ui, sans-serif"

# Theme hues: the brief's palette, with SUST and BRAND re-stepped in lightness because the original
# olive/terracotta pair was indistinguishable under deuteranopia (validator dE 0.6 -> 4.6). Theme colour is
# never the only cue: every theme mark is accompanied by its text label.
THEME_COLORS = {
    "PROD": "#1F3A5F",
    "DIST": "#3E7C78",
    "DIGI": "#5B6FA8",
    "OPS": "#B8893B",
    "SUST": "#5E7A28",
    "BRAND": "#C0604A",
    "OTHER": "#A39E96",
}
# Company hues: validated categorical set (passes lightness, chroma, CVD and normal-vision checks on white).
COMPANY_COLORS = {"HUL": "#24578A", "DABUR": "#C27C2E", "NESTLE": "#7D4E9E"}
FALLBACK_COMPANY_COLORS = ["#4F6D7A", "#8C6D46", "#5E5A80"]

# Sequential single-hue scale for heatmaps (light -> navy).
SEQUENTIAL = [[0.0, "#F4F1EA"], [0.25, "#D5DCE6"], [0.55, "#8EA3BF"], [0.8, "#47648C"], [1.0, "#1F3A5F"]]


def company_color(code: str, i: int = 0) -> str:
    return COMPANY_COLORS.get(code, FALLBACK_COMPANY_COLORS[i % len(FALLBACK_COMPANY_COLORS)])


def tint(hex_color: str, alpha: float = 0.15) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _luminance(hex_color: str) -> float:
    def ch(c):
        c /= 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def contrast(a: str, b: str) -> float:
    la, lb = sorted([_luminance(a), _luminance(b)], reverse=True)
    return (la + 0.05) / (lb + 0.05)


TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(family=SANS, size=12, color=TEXT),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        margin=dict(l=8, r=16, t=8, b=8),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER, font=dict(family=SANS, size=12, color=TEXT)),
        xaxis=dict(showgrid=False, zeroline=False, linecolor=BORDER, ticks="", tickfont=dict(color=MUTED, size=11),
                   title=dict(font=dict(color=MUTED, size=11))),
        yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=BORDER, ticks="",
                   tickfont=dict(color=MUTED, size=11), title=dict(font=dict(color=MUTED, size=11))),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, font=dict(size=11, color=MUTED),
                    bgcolor="rgba(0,0,0,0)"),
        bargap=0.28,
        bargroupgap=0.08,
    )
)
pio.templates["stratlens"] = TEMPLATE
pio.templates.default = "simple_white+stratlens"

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}
