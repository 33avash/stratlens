"""Home: the three headline findings and the strategy fingerprint, each one click from its evidence."""
import streamlit as st

from stratlens import insights, ui
from stratlens.config import APP

an = ui.load_analytics()
ui.header(APP.get("hero", "StratLens"))
if an is None:
    ui.empty_state("No reports have been processed yet.", "Run <code>python -m stratlens.build</code>.")
    st.stop()

ui.pipeline_strip()

n_passages = sum(r["n_passages"] for r in an["reports"].values())
ui.meta(f"{len(an['reports'])} reports · {len(an['companies'])} companies · {n_passages:,} passages read"
        + (f" · missing: {', '.join(ui.report_label(k) for k in an['missing'])}" if an["missing"] else ""))

with st.spinner("Loading findings…"):
    cards = insights.findings(an)
if cards:
    ui.finding_cards(cards)
else:
    ui.empty_state("Not enough reports for findings yet.")

years = an["years"]
with ui.card("fingerprint", "Strategy fingerprint",
             meta="share of each report's strategy passages · click a cell for its passages"):
    opts = years + (["Change"] if an["yoy"] else [])
    mode = st.segmented_control("View", opts, default=years[-1], key="fp_mode", label_visibility="collapsed") \
        or years[-1]
    ui.fingerprint_grid(an, mode)
    if mode == "Change":
        ui.meta(f"Change from {years[-2]} to {years[-1]} in percentage points. Coloured = bigger than chance; "
                "grey = within noise. Click a cell to compare the two years.")
    else:
        ui.meta("Stronger colour = more of the strategy passages. This measures what a report talks about, not what the "
                "company spends.")
