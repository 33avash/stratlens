"""Model card: renders the generated MODEL_CARD.md with a download option."""
import streamlit as st

from stratlens import ui
from stratlens.config import MODEL_CARD

h1, h2 = st.columns([4, 1], vertical_alignment="bottom")
with h1:
    ui.header("Model card")
if not MODEL_CARD.exists():
    ui.empty_state("MODEL_CARD.md not found.", "Run <code>python train.py</code>.")
    st.stop()
text = MODEL_CARD.read_text(encoding="utf-8")
with h2:
    st.download_button("Download", text.encode("utf-8"), file_name="StratLens_MODEL_CARD.md", mime="text/markdown",
                       icon=":material/download:", use_container_width=True)
body = "\n".join(line for line in text.splitlines() if not line.startswith("# ") and not line.startswith("*Generated"))
with ui.card("mc"):
    st.markdown(body)
