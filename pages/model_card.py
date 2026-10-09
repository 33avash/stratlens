"""Model card: renders MODEL_CARD.md with a download option."""
import streamlit as st

from stratlens import ui
from stratlens.config import MODEL_CARD

if not MODEL_CARD.exists():
    ui.page_header("Model card", "Purpose, data, results and limitations of the StratLens classifier.")
    ui.empty_state("MODEL_CARD.md was not found.", "Restore it from the project folder.")
    st.stop()

text = MODEL_CARD.read_text(encoding="utf-8")
c1, c2 = st.columns([3.6, 1.4])
with c2:
    st.download_button("Download model card", text.encode("utf-8"), file_name="StratLens_MODEL_CARD.md",
                       mime="text/markdown")
st.markdown(text.replace("<!-- AGREEMENT:START -->", "").replace("<!-- AGREEMENT:END -->", ""))
