"""Compare human review labels with AI-drafted labels and update the model card.

Reads data/labels/human_review_sample.csv (filled by a person), joins the hidden AI labels,
computes percent agreement, Cohen's kappa and per-class agreement, writes reports/agreement.json,
and rewrites the block between <!-- AGREEMENT:START --> and <!-- AGREEMENT:END --> in MODEL_CARD.md.
Usage: python scripts/compute_agreement.py
"""
import json
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from stratlens.config import AGREEMENT_JSON, LABELS_CSV, MODEL_CARD, REVIEW_CSV
from stratlens.dataset import agreement

START, END = "<!-- AGREEMENT:START -->", "<!-- AGREEMENT:END -->"


def interpret(kappa: float) -> str:
    # Landis & Koch (1977) bands
    for cut, word in [(0.2, "slight"), (0.4, "fair"), (0.6, "moderate"), (0.8, "substantial")]:
        if kappa <= cut:
            return word
    return "almost perfect"


def card_block(res: dict, n_sample: int) -> str:
    if res["n"] == 0:
        return (f"**Not yet verified.** A blind review sample of {n_sample} passages is in "
                "`data/labels/human_review_sample.csv`; agreement will appear here after it is filled in.")
    lines = [
        f"**Human-verified sample:** {res['n']} of {n_sample} review passages labelled by a person "
        f"(updated {date.today().isoformat()}).",
        f"Percent agreement with AI-drafted labels: **{res['percent_agreement']:.1%}**; "
        f"Cohen's kappa: **{res['cohen_kappa']:.2f}** ({interpret(res['cohen_kappa'])} agreement).",
        "Per-class agreement (share of AI labels the reviewer confirmed): " + ", ".join(
            f"{k} {v['agree']:.0%} (n={v['n_ai']})" for k, v in res["per_class"].items()) + ".",
    ]
    if res["n"] < n_sample:
        lines.append(f"Partial review: {n_sample - res['n']} passages still unlabelled.")
    if res.get("invalid_labels"):
        lines.append(f"Invalid reviewer labels ignored as disagreements: {res['invalid_labels']}.")
    return "\n\n".join(lines)


def main() -> None:
    if not REVIEW_CSV.exists():
        sys.exit("No review sample found. Run scripts/make_review_sample.py first.")
    review = pd.read_csv(REVIEW_CSV, dtype={"human_label": str})
    ai = pd.read_csv(LABELS_CSV)[["passage_id", "label"]]
    merged = review.merge(ai, on="passage_id", how="left")
    if merged["label"].isna().any():
        sys.exit("Some review passages are no longer in labels.csv; regenerate the review sample.")
    res = agreement(merged["human_label"], merged["label"])
    res["n_sample"] = int(len(review))
    AGREEMENT_JSON.parent.mkdir(parents=True, exist_ok=True)
    AGREEMENT_JSON.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))

    if MODEL_CARD.exists():
        text = MODEL_CARD.read_text(encoding="utf-8")
        if START in text and END in text:
            new = re.sub(re.escape(START) + r".*?" + re.escape(END),
                         lambda _: f"{START}\n{card_block(res, len(review))}\n{END}", text, flags=re.S)
            MODEL_CARD.write_text(new, encoding="utf-8")
            print(f"updated verification block in {MODEL_CARD.name}")
        else:
            print(f"warning: agreement markers not found in {MODEL_CARD.name}; card not updated")


if __name__ == "__main__":
    main()
