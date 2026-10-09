"""Write the blind human-review sample: data/labels/human_review_sample.csv.

150 labelled passages stratified by AI-drafted label x company, shuffled, WITHOUT the AI label,
with an empty `human_label` column. Fill it with one of PROD, DIST, DIGI, OPS, SUST, BRAND, OTHER
following docs/labelling_guidelines.md, then run scripts/compute_agreement.py.
Will not overwrite an existing file that already has human labels (pass --force to override).
Usage: python scripts/make_review_sample.py [--force]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from stratlens.config import LABELS_CSV, REVIEW_CSV
from stratlens.dataset import review_sample


def main(force: bool) -> None:
    if REVIEW_CSV.exists() and not force:
        existing = pd.read_csv(REVIEW_CSV)
        if existing["human_label"].fillna("").astype(str).str.strip().ne("").any():
            sys.exit(f"{REVIEW_CSV} already contains human labels; refusing to overwrite (use --force).")
    labels = pd.read_csv(LABELS_CSV)
    sample = review_sample(labels)
    sample.to_csv(REVIEW_CSV, index=False, encoding="utf-8-sig")  # BOM so Excel opens it as UTF-8
    print(f"review sample: {len(sample)} passages -> {REVIEW_CSV}")
    ai = labels.set_index("passage_id").loc[sample.passage_id, "label"]
    print("AI-label mix (hidden from the file):", ai.value_counts().to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args().force)
