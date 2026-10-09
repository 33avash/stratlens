"""Merge drafted label batches (data/labels/drafts/*.txt) with the labelling sample into data/labels/labels.csv.

Draft line format: "<row index in labelling_sample.csv> <LABEL> [optional note]".
Labels were drafted by Claude during the build session (not at app runtime) and are marked
label_source = "ai_drafted" until verified by a human (see scripts/compute_agreement.py).
Usage: python scripts/merge_labels.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from stratlens.config import LABELS, LABELS_CSV, THEMES

SAMPLE_CSV = LABELS / "labelling_sample.csv"
DRAFTS = LABELS / "drafts"


def main() -> None:
    sample = pd.read_csv(SAMPLE_CSV)
    rows = []
    for f in sorted(DRAFTS.glob("*.txt")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split(maxsplit=2)
            idx, label = int(parts[0]), parts[1].upper()
            if label not in THEMES:
                sys.exit(f"{f.name}: unknown label {label!r} on row {idx}")
            rows.append({"row": idx, "label": label, "note": parts[2] if len(parts) > 2 else ""})
    drafts = pd.DataFrame(rows)
    dup = drafts[drafts.row.duplicated()]
    if not dup.empty:
        sys.exit(f"duplicate rows in drafts: {dup.row.tolist()}")
    missing = sorted(set(range(len(sample))) - set(drafts.row))
    labelled = sample.join(drafts.set_index("row"), how="inner")
    labelled["label_source"] = "ai_drafted"
    labelled["labeller"] = "Claude (build-time drafting, not runtime)"
    labelled.to_csv(LABELS_CSV, index=False)
    print(f"labels: {len(labelled)} rows -> {LABELS_CSV}" + (f"; {len(missing)} sample rows unlabelled" if missing else ""))
    print(labelled.label.value_counts().reindex(THEMES).to_string())


if __name__ == "__main__":
    main()
