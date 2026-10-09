"""Classify every corpus passage with the saved model -> data/processed/benchmark_predictions.csv.

The app's Benchmark page reads this file; it never retrains.
Usage: python scripts/build_benchmark.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from stratlens import model as mdl
from stratlens.config import BENCHMARK_CSV, CORPUS_CSV, LABELS_CSV, TRAIN_CSV


def main() -> None:
    clf, meta = mdl.load()
    corpus = pd.read_csv(CORPUS_CSV)
    preds = mdl.predict_frame(clf, corpus["text"], threshold=meta["threshold"])
    out = pd.concat([corpus.reset_index(drop=True), preds], axis=1)
    # Provenance flags so the UI can be honest about which passages the model saw in training.
    train_texts = set(pd.read_csv(TRAIN_CSV)["text"]) if TRAIN_CSV.exists() else set()
    out["in_training_set"] = out["text"].isin(train_texts)
    labels = pd.read_csv(LABELS_CSV)[["text", "label"]].drop_duplicates("text") if LABELS_CSV.exists() else None
    if labels is not None:
        out = out.merge(labels.rename(columns={"label": "ai_drafted_label"}), on="text", how="left")
    out.to_csv(BENCHMARK_CSV, index=False)
    print(f"benchmark: {len(out)} passages -> {BENCHMARK_CSV}")
    print(out.groupby(["company", "year"])["pred"].value_counts(normalize=True).unstack(fill_value=0).round(3))


if __name__ == "__main__":
    main()
