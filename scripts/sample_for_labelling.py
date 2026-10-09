"""Draw the labelling sample from the corpus: a fixed fraction of every company x year x section stratum.

Sampling by fraction (rather than a global n) means a report added later is sampled at the same rate
without redrawing passages already labelled. Already-sampled passage ids are kept.
Letter passages are scarce, so they are sampled at --letter-frac (default 1.0) to give the writing-style
subgroup enough test examples.
Usage: python scripts/sample_for_labelling.py [--frac 0.7] [--letter-frac 1.0]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from stratlens.config import CORPUS_CSV, LABELS, SEED

SAMPLE_CSV = LABELS / "labelling_sample.csv"


def main(frac: float, letter_frac: float) -> None:
    corpus = pd.read_csv(CORPUS_CSV)
    existing = pd.read_csv(SAMPLE_CSV) if SAMPLE_CSV.exists() else pd.DataFrame(columns=corpus.columns)
    done_strata = set(map(tuple, existing[["company", "year", "section"]].drop_duplicates().values))
    new = []
    for key, g in corpus.groupby(["company", "year", "section"]):
        if key in done_strata:
            continue
        f = letter_frac if key[2] == "LETTER" else frac  # letters are scarce: oversample for subgroup audit power
        new.append(g.sample(frac=f, random_state=SEED))
    sample = pd.concat([existing[existing.passage_id.isin(corpus.passage_id)], *new], ignore_index=True)
    LABELS.mkdir(parents=True, exist_ok=True)
    sample.to_csv(SAMPLE_CSV, index=False)
    print(f"labelling sample: {len(sample)} passages ({sum(len(n) for n in new)} new)")
    print(sample.groupby(["company", "year", "section"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.7)
    ap.add_argument("--letter-frac", type=float, default=1.0)
    a = ap.parse_args()
    main(a.frac, a.letter_frac)
