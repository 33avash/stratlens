"""Extract, section and segment every report in data/raw into data/processed/corpus.csv.

Also writes data/sections.csv (pages used per report) and data/processed/drop_log.csv.
Usage: python scripts/build_corpus.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from stratlens.config import CORPUS_CSV, DROPS_CSV, PROCESSED, RAW, SECTIONS_CSV
from stratlens.extract import extract_file, parse_filename
from stratlens.preprocess import segment
from stratlens.sections import resolve_sections


def main() -> None:
    files = sorted(p for p in RAW.iterdir() if p.suffix.lower() in {".pdf", ".txt"})
    if not files:
        sys.exit(f"No reports found in {RAW}. Add files named like HUL_FY2025.pdf.")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    passages, drops, sections = [], [], []
    for path in files:
        company, year = parse_filename(path.name)
        if not company:
            print(f"skip {path.name}: name must look like COMPANY_FY2025.pdf")
            continue
        print(f"Extracting {path.name} ...", flush=True)
        doc = extract_file(path)
        secs = resolve_sections(doc, company, year)
        if not secs:
            print(f"  WARNING: no strategy sections found in {path.name}; add rows to config/section_overrides.csv")
            continue
        for s in secs:
            chars = sum(len(doc.pages[p - 1].text) for p in s.pages() if p <= doc.page_count)
            sections.append({"company": company, "year": year, "file": path.name, "section": s.section,
                             "start_page": s.start_page, "end_page": s.end_page,
                             "n_pages": len(s.pages()), "chars": chars, "method": s.method})
        res = segment(doc, secs, company, year)
        passages.append(res.passages)
        drops.append(res.drops)
        print(f"  {doc.page_count} pages; sections {[(s.section, s.start_page, s.end_page) for s in secs]}; "
              f"kept {len(res.passages)} passages, dropped {len(res.drops)}")
    corpus = pd.concat(passages, ignore_index=True)
    corpus.to_csv(CORPUS_CSV, index=False)
    pd.concat(drops, ignore_index=True).to_csv(DROPS_CSV, index=False)
    pd.DataFrame(sections).to_csv(SECTIONS_CSV, index=False)
    print(f"\ncorpus: {len(corpus)} passages -> {CORPUS_CSV}")
    print(corpus.groupby(["company", "year", "section"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
