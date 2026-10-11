"""One-command offline build: python -m stratlens.build [--force] [--only CODE_FYyyyy]

For every report in data/raw (named like britannia_FY2026.pdf):
  1. segment   extract text -> find the Chairman/CEO letter and MD&A -> clean -> passages
  2. predict   theme + confidence (saved classifier) -> concreteness (regex + spaCy NER) -> MiniLM embeddings
Then, for all reports together: passages.csv, embeddings.npz, analytics.json, sections.csv, drop_log.csv,
and the text the app shows (narratives.json: briefs and comparison summaries; ask_examples.json).

Idempotent. Each step has a key (hash of its inputs, recorded in data/processed/manifest.json):
  segment key  PDF bytes + section overrides for that report + preprocessing/section code
  predict key  segment key + model file + gazetteer + concreteness code + label files
A report is only redone when its key changes. A processed report whose PDF is absent (raw PDFs are not in
the repo) is kept, and re-predicted from its stored passages if the model changed.
Section detection must pass sections.validate(); otherwise the report is SKIPPED with a warning. It never
falls back to the whole report. Confirmed ranges go in config/section_overrides.csv.
The app only reads the files written here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import analytics, embeddings, explain, precompute, specificity
from . import model as mdl
from .config import (ANALYTICS_JSON, COMPANIES, CORPUS_CSV, DROPS_CSV, EMBEDDINGS_NPZ, GAZETTEER_YAML, LABELS_CSV,
                     MANIFEST_JSON, MODEL_PATH, OVERRIDES_CSV, PASSAGES_CSV, PROCESSED, RAW, REPORTS_DIR, ROOT,
                     SECTIONS_CSV, STRATEGIC_THEMES, TRAIN_CSV)
from .extract import ExtractionError, extract_file, parse_filename
from .preprocess import segment
from .sections import heading, resolve_sections, validate

PKG = ROOT / "stratlens"
SEGMENT_CODE = ["extract.py", "sections.py", "preprocess.py"]
PREDICT_CODE = ["masking.py", "model.py", "specificity.py", "explain.py"]


# ---------------------------------------------------------------- hashing
def _md5_bytes(*chunks: bytes) -> str:
    h = hashlib.md5()
    for c in chunks:
        h.update(c)
    return h.hexdigest()


def _file_md5(path: Path) -> str:
    return _md5_bytes(path.read_bytes()) if path.exists() else "absent"


def _overrides_rows(code: str, year: str) -> str:
    try:
        df = pd.read_csv(OVERRIDES_CSV)
    except FileNotFoundError:
        return ""
    return df[(df.company == code) & (df.year == year)].to_csv(index=False)


def segment_key(pdf: Path, code: str, year: str) -> str:
    return _md5_bytes(_file_md5(pdf).encode(), _overrides_rows(code, year).encode(),
                      *[(PKG / f).read_bytes() for f in SEGMENT_CODE])


def predict_key(seg_key: str) -> str:
    return _md5_bytes(seg_key.encode(), *[_file_md5(p).encode() for p in (MODEL_PATH, GAZETTEER_YAML, TRAIN_CSV,
                                                                           LABELS_CSV)],
                      *[(PKG / f).read_bytes() for f in PREDICT_CODE])


def _load_manifest() -> dict:
    try:
        return json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


# ---------------------------------------------------------------- steps
def run_segment(pdf: Path, code: str, year: str) -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    """Passages, drops and section info for one PDF, or None (with a printed warning) if it must be skipped."""
    try:
        doc = extract_file(pdf)
    except ExtractionError as exc:
        print(f"  !! SKIPPED {pdf.name}: {exc}")
        return None
    if doc.looks_scanned:
        print(f"  !! SKIPPED {pdf.name}: {doc.scanned_share:.0%} of pages look scanned (no text layer). Run OCR first.")
        return None
    secs = resolve_sections(doc, code, year)
    for s in secs:
        print(f"     {s.section:<6} p{s.start_page}-{s.end_page} ({len(s.pages())} pages, {s.method}) "
              f"starts: \"{heading(doc, s.start_page)}\"")
    problems = validate(doc, secs)
    if problems:
        print("  " + "!" * 70)
        print(f"  !! SKIPPED {pdf.name}: section detection is not trustworthy:")
        for p in problems:
            print(f"  !!   - {p}")
        print(f"  !! Check the PDF and add confirmed page ranges to {OVERRIDES_CSV.relative_to(ROOT)}, then rerun.")
        print("  " + "!" * 70)
        return None
    res = segment(doc, secs, code, year)
    if res.passages.empty:
        print(f"  !! SKIPPED {pdf.name}: no passages survived cleaning; check the page ranges.")
        return None
    info = {"file": pdf.name, "page_count": doc.page_count,
            "pages_kept": int(sum(len(s.pages()) for s in secs)),
            "sections": [{"section": s.section, "start_page": s.start_page, "end_page": s.end_page,
                          "n_pages": len(s.pages()), "method": s.method,
                          "chars": sum(len(doc.pages[p - 1].text) for p in s.pages() if p <= doc.page_count),
                          "starts_with": heading(doc, s.start_page)} for s in secs],
            "n_dropped": int(len(res.drops))}
    return res.passages, res.drops, info


def run_predict(passages: pd.DataFrame, model, meta: dict) -> pd.DataFrame:
    base = passages[["passage_id", "company", "year", "section", "page", "text", "word_count"]].reset_index(drop=True)
    pf = mdl.predict_frame(model, base["text"], threshold=meta["threshold"])
    out = pd.concat([base, pf], axis=1)
    out["why_terms"] = [json.dumps(t, ensure_ascii=False) for t in explain.why_terms(model, base["text"])]
    spec = specificity.score_texts(out["text"].tolist())
    out["specificity"] = [s["specificity"] for s in spec]
    out["spec_tier"] = [s["spec_tier"] for s in spec]
    out["spec_signals"] = [json.dumps(s["spec_signals"], ensure_ascii=False) for s in spec]
    # Provenance, so the UI can be honest about which passages the model saw in training.
    train_texts = set(pd.read_csv(TRAIN_CSV)["text"]) if TRAIN_CSV.exists() else set()
    out["in_training_set"] = out["text"].isin(train_texts)
    if LABELS_CSV.exists():
        labels = pd.read_csv(LABELS_CSV)[["text", "label"]].drop_duplicates("text")
        out = out.merge(labels.rename(columns={"label": "ai_drafted_label"}), on="text", how="left")
    else:
        out["ai_drafted_label"] = np.nan
    if embeddings.available():
        embeddings.encode(out["text"].tolist())  # fills the on-disk cache; aggregated into embeddings.npz below
    return out


# ---------------------------------------------------------------- aggregate
def _summary_row(key: str, status: str, df: pd.DataFrame | None, info: dict | None) -> dict:
    if df is None:
        return {"report": key, "status": status, "pages kept": "", "passages": "", "% strategy": "",
                "% low confidence": ""}
    return {"report": key, "status": status, "pages kept": info.get("pages_kept", "") if info else "",
            "passages": len(df), "% strategy": f"{df.pred.isin(STRATEGIC_THEMES).mean():.0%}",
            "% low confidence": f"{df.needs_review.astype(bool).mean():.0%}"}


def aggregate(keys: list[str]) -> pd.DataFrame:
    frames, drops, secs = [], [], []
    for k in keys:
        frames.append(pd.read_csv(REPORTS_DIR / f"{k}.csv"))
        d = REPORTS_DIR / f"{k}_drops.csv"
        if d.exists():
            drops.append(pd.read_csv(d))
        info = json.loads((REPORTS_DIR / f"{k}.json").read_text(encoding="utf-8"))
        code, year = k.split("_")
        for s in info["sections"]:
            secs.append({"company": code, "year": year, "file": info["file"], **{x: s[x] for x in
                         ("section", "start_page", "end_page", "n_pages", "chars", "method")}})
    passages = pd.concat(frames, ignore_index=True).sort_values(["company", "year", "passage_id"])
    passages.to_csv(PASSAGES_CSV, index=False)
    passages[["passage_id", "company", "year", "section", "page", "text", "word_count"]].to_csv(CORPUS_CSV,
                                                                                                index=False)
    if drops:
        pd.concat(drops, ignore_index=True).to_csv(DROPS_CSV, index=False)
    pd.DataFrame(secs).to_csv(SECTIONS_CSV, index=False)
    if embeddings.available():
        vecs = embeddings.encode(passages["text"].tolist())
        np.savez_compressed(EMBEDDINGS_NPZ, passage_id=passages["passage_id"].to_numpy(dtype=str),
                            vecs=vecs.astype(np.float32))
    else:
        print("  WARNING: MiniLM model unavailable; embeddings.npz not updated.")
    an = analytics.compute(passages)
    ANALYTICS_JSON.write_text(json.dumps(an, ensure_ascii=False), encoding="utf-8")
    return passages


def write_narratives(passages: pd.DataFrame) -> None:
    an = json.loads(ANALYTICS_JSON.read_text(encoding="utf-8"))
    print("Writing briefs, comparison summaries and example answers (cached LLM outputs are reused) ...",
          flush=True)
    stats = precompute.write_all(an, passages)
    print("  " + " · ".join(f"{k}: {v}" for k, v in stats.items()))


# ---------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stratlens.build", description=__doc__.split("\n")[0])
    ap.add_argument("--force", action="store_true", help="redo every report even if nothing changed")
    ap.add_argument("--only", action="append", default=[], help="limit to a report, e.g. MARICO_FY2026")
    ap.add_argument("--narratives", action="store_true", help="rewrite briefs, summaries and example answers")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # PDF text includes ₹, ’ and – on Windows consoles

    if not MODEL_PATH.exists():
        print(f"No trained model at {MODEL_PATH}. Run python train.py first.")
        return 1
    model, meta = mdl.load(MODEL_PATH)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()

    pdfs = {}
    for p in sorted(RAW.glob("*")) if RAW.exists() else []:
        if p.suffix.lower() not in {".pdf", ".txt"}:
            continue
        code, year = parse_filename(p.name)
        if not code:
            print(f"skip {p.name}: name must look like britannia_FY2026.pdf")
            continue
        if code not in COMPANIES:
            print(f"  WARNING: {code} is not in config/companies.yaml; it will have no peer group or colour.")
        pdfs[f"{code}_{year}"] = p
    stored = {p.stem for p in REPORTS_DIR.glob("*.json")}
    keys = sorted(set(pdfs) | stored)
    if args.only:
        keys = [k for k in keys if k in {o.upper() for o in args.only}]
    if not keys:
        print(f"No reports found in {RAW} or {REPORTS_DIR}.")
        return 1

    rows, ok_keys, changed = [], [], False
    for key in keys:
        code, year = key.split("_")
        csv, js = REPORTS_DIR / f"{key}.csv", REPORTS_DIR / f"{key}.json"
        entry = manifest.get(key, {})
        pdf = pdfs.get(key)
        seg_key = segment_key(pdf, code, year) if pdf else entry.get("segment_key")
        pred_key = predict_key(seg_key) if seg_key else None
        have = csv.exists() and js.exists()

        if have and not args.force and entry.get("segment_key") == seg_key and entry.get("predict_key") == pred_key:
            rows.append(_summary_row(key, "up to date", pd.read_csv(csv), json.loads(js.read_text(encoding="utf-8"))))
            ok_keys.append(key)
            continue

        if pdf is not None and (args.force or not have or entry.get("segment_key") != seg_key):
            print(f"\n{key}: reading {pdf.name}", flush=True)
            res = run_segment(pdf, code, year)
            if res is None:
                rows.append(_summary_row(key, "SKIPPED", None, None))
                if have:  # keep the last good version rather than silently losing it
                    print(f"  keeping the previously processed {key}")
                    ok_keys.append(key)
                continue
            passages, drops, info = res
            drops.to_csv(REPORTS_DIR / f"{key}_drops.csv", index=False)
            status = "built"
        elif have:
            print(f"\n{key}: re-predicting from stored passages" + ("" if pdf else " (PDF not in data/raw)"),
                  flush=True)
            passages, info = pd.read_csv(csv), json.loads(js.read_text(encoding="utf-8"))
            status = "re-predicted"
        else:
            rows.append(_summary_row(key, "SKIPPED", None, None))
            continue

        out = run_predict(passages, model, meta)
        out.to_csv(csv, index=False)
        info.update({"company": code, "year": year, "n_passages": int(len(out)), "built": date.today().isoformat()})
        js.write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
        manifest[key] = {"segment_key": seg_key, "predict_key": pred_key, "built": date.today().isoformat()}
        rows.append(_summary_row(key, status, out, info))
        ok_keys.append(key)
        changed = True

    all_keys = sorted({p.stem for p in REPORTS_DIR.glob("*.json")} if args.only else ok_keys)
    if changed or args.force or not PASSAGES_CSV.exists() or not ANALYTICS_JSON.exists():
        print("\nAggregating passages, embeddings and analytics ...", flush=True)
        passages = aggregate(all_keys)
    else:
        passages = pd.read_csv(PASSAGES_CSV)
    if changed or args.force or args.narratives or not precompute.complete():
        write_narratives(passages)
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    table = pd.DataFrame(rows)
    print("\n" + table.to_string(index=False))
    n_co = passages.company.nunique()
    print(f"\n{passages[['company', 'year']].drop_duplicates().shape[0]} reports · {n_co} companies · "
          f"{len(passages):,} passages -> {PASSAGES_CSV.relative_to(ROOT)}")
    skipped = table[table.status == "SKIPPED"]
    if not skipped.empty:
        print(f"!! {len(skipped)} report(s) skipped: {', '.join(skipped.report)}. See the warnings above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
