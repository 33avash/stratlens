"""Central configuration: paths, taxonomy, seeds and thresholds."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
LABELS = DATA / "labels"
SPLITS = DATA / "splits"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
CONFIG_DIR = ROOT / "config"

SECTIONS_CSV = DATA / "sections.csv"
OVERRIDES_CSV = CONFIG_DIR / "section_overrides.csv"
CORPUS_CSV = PROCESSED / "corpus.csv"
DROPS_CSV = PROCESSED / "drop_log.csv"
LABELS_CSV = LABELS / "labels.csv"
TRAIN_CSV = SPLITS / "train.csv"
TEST_CSV = SPLITS / "test.csv"
MODEL_PATH = MODELS / "classifier.joblib"
METRICS_JSON = REPORTS / "metrics.json"
ERROR_NOTES_CSV = REPORTS / "error_notes.csv"
REPORTS_DIR = PROCESSED / "reports"          # one CSV + JSON per report (python -m stratlens.build)
PASSAGES_CSV = PROCESSED / "passages.csv"     # every report, classified and scored: what the app reads
BENCHMARK_CSV = PASSAGES_CSV
EMBEDDINGS_NPZ = PROCESSED / "embeddings.npz"  # MiniLM vectors aligned to passage_id
ANALYTICS_JSON = PROCESSED / "analytics.json"
MANIFEST_JSON = PROCESSED / "manifest.json"
MODEL_CARD = ROOT / "MODEL_CARD.md"

SEED = 42
TEST_SIZE = 0.2
REVIEW_THRESHOLD = 0.45
MIN_WORDS = 8
MAX_WORDS = 70
SMALL_N = 30

COMPANIES_YAML = CONFIG_DIR / "companies.yaml"
THEMES_YAML = CONFIG_DIR / "themes.yaml"
GAZETTEER_YAML = CONFIG_DIR / "gazetteer.yaml"


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


# ---- themes (config/themes.yaml)
_themes = _yaml(THEMES_YAML)["themes"]
THEMES = [t["code"] for t in _themes]
THEME_NAMES = {t["code"]: t["name"] for t in _themes}
THEME_SHORT = {t["code"]: t["short"] for t in _themes}
THEME_COLORS = {t["code"]: t["color"] for t in _themes}
STRATEGIC_THEMES = [t for t in THEMES if t != "OTHER"]

# ---- companies (config/companies.yaml)
_companies = _yaml(COMPANIES_YAML)
COMPANIES = {c["code"]: c for c in _companies["companies"]}
COMPANY_NAMES = {k: c["name"] for k, c in COMPANIES.items()}
COMPANY_SHORT = {k: c["short"] for k, c in COMPANIES.items()}
COMPANY_COLORS = {k: c["color"] for k, c in COMPANIES.items()}
COMPANY_ALIASES = {k: c.get("aliases", [k.lower()]) for k, c in COMPANIES.items()}
PEER_GROUPS = _companies.get("peer_groups", {})
EXPECTED_YEARS = _companies.get("years", [])


def peers_of(code: str) -> list[str]:
    """Other companies in the same peer group (empty for an unknown company)."""
    group = COMPANIES.get(code, {}).get("peer_group")
    return [k for k, c in COMPANIES.items() if k != code and group and c.get("peer_group") == group]


SECTION_NAMES = {"LETTER": "Chairman/CEO letter", "MDA": "MD&A"}

# Plain names for the systems in metrics.json (technique in brackets).
SYSTEM_NAMES = {
    "TF-IDF + LR (masked)": "Word-weight classifier (TF-IDF)",
    "MiniLM embeddings + LR (masked)": "Meaning-based classifier (MiniLM embeddings)",
    "TF-IDF + LR (unmasked)": "Word-weight classifier, names not hidden",
    "Keyword baseline": "Keyword matching (lexicon)",
}
