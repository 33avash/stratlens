"""Central configuration: paths, taxonomy, seeds and thresholds."""
from pathlib import Path

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
BENCHMARK_CSV = PROCESSED / "benchmark_predictions.csv"
MODEL_CARD = ROOT / "MODEL_CARD.md"

SEED = 42
TEST_SIZE = 0.2
REVIEW_THRESHOLD = 0.45
MIN_WORDS = 8
MAX_WORDS = 70
SMALL_N = 30

THEMES = ["PROD", "DIST", "DIGI", "OPS", "SUST", "BRAND", "OTHER"]
THEME_NAMES = {
    "PROD": "Product innovation & portfolio",
    "DIST": "Market expansion & distribution",
    "DIGI": "Digital & e-commerce",
    "OPS": "Operations & supply chain",
    "SUST": "Sustainability & sourcing",
    "BRAND": "Consumer & brand strategy",
    "OTHER": "Other / unclear",
}
STRATEGIC_THEMES = [t for t in THEMES if t != "OTHER"]

COMPANY_NAMES = {"HUL": "Hindustan Unilever", "DABUR": "Dabur India", "NESTLE": "Nestlé India"}
SECTION_NAMES = {"LETTER": "Chairman/CEO letter", "MDA": "MD&A"}

# Plain names for the systems in metrics.json (technique in brackets).
SYSTEM_NAMES = {
    "TF-IDF + LR (masked)": "Word-weight classifier (TF-IDF)",
    "MiniLM embeddings + LR (masked)": "Meaning-based classifier (MiniLM embeddings)",
    "TF-IDF + LR (unmasked)": "Word-weight classifier, names not hidden",
    "Keyword baseline": "Keyword matching (lexicon)",
}
