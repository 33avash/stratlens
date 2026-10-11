"""Sentence embeddings (all-MiniLM-L6-v2) with an on-disk cache, plus an sklearn-compatible embedding classifier.

The encoder is optional at runtime: `available()` is False when sentence-transformers cannot be imported or
the model cannot be loaded, and callers fall back (e.g. Ask uses TF-IDF retrieval).
"""
from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from .config import DATA, SEED

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE = DATA / "cache" / "emb_minilm.npz"
DIM = 384


@lru_cache(maxsize=1)
def _encoder():
    import os
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        from sentence_transformers import SentenceTransformer
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        return None
    try:  # offline first: use the locally cached model without contacting the Hugging Face Hub
        return SentenceTransformer(MODEL_NAME, device="cpu", local_files_only=True)
    except Exception:
        pass
    try:  # first run: download (~90 MB)
        return SentenceTransformer(MODEL_NAME, device="cpu")
    except Exception:
        return None


def available() -> bool:
    return _encoder() is not None


def _key(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


_mem: dict[str, np.ndarray] = {}


def _load_cache() -> None:
    if _mem or not CACHE.exists():
        return
    try:
        z = np.load(CACHE, allow_pickle=False)
        _mem.update(zip(z["keys"].tolist(), z["vecs"]))
    except Exception:
        pass


def _save_cache() -> None:
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(_mem.keys()))
        vecs = np.stack(list(_mem.values())).astype(np.float32)
        np.savez_compressed(CACHE, keys=keys, vecs=vecs)
    except OSError:
        pass


def encode(texts, cache: bool = True) -> np.ndarray:
    """L2-normalised embeddings, shape (n, 384). Raises RuntimeError if the encoder is unavailable."""
    texts = list(texts)
    if cache:
        _load_cache()
    missing = list(dict.fromkeys(t for t in texts if _key(t) not in _mem))
    if missing:
        enc = _encoder()
        if enc is None:
            raise RuntimeError("sentence-transformers model is not available")
        vecs = enc.encode(missing, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        for t, v in zip(missing, vecs):
            _mem[_key(t)] = np.asarray(v, dtype=np.float32)
        if cache:
            _save_cache()
    if not texts:
        return np.zeros((0, DIM), dtype=np.float32)
    return np.stack([_mem[_key(t)] for t in texts])


class EmbeddingLR(BaseEstimator, ClassifierMixin):
    """Logistic regression on sentence embeddings; takes raw text like the TF-IDF pipeline."""

    def __init__(self, C: float = 1.0, preprocess=None):
        self.C = C
        self.preprocess = preprocess

    def _x(self, texts):
        texts = list(texts)
        if self.preprocess is not None:
            texts = [self.preprocess(t) for t in texts]
        return encode(texts)

    def fit(self, texts, y):
        self.clf_ = LogisticRegression(C=self.C, class_weight="balanced", max_iter=5000, random_state=SEED)
        self.clf_.fit(self._x(texts), y)
        self.classes_ = self.clf_.classes_
        return self

    def predict_proba(self, texts):
        return self.clf_.predict_proba(self._x(texts))

    def predict(self, texts):
        return self.clf_.predict(self._x(texts))
