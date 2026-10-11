"""LLM client behind a small interface. Only Gemini is implemented; others (e.g. Ollama) can be added
by subclassing LLMClient and extending get_client().

Configuration (environment or .env in the project root):
  GEMINI_API_KEY   required to enable AI writing
  GEMINI_MODEL     optional, default DEFAULT_MODEL
With no key, get_client() returns None and every caller falls back to template text.
Generated text is cached on disk (data/cache/llm.json) keyed by model + prompt, so repeated demos are free.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from functools import lru_cache

from ..config import DATA, ROOT

DEFAULT_MODEL = "gemini-3.5-flash"   # current fast, low-cost Flash model (checked against models.list)
CACHE = DATA / "cache" / "llm.json"
TIMEOUT_MS = 45_000
RETRIES = 2


MAX_WAIT_S = 70  # wait out per-minute rate limits up to this long; longer waits mean the daily quota is gone


def _retry_seconds(msg: str) -> float | None:
    """'Please retry in 7h29m13.7s' -> seconds, or None if the API gave no delay."""
    m = re.search(r"retry in\s+(?:(\d+)h)?(?:(\d+)m(?!s))?(?:([\d.]+)s)?", msg)
    if not m or not any(m.groups()):
        m2 = re.search(r"retryDelay['\"]?:\s*['\"]?([\d.]+)s", msg)
        return float(m2.group(1)) if m2 else None
    h, mi, s = m.groups()
    return int(h or 0) * 3600 + int(mi or 0) * 60 + float(s or 0)


class LLMError(Exception):
    """Raised with a short, user-facing message."""


class LLMClient:
    name = "base"
    model = ""

    def generate(self, system: str, prompt: str, max_tokens: int = 1200, json_mode: bool = False) -> str:
        raise NotImplementedError


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
    except Exception:
        pass


def _cache_get(key: str) -> str | None:
    try:
        return json.loads(CACHE.read_text(encoding="utf-8")).get(key)
    except Exception:
        return None


def _cache_put(key: str, value: str) -> None:
    try:
        data = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
        data[key] = value
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


class GeminiClient(LLMClient):
    name = "Gemini"

    def __init__(self, api_key: str, model: str | None = None):
        from google import genai
        from google.genai import types
        self._types = types
        self.model = model or DEFAULT_MODEL
        self.exhausted = False  # set when the daily quota runs out; later calls fail fast
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=TIMEOUT_MS))

    def _config(self, system: str, max_tokens: int, json_mode: bool):
        t = self._types
        kw = dict(system_instruction=system, temperature=0.2, max_output_tokens=max_tokens)
        if json_mode:
            kw["response_mime_type"] = "application/json"
        if self.model.startswith("gemini-2.5"):
            kw["thinking_config"] = t.ThinkingConfig(thinking_budget=0)
        elif self.model.startswith("gemini-3"):
            kw["thinking_config"] = t.ThinkingConfig(thinking_level="low")
        return t.GenerateContentConfig(**kw)

    def _key(self, system: str, prompt: str, json_mode: bool) -> str:
        return hashlib.md5(f"{self.model}\n{system}\n{prompt}\n{json_mode}".encode("utf-8")).hexdigest()

    def cached(self, system: str, prompt: str, json_mode: bool = False) -> str | None:
        """The stored output for this exact request, without any network call (None if never generated)."""
        return _cache_get(self._key(system, prompt, json_mode))

    def generate(self, system: str, prompt: str, max_tokens: int = 1200, json_mode: bool = False) -> str:
        key = self._key(system, prompt, json_mode)
        cached = _cache_get(key)
        if cached is not None:
            return cached
        if self.exhausted:
            raise LLMError("The AI service's daily quota is used up. Try again tomorrow.")
        last = None
        for attempt in range(RETRIES + 1):
            try:
                resp = self._client.models.generate_content(model=self.model, contents=prompt,
                                                            config=self._config(system, max_tokens, json_mode))
                text = (resp.text or "").strip()
                if not text:
                    raise LLMError("The model returned an empty answer.")
                _cache_put(key, text)
                return text
            except LLMError:
                raise
            except Exception as exc:  # google.genai.errors.APIError and network errors
                last = exc
                code = getattr(exc, "code", None)
                wait = _retry_seconds(str(exc))
                if code == 429 and wait is not None:
                    if wait > MAX_WAIT_S or attempt >= RETRIES:
                        self.exhausted = wait > MAX_WAIT_S
                        break  # daily quota exhausted (retry in hours): waiting here cannot help
                    time.sleep(wait + 1)  # per-minute limit: wait as long as the API asks, then retry
                    continue
                if code in (429, 500, 503) and attempt < RETRIES:
                    time.sleep(2 + 3 * attempt)
                    continue
                break
        code = getattr(last, "code", None)
        if code == 429:
            raise LLMError("The AI service is rate-limiting requests. Wait a minute and try again.")
        if code in (500, 503):
            raise LLMError("The AI service is temporarily overloaded. Try again shortly.")
        if code in (400, 401, 403):
            raise LLMError("The AI service rejected the request. Check the API key and model in .env.")
        raise LLMError("Could not reach the AI service (network or timeout).")


@lru_cache(maxsize=1)
def get_client() -> LLMClient | None:
    _load_env()
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        return None
    try:
        return GeminiClient(key, os.environ.get("GEMINI_MODEL", "").strip() or None)
    except Exception:
        return None


def status() -> str:
    c = get_client()
    return "AI writing on" if c else "AI writing off"
