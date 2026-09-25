"""Central runtime configuration.

Every environment variable Zenic reads is declared here — once. Modules ask for
``get_settings()`` instead of touching ``os.environ`` directly, so that:

  * a missing key fails with an actionable message instead of a bare ``KeyError``
    surfacing from three call frames deep;
  * defaults live in one place rather than being repeated at each call site;
  * tests can swap the whole configuration by calling :func:`reload_settings`.

Secrets are never logged or included in ``repr`` output — see :meth:`Settings.redacted`.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from dotenv import load_dotenv

from zenic.errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Env vars holding credentials — redacted in every diagnostic output.
_SECRET_FIELDS = frozenset(
    {
        "groq_api_key",
        "qdrant_api_key",
        "usda_api_key",
        "openfda_api_key",
        "google_api_key",
    }
)

_VALID_ENVS = ("development", "production")

#: Matches a trailing `  # comment` — whitespace is required before the hash.
_COMMENT_RE = re.compile(r"\s+#")


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    # `KEY=value  # comment` is valid in .env files but python-dotenv only strips
    # the comment for unquoted values; be defensive and strip it either way.
    # Requires whitespace before the '#', so a value that legitimately contains
    # one (a generated password, say) survives intact.
    return _COMMENT_RE.split(raw, maxsplit=1)[0].strip()


def _env_optional(name: str) -> str | None:
    value = _env_str(name, "")
    return value or None


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = _env_str(name, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the process configuration."""

    # ── Deployment ─────────────────────────────────────────────────────────
    env: str
    log_level: str
    log_format: str

    # ── Credentials (optional at import time, required at point of use) ────
    groq_api_key: str | None = field(repr=False)
    qdrant_url: str | None
    qdrant_api_key: str | None = field(repr=False)
    usda_api_key: str | None = field(repr=False)
    openfda_api_key: str | None = field(repr=False)
    google_api_key: str | None = field(repr=False)

    # ── Models ─────────────────────────────────────────────────────────────
    groq_model: str
    groq_plan_model: str
    embed_model: str
    reranker_model: str
    embedding_dimensions: int

    # ── Vector store ───────────────────────────────────────────────────────
    collection_name: str
    chroma_path: Path
    bm25_corpus_path: Path

    # ── Retrieval tuning ───────────────────────────────────────────────────
    retrieval_candidate_pool: int
    retrieval_max_per_source: int
    retrieval_top_k: int
    rerank_batch_size: int
    multi_query_enabled: bool
    multi_query_count: int

    # ── Resilience ─────────────────────────────────────────────────────────
    llm_timeout_seconds: float
    llm_max_retries: int
    http_timeout_seconds: float
    http_max_retries: int
    vector_store_timeout_seconds: float

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    def redacted(self) -> dict[str, Any]:
        """Config as a dict with every secret replaced by a presence marker."""
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in _SECRET_FIELDS:
                out[f.name] = "<set>" if value else "<unset>"
            else:
                out[f.name] = str(value) if isinstance(value, Path) else value
        return out

    # ── Point-of-use credential accessors ──────────────────────────────────
    # Raising here, rather than at construction, keeps the unit-test suite and
    # the ingestion scripts runnable without production credentials.

    def require_groq_api_key(self) -> str:
        return self._require("groq_api_key", "GROQ_API_KEY", "all LLM calls")

    def require_usda_api_key(self) -> str:
        return self._require("usda_api_key", "USDA_API_KEY", "the USDA food API fallback")

    def require_google_api_key(self) -> str:
        return self._require("google_api_key", "GOOGLE_API_KEY", "the Pillar 3 RAGAS judge")

    def require_qdrant(self) -> tuple[str, str]:
        url = self._require("qdrant_url", "QDRANT_URL", "the production vector store")
        key = self._require("qdrant_api_key", "QDRANT_API_KEY", "the production vector store")
        return url, key

    def _require(self, attr: str, env_name: str, purpose: str) -> str:
        value = getattr(self, attr)
        if not value:
            raise ConfigError(
                f"{env_name} is not set but is required for {purpose}. "
                f"Add it to your .env file (see .env.example) or export it in the environment."
            )
        return value


_dotenv_loaded = False


def _load_dotenv_once() -> None:
    """Read .env into the process environment, exactly once.

    Deployment environment variables take precedence over the local .env file. Doing this only on the first build means a later
    :func:`reload_settings` picks up variables the caller has since set —
    without .env silently overwriting them again.
    """
    global _dotenv_loaded
    if not _dotenv_loaded:
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        _dotenv_loaded = True


def _build_settings() -> Settings:
    _load_dotenv_once()

    env = _env_str("ENV", "development").lower()
    if env not in _VALID_ENVS:
        raise ConfigError(f"ENV must be one of {_VALID_ENVS}, got {env!r}")

    log_format = _env_str("LOG_FORMAT", "json" if env == "production" else "text").lower()
    if log_format not in ("text", "json"):
        raise ConfigError(f"LOG_FORMAT must be 'text' or 'json', got {log_format!r}")

    settings = Settings(
        env=env,
        log_level=_env_str("LOG_LEVEL", "INFO").upper(),
        log_format=log_format,
        groq_api_key=_env_optional("GROQ_API_KEY"),
        qdrant_url=_env_optional("QDRANT_URL"),
        qdrant_api_key=_env_optional("QDRANT_API_KEY"),
        usda_api_key=_env_optional("USDA_API_KEY"),
        openfda_api_key=_env_optional("OPENFDA_API_KEY"),
        google_api_key=_env_optional("GOOGLE_API_KEY"),
        groq_model=_env_str("GROQ_MODEL", "openai/gpt-oss-20b"),
        groq_plan_model=_env_str("GROQ_PLAN_MODEL", "openai/gpt-oss-120b"),
        embed_model=_env_str("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        reranker_model=_env_str("RERANKER_MODEL", "BAAI/bge-reranker-base"),
        embedding_dimensions=_env_int("EMBEDDING_DIMENSIONS", 384),
        collection_name=_env_str("VECTOR_COLLECTION", "zenic_knowledge"),
        chroma_path=Path(_env_str("CHROMA_PATH", str(PROJECT_ROOT / "chroma_db"))),
        bm25_corpus_path=Path(
            _env_str("BM25_CORPUS_PATH", str(PROJECT_ROOT / "data" / "bm25_corpus.json"))
        ),
        retrieval_candidate_pool=_env_int("RETRIEVAL_CANDIDATE_POOL", 12),
        retrieval_max_per_source=_env_int("RETRIEVAL_MAX_PER_SOURCE", 12),
        retrieval_top_k=_env_int("RETRIEVAL_TOP_K", 7),
        # Small batches minimise padding waste in the cross-encoder — see
        # pipeline._rerank_scores. Larger is NOT faster here.
        rerank_batch_size=_env_int("RERANK_BATCH_SIZE", 4),
        multi_query_enabled=_env_str("MULTI_QUERY_ENABLED", "true").lower() != "false",
        multi_query_count=_env_int("MULTI_QUERY_COUNT", 3),
        llm_timeout_seconds=_env_float("LLM_TIMEOUT_SECONDS", 60.0),
        llm_max_retries=_env_int("LLM_MAX_RETRIES", 3),
        http_timeout_seconds=_env_float("HTTP_TIMEOUT_SECONDS", 10.0),
        http_max_retries=_env_int("HTTP_MAX_RETRIES", 2),
        vector_store_timeout_seconds=_env_float("VECTOR_STORE_TIMEOUT_SECONDS", 30.0),
    )
    _validate(settings)
    return settings


def _validate(settings: Settings) -> None:
    """Reject configurations that cannot possibly work, as early as possible."""
    if settings.is_production and not (settings.qdrant_url and settings.qdrant_api_key):
        raise ConfigError(
            "ENV=production requires QDRANT_URL and QDRANT_API_KEY. "
            "Set ENV=development to use the local ChromaDB store instead."
        )
    if settings.qdrant_url and not settings.qdrant_url.startswith(("http://", "https://")):
        raise ConfigError(
            "QDRANT_URL must start with http:// or https://"
        )
    if settings.qdrant_url:
        parsed = urlsplit(settings.qdrant_url)
        if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ConfigError("QDRANT_URL must be a server URL without embedded credentials, query, or fragment")
    if settings.retrieval_top_k > settings.retrieval_candidate_pool:
        raise ConfigError(
            "RETRIEVAL_TOP_K cannot exceed RETRIEVAL_CANDIDATE_POOL "
            f"({settings.retrieval_top_k} > {settings.retrieval_candidate_pool})"
        )
    for name in ("embedding_dimensions", "retrieval_candidate_pool", "retrieval_max_per_source",
                 "retrieval_top_k", "rerank_batch_size", "llm_timeout_seconds",
                 "http_timeout_seconds", "vector_store_timeout_seconds"):
        value = getattr(settings, name)
        if not math.isfinite(value) or value <= 0:
            raise ConfigError(f"{name.upper()} must be finite and positive")
    for name, maximum in (("llm_max_retries", 5), ("http_max_retries", 5), ("multi_query_count", 5)):
        if not 0 <= getattr(settings, name) <= maximum:
            raise ConfigError(f"{name.upper()} must be between 0 and {maximum}")
    if settings.is_production and not settings.qdrant_url.startswith("https://"):
        raise ConfigError("Production QDRANT_URL must use HTTPS")
    if settings.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        raise ConfigError("LOG_LEVEL is invalid")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, built once on first access."""
    return _build_settings()


def reload_settings() -> Settings:
    """Discard the cached settings and rebuild from the current environment.

    Intended for tests and for scripts that mutate ``os.environ`` before use.
    """
    get_settings.cache_clear()
    return get_settings()
