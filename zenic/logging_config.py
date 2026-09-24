"""Structured logging for Zenic.

Two output formats, selected by ``LOG_FORMAT``:
  * ``text`` (default) — human-readable, for local development
  * ``json``           — one JSON object per line, for log aggregation

Every log record can carry a correlation id so a single user turn can be traced
across the router, retrieval, and generation nodes. Set it once per request with
:func:`bind_correlation_id`; it propagates via a ContextVar, so it survives the
thread hops LangGraph and Streamlit make.
"""
from __future__ import annotations

import contextvars
import json
import logging
import sys
import uuid
from typing import Any

_CORRELATION_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "zenic_correlation_id", default="-"
)

_RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}

_configured = False


def bind_correlation_id(correlation_id: str | None = None) -> str:
    """Attach a correlation id to the current context. Returns the id used."""
    cid = correlation_id or uuid.uuid4().hex[:12]
    _CORRELATION_ID.set(cid)
    return cid


def current_correlation_id() -> str:
    return _CORRELATION_ID.get()


class _CorrelationFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = _CORRELATION_ID.get()
        return True


class JsonFormatter(logging.Formatter):
    """Render records as single-line JSON, including any `extra` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", "-"),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key != "correlation_id":
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable format that still shows the structured `extra` fields."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED and key != "correlation_id"
        }
        if extras:
            base += "  " + " ".join(f"{k}={v}" for k, v in extras.items())
        return base


_TEXT_FORMAT = "%(asctime)s %(levelname)-7s [%(correlation_id)s] %(name)s: %(message)s"


def _settings_or_defaults() -> tuple[str, str]:
    """Read the log level and format from settings, tolerating a bad config.

    Modules call :func:`get_logger` at import time, so this must never raise —
    otherwise a missing environment variable turns into an import error with no
    log output to explain it.
    """
    try:
        # Imported lazily: config imports this module, so a top-level import would cycle.
        from zenic.config import get_settings

        settings = get_settings()
        return settings.log_level, settings.log_format
    except Exception:
        return "INFO", "text"


def configure_logging(level: str | None = None, fmt: str | None = None) -> None:
    """Install Zenic's handler on the root logger. Idempotent.

    Called automatically by :mod:`zenic.config`; call it directly only to
    override the level or format at runtime.
    """
    global _configured

    if level is None or fmt is None:
        settings_level, settings_fmt = _settings_or_defaults()
        level = level or settings_level
        fmt = fmt or settings_fmt
    level = level.upper()
    fmt = fmt.lower()

    root = logging.getLogger()
    if _configured:
        root.setLevel(level)
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(_CorrelationFilter())
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter(_TEXT_FORMAT))

    root.handlers[:] = [handler]
    root.setLevel(level)

    # These libraries log every HTTP request at INFO — far too chatty for us.
    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers", "chromadb", "groq", "openai", "langsmith", "qdrant_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger. Safe to call at module import time."""
    configure_logging()
    return logging.getLogger(name)
