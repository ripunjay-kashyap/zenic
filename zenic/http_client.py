"""Shared HTTP client for third-party APIs (USDA, wger, OpenFDA).

The previous call sites used bare ``httpx.get`` with a fixed timeout and no
retry, so a single dropped connection failed the user's turn. This module adds
one pooled client plus a retry policy for the failures that are actually worth
retrying — connection errors, 429, and 5xx — and leaves 4xx alone.
"""
from __future__ import annotations

import random
import threading
import time
from typing import Any

import httpx

from zenic.config import get_settings
from zenic.errors import ExternalAPIError
from zenic.logging_config import get_logger

logger = get_logger(__name__)

_client: httpx.Client | None = None
_client_lock = threading.Lock()

#: Status codes worth a second attempt. Anything else is a caller error.
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

_BACKOFF_BASE_SECONDS = 0.5


def get_client() -> httpx.Client:
    """Return the process-wide HTTP client, building it on first use."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                settings = get_settings()
                _client = httpx.Client(
                    timeout=settings.http_timeout_seconds,
                    follow_redirects=True,
                    headers={"User-Agent": "Zenic/1.0 (+health-assistant)"},
                    # One retry budget in get_json covers transport and status failures.
                    transport=httpx.HTTPTransport(retries=0),
                )
    return _client


def reset_client() -> None:
    """Close and drop the cached client. For tests and clean shutdown."""
    global _client
    with _client_lock:
        if _client is not None:
            _client.close()
        _client = None


def get_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    service: str,
    timeout: float | None = None,
) -> dict[str, Any]:
    """GET ``url`` and return the decoded JSON body.

    ``service`` names the upstream in logs and error messages. Retries on
    connection errors, 429, and 5xx with jittered exponential backoff.

    Raises:
        ExternalAPIError: once the retry budget is exhausted, or on a
            non-retryable status, or if the body is not JSON.
    """
    settings = get_settings()
    client = get_client()
    attempts = settings.http_max_retries + 1
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        started = time.perf_counter()
        try:
            response = client.get(url, params=params, timeout=timeout if timeout is not None else settings.http_timeout_seconds)
        except httpx.HTTPError as exc:
            last_error = exc
            logger.warning(
                "http request failed",
                extra={"service": service, "attempt": attempt, "error_type": type(exc).__name__},
            )
            _sleep_backoff(attempt, attempts)
            continue

        elapsed_ms = round((time.perf_counter() - started) * 1000)
        if response.status_code in _RETRYABLE_STATUS:
            last_error = httpx.HTTPStatusError(
                f"HTTP {response.status_code}", request=response.request, response=response
            )
            logger.warning(
                "http retryable status",
                extra={
                    "service": service,
                    "attempt": attempt,
                    "status": response.status_code,
                    "elapsed_ms": elapsed_ms,
                },
            )
            _sleep_backoff(attempt, attempts)
            continue

        if response.is_error:
            logger.error(
                "http client error",
                extra={"service": service, "status": response.status_code},
            )
            raise ExternalAPIError(
                f"{service} returned HTTP {response.status_code}."
            )

        logger.debug(
            "http request complete",
            extra={"service": service, "status": response.status_code, "elapsed_ms": elapsed_ms},
        )
        try:
            payload = response.json()
        except ValueError as exc:
            raise ExternalAPIError(f"{service} returned a non-JSON response.") from exc
        if not isinstance(payload, dict):
            raise ExternalAPIError(
                f"{service} returned {type(payload).__name__}, expected a JSON object."
            )
        return payload

    raise ExternalAPIError(
        f"{service} is unreachable after {attempts} attempts."
    ) from last_error


def _sleep_backoff(attempt: int, attempts: int) -> None:
    """Sleep before the next attempt — skipped after the final one."""
    if attempt >= attempts:
        return
    delay = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
    time.sleep(delay + random.uniform(0, delay * 0.25))
