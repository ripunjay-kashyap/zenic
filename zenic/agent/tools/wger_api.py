"""
wger exercise database live API — fallback when exercise not found in local index.
"""
from __future__ import annotations

from typing import Any

from zenic.http_client import get_json
from zenic.logging_config import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://wger.de/api/v2"
_ENGLISH = 2


def search_exercises(
    muscle_group: str | None = None,
    equipment: str | None = None,
    language: int = _ENGLISH,
) -> list[dict]:
    """Return exercises matching the filters. language=2 is English.

    Raises:
        ExternalAPIError: if the API is unreachable or returns an error.
    """
    params: dict[str, Any] = {"language": language, "format": "json"}
    if muscle_group:
        params["muscles"] = muscle_group
    if equipment:
        params["equipment"] = equipment

    payload = get_json(f"{_BASE_URL}/exercise/", params=params, service="wger")
    results = payload.get("results") or []
    logger.debug("wger search complete", extra={"results": len(results)})
    return [
        {
            "name": ex.get("name"),
            "description": ex.get("description", ""),
            "muscles": ex.get("muscles", []),
            "equipment": ex.get("equipment", []),
        }
        for ex in results
        if isinstance(ex, dict)
    ]
