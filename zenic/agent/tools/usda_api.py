"""
USDA FoodData Central live API — fallback when item not found in local index.
Only called when RAG retrieval returns no relevant results for a food query.
"""
from __future__ import annotations

from typing import Any

from zenic.config import get_settings
from zenic.http_client import get_json
from zenic.logging_config import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://api.nal.usda.gov/fdc/v1"
_MAX_NUTRIENTS = 10


def search_food(query: str, page_size: int = 5) -> list[dict]:
    """Return foods with nutrient summaries.

    Raises:
        ConfigError: if USDA_API_KEY is not set.
        ExternalAPIError: if the API is unreachable or returns an error.
    """
    payload = get_json(
        f"{_BASE_URL}/foods/search",
        params={
            "query": query,
            "pageSize": page_size,
            "api_key": get_settings().require_usda_api_key(),
        },
        service="USDA FoodData Central",
    )
    items = payload.get("foods") or []
    logger.debug("usda search complete", extra={"results": len(items)})
    return [_to_food(item) for item in items if isinstance(item, dict)]


def _to_food(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": item.get("description"),
        "fdcId": item.get("fdcId"),
        "nutrients": _to_nutrients(item.get("foodNutrients") or []),
    }


def _to_nutrients(raw: list[Any]) -> dict[str, str]:
    """Format the first few nutrients as 'name: value unit'.

    Entries missing a name, value, or unit are skipped — the API omits them for
    some records, and the previous dict comprehension raised KeyError on those.
    """
    nutrients: dict[str, str] = {}
    for entry in raw[:_MAX_NUTRIENTS]:
        if not isinstance(entry, dict):
            continue
        name = entry.get("nutrientName")
        value = entry.get("value")
        if not name or value is None:
            continue
        unit = entry.get("unitName", "")
        nutrients[name] = f"{value} {unit}".strip()
    return nutrients
