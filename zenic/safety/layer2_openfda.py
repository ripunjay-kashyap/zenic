"""
Layer 2: OpenFDA adverse event lookup.
Research utility, not wired into the graph. Adverse-event reports cannot
establish clinical safety or causality.
"""
from __future__ import annotations

import re

from zenic.config import get_settings
from zenic.errors import ExternalAPIError
from zenic.http_client import get_json
from zenic.logging_config import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://api.fda.gov/drug/event.json"

_MAX_REACTIONS = 5


def check_substance(substance: str) -> dict:
    """
    Returns descriptive adverse-event data; never infers clinical safety.

    Raises:
        ExternalAPIError: if OpenFDA is unreachable — callers should treat an
            unavailable lookup as "unknown", not as "safe".
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ()-]{0,99}", substance):
        raise ValueError("Use a plain substance name of at most 100 characters")
    params = {
        "search": f'patient.drug.medicinalproduct:"{substance}"',
        "count": "patient.reaction.reactionmeddrapt.exact",
        "limit": str(_MAX_REACTIONS),
    }
    api_key = get_settings().openfda_api_key
    if api_key:
        params["api_key"] = api_key

    try:
        data = get_json(_BASE_URL, params=params, service="OpenFDA")
    except ExternalAPIError:
        logger.warning("openfda lookup failed", extra={"service": "OpenFDA"})
        raise

    results = data.get("results") or []
    total_events = int(((data.get("meta") or {}).get("results") or {}).get("total", 0))
    top_reactions = [r["term"] for r in results[:_MAX_REACTIONS] if isinstance(r, dict) and "term" in r]

    return {
        "safe": None,
        "adverse_event_count": total_events,
        "top_reactions": top_reactions,
    }
