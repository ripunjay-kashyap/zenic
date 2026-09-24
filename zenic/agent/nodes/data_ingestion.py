"""Load weekly tracking data (V1: seeded JSON file; V2: database query)."""
from __future__ import annotations

import json
from datetime import date, timedelta

from zenic.agent.state import ZenicState
from zenic.config import PROJECT_ROOT
from zenic.logging_config import get_logger

logger = get_logger(__name__)

_DATA_PATH = PROJECT_ROOT / "data" / "mock_weekly.json"
_WEEK_DAYS = 7


def run(state: ZenicState) -> dict:
    entries = _load_entries()
    weekly = _select_week(entries)
    logger.info("weekly data loaded", extra={"days": len(weekly), "source_entries": len(entries)})
    return {"tool_results": {**(state.get("tool_results") or {}), "weekly_data": weekly}}


def _load_entries() -> list[dict]:
    """Read the tracking file. Returns [] if it is missing or unreadable."""
    try:
        with open(_DATA_PATH, encoding="utf-8") as f:
            entries = json.load(f)
    except FileNotFoundError:
        logger.info("no tracking data file", extra={"path": str(_DATA_PATH)})
        return []
    except (OSError, json.JSONDecodeError):
        logger.error("tracking data file is unreadable", extra={"path": str(_DATA_PATH)})
        return []

    if not isinstance(entries, list):
        logger.error("tracking data file is not a list of entries")
        return []
    return [e for e in entries if isinstance(e, dict)]


def _select_week(entries: list[dict]) -> list[dict]:
    """Return the last seven days of entries.

    Prefers the trailing calendar week. The bundled demo dataset is fixed-date,
    so once it ages out of that window the strict filter returns nothing and the
    summary comes back empty; in that case fall back to the most recent seven
    entries in the file so the workflow still demonstrates end to end.
    """
    if not entries:
        return _placeholder_week()

    end = date.today()
    start = end - timedelta(days=_WEEK_DAYS - 1)
    current = [
        e for e in entries if start.isoformat() <= str(e.get("date", "")) <= end.isoformat()
    ]
    if current:
        return current

    logger.info("no entries in the trailing week — using the most recent recorded week")
    return sorted(entries, key=lambda e: str(e.get("date", "")))[-_WEEK_DAYS:]


def _placeholder_week() -> list[dict]:
    start = date.today() - timedelta(days=_WEEK_DAYS - 1)
    return [
        {
            "date": (start + timedelta(days=i)).isoformat(),
            "calories": 0,
            "protein_g": 0,
            "workout_planned": False,
            "workout_completed": False,
        }
        for i in range(_WEEK_DAYS)
    ]
