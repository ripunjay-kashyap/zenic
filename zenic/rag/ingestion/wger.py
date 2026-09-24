"""
wger exercise database ingestion.
Source: https://wger.de/api/v2/exerciseinfo/ (open REST API, no key needed)

Strategy: paginate through all English exercises using exerciseinfo endpoint
(resolves category, equipment, and muscle names inline — no separate ID lookup).
One chunk per exercise. No overlap.
Metadata: muscle_group, muscles_secondary, equipment, category, source="wger"
"""
import re
import time

import httpx

from zenic.logging_config import get_logger

logger = get_logger(__name__)

_BASE = "https://wger.de/api/v2"
_LANGUAGE_ENGLISH = 2
_PAGE_SIZE = 100
_RETRY_DELAY = 2  # seconds between retries on rate limit


def fetch_exercises(page_size: int = _PAGE_SIZE) -> list[dict]:
    """
    Fetch all English exercises from wger exerciseinfo endpoint.
    Returns raw API results (list of exerciseinfo objects).
    """
    results = []
    url = f"{_BASE}/exerciseinfo/?format=json&language={_LANGUAGE_ENGLISH}&limit={page_size}"

    with httpx.Client(timeout=30) as client:
        while url:
            resp = client.get(url)
            if resp.status_code == 429:
                logger.info(f"  Rate limited — waiting {_RETRY_DELAY}s...")
                time.sleep(_RETRY_DELAY)
                resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()
            results.extend(data["results"])
            url = data.get("next")
            logger.info(f"  Fetched {len(results)} exercises...")

    return results


def _extract_english_translation(exercise: dict) -> tuple[str, str]:
    """Return (name, description) for the English translation, or ('', '') if absent."""
    for t in exercise.get("translations", []):
        if t.get("language") == _LANGUAGE_ENGLISH:
            name = t.get("name", "").strip()
            desc = re.sub(r"<[^>]+>", " ", t.get("description", "")).strip()
            desc = re.sub(r"\s+", " ", desc)
            return name, desc
    return "", ""


def _format_exercise_doc(exercise: dict) -> dict | None:
    """Convert a raw exerciseinfo object into an indexable document. Returns None if no English name."""
    name, description = _extract_english_translation(exercise)
    if not name:
        return None

    ex_id = exercise["id"]
    category = exercise.get("category", {}).get("name", "")
    muscles = [m.get("name_en") or m.get("name", "") for m in exercise.get("muscles", [])]
    muscles_secondary = [m.get("name_en") or m.get("name", "") for m in exercise.get("muscles_secondary", [])]
    equipment = [e.get("name", "") for e in exercise.get("equipment", [])]

    # Build human-readable chunk text
    parts = [f"Exercise: {name}"]
    if category:
        parts.append(f"Category: {category}")
    if muscles:
        parts.append(f"Primary muscles: {', '.join(muscles)}")
    if muscles_secondary:
        parts.append(f"Secondary muscles: {', '.join(muscles_secondary)}")
    if equipment:
        parts.append(f"Equipment: {', '.join(equipment)}")
    if description:
        parts.append(f"Description: {description}")

    text = "\n".join(parts)

    return {
        "id": f"wger_{ex_id}",
        "text": text,
        "metadata": {
            "source": "wger",
            "exercise_id": ex_id,
            "name": name,
            "category": category,
            "muscle_group": ", ".join(muscles) if muscles else "",
            "muscles_secondary": ", ".join(muscles_secondary),
            "equipment": ", ".join(equipment) if equipment else "bodyweight",
        },
    }


def ingest_wger_exercises() -> list[dict]:
    """
    Fetch all wger exercises, format them as documents, and return the list.
    Call indexer.index_documents() on the result to embed and persist.
    """
    logger.info("Fetching wger exercises...")
    raw = fetch_exercises()
    docs = [_format_exercise_doc(ex) for ex in raw]
    docs = [d for d in docs if d is not None]
    logger.info(f"Prepared {len(docs)} exercise documents (from {len(raw)} raw)")
    return docs
