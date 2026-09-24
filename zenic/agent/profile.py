"""User profile schema, validation, and normalisation.

The profile is populated by an LLM extraction step, so its values cannot be
trusted to match the enums the deterministic calculators expect. Previously an
extraction of ``activity_level="moderately active"`` passed the completeness
check and then raised ``ValueError`` inside ``calculate_tdee`` — an unhandled
crash mid-conversation. Everything written into the profile now goes through
:func:`normalize_profile`, which coerces what it can, drops what it cannot, and
reports why.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from zenic.logging_config import get_logger

logger = get_logger(__name__)

ACTIVITY_LEVELS = ("sedentary", "light", "moderate", "active", "very_active")
GOALS = ("maintenance", "cutting", "bulking")
EXPERIENCE_LEVELS = ("beginner", "intermediate", "advanced")
GENDERS = ("male", "female", "other")

#: Free-text an LLM plausibly returns → the canonical enum value.
_ACTIVITY_SYNONYMS = {
    "sedentary": "sedentary",
    "inactive": "sedentary",
    "none": "sedentary",
    "light": "light",
    "lightly_active": "light",
    "lightly active": "light",
    "moderate": "moderate",
    "moderately_active": "moderate",
    "moderately active": "moderate",
    "active": "active",
    "very_active": "very_active",
    "very active": "very_active",
    "extremely_active": "very_active",
    "extremely active": "very_active",
    "athlete": "very_active",
}

_GOAL_SYNONYMS = {
    "maintenance": "maintenance",
    "maintain": "maintenance",
    "recomp": "maintenance",
    "cutting": "cutting",
    "cut": "cutting",
    "fat_loss": "cutting",
    "fat loss": "cutting",
    "weight_loss": "cutting",
    "weight loss": "cutting",
    "lose weight": "cutting",
    "bulking": "bulking",
    "bulk": "bulking",
    "muscle_gain": "bulking",
    "muscle gain": "bulking",
    "gain weight": "bulking",
    "hypertrophy": "bulking",
}

_EXPERIENCE_SYNONYMS = {
    "beginner": "beginner",
    "novice": "beginner",
    "new": "beginner",
    "intermediate": "intermediate",
    "advanced": "advanced",
    "expert": "advanced",
}

_GENDER_SYNONYMS = {
    "male": "male",
    "m": "male",
    "man": "male",
    "female": "female",
    "f": "female",
    "woman": "female",
    "other": "other",
    "non-binary": "other",
    "nonbinary": "other",
    "prefer not to say": "other",
}


class _Field:
    """One profile field: how to coerce a raw value, and what counts as valid."""

    def __init__(self, coerce: Callable[[Any], Any]):
        self.coerce = coerce


def _bounded_number(low: float, high: float, as_int: bool = False) -> Callable[[Any], Any]:
    def coerce(value: Any) -> Any:
        if isinstance(value, bool):  # bool is an int subclass — never a measurement
            raise ValueError("boolean is not a number")
        number = float(value)
        if not low <= number <= high:
            raise ValueError(f"outside the plausible range {low}–{high}")
        return round(number) if as_int else round(number, 1)

    return coerce


def _enum(synonyms: dict[str, str]) -> Callable[[Any], str]:
    def coerce(value: Any) -> str:
        key = str(value).strip().lower().replace("-", "_")
        if key in synonyms:
            return synonyms[key]
        spaced = key.replace("_", " ")
        if spaced in synonyms:
            return synonyms[spaced]
        raise ValueError(f"not one of {sorted(set(synonyms.values()))}")

    return coerce


def _free_text(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("empty")
    return text[:200]


#: Every field the profile may contain, and how to validate it.
FIELD_SCHEMA: dict[str, _Field] = {
    "weight_kg": _Field(_bounded_number(20, 500)),
    "height_cm": _Field(_bounded_number(50, 280)),
    "age": _Field(_bounded_number(1, 120, as_int=True)),
    "gender": _Field(_enum(_GENDER_SYNONYMS)),
    "activity_level": _Field(_enum(_ACTIVITY_SYNONYMS)),
    "goal": _Field(_enum(_GOAL_SYNONYMS)),
    "dietary_restrictions": _Field(_free_text),
    "experience_level": _Field(_enum(_EXPERIENCE_SYNONYMS)),
    "available_days": _Field(_bounded_number(1, 7, as_int=True)),
    "equipment": _Field(_free_text),
}

PROFILE_FIELDS = tuple(FIELD_SCHEMA)


def normalize_profile(raw: dict[str, Any]) -> dict[str, Any]:
    """Return a new profile containing only recognised, valid, canonical values.

    Unknown keys and values that cannot be coerced are dropped with a warning
    rather than propagated into the calculators.
    """
    clean: dict[str, Any] = {}
    for key, value in (raw or {}).items():
        field = FIELD_SCHEMA.get(key)
        if field is None:
            logger.debug("dropping unknown profile field", extra={"field": key})
            continue
        if value is None or value == "":
            continue
        try:
            clean[key] = field.coerce(value)
        except (TypeError, ValueError):
            logger.info(
                "dropping invalid profile value",
                extra={"field": key},
            )
    return clean


def merge_profile(existing: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Merge validated ``updates`` over ``existing`` without mutating either."""
    return {**normalize_profile(existing), **normalize_profile(updates)}
