"""Extract profile fields from the user's message and report what is still missing.

The extraction is best-effort: if the LLM is unavailable or returns nonsense,
the node still reports completeness from whatever profile it already had, and
the graph routes to profile_gather to ask the user directly.
"""
from __future__ import annotations

import json
from typing import Any

from zenic.agent.messages import last_user_message
from zenic.agent.profile import (
    ACTIVITY_LEVELS,
    EXPERIENCE_LEVELS,
    GOALS,
    merge_profile,
    normalize_profile,
)
from zenic.agent.state import ZenicState
from zenic.errors import LLMError
from zenic.llm import chat_completion_json
from zenic.logging_config import get_logger

logger = get_logger(__name__)

REQUIRED_FIELDS: dict[str, list[str]] = {
    "calculate":     ["weight_kg", "height_cm", "age", "gender", "activity_level", "goal"],
    "meal_plan":     ["weight_kg", "height_cm", "age", "gender", "activity_level", "goal", "dietary_restrictions"],
    "workout_plan":  ["goal", "experience_level", "available_days", "equipment"],
    "weekly_summary": [],
}

_EXTRACTION_PROMPT = (
    "Extract physical and fitness profile data from the user's message. "
    "Currently known profile: {profile}. "
    "Return a JSON object containing any NEW or UPDATED fields from this list: "
    "weight_kg (number), height_cm (number), age (number), "
    "gender (MUST BE exactly one of: {genders}), "
    "activity_level (MUST BE exactly one of: {activity_levels}), "
    "goal (MUST BE exactly one of: {goals}), "
    "dietary_restrictions (string), "
    "experience_level (MUST BE exactly one of: {experience_levels}), "
    "available_days (number), equipment (string). "
    "If none are found, return {{}}. User message: '{message}'"
)


def run(state: ZenicState) -> dict:
    intent = state.get("intent", "")
    profile = normalize_profile(state.get("user_profile") or {})
    required = REQUIRED_FIELDS.get(intent, [])

    message = last_user_message(state)
    if required and message:
        extracted = _extract_fields(message, profile)
        if extracted:
            profile = merge_profile(profile, extracted)

    missing = [field for field in required if not profile.get(field)]
    logger.info(
        "profile checked",
        extra={"intent": intent, "known_fields": sorted(profile), "missing_fields": missing},
    )
    return {
        "user_profile": profile,
        "profile_complete": not missing,
        "missing_fields": missing,
        "awaiting_input": bool(missing),
    }


def _extract_fields(message: str, profile: dict[str, Any]) -> dict[str, Any]:
    """Ask the LLM for profile fields present in ``message``. Never raises."""
    prompt = _EXTRACTION_PROMPT.format(
        profile=json.dumps(profile),
        genders="'male', 'female', 'other'",
        activity_levels=", ".join(repr(v) for v in ACTIVITY_LEVELS),
        goals=", ".join(repr(v) for v in GOALS),
        experience_levels=", ".join(repr(v) for v in EXPERIENCE_LEVELS),
        message=message,
    )
    try:
        return chat_completion_json(
            [{"role": "user", "content": prompt}], purpose="profile_extraction"
        )
    except LLMError:
        logger.warning("profile extraction unavailable — continuing with the known profile")
        return {}
