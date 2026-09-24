"""Run deterministic nutrition calculations from user_profile."""
from __future__ import annotations

from zenic.agent.nodes.profile_check import REQUIRED_FIELDS
from zenic.agent.state import ZenicState
from zenic.agent.tools.calculations import (
    calculate_bmr,
    calculate_macros,
    calculate_protein_range,
    calculate_tdee,
)
from zenic.errors import ZenicError
from zenic.logging_config import get_logger

logger = get_logger(__name__)


def run(state: ZenicState) -> dict:
    profile = state.get("user_profile") or {}
    # The graph only routes here once profile_check reports completeness, but a
    # direct invocation (tests, scripts) can arrive with gaps — fail with a
    # message that names the missing field instead of a bare KeyError.
    missing = [f for f in REQUIRED_FIELDS["calculate"] if profile.get(f) in (None, "")]
    if missing:
        raise ZenicError(
            f"Cannot calculate: the profile is missing {', '.join(missing)}."
        )

    if float(profile.get("age", 0)) < 18:
        raise ZenicError("This calculator supports adults only. Please consult a qualified clinician for younger people.")
    if profile.get("gender") not in ("male", "female"):
        raise ZenicError("This equation requires a male or female physiology coefficient; I cannot safely infer one from gender identity.")


    bmr = calculate_bmr(
        profile["weight_kg"], profile["height_cm"], profile["age"], profile["gender"]
    )
    tdee = calculate_tdee(bmr, profile["activity_level"])
    macros = calculate_macros(tdee, profile["goal"])
    protein = calculate_protein_range(profile["weight_kg"], profile["goal"])

    logger.info("metrics calculated")
    return {
        "tool_results": {
            **(state.get("tool_results") or {}),
            "bmr": bmr,
            "tdee": tdee,
            **macros,
            "protein_min_g": protein["min_g"],
            "protein_max_g": protein["max_g"],
        }
    }
