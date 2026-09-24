"""Determine workout split and retrieve exercises from the indexed wger data."""
from __future__ import annotations

from zenic.agent.state import ZenicState
from zenic.errors import RetrievalError
from zenic.logging_config import get_logger
from zenic.rag.pipeline import retrieve

logger = get_logger(__name__)

_SPLIT_MAP = {
    3: "full_body",
    4: "upper_lower",
    5: "ulppl",
    6: "ppl",
}

_DEFAULT_SPLIT = "full_body"
_DEFAULT_DAYS = 3


def _select_split(available_days: int, goal: str) -> str:
    """Pick a training split from weekly availability and goal.

    A cutting goal always takes the conditioning-oriented split. (This branch
    previously tested for the string "fat_loss", which the profile normaliser
    canonicalises to "cutting" — so it could never fire.)
    """
    if goal == "cutting":
        return "full_body_cardio"
    try:
        days = int(available_days)
    except (TypeError, ValueError):
        days = _DEFAULT_DAYS
    return _SPLIT_MAP.get(min(max(days, 3), 6), _DEFAULT_SPLIT)


def run(state: ZenicState) -> dict:
    profile = state.get("user_profile") or {}
    goal = profile.get("goal", "maintenance")
    split = _select_split(profile.get("available_days", _DEFAULT_DAYS), goal)
    equipment = profile.get("equipment", "barbell")

    query = f"{split} workout exercises {equipment} {goal}".strip()
    try:
        chunks = retrieve(query)
    except RetrievalError:
        logger.error("exercise retrieval failed — no plan will be generated without evidence")
        chunks = []

    logger.info("exercises retrieved", extra={"split": split, "chunks": len(chunks)})
    return {
        "tool_results": {
            **(state.get("tool_results") or {}),
            "split_type": split,
            "exercise_chunks": chunks,
        }
    }
