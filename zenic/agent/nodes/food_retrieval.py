"""Retrieve foods matching macro targets and dietary restrictions."""
from __future__ import annotations

from zenic.agent.state import ZenicState
from zenic.errors import RetrievalError
from zenic.logging_config import get_logger
from zenic.rag.pipeline import retrieve

logger = get_logger(__name__)


def run(state: ZenicState) -> dict:
    profile = state.get("user_profile") or {}
    tool_results = state.get("tool_results") or {}

    restrictions = profile.get("dietary_restrictions", "none")
    goal = profile.get("goal", "maintenance")
    # Populated by the calculator node, which the meal_plan route runs first.
    protein_target = tool_results.get("protein_min_g")

    query = f"high protein foods {restrictions} diet {goal}"
    if protein_target:
        query += f" {protein_target}g protein"

    try:
        chunks = retrieve(query)
    except RetrievalError:
        logger.error("food retrieval failed — no plan will be generated without evidence")
        chunks = []

    logger.info("foods retrieved", extra={"chunks": len(chunks), "goal": goal})
    return {"tool_results": {**tool_results, "food_chunks": chunks}}
