"""Run one graph turn and report progress without depending on a UI framework."""
from __future__ import annotations

import time
from collections.abc import Callable

from zenic.agent.graph import app, initial_state
from zenic.agent.messages import message_content, message_role
from zenic.errors import ZenicError
from zenic.logging_config import bind_correlation_id, get_logger

logger = get_logger(__name__)

NODE_LABELS = {
    "safety_check": "Checking your message",
    "router": "Understanding your question",
    "profile_check": "Reviewing your profile",
    "profile_gather": "Working out what's missing",
    "rag_retrieval": "Searching the knowledge base",
    "calculator": "Running your numbers",
    "food_retrieval": "Finding foods that fit",
    "exercise_retrieval": "Selecting exercises",
    "plan_compose": "Composing your plan",
    "pdf_generate": "Building your PDF",
    "data_ingestion": "Loading your week",
    "trend_analysis": "Analysing trends",
    "insight_generation": "Drawing out insights",
    "generate": "Writing the answer",
    "safety_response": "Preparing a response",
}

AFTER_ROUTER = {
    "nutrition_qa": NODE_LABELS["rag_retrieval"],
    "calculate": NODE_LABELS["profile_check"],
    "meal_plan": NODE_LABELS["profile_check"],
    "workout_plan": NODE_LABELS["profile_check"],
    "weekly_summary": NODE_LABELS["data_ingestion"],
    "general_chat": NODE_LABELS["generate"],
}


def _merge(state: dict, partial: object) -> None:
    if not isinstance(partial, dict):
        return
    for key, value in partial.items():
        if key == "messages":
            state[key] = list(state.get(key) or []) + (value if isinstance(value, list) else [value])
        elif key in ("tool_results", "plan_data", "user_profile") and isinstance(value, dict):
            state[key] = {**(state.get(key) or {}), **value}
        else:
            state[key] = value


def _next_label(node: str, state: dict) -> str:
    if node == "router":
        return AFTER_ROUTER.get(state.get("intent", ""), "Working on it")
    if node in ("rag_retrieval", "calculator", "plan_compose"):
        return NODE_LABELS["generate"]
    if node in ("food_retrieval", "exercise_retrieval"):
        return NODE_LABELS["plan_compose"]
    return NODE_LABELS.get(node, "Working on it")


def extract_reply(state: dict) -> str:
    for message in reversed(state.get("messages") or []):
        if message_role(message) == "assistant":
            return message_content(message)
    return "Sorry, I couldn't generate a response."


def run_turn(
    messages: list[dict[str, str]],
    profile: dict,
    *,
    pending_intent: str | None = None,
    pending_missing_fields: list[str] | None = None,
    on_stage: Callable[[str, int], None] | None = None,
) -> tuple[dict | None, str | None, dict[str, int]]:
    """Return final state, a safe error (if any), and per-node elapsed milliseconds."""
    correlation_id = bind_correlation_id()
    state = initial_state(messages[-12:], profile, pending_intent, pending_missing_fields)
    accumulated = dict(state)
    timings: dict[str, int] = {}
    started = time.perf_counter()
    last_node_at = started
    try:
        for step in app.stream(state):
            for node, partial in step.items():
                now = time.perf_counter()
                timings[node] = round((now - last_node_at) * 1000)
                last_node_at = now
                _merge(accumulated, partial)
                if on_stage:
                    on_stage(_next_label(node, accumulated), round((now - started) * 1000))
        timings["total"] = round((time.perf_counter() - started) * 1000)
        return accumulated, None, timings
    except ZenicError as exc:
        logger.warning("turn failed", extra={"error_type": type(exc).__name__})
        return None, str(exc), timings
    except Exception as exc:
        logger.error("unhandled error during turn", extra={"error_type": type(exc).__name__})
        return None, f"Something went wrong. Please try again. Reference: {correlation_id}", timings
