"""Run the agent with tool-call tracing for Pillar 2 evaluation."""
from __future__ import annotations

import time
import uuid
from typing import Any

from zenic.agent.graph import app, initial_state
from zenic.logging_config import bind_correlation_id, get_logger

logger = get_logger(__name__)

#: Fields whose per-node partials must be merged rather than replaced, so the
#: final state carries everything the run produced.
_MERGED_DICT_FIELDS = ("tool_results", "plan_data", "user_profile")


def run_with_trace(query: str, user_profile: dict | None = None) -> dict:
    """
    Execute the graph and return the final state plus a list of nodes visited.
    Used by Pillar 2 test suite to verify tool_call sequences.

    tools_called contains the LangGraph node names in visit order, plus any
    virtual tool names injected from state (e.g. "usda_api" when the RAG
    fallback triggers) so that rag_vs_api_check.py can detect them.

    The returned final_state is the *accumulated* state across every node. The
    previous implementation returned only the last node's partial output, so
    fields set earlier in the run (intent, retrieved_context, user_profile)
    were missing from it.
    """
    trace_id = str(uuid.uuid4())
    bind_correlation_id(trace_id[:12])
    started = time.perf_counter()

    state: dict[str, Any] = dict(
        initial_state([{"role": "user", "content": query}], user_profile)
    )
    nodes_visited: list[str] = []

    for step in app.stream(state):
        for node_name, partial in step.items():
            nodes_visited.append(node_name)
            _merge_partial(state, partial)

    # Expose API fallback as a virtual tool name so callers can detect it with
    # a simple `"usda_api" in tools_called` check (mirrors node-name convention).
    api_fallback = (state.get("tool_results") or {}).get("api_fallback_used")
    if api_fallback:
        nodes_visited.append(api_fallback)

    logger.info(
        "trace complete",
        extra={
            "nodes": nodes_visited,
            "intent": state.get("intent"),
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        },
    )
    return {
        "id": trace_id,
        "tools_called": nodes_visited,
        "final_state": state,
    }


def _merge_partial(state: dict[str, Any], partial: Any) -> None:
    """Fold one node's returned fields into the accumulated state."""
    if not isinstance(partial, dict):
        return
    for key, value in partial.items():
        if key == "messages":
            state["messages"] = list(state.get("messages") or []) + _as_list(value)
        elif key in _MERGED_DICT_FIELDS and isinstance(value, dict):
            state[key] = {**(state.get(key) or {}), **value}
        else:
            state[key] = value


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else [value]
