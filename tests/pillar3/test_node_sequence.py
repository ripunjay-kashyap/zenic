"""
Tool-call / node-sequence integration tests — Pillar 3.

Verifies that the LangGraph graph routes each intent through the correct node
sequence without making any real LLM or API calls. All node run() functions are
replaced with lightweight mocks that record their name and return the state keys
needed to drive the conditional routing logic.

No API keys required. These tests are fast (~1–2 s total).
"""
from contextlib import ExitStack
from unittest.mock import patch

import pytest

import zenic.agent.nodes.calculator as _calculator
import zenic.agent.nodes.data_ingestion as _data_ingestion
import zenic.agent.nodes.exercise_retrieval as _exercise_retrieval
import zenic.agent.nodes.food_retrieval as _food_retrieval
import zenic.agent.nodes.generate as _generate
import zenic.agent.nodes.insight_generation as _insight_generation
import zenic.agent.nodes.pdf_generate as _pdf_generate
import zenic.agent.nodes.plan_compose as _plan_compose
import zenic.agent.nodes.profile_check as _profile_check
import zenic.agent.nodes.profile_gather as _profile_gather
import zenic.agent.nodes.rag_retrieval as _rag_retrieval
import zenic.agent.nodes.router as _router
import zenic.agent.nodes.safety_check as _safety_check
import zenic.agent.nodes.safety_response as _safety_response
import zenic.agent.nodes.trend_analysis as _trend_analysis

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Maps module objects → their node name in the graph
_MODULE_TO_NAME = {
    _safety_check:      "safety_check",
    _router:            "router",
    _profile_check:     "profile_check",
    _profile_gather:    "profile_gather",
    _rag_retrieval:     "rag_retrieval",
    _calculator:        "calculator",
    _food_retrieval:    "food_retrieval",
    _exercise_retrieval: "exercise_retrieval",
    _plan_compose:      "plan_compose",
    _pdf_generate:      "pdf_generate",
    _data_ingestion:    "data_ingestion",
    _trend_analysis:    "trend_analysis",
    _insight_generation: "insight_generation",
    _generate:          "generate",
    _safety_response:   "safety_response",
}

# Minimal valid initial state
_BASE_STATE = {
    "messages": [],
    "user_profile": {},
    "intent": "",
    "profile_complete": False,
    "missing_fields": [],
    "awaiting_input": False,
    "retrieved_context": [],
    "tool_results": {},
    "plan_data": {},
    "safety_flag": False,
    "safety_reason": "",
}


def _invoke_with_mocks(node_returns: dict) -> list[str]:
    """
    Build a fresh LangGraph app with mocked node run() functions, invoke it,
    and return the list of node names that were called in order.

    node_returns maps module object → dict to return from that node's run().
    The return dict must include the state keys that drive routing
    (e.g. {"safety_flag": False}, {"intent": "nutrition_qa"}).
    """
    call_order: list[str] = []

    def _make_mock(name: str, retval: dict):
        def _mock(_state):
            call_order.append(name)
            return retval
        return _mock

    with ExitStack() as stack:
        for mod, retval in node_returns.items():
            name = _MODULE_TO_NAME[mod]
            stack.enter_context(patch.object(mod, "run", _make_mock(name, retval)))

        # Build the graph inside the patched context so add_node() picks up mocks
        from zenic.agent.graph import build_graph
        app = build_graph().compile()
        app.invoke(dict(_BASE_STATE))

    return call_order


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,node_returns,expected_sequence", [
    (
        "nutrition_qa",
        {
            _safety_check:  {"safety_flag": False},
            _router:        {"intent": "nutrition_qa"},
            _rag_retrieval: {"retrieved_context": []},
            _generate:      {},
        },
        ["safety_check", "router", "rag_retrieval", "generate"],
    ),
    (
        "calculate_complete_profile",
        {
            _safety_check:  {"safety_flag": False},
            _router:        {"intent": "calculate"},
            _profile_check: {"profile_complete": True},
            _calculator:    {"tool_results": {}},
            _generate:      {},
        },
        ["safety_check", "router", "profile_check", "calculator", "generate"],
    ),
    (
        "calculate_incomplete_profile",
        {
            _safety_check:  {"safety_flag": False},
            _router:        {"intent": "calculate"},
            _profile_check: {"profile_complete": False, "missing_fields": ["age"]},
            _profile_gather: {},
        },
        ["safety_check", "router", "profile_check", "profile_gather"],
    ),
    (
        # meal_plan runs the calculator first so plan_compose gets real macro targets.
        "meal_plan_complete_profile",
        {
            _safety_check:   {"safety_flag": False},
            _router:         {"intent": "meal_plan"},
            _profile_check:  {"profile_complete": True},
            _calculator:     {"tool_results": {"tdee": 2400}},
            _food_retrieval: {"retrieved_context": []},
            _plan_compose:   {"plan_data": {}},
            _pdf_generate:   {},
        },
        [
            "safety_check", "router", "profile_check", "calculator",
            "food_retrieval", "plan_compose", "pdf_generate",
        ],
    ),
    (
        "workout_plan_complete_profile",
        {
            _safety_check:       {"safety_flag": False},
            _router:             {"intent": "workout_plan"},
            _profile_check:      {"profile_complete": True},
            _exercise_retrieval: {"retrieved_context": []},
            _plan_compose:       {"plan_data": {}},
            _pdf_generate:       {},
        },
        ["safety_check", "router", "profile_check", "exercise_retrieval", "plan_compose", "pdf_generate"],
    ),
    (
        "weekly_summary",
        {
            _safety_check:      {"safety_flag": False},
            _router:            {"intent": "weekly_summary"},
            _data_ingestion:    {"tool_results": {}},
            _trend_analysis:    {"tool_results": {}},
            _insight_generation: {"plan_data": {}},
            _pdf_generate:      {},
        },
        ["safety_check", "router", "data_ingestion", "trend_analysis", "insight_generation", "pdf_generate"],
    ),
    (
        "general_chat",
        {
            _safety_check: {"safety_flag": False},
            _router:       {"intent": "general_chat"},
            _generate:     {},
        },
        ["safety_check", "router", "generate"],
    ),
    (
        "safety_blocked",
        {
            _safety_check:   {"safety_flag": True, "safety_reason": "harmful content"},
            _safety_response: {},
        },
        ["safety_check", "safety_response"],
    ),
])
def test_node_sequence(label, node_returns, expected_sequence):
    actual = _invoke_with_mocks(node_returns)
    assert actual == expected_sequence, (
        f"[{label}] Expected sequence:\n  {expected_sequence}\n"
        f"Got:\n  {actual}"
    )
