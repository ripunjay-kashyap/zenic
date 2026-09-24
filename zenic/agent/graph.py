"""
LangGraph StateGraph for Zenic.

Entry point: safety_check
Workflows:
  nutrition_qa     → rag_retrieval → generate → END
  calculate        → profile_check → [profile_gather → END | calculator → generate → END]
  meal_plan        → profile_check → [profile_gather → END | calculator → food_retrieval → plan_compose → pdf_generate → END]
  workout_plan     → profile_check → [profile_gather → END | exercise_retrieval → plan_compose → pdf_generate → END]
  weekly_summary   → data_ingestion → trend_analysis → insight_generation → pdf_generate → END
  general_chat     → generate → END

meal_plan runs the calculator before retrieval so that plan_compose has real
macro targets; without it the composer was prompted with empty TDEE and macro
values and invented its own.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from zenic.agent.nodes import (
    calculator,
    data_ingestion,
    exercise_retrieval,
    food_retrieval,
    generate,
    insight_generation,
    pdf_generate,
    plan_compose,
    profile_check,
    profile_gather,
    rag_retrieval,
    router,
    safety_check,
    safety_response,
    trend_analysis,
)
from zenic.agent.state import ZenicState

#: Intents that require a complete user profile before any work is done.
_PROFILE_GATED_INTENTS = ("calculate", "meal_plan", "workout_plan")


def _route_after_safety(state: ZenicState) -> str:
    return "safety_response" if state.get("safety_flag") else "router"


def _route_after_router(state: ZenicState) -> str:
    intent = state.get("intent", "general_chat")
    if intent == "nutrition_qa":
        return "rag_retrieval"
    if intent in _PROFILE_GATED_INTENTS:
        return "profile_check"
    if intent == "weekly_summary":
        return "data_ingestion"
    return "generate"  # general_chat


def _route_after_profile_check(state: ZenicState) -> str:
    if not state.get("profile_complete"):
        return "profile_gather"
    intent = state.get("intent")
    # meal_plan shares the calculator with calculate — it needs TDEE and macro
    # targets before it can pick foods.
    if intent in ("calculate", "meal_plan"):
        return "calculator"
    if intent == "workout_plan":
        return "exercise_retrieval"
    return "generate"


def _route_after_calculator(state: ZenicState) -> str:
    return "food_retrieval" if state.get("intent") == "meal_plan" else "generate"


def build_graph() -> StateGraph:
    g = StateGraph(ZenicState)

    g.add_node("safety_check",       safety_check.run)
    g.add_node("router",             router.run)
    g.add_node("profile_check",      profile_check.run)
    g.add_node("profile_gather",     profile_gather.run)
    g.add_node("rag_retrieval",      rag_retrieval.run)
    g.add_node("calculator",         calculator.run)
    g.add_node("exercise_retrieval", exercise_retrieval.run)
    g.add_node("food_retrieval",     food_retrieval.run)
    g.add_node("plan_compose",       plan_compose.run)
    g.add_node("pdf_generate",       pdf_generate.run)
    g.add_node("data_ingestion",     data_ingestion.run)
    g.add_node("trend_analysis",     trend_analysis.run)
    g.add_node("insight_generation", insight_generation.run)
    g.add_node("generate",           generate.run)
    g.add_node("safety_response",    safety_response.run)

    g.set_entry_point("safety_check")

    g.add_conditional_edges("safety_check", _route_after_safety, {
        "safety_response": "safety_response",
        "router":          "router",
    })
    g.add_conditional_edges("router", _route_after_router, {
        "rag_retrieval":  "rag_retrieval",
        "profile_check":  "profile_check",
        "data_ingestion": "data_ingestion",
        "generate":       "generate",
    })
    g.add_conditional_edges("profile_check", _route_after_profile_check, {
        "profile_gather":     "profile_gather",
        "calculator":         "calculator",
        "exercise_retrieval": "exercise_retrieval",
        "generate":           "generate",
    })
    g.add_conditional_edges("calculator", _route_after_calculator, {
        "food_retrieval": "food_retrieval",
        "generate":       "generate",
    })

    g.add_edge("rag_retrieval",      "generate")
    g.add_edge("food_retrieval",     "plan_compose")
    g.add_edge("exercise_retrieval", "plan_compose")
    g.add_edge("plan_compose",       "pdf_generate")
    g.add_edge("data_ingestion",     "trend_analysis")
    g.add_edge("trend_analysis",     "insight_generation")
    g.add_edge("insight_generation", "pdf_generate")
    g.add_edge("generate",           END)
    g.add_edge("pdf_generate",       END)
    g.add_edge("profile_gather",     END)
    g.add_edge("safety_response",    END)

    return g


app = build_graph().compile()


def initial_state(
    messages: list[dict] | None = None,
    user_profile: dict | None = None,
    pending_intent: str | None = None,
    pending_missing_fields: list[str] | None = None,
) -> ZenicState:
    """Build a fully-populated starting state.

    Every caller (UI, trace runner, scripts) previously hand-rolled this dict,
    so adding a state field meant editing three places and silently breaking any
    that were missed.
    """
    return {
        "messages": messages or [],
        "user_profile": user_profile or {},
        "intent": pending_intent or "",
        "profile_complete": False,
        "missing_fields": pending_missing_fields or [],
        "awaiting_input": bool(pending_intent),
        "retrieved_context": [],
        "tool_results": {},
        "plan_data": {},
        "safety_flag": False,
        "safety_reason": "",
    }
