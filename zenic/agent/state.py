from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class ZenicState(TypedDict):
    # Conversation
    messages: Annotated[list, add_messages]

    # User profile (populated progressively through profile_gather)
    user_profile: dict  # weight_kg, height_cm, age, gender, activity_level, goal, dietary_restrictions, experience_level, available_days, equipment

    # Routing
    intent: str  # nutrition_qa | calculate | meal_plan | workout_plan | weekly_summary | general_chat

    # Profile completeness
    profile_complete: bool
    missing_fields: list[str]
    awaiting_input: bool

    # Retrieval
    retrieved_context: list  # list of chunks with source metadata
    retrieval_query: str  # standalone wording for referential follow-ups

    # Calculation and tool results
    tool_results: dict  # bmr, tdee, macros, protein_range, split_type, exercises, weekly_stats, etc.

    # Plan output (structured JSON, pre-PDF)
    plan_data: dict

    # Safety
    safety_flag: bool
    safety_reason: str
