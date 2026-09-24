"""Router accuracy tests. Requires GROQ_API_KEY. Mark as integration tests."""
import pytest

pytestmark = pytest.mark.integration

from langchain_core.messages import HumanMessage

from zenic.agent.nodes.router import run as router_run

_CASES = [
    ("How much protein is in chicken breast?",        "nutrition_qa"),
    ("What foods are high in iron?",                  "nutrition_qa"),
    ("What's my TDEE? I'm 80kg moderate activity",    "calculate"),
    ("Calculate my daily protein target",             "calculate"),
    ("Make me a high-protein vegetarian meal plan",   "meal_plan"),
    ("Create a 7-day bulking meal plan",              "meal_plan"),
    ("Give me a PPL workout split for muscle gain",   "workout_plan"),
    ("Build me a 4-day upper lower routine",          "workout_plan"),
    ("Summarize my week",                             "weekly_summary"),
    ("Show me my weekly progress",                    "weekly_summary"),
    ("What can you help me with?",                    "general_chat"),
    ("Hey, how does this work?",                      "general_chat"),
]


@pytest.mark.parametrize("query,expected_intent", _CASES)
def test_router_classifies_correctly(query, expected_intent):
    state = {"messages": [HumanMessage(content=query)]}
    result = router_run(state)
    assert result["intent"] == expected_intent, (
        f"Query: '{query}'\nExpected: {expected_intent}\nGot: {result['intent']}"
    )
