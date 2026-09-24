"""Profile check node — deterministic, no LLM needed."""
from zenic.agent.nodes.profile_check import run


def test_complete_calculate_profile():
    state = {
        "intent": "calculate",
        "user_profile": {
            "weight_kg": 80, "height_cm": 178, "age": 28,
            "gender": "male", "activity_level": "moderate", "goal": "cutting"
        },
    }
    result = run(state)
    assert result["profile_complete"] is True
    assert result["missing_fields"] == []


def test_incomplete_calculate_profile():
    state = {
        "intent": "calculate",
        "user_profile": {"weight_kg": 80, "height_cm": 178},
    }
    result = run(state)
    assert result["profile_complete"] is False
    assert "age" in result["missing_fields"]
    assert "gender" in result["missing_fields"]


def test_workout_plan_missing_fields():
    state = {
        "intent": "workout_plan",
        "user_profile": {"goal": "bulking"},
    }
    result = run(state)
    assert result["profile_complete"] is False
    assert "experience_level" in result["missing_fields"]
    assert "available_days" in result["missing_fields"]


def test_weekly_summary_needs_no_profile():
    state = {"intent": "weekly_summary", "user_profile": {}}
    result = run(state)
    assert result["profile_complete"] is True
