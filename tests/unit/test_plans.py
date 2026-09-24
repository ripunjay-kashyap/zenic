"""Validate provider schema selection and reject unusable generated plans."""

from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from groq import APIStatusError
from pydantic import ValidationError

from zenic import llm
from zenic.agent.plans import MealPlan, WorkoutPlan
from zenic.errors import LLMContextLimitError, LLMError, LLMOutputError


@pytest.mark.parametrize(
    "model,format_type",
    [("openai/gpt-oss-20b", "json_schema"), ("custom-model", "json_object")],
)
def test_plan_response_format(settings_env, monkeypatch, model, format_type):
    settings_env(GROQ_MODEL=model)
    create = Mock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))], usage=None,
    ))
    monkeypatch.setattr(llm, "get_client", lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    ))
    llm.chat_completion_json(
        [{"role": "user", "content": "Return JSON"}],
        purpose="plan_compose:workout_plan", schema=WorkoutPlan.model_json_schema(),
        temperature=0,
    )
    kwargs = create.call_args.kwargs
    assert kwargs["response_format"]["type"] == format_type
    assert kwargs["max_completion_tokens"] == 8192
    if format_type == "json_schema":
        assert kwargs["response_format"]["json_schema"]["strict"] is True


def test_workout_requires_exercise_details():
    with pytest.raises(ValidationError):
        WorkoutPlan.model_validate({
            "split_name": "Example", "notes": "Educational example",
            "days": [{"name": "Day 1", "exercises": [{}]}],
        })


@pytest.mark.parametrize("days,calories", [(6, 2000), (7, -1), (7, float("nan"))])
def test_meal_plan_rejects_incomplete_week_or_invalid_numbers(days, calories):
    targets = {"calories": calories, "protein_g": 100, "carbs_g": 200, "fat_g": 60}
    with pytest.raises(ValidationError):
        MealPlan.model_validate({
            "daily_targets": targets, "notes": "Estimates only",
            "days": [{"name": f"Day {day}", "meals": [
                {**targets, "meal": "Example", "foods": ["Example food"]},
            ]} for day in range(days)],
        })


@pytest.mark.parametrize("first", [LLMOutputError("provider invalid JSON"), "not JSON", "[]"])
def test_structured_output_recovers_once(monkeypatch, first):
    complete = Mock(side_effect=[first, '{"ok": true}'])
    monkeypatch.setattr(llm, "chat_completion", complete)
    assert llm.chat_completion_json([], purpose="test") == {"ok": True}
    assert complete.call_count == 2


def test_structured_output_retry_is_bounded(monkeypatch):
    complete = Mock(side_effect=LLMOutputError("invalid"))
    monkeypatch.setattr(llm, "chat_completion", complete)
    with pytest.raises(LLMOutputError):
        llm.chat_completion_json([], purpose="test")
    assert complete.call_count == 2


def test_structured_output_does_not_retry_other_provider_errors(monkeypatch):
    complete = Mock(side_effect=LLMError("authentication failed"))
    monkeypatch.setattr(llm, "chat_completion", complete)
    with pytest.raises(LLMError):
        llm.chat_completion_json([], purpose="test")
    assert complete.call_count == 1


@pytest.mark.parametrize("status,code,expected", [
    (413, "rate_limit_exceeded", LLMContextLimitError),
    (400, "json_validate_failed", LLMOutputError),
    (400, "invalid_request", LLMError),
])
def test_provider_errors_are_classified_without_logging_body(monkeypatch, caplog, status, code, expected):
    error = APIStatusError(
        "synthetic provider error", response=httpx.Response(
            status, request=httpx.Request("POST", "https://example.invalid"),
        ), body={"error": {"code": code, "failed_generation": "private-model-output"}},
    )
    create = Mock(side_effect=error)
    monkeypatch.setattr(llm, "get_client", lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    ))
    with pytest.raises(expected):
        llm.chat_completion([], purpose="test", json_mode=True)
    assert "private-model-output" not in caplog.text
