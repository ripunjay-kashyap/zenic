"""Profile normalisation — the guard between LLM extraction and the calculators."""
import pytest

from zenic.agent.profile import merge_profile, normalize_profile


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("moderately active", "moderate"),
        ("moderately_active", "moderate"),
        ("Sedentary", "sedentary"),
        ("very active", "very_active"),
        ("athlete", "very_active"),
    ],
)
def test_activity_level_synonyms_are_canonicalised(raw, expected):
    assert normalize_profile({"activity_level": raw})["activity_level"] == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("fat loss", "cutting"),
        ("fat_loss", "cutting"),
        ("weight loss", "cutting"),
        ("muscle gain", "bulking"),
        ("hypertrophy", "bulking"),
        ("maintain", "maintenance"),
    ],
)
def test_goal_synonyms_are_canonicalised(raw, expected):
    assert normalize_profile({"goal": raw})["goal"] == expected


def test_uncoercible_enum_is_dropped_not_passed_through():
    """A value the calculators would reject must never reach them."""
    assert "activity_level" not in normalize_profile({"activity_level": "sometimes I walk"})


def test_numeric_strings_are_coerced():
    profile = normalize_profile({"weight_kg": "82.5", "age": "28"})
    assert profile["weight_kg"] == 82.5
    assert profile["age"] == 28
    assert isinstance(profile["age"], int)


@pytest.mark.parametrize("weight", [0, -5, 1200, "heavy"])
def test_implausible_weights_are_dropped(weight):
    assert "weight_kg" not in normalize_profile({"weight_kg": weight})


def test_booleans_are_not_treated_as_numbers():
    assert "age" not in normalize_profile({"age": True})


def test_unknown_fields_are_dropped():
    assert normalize_profile({"favourite_colour": "blue"}) == {}


def test_empty_values_are_dropped():
    assert normalize_profile({"gender": "", "goal": None}) == {}


def test_available_days_is_clamped_to_a_week():
    assert "available_days" not in normalize_profile({"available_days": 9})
    assert normalize_profile({"available_days": 4})["available_days"] == 4


def test_merge_does_not_mutate_its_inputs():
    existing = {"weight_kg": 80}
    updates = {"weight_kg": 82}
    merged = merge_profile(existing, updates)
    assert merged["weight_kg"] == 82
    assert existing == {"weight_kg": 80}


def test_merge_keeps_fields_absent_from_the_update():
    merged = merge_profile({"weight_kg": 80, "goal": "cutting"}, {"age": 30})
    assert merged == {"weight_kg": 80, "goal": "cutting", "age": 30}
