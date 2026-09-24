"""Unit tests for deterministic calculation tools. Zero tolerance for numeric error."""
import pytest

from zenic.agent.tools.calculations import (
    calculate_bmr,
    calculate_macros,
    calculate_protein_range,
    calculate_tdee,
)


def test_bmr_male():
    # 80kg, 178cm, 28yo male → 10*80 + 6.25*178 - 5*28 + 5 = 1777.5
    assert calculate_bmr(80, 178, 28, "male") == pytest.approx(1777.5)


def test_bmr_female():
    # 60kg, 165cm, 25yo female → 10*60 + 6.25*165 - 5*25 - 161 = 1345.25
    assert calculate_bmr(60, 165, 25, "female") == pytest.approx(1345.25)


def test_tdee_moderate():
    tdee = calculate_tdee(1902.5, "moderate")
    assert tdee == pytest.approx(1902.5 * 1.55, rel=0.01)


def test_tdee_invalid_activity():
    with pytest.raises(ValueError, match="Unknown activity_level"):
        calculate_tdee(1900, "super_active")


def test_macros_cutting():
    macros = calculate_macros(2000, "cutting")
    assert macros["protein_g"] == pytest.approx(2000 * 0.40 / 4, rel=0.01)
    assert macros["carbs_g"] == pytest.approx(2000 * 0.40 / 4, rel=0.01)
    assert macros["fat_g"] == pytest.approx(2000 * 0.20 / 9, rel=0.01)


def test_macros_invalid_goal():
    with pytest.raises(ValueError, match="Unknown goal"):
        calculate_macros(2000, "shredding")


def test_protein_range_bulking():
    result = calculate_protein_range(80, "bulking")
    assert result["min_g"] == pytest.approx(80 * 1.6, rel=0.01)
    assert result["max_g"] == pytest.approx(80 * 2.2, rel=0.01)


def test_protein_range_sedentary():
    result = calculate_protein_range(70, "sedentary")
    assert result["min_g"] == pytest.approx(70 * 0.8, rel=0.01)
