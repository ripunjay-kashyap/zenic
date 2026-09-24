"""
Deterministic calculation tools. No LLM involvement — math must be exact.
The RAG explains *why* these numbers matter; these functions calculate *what* they are.
"""


_ACTIVITY_MULTIPLIERS = {
    "sedentary": 1.2,       # little or no exercise
    "light": 1.375,         # light exercise 1-3 days/week
    "moderate": 1.55,       # moderate exercise 3-5 days/week
    "active": 1.725,        # hard exercise 6-7 days/week
    "very_active": 1.9,     # very hard exercise + physical job
}

_FAT_SHARE = {
    "maintenance": 0.30,
    "cutting": 0.20,
    "bulking": 0.20,
}

_PROTEIN_RANGES = {
    # Illustrative (min g/kg, max g/kg) bands, not individualized prescriptions.
    "sedentary":   (0.8,  1.0),
    "maintenance": (1.2,  1.6),
    "cutting":     (1.6,  2.2),
    "bulking":     (1.6,  2.2),
    "athlete":     (1.6,  2.2),
}


def calculate_bmr(weight_kg: float, height_cm: float, age: int, gender: str) -> float:
    """Mifflin-St Jeor equation. Returns BMR in kcal/day."""
    if gender.lower() in ("male", "m"):
        return 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        return 10 * weight_kg + 6.25 * height_cm - 5 * age - 161


def calculate_tdee(bmr: float, activity_level: str) -> float:
    """Returns TDEE in kcal/day."""
    multiplier = _ACTIVITY_MULTIPLIERS.get(activity_level.lower())
    if multiplier is None:
        raise ValueError(f"Unknown activity_level '{activity_level}'. Choose from: {list(_ACTIVITY_MULTIPLIERS)}")
    return round(bmr * multiplier, 1)


def calculate_macros(tdee: float, goal: str, weight_kg: float) -> dict:
    """Illustrative macros at estimated TDEE, with protein inside the displayed range.

    A fixed percentage of calories made protein climb far above the weight-based
    range shown alongside it. Use that range's midpoint, retain the existing
    goal-specific fat share, and allocate the remaining energy to carbohydrates.
    This is a maintenance-calorie estimate, not a prescribed deficit or surplus.
    """
    fat_share = _FAT_SHARE.get(goal.lower())
    if fat_share is None:
        raise ValueError(f"Unknown goal '{goal}'. Choose from: {list(_FAT_SHARE)}")
    if tdee <= 0 or weight_kg <= 0:
        raise ValueError("TDEE and weight must be positive")
    protein_range = calculate_protein_range(weight_kg, goal)
    protein_g = round((protein_range["min_g"] + protein_range["max_g"]) / 2, 1)
    fat_g = round(tdee * fat_share / 9, 1)
    carbs_g = round((tdee - protein_g * 4 - fat_g * 9) / 4, 1)
    if carbs_g < 0:
        raise ValueError("The estimated calorie intake is too low for these macro targets")
    return {
        "protein_g": protein_g,
        "carbs_g": carbs_g,
        "fat_g": fat_g,
    }


def calculate_protein_range(weight_kg: float, goal: str) -> dict:
    """Return an illustrative daily protein range from body weight and goal."""
    key = goal.lower() if goal.lower() in _PROTEIN_RANGES else "maintenance"
    min_ratio, max_ratio = _PROTEIN_RANGES[key]
    return {
        "min_g": round(weight_kg * min_ratio, 1),
        "max_g": round(weight_kg * max_ratio, 1),
    }
