"""Schemas shared by constrained plan generation and local output validation."""

from pydantic import BaseModel, ConfigDict, Field


class _PlanRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Exercise(_PlanRecord):
    name: str
    sets: int = Field(ge=1, le=20)
    reps: str
    muscles: str


class WorkoutDay(_PlanRecord):
    name: str
    exercises: list[Exercise] = Field(min_length=1, max_length=20)


class WorkoutPlan(_PlanRecord):
    split_name: str
    days: list[WorkoutDay] = Field(min_length=1, max_length=7)
    notes: str


class DailyTargets(_PlanRecord):
    calories: float = Field(ge=0)
    protein_g: float = Field(ge=0)
    carbs_g: float = Field(ge=0)
    fat_g: float = Field(ge=0)


class Meal(DailyTargets):
    meal: str
    foods: list[str] = Field(min_length=1, max_length=20)


class MealDay(_PlanRecord):
    name: str
    meals: list[Meal] = Field(min_length=1, max_length=20)


class MealPlan(_PlanRecord):
    daily_targets: DailyTargets
    days: list[MealDay] = Field(min_length=7, max_length=7)
    notes: str
