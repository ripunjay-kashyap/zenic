"""Generate a formatted PDF from plan_data using FPDF2."""
from __future__ import annotations

import tempfile
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from zenic.agent.state import ZenicState
from zenic.logging_config import get_logger

logger = get_logger(__name__)

_TITLES = {
    "meal_plan": "Meal Plan",
    "workout_plan": "Workout Plan",
    "weekly_summary": "Demo Weekly Summary",
}


def _output_path(intent: str) -> Path:
    """A unique path per generated document.

    The previous fixed name (``zenic_<intent>_plan.pdf``) is shared process-wide,
    so two users generating a plan at the same time would each download whichever
    file was written last.
    """
    directory = Path(tempfile.mkdtemp(prefix="zenic_plan_"))
    return directory / "plan.pdf"



def run(state: ZenicState) -> dict:
    intent = state.get("intent") or ""
    plan_data = state.get("plan_data") or {}
    output_path = _output_path(intent)

    pdf = FPDF()
    pdf.add_page()

    if intent == "meal_plan":
        _render_meal_plan(pdf, plan_data)
    elif intent == "workout_plan":
        _render_workout_plan(pdf, plan_data)
    elif intent == "weekly_summary":
        _render_weekly_summary(pdf, plan_data)
    else:
        _render_generic(pdf, plan_data)
    title = _TITLES.get(intent) or (intent.replace("_", " ").title() if intent else "Plan")

    _body(pdf, "General information only; not a diagnosis or personalized medical advice.")
    pdf.output(str(output_path))
    logger.info("pdf generated", extra={"intent": intent})
    return {
        "messages": [{
            "role": "assistant",
            "content": f"Your {title.lower()} is ready. Download it from the sidebar.",
        }],
        "tool_results": {
            **(state.get("tool_results") or {}),
            "pdf_path": str(output_path),
        },
    }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _safe(text: str) -> str:
    """Sanitize to latin-1 so FPDF core fonts don't error on Unicode chars."""
    return str(text).encode("latin-1", errors="replace").decode("latin-1")


def _header(pdf: FPDF, text: str) -> None:
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _safe(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.set_draw_color(100, 100, 100)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)


def _section_title(pdf: FPDF, text: str) -> None:
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(0, 7, _safe(f"  {text}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.ln(2)


def _body(pdf: FPDF, text: str) -> None:
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5, _safe(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)


# ---------------------------------------------------------------------------
# meal_plan template
# ---------------------------------------------------------------------------

def _render_meal_plan(pdf: FPDF, plan_data: dict) -> None:
    _header(pdf, "Zenic - 7-Day Meal Plan")

    targets = plan_data.get("daily_targets", {})
    if targets:
        _section_title(pdf, "Daily Targets")
        pdf.set_font("Helvetica", size=10)
        parts = []
        for k, v in targets.items():
            label = k.replace("_g", "g").replace("_kcal", " kcal").replace("_", " ").title()
            parts.append(f"{label}: {v}")
        _body(pdf, "  " + "   |   ".join(parts))
        pdf.ln(3)

    days = plan_data.get("days", [])
    for day in days:
        day_name = day.get("day") or day.get("name") or day.get("date") or "Day"
        _section_title(pdf, day_name)

        meals = day.get("meals", [])
        if meals and isinstance(meals[0], dict):
            # Table Header
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_fill_color(220, 220, 220)
            pdf.cell(30, 6, "Meal", border=1, fill=True)
            pdf.cell(85, 6, "Foods", border=1, fill=True)
            pdf.cell(15, 6, "Cals", border=1, fill=True, align="C")
            pdf.cell(20, 6, "Prot", border=1, fill=True, align="C")
            pdf.cell(20, 6, "Carb", border=1, fill=True, align="C")
            pdf.cell(20, 6, "Fat", border=1, fill=True, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            pdf.set_font("Helvetica", size=8)
            for meal in meals:
                meal_name = _safe(str(meal.get("meal") or meal.get("name") or "Meal")[:18])
                foods_val = meal.get("foods") or meal.get("items") or ""
                if isinstance(foods_val, list):
                    foods_str = ", ".join(str(food) for food in foods_val)
                else:
                    foods_str = str(foods_val)
                foods_str = _safe(foods_str[:60])

                cals = _safe(str(meal.get("calories") or "-"))
                prot = _safe(str(meal.get("protein_g") or "-"))
                carb = _safe(str(meal.get("carbs_g") or "-"))
                fat = _safe(str(meal.get("fat_g") or "-"))

                pdf.cell(30, 6, meal_name, border=1)
                pdf.cell(85, 6, foods_str, border=1)
                pdf.cell(15, 6, cals, border=1, align="C")
                pdf.cell(20, 6, prot, border=1, align="C")
                pdf.cell(20, 6, carb, border=1, align="C")
                pdf.cell(20, 6, fat, border=1, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            # List format fallback
            for meal in meals:
                if isinstance(meal, dict):
                    meal_name = meal.get("meal") or meal.get("name") or "Meal"
                    _body(pdf, f"  - {meal_name}: {meal.get('foods', '')}")
                else:
                    _body(pdf, f"  - {meal}")
        pdf.ln(4)

    notes = plan_data.get("notes")
    if notes:
        _section_title(pdf, "Notes")
        _body(pdf, f"  {notes}")


# ---------------------------------------------------------------------------
# workout_plan template
# ---------------------------------------------------------------------------

def _render_workout_plan(pdf: FPDF, plan_data: dict) -> None:
    split_name = plan_data.get("split_name", "Workout Plan")
    _header(pdf, f"Zenic - {split_name}")

    days = plan_data.get("days", [])
    for day in days:
        day_name = day.get("name") or day.get("day") or "Day"
        _section_title(pdf, day_name)

        exercises = day.get("exercises", [])
        if exercises and isinstance(exercises[0], dict):
            # Structured exercise objects — render as table
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_fill_color(220, 220, 220)
            pdf.cell(75, 6, "Exercise", border=1, fill=True)
            pdf.cell(20, 6, "Sets", border=1, fill=True, align="C")
            pdf.cell(25, 6, "Reps", border=1, fill=True, align="C")
            pdf.cell(70, 6, "Muscles", border=1, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            pdf.set_font("Helvetica", size=9)
            for ex in exercises:
                name = _safe(str(ex.get("name") or ex.get("exercise") or "")[:42])
                sets = _safe(str(ex.get("sets") or ""))
                reps = _safe(str(ex.get("reps") or ex.get("rep_range") or ""))
                muscles = _safe(str(ex.get("muscles") or ex.get("muscle_groups") or "")[:38])
                pdf.cell(75, 6, name, border=1)
                pdf.cell(20, 6, sets, border=1, align="C")
                pdf.cell(25, 6, reps, border=1, align="C")
                pdf.cell(70, 6, muscles, border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            # Plain string list
            pdf.set_font("Helvetica", size=10)
            for ex in exercises:
                pdf.cell(0, 6, _safe(f"  - {ex}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(4)

    notes = plan_data.get("notes")
    if notes:
        _section_title(pdf, "Notes")
        _body(pdf, f"  {notes}")


# ---------------------------------------------------------------------------
# weekly_summary template
# ---------------------------------------------------------------------------

def _render_weekly_summary(pdf: FPDF, plan_data: dict) -> None:
    _header(pdf, "Zenic - Weekly Health Summary")

    stats = plan_data.get("weekly_stats", {})
    if stats:
        _section_title(pdf, "Weekly Performance Metrics")
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(95, 7, " Metric", border=1, fill=True)
        pdf.cell(95, 7, " Value", border=1, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_font("Helvetica", size=10)
        stat_labels = [
            ("avg_calories", "Avg. Daily Calories (kcal)"),
            ("avg_protein_g", "Avg. Daily Protein (g)"),
            ("protein_consistency_std", "Protein Consistency (Std Dev)"),
            ("workout_adherence_pct", "Workout Adherence (%)"),
            ("weight_change_kg", "Weekly Weight Change (kg)"),
        ]
        for key, label in stat_labels:
            val = stats.get(key)
            if val is not None:
                pdf.cell(95, 7, f" {label}", border=1)
                pdf.cell(95, 7, f" {val}", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(6)

    insights = plan_data.get("insights", "")
    if insights:
        _section_title(pdf, "Insights & Expert Recommendations")
        pdf.set_font("Helvetica", size=10)
        # Handle markdown-style bullet points or lists if present
        for line in insights.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("-") or line.startswith("*") or (len(line) > 1 and line[0].isdigit() and line[1] == "."):
                pdf.multi_cell(0, 6, _safe(f"  {line}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            else:
                pdf.multi_cell(0, 6, _safe(f"    - {line}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(3)


# ---------------------------------------------------------------------------
# Fallback: generic key-value render
# ---------------------------------------------------------------------------

def _render_generic(pdf: FPDF, plan_data: dict) -> None:
    pdf.set_font("Helvetica", size=11)
    for key, value in plan_data.items():
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, _safe(str(key).replace("_", " ").title()), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", size=10)
        if isinstance(value, list):
            for item in value:
                pdf.multi_cell(0, 6, _safe(f"  {item}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            pdf.multi_cell(0, 6, _safe(f"  {value}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)
