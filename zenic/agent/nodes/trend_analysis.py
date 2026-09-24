"""Deterministic weekly stats from raw tracking data."""
from __future__ import annotations

import statistics
from typing import Any

from zenic.agent.state import ZenicState
from zenic.logging_config import get_logger

logger = get_logger(__name__)


def run(state: ZenicState) -> dict:
    tool_results = state.get("tool_results") or {}
    data = tool_results.get("weekly_data") or []

    stats = _compute_stats(data)
    logger.info("weekly stats computed", extra={"days": len(data), "has_stats": bool(stats)})
    return {"tool_results": {**tool_results, "weekly_stats": stats}}


def _compute_stats(data: list[dict]) -> dict[str, Any]:
    """Summarise a week of entries. Returns {} when there is nothing logged."""
    if not data:
        return {}

    # Sort by date so first/last are chronological — the caller's ordering is
    # not guaranteed, and weight change depends on it.
    ordered = sorted(data, key=lambda d: str(d.get("date", "")))

    calories = _numeric(ordered, "calories")
    protein = _numeric(ordered, "protein_g")
    workouts_planned = sum(1 for d in ordered if d.get("workout_planned"))
    workouts_done = sum(1 for d in ordered if d.get("workout_completed"))

    # An all-zero week is the placeholder the ingestion node emits when nothing
    # has been tracked; report it as "no data" rather than as a real zero-calorie week.
    if not any(calories) and not any(protein) and not workouts_planned:
        return {}

    return {
        "avg_calories": round(statistics.mean(calories), 1) if calories else None,
        "avg_protein_g": round(statistics.mean(protein), 1) if protein else None,
        "protein_consistency_std": (
            round(statistics.stdev(protein), 1) if len(protein) > 1 else 0.0
        ),
        "workout_adherence_pct": (
            round(workouts_done / workouts_planned * 100) if workouts_planned else None
        ),
        "weight_change_kg": _weight_change(ordered),
    }


def _numeric(entries: list[dict], key: str) -> list[float]:
    """Values for ``key`` that are actually numbers — bad rows are skipped, not fatal."""
    values: list[float] = []
    for entry in entries:
        value = entry.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        values.append(float(value))
    return values


def _weight_change(ordered: list[dict]) -> float | None:
    weights = [
        (d.get("date"), d["weight_kg"])
        for d in ordered
        if isinstance(d.get("weight_kg"), (int, float)) and not isinstance(d.get("weight_kg"), bool)
    ]
    if len(weights) < 2:
        return None
    return round(float(weights[-1][1]) - float(weights[0][1]), 2)
