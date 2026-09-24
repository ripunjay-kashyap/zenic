"""Ask the user for missing profile fields in a single friendly message."""
from __future__ import annotations

from zenic.agent.state import ZenicState
from zenic.errors import LLMError
from zenic.llm import chat_completion
from zenic.logging_config import get_logger

logger = get_logger(__name__)

_FIELD_LABELS = {
    "weight_kg": "your weight in kg",
    "height_cm": "your height in cm",
    "age": "your age",
    "gender": "your gender",
    "activity_level": "your activity level (sedentary, light, moderate, active, or very active)",
    "goal": "your goal (maintenance, cutting, or bulking)",
    "dietary_restrictions": "any dietary restrictions",
    "experience_level": "your training experience (beginner, intermediate, or advanced)",
    "available_days": "how many days a week you can train",
    "equipment": "what equipment you have access to",
}


def run(state: ZenicState) -> dict:
    missing = state.get("missing_fields", []) or []
    intent = state.get("intent", "")
    labels = [_FIELD_LABELS.get(field, field.replace("_", " ")) for field in missing]

    prompt = (
        f"The user wants a {intent.replace('_', ' ')}. "
        f"You need the following information to proceed: {', '.join(labels)}. "
        "Ask the user for all of these in a single, natural, friendly message."
    )
    try:
        question = chat_completion(
            [{"role": "user", "content": prompt}], purpose="profile_gather"
        )
    except LLMError:
        logger.warning("profile_gather LLM call failed — using the static fallback prompt")
        question = _fallback_question(intent, labels)

    return {
        "messages": [{"role": "assistant", "content": question}],
        "awaiting_input": True,
    }


def _fallback_question(intent: str, labels: list[str]) -> str:
    """Deterministic prompt used when the model is unavailable."""
    if not labels:
        return "Could you tell me a bit more about yourself so I can help?"
    bullets = "\n".join(f"- {label}" for label in labels)
    return (
        f"To build your {intent.replace('_', ' ')}, I need a few details:\n\n{bullets}\n\n"
        "You can give them all in one message."
    )
