"""
Layer 1: Keyword filter + lightweight classification.
Fast, cheap — catches obvious harmful requests before any LLM or API calls.
"""
import re

_BANNED_SUBSTANCE = "banned or restricted substance"
_MEDICAL_ADVICE = "medical diagnosis or treatment"
_EXTREME_RESTRICTION = "extreme calorie restriction"

_HARD_BLOCK_PATTERNS: list[tuple[str, str]] = [
    (r"\bsteroids?\b", _BANNED_SUBSTANCE),
    (r"\bsarms?\b", _BANNED_SUBSTANCE),
    (r"\bgrowth hormone\b", _BANNED_SUBSTANCE),
    (r"\bhgh\b", _BANNED_SUBSTANCE),
    (r"\bephedrine\b", _BANNED_SUBSTANCE),
    (r"\bdnp\b", _BANNED_SUBSTANCE),
    (r"\bdinitrophenol\b", _BANNED_SUBSTANCE),
    (r"\bclenbuterol\b", _BANNED_SUBSTANCE),
    (r"\bmedical diagnos\w+\b", _MEDICAL_ADVICE),
    (r"\bcure\s+(my\s+)?(cancer|diabetes|heart)", _MEDICAL_ADVICE),
    (r"\b(500|400|300)\s*calories?\s*(a\s+day|per\s+day|daily)\b", _EXTREME_RESTRICTION),
]

_COMPILED = [(re.compile(p, re.IGNORECASE), category) for p, category in _HARD_BLOCK_PATTERNS]


def is_harmful(text: str) -> tuple[bool, str | None]:
    """Returns (is_harmful, friendly_category_or_None).

    The category is user-facing — never include raw regex patterns or other
    internals here, since the safety_response node surfaces this string in chat.
    """
    for pattern, category in _COMPILED:
        if pattern.search(text):
            return True, category
    return False, None
