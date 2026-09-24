"""Intent classification using structured LLM output."""
from __future__ import annotations

from zenic.agent.messages import to_openai_messages
from zenic.agent.state import ZenicState
from zenic.errors import LLMError
from zenic.llm import chat_completion_json
from zenic.logging_config import get_logger

logger = get_logger(__name__)

INTENTS = (
    "nutrition_qa",
    "calculate",
    "meal_plan",
    "workout_plan",
    "weekly_summary",
    "general_chat",
)

#: Unknown classifications still require evidence before answering health questions.
DEFAULT_INTENT = "nutrition_qa"

_SYSTEM_PROMPT = f"""Classify the user's message into exactly one intent.
Return a JSON object with a single key "intent" from this list: {list(INTENTS)}.

nutrition_qa covers ANY factual lookup answered from the knowledge base:
food nutrients, supplement guidelines, exercise descriptions, muscles worked,
equipment needed, dietary recommendations, ingredient info, etc.

Examples:
- "How much protein is in chicken?" → {{"intent": "nutrition_qa"}}
- "What muscles does the barbell row work?" → {{"intent": "nutrition_qa"}}
- "iron content in spinach" → {{"intent": "nutrition_qa"}}
- "ISSN creatine loading recommendations" → {{"intent": "nutrition_qa"}}
- "What exercises target the back with a barbell?" → {{"intent": "nutrition_qa"}}
- "vitamin D upper intake level for adults" → {{"intent": "nutrition_qa"}}
- "What's my TDEE?" → {{"intent": "calculate"}}
- "Make me a high-protein meal plan" → {{"intent": "meal_plan"}}
- "Give me a PPL workout split" → {{"intent": "workout_plan"}}
- "Summarize my week" → {{"intent": "weekly_summary"}}
- "What can you do?" → {{"intent": "general_chat"}}
- "What is Zenic?" → {{"intent": "general_chat"}}
- "Hello" → {{"intent": "general_chat"}}
- "Tell me about yourself" → {{"intent": "general_chat"}}

general_chat covers greetings, questions about Zenic itself, and anything that
is NOT a factual nutrition/exercise lookup, calculation, plan request, or summary.
"""


def run(state: ZenicState) -> dict:
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages.extend(to_openai_messages(state.get("messages", [])))

    try:
        result = chat_completion_json(messages, purpose="router")
    except LLMError:
        logger.warning("intent classification failed — defaulting to %s", DEFAULT_INTENT)
        return {"intent": DEFAULT_INTENT}

    intent = result.get("intent")
    if intent not in INTENTS:
        logger.warning(
            "router returned an unknown intent — defaulting",
            extra={"default": DEFAULT_INTENT},
        )
        intent = DEFAULT_INTENT

    logger.info("intent classified", extra={"intent": intent})
    return {"intent": intent}
