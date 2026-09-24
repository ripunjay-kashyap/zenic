"""Final LLM generation node — formats the response with source citations."""
from __future__ import annotations

from zenic.agent.messages import last_user_message, to_openai_messages
from zenic.agent.state import ZenicState
from zenic.logging_config import get_logger
from zenic.rag.pipeline import generate as rag_generate

logger = get_logger(__name__)

#: Keys used for internal bookkeeping — never shown to the model as "results".
_INTERNAL_KEYS = frozenset({"api_fallback_used", "pdf_path"})

#: Values that are retrieval payloads rather than computed numbers. Injecting
#: these into the calculator block would dump whole chunk lists into the prompt.
_CHUNK_KEYS = frozenset({"exercise_chunks", "food_chunks", "weekly_data", "weekly_stats"})


def run(state: ZenicState) -> dict:
    query = last_user_message(state)
    context = list(state.get("retrieved_context") or [])

    tool_results = {
        k: v
        for k, v in (state.get("tool_results") or {}).items()
        if k not in _INTERNAL_KEYS and k not in _CHUNK_KEYS
    }

    # Inject calculation results into context if present
    if tool_results:
        tool_text = "Calculation results:\n" + "\n".join(
            f"  {k}: {v}" for k, v in tool_results.items()
        )
        context.insert(0, {"text": tool_text, "metadata": {"source": "Zenic Calculator"}})

    history = to_openai_messages(state.get("messages", []))
    answer = rag_generate(query, context, intent=state.get("intent", ""), history=history)
    logger.info(
        "response generated",
        extra={"intent": state.get("intent", ""), "context_chunks": len(context)},
    )
    return {"messages": [{"role": "assistant", "content": answer}]}
