"""Final LLM generation node — formats the response with source citations."""
from zenic.agent.state import ZenicState
from zenic.rag.pipeline import generate as rag_generate

_LC_TYPE_TO_ROLE = {"human": "user", "ai": "assistant", "system": "system"}


def _to_openai_messages(messages) -> list[dict]:
    """Normalize LangChain message objects + dicts to OpenAI-format dicts."""
    out = []
    for m in messages or []:
        if isinstance(m, dict):
            out.append({"role": m.get("role", "user"), "content": m.get("content", "")})
        else:
            role = _LC_TYPE_TO_ROLE.get(getattr(m, "type", "human"), "user")
            out.append({"role": role, "content": getattr(m, "content", "")})
    return out


def run(state: ZenicState) -> dict:
    query = state["messages"][-1].content
    context = state.get("retrieved_context") or []
    # Strip internal tracking keys before injecting into context
    _INTERNAL_KEYS = {"api_fallback_used"}
    tool_results = {
        k: v for k, v in (state.get("tool_results") or {}).items()
        if k not in _INTERNAL_KEYS
    }

    # Inject calculation results into context if present
    if tool_results:
        tool_text = "Calculation results:\n" + "\n".join(f"  {k}: {v}" for k, v in tool_results.items())
        context = [{"text": tool_text, "metadata": {"source": "Zenic Calculator"}}] + context

    history = _to_openai_messages(state.get("messages", []))
    answer = rag_generate(query, context, intent=state.get("intent", ""), history=history)
    return {"messages": [{"role": "assistant", "content": answer}]}
