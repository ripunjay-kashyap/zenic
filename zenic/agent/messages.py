"""Message normalisation helpers.

``ZenicState["messages"]`` is annotated with LangGraph's ``add_messages``
reducer, which converts plain dicts into LangChain message objects — but only
for state that has already passed through the reducer. Nodes therefore see
either shape depending on how they were invoked (graph run vs. direct unit
test), and the original code assumed one or the other in different places,
crashing with ``AttributeError`` on the mismatch. These helpers accept both.
"""
from __future__ import annotations

from typing import Any

#: LangChain message types → OpenAI/Groq API roles.
_TYPE_TO_ROLE = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}


def message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role", "user"))
    return _TYPE_TO_ROLE.get(getattr(message, "type", "human"), "user")


def message_content(message: Any) -> str:
    """Extract text content from a dict or LangChain message.

    LangChain permits list-of-blocks content; only the text blocks are kept.
    """
    raw = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
    if isinstance(raw, list):
        parts = [
            block.get("text", "") if isinstance(block, dict) else str(block) for block in raw
        ]
        return "".join(parts)
    return raw if isinstance(raw, str) else str(raw)


def to_openai_messages(messages: Any) -> list[dict[str, str]]:
    """Normalise a message list to OpenAI-format dicts."""
    return [
        {"role": message_role(m), "content": message_content(m)[:4000]}
        for m in (messages or [])[-12:]
        if message_role(m) in ("user", "assistant")
    ]


def last_user_message(state: Any) -> str:
    """Text of the most recent user turn, or the last message if none is tagged user.

    Returns an empty string when there are no messages at all, so callers can
    make their own decision rather than handling an IndexError.
    """
    messages = (state or {}).get("messages") or []
    if not messages:
        return ""
    for message in reversed(messages):
        if message_role(message) == "user":
            return message_content(message)
    return message_content(messages[-1])
