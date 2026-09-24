"""Message normalisation across dict and LangChain shapes."""
from langchain_core.messages import AIMessage, HumanMessage

from zenic.agent.messages import (
    last_user_message,
    message_content,
    message_role,
    to_openai_messages,
)


def test_dict_message_roles_and_content():
    message = {"role": "assistant", "content": "hi"}
    assert message_role(message) == "assistant"
    assert message_content(message) == "hi"


def test_langchain_messages_map_to_openai_roles():
    assert message_role(HumanMessage(content="q")) == "user"
    assert message_role(AIMessage(content="a")) == "assistant"


def test_block_content_is_flattened():
    message = {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
    assert message_content(message) == "ab"


def test_missing_content_is_an_empty_string():
    assert message_content({"role": "user"}) == ""


def test_to_openai_messages_handles_mixed_shapes():
    result = to_openai_messages([HumanMessage(content="q"), {"role": "assistant", "content": "a"}])
    assert result == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]


def test_to_openai_messages_of_none_is_empty():
    assert to_openai_messages(None) == []


def test_last_user_message_skips_trailing_assistant_turns():
    """Nodes downstream of a reply must still see the user's question."""
    state = {
        "messages": [
            {"role": "user", "content": "what is my tdee?"},
            {"role": "assistant", "content": "I need your weight"},
        ]
    }
    assert last_user_message(state) == "what is my tdee?"


def test_last_user_message_with_langchain_objects():
    state = {"messages": [HumanMessage(content="hello"), AIMessage(content="hi")]}
    assert last_user_message(state) == "hello"


def test_last_user_message_with_no_messages_is_empty():
    assert last_user_message({"messages": []}) == ""
    assert last_user_message({}) == ""
