"""Single entry point for every Groq LLM call.

Before this module existed, five nodes each constructed their own ``Groq()``
client per invocation with no timeout, no retry policy, and no error handling —
so a transient 503 from the provider surfaced as an unhandled exception in the
browser UI. Everything now goes through :func:`chat_completion`, which owns:

  * one shared, lazily-built client (connection pooling, thread-safe)
  * a bounded timeout and provider-level retries with exponential backoff
  * latency and token-usage logging under the current correlation id
  * translation of provider failures into :class:`~zenic.errors.LLMError`
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any

from groq import APIError, APIStatusError, Groq, RateLimitError

from zenic.config import get_settings
from zenic.errors import LLMContextLimitError, LLMError, LLMOutputError
from zenic.logging_config import get_logger

logger = get_logger(__name__)

_client: Groq | None = None
_client_lock = threading.Lock()

Message = dict[str, str]


def get_client() -> Groq:
    """Return the process-wide Groq client, building it on first use."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                settings = get_settings()
                _client = Groq(
                    api_key=settings.require_groq_api_key(),
                    timeout=settings.llm_timeout_seconds,
                    max_retries=settings.llm_max_retries,
                )
    return _client


def reset_client() -> None:
    """Drop the cached client. For tests, and after a settings reload."""
    global _client
    with _client_lock:
        if _client is not None:
            _client.close()
        _client = None


def chat_completion(
    messages: list[Message],
    *,
    purpose: str,
    model: str | None = None,
    temperature: float | None = None,
    json_mode: bool = False,
    json_schema: dict[str, Any] | None = None,
) -> str:
    """Run a chat completion and return the assistant's message content.

    ``purpose`` is a short label ("router", "multi_query", …) used in logs so a
    slow or failing call can be attributed to a specific stage of the graph.

    Raises:
        LLMError: on transport failure, provider error, or an empty response.
    """
    settings = get_settings()
    client = get_client()
    kwargs: dict[str, Any] = {
        "model": model or settings.groq_model,
        "messages": messages,
        "max_completion_tokens": 8192 if purpose.startswith("plan_compose:") else 4096,
    }
    if kwargs["model"].startswith("openai/gpt-oss-"):
        kwargs["reasoning_effort"] = "low"
    if temperature is not None:
        kwargs["temperature"] = temperature
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
        if json_schema and kwargs["model"] in {
            "openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b",
        }:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "zenic_plan", "strict": True, "schema": json_schema},
            }

    started = time.perf_counter()
    try:
        response = client.chat.completions.create(**kwargs)
    except RateLimitError as exc:
        logger.warning("llm rate limited", extra={"purpose": purpose, "model": kwargs["model"]})
        raise LLMError(
            "The language model provider is rate limiting requests. Please try again shortly."
        ) from exc
    except APIStatusError as exc:
        logger.error(
            "llm provider error",
            extra={"purpose": purpose, "model": kwargs["model"], "status": exc.status_code},
        )
        if exc.status_code == 413:
            raise LLMContextLimitError("The request exceeds the model provider's input limit.") from exc
        body = exc.body if isinstance(exc.body, dict) else {}
        error = body.get("error", body)
        if json_mode and exc.status_code == 400 and isinstance(error, dict) and error.get("code") == "json_validate_failed":
            raise LLMOutputError("The model provider could not generate valid JSON.") from exc
        raise LLMError(
            f"The language model provider returned an error (HTTP {exc.status_code})."
        ) from exc
    except APIError as exc:
        logger.error("llm transport error", extra={"purpose": purpose})
        raise LLMError("Could not reach the language model provider.") from exc

    elapsed_ms = round((time.perf_counter() - started) * 1000)
    usage = getattr(response, "usage", None)
    logger.info(
        "llm call complete",
        extra={
            "purpose": purpose,
            "model": kwargs["model"],
            "elapsed_ms": elapsed_ms,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
        },
    )

    if not response.choices:
        raise LLMError(f"The language model returned no choices for {purpose}.")
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise LLMError(f"The language model returned an empty response for {purpose}.")
    return content


def chat_completion_json(
    messages: list[Message],
    *,
    purpose: str,
    model: str | None = None,
    temperature: float | None = None,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a chat completion in JSON mode and parse the result.

    Raises:
        LLMError: if the call fails or the response is not a JSON object.
    """
    # Even constrained decoding can fail at the provider. Retry only malformed
    # output, once; auth, configuration, and other HTTP errors remain failures.
    attempt_messages = messages
    for attempt in range(2):
        try:
            raw = chat_completion(
                attempt_messages, purpose=purpose, model=model,
                temperature=temperature if attempt == 0 else 0.2,
                json_mode=True, json_schema=schema,
            )
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise LLMOutputError(f"The language model returned malformed JSON for {purpose}.") from exc
            if not isinstance(parsed, dict):
                raise LLMOutputError(f"Expected a JSON object for {purpose}.")
            return parsed
        except LLMOutputError:
            if attempt:
                raise
            logger.warning("retrying invalid structured output", extra={"purpose": purpose})
            attempt_messages = [*messages, {
                "role": "user",
                "content": "For the original request, return exactly one valid JSON object matching the requested schema. Escape quotes inside strings. Use commas between every field and array item; no comments or trailing commas.",
            }]
    raise LLMOutputError("The model did not return usable JSON.")  # defensive
