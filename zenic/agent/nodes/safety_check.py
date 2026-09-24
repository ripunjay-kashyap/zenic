"""Layer 1 safety filter — runs before every LLM call."""
from __future__ import annotations

from zenic.agent.messages import last_user_message
from zenic.agent.state import ZenicState
from zenic.errors import ZenicError
from zenic.logging_config import get_logger
from zenic.safety.layer1_classifier import is_harmful

logger = get_logger(__name__)


def run(state: ZenicState) -> dict:
    query = last_user_message(state)
    if len(query) > 4000:
        raise ZenicError("Please keep your message under 4,000 characters.")
    flagged, reason = is_harmful(query)
    if flagged:
        logger.info("message blocked by layer 1 safety filter", extra={"category": reason})
    return {"safety_flag": flagged, "safety_reason": reason or ""}
