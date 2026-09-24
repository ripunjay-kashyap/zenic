"""
Wrapper node — delegates to the Pillar 1 pipeline.

RAG-first policy: always try the local index first. If the top reranked
result scores below FALLBACK_THRESHOLD (meaning the index doesn't have a
good answer), fall back to the live USDA FoodData Central API and record
the fallback so trace.py / rag_vs_api_check.py can detect it.
"""
from __future__ import annotations

from zenic.agent.messages import last_user_message
from zenic.agent.state import ZenicState
from zenic.config import get_settings
from zenic.errors import ExternalAPIError, RetrievalError
from zenic.logging_config import get_logger
from zenic.rag.pipeline import rerank, retrieve

logger = get_logger(__name__)

# Threshold for "index returned nothing useful". BAAI/bge-reranker-base
# uses its default activation; the operational threshold is tuned to its returned scores.
# Tune this value against retrieval_spot_check output if needed.
_FALLBACK_SCORE_THRESHOLD = 0.5


def _poor_retrieval(chunks: list[dict]) -> bool:
    """True when RAG found nothing relevant for the query."""
    if not chunks:
        return True
    return chunks[0].get("rerank_score", 0.0) < _FALLBACK_SCORE_THRESHOLD


def run(state: ZenicState) -> dict:
    query = last_user_message(state)
    if not query:
        return {"retrieved_context": []}

    try:
        chunks = retrieve(query)
    except RetrievalError:
        # The knowledge base is down. Rather than failing the turn, hand an
        # empty context to generate(), which abstains if no fallback evidence exists.
        logger.warning("knowledge base unavailable — evidence required before answering")
        chunks = []

    if _poor_retrieval(chunks) and get_settings().usda_api_key:
        api_chunks = _usda_fallback(query)
        api_chunks = rerank(query, api_chunks) if api_chunks else []
        if api_chunks and not _poor_retrieval(api_chunks):
            logger.info("rag miss — served from the live USDA API", extra={"results": len(api_chunks)})
            return {
                "retrieved_context": api_chunks,
                "tool_results": {
                    **(state.get("tool_results") or {}),
                    "api_fallback_used": "usda_api",
                },
            }

    return {"retrieved_context": chunks}


def _usda_fallback(query: str) -> list[dict]:
    """Query the live USDA API. Returns [] on any failure — never raises."""
    from zenic.agent.tools import usda_api

    try:
        items = usda_api.search_food(query)
    except ExternalAPIError:
        logger.warning("usda fallback unavailable")
        return []

    return [
        {
            "text": (
                f"{item['name']}\n"
                + "\n".join(f"  {k}: {v}" for k, v in item.get("nutrients", {}).items())
            ),
            "metadata": {"source": "USDA FoodData Central (live)", "url": "https://fdc.nal.usda.gov/"},
        }
        for item in items
    ]
