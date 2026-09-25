"""
Pillar 1 automated spot-check tests.

Runs all 12 cases from eval_data/pillar1_spot_check.json and asserts:
  - Expected source appears in top-7 retrieved chunks (when expected_source is not null)
  - At least one expected keyword is found across top-7 chunks

Cases p1_004 (ISSN protein) and p1_011 (ISSN creatine) require ISSN PDFs to be indexed
and are skipped if no ISSN chunks exist. p1_006 (dietary guidelines) similarly.
p1_009 and p1_010 are RAG-vs-API enforcement checks handled by rag_vs_api_check.py.

Run with: pytest tests/pillar1/test_retrieval_spot_check.py -v
Requires: GROQ_API_KEY (for multi-query expansion)
"""
import json
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

# All tests in this file call Groq (multi-query expansion) — skip without API key
pytestmark = pytest.mark.integration

# Source availability — skip cases if source not indexed
_SOURCE_SKIP = {
    "ISSN": "ISSN PDFs not indexed — place PDFs in data/issn/ and run ingestion",
    "DietaryGuidelines": "Dietary Guidelines PDF not indexed — place PDF in data/dietary_guidelines/ and run ingestion",
    "USDA API (live)": None,   # p1_009 — RAG-vs-API enforcement, skip in unit test
}

_EVAL_PATH = Path("eval_data/pillar1_spot_check.json")
_SKIP_CASE_IDS = {"p1_009", "p1_010"}  # RAG-vs-API enforcement — covered by rag_vs_api_check.py


def _load_cases():
    with open(_EVAL_PATH, encoding="utf-8") as f:
        return json.load(f)


def _check_source_indexed(source: str) -> bool:
    """Return True if the source has at least one chunk in ChromaDB."""
    from zenic.rag.vector_store import get_vector_store
    store = get_vector_store()
    chunks = store.sample_chunks(where={"source": source}, n=1)
    return len(chunks) > 0


@pytest.fixture(scope="module")
def pipeline():
    """Load the retrieval pipeline once for all tests in this module."""
    from zenic.rag.pipeline import (
        _try_load_bm25_from_disk,
        generate_multi_queries,
        hybrid_search,
        rerank,
    )
    _try_load_bm25_from_disk()
    return generate_multi_queries, hybrid_search, rerank


@pytest.mark.parametrize("case", _load_cases(), ids=[c["id"] for c in _load_cases()])
def test_retrieval_spot_check(case, pipeline):
    case_id = case["id"]
    expected_source = case.get("expected_source")
    expected_keywords = case.get("expected_keywords_in_context", [])

    # Skip cases handled elsewhere
    if case_id in _SKIP_CASE_IDS:
        pytest.skip(f"{case_id} — RAG-vs-API enforcement: run rag_vs_api_check.py instead")

    # Skip if required source not indexed
    if expected_source and expected_source in _SOURCE_SKIP:
        skip_reason = _SOURCE_SKIP[expected_source]
        if skip_reason and not _check_source_indexed(expected_source.split(" ")[0]):
            pytest.skip(f"{case_id} — {skip_reason}")

    generate_multi_queries, hybrid_search, rerank = pipeline

    query = case["query"]
    variants = generate_multi_queries(query, n=2)   # n=2 to stay fast
    from zenic.config import get_settings

    candidates = hybrid_search(variants, top_k=get_settings().retrieval_candidate_pool)
    chunks = rerank(query, candidates, top_k=7)

    # For p1_012 (graceful failure) — just check retrieval doesn't crash
    if case_id == "p1_012":
        assert isinstance(chunks, list)
        return

    all_text = " ".join(c["text"].lower() for c in chunks)
    sources_found = [c["metadata"].get("source", "") for c in chunks]

    # Assert at least one expected keyword appears in top-7 chunks
    if expected_keywords:
        kw_found = [kw for kw in expected_keywords if kw.lower() in all_text]
        assert kw_found, (
            f"{case_id}: None of {expected_keywords} found in top-7 chunks.\n"
            f"Sources retrieved: {sources_found}\n"
            f"Query: {query}"
        )

    # General food-list queries can be answered by authoritative guidelines,
    # while nutrient-quantity queries below continue to require USDA tables.
    if acceptable_sources := case.get("acceptable_sources"):
        assert set(sources_found) & set(acceptable_sources)
        food_matches = [word for word in case["required_food_keywords"] if word in all_text]
        assert len(food_matches) >= case["minimum_food_matches"], (
            f"{case_id}: expected named plant foods, found {food_matches}"
        )

    # Assert expected source is present in top-7 (if specified and indexed)
    if expected_source and expected_source not in _SOURCE_SKIP:
        expected_src_short = expected_source.split(" ")[0].upper()
        assert any(expected_src_short in s.upper() for s in sources_found), (
            f"{case_id}: Expected source '{expected_source}' not found in top-7.\n"
            f"Sources found: {sources_found}"
        )
