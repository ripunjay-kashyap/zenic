"""Measure end-to-end and per-stage latency for the chat path.

    PYTHONPATH=. python scripts/latency_check.py
    PYTHONPATH=. python scripts/latency_check.py --runs 3 --no-multi-query

Reports a per-stage breakdown (router, query expansion, embedding, vector
search, BM25, rerank, generation) so it is clear which stage dominates a turn
rather than only that a turn is slow.

Model load time is excluded from the per-turn figures and reported separately —
it is paid once per process, and the container pre-bakes the models.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from collections import defaultdict

_STAGES: dict[str, list[float]] = defaultdict(list)


def _timed(label: str, fn, *args, **kwargs):
    started = time.perf_counter()
    result = fn(*args, **kwargs)
    _STAGES[label].append((time.perf_counter() - started) * 1000)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure chat-path latency.")
    parser.add_argument("--runs", type=int, default=3, help="timed runs per query")
    parser.add_argument(
        "--no-multi-query",
        action="store_true",
        help="disable query expansion (removes one LLM round trip per turn)",
    )
    return parser.parse_args()


QUERIES = [
    ("nutrition_qa", "What does the ISSN recommend for protein intake for athletes?"),
    ("nutrition_qa", "What is the vitamin D upper intake level for adults?"),
    ("general_chat", "What can you do?"),
]


def main() -> int:
    args = parse_args()
    if args.no_multi_query:
        os.environ["MULTI_QUERY_ENABLED"] = "false"

    from zenic.config import reload_settings

    settings = reload_settings()

    from zenic.rag import pipeline
    from zenic.rag.vector_store import get_vector_store

    # --- one-time warmup, reported separately -------------------------------
    started = time.perf_counter()
    pipeline._embed_model_instance()
    embed_load_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    pipeline._reranker_instance()
    rerank_load_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    pipeline._try_load_bm25_from_disk()
    bm25_load_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    get_vector_store()
    store_init_ms = (time.perf_counter() - started) * 1000

    print("=" * 68)
    print("COLD START (once per process; models are pre-baked into the image)")
    print("=" * 68)
    print(f"  embedding model load   {embed_load_ms:8.0f} ms")
    print(f"  reranker model load    {rerank_load_ms:8.0f} ms")
    print(f"  BM25 index build       {bm25_load_ms:8.0f} ms")
    print(f"  vector store connect   {store_init_ms:8.0f} ms")
    print(f"  {'TOTAL':<22} {embed_load_ms + rerank_load_ms + bm25_load_ms + store_init_ms:8.0f} ms")

    # --- timed turns --------------------------------------------------------
    from zenic.agent.nodes import router
    from zenic.rag.pipeline import embed_texts, generate, rerank

    totals: list[float] = []
    for label, query in QUERIES:
        for _ in range(args.runs):
            turn_start = time.perf_counter()

            _timed("router (LLM)", router.run, {"messages": [{"role": "user", "content": query}]})

            if label == "nutrition_qa":
                variants = _timed("query expansion (LLM)", pipeline.generate_multi_queries, query)
                _timed("embedding", embed_texts, variants)
                candidates = _timed(
                    "hybrid search (vector+BM25)", pipeline.hybrid_search, variants
                )
                chunks = _timed("rerank (cross-encoder)", rerank, query, candidates)
            else:
                chunks = []

            _timed("generation (LLM)", generate, query, chunks, label)
            totals.append((time.perf_counter() - turn_start) * 1000)

    # --- report -------------------------------------------------------------
    print()
    print("=" * 68)
    print(f"PER-TURN LATENCY  ({len(totals)} turns, multi_query={settings.multi_query_enabled})")
    print("=" * 68)
    print(f"  {'stage':<30} {'median':>9} {'min':>9} {'max':>9}")
    print(f"  {'-' * 30} {'-' * 9} {'-' * 9} {'-' * 9}")
    grand = 0.0
    for stage, samples in _STAGES.items():
        median = statistics.median(samples)
        grand += median
        print(f"  {stage:<30} {median:8.0f}ms {min(samples):8.0f}ms {max(samples):8.0f}ms")
    print(f"  {'-' * 30} {'-' * 9} {'-' * 9} {'-' * 9}")
    print(f"  {'END-TO-END (median turn)':<30} {statistics.median(totals):8.0f}ms")
    print(f"  {'p95 turn':<30} {sorted(totals)[int(len(totals) * 0.95) - 1]:8.0f}ms")

    llm_ms = sum(statistics.median(v) for k, v in _STAGES.items() if "LLM" in k)
    print()
    print(f"  LLM round trips account for {llm_ms / grand * 100:.0f}% of a turn")
    return 0


if __name__ == "__main__":
    sys.exit(main())
