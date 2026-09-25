"""
Run all 12 spot-check queries and print stage-by-stage retrieval output.
Use --save and --compare flags to A/B test retrieval changes.
"""
import argparse
import json

from zenic.config import get_settings
from zenic.rag.pipeline import generate_multi_queries, hybrid_search, rerank


def spot_check(query: str, verbose: bool = True) -> dict:
    print(f"\n{'#' * 70}\nQUERY: {query}\n{'#' * 70}")

    variants = generate_multi_queries(query)
    print(f"\n[1] MULTI-QUERY VARIANTS ({len(variants)}):")
    for v in variants:
        print(f"    - {v}")

    pool = get_settings().retrieval_candidate_pool
    candidates = hybrid_search(variants, top_k=pool)
    print(f"\n[2] HYBRID SEARCH — TOP {pool}:")
    for i, c in enumerate(candidates, 1):
        snippet = c["text"][:120].replace("\n", " ")
        print(f"  {i:>2}. [vec={c.get('vector_score', 0):.3f} bm25={c.get('bm25_score', 0):.3f}] "
              f"src={c['metadata'].get('source')} :: {snippet}...")

    reranked = rerank(query, candidates, top_k=7)
    print("\n[3] AFTER RERANK — TOP 7:")
    for i, c in enumerate(reranked, 1):
        snippet = c["text"][:120].replace("\n", " ")
        print(f"  {i}. [rerank={c.get('rerank_score', 0):.3f}] "
              f"src={c['metadata'].get('source')} :: {snippet}...")

    if verbose:
        print("\n[4] FULL TEXT — TOP 3:")
        for i, c in enumerate(reranked[:3], 1):
            print(f"\n--- Chunk {i} ---\nMetadata: {c['metadata']}\n{c['text']}")

    return {"query": query, "top3": reranked[:3]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", help="Save results to JSON file")
    parser.add_argument("--compare", help="Compare against a previously saved baseline JSON")
    args = parser.parse_args()

    with open("eval_data/pillar1_spot_check.json", encoding="utf-8") as f:
        cases = json.load(f)

    results = []
    for case in cases:
        result = spot_check(case["query"])
        results.append(result)
        input("\n[Press Enter for next query...]")

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.save}")

    if args.compare:
        with open(args.compare, encoding="utf-8") as f:
            baseline = json.load(f)
        print("\n=== COMPARISON vs BASELINE ===")
        for new, old in zip(results, baseline, strict=False):
            new_sources = [c["metadata"].get("source") for c in new["top3"]]
            old_sources = [c["metadata"].get("source") for c in old["top3"]]
            if new_sources != old_sources:
                print(f"\nCHANGED: {new['query']}")
                print(f"  Before: {old_sources}")
                print(f"  After : {new_sources}")


if __name__ == "__main__":
    main()
