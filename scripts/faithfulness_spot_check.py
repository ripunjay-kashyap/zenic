"""
Faithfulness spot-check for manual evaluation.

Runs 5 representative queries through the RAG pipeline (retrieve + generate)
and prints the retrieved context alongside the generated answer so you can
eyeball whether the LLM fabricated any numbers, sources, or claims.

Usage:
    python scripts/faithfulness_spot_check.py              # run all cases
    python scripts/faithfulness_spot_check.py --skip f1    # skip known data-gap cases
    python scripts/faithfulness_spot_check.py --only f2,f3,f5

Requires: GROQ_API_KEY in .env, populated vector DB + BM25 corpus.
"""
import argparse
import sys
import textwrap

from dotenv import load_dotenv

# Ensure Unicode output works on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

from zenic.rag.pipeline import generate, hybrid_search, rerank, retrieve

# ---------------------------------------------------------------------------
# 5 queries chosen to cover diverse failure modes
# ---------------------------------------------------------------------------
CASES = [
    {
        "id": "f1",
        "query": "How much protein is in 100g of cooked chicken breast?",
        "check": [
            "Answer mentions ~31 g protein (not a made-up number)",
            "Source cited is USDA FoodData Central",
            "No non-USDA source used for the numeric value",
        ],
    },
    {
        "id": "f2",
        "query": "What is the tolerable upper intake level for vitamin D in adults?",
        "check": [
            "Answer states 4,000 IU / 100 mcg as the adult UL",
            "Source cited is NIH Office of Dietary Supplements",
            "UL value is not confused with the RDA (~600 IU)",
        ],
    },
    {
        "id": "f3",
        "query": "What does research say about protein requirements for resistance-trained athletes?",
        "check": [
            "Answer gives a range of 1.6–2.2 g/kg body weight",
            "Source cited is the ISSN position stand (with year)",
            "No numbers outside the 1.6–2.2 g/kg range attributed to ISSN",
        ],
    },
    {
        "id": "f4",
        "query": "How much calcium do pregnant women need per day?",
        "check": [
            "Answer states 1,000 mg/day for pregnant adults",
            "Source cited is NIH Office of Dietary Supplements",
            "Answer distinguishes pregnant adults from pregnant teens (1,300 mg) if both chunks retrieved",
        ],
    },
    {
        "id": "f5",
        "query": "What does the Dietary Guidelines say about the Mediterranean diet?",
        "check": [
            "Answer mentions olive oil, fish, vegetables, or fruits",
            "Source cited is Dietary Guidelines for Americans 2020-2025",
            "No foods or claims attributed to DGA that aren't in the context below",
        ],
    },
]

_SEP = "=" * 72
_SUB = "-" * 72


def _wrap(text: str, width: int = 80, indent: str = "  ") -> str:
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


_NO_MULTI_QUERY: bool = False  # set by CLI arg; True = skip query expansion (saves ~3 Groq calls/case)


def run_case(case: dict) -> None:
    print(_SEP)
    print(f"[{case['id']}] {case['query']}")
    print(_SEP)

    # --- Retrieval ---
    if _NO_MULTI_QUERY:
        # Single-query mode: skips LLM expansion, uses only the original query.
        # Saves ~75% of Groq tokens at the cost of slightly lower recall.
        candidates = hybrid_search([case["query"]], top_k=30, max_per_source=12)
        chunks = rerank(case["query"], candidates, top_k=9)
    else:
        chunks = retrieve(case["query"])

    print(f"\nRETRIEVED CHUNKS ({len(chunks)} returned, showing top 5):")
    print(_SUB)
    for i, chunk in enumerate(chunks[:5], 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "Unknown")
        year = meta.get("year", "")
        score = chunk.get("rerank_score", "n/a")
        score_str = f"{score:.3f}" if isinstance(score, float) else str(score)
        print(f"  [{i}] Source: {source}  Year: {year}  Rerank: {score_str}")
        # Print first 200 chars of chunk text
        snippet = chunk.get("text", "")[:200].replace("\n", " ").strip()
        print(f"      {snippet}...")
    print()

    # --- Generation (with rate-limit retry) ---
    answer = generate(case["query"], chunks)

    print("GENERATED ANSWER:")
    print(_SUB)
    print(_wrap(answer, indent="  "))
    print()

    # --- Manual checklist ---
    print("FAITHFULNESS CHECKLIST (tick each manually):")
    for item in case["check"]:
        print(f"  [ ] {item}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Faithfulness spot-check")
    parser.add_argument("--skip", metavar="IDs", default="",
                        help="Comma-separated case IDs to skip (e.g. --skip f1)")
    parser.add_argument("--only", metavar="IDs", default="",
                        help="Comma-separated case IDs to run (e.g. --only f2,f3,f5)")
    parser.add_argument("--no-multi-query", action="store_true",
                        help="Skip LLM query expansion; use only the original query. "
                             "Saves ~3 Groq calls per case (~75%% tokens). "
                             "Use when close to the 100k/day token limit.")
    args = parser.parse_args()

    global _NO_MULTI_QUERY
    _NO_MULTI_QUERY = args.no_multi_query

    skip_ids = {s.strip() for s in args.skip.split(",") if s.strip()}
    only_ids = {s.strip() for s in args.only.split(",") if s.strip()}

    cases_to_run = [
        c for c in CASES
        if c["id"] not in skip_ids
        and (not only_ids or c["id"] in only_ids)
    ]

    print("\nZenic Faithfulness Spot-Check")
    print("Run the checks below and tick each box.\n")
    print("PASS criteria: all boxes ticked for all cases.")
    print("FAIL criteria: any number is fabricated, any source is invented,")
    print("               or the answer directly contradicts the context.\n")
    if skip_ids:
        print(f"Skipping: {', '.join(sorted(skip_ids))}\n")
    if only_ids:
        print(f"Running only: {', '.join(sorted(only_ids))}\n")
    if _NO_MULTI_QUERY:
        print("Mode: single-query (--no-multi-query) — query expansion disabled\n")

    for case in cases_to_run:
        run_case(case)

    print(_SEP)
    print(f"{len(cases_to_run)} case(s) complete. Record results in dev log / summary.txt.")
    print(_SEP)


if __name__ == "__main__":
    main()
