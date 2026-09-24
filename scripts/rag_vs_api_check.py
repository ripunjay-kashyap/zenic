"""
Check RAG versus live USDA routing for the bundled corpus.

Requires configured Groq and USDA credentials and a populated vector store.
These fixed expectations must be reviewed if the corpus changes. Missing foods
may use relevant API evidence or abstain if that evidence is unavailable. Any
route mismatch exits nonzero; this is a functional check, not a quality score.
"""
import json

from dotenv import load_dotenv

load_dotenv()
from zenic.agent.messages import message_content
from zenic.agent.trace import run_with_trace
from zenic.rag.pipeline import NO_EVIDENCE_RESPONSE

_CASES = [
    # --- Confirmed IN-INDEX (should use RAG, score >> 0.5) ---
    {"query": "protein in 100g chicken breast",            "should_use_rag": True},  # USDA
    {"query": "vitamin D upper intake level",              "should_use_rag": True},  # NIH ODS
    {"query": "ISSN protein recommendations for athletes", "should_use_rag": True},  # ISSN paper
    {"query": "barbell row muscles worked",                "should_use_rag": True},  # wger
    # --- Confirmed NOT IN INDEX (should trigger API fallback, score < 0.5) ---
    {"query": "calories in a medium banana",               "should_use_rag": False}, # absent from USDA subset
    {"query": "macros in boiled jackfruit seeds",          "should_use_rag": False}, # absent from all sources
]


def main():
    stats = {"total": 0, "correctly_used_rag": 0, "correctly_used_api": 0, "correctly_abstained": 0,
             "false_api_fallback": 0, "false_rag_attempt": 0,
             "missing_rag_route": 0, "failures": []}

    for case in _CASES:
        trace = run_with_trace(case["query"])
        tools = trace["tools_called"]
        used_rag = "rag_retrieval" in tools
        used_api = "usda_api" in tools or "wger_api" in tools
        messages = trace.get("final_state", {}).get("messages", [])
        answer = message_content(messages[-1]) if messages else ""
        stats["total"] += 1

        if case["should_use_rag"] and used_rag and not used_api:
            stats["correctly_used_rag"] += 1
        elif not case["should_use_rag"] and used_api:
            stats["correctly_used_api"] += 1
        elif not case["should_use_rag"] and used_rag and answer == NO_EVIDENCE_RESPONSE:
            stats["correctly_abstained"] += 1
        elif case["should_use_rag"]:
            stats["false_api_fallback" if used_api else "missing_rag_route"] += 1
            stats["failures"].append({"query": case["query"], "expected": "rag", "actual": tools})
        else:
            stats["false_rag_attempt"] += 1
            stats["failures"].append({"query": case["query"], "expected": "api", "actual": tools})

    in_scope = sum(1 for c in _CASES if c["should_use_rag"])
    rag_rate = stats["correctly_used_rag"] / in_scope if in_scope else 0

    print(json.dumps(stats, indent=2))
    print(f"\nRAG HIT RATE (in-scope queries): {rag_rate:.0%}")
    print(f"FALSE API FALLBACKS:             {stats['false_api_fallback']}")

    passed = not stats["failures"]
    print("\nPASS — all routing checks passed" if passed else "\nFAIL — routing mismatches found")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
