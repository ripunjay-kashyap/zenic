"""
RAGAS automated evaluation.

Runs spot-check queries through the RAG pipeline (retrieve + generate) and
scores the results with RAGAS using Gemma 4 as the judge LLM.

Metrics
-------
  faithfulness      — is the answer grounded in the retrieved context?
  context_precision — are the retrieved chunks relevant to the question?

Evaluation targets: faithfulness > 0.85, context_precision > 0.75

Usage
-----
  PYTHONPATH=. python scripts/ragas_eval.py
  PYTHONPATH=. python scripts/ragas_eval.py --skip p1_001,p1_002 --no-multi-query
  PYTHONPATH=. python scripts/ragas_eval.py --only p1_003,p1_004,p1_006

Requires: GROQ_API_KEY + GOOGLE_API_KEY in .env, populated vector DB + BM25 corpus.

All evaluation cases run by default. Use --skip only for an explicitly scoped
investigation; excluded IDs are printed and saved with the result.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv

# Silence deprecation noise from older google-generativeai/RAGAS combos
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

# ---------------------------------------------------------------------------
# Eval dataset (all spot-check cases that don't require USDA re-ingest)
# ---------------------------------------------------------------------------

_EVAL_DATA_PATH = Path("eval_data/pillar1_spot_check.json")

def _load_cases(skip_ids: set[str], only_ids: set[str]) -> list[dict]:
    cases = json.loads(_EVAL_DATA_PATH.read_text(encoding="utf-8"))
    if only_ids:
        cases = [c for c in cases if c["id"] in only_ids]
    if skip_ids:
        cases = [c for c in cases if c["id"] not in skip_ids]
    return cases


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _run_case(case: dict, multi_query: bool) -> dict:
    """Run one spot-check case through the RAG pipeline and return eval row."""
    from zenic.rag.pipeline import generate, retrieve

    query = case["query"]

    # retrieve() internally runs multi-query expansion unless we patch it out
    if not multi_query:
        # Bypass multi-query: call hybrid_search + rerank directly with same
        # params as retrieve() so chunk quality is identical.
        from zenic.rag.pipeline import hybrid_search, rerank
        candidates = hybrid_search([query], top_k=30, max_per_source=12)
        chunks = rerank(query, candidates, top_k=7)
    else:
        chunks = retrieve(query)

    answer = generate(query, chunks)

    # RAGAS expects contexts as list of strings
    contexts = [c.get("text", c.get("content", "")) for c in chunks]

    return {
        "question": query,
        "answer": answer,
        "contexts": contexts,
    }


# ---------------------------------------------------------------------------
# RAGAS setup
# ---------------------------------------------------------------------------

def _build_llm():
    import os

    from langchain_google_genai import ChatGoogleGenerativeAI
    from ragas.llms import LangchainLLMWrapper

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise OSError("GOOGLE_API_KEY not set in environment / .env")

    gemini = ChatGoogleGenerativeAI(
        model="gemma-4-31b-it",
        google_api_key=api_key,
        max_output_tokens=4096,
        timeout=300,
        temperature=0,
        # Pass thinking_config via model_kwargs so langchain-google-genai forwards
        # it directly to the API call — thinking_budget=0 suppresses Gemma 4's
        # reasoning preamble so RAGAS receives clean JSON output.
        model_kwargs={"thinking_config": {"thinking_budget": 0}},
    )
    return LangchainLLMWrapper(gemini)


def _build_embeddings():
    import os

    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    api_key = os.environ.get("GOOGLE_API_KEY")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key,
    )
    return LangchainEmbeddingsWrapper(embeddings)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS automated evaluation")
    parser.add_argument("--skip", metavar="IDs", default="",
                        help="Comma-separated case IDs to skip (e.g. --skip p1_001,p1_002)")
    parser.add_argument("--only", metavar="IDs", default="",
                        help="Comma-separated case IDs to run (e.g. --only p1_003,p1_004)")
    parser.add_argument("--no-multi-query", action="store_true",
                        help="Bypass LLM query expansion (~3x fewer Groq tokens)")
    args = parser.parse_args()

    only_ids = {s.strip() for s in args.only.split(",") if s.strip()}
    skip_ids = {s.strip() for s in args.skip.split(",") if s.strip()}
    multi_query = not args.no_multi_query

    cases = _load_cases(skip_ids, only_ids)
    if not cases:
        print("No cases to run after applying --skip / --only filters.")
        raise SystemExit(1)

    print("\nZenic RAGAS Evaluation")
    print("Judge LLM : gemma-4-31b-it/thinking_budget=0 (GOOGLE_API_KEY)")
    print("Metrics   : faithfulness (target >0.85), context_precision/no-ref (target >0.75)")
    print(f"Cases     : {len(cases)}")
    if skip_ids:
        print(f"Skipped   : {', '.join(sorted(skip_ids))}")
    if not multi_query:
        print("Mode      : single-query (--no-multi-query)")
    print("=" * 72)

    # --- Step 1: collect pipeline outputs ----------------------------------
    rows = {"question": [], "answer": [], "contexts": []}
    run_ids: list[str] = []
    failed_ids: list[str] = []

    for case in cases:
        print(f"\n[{case['id']}] {case['query']}")
        try:
            row = _run_case(case, multi_query=multi_query)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            failed_ids.append(case["id"])
            continue

        print(f"  Answer   : {row['answer'][:120].replace(chr(10), ' ')}{'...' if len(row['answer']) > 120 else ''}")
        print(f"  Contexts : {len(row['contexts'])} chunks retrieved")
        rows["question"].append(row["question"])
        rows["answer"].append(row["answer"])
        rows["contexts"].append(row["contexts"])
        run_ids.append(case["id"])

    if not rows["question"]:
        print("\nNo rows collected — aborting RAGAS scoring.")
        raise SystemExit(1)

    # --- Step 2: score with RAGAS -----------------------------------------
    print(f"\n{'=' * 72}")
    print(f"Scoring {len(rows['question'])} case(s) with RAGAS...")

    import numpy as np
    from datasets import Dataset as HFDataset
    from ragas import evaluate
    from ragas.metrics import faithfulness
    from ragas.metrics._context_precision import LLMContextPrecisionWithoutReference
    from ragas.run_config import RunConfig

    context_precision_nr = LLMContextPrecisionWithoutReference()

    dataset = HFDataset.from_dict(rows)
    llm = _build_llm()

    # max_workers=2: stay within Gemma 4's 15 RPM free-tier limit.
    # thinking_budget=0 removes multi-KB preamble so calls complete well within timeout.
    run_config = RunConfig(timeout=300, max_retries=5, max_wait=90, max_workers=2)

    result = evaluate(
        dataset,
        metrics=[faithfulness, context_precision_nr],
        llm=llm,
        run_config=run_config,
        raise_exceptions=False,
    )

    # --- Step 3: report ----------------------------------------------------
    print(f"\n{'=' * 72}")
    print("RAGAS Results")
    print("=" * 72)

    df = result.to_pandas()
    # Show available averages, but incomplete scores cannot pass the evaluation.
    faith_key = "faithfulness"
    prec_key = next((c for c in df.columns if "context_precision" in c), None)

    faith_score = float(np.nanmean(df[faith_key])) if faith_key in df.columns else float("nan")
    prec_score = float(np.nanmean(df[prec_key])) if prec_key else float("nan")

    faith_status = "PASS ✅" if faith_score >= 0.85 else "FAIL ❌"
    prec_status  = "PASS ✅" if prec_score >= 0.75 else "FAIL ❌"

    print(f"  faithfulness      : {faith_score:.3f}  (target >0.85)  {faith_status}")
    print(f"  context_precision : {prec_score:.3f}  (target >0.75)  {prec_status}")
    print()

    complete = (
        not failed_ids
        and len(df) == len(run_ids)
        and faith_key in df.columns
        and prec_key is not None
        and bool(np.isfinite(df[faith_key].to_numpy(dtype=float)).all())
        and bool(np.isfinite(df[prec_key].to_numpy(dtype=float)).all())
    )
    overall = complete and faith_score >= 0.85 and prec_score >= 0.75
    if failed_ids:
        print(f"Pipeline failures: {', '.join(failed_ids)}")
    if not complete:
        print("Some selected cases or judge scores are missing; evaluation cannot pass.")
    if overall:
        print("OVERALL: PASS ✅  RAGAS targets met for the scored cases.")
    else:
        print("OVERALL: FAIL ❌  One or more targets not met.")

    print(f"\n{'=' * 72}")
    print("Per-case scores:")
    per_case = []
    for i, row in df.iterrows():
        case_id = run_ids[i] if i < len(run_ids) else f"case_{i}"
        f_val = row.get(faith_key, float("nan"))
        p_val = row.get(prec_key, float("nan")) if prec_key else float("nan")
        print(f"  [{case_id}]  faithfulness={f_val:.3f}  context_precision={p_val:.3f}")
        per_case.append({
            "id": case_id,
            "faithfulness": None if np.isnan(f_val) else round(float(f_val), 4),
            "context_precision": None if np.isnan(p_val) else round(float(p_val), 4),
        })

    # --- Step 4: save JSON results -----------------------------------------
    import datetime
    results_dir = Path("eval_results")
    results_dir.mkdir(exist_ok=True)
    json_path = results_dir / "ragas_latest.json"
    json_payload = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "judge": "gemma-4-31b-it",
        "mode": "single-query" if not multi_query else "multi-query",
        "skipped": sorted(skip_ids),
        "failed": failed_ids,
        "evaluated": run_ids,
        "averages": {
            "faithfulness": None if np.isnan(faith_score) else round(faith_score, 4),
            "context_precision": None if np.isnan(prec_score) else round(prec_score, 4),
        },
        "targets": {"faithfulness": 0.85, "context_precision": 0.75},
        "overall_pass": overall,
        "per_case": per_case,
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    print(f"\nResults saved → {json_path}")
    if not overall:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
