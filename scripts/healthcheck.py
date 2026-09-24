"""Pre-flight check: is this deployment actually able to serve traffic?

    PYTHONPATH=. python scripts/healthcheck.py

Verifies configuration, vector-store reachability, corpus presence, and (with
--llm) that the model provider answers. Exits non-zero if anything required is
broken, so it can gate a deploy or run as a scheduled probe.
"""
from __future__ import annotations

import argparse
import sys

from zenic.config import get_settings
from zenic.errors import ZenicError

_OK = "PASS"
_FAIL = "FAIL"
_WARN = "WARN"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check that Zenic can serve traffic.")
    parser.add_argument(
        "--llm", action="store_true", help="also send a live probe to the model provider"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results: list[tuple[str, str, str]] = []

    settings = get_settings()
    results.append((_OK, "config", f"ENV={settings.env}, model={settings.groq_model}"))

    # --- credentials ---------------------------------------------------------
    try:
        settings.require_groq_api_key()
        results.append((_OK, "groq credentials", "present"))
    except ZenicError as exc:
        results.append((_FAIL, "groq credentials", str(exc)))

    if settings.usda_api_key:
        results.append((_OK, "usda credentials", "present (live fallback enabled)"))
    else:
        results.append((_WARN, "usda credentials", "absent — live food fallback disabled"))

    # --- vector store --------------------------------------------------------
    try:
        from zenic.rag.vector_store import get_vector_store

        store = get_vector_store()
        if not store.health_check():
            results.append(
                (
                    _FAIL,
                    "vector store",
                    "service unavailable or collection missing; check network, credentials, and collection",
                )
            )
        else:
            count = store.count()
            level = _OK if count > 0 else _FAIL
            results.append((level, "vector store", f"{count} points in {settings.collection_name}"))

            # Qdrant rejects every filtered query when the payload field has no
            # keyword index — unfiltered search still works, so this failure is
            # invisible until something tries to filter or re-ingest.
            try:
                store.sample_chunks(where={"source": "ISSN"}, n=1)
                results.append((_OK, "payload index", "filtered queries work"))
            except ZenicError:
                results.append(
                    (
                        _FAIL,
                        "payload index",
                        "filtered queries fail — no keyword index on 'source'; "
                        "run ensure_collection() or the migration script",
                    )
                )
    except ZenicError as exc:
        results.append((_FAIL, "vector store", str(exc)))

    # --- BM25 corpus ---------------------------------------------------------
    corpus = settings.bm25_corpus_path
    if corpus.exists():
        size_mb = corpus.stat().st_size / 1e6
        results.append((_OK, "bm25 corpus", f"{corpus} ({size_mb:.1f} MB)"))
    else:
        results.append((_WARN, "bm25 corpus", f"missing at {corpus} — vector-only retrieval"))

    # --- model provider ------------------------------------------------------
    if args.llm:
        from zenic.llm import chat_completion

        models = {"llm provider": settings.groq_model}
        if settings.groq_plan_model != settings.groq_model:
            models["plan model"] = settings.groq_plan_model
        for label, model in models.items():
            try:
                chat_completion(
                    [{"role": "user", "content": "Reply with the single word: ok"}],
                    purpose="healthcheck", model=model,
                )
                results.append((_OK, label, f"{model} responded"))
            except ZenicError as exc:
                results.append((_FAIL, label, str(exc)))

    # --- report --------------------------------------------------------------
    width = max(len(name) for _, name, _ in results)
    for level, name, detail in results:
        print(f"[{level}] {name.ljust(width)}  {detail}")

    failed = sum(1 for level, _, _ in results if level == _FAIL)
    print()
    print(f"{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
