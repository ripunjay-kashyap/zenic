"""One-time migration: rebuild the production Qdrant collection from the BM25 corpus.

Run locally (NOT in the container), once, before the first deploy — and again
after rotating to a new Qdrant cluster, which starts empty:

    PYTHONPATH=. python scripts/migrate_to_qdrant.py
    PYTHONPATH=. python scripts/migrate_to_qdrant.py --dry-run   # check config only

Requires QDRANT_URL and QDRANT_API_KEY in .env. Embeds every chunk with the
configured embedding model and upserts to Qdrant Cloud.
Idempotent: re-running upserts the same point ids rather than duplicating.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Force the Qdrant adapter regardless of the ambient ENV. This must happen
# before zenic.config is imported, since settings are cached on first access.
os.environ["ENV"] = "production"

from zenic.config import get_settings
from zenic.errors import ZenicError
from zenic.logging_config import get_logger
from zenic.rag.pipeline import embed_texts
from zenic.rag.vector_store import get_vector_store

logger = get_logger("migrate_to_qdrant")

BATCH = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate the chunk corpus into Qdrant Cloud.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate configuration and corpus, then exit without writing",
    )
    parser.add_argument(
        "--batch", type=int, default=BATCH, help=f"embedding batch size (default {BATCH})"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch <= 0:
        raise ValueError("--batch must be positive")
    settings = get_settings()

    corpus_path = settings.bm25_corpus_path
    if not corpus_path.exists():
        logger.error("corpus not found", extra={"path": str(corpus_path)})
        return 1

    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)
    logger.info("corpus loaded", extra={"chunks": len(corpus), "path": str(corpus_path)})

    invalid = [i for i, c in enumerate(corpus) if not c.get("id") or not c.get("text")]
    if invalid:
        logger.error(
            "corpus has entries without an id or text",
            extra={"count": len(invalid), "first_index": invalid[0]},
        )
        return 1

    if args.dry_run:
        logger.info("dry run complete — nothing written")
        return 0

    store = get_vector_store()
    store.ensure_collection(settings.embedding_dimensions)
    logger.info(
        "collection ready",
        extra={
            "collection": settings.collection_name,
            "vector_size": settings.embedding_dimensions,
            "existing_points": store.count(),
        },
    )

    started = time.perf_counter()
    total = 0
    for start in range(0, len(corpus), args.batch):
        batch = corpus[start : start + args.batch]
        embeddings = embed_texts([c["text"] for c in batch])
        store.upsert(
            [
                {
                    "id": c["id"],
                    "text": c["text"],
                    "metadata": c["metadata"] if isinstance(c.get("metadata"), dict) else {},
                    "embedding": embedding,
                }
                for c, embedding in zip(batch, embeddings, strict=True)
            ]
        )
        total += len(batch)
        if total % (args.batch * 10) == 0 or total == len(corpus):
            logger.info("upsert progress", extra={"progress": f"{total}/{len(corpus)}"})

    logger.info(
        "migration complete",
        extra={
            "chunks": total,
            "collection": settings.collection_name,
            "points_in_collection": store.count(),
            "elapsed_s": round(time.perf_counter() - started, 1),
        },
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ZenicError as exc:
        logger.error("migration failed: %s", exc)
        sys.exit(1)
