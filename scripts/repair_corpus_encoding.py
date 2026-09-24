"""Repair mojibake in the chunk corpus, and optionally re-upsert the fixed chunks.

    PYTHONPATH=. python scripts/repair_corpus_encoding.py            # report only
    PYTHONPATH=. python scripts/repair_corpus_encoding.py --write    # fix the corpus file
    PYTHONPATH=. python scripts/repair_corpus_encoding.py --write --reindex

Background: ISSN chunks were ingested by a run that read the paper metadata
without an explicit encoding, so UTF-8 bytes were decoded as cp1252 — "Jäger"
became "JÃ¤ger" in the citation prefix of 44 chunks. The ingestion bug is fixed
in zenic/rag/ingestion/issn.py; this repairs the corpus that bug already
produced, so a full re-ingest is not required.

The repair is the standard round-trip: re-encode the text back to the bytes it
was mistakenly decoded from, then decode those bytes as UTF-8. Chunks that do
not round-trip cleanly are left untouched.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

from zenic.config import get_settings
from zenic.logging_config import get_logger

logger = get_logger("repair_corpus_encoding")

#: Signature of UTF-8 bytes decoded as a single-byte codec.
_MOJIBAKE = re.compile(r"[ÃÂ][\x80-\xbf -ÿ]")


#: A maximal run of latin-1-range characters beginning with the signature. Runs
#: are repaired individually rather than round-tripping the whole chunk: a chunk
#: that mixes corrupted text with correctly-decoded characters (an em dash, say)
#: cannot be re-encoded as a whole, which left 43 of 44 chunks unrepairable when
#: this was attempted string-at-a-time.
_MOJIBAKE_RUN = re.compile("[\u00c3\u00c2][\u0080-\u00ff]+")


def looks_mojibaked(text: str) -> bool:
    return bool(_MOJIBAKE.search(text))


def _decode_run(match):
    run = match.group(0)
    try:
        return run.encode("latin-1", errors="strict").decode("utf-8", errors="strict")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return run  # not actually mojibake — leave it alone


def repair(text: str) -> str | None:
    """Return the corrected text, or None if the signature could not be removed."""
    candidate = _MOJIBAKE_RUN.sub(_decode_run, text)
    if candidate == text or looks_mojibaked(candidate):
        return None
    return candidate

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair mojibake in the chunk corpus.")
    parser.add_argument("--write", action="store_true", help="write the repaired corpus back")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="re-embed and upsert the repaired chunks into the vector store",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = get_settings().bm25_corpus_path
    if not path.exists():
        logger.error("corpus not found", extra={"path": str(path)})
        return 1

    with open(path, encoding="utf-8") as f:
        corpus = json.load(f)

    repaired: list[dict] = []
    unrepairable = 0
    for chunk in corpus:
        text = chunk.get("text", "")
        if not looks_mojibaked(text):
            continue
        fixed = repair(text)
        if fixed is None:
            unrepairable += 1
            continue
        chunk["text"] = fixed
        repaired.append(chunk)

    logger.info(
        "scan complete",
        extra={"total": len(corpus), "repaired": len(repaired), "unrepairable": unrepairable},
    )
    for chunk in repaired[:3]:
        print(f"  {chunk.get('id')}: {chunk['text'][:80]!r}")

    if not repaired:
        return 0

    if not args.write:
        print("\nDry run — re-run with --write to apply.")
        return 0

    with open(path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)
    logger.info("corpus rewritten", extra={"path": str(path), "chunks": len(repaired)})

    if args.reindex:
        from zenic.rag.pipeline import embed_texts
        from zenic.rag.vector_store import get_vector_store

        store = get_vector_store()
        embeddings = embed_texts([c["text"] for c in repaired])
        store.upsert(
            [
                {
                    "id": c["id"],
                    "text": c["text"],
                    "metadata": c.get("metadata") or {},
                    "embedding": embedding,
                }
                for c, embedding in zip(repaired, embeddings, strict=True)
            ]
        )
        logger.info("repaired chunks re-indexed", extra={"chunks": len(repaired)})

    return 0


if __name__ == "__main__":
    sys.exit(main())
