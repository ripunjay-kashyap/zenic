"""Rebuild vectors from the bundled BM25 corpus without downloading source documents."""
from __future__ import annotations

import json

from zenic.config import get_settings
from zenic.rag.ingestion.indexer import index_documents


def main() -> None:
    settings = get_settings()
    with settings.bm25_corpus_path.open(encoding="utf-8") as handle:
        documents = json.load(handle)
    if not isinstance(documents, list) or not documents:
        raise ValueError("Corpus must be a non-empty list of documents")
    ids = set()
    for doc in documents:
        if not isinstance(doc, dict) or not all(key in doc for key in ("id", "text", "metadata")):
            raise ValueError("Each corpus document needs id, text, and metadata")
        if not isinstance(doc["id"], str) or not isinstance(doc["text"], str) or not doc["text"].strip() or not isinstance(doc["metadata"], dict):
            raise ValueError("Corpus IDs and passages must be text, with object metadata")
        if doc["id"] in ids:
            raise ValueError("Corpus IDs must be unique")
        ids.add(doc["id"])
    index_documents(documents)
    print(f"Indexed {len(documents)} documents for ENV={settings.env}")


if __name__ == "__main__":
    main()
