"""
Shared indexing logic — embeds documents and upserts them into the vector store.
All ingestion modules produce list[dict] with keys: id, text, metadata.
This module adds embeddings and persists to the store.
"""
from collections.abc import Callable

from zenic.config import get_settings
from zenic.logging_config import get_logger
from zenic.rag.pipeline import embed_texts
from zenic.rag.vector_store import get_vector_store

logger = get_logger(__name__)

_BATCH_SIZE = 64


def index_documents(
    documents: list[dict],
    progress_cb: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """
    Embed and upsert a list of {id, text, metadata} documents.
    Returns the same list (for BM25 index building in the caller).
    Each document must have a unique `id`.
    """
    store = get_vector_store()
    store.ensure_collection(get_settings().embedding_dimensions)
    total = len(documents)
    logger.info("indexing documents", extra={"documents": total})

    for start in range(0, total, _BATCH_SIZE):
        batch = documents[start : start + _BATCH_SIZE]
        embeddings = embed_texts([d["text"] for d in batch])
        enriched = [
            {**d, "embedding": emb}
            for d, emb in zip(batch, embeddings, strict=True)
        ]
        store.upsert(enriched)
        if progress_cb:
            progress_cb(min(start + _BATCH_SIZE, total), total)

    return documents
