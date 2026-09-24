"""Vector store factory — ChromaDB for local development, Qdrant Cloud for production.

Both adapters implement the :class:`VectorStore` protocol and normalise their
results to a common shape::

    {"text": str, "metadata": dict, "vector_score": float}

``vector_score`` is always a cosine *similarity* in ``[-1, 1]`` where higher is
better, regardless of the distance metric the backend uses internally.
"""
from __future__ import annotations

import threading
import uuid
from typing import Any, Protocol, runtime_checkable

from zenic.config import get_settings
from zenic.errors import VectorStoreError
from zenic.logging_config import get_logger

logger = get_logger(__name__)

# Fixed namespace so a given chunk id always maps to the same Qdrant point id
# (makes re-running the migration an idempotent upsert rather than a duplicate insert).
# The literal is just any constant valid UUID — its only job is to stay fixed.
_POINT_ID_NAMESPACE = uuid.UUID("c3f1a2b4-1111-4000-8000-000000000001")

_store: VectorStore | None = None
_store_lock = threading.Lock()


def _to_point_id(raw_id: str) -> str:
    """Map a string chunk id (e.g. 'usda_167782') to a deterministic UUID string.
    Qdrant point IDs must be an unsigned int or a UUID — not an arbitrary string."""
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, raw_id))


@runtime_checkable
class VectorStore(Protocol):
    def ensure_collection(self, vector_size: int) -> None: ...
    def upsert(self, documents: list[dict]) -> None: ...
    def search(
        self, query_embedding: list[float], top_k: int, where: dict | None = None
    ) -> list[dict]: ...
    def sample_chunks(self, where: dict | None = None, n: int = 10) -> list[dict]: ...
    def delete_by_source(self, source: str) -> int: ...
    def count(self) -> int: ...
    def health_check(self) -> bool: ...


def get_vector_store() -> VectorStore:
    """Return the process-wide vector store for the current ``ENV``.

    Cached: building a client opens a connection (Qdrant) or a local database
    handle (Chroma), neither of which should happen per query.
    """
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                settings = get_settings()
                _store = (
                    _build_qdrant_store() if settings.is_production else _build_chroma_store()
                )
                logger.info(
                    "vector store initialised",
                    extra={"backend": type(_store).__name__, "env": settings.env},
                )
    return _store


def reset_vector_store() -> None:
    """Drop the cached store. For tests and after a settings reload."""
    global _store
    with _store_lock:
        _store = None


def _build_chroma_store() -> VectorStore:
    settings = get_settings()
    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise VectorStoreError(
            "chromadb is required for ENV=development but is not installed."
        ) from exc
    try:
        client = chromadb.PersistentClient(path=str(settings.chroma_path))
    except Exception as exc:
        raise VectorStoreError(
            "Could not open the local Chroma database; check its path and permissions."
        ) from exc
    return _ChromaAdapter(client, settings.collection_name)


def _build_qdrant_store() -> VectorStore:
    settings = get_settings()
    url, api_key = settings.require_qdrant()
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise VectorStoreError(
            "qdrant-client is required for ENV=production but is not installed."
        ) from exc
    try:
        client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=max(1, int(settings.vector_store_timeout_seconds)),
        )
    except Exception as exc:
        raise VectorStoreError("Could not connect to Qdrant; check server configuration and connectivity.") from exc
    return _QdrantAdapter(client, settings.collection_name)


class _ChromaAdapter:
    """Local development backend.

    Chroma defaults to squared L2 distance, but the old ``1 - distance``
    conversion here only makes sense for cosine space — under L2 it produced
    large negative "similarities" that the hybrid merge then compared directly
    against BM25 scores. New collections are created in cosine space; existing
    ones keep whatever space they were built with, and the distance is converted
    accordingly by :meth:`_to_similarity`.
    """

    def __init__(self, client: Any, collection: str):
        self._collection_name = collection
        try:
            self._col = client.get_or_create_collection(
                collection, metadata={"hnsw:space": "cosine"}
            )
        except Exception:
            # A collection created with different metadata already exists — reuse it.
            try:
                self._col = client.get_collection(collection)
            except Exception as exc:
                raise VectorStoreError(
                    "Could not open the Chroma collection; check database configuration."
                ) from exc

        self._space = str((self._col.metadata or {}).get("hnsw:space", "l2")).lower()
        if self._space != "cosine":
            logger.warning(
                "chroma collection uses a non-cosine distance; scores are converted heuristically",
                extra={"collection": collection, "space": self._space},
            )

    def _to_similarity(self, distance: float) -> float:
        """Convert a backend distance into a higher-is-better similarity."""
        distance = float(distance)
        if self._space == "cosine":
            return 1.0 - distance
        if self._space == "ip":  # inner product: already higher-is-better, negated
            return 1.0 - distance
        # l2 / squared-l2: unbounded above, so map monotonically into (0, 1].
        return 1.0 / (1.0 + distance)

    def ensure_collection(self, vector_size: int) -> None:
        # Chroma creates the collection in __init__ via get_or_create_collection and
        # infers dimensionality from the first upsert; this is a no-op so the adapter
        # satisfies the VectorStore protocol.
        return None

    def upsert(self, documents: list[dict]) -> None:
        if not documents:
            return
        try:
            self._col.upsert(
                ids=[d["id"] for d in documents],
                documents=[d["text"] for d in documents],
                embeddings=[d["embedding"] for d in documents],
                metadatas=[{**d["metadata"], "chunk_id": d["id"]} for d in documents],
            )
        except KeyError as exc:
            raise VectorStoreError(f"Document is missing required key {exc}.") from exc
        except Exception as exc:
            raise VectorStoreError("Chroma upsert failed; check server configuration and connectivity.") from exc

    def search(
        self, query_embedding: list[float], top_k: int, where: dict | None = None
    ) -> list[dict]:
        kwargs: dict[str, Any] = {"query_embeddings": [query_embedding], "n_results": top_k}
        if where:
            kwargs["where"] = where
        try:
            results = self._col.query(**kwargs)
        except Exception as exc:
            raise VectorStoreError("Chroma search failed; check server configuration and connectivity.") from exc

        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        return [
            {
                "text": doc,
                "metadata": {**(meta or {}), **({"chunk_id": chunk_id} if chunk_id else {})},
                "vector_score": self._to_similarity(dist),
            }
            for doc, meta, dist, chunk_id in zip(
                documents, metadatas, distances,
                (results.get("ids") or [[None] * len(documents)])[0], strict=False
            )
        ]

    def sample_chunks(self, where: dict | None = None, n: int = 10) -> list[dict]:
        kwargs: dict[str, Any] = {"limit": n}
        if where:
            kwargs["where"] = where
        try:
            results = self._col.get(**kwargs)
        except Exception as exc:
            raise VectorStoreError("Chroma scan failed; check server configuration and connectivity.") from exc
        return [
            {"text": doc, "metadata": dict(meta or {})}
            for doc, meta in zip(
                results.get("documents") or [], results.get("metadatas") or [], strict=False
            )
        ]

    def delete_by_source(self, source: str) -> int:
        try:
            results = self._col.get(where={"source": source}, include=[])
            ids = results.get("ids") or []
            if ids:
                self._col.delete(ids=ids)
        except Exception as exc:
            raise VectorStoreError("Chroma delete failed; check server configuration and connectivity.") from exc
        logger.info("deleted chunks", extra={"source": source, "count": len(ids)})
        return len(ids)

    def count(self) -> int:
        try:
            return int(self._col.count())
        except Exception as exc:
            raise VectorStoreError("Chroma count failed; check server configuration and connectivity.") from exc

    def health_check(self) -> bool:
        try:
            self._col.count()
            return True
        except Exception:
            logger.warning("chroma health check failed")
            return False


class _QdrantAdapter:
    """Production backend (Qdrant Cloud), cosine distance."""

    def __init__(self, client: Any, collection: str):
        self._client = client
        self._collection = collection

    def _filter(self, where: dict | None):
        """Translate a flat {field: value} mapping into a Qdrant filter."""
        if not where:
            return None
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        return Filter(
            must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in where.items()]
        )

    #: Payload fields that must carry a keyword index. Unlike Chroma, Qdrant
    #: rejects *any* filtered query on an unindexed field with a 400 — so
    #: without this, delete_by_source() and every filtered search fail in
    #: production while unfiltered search works fine, hiding the problem.
    _INDEXED_FIELDS = ("source",)

    def ensure_payload_indexes(self) -> None:
        """Create the keyword indexes filtered queries depend on. Idempotent."""
        from qdrant_client.models import PayloadSchemaType

        for field in self._INDEXED_FIELDS:
            try:
                self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                    wait=True,
                )
                logger.info("payload index ensured", extra={"field": field})
            except Exception as exc:
                raise VectorStoreError("Could not ensure Qdrant payload indexes; check permissions and connectivity.") from exc

    def ensure_collection(self, vector_size: int) -> None:
        """Create the collection and its payload indexes if not already present."""
        from qdrant_client.models import Distance, VectorParams

        try:
            existing = {c.name for c in self._client.get_collections().collections}
            if self._collection not in existing:
                self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
                logger.info(
                    "created qdrant collection",
                    extra={"collection": self._collection, "vector_size": vector_size},
                )
        except Exception as exc:
            raise VectorStoreError("Could not ensure Qdrant collection; check server configuration and connectivity.") from exc

        # Run for existing collections too — a collection migrated before these
        # indexes existed still needs them.
        self.ensure_payload_indexes()

    def upsert(self, documents: list[dict]) -> None:
        if not documents:
            return
        from qdrant_client.models import PointStruct

        try:
            points = [
                PointStruct(
                    id=_to_point_id(d["id"]),
                    vector=d["embedding"],
                    payload={**d["metadata"], "text": d["text"], "chunk_id": d["id"]},
                )
                for d in documents
            ]
        except KeyError as exc:
            raise VectorStoreError(f"Document is missing required key {exc}.") from exc
        try:
            self._client.upsert(collection_name=self._collection, points=points)
        except Exception as exc:
            raise VectorStoreError("Qdrant upsert failed; check server configuration and connectivity.") from exc

    def search(
        self, query_embedding: list[float], top_k: int, where: dict | None = None
    ) -> list[dict]:
        try:
            results = self._client.query_points(
                collection_name=self._collection,
                query=query_embedding,
                limit=top_k,
                query_filter=self._filter(where),
                with_payload=True,
            ).points
        except Exception as exc:
            raise VectorStoreError("Qdrant search failed; check server configuration and connectivity.") from exc
        return [_payload_to_chunk(r.payload, score=r.score) for r in results]

    def sample_chunks(self, where: dict | None = None, n: int = 10) -> list[dict]:
        try:
            results, _ = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=self._filter(where),
                limit=n,
                with_payload=True,
            )
        except Exception as exc:
            raise VectorStoreError("Qdrant scan failed; check server configuration and connectivity.") from exc
        return [_payload_to_chunk(r.payload) for r in results]

    def delete_by_source(self, source: str) -> int:
        """Delete every point from ``source``. Returns the number removed."""
        query_filter = self._filter({"source": source})
        try:
            before = self._count_matching(query_filter)
            self._client.delete(
                collection_name=self._collection,
                points_selector=query_filter,
                wait=True,
            )
            after = self._count_matching(query_filter)
        except Exception as exc:
            raise VectorStoreError("Qdrant delete failed; check server configuration and connectivity.") from exc
        removed = max(before - after, 0)
        logger.info("deleted chunks", extra={"source": source, "count": removed})
        return removed

    def _count_matching(self, query_filter) -> int:
        return int(
            self._client.count(
                collection_name=self._collection,
                count_filter=query_filter,
                exact=True,
            ).count
        )

    def count(self) -> int:
        try:
            return int(self._client.count(collection_name=self._collection, exact=True).count)
        except Exception as exc:
            raise VectorStoreError("Qdrant count failed; check server configuration and connectivity.") from exc

    def health_check(self) -> bool:
        """True when the collection exists and is reachable."""
        try:
            return self._client.collection_exists(self._collection)
        except Exception:
            logger.warning("qdrant health check failed")
            return False


def _payload_to_chunk(payload: dict | None, score: float | None = None) -> dict:
    """Normalise a Qdrant payload into the common chunk shape.

    Points written by an older migration may lack ``text``; treat that as empty
    rather than raising a KeyError in the middle of a user's query.
    """
    payload = payload or {}
    chunk = {
        "text": payload.get("text", ""),
        "metadata": {k: v for k, v in payload.items() if k != "text"},
    }
    if score is not None:
        chunk["vector_score"] = float(score)
    return chunk
