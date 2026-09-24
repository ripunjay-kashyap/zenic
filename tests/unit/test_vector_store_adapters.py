"""Vector store adapters, exercised against fake backend clients.

These cover the normalisation contract — score direction, filter translation,
payload shape — without needing a live Chroma or Qdrant instance.
"""
import uuid

import pytest

from zenic.errors import VectorStoreError
from zenic.rag.vector_store import _ChromaAdapter, _payload_to_chunk, _QdrantAdapter, _to_point_id

# ---------------------------------------------------------------------------
# Point ids
# ---------------------------------------------------------------------------

def test_to_point_id_is_deterministic():
    assert _to_point_id("usda_167782") == _to_point_id("usda_167782")


def test_to_point_id_differs_per_input():
    assert _to_point_id("usda_167782") != _to_point_id("usda_167783")


def test_to_point_id_is_a_valid_uuid_string():
    uuid.UUID(_to_point_id("issn_protein_01"))  # must not raise


# ---------------------------------------------------------------------------
# Chroma
# ---------------------------------------------------------------------------

class _FakeCollection:
    def __init__(self, space="cosine"):
        self.metadata = {"hnsw:space": space}
        self.query_kwargs = None
        self.get_kwargs = None
        self.deleted_ids = None
        self._distances = [0.1, 0.4]

    def query(self, **kwargs):
        self.query_kwargs = kwargs
        return {
            "documents": [["alpha", "beta"]],
            "metadatas": [[{"source": "USDA"}, {"source": "NIH_ODS"}]],
            "distances": [self._distances],
        }

    def get(self, **kwargs):
        self.get_kwargs = kwargs
        return {"ids": ["a", "b"], "documents": ["alpha", "beta"], "metadatas": [{}, {}]}

    def upsert(self, **kwargs):
        self.upserted = kwargs

    def delete(self, ids):
        self.deleted_ids = ids

    def count(self):
        return 2


class _FakeChromaClient:
    def __init__(self, collection):
        self._collection = collection

    def get_or_create_collection(self, name, metadata=None):
        return self._collection

    def get_collection(self, name):
        return self._collection


def _chroma(space="cosine"):
    collection = _FakeCollection(space)
    return _ChromaAdapter(_FakeChromaClient(collection), "zenic_knowledge"), collection


def test_chroma_cosine_distance_becomes_similarity():
    adapter, _ = _chroma("cosine")
    results = adapter.search([0.0] * 3, top_k=2)
    assert results[0]["vector_score"] == pytest.approx(0.9)
    assert results[1]["vector_score"] == pytest.approx(0.6)


def test_chroma_l2_distance_stays_positive_and_ordered():
    """Under L2 the old `1 - distance` produced negative pseudo-similarities."""
    adapter, _ = _chroma("l2")
    results = adapter.search([0.0] * 3, top_k=2)
    assert all(r["vector_score"] > 0 for r in results)
    assert results[0]["vector_score"] > results[1]["vector_score"]


def test_chroma_search_passes_the_filter_through():
    adapter, collection = _chroma()
    adapter.search([0.0] * 3, top_k=2, where={"source": "USDA"})
    assert collection.query_kwargs["where"] == {"source": "USDA"}


def test_chroma_delete_returns_the_number_removed():
    adapter, collection = _chroma()
    assert adapter.delete_by_source("USDA") == 2
    assert collection.deleted_ids == ["a", "b"]


def test_chroma_upsert_reports_the_missing_key():
    adapter, _ = _chroma()
    with pytest.raises(VectorStoreError, match="missing required key"):
        adapter.upsert([{"id": "a", "text": "t"}])  # no embedding/metadata


def test_chroma_upsert_of_nothing_is_a_no_op():
    adapter, _ = _chroma()
    adapter.upsert([])  # must not raise


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

class _Point:
    def __init__(self, payload, score=0.0):
        self.payload = payload
        self.score = score


class _CountResult:
    def __init__(self, count):
        self.count = count


class _Collections:
    def __init__(self, names):
        self.collections = [type("C", (), {"name": n})() for n in names]


class _FakeQdrantClient:
    def __init__(self, points=None, counts=None, existing=("zenic_knowledge",)):
        self._points = points or []
        self._counts = list(counts or [0])
        self._existing = existing
        self.scroll_kwargs = None
        self.query_kwargs = None
        self.delete_kwargs = None
        self.indexed_fields = []
        self.created_collections = []

    def get_collections(self):
        return _Collections(self._existing)

    def create_collection(self, collection_name, vectors_config):
        self.created_collections.append(collection_name)

    def create_payload_index(self, collection_name, field_name, field_schema, wait=False):
        self.indexed_fields.append(field_name)

    def query_points(self, **kwargs):
        self.query_kwargs = kwargs
        return type("R", (), {"points": self._points})()

    def scroll(self, **kwargs):
        self.scroll_kwargs = kwargs
        return self._points, None

    def delete(self, **kwargs):
        self.delete_kwargs = kwargs

    def count(self, **kwargs):
        return _CountResult(self._counts.pop(0) if len(self._counts) > 1 else self._counts[0])

    def collection_exists(self, name):
        return True


def test_qdrant_search_separates_text_from_metadata():
    client = _FakeQdrantClient([_Point({"text": "alpha", "source": "USDA", "year": "2024"}, 0.82)])
    adapter = _QdrantAdapter(client, "zenic_knowledge")
    result = adapter.search([0.0] * 3, top_k=1)[0]
    assert result["text"] == "alpha"
    assert result["metadata"] == {"source": "USDA", "year": "2024"}
    assert result["vector_score"] == pytest.approx(0.82)


def test_qdrant_sample_chunks_applies_the_filter():
    """scroll() previously ignored `where` entirely, returning arbitrary chunks."""
    client = _FakeQdrantClient([_Point({"text": "alpha", "source": "NIH_ODS"})])
    adapter = _QdrantAdapter(client, "zenic_knowledge")
    adapter.sample_chunks(where={"source": "NIH_ODS"}, n=5)
    assert client.scroll_kwargs["scroll_filter"] is not None
    assert client.scroll_kwargs["limit"] == 5


def test_qdrant_sample_chunks_without_a_filter_sends_none():
    client = _FakeQdrantClient([_Point({"text": "alpha"})])
    adapter = _QdrantAdapter(client, "zenic_knowledge")
    adapter.sample_chunks(n=3)
    assert client.scroll_kwargs["scroll_filter"] is None


def test_qdrant_delete_returns_a_count_not_an_operation_id():
    client = _FakeQdrantClient(counts=[12, 0])
    adapter = _QdrantAdapter(client, "zenic_knowledge")
    assert adapter.delete_by_source("USDA") == 12


def test_qdrant_search_failure_becomes_a_vector_store_error():
    class _Broken(_FakeQdrantClient):
        def query_points(self, **kwargs):
            raise RuntimeError("connection reset")

    adapter = _QdrantAdapter(_Broken(), "zenic_knowledge")
    with pytest.raises(VectorStoreError, match="Qdrant search failed"):
        adapter.search([0.0] * 3, top_k=1)


def test_health_check_is_false_when_the_backend_raises():
    class _Broken(_FakeQdrantClient):
        def collection_exists(self, name):
            raise RuntimeError("unauthorised")

    assert _QdrantAdapter(_Broken(), "zenic_knowledge").health_check() is False


def test_ensure_collection_creates_the_source_payload_index():
    """Qdrant 400s on any filtered query against an unindexed field, so
    delete_by_source and filtered search fail without this index."""
    client = _FakeQdrantClient()
    _QdrantAdapter(client, "zenic_knowledge").ensure_collection(384)
    assert "source" in client.indexed_fields


def test_payload_indexes_are_added_to_a_preexisting_collection():
    """A collection migrated before the indexes existed still needs them."""
    client = _FakeQdrantClient(existing=("zenic_knowledge",))
    _QdrantAdapter(client, "zenic_knowledge").ensure_collection(384)
    assert client.created_collections == []          # not recreated
    assert client.indexed_fields == ["source"]       # but indexed


def test_index_creation_failure_is_not_silently_ignored():
    """A permission or transport failure must not report a successful migration."""
    class _Existing(_FakeQdrantClient):
        def create_payload_index(self, **kwargs):
            raise RuntimeError("index already exists")

    with pytest.raises(VectorStoreError, match="payload indexes"):
        _QdrantAdapter(_Existing(), "zenic_knowledge").ensure_collection(384)


def test_payload_without_text_does_not_raise():
    """Points written by an older migration may lack the text field."""
    assert _payload_to_chunk({"source": "USDA"})["text"] == ""


def test_payload_none_is_tolerated():
    assert _payload_to_chunk(None) == {"text": "", "metadata": {}}
