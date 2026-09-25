"""Hybrid-search merge, source diversity, tokenisation, and prompt selection.

No models are loaded: the embedder and vector store are replaced with fakes.
"""
import pytest

from zenic.rag import pipeline


@pytest.fixture
def fake_search(monkeypatch):
    """Replace embedding and vector search; return a setter for the results."""
    monkeypatch.setattr(pipeline, "embed_texts", lambda texts: [[0.0, 0.0] for _ in texts])
    monkeypatch.setattr(pipeline, "_bm25_index", None)
    monkeypatch.setattr(pipeline, "_bm25_corpus", None)
    monkeypatch.setattr(pipeline, "_bm25_load_attempted", True)

    holder = {"results": []}

    class _Store:
        def search(self, query_embedding, top_k, where=None):
            return [dict(r) for r in holder["results"]]

    monkeypatch.setattr(pipeline, "get_vector_store", lambda: _Store())
    return holder


def _chunk(chunk_id, source, score, text=None):
    return {
        "text": text or f"text for {chunk_id}",
        "metadata": {"chunk_id": chunk_id, "source": source},
        "vector_score": score,
    }


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def test_tokenizer_strips_punctuation():
    """`"protein,".split()` never matched the corpus token "protein"."""
    assert pipeline._tokenize("How much protein, exactly?") == [
        "how", "much", "protein", "exactly"
    ]


def test_tokenizer_keeps_decimals_and_hyphens():
    assert pipeline._tokenize("1.6 g/kg bge-small") == ["1.6", "g", "kg", "bge-small"]


def test_tokenizer_is_case_insensitive():
    assert pipeline._tokenize("Vitamin D") == pipeline._tokenize("vitamin d")


# ---------------------------------------------------------------------------
# Candidate identity
# ---------------------------------------------------------------------------

def test_chunks_sharing_a_prefix_are_not_collapsed(fake_search):
    """The old 80-character prefix key merged distinct entries with common openings."""
    shared = "Beef, ground, raw, nutrient values per 100 g of edible portion as reported"
    fake_search["results"] = [
        _chunk("usda_1", "USDA", 0.9, text=shared + " ... 80% lean"),
        _chunk("usda_2", "USDA", 0.8, text=shared + " ... 93% lean"),
    ]
    assert len(pipeline.hybrid_search(["beef"], top_k=10, max_per_source=10)) == 2


def test_the_same_chunk_across_variants_is_deduplicated(fake_search):
    fake_search["results"] = [_chunk("usda_1", "USDA", 0.5)]
    results = pipeline.hybrid_search(["a", "b", "c"], top_k=10, max_per_source=10)
    assert len(results) == 1


def test_duplicate_keeps_the_best_vector_score(monkeypatch, fake_search):
    scores = iter([0.4, 0.9])

    class _Store:
        def search(self, query_embedding, top_k, where=None):
            return [_chunk("usda_1", "USDA", next(scores))]

    monkeypatch.setattr(pipeline, "get_vector_store", lambda: _Store())
    results = pipeline.hybrid_search(["a", "b"], top_k=10, max_per_source=10)
    assert results[0]["vector_score"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Source diversity
# ---------------------------------------------------------------------------

def test_a_dominant_source_cannot_crowd_out_the_others(fake_search):
    """NIH_ODS has 6k+ chunks and would otherwise take every top slot.

    The cap applies first; only then does overflow backfill any slots left over,
    so a lower-scoring source still reaches the reranker.
    """
    fake_search["results"] = [_chunk(f"nih_{i}", "NIH_ODS", 0.9 - i / 100) for i in range(10)] + [
        _chunk("usda_1", "USDA", 0.1)
    ]
    results = pipeline.hybrid_search(["q"], top_k=4, max_per_source=2)
    sources = [r["metadata"]["source"] for r in results]
    assert "USDA" in sources
    assert sources.index("USDA") < 3  # promoted ahead of the capped overflow


def test_overflow_backfills_when_diverse_candidates_run_out(fake_search):
    fake_search["results"] = [_chunk(f"nih_{i}", "NIH_ODS", 0.9) for i in range(6)]
    results = pipeline.hybrid_search(["q"], top_k=5, max_per_source=2)
    assert len(results) == 5


def test_nih_boilerplate_is_dropped(fake_search):
    boilerplate = (
        "Recommended Intakes Intake recommendations for vitamin D are provided in the "
        "Dietary Reference Intakes developed by the Food and Nutrition Board."
    )
    fake_search["results"] = [
        _chunk("nih_boiler", "NIH_ODS", 0.99, text=boilerplate),
        _chunk("nih_real", "NIH_ODS", 0.5, text="The UL for adults is 100 mcg (4,000 IU)."),
    ]
    texts = [r["text"] for r in pipeline.hybrid_search(["vitamin d ul"], top_k=5, max_per_source=5)]
    assert boilerplate not in texts
    assert any("4,000 IU" in t for t in texts)


def test_boilerplate_from_other_sources_is_kept(fake_search):
    boilerplate = (
        "Recommended Intakes Intake recommendations for protein are provided in the "
        "Dietary Reference Intakes."
    )
    fake_search["results"] = [_chunk("dga_1", "Dietary_Guidelines", 0.9, text=boilerplate)]
    assert len(pipeline.hybrid_search(["q"], top_k=5, max_per_source=5)) == 1


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------

def test_rerank_of_nothing_returns_nothing():
    assert pipeline.rerank("q", []) == []


class _FakeTokenizer:
    """Stands in for the CrossEncoder tokenizer: token count == character count,
    which is enough to drive the length-sorting path."""

    def __call__(self, query, text, truncation=True, max_length=512):
        return {"input_ids": [0] * min(len(text), max_length)}


class _FakeReranker:
    """Scores a passage by looking its text up, so the expected ranking is known
    regardless of the order length-sorted batching feeds the pairs in."""

    max_seq_length = 512

    def __init__(self, scores: dict[str, float]):
        self._scores = scores
        self.batch_sizes: list[int] = []
        self.tokenizer = _FakeTokenizer()

    def predict(self, pairs, batch_size=32):
        self.batch_sizes.append(batch_size)
        return [self._scores[text] for _query, text in pairs]


def test_rerank_orders_by_score_and_truncates(monkeypatch):
    fake = _FakeReranker({"a": 0.1, "b": 0.9, "c": 0.5})
    monkeypatch.setattr(pipeline, "_reranker_instance", lambda: fake)
    candidates = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    result = pipeline.rerank("q", candidates, top_k=2)
    assert [c["text"] for c in result] == ["b", "c"]


def test_rerank_maps_scores_back_to_the_right_candidate(monkeypatch):
    """Length-sorted batching reorders pairs internally; a mapping bug here
    would silently attach each score to the wrong chunk."""
    fake = _FakeReranker({"short": 0.2, "a much longer passage": 0.95, "mid text": 0.5})
    monkeypatch.setattr(pipeline, "_reranker_instance", lambda: fake)
    candidates = [{"text": "short"}, {"text": "a much longer passage"}, {"text": "mid text"}]
    result = pipeline.rerank("q", candidates, top_k=3)
    assert [c["text"] for c in result] == ["a much longer passage", "mid text", "short"]
    assert result[0]["rerank_score"] == pytest.approx(0.95)
    assert result[-1]["rerank_score"] == pytest.approx(0.2)


def test_rerank_uses_small_batches(monkeypatch):
    """Large batches pad every pair to the longest in the batch — 3x slower."""
    fake = _FakeReranker({"a": 0.1})
    monkeypatch.setattr(pipeline, "_reranker_instance", lambda: fake)
    pipeline.rerank("q", [{"text": "a"}])
    assert fake.batch_sizes == [pipeline.get_settings().rerank_batch_size]


def test_rerank_includes_document_title_for_contextless_table(monkeypatch):
    fake = _FakeReranker({"Vitamin D - Health Professional\n>70 years | 20 mcg (800 IU)": 0.9})
    monkeypatch.setattr(pipeline, "_reranker_instance", lambda: fake)
    candidate = {"text": ">70 years | 20 mcg (800 IU)",
                 "metadata": {"source": "NIH_ODS", "nutrient_name": "Vitamin D - Health Professional"}}
    assert pipeline.rerank("Vitamin D intake for >70?", [candidate])[0]["rerank_score"] == 0.9


def test_age_table_reranking_focuses_current_age_without_changing_evidence():
    text = "Age | Male | Female\n19–50 years | 15 mcg (600 IU) | 15 mcg (600 IU)\n" \
           "51–70 years | 15 mcg (600 IU) | 15 mcg (600 IU)\n" \
           ">70 years | 20 mcg (800 IU) | 20 mcg (800 IU)"
    candidate = {"text": text, "metadata": {"nutrient_name": "Vitamin D - Health Professional"}}
    focused = pipeline._rerank_passage("Vitamin D intake for adults older than 70", candidate)
    assert ">70 years | 20 mcg" in focused
    assert "19–50 years" not in focused
    assert candidate["text"] == text
    assert pipeline._age_table_rows("Vitamin D upper limit for adults older than 70", candidate) == []


def test_recommended_intake_prefers_matching_age_table_over_ul_summary(monkeypatch):
    table = {"text": "Age | Male | Female\n>70 years | 20 mcg (800 IU) | 20 mcg (800 IU)",
             "metadata": {"nutrient_name": "Vitamin D - Health Professional"}}
    upper_limit = {"text": "Vitamin D Tolerable Upper Intake Level: 4,000 IU",
                   "metadata": {"note": "synthetic UL summary"}}
    monkeypatch.setattr(pipeline, "_reranker_instance", lambda: object())
    monkeypatch.setattr(pipeline, "_rerank_scores", lambda _model, pairs: [0.2] * len(pairs))
    result = pipeline.rerank("Recommended vitamin D intake for adults older than 70?",
                             [upper_limit, table])
    assert result == [table]
    assert table["rerank_score"] == 0.75


def test_empty_query_short_circuits(monkeypatch):
    monkeypatch.setattr(
        pipeline, "hybrid_search", lambda *a, **k: pytest.fail("should not search")
    )
    assert pipeline.retrieve("   ") == []


# ---------------------------------------------------------------------------
# Prompt selection
# ---------------------------------------------------------------------------

def test_calculate_intent_uses_the_precomputed_results():
    messages = pipeline._build_generation_messages(
        "what's my tdee?",
        [{"text": "tdee: 2400", "metadata": {"source": "Zenic Calculator"}}],
        "calculate",
        None,
    )
    assert "DO NOT recalculate" in messages[0]["content"]
    assert "tdee: 2400" in messages[1]["content"]


def test_rag_intent_cites_sources_in_the_context_block():
    messages = pipeline._build_generation_messages(
        "protein needs?",
        [{"text": "1.6 g/kg", "metadata": {"source": "ISSN", "year": "2017"}, "rerank_score": 5.0}],
        "nutrition_qa",
        None,
    )
    assert '"source": "ISSN"' in messages[1]["content"]
    assert "Clinical Data Retrieval" in messages[0]["content"]


def test_low_quality_context_keeps_strict_grounding():
    messages = pipeline._build_generation_messages(
        "hello",
        [{"text": "irrelevant", "metadata": {}, "rerank_score": -3.0}],
        "nutrition_qa",
        None,
    )
    assert "Clinical Data Retrieval" in messages[0]["content"]


def test_general_chat_includes_recent_history():
    history = [
        {"role": "user", "content": "my name is Sam"},
        {"role": "assistant", "content": "Hi Sam"},
        {"role": "user", "content": "what is my name?"},
    ]
    messages = pipeline._build_generation_messages("what is my name?", [], "general_chat", history)
    assert any("my name is Sam" in m["content"] for m in messages)
    assert messages[-1]["content"] == "what is my name?"


def test_general_chat_does_not_duplicate_the_trailing_query():
    history = [{"role": "user", "content": "hello"}]
    messages = pipeline._build_generation_messages("hello", [], "general_chat", history)
    assert sum(1 for m in messages if m["content"] == "hello") == 1


def test_rrf_does_not_add_incompatible_raw_scores(monkeypatch, fake_search):
    """A huge lexical score must not erase a high-ranking semantic result."""
    fake_search["results"] = [_chunk("semantic", "USDA", 0.9)]
    corpus = [{"id": "lexical", "text": "keyword", "metadata": {"source": "NIH_ODS"}}]
    monkeypatch.setattr(pipeline, "_bm25_corpus", corpus)
    class Index:
        def get_scores(self, tokens):
            return [1000000.0]
    monkeypatch.setattr(pipeline, "_bm25_index", Index())
    result = pipeline.hybrid_search(["keyword"], top_k=2)
    assert len(result) == 2
    assert result[0]["metadata"]["source"] == "USDA"
    assert result[0]["rrf_score"] == pytest.approx(result[1]["rrf_score"])


def test_rrf_rewards_agreement_between_retrievers(monkeypatch, fake_search):
    fake_search["results"] = [_chunk("a", "USDA", 0.9), _chunk("b", "USDA", 0.8)]
    monkeypatch.setattr(pipeline, "_bm25_corpus", [{"id": "b", "text": "text for b", "metadata": {"source": "USDA"}}])
    class Index:
        def get_scores(self, tokens):
            return [1.0]
    monkeypatch.setattr(pipeline, "_bm25_index", Index())
    result = pipeline.hybrid_search(["q"], top_k=2)
    assert result[0]["text"] == "text for b"


def test_all_sources_reach_a_pool_smaller_than_combined_caps(fake_search):
    """A 12/source cap cannot protect the fourth source in a 20-slot pool."""
    fake_search["results"] = [
        _chunk(f"{source}_{i}", source, 1 - group / 10 - i / 1000)
        for group, source in enumerate(("NIH_ODS", "ISSN", "DietaryGuidelines", "USDA"))
        for i in range(12)
    ]
    result = pipeline.hybrid_search(["foods"], top_k=20, max_per_source=12)
    assert {c["metadata"]["source"] for c in result} == {"NIH_ODS", "ISSN", "DietaryGuidelines", "USDA"}
    assert len(result) == 20
