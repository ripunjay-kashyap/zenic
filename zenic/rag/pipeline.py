"""
Pillar 1 retrieval pipeline.

Flow:
  query → multi-query expansion → hybrid search (vector + BM25) → cross-encoder rerank → top chunks
"""
import os
import re
from typing import Any

from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from zenic.rag.vector_store import get_vector_store

_embed_model: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None
_bm25_index: BM25Okapi | None = None
_bm25_corpus: list[dict] | None = None
_groq_client: Groq | None = None

_BM25_CORPUS_PATH = "data/bm25_corpus.json"


def _try_load_bm25_from_disk() -> None:
    """Auto-load BM25 index from the persisted corpus file if it exists."""
    global _bm25_index, _bm25_corpus
    import json
    from pathlib import Path
    if _bm25_index is not None:
        return
    path = Path(_BM25_CORPUS_PATH)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            corpus = json.load(f)
        load_bm25_index(corpus)
        print(f"[pipeline] BM25 index loaded from {path} ({len(corpus)} docs)")


def _embed_model_instance() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        model_name = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
        _embed_model = SentenceTransformer(model_name)
    return _embed_model


def _reranker_instance() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        model_name = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
        _reranker = CrossEncoder(model_name)
    return _reranker


def _groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _groq_client


# ---------------------------------------------------------------------------
# Public pipeline functions
# ---------------------------------------------------------------------------

def generate_multi_queries(query: str, n: int = 3) -> list[str]:
    """Generate alternative phrasings of the query to improve recall."""
    prompt = (
        f"Generate {n} alternative phrasings of the following health or nutrition question. "
        "Each phrasing should approach the same information need from a different angle. "
        "If the question asks about research, studies, or expert recommendations, include at least one "
        "phrasing that names a specific authoritative source (e.g., ISSN, NIH, WHO, USDA, Dietary Guidelines). "
        "Return only the list, one per line, no numbering.\n\n"
        f"Original question: {query}"
    )
    response = _groq().chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    variants = [line.strip() for line in response.choices[0].message.content.strip().splitlines() if line.strip()]
    return [query] + variants[:n]


def hybrid_search(queries: list[str], top_k: int = 20, max_per_source: int = 7) -> list[dict]:
    """
    Run vector + BM25 search for all query variants and merge candidates.
    Returns up to top_k unique candidates with combined scores.

    max_per_source caps how many candidates from any single source are allowed
    in the output, preventing a numerically dominant source (e.g. NIH_ODS with
    6k+ chunks) from crowding out other sources before the reranker can evaluate
    cross-source relevance.  Overflow candidates fill remaining slots.
    """
    _try_load_bm25_from_disk()
    store = get_vector_store()
    embedder = _embed_model_instance()

    seen_ids: dict[str, dict] = {}

    for q in queries:
        embedding = embedder.encode(q).tolist()
        vector_results = store.search(query_embedding=embedding, top_k=top_k)
        for r in vector_results:
            key = r["text"][:80]
            if key not in seen_ids:
                r["bm25_score"] = 0.0
                seen_ids[key] = r
            else:
                seen_ids[key]["vector_score"] = max(seen_ids[key].get("vector_score", 0), r.get("vector_score", 0))

    # BM25 pass — two roles:
    #   1. Score existing vector candidates (boost if BM25 also likes them)
    #   2. Inject top BM25 candidates that vector search missed entirely
    #      (true hybrid: prevents terse structured docs like USDA nutrient tables
    #       from being excluded when they lack the conceptual vocabulary the query uses)
    if _bm25_index is not None and _bm25_corpus is not None:
        for q in queries:
            tokens = q.lower().split()
            scores = _bm25_index.get_scores(tokens)
            # Collect top-k by BM25 score for injection
            top_bm25 = sorted(
                ((idx, s) for idx, s in enumerate(scores) if s > 0 and idx < len(_bm25_corpus)),
                key=lambda x: x[1],
                reverse=True,
            )[:top_k]
            for idx, score in top_bm25:
                chunk = _bm25_corpus[idx]
                key = chunk["text"][:80]
                if key in seen_ids:
                    # Already a vector candidate — just update BM25 score
                    seen_ids[key]["bm25_score"] = max(seen_ids[key].get("bm25_score", 0), score)
                else:
                    # New candidate from BM25 — vector score unknown, use 0
                    seen_ids[key] = {
                        "text": chunk["text"],
                        "metadata": chunk["metadata"],
                        "vector_score": 0.0,
                        "bm25_score": score,
                    }

    candidates = list(seen_ids.values())
    candidates.sort(key=lambda c: c.get("vector_score", 0) + c.get("bm25_score", 0), reverse=True)

    # Per-source diversity cap — prevents one large source from filling all slots
    # before the cross-encoder reranker can make cross-source comparisons.
    source_counts: dict[str, int] = {}
    diverse: list[dict] = []
    overflow: list[dict] = []
    for c in candidates:
        src = c.get("metadata", {}).get("source", "unknown")
        if source_counts.get(src, 0) < max_per_source:
            diverse.append(c)
            source_counts[src] = source_counts.get(src, 0) + 1
        else:
            overflow.append(c)

    result = diverse[:top_k]
    if len(result) < top_k:
        result.extend(overflow[: top_k - len(result)])

    # Strip generic NIH ODS boilerplate intro paragraphs — every nutrient fact
    # sheet starts with the same "Recommended Intakes / DRI framework" paragraph
    # that matches nutrient-specific queries (e.g. "vitamin D upper intake level")
    # without containing any actual values, flooding the top slots.
    _BOILERPLATE = re.compile(
        r"^(Recommended Intakes\s+Intake recommendations for [^\n]+ are provided in the "
        r"Dietary Reference Intakes|Nutrient Intake Recommendations and Upper Limits)"
    )
    result = [
        c for c in result
        if not (
            c.get("metadata", {}).get("source") == "NIH_ODS"
            and _BOILERPLATE.match(c.get("text", ""))
        )
    ]
    return result


def rerank(query: str, candidates: list[dict], top_k: int = 7) -> list[dict]:
    """Cross-encoder reranking — precision pass after recall-optimised retrieval."""
    reranker = _reranker_instance()
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)
    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:top_k]


def retrieve(query: str) -> list[dict]:
    """Full retrieval pipeline: multi-query → hybrid search → rerank."""
    queries = generate_multi_queries(query)
    # Larger pool (top_k=30, max_per_source=12) ensures the reranker sees enough
    # candidates from each source — especially NIH_ODS which has 6k+ chunks and
    # needs more slots to surface nutrient-specific content past generic neighbors.
    candidates = hybrid_search(queries, top_k=30, max_per_source=12)
    return rerank(query, candidates, top_k=7)


def _all_low_quality(chunks: list[dict], threshold: float = 0.3) -> bool:
    """True when every retrieved chunk scored below the quality threshold."""
    if not chunks:
        return True
    return all(c.get("rerank_score", 0.0) < threshold for c in chunks)


def generate(
    query: str,
    context_chunks: list[dict],
    intent: str = "",
    history: list[dict] | None = None,
) -> str:
    """Generate a grounded answer with source citations.

    history is an optional list of prior turns ({"role": "user"|"assistant", "content": str}),
    used only by the general_chat branch so the model can reference earlier context.
    """

    # --- Calculate intent: present pre-computed deterministic results ---
    if intent == "calculate":
        calc_chunk = next(
            (c for c in context_chunks if c.get("metadata", {}).get("source") == "Zenic Calculator"),
            None,
        )
        if calc_chunk:
            system_prompt = (
                "You are Zenic, a health and nutrition assistant. "
                "You have already computed the user's metrics using the Harris-Benedict equation. "
                "Present the pre-computed results below clearly and concisely. "
                "DO NOT recalculate. DO NOT use different numbers. "
                "Format the results with clear labels (BMR, TDEE, macros). "
                "Add a brief 1-sentence practical tip at the end."
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Pre-computed results:\n{calc_chunk['text']}\n\nUser query: {query}"},
            ]
            response = _groq().chat.completions.create(
                model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                messages=messages,
            )
            return response.choices[0].message.content

    # --- General chat: conversational LLM ---
    if intent == "general_chat" or not context_chunks or _all_low_quality(context_chunks):
        system_prompt = (
            "You are Zenic, a friendly AI health and nutrition assistant. "
            "Answer the user's question conversationally and helpfully. "
            "You can discuss nutrition, fitness, meal planning, and general health topics. "
            "Never provide medical diagnoses and never recommend supplement dosages above established Upper Intake Levels. "
            "Keep your answers concise and practical."
        )
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history[-6:])
        if not messages or messages[-1].get("content") != query:
            messages.append({"role": "user", "content": query})
    else:
        # --- RAG intents: strict Clinical Data Retrieval ---
        context_text = "\n\n".join(
            f"[Source: {c['metadata'].get('source', 'Unknown')}, "
            f"{c['metadata'].get('year', '')}]\n{c['text']}"
            for c in context_chunks
        )
        system_prompt = (
            "You are a Clinical Data Retrieval Assistant. Your ONLY goal is to answer the user's query "
            "based strictly on the provided context chunks from USDA, NIH, and wger. "
            "STRICT RULE: Do not provide advice, tips, or facts not present in the context. "
            "If the context says 'Perform 3 sets' and you know '5 sets' is better, you MUST say '3 sets.' "
            "If the information is missing, state 'The provided documentation does not contain this information.' "
            "No conversational filler. No creative extrapolation. "
            "When referencing information, always cite the source name and year inline. "
            "Never provide medical diagnoses and never recommend supplement dosages above established Upper Intake Levels."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"},
        ]
    response = _groq().chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=messages,
    )
    return response.choices[0].message.content


def load_bm25_index(corpus: list[dict]) -> None:
    """Call during ingestion to build the in-memory BM25 index over the full corpus."""
    global _bm25_index, _bm25_corpus
    tokenized = [doc["text"].lower().split() for doc in corpus]
    _bm25_index = BM25Okapi(tokenized)
    _bm25_corpus = corpus
