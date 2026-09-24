"""
Upsert the synthetic Mediterranean-Style Dietary Pattern chunk into ChromaDB
so it participates in vector search (not just BM25).

Run once after patch_corpus.py:
    PYTHONPATH=. python scripts/upsert_mediterranean_chunk.py
"""
import hashlib
import os

from dotenv import load_dotenv

load_dotenv()

from sentence_transformers import SentenceTransformer

from zenic.rag.vector_store import get_vector_store

MED_TEXT = (
    "Healthy Mediterranean-Style Dietary Pattern "
    "(Dietary Guidelines for Americans, 2020-2025, Appendix 3, Table A3-5)\n\n"
    "The Healthy Mediterranean-Style Dietary Pattern is adapted from traditional diets "
    "of countries bordering the Mediterranean Sea. It is one of three USDA Dietary Patterns "
    "in the Dietary Guidelines for Americans 2020-2025.\n\n"
    "Compared to the Healthy U.S.-Style Dietary Pattern, the Mediterranean-Style pattern "
    "includes:\n"
    "- More fruits (2.5 cup equivalents/day vs 2 cup eq/day)\n"
    "- More seafood (15 oz equivalents/week vs 8 oz eq/week) — fish such as salmon, sardines, "
    "trout, herring, and other seafood are emphasized\n"
    "- Less dairy (2 cup equivalents/day vs 3 cup eq/day)\n"
    "- Similar vegetables (2.5 cup equivalents/day), emphasizing dark-green, red and orange, "
    "beans, peas, and lentils\n"
    "- Oils including olive oil (27 grams/day)\n"
    "- Whole grains (at least half of 6 oz eq/day)\n"
    "- Nuts, seeds, and legumes\n\n"
    "The pattern limits added sugars, saturated fat, and sodium, consistent with all USDA "
    "Dietary Patterns. It does not prescribe specific foods but provides a flexible framework "
    "for customization."
)

chunk_id = hashlib.md5(b"DGA_mediterranean_style_A3-5").hexdigest()[:12]

print("Loading embedding model...")
model = SentenceTransformer(os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
embedding = model.encode(MED_TEXT).tolist()

store = get_vector_store()
store.upsert([{
    "id": f"dietary_{chunk_id}",
    "text": MED_TEXT,
    "embedding": embedding,
    "metadata": {
        "source": "DietaryGuidelines",
        "source_name": "Dietary Guidelines for Americans 2020-2025",
        "category": "dietary_guidelines",
        "chunk_index": 9999,
    },
}])

print(f"Upserted Mediterranean-Style chunk (id=dietary_{chunk_id}) into ChromaDB.")
