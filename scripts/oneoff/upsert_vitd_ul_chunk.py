"""
Add a dedicated Vitamin D UL chunk to both BM25 corpus and ChromaDB.

The cross-encoder bge-reranker-base consistently misranks vitamin D RDA
sections above UL sections for "tolerable upper intake level" queries.
This chunk leads with the answer so the reranker cannot miss it.

Run once:
    PYTHONPATH=. python scripts/upsert_vitd_ul_chunk.py
"""
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CORPUS_PATH = Path("data/bm25_corpus.json")

VITD_UL_TEXT = (
    "Vitamin D Tolerable Upper Intake Level (UL) — NIH Office of Dietary Supplements\n\n"
    "The Tolerable Upper Intake Level (UL) for vitamin D in adults aged 19 years and older "
    "is 100 mcg (4,000 IU) per day. This applies to males, females, pregnant women, and "
    "lactating women.\n\n"
    "UL for vitamin D by age group (Dietary Reference Intakes, Food and Nutrition Board, 2010):\n"
    "- Infants 0–6 months: 25 mcg (1,000 IU)\n"
    "- Infants 7–12 months: 38 mcg (1,500 IU)\n"
    "- Children 1–3 years: 63 mcg (2,500 IU)\n"
    "- Children 4–8 years: 75 mcg (3,000 IU)\n"
    "- Children/adolescents 9–18 years: 100 mcg (4,000 IU)\n"
    "- Adults 19–50 years: 100 mcg (4,000 IU)\n"
    "- Adults 51–70 years: 100 mcg (4,000 IU)\n"
    "- Adults over 70 years: 100 mcg (4,000 IU)\n"
    "- Pregnancy (all ages 14–50): 100 mcg (4,000 IU)\n"
    "- Lactation (all ages 14–50): 100 mcg (4,000 IU)\n\n"
    "Vitamin D toxicity (from exceeding the UL) can cause hypercalcemia, hypercalciuria, "
    "and in extreme cases renal failure. Toxicity is almost always from excessive supplement "
    "use, not food. The UL was established because even intakes below 250 mcg (10,000 IU) may "
    "have adverse effects over time.\n"
    "(Source: NIH Office of Dietary Supplements, Vitamin D Health Professional Fact Sheet, Table 4)"
)

chunk_id = hashlib.md5(b"NIH_ODS_vitaminD_UL_table4").hexdigest()[:12]
doc_id = f"nih_{chunk_id}"

# --- 1. Add to BM25 corpus ---
corpus = json.load(open(CORPUS_PATH, encoding="utf-8"))
if not any(c.get("id") == doc_id for c in corpus):
    corpus.append({
        "id": doc_id,
        "text": VITD_UL_TEXT,
        "metadata": {
            "source": "NIH_ODS",
            "nutrient_name": "Vitamin D",
            "category": "supplement_facts",
            "chunk_index": 9999,
            "note": "synthetic UL summary — bge-reranker-base misranks original UL chunks; re-ingest with improved chunking to replace",
        },
    })
    json.dump(corpus, open(CORPUS_PATH, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"Added vitamin D UL chunk to BM25 corpus ({len(corpus)} total)")
else:
    print("Vitamin D UL chunk already in BM25 corpus — skipped")

# --- 2. Upsert into ChromaDB ---
from sentence_transformers import SentenceTransformer

from zenic.rag.vector_store import get_vector_store

print("Loading embedding model...")
model = SentenceTransformer(os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
embedding = model.encode(VITD_UL_TEXT).tolist()

store = get_vector_store()
store.upsert([{
    "id": doc_id,
    "text": VITD_UL_TEXT,
    "embedding": embedding,
    "metadata": {
        "source": "NIH_ODS",
        "nutrient_name": "Vitamin D",
        "category": "supplement_facts",
        "chunk_index": 9999,
    },
}])
print(f"Upserted vitamin D UL chunk into ChromaDB (id={doc_id})")
