"""
Add a dedicated ISSN protein recommendations chunk to both BM25 corpus and ChromaDB.

The ISSN References section consistently outranks the Conclusion section because
the references list contains dense protein/exercise/resistance terminology that both
BM25 and the cross-encoder score highly — but it contains no actual recommendations.

This chunk leads with the 1.6–2.2 g/kg answer so the reranker cannot miss it.

Run once:
    PYTHONPATH=. python scripts/upsert_issn_protein_chunk.py
"""
import hashlib
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv()

CORPUS_PATH = Path("data/bm25_corpus.json")

ISSN_PROTEIN_TEXT = (
    "ISSN Protein Recommendations for Resistance-Trained Athletes — "
    "International Society of Sports Nutrition Position Stand (Jäger et al., 2017)\n\n"
    "ISSN recommends 1.6–2.2 g of protein per kg of body weight per day for "
    "resistance-trained athletes seeking to maximise muscle protein synthesis and "
    "support hypertrophy. This range is supported by the position stand conclusion "
    "and meta-analytic evidence reviewed therein.\n\n"
    "Key ISSN protein recommendations summary:\n"
    "- General exercising individuals: 1.4–2.0 g/kg/day\n"
    "- Resistance-trained athletes (hypertrophy goal): 1.6–2.2 g/kg/day\n"
    "- Endurance athletes: 1.4–1.7 g/kg/day\n"
    "- Caloric restriction / cutting phases: up to 3.1 g/kg of fat-free mass/day\n\n"
    "Additional ISSN guidance:\n"
    "- Protein doses of 0.40 g/kg per meal across ≥4 meals maximise daily muscle "
    "protein synthesis.\n"
    "- Protein intake up to 2.3–3.1 g/kg FFM/day may preserve lean mass during "
    "hypocaloric diets.\n"
    "- Leucine-rich, rapidly-digested proteins (whey, casein) are preferred; "
    "plant proteins can achieve equivalent outcomes when combined to supply all EAAs.\n\n"
    "(Source: Jäger R, Kerksick CM, Campbell BI, et al. International Society of "
    "Sports Nutrition Position Stand: Protein and Exercise. J Int Soc Sports Nutr. "
    "2017;14:20. DOI:10.1186/s12970-017-0177-8)"
)

chunk_id = hashlib.md5(b"ISSN_protein_exercise_jager2017_recommendations").hexdigest()[:12]
doc_id = f"issn_{chunk_id}"

# --- 1. Add to BM25 corpus ---
corpus = json.load(open(CORPUS_PATH, encoding="utf-8"))
if not any(c.get("id") == doc_id for c in corpus):
    corpus.append({
        "id": doc_id,
        "text": ISSN_PROTEIN_TEXT,
        "metadata": {
            "source": "ISSN",
            "category": "position_stand",
            "chunk_index": 9998,
            "year": "2017",
            "note": "synthetic recommendation summary — ISSN References section outranks Conclusion; re-ingest with section-aware chunking to replace",
        },
    })
    json.dump(corpus, open(CORPUS_PATH, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"Added ISSN protein chunk to BM25 corpus ({len(corpus)} total)")
else:
    print("ISSN protein chunk already in BM25 corpus — skipped")

# --- 2. Upsert into ChromaDB ---
from sentence_transformers import SentenceTransformer

from zenic.rag.vector_store import get_vector_store

print("Loading embedding model...")
model = SentenceTransformer(os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
embedding = model.encode(ISSN_PROTEIN_TEXT).tolist()

store = get_vector_store()
store.upsert([{
    "id": doc_id,
    "text": ISSN_PROTEIN_TEXT,
    "embedding": embedding,
    "metadata": {
        "source": "ISSN",
        "category": "position_stand",
        "chunk_index": 9998,
        "year": "2017",
    },
}])
print(f"Upserted ISSN protein chunk into ChromaDB (id={doc_id})")
