"""Eyeball random chunks from each source. Run after any chunking/ingestion change."""
from zenic.rag.vector_store import get_vector_store


def inspect_chunks(source_filter: str, n: int = 5) -> None:
    store = get_vector_store()
    chunks = store.sample_chunks(where={"source": source_filter}, n=n)
    print(f"\n{'=' * 70}")
    print(f"SOURCE: {source_filter}  ({len(chunks)} chunks shown)")
    print(f"{'=' * 70}")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} ---")
        print(f"Metadata : {chunk['metadata']}")
        print(f"Length   : {len(chunk['text'])} chars, ~{len(chunk['text'].split())} words")
        print(f"Preview  : {chunk['text'][:300]}...")


if __name__ == "__main__":
    for source in ["USDA", "NIH_ODS", "ISSN", "wger", "DietaryGuidelines"]:
        inspect_chunks(source, n=5)
