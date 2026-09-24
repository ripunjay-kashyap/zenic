"""
Ingestion runner — builds the local knowledge base.
Run once before starting the app or after adding new data.

Usage:
  python scripts/run_ingestion.py                        # all sources
  python scripts/run_ingestion.py --sources wger usda    # specific sources only
  python scripts/run_ingestion.py --usda-file data/usda/FoundationDownload.json
  python scripts/run_ingestion.py --usda-n 3000          # top N via API (slower)

After ingestion completes, the BM25 index is persisted to data/bm25_corpus.json
and loaded automatically when the app starts.
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from zenic.rag.ingestion.indexer import index_documents
from zenic.rag.pipeline import load_bm25_index

_BM25_CORPUS_PATH = "data/bm25_corpus.json"


def _progress(done: int, total: int) -> None:
    pct = done / total * 100
    filled = int(pct / 5)
    bar = "#" * filled + "-" * (20 - filled)
    print(f"\r  [{bar}] {done}/{total} ({pct:.0f}%)", end="", flush=True)


def run_wger() -> list[dict]:
    from zenic.rag.ingestion.wger import ingest_wger_exercises
    docs = ingest_wger_exercises()
    print(f"\nIndexing {len(docs)} wger exercise docs...")
    index_documents(docs, progress_cb=_progress)
    print()
    return docs


def run_usda(bulk_file: str | None = None, n: int = 3000) -> list[dict]:
    if bulk_file:
        from zenic.rag.ingestion.usda import ingest_usda_bulk
        docs = ingest_usda_bulk(bulk_file)
    else:
        from zenic.rag.ingestion.usda import ingest_usda_api
        docs = ingest_usda_api(n=n)
    print(f"\nIndexing {len(docs)} USDA food docs...")
    index_documents(docs, progress_cb=_progress)
    print()
    return docs


def run_nih(limit: int | None = None) -> list[dict]:
    from zenic.rag.ingestion.nih import ingest_nih_fact_sheets
    docs = ingest_nih_fact_sheets(limit=limit)
    print(f"\nIndexing {len(docs)} NIH ODS docs...")
    index_documents(docs, progress_cb=_progress)
    print()
    return docs


def run_dietary(pdf_dir: str = "data/dietary_guidelines") -> list[dict]:
    if not list(Path(pdf_dir).glob("*.pdf")):
        print(f"No PDFs in {pdf_dir} — skipping dietary guidelines")
        return []
    from zenic.rag.ingestion.dietary_guidelines import ingest_dietary_guidelines
    docs = ingest_dietary_guidelines(pdf_dir)
    print(f"\nIndexing {len(docs)} dietary guidelines docs...")
    index_documents(docs, progress_cb=_progress)
    print()
    return docs


def run_issn(papers_dir: str = "data/issn") -> list[dict]:
    if not list(Path(papers_dir).glob("*.pdf")):
        print(f"No PDFs in {papers_dir} — skipping ISSN papers")
        return []
    from zenic.rag.ingestion.issn import ingest_issn_papers
    docs = ingest_issn_papers(papers_dir)
    print(f"\nIndexing {len(docs)} ISSN paper docs...")
    index_documents(docs, progress_cb=_progress)
    print()
    return docs


def save_bm25_corpus(corpus: list[dict], merge: bool = True) -> None:
    """
    Persist the BM25 corpus to disk.
    merge=True (default): load existing corpus and merge by id so partial
    ingestion runs (e.g. --sources nih wger) don't wipe previously indexed
    sources.  merge=False overwrites the file entirely.
    """
    Path(_BM25_CORPUS_PATH).parent.mkdir(parents=True, exist_ok=True)
    slim = [{"id": d["id"], "text": d["text"], "metadata": d["metadata"]} for d in corpus]

    if merge and Path(_BM25_CORPUS_PATH).exists():
        with open(_BM25_CORPUS_PATH, encoding="utf-8") as f:
            existing = json.load(f)
        existing_by_id = {d["id"]: d for d in existing}
        for doc in slim:
            existing_by_id[doc["id"]] = doc  # new doc wins on collision
        slim = list(existing_by_id.values())

    target = Path(_BM25_CORPUS_PATH)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, delete=False) as f:
            temporary = Path(f.name)
            json.dump(slim, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"BM25 corpus saved: {_BM25_CORPUS_PATH} ({len(slim)} total docs)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Zenic knowledge base")
    parser.add_argument("--sources", nargs="+",
                        choices=["wger", "usda", "nih", "dietary", "issn"],
                        help="Sources to ingest (default: all)")
    parser.add_argument("--usda-file", help="Path to downloaded USDA bulk JSON file")
    parser.add_argument("--usda-n", type=int, default=3000,
                        help="Number of foods to fetch via USDA API (default: 3000)")
    parser.add_argument("--nih-limit", type=int, default=None,
                        help="Max NIH fact sheets to scrape (default: all)")
    parser.add_argument("--dietary-dir", default="data/dietary_guidelines",
                        help="Directory containing dietary guidelines PDFs")
    parser.add_argument("--issn-dir", default="data/issn",
                        help="Directory containing ISSN paper PDFs + .json metadata")
    args = parser.parse_args()

    sources = set(args.sources) if args.sources else {"wger", "usda", "nih", "dietary", "issn"}
    all_docs = []

    print("=" * 60)
    print("Zenic Knowledge Base Ingestion")
    print("=" * 60)

    if "wger" in sources:
        all_docs.extend(run_wger())

    if "usda" in sources:
        all_docs.extend(run_usda(bulk_file=args.usda_file, n=args.usda_n))

    if "nih" in sources:
        all_docs.extend(run_nih(limit=args.nih_limit))

    if "dietary" in sources:
        all_docs.extend(run_dietary(pdf_dir=args.dietary_dir))

    if "issn" in sources:
        all_docs.extend(run_issn(papers_dir=args.issn_dir))

    if all_docs:
        print(f"\nTotal documents indexed: {len(all_docs)}")
        save_bm25_corpus(all_docs)
        load_bm25_index(all_docs)
        print("\nIngestion complete. Knowledge base is ready.")
    else:
        print("\nNo documents ingested.")


if __name__ == "__main__":
    main()
