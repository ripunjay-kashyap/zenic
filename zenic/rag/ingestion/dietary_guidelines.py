"""
Dietary Guidelines ingestion.
Sources:
  - DietaryGuidelines.gov PDFs (public domain)
  - WHO Dietary Guidelines (public domain)

Chunking: recursive text splitting, 500-800 tokens (~2500-4000 chars) with
100-token (~500 char) overlap. Each chunk tagged with source, section topic.

Metadata: category, topic, source_name, source="DietaryGuidelines", chunk_index
"""
import hashlib
import re
from pathlib import Path

from zenic.logging_config import get_logger

logger = get_logger(__name__)


def _extract_pdf_text(pdf_path: str) -> str:
    """Extract full text from a PDF using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("Install pypdf: pip install pypdf") from exc

    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n\n".join(pages)


def _recursive_split(text: str, chunk_size: int = 3000, overlap: int = 500) -> list[str]:
    """
    Recursive text splitter. Tries to split on paragraph breaks first,
    then sentence breaks, then character boundaries.
    """
    from zenic.rag.ingestion.issn import _split_section

    # Sentence-aware windows always advance, cap size, and preserve overlap.
    return _split_section(text, chunk_size=chunk_size, overlap=overlap)


def ingest_pdf(
    pdf_path: str,
    source_name: str,
    category: str = "dietary_guidelines",
    chunk_size: int = 3000,
    overlap: int = 800,
) -> list[dict]:
    """
    Ingest a single dietary guidelines PDF.

    Args:
        pdf_path: path to the PDF file
        source_name: display name (e.g. "Dietary Guidelines for Americans 2020-2025")
        category: metadata category tag
        chunk_size: target chars per chunk (~600 tokens at ~5 chars/token)
        overlap: overlap chars between adjacent chunks
    """
    logger.info(f"Extracting text from: {Path(pdf_path).name}")
    text = _extract_pdf_text(pdf_path)

    # Clean up common PDF extraction artefacts
    text = re.sub(r"-\n", "", text)          # hyphenated line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)   # excess blank lines
    text = re.sub(r"[ \t]{2,}", " ", text)   # excess whitespace

    logger.info(f"  Extracted {len(text)} chars — splitting...")
    raw_chunks = _recursive_split(text, chunk_size=chunk_size, overlap=overlap)
    logger.info(f"  {len(raw_chunks)} chunks")

    docs = []
    for i, chunk_text in enumerate(raw_chunks):
        chunk_id = hashlib.md5(f"{source_name}_{i}".encode()).hexdigest()[:12]
        docs.append({
            "id": f"dietary_{chunk_id}",
            "text": chunk_text,
            "metadata": {
                "source": "DietaryGuidelines",
                "source_name": source_name,
                "category": category,
                "chunk_index": i,
            },
        })

    return docs


def ingest_dietary_guidelines(pdf_dir: str) -> list[dict]:
    """
    Ingest all PDFs in a directory as dietary guidelines.
    Each PDF should be a DietaryGuidelines.gov or WHO guidelines document.
    The filename (without extension) is used as the source_name.

    Place downloaded PDFs in: data/dietary_guidelines/
    """
    pdf_dir_path = Path(pdf_dir)
    pdfs = list(pdf_dir_path.glob("*.pdf"))
    if not pdfs:
        logger.info(f"No PDFs found in {pdf_dir}")
        return []

    all_docs = []
    for pdf in pdfs:
        source_name = pdf.stem.replace("_", " ").replace("-", " ")
        docs = ingest_pdf(str(pdf), source_name=source_name)
        all_docs.extend(docs)

    logger.info(f"Total dietary guidelines documents: {len(all_docs)}")
    return all_docs
