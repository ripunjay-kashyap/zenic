"""
ISSN Position Papers ingestion (PubMed Central open-access).

The key design decision: SECTION-AWARE chunking.
Each chunk starts with a metadata prefix injected into the text itself:
  "Source: {title} ({authors}, {year}), Section: {section_name}"

This prefix is what enables the LLM to cite the source accurately in its response —
the citation appears inside the retrieved chunk, not just as a metadata field.

Chunk size: 800-1000 tokens (~4000-5000 chars) with 100-token (~500 char) overlap.
Metadata: source_title, authors, year, section, topic, source="ISSN"

To add an ISSN paper, download it as a PDF from PubMed Central and call:
  ingest_issn_paper(pdf_path, title="...", authors="...", year=2017)
"""
import hashlib
import re
from pathlib import Path

from zenic.logging_config import get_logger

logger = get_logger(__name__)

# Common ISSN position stand section headers (used to detect section boundaries)
_SECTION_PATTERNS = [
    r"^(?:Abstract|Introduction|Background|Methods?|Results?|Discussion|Conclusions?|"
    r"Recommendations?|Summary|References?|Acknowledgements?|"
    r"Position\s+Statement|Literature\s+Review|"
    r"\d+\.\s+[A-Z][^\n]{3,60})$",
]
_SECTION_RE = re.compile("|".join(_SECTION_PATTERNS), re.MULTILINE | re.IGNORECASE)


def _extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("Install pypdf: pip install pypdf") from exc

    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    full = "\n\n".join(pages)
    # Clean PDF extraction artefacts
    full = re.sub(r"-\n", "", full)
    full = re.sub(r"\n{3,}", "\n\n", full)
    full = re.sub(r"[ \t]{2,}", " ", full)
    return full


def _detect_sections(text: str) -> list[tuple[str, int]]:
    """
    Return a list of (section_name, start_char_index) tuples.
    Falls back to a single "Full Text" section if no headers detected.
    """
    matches = [(m.group(0).strip(), m.start()) for m in _SECTION_RE.finditer(text)]
    if not matches:
        return [("Full Text", 0)]
    # Ensure we start from the beginning
    if matches[0][1] > 0:
        matches = [("Introduction", 0), *matches]
    return matches


def _split_section(
    section_text: str,
    chunk_size: int = 4500,
    overlap: int = 500,
) -> list[str]:
    """Split a section into overlapping chunks if it exceeds chunk_size."""
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be nonnegative and smaller than chunk_size")
    if len(section_text) <= chunk_size:
        return [section_text.strip()] if section_text.strip() else []
    chunks = []
    start = 0
    while start < len(section_text):
        end = start + chunk_size
        # Try to break at a sentence boundary
        if end < len(section_text):
            boundary = section_text.rfind(". ", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        chunk = section_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(section_text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def ingest_issn_paper(
    pdf_path: str,
    title: str,
    authors: str,
    year: int,
    topic: str = "",
) -> list[dict]:
    """
    Ingest a single ISSN position stand PDF.

    Args:
        pdf_path:  path to the downloaded PDF
        title:     e.g. "Position Stand on Protein and Exercise"
        authors:   e.g. "Jäger et al."
        year:      e.g. 2017
        topic:     optional topic tag, e.g. "protein", "creatine", "caffeine"
    """
    logger.info(f"Ingesting ISSN paper: {title} ({authors}, {year})")
    text = _extract_pdf_text(pdf_path)
    sections = _detect_sections(text)
    logger.info(f"  Detected {len(sections)} sections")

    citation_prefix = f"Source: ISSN {title} ({authors}, {year})"

    docs = []
    for i, (section_name, start) in enumerate(sections):
        end = sections[i + 1][1] if i + 1 < len(sections) else len(text)
        section_text = text[start:end].strip()

        if len(section_text) < 100:
            continue

        sub_chunks = _split_section(section_text)
        for j, chunk_text in enumerate(sub_chunks):
            # Inject citation prefix INTO the chunk text — this is what the LLM cites
            prefixed = f"{citation_prefix}, Section: {section_name}\n\n{chunk_text}"
            chunk_id = hashlib.md5(f"{title}_{i}_{section_name}_{j}".encode()).hexdigest()[:12]

            docs.append({
                "id": f"issn_{chunk_id}",
                "text": prefixed,
                "metadata": {
                    "source": "ISSN",
                    "source_title": f"ISSN {title}",
                    "authors": authors,
                    "year": str(year),
                    "section": section_name,
                    "topic": topic,
                    "chunk_index": j,
                },
            })

    logger.info(f"  Produced {len(docs)} chunks")
    return docs


def ingest_issn_papers(papers_dir: str) -> list[dict]:
    """
    Ingest all ISSN papers from a directory.
    Each PDF must have a companion metadata .json file with the same stem:
      {
        "title": "Position Stand on Protein and Exercise",
        "authors": "Jäger et al.",
        "year": 2017,
        "topic": "protein"
      }

    Place downloaded PDFs in: data/issn/
    """
    import json

    dir_path = Path(papers_dir)
    all_docs = []

    for pdf in sorted(dir_path.glob("*.pdf")):
        meta_path = pdf.with_suffix(".json")
        if not meta_path.exists():
            logger.info(f"  Skipping {pdf.name} — no companion .json metadata file")
            continue
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        docs = ingest_issn_paper(
            str(pdf),
            title=meta["title"],
            authors=meta["authors"],
            year=int(meta["year"]),
            topic=meta.get("topic", ""),
        )
        all_docs.extend(docs)

    logger.info(f"Total ISSN documents: {len(all_docs)}")
    return all_docs
