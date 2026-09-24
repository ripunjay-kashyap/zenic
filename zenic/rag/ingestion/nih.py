"""
NIH Office of Dietary Supplements ingestion.
Source: https://ods.od.nih.gov/factsheets/list-all/

Strategy: scrape English health professional fact sheets.
One document per nutrient/supplement. Chunking preserves the RDA/UL table intact —
never split numeric values from their life-stage context (age/gender/condition).

Metadata: nutrient_name, category, source="NIH_ODS", url
"""
import hashlib
import re
import time
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from zenic.logging_config import get_logger

logger = get_logger(__name__)

_BASE = "https://ods.od.nih.gov"
_LIST_URL = f"{_BASE}/factsheets/list-all/"
_HEALTH_PROF_SUFFIX = "-HealthProfessional"  # ODS dropped the trailing slash
_REQUEST_DELAY = 0.5  # seconds between page fetches (be polite)


def _fetch_factsheet_urls() -> list[tuple[str, str]]:
    """Return [(nutrient_name, url), ...] for all health professional fact sheets."""
    resp = httpx.get(_LIST_URL, timeout=20, follow_redirects=True)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    results = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if _HEALTH_PROF_SUFFIX.lower() in href.lower():
            full_url = urljoin(_BASE, href)
            name = a.get_text(strip=True)
            if name:
                results.append((name, full_url))

    # Deduplicate by URL
    seen = set()
    unique = []
    for name, url in results:
        if url not in seen:
            seen.add(url)
            unique.append((name, url))

    return unique


def _extract_factsheet_text(url: str) -> str:
    """Fetch a fact sheet page and extract the main content text."""
    resp = httpx.get(url, timeout=20, follow_redirects=True)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Target the main content div (ODS uses id="fs-section" or class="fact-sheet-content")
    main = (
        soup.find(id="fs-section")
        or soup.find(class_="fact-sheet-content")
        or soup.find("main")
        or soup.find("article")
        or soup.body
    )

    if main is None:
        return ""

    # Remove nav, footer, script, style
    for tag in main.find_all(["nav", "footer", "script", "style", "aside"]):
        tag.decompose()

    # Preserve table structure by converting cells to readable text
    for table in main.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(separator=" ", strip=True) for td in tr.find_all(["td", "th"])]
            rows.append(" | ".join(cells))
        table.replace_with("\n".join(rows))

    text = main.get_text(separator="\n", strip=True)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_into_sections(text: str, max_chars: int = 2500) -> list[str]:
    """
    Split fact sheet text into sections that keep RDA/UL context intact.
    Splits on double newlines (section boundaries) but won't split a
    paragraph that contains a table (to preserve life-stage specificity).
    Falls back to max_chars split if a section is too long.
    """
    # First try to split on section headers (lines that look like headings)
    sections = re.split(r"\n(?=[A-Z][^\n]{2,60}\n)", text)
    chunks = []
    for section in sections:
        if len(section) <= max_chars:
            chunks.append(section.strip())
        else:
            # Hard split on double newlines
            paragraphs = re.split(r"\n{2,}", section)
            current = ""
            for para in paragraphs:
                if len(current) + len(para) + 2 <= max_chars:
                    current = (current + "\n\n" + para).strip()
                else:
                    if current:
                        chunks.append(current)
                    current = para.strip()
            if current:
                chunks.append(current)

    # Drop generic DRI boilerplate intro paragraphs — identical across all
    # nutrient fact sheets, contain no numeric values, flood retrieval.
    _boilerplate = re.compile(
        r"^(Recommended Intakes\s+Intake recommendations for [^\n]+ are provided in the "
        r"Dietary Reference Intakes|Nutrient Intake Recommendations and Upper Limits)"
    )
    return [c for c in chunks if len(c) > 50 and not _boilerplate.match(c)]


def ingest_nih_fact_sheets(limit: int | None = None) -> list[dict]:
    """
    Scrape NIH ODS health professional fact sheets and return indexable documents.
    One document per fact sheet (single chunk — fact sheets are concise enough).
    If a fact sheet is long, it's split into sections that keep tables intact.

    limit: max number of fact sheets to fetch (None = all ~100+)
    """
    logger.info("Fetching NIH ODS fact sheet list...")
    sheets = _fetch_factsheet_urls()
    if limit:
        sheets = sheets[:limit]
    logger.info(f"  Found {len(sheets)} fact sheets")

    docs = []
    for i, (name, url) in enumerate(sheets, 1):
        logger.info(f"  [{i}/{len(sheets)}] {name}")
        try:
            text = _extract_factsheet_text(url)
            if not text:
                continue

            sections = _split_into_sections(text)
            for j, section_text in enumerate(sections):
                chunk_id = hashlib.md5(f"{url}_{j}".encode()).hexdigest()[:12]
                docs.append({
                    "id": f"nih_{chunk_id}",
                    "text": section_text,
                    "metadata": {
                        "source": "NIH_ODS",
                        "nutrient_name": name,
                        "category": "supplement_facts",
                        "url": url,
                        "chunk_index": j,
                    },
                })
        except Exception as e:
            logger.info(f"    ERROR: {e}")

        time.sleep(_REQUEST_DELAY)

    logger.info(f"Prepared {len(docs)} NIH ODS documents")
    return docs
