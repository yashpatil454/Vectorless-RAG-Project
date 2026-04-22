from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from langchain_community.document_loaders import PyPDFLoader

from utils.logging_utils import get_logger
from utils.section_detector import detect_sections

logger = get_logger(__name__)


def parse_pdf(file_path: Path) -> List[Dict[str, Any]]:
    """Parse a PDF and return section-aware chunks via detect_sections().

    Each chunk contains:
        - ``text``        : full text of the section (may span multiple pages).
        - ``section``     : L1 heading label.
        - ``subsection``  : L2 heading label (empty string if none).
        - ``level``       : 1 or 2.
        - ``page_range``  : [first_page, last_page] (1-based).
        - ``page_number`` : first_page (backwards-compat alias).
        - ``source``      : absolute path of the source file.
        - ``char_count``  : character count of the text.

    Falls back to LLM-based section detection when fewer than 2 regex headings
    are found, and ultimately to one-chunk-per-page if the LLM also fails.
    """
    file_path = Path(file_path)
    logger.info("Parsing PDF: %s", file_path.name)

    try:
        loader = PyPDFLoader(str(file_path.resolve()))
        raw_pages = loader.load()
    except Exception as exc:
        logger.error("Failed to load '%s': %s", file_path.name, exc)
        raise

    logger.info("Total pages loaded from PyPDFLoader: %d", len(raw_pages))

    # Convert LangChain Document objects into plain dicts expected by detect_sections
    source = str(file_path.resolve())
    page_dicts: List[Dict[str, Any]] = []
    for page_doc in raw_pages:
        text = (page_doc.page_content or "").strip()
        if not text:
            logger.warning(
                "Page %s of '%s' yielded no text — skipping.",
                page_doc.metadata.get("page", "?"),
                file_path.name,
            )
            continue
        page_number = int(page_doc.metadata.get("page", 0)) + 1  # 0-based → 1-based
        page_dicts.append({"text": text, "page_number": page_number, "source": source})

    chunks = detect_sections(page_dicts)
    logger.info("Produced %d section chunks from '%s'.", len(chunks), file_path.name)
    return chunks
