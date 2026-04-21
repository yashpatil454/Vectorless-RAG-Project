from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from langchain_community.document_loaders import PyPDFLoader

from utils.logging_utils import get_logger

logger = get_logger(__name__)


def parse_pdf(file_path: Path) -> List[Dict[str, Any]]:
    """Parse a PDF file and return one chunk dict per page using PyPDFLoader.

    Each chunk contains:
        - ``text``: the extracted text for the page (stripped).
        - ``page_number``: 1-based page index.
        - ``source``: absolute path of the source file as a string.
        - ``char_count``: character count of the extracted text.

    Pages with no extractable text are skipped with a warning.
    """
    file_path = Path(file_path)
    logger.info("Parsing PDF: %s", file_path.name)

    try:
        loader = PyPDFLoader(str(file_path.resolve()))
        pages = loader.load()
    except Exception as exc:
        logger.error("Failed to parse '%s': %s", file_path.name, exc)
        raise

    logger.info("Total pages loaded: %d", len(pages))

    chunks: List[Dict[str, Any]] = []
    for page_doc in pages:
        text = (page_doc.page_content or "").strip()
        if not text:
            logger.warning(
                "Page %s of '%s' yielded no text — skipping.",
                page_doc.metadata.get("page", "?"),
                file_path.name,
            )
            continue
        # PyPDFLoader stores 0-based page index; convert to 1-based
        page_number = int(page_doc.metadata.get("page", 0)) + 1
        chunks.append(
            {
                "text": text,
                "page_number": page_number,
                "source": str(file_path.resolve()),
                "char_count": len(text),
            }
        )

    logger.info("Extracted %d non-empty pages from '%s'.", len(chunks), file_path.name)
    return chunks
