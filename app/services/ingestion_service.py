from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import settings
from utils.logging_utils import get_logger
from utils.metadata_extractor import attach_metadata
from utils.pdf_parser import parse_pdf

logger = get_logger(__name__)


def ingest_document(
    file_bytes: bytes,
    filename: str,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Persist a PDF upload to disk, parse it, and return ingestion metadata.

    Steps:
        1. Write the raw file to ``UPLOAD_DIR / filename``.
        2. Parse the PDF into per-page chunks via ``parse_pdf()``.
        3. Enrich each chunk with document metadata via ``attach_metadata()``.
        4. Persist enriched chunks as JSON to ``PARSED_DIR / <stem>.json``.

    Returns a dict containing:
        - ``filename``: original filename.
        - ``upload_path``: absolute path where the raw file was saved.
        - ``parsed_path``: absolute path where the parsed JSON was saved.
        - ``chunk_count``: number of non-empty page chunks extracted.
        - ``upload_timestamp``: ISO-8601 UTC timestamp.
    """
    settings.ensure_dirs()

    upload_path = settings.upload_dir / filename
    logger.info("Saving uploaded file to: %s", upload_path)
    upload_path.write_bytes(file_bytes)

    # Parse PDF → raw chunks
    try:
        raw_chunks = parse_pdf(upload_path)
    except Exception as exc:
        logger.error("PDF parse error for '%s': %s", filename, exc)
        raise ValueError(f"Could not parse PDF '{filename}': {exc}") from exc

    if not raw_chunks:
        raise ValueError(f"No text could be extracted from '{filename}'.")

    # Attach metadata
    enriched = attach_metadata(raw_chunks, filename=filename, tags=tags)

    # Store section metadata only — summaries are generated at index build time
    for chunk in enriched:
        chunk.setdefault("metadata", {})
        chunk["metadata"]["section"] = chunk.get("section", "")
        chunk["metadata"]["subsection"] = chunk.get("subsection", "")
        chunk["metadata"]["page_range"] = chunk.get("page_range", [chunk.get("page_number", 0)])

    # Persist parsed chunks
    stem = Path(filename).stem
    parsed_path = settings.parsed_dir / f"{stem}.json"
    parsed_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved %d chunks to: %s", len(enriched), parsed_path)

    return {
        "filename": filename,
        "upload_path": str(upload_path.resolve()),
        "parsed_path": str(parsed_path.resolve()),
        "chunk_count": len(enriched),
        "upload_timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }


def load_parsed_chunks(filename: str) -> List[Dict[str, Any]]:
    """Load previously parsed chunks for a given original filename.

    Raises ``FileNotFoundError`` if the parsed JSON does not exist.
    """
    stem = Path(filename).stem
    parsed_path = settings.parsed_dir / f"{stem}.json"
    if not parsed_path.exists():
        raise FileNotFoundError(f"No parsed data found for '{filename}' at {parsed_path}")

    return json.loads(parsed_path.read_text(encoding="utf-8"))


def list_ingested_documents() -> List[str]:
    """Return the filenames of all documents that have been parsed."""
    settings.ensure_dirs()
    return [p.stem for p in settings.parsed_dir.glob("*.json")]
