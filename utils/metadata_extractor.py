from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def attach_metadata(
    chunks: List[Dict[str, Any]],
    filename: str,
    tags: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Enrich each chunk dict with document-level metadata.

    Added fields (will not overwrite existing keys already set in ``chunks``):
        - ``doc_name``: filename (no directory component).
        - ``upload_timestamp``: ISO-8601 UTC timestamp at the moment of calling.
        - ``tags``: list of user-supplied tags (empty list if not provided).

    The function is non-destructive: it returns a new list of new dicts so the
    caller's original list is not mutated.
    """
    upload_ts = datetime.now(tz=timezone.utc).isoformat()
    enriched: List[Dict[str, Any]] = []

    for chunk in chunks:
        enriched_chunk: Dict[str, Any] = {**chunk}
        enriched_chunk.setdefault("doc_name", filename)
        enriched_chunk.setdefault("upload_timestamp", upload_ts)
        enriched_chunk.setdefault("tags", tags or [])
        enriched.append(enriched_chunk)

    return enriched
