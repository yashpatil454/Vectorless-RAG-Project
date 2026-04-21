from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from app.config import settings
from app.models.schemas import HistoryEntry, QueryResponse
from utils.logging_utils import get_logger

logger = get_logger(__name__)

# In-process list of history entries (fast access for the current session).
_history: List[HistoryEntry] = []


def _load_from_disk() -> None:
    """Populate the in-memory list from the history JSON file if it exists."""
    global _history
    path = settings.history_file
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            _history = [HistoryEntry(**e) for e in raw]
            logger.info("Loaded %d history entries from disk.", len(_history))
        except Exception as exc:
            logger.warning("Could not load history from disk: %s", exc)
            _history = []


def _persist_to_disk() -> None:
    """Write the in-memory history to disk (best-effort)."""
    path = settings.history_file
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([e.model_dump() for e in _history], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("Could not persist history to disk: %s", exc)


def add_entry(query_response: QueryResponse) -> HistoryEntry:
    """Create and store a history entry from a completed QueryResponse."""
    entry = HistoryEntry(
        id=str(uuid.uuid4()),
        query=query_response.query,
        answer=query_response.answer,
        status=query_response.status,
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
        evidence_count=len(query_response.evidence),
        tokens_used=query_response.tokens_used,
        llm_calls=query_response.llm_calls,
    )
    _history.append(entry)
    _persist_to_disk()
    return entry


def get_history(limit: int = 50) -> List[HistoryEntry]:
    """Return the most recent ``limit`` history entries (newest first)."""
    return list(reversed(_history[-limit:]))


def clear_history() -> int:
    """Remove all history entries.  Returns the count deleted."""
    global _history
    count = len(_history)
    _history = []
    _persist_to_disk()
    return count


# Load existing history when the module is first imported.
_load_from_disk()
