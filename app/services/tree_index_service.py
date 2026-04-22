from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from llama_index.core import (
    Document,
    StorageContext,
    TreeIndex,
    load_index_from_storage,
)
from llama_index.core.indices.tree.base import TreeIndex as _TreeIndex
from llama_index.llms.gemini import Gemini

from app.config import settings
from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Module-level cache so the index is only built/loaded once per process.
_index_cache: Optional[_TreeIndex] = None


def _get_llm() -> Gemini:
    """Return a Google Gemini LLM configured from settings."""
    return Gemini(
        model=settings.model_name,
        api_key=settings.google_api_key or os.environ.get("GOOGLE_API_KEY", ""),
    )


def _chunks_to_documents(chunks: List[Dict[str, Any]]) -> List[Document]:
    """Convert parsed chunk dicts into LlamaIndex Document objects."""
    docs: List[Document] = []
    for chunk in chunks:
        nested_meta = chunk.get("metadata", {})
        doc = Document(
            text=chunk["text"],
            metadata={
                "doc_name": chunk.get("doc_name", ""),
                "page_number": chunk.get("page_number", 0),
                "upload_timestamp": chunk.get("upload_timestamp", ""),
                "tags": chunk.get("tags", []),
                "source": chunk.get("source", ""),
                # Section-aware chunking fields (stored in nested metadata by ingestion_service)
                "summary": nested_meta.get("summary", ""),
                "section": nested_meta.get("section", chunk.get("section", "")),
                "subsection": nested_meta.get("subsection", chunk.get("subsection", "")),
                "page_range": nested_meta.get("page_range", chunk.get("page_range", [])),
            },
        )
        docs.append(doc)
    return docs


def build_index(chunks: List[Dict[str, Any]]) -> _TreeIndex:
    """Build a LlamaIndex TreeIndex from parsed chunk dicts and persist it.

    The index is written to ``settings.index_store_dir`` via
    ``StorageContext.persist()``.  A module-level cache is updated so
    subsequent calls to ``load_index()`` return immediately.
    """
    global _index_cache

    settings.ensure_dirs()
    documents = _chunks_to_documents(chunks)
    logger.info("Building TreeIndex from %d document chunks …", len(documents))

    llm = _get_llm()
    from llama_index.core import Settings as LISettings

    LISettings.llm = llm
    # Disable the default vector embed model — tree index uses LLM traversal
    LISettings.embed_model = None  # type: ignore[assignment]

    index = TreeIndex.from_documents(documents)
    persist_path = str(settings.index_store_dir)
    index.storage_context.persist(persist_dir=persist_path)
    logger.info("TreeIndex persisted to: %s", persist_path)

    _index_cache = index
    return index


def load_index() -> Optional[_TreeIndex]:
    """Load a previously persisted TreeIndex from disk.

    Returns ``None`` if no persisted index is found.
    """
    global _index_cache

    if _index_cache is not None:
        return _index_cache

    persist_path = settings.index_store_dir
    docstore_file = persist_path / "docstore.json"

    if not docstore_file.exists():
        logger.info("No persisted index found at: %s", persist_path)
        return None

    logger.info("Loading TreeIndex from: %s", persist_path)
    try:
        llm = _get_llm()
        from llama_index.core import Settings as LISettings

        LISettings.llm = llm
        LISettings.embed_model = None  # type: ignore[assignment]

        storage_context = StorageContext.from_defaults(persist_dir=str(persist_path))
        index = load_index_from_storage(storage_context)
        _index_cache = index
        logger.info("TreeIndex loaded successfully.")
        return index
    except Exception as exc:
        logger.error("Failed to load index: %s", exc)
        return None


def get_or_build_index(chunks: List[Dict[str, Any]]) -> _TreeIndex:
    """Return the cached or persisted index, building it from chunks if needed."""
    existing = load_index()
    if existing is not None:
        return existing
    return build_index(chunks)


def get_index_node_count() -> Optional[int]:
    """Return the number of leaf nodes in the loaded index, or None."""
    index = load_index()
    if index is None:
        return None
    try:
        return len(index.index_struct.all_nodes)
    except Exception:
        return None


def invalidate_cache() -> None:
    """Clear the module-level index cache to force a reload from disk."""
    global _index_cache
    _index_cache = None
