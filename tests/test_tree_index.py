from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunks(n: int = 3) -> List[Dict[str, Any]]:
    return [
        {
            "text": f"This is page {i} of the test document.",
            "page_number": i,
            "source": "/tmp/test.pdf",
            "doc_name": "test.pdf",
            "upload_timestamp": "2026-01-01T00:00:00+00:00",
            "tags": ["test"],
        }
        for i in range(1, n + 1)
    ]


# ---------------------------------------------------------------------------
# Tree index service — unit tests with mocked LlamaIndex
# ---------------------------------------------------------------------------

class TestTreeIndexService:
    def test_chunks_to_documents_count(self):
        """_chunks_to_documents should produce one Document per chunk."""
        from llama_index.core import Document

        from app.services.tree_index_service import _chunks_to_documents

        chunks = _make_chunks(5)
        docs = _chunks_to_documents(chunks)
        assert len(docs) == 5
        assert all(isinstance(d, Document) for d in docs)

    def test_chunks_to_documents_metadata(self):
        """Metadata fields should be transferred correctly."""
        from app.services.tree_index_service import _chunks_to_documents

        chunks = _make_chunks(1)
        docs = _chunks_to_documents(chunks)
        assert docs[0].metadata["doc_name"] == "test.pdf"
        assert docs[0].metadata["page_number"] == 1

    def test_load_index_returns_none_when_no_store(self, tmp_path: Path, monkeypatch):
        """load_index() should return None when the index store directory is empty."""
        import app.services.tree_index_service as tis

        monkeypatch.setattr(tis, "_index_cache", None)

        from app.config import settings

        monkeypatch.setattr(settings, "index_store_dir", tmp_path)

        result = tis.load_index()
        assert result is None

    def test_invalidate_cache_clears_module_cache(self, monkeypatch):
        """invalidate_cache() should set the module-level cache to None."""
        import app.services.tree_index_service as tis

        tis._index_cache = MagicMock()  # Simulate a loaded index
        tis.invalidate_cache()
        assert tis._index_cache is None
