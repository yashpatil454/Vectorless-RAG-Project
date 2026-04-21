from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from utils.metadata_extractor import attach_metadata
from utils.pdf_parser import parse_pdf


# ---------------------------------------------------------------------------
# pdf_parser tests
# ---------------------------------------------------------------------------

class TestParsePdf:
    def test_raises_on_invalid_file(self, tmp_path: Path):
        bad_file = tmp_path / "not_a_pdf.pdf"
        bad_file.write_bytes(b"this is not a PDF")
        with pytest.raises(Exception):
            parse_pdf(bad_file)

    def test_returns_list(self, tmp_path: Path):
        # Minimal valid PDF with one page of text (created with fpdf-like bytes is
        # non-trivial; we use a pre-built minimal PDF byte string instead).
        # Since we cannot guarantee pdfplumber works on an arbitrary byte string,
        # we skip if fpdf is not available and only test type contract.
        try:
            from fpdf import FPDF  # type: ignore

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Hello world from test PDF.", ln=True)
            pdf_bytes = pdf.output(dest="S").encode("latin-1")
        except ImportError:
            pytest.skip("fpdf not installed; skipping full parse test.")

        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(pdf_bytes)

        chunks = parse_pdf(pdf_file)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        assert "text" in chunks[0]
        assert "page_number" in chunks[0]
        assert "source" in chunks[0]


# ---------------------------------------------------------------------------
# metadata_extractor tests
# ---------------------------------------------------------------------------

class TestAttachMetadata:
    def test_attaches_doc_name(self):
        chunks = [{"text": "Hello", "page_number": 1, "source": "/tmp/a.pdf"}]
        enriched = attach_metadata(chunks, filename="a.pdf")
        assert enriched[0]["doc_name"] == "a.pdf"

    def test_attaches_tags(self):
        chunks = [{"text": "Hello", "page_number": 1, "source": "/tmp/a.pdf"}]
        enriched = attach_metadata(chunks, filename="a.pdf", tags=["finance", "2024"])
        assert "finance" in enriched[0]["tags"]

    def test_does_not_mutate_original(self):
        original = [{"text": "Hello", "page_number": 1, "source": "/tmp/a.pdf"}]
        attach_metadata(original, filename="a.pdf")
        assert "doc_name" not in original[0]

    def test_upload_timestamp_added(self):
        chunks = [{"text": "Hello", "page_number": 1, "source": "/tmp/a.pdf"}]
        enriched = attach_metadata(chunks, filename="a.pdf")
        assert "upload_timestamp" in enriched[0]

    def test_empty_chunks_returns_empty(self):
        assert attach_metadata([], filename="a.pdf") == []
