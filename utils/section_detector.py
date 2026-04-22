from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Heading patterns
# ---------------------------------------------------------------------------

# Level-1: "1. Introduction", "1 INTRODUCTION", bare known names, or ALL-CAPS line
_L1_NUMBERED = re.compile(r"^\d+\.?\s+[A-Z]")
_L1_NAMED = re.compile(
    r"^\s*(abstract|introduction|background|related\s+work|literature\s+review|"
    r"methodology|methods?|materials?\s+and\s+methods?|experiments?|results?|"
    r"discussion|conclusion|conclusions?\s+and\s+future\s+work|"
    r"acknowledgements?|references?|appendix|overview|summary)\s*$",
    re.IGNORECASE,
)
_L1_ALLCAPS = re.compile(r"^[A-Z][A-Z\s\-]{4,50}$")

# Level-2: "1.1 Background", "2.3. Data Collection"
_L2_NUMBERED = re.compile(r"^\d+\.\d+\.?\s+[A-Z]")


def _classify_line(line: str) -> Optional[int]:
    """Return heading level (1 or 2) or None if not a heading."""
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return None
    if _L2_NUMBERED.match(stripped):
        return 2
    if _L1_NUMBERED.match(stripped) or _L1_NAMED.match(stripped) or _L1_ALLCAPS.match(stripped):
        return 1
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def detect_sections(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert a list of per-page dicts into section-aware chunks.

    Each input page dict must have at minimum:
        - ``text`` (str)
        - ``page_number`` (int, 1-based)
        - ``source`` (str)

    Returns a list of section-chunk dicts with:
        - ``text``         : full text of the section (may span multiple pages)
        - ``section``      : L1 heading label (str)
        - ``subsection``   : L2 heading label (str, empty string if none)
        - ``level``        : 1 or 2
        - ``page_range``   : [first_page, last_page]  (both 1-based)
        - ``page_number``  : first_page (backwards-compat alias)
        - ``source``       : inherited from first page
        - ``char_count``   : len(text)

    Falls back to LLM-based structure detection when fewer than 2 headings are
    found across all pages.
    """
    if not pages:
        return []

    source = pages[0].get("source", "")

    # ------------------------------------------------------------------
    # Pass 1: detect headings across all pages
    # ------------------------------------------------------------------
    heading_count = 0
    for page in pages:
        for line in (page.get("text") or "").splitlines():
            if _classify_line(line) is not None:
                heading_count += 1

    if heading_count < 2:
        logger.info(
            "Only %d heading(s) detected — using LLM fallback for section detection.", heading_count
        )
        return _llm_fallback(pages, source)

    # ------------------------------------------------------------------
    # Pass 2: build section chunks
    # ------------------------------------------------------------------
    chunks: List[Dict[str, Any]] = []
    current_section = "Preamble"
    current_subsection = ""
    current_level = 1
    current_text_lines: List[str] = []
    current_start_page = pages[0]["page_number"]
    current_end_page = pages[0]["page_number"]

    def _flush(end_page: int) -> None:
        text = "\n".join(current_text_lines).strip()
        if text:
            chunks.append(
                {
                    "text": text,
                    "section": current_section,
                    "subsection": current_subsection,
                    "level": current_level,
                    "page_range": [current_start_page, end_page],
                    "page_number": current_start_page,
                    "source": source,
                    "char_count": len(text),
                }
            )

    for page in pages:
        page_num = page["page_number"]
        current_end_page = page_num

        for line in (page.get("text") or "").splitlines():
            level = _classify_line(line)

            if level == 1:
                _flush(current_end_page)
                current_section = line.strip()
                current_subsection = ""
                current_level = 1
                current_text_lines = []
                current_start_page = page_num
            elif level == 2:
                _flush(current_end_page)
                current_subsection = line.strip()
                current_level = 2
                current_text_lines = []
                current_start_page = page_num
            else:
                current_text_lines.append(line)

    _flush(current_end_page)

    logger.info("Detected %d section chunks from %d pages.", len(chunks), len(pages))
    return chunks if chunks else _page_fallback(pages, source)


# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------

def _page_fallback(pages: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    """Return one chunk per page with a synthetic section label (last resort)."""
    chunks = []
    for page in pages:
        text = (page.get("text") or "").strip()
        if text:
            pn = page["page_number"]
            chunks.append(
                {
                    "text": text,
                    "section": f"Page {pn}",
                    "subsection": "",
                    "level": 1,
                    "page_range": [pn, pn],
                    "page_number": pn,
                    "source": source,
                    "char_count": len(text),
                }
            )
    return chunks


def _llm_fallback(pages: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    """Use Gemini to infer structure when regex detects no headings."""
    import json

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        from app.config import settings

        llm = ChatGoogleGenerativeAI(
            model=settings.model_name,
            google_api_key=settings.google_api_key or os.environ.get("GOOGLE_API_KEY", ""),
            temperature=0,
        )

        full_text = "\n\n".join(
            f"[Page {p['page_number']}]\n{p.get('text', '')}" for p in pages
        )
        # Limit context to avoid token overflow (~12 000 chars ≈ ~3 000 tokens)
        truncated = full_text[:12_000]

        prompt = (
            "You are a document structure analyser. "
            "Given the text below, identify the logical sections. "
            "Return ONLY a JSON array where each element is:\n"
            '{"title": "<section title>", "content": "<full section text>"}\n'
            "Do not include any other text outside the JSON.\n\n"
            f"DOCUMENT:\n{truncated}"
        )

        response = llm.invoke(prompt)
        raw = response.content.strip()

        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        sections = json.loads(raw)
        if not isinstance(sections, list):
            raise ValueError("LLM did not return a list")

    except Exception as exc:
        logger.warning("LLM section fallback failed (%s); using page-level chunks.", exc)
        return _page_fallback(pages, source)

    # Map LLM sections back to page numbers (best-effort: use first page that
    # contains a significant portion of the section text)
    chunks: List[Dict[str, Any]] = []
    page_texts = {p["page_number"]: (p.get("text") or "") for p in pages}

    for i, sec in enumerate(sections):
        title = (sec.get("title") or f"Section {i + 1}").strip()
        content = (sec.get("content") or "").strip()
        if not content:
            continue

        # Find first page whose text overlaps with the section content
        first_words = " ".join(content.split()[:8]).lower()
        matched_page = pages[0]["page_number"]
        for pn, pt in page_texts.items():
            if first_words and first_words[:40] in pt.lower():
                matched_page = pn
                break

        chunks.append(
            {
                "text": content,
                "section": title,
                "subsection": "",
                "level": 1,
                "page_range": [matched_page, matched_page],
                "page_number": matched_page,
                "source": source,
                "char_count": len(content),
            }
        )

    logger.info("LLM fallback produced %d section chunks.", len(chunks))
    return chunks if chunks else _page_fallback(pages, source)
