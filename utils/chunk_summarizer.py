from __future__ import annotations

import os

from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Chunks shorter than this are returned as-is (no LLM call).
_MIN_LEN_FOR_SUMMARY = 200


def summarize_chunk(text: str, section: str, doc_name: str) -> str:
    """Return a 2-3 sentence Gemini summary of a section chunk.

    If ``text`` is shorter than ``_MIN_LEN_FOR_SUMMARY`` characters the text
    itself is returned unchanged to avoid a needless API call.

    Args:
        text:     Full text of the section chunk.
        section:  Section heading label (used to ground the summary prompt).
        doc_name: Document filename (used for logging only).

    Returns:
        A 2-3 sentence summary string, or the original text if it is too short
        to summarise or if the LLM call fails.
    """
    if len(text) < _MIN_LEN_FOR_SUMMARY:
        return text

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        from app.config import settings

        llm = ChatGoogleGenerativeAI(
            model=settings.model_name,
            google_api_key=settings.google_api_key or os.environ.get("GOOGLE_API_KEY", ""),
            temperature=0,
        )

        prompt = (
            f"Summarise the following section from a document in 2-3 concise sentences. "
            f"Focus on the key points only. Do not include preamble or meta-commentary.\n\n"
            f"Section: {section}\n\n"
            f"{text[:4000]}"  # cap to avoid token overflow
        )

        response = llm.invoke(prompt)
        summary = (response.content or "").strip()
        if not summary:
            raise ValueError("Empty response from LLM")

        logger.debug("Summarised section '%s' of '%s' (%d chars → %d chars).",
                     section, doc_name, len(text), len(summary))
        return summary

    except Exception as exc:
        logger.warning(
            "Could not summarise section '%s' of '%s': %s — using raw text truncation.",
            section, doc_name, exc,
        )
        # Graceful degradation: first 300 chars as a pseudo-summary
        return text[:300].strip()
