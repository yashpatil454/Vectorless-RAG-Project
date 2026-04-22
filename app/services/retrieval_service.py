from __future__ import annotations

from typing import Any, Dict, List, Optional

from llama_index.core.indices.tree.base import TreeIndex as _TreeIndex

from app.models.schemas import EvidenceNode
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def filter_by_metadata(
    chunks: List[Dict[str, Any]],
    document_filter: Optional[str] = None,
    tags_filter: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Return only the chunks that match the supplied metadata filters.

    Filters are applied with AND logic:
    - ``document_filter``: case-insensitive substring match against ``doc_name``.
    - ``tags_filter``: every supplied tag must be present in the chunk's ``tags`` list.

    If both filters are ``None`` / empty, all chunks are returned unchanged.
    """
    result = chunks

    if document_filter:
        doc_lower = document_filter.lower()
        result = [c for c in result if doc_lower in c.get("doc_name", "").lower()]
        logger.info("After document filter ('%s'): %d chunks remain.", document_filter, len(result))

    if tags_filter:
        def _has_all_tags(chunk: Dict[str, Any]) -> bool:
            chunk_tags = [t.lower() for t in chunk.get("tags", [])]
            return all(t.lower() in chunk_tags for t in tags_filter)

        result = [c for c in result if _has_all_tags(c)]
        logger.info("After tags filter %s: %d chunks remain.", tags_filter, len(result))

    return result


def retrieve_from_tree(
    index: _TreeIndex,
    query: str,
    top_k: int = 5,
) -> List[EvidenceNode]:
    """Query the TreeIndex and return matching nodes as EvidenceNode objects.

    Uses LlamaIndex's built-in ``as_query_engine()`` which traverses the tree
    via LLM-based node selection (no vector similarity).

    Args:
        index: A loaded LlamaIndex TreeIndex.
        query: The natural-language query string.
        top_k: Maximum number of evidence nodes to return.

    Returns:
        A list of ``EvidenceNode`` instances ordered by relevance as judged
        by the tree traversal.
    """
    logger.info("Querying TreeIndex with: '%s' (top_k=%d)", query, top_k)

    query_engine = index.as_query_engine(
        child_branch_factor=2,
        verbose=False,
    )

    response = query_engine.query(query)
    evidence: List[EvidenceNode] = []

    source_nodes = getattr(response, "source_nodes", []) or []
    for i, node_with_score in enumerate(source_nodes[:top_k]):
        node = node_with_score.node
        evidence.append(
            EvidenceNode(
                node_id=node.node_id,
                summary=node.metadata.get("summary") or node.get_content()[:300],
                metadata=node.metadata or {},
            )
        )

    logger.info("Retrieved %d evidence nodes.", len(evidence))
    return evidence
