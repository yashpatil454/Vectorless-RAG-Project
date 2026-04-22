from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings
from app.models.schemas import EvidenceNode
from utils.logging_utils import get_logger
from utils.tree_model import CustomTreeIndex, TreeNode

logger = get_logger(__name__)

_RELEVANCE_THRESHOLD = 3

# Ask the LLM to rate each passage 1-5 in a single call per tree level.
_SCORE_PROMPT = """\
Query: {query}

Rate the relevance of each passage below to the query on a scale of 1 (not relevant) to 5 (highly relevant).

{items}

Return ONLY a JSON array, e.g.: [{{"index": 0, "score": 4}}, {{"index": 1, "score": 2}}]
No explanation. No markdown code fences."""


def filter_by_metadata(
    chunks: List[Dict[str, Any]],
    document_filter: Optional[str] = None,
    tags_filter: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Return only the chunks that match the supplied metadata filters (AND logic)."""
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


def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.model_name,
        google_api_key=settings.google_api_key or os.environ.get("GOOGLE_API_KEY", ""),
        temperature=0.0,
    )


def _score_batch(
    nodes: List[TreeNode],
    query: str,
    llm: ChatGoogleGenerativeAI,
) -> List[Tuple[TreeNode, int]]:
    """Score a batch of nodes in a single LLM call. Returns (node, score) pairs."""
    if not nodes:
        return []

    items = "\n\n".join(
        f"[{i}] {(node.summary or node.text)[:500]}" for i, node in enumerate(nodes)
    )
    prompt = _SCORE_PROMPT.format(query=query, items=items)

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        # Strip markdown code fences if the model adds them anyway
        if content.startswith("```"):
            content = "\n".join(
                line for line in content.splitlines()
                if not line.startswith("```")
            ).strip()
        scores_raw = json.loads(content)
        score_map = {int(e["index"]): int(e["score"]) for e in scores_raw}
    except Exception as exc:
        logger.warning("Batch scoring failed (%s); defaulting all nodes to score 3.", exc)
        score_map = {i: 3 for i in range(len(nodes))}

    return [(nodes[i], score_map.get(i, 1)) for i in range(len(nodes))]


def retrieve_from_tree(
    index: CustomTreeIndex,
    query: str,
    top_k: int = 5,
) -> List[EvidenceNode]:
    """BFS traversal of the custom tree, scoring each level in one LLM call.

    Nodes scoring >= _RELEVANCE_THRESHOLD are explored further (or collected if
    they are leaves). Nodes below the threshold prune that entire branch.
    """
    logger.info("Traversing custom tree for: '%s' (top_k=%d)", query, top_k)
    llm = _get_llm()

    collected: List[Tuple[TreeNode, int]] = []
    visited: set = set()
    frontier: List[str] = list(index.root_ids)

    while frontier:
        batch = [index.nodes[nid] for nid in frontier if nid in index.nodes]
        if not batch:
            break

        scored = _score_batch(batch, query, llm)
        next_frontier: List[str] = []

        for node, score in scored:
            if node.node_id in visited:
                continue
            visited.add(node.node_id)

            if not node.children:
                # Leaf — collect if relevant
                if score >= _RELEVANCE_THRESHOLD:
                    collected.append((node, score))
            else:
                # Internal node — recurse into children if relevant
                if score >= _RELEVANCE_THRESHOLD:
                    next_frontier.extend(node.children)

        frontier = next_frontier

    # Sort by relevance score descending, return top_k
    collected.sort(key=lambda x: x[1], reverse=True)
    evidence = [
        EvidenceNode(
            node_id=node.node_id,
            summary=node.summary or node.text[:300],
            metadata={**node.metadata, "relevance_score": score},
        )
        for node, score in collected[:top_k]
    ]
    logger.info("Retrieved %d evidence nodes.", len(evidence))
    return evidence

