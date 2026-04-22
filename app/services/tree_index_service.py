from __future__ import annotations

import os
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.data_structs.data_structs import IndexGraph
from llama_index.core.indices.tree.base import TreeIndex as _TreeIndex
from llama_index.core.schema import TextNode
from llama_index.llms.gemini import Gemini

from app.config import settings
from utils.logging_utils import get_logger

logger = get_logger(__name__)

_index_cache: Optional[_TreeIndex] = None


def _get_llm() -> Gemini:
    return Gemini(
        model=settings.model_name,
        api_key=settings.google_api_key or os.environ.get("GOOGLE_API_KEY", ""),
    )


def _get_langchain_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.model_name,
        google_api_key=settings.google_api_key or os.environ.get("GOOGLE_API_KEY", ""),
        temperature=0.0,
    )


def _make_node(text: str, metadata: Dict[str, Any]) -> TextNode:
    return TextNode(id_=str(uuid.uuid4()), text=text, metadata=metadata)


def _llm_call(prompt: str, llm: ChatGoogleGenerativeAI, fallback: str) -> Tuple[str, int]:
    """Single LLM call. Returns (text, tokens_used)."""
    try:
        response = llm.invoke(prompt)
        tokens = 0
        um = getattr(response, "usage_metadata", None)
        if um is not None:
            # UsageMetadata may be a TypedDict (dict) or a Pydantic/dataclass object
            if isinstance(um, dict):
                tokens = int(um.get("total_tokens", 0))
            else:
                tokens = int(
                    getattr(um, "total_tokens", None)
                    or getattr(um, "totalTokenCount", None)
                    or 0
                )
        logger.debug("LLM call OK — tokens: %d", tokens)
        return (response.content.strip(), tokens)
    except Exception as exc:
        logger.warning("LLM call failed: %s — using fallback.", exc)
        return (fallback, 0)


def _summarize_leaf(
    text: str, section: str, doc_name: str, llm: ChatGoogleGenerativeAI
) -> Tuple[str, int]:
    """Generate a 2-3 sentence summary for a leaf chunk."""
    if len(text) < 150:
        return (text, 0)
    prompt = (
        f"Summarise this content in 2-3 concise sentences. "
        f"Document: {doc_name}, Section: {section}\n\n{text[:4000]}"
    )
    return _llm_call(prompt, llm, text[:300])


def _summarize_group(
    texts: List[str], label: str, llm: ChatGoogleGenerativeAI
) -> Tuple[str, int]:
    """Generate a 3-5 sentence summary for a group of texts (section or doc)."""
    combined = "\n\n---\n\n".join(t[:2000] for t in texts if t.strip())
    if not combined:
        return ("", 0)
    prompt = (
        f"Summarise the following content from '{label}' in 3-5 concise sentences "
        f"for a document QA system:\n\n{combined}"
    )
    return _llm_call(prompt, llm, combined[:400])


def _build_hierarchical_tree(
    chunks: List[Dict[str, Any]],
) -> Tuple[IndexGraph, Dict[str, TextNode], Dict[str, Any]]:
    """Build a Doc → Section → Subsection → Leaf hierarchy.

    All summarization happens here at build time:
    - Leaf nodes:        1 LLM call per chunk (leaf summary)
    - Subsection nodes:  no LLM call (concat of leaf summaries)
    - Section nodes:     1 LLM call per section
    - Doc root nodes:    1 LLM call per document
    """
    llm = _get_langchain_llm()
    total_tokens = 0
    total_llm_calls = 0

    # Group: {doc_name: {section: {subsection: [chunk]}}}
    grouped: Dict[str, Dict[str, Dict[str, List[Dict]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for chunk in chunks:
        nested = chunk.get("metadata", {})
        doc = chunk.get("doc_name") or nested.get("doc_name", "unknown")
        sec = (nested.get("section") or chunk.get("section") or "Document").strip() or "Document"
        sub = (nested.get("subsection") or chunk.get("subsection") or "").strip()
        grouped[doc][sec][sub].append(chunk)

    index_graph = IndexGraph()
    all_nodes: Dict[str, TextNode] = {}
    root_counter = 0

    for doc_name, sections in grouped.items():
        section_nodes: List[TextNode] = []

        for sec_title, subsections in sections.items():
            subsec_nodes: List[TextNode] = []

            for sub_title, leaf_chunks in subsections.items():
                # ── Leaf nodes (1 LLM call per chunk) ───────────────────────────────
                leaves: List[TextNode] = []
                for c in leaf_chunks:
                    nested = c.get("metadata", {})
                    leaf_summary, leaf_tokens = _summarize_leaf(
                        text=c["text"],
                        section=sec_title,
                        doc_name=doc_name,
                        llm=llm,
                    )
                    total_tokens += leaf_tokens
                    total_llm_calls += 1

                    leaf = _make_node(
                        text=c["text"],
                        metadata={
                            "doc_name": doc_name,
                            "page_number": c.get("page_number", 0),
                            "upload_timestamp": c.get("upload_timestamp", ""),
                            "tags": c.get("tags", []),
                            "source": c.get("source", ""),
                            "summary": leaf_summary,
                            "section": sec_title,
                            "subsection": sub_title,
                            "page_range": nested.get("page_range", c.get("page_range", [])),
                        },
                    )
                    leaves.append(leaf)
                    all_nodes[leaf.node_id] = leaf
                    # Leaf nodes have no children — register with empty list so
                    # LlamaIndex TreeRetriever never gets a KeyError on dict lookup
                    index_graph.node_id_to_children_ids[leaf.node_id] = []

                # ── Subsection node (no LLM — concat leaf summaries) ─────────────
                if sub_title:
                    sub_text = "\n\n".join(
                        n.metadata.get("summary") or n.text[:300] for n in leaves
                    )
                    sub_node = _make_node(
                        text=sub_text,
                        metadata={
                            "doc_name": doc_name,
                            "section": sec_title,
                            "subsection": sub_title,
                            "node_type": "subsection",
                        },
                    )
                    all_nodes[sub_node.node_id] = sub_node
                    index_graph.node_id_to_children_ids[sub_node.node_id] = [
                        n.node_id for n in leaves
                    ]
                    subsec_nodes.append(sub_node)
                else:
                    # No subsection label — leaves sit directly under section
                    subsec_nodes.extend(leaves)

            # ── Section node (1 LLM call) ───────────────────────────────────
            sec_texts = [n.text[:1500] for n in subsec_nodes]
            sec_summary, sec_tokens = _summarize_group(
                sec_texts, f"{doc_name}/{sec_title}", llm
            )
            total_tokens += sec_tokens
            total_llm_calls += 1

            sec_node = _make_node(
                text=sec_summary or "\n\n".join(sec_texts)[:600],
                metadata={
                    "doc_name": doc_name,
                    "section": sec_title,
                    "node_type": "section",
                },
            )
            all_nodes[sec_node.node_id] = sec_node
            index_graph.node_id_to_children_ids[sec_node.node_id] = [
                n.node_id for n in subsec_nodes
            ]
            section_nodes.append(sec_node)

        # ── Doc root node (1 LLM call) ─────────────────────────────────────
        doc_texts = [n.text[:1500] for n in section_nodes]
        doc_summary, doc_tokens = _summarize_group(doc_texts, doc_name, llm)
        total_tokens += doc_tokens
        total_llm_calls += 1

        doc_node = _make_node(
            text=doc_summary or "\n\n".join(doc_texts)[:600],
            metadata={"doc_name": doc_name, "node_type": "document"},
        )
        all_nodes[doc_node.node_id] = doc_node
        index_graph.node_id_to_children_ids[doc_node.node_id] = [
            n.node_id for n in section_nodes
        ]
        index_graph.root_nodes[root_counter] = doc_node.node_id
        root_counter += 1

    # Populate flat integer index required internally by LlamaIndex
    for i, node_id in enumerate(all_nodes):
        index_graph.all_nodes[i] = node_id

    build_stats = {
        "tokens_used": total_tokens,
        "llm_calls": total_llm_calls,
        "node_count": len(all_nodes),
    }
    logger.info(
        "Hierarchical tree built: %d nodes, %d LLM calls, %d tokens",
        len(all_nodes), total_llm_calls, total_tokens,
    )
    return index_graph, all_nodes, build_stats


def build_index(chunks: List[Dict[str, Any]]) -> Tuple[_TreeIndex, Dict[str, Any]]:
    """Build a hierarchical TreeIndex and persist it.

    Returns (index, build_stats) where build_stats has:
        tokens_used, llm_calls, node_count
    """
    global _index_cache
    settings.ensure_dirs()

    from llama_index.core import Settings as LISettings

    llm = _get_llm()
    LISettings.llm = llm
    LISettings.embed_model = None  # type: ignore[assignment]

    logger.info("Building hierarchical TreeIndex from %d chunks …", len(chunks))
    index_graph, all_nodes, build_stats = _build_hierarchical_tree(chunks)

    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(list(all_nodes.values()))

    index = _TreeIndex(
        index_struct=index_graph,
        storage_context=storage_context,
    )
    persist_path = str(settings.index_store_dir)
    index.storage_context.persist(persist_dir=persist_path)
    logger.info("TreeIndex persisted to: %s", persist_path)

    _index_cache = index
    return index, build_stats


def load_index() -> Optional[_TreeIndex]:
    """Load a previously persisted TreeIndex from disk.

    Returns ``None`` if no persisted index is found.
    """
    global _index_cache

    if _index_cache is not None:
        return _index_cache

    persist_path = settings.index_store_dir
    if not (persist_path / "docstore.json").exists():
        logger.info("No persisted index found at: %s", persist_path)
        return None

    logger.info("Loading TreeIndex from: %s", persist_path)
    try:
        from llama_index.core import Settings as LISettings

        llm = _get_llm()
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
    index, _ = build_index(chunks)
    return index


def get_index_node_count() -> Optional[int]:
    """Return the number of nodes in the loaded index, or None."""
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
