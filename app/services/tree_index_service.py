from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings
from utils.logging_utils import get_logger
from utils.tree_model import CustomTreeIndex, TreeNode

logger = get_logger(__name__)

_index_cache: Optional[CustomTreeIndex] = None
_TREE_FILE = "tree.json"


def _get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.model_name,
        google_api_key=settings.google_api_key or os.environ.get("GOOGLE_API_KEY", ""),
        temperature=0.0,
    )


def _extract_tokens(response: Any) -> int:
    rm = getattr(response, "response_metadata", None) or {}
    rm_um = rm.get("usage_metadata", {}) if isinstance(rm, dict) else {}
    if rm_um:
        v = rm_um.get("total_token_count") or rm_um.get("totalTokenCount")
        if v:
            return int(v)
    um = getattr(response, "usage_metadata", None)
    if um is not None:
        v = um.get("total_tokens", 0) if isinstance(um, dict) else getattr(um, "total_tokens", 0)
        if v:
            return int(v)
    return 0


def _llm_call(prompt: str, llm: ChatGoogleGenerativeAI, fallback: str) -> Tuple[str, int]:
    try:
        response = llm.invoke(prompt)
        return (response.content.strip(), _extract_tokens(response))
    except Exception as exc:
        logger.warning("LLM call failed: %s - using fallback.", exc)
        return (fallback, 0)


def _summarize_leaf(
    text: str, section: str, doc_name: str, llm: ChatGoogleGenerativeAI
) -> Tuple[str, int]:
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
) -> Tuple[CustomTreeIndex, Dict[str, Any]]:
    llm = _get_llm()
    total_tokens = 0
    total_llm_calls = 0
    all_nodes: Dict[str, TreeNode] = {}
    root_ids: List[str] = []

    grouped: Dict[str, Dict[str, Dict[str, List[Dict]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for chunk in chunks:
        nested = chunk.get("metadata", {})
        doc = (chunk.get("doc_name") or nested.get("doc_name") or "unknown").strip()
        sec = (nested.get("section") or chunk.get("section") or "Document").strip() or "Document"
        sub = (nested.get("subsection") or chunk.get("subsection") or "").strip()
        grouped[doc][sec][sub].append(chunk)

    for doc_name, sections in grouped.items():
        section_node_ids: List[str] = []

        for sec_title, subsections in sections.items():
            subsec_node_ids: List[str] = []

            for sub_title, leaf_chunks in subsections.items():
                leaf_ids: List[str] = []

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

                    nid = CustomTreeIndex.make_node_id()
                    all_nodes[nid] = TreeNode(
                        node_id=nid,
                        node_type="leaf",
                        text=c["text"],
                        summary=leaf_summary,
                        metadata={
                            "doc_name": doc_name,
                            "page_number": c.get("page_number", 0),
                            "upload_timestamp": c.get("upload_timestamp", ""),
                            "tags": c.get("tags", []),
                            "source": c.get("source", ""),
                            "section": sec_title,
                            "subsection": sub_title,
                            "page_range": nested.get("page_range", c.get("page_range", [])),
                        },
                        children=[],
                    )
                    leaf_ids.append(nid)

                if sub_title:
                    sub_text = "\n\n".join(
                        all_nodes[lid].summary or all_nodes[lid].text[:300]
                        for lid in leaf_ids
                    )
                    sub_id = CustomTreeIndex.make_node_id()
                    all_nodes[sub_id] = TreeNode(
                        node_id=sub_id,
                        node_type="subsection",
                        text=sub_text,
                        summary=sub_text[:400],
                        metadata={
                            "doc_name": doc_name,
                            "section": sec_title,
                            "subsection": sub_title,
                        },
                        children=leaf_ids,
                    )
                    subsec_node_ids.append(sub_id)
                else:
                    subsec_node_ids.extend(leaf_ids)

            sec_texts = [
                all_nodes[nid].summary or all_nodes[nid].text[:300]
                for nid in subsec_node_ids
            ]
            sec_summary, sec_tokens = _summarize_group(
                sec_texts, f"{doc_name} / {sec_title}", llm
            )
            total_tokens += sec_tokens
            total_llm_calls += 1

            sec_id = CustomTreeIndex.make_node_id()
            all_nodes[sec_id] = TreeNode(
                node_id=sec_id,
                node_type="section",
                text=sec_summary or "\n\n".join(sec_texts)[:600],
                summary=sec_summary or "\n\n".join(sec_texts)[:300],
                metadata={"doc_name": doc_name, "section": sec_title},
                children=subsec_node_ids,
            )
            section_node_ids.append(sec_id)

        doc_texts = [
            all_nodes[nid].summary or all_nodes[nid].text[:300]
            for nid in section_node_ids
        ]
        doc_summary, doc_tokens = _summarize_group(doc_texts, doc_name, llm)
        total_tokens += doc_tokens
        total_llm_calls += 1

        doc_id = CustomTreeIndex.make_node_id()
        all_nodes[doc_id] = TreeNode(
            node_id=doc_id,
            node_type="document",
            text=doc_summary or "\n\n".join(doc_texts)[:600],
            summary=doc_summary or "\n\n".join(doc_texts)[:300],
            metadata={"doc_name": doc_name},
            children=section_node_ids,
        )
        root_ids.append(doc_id)

    tree = CustomTreeIndex(root_ids=root_ids, nodes=all_nodes)
    build_stats = {
        "tokens_used": total_tokens,
        "llm_calls": total_llm_calls,
        "node_count": len(all_nodes),
    }
    logger.info(
        "Tree built: %d nodes, %d LLM calls, %d tokens",
        len(all_nodes), total_llm_calls, total_tokens,
    )
    return tree, build_stats


def build_index(chunks: List[Dict[str, Any]]) -> Tuple[CustomTreeIndex, Dict[str, Any]]:
    global _index_cache
    settings.ensure_dirs()
    logger.info("Building custom tree from %d chunks ...", len(chunks))
    tree, build_stats = _build_hierarchical_tree(chunks)
    tree_path = settings.index_store_dir / _TREE_FILE
    tree.save(tree_path)
    logger.info("Tree persisted to: %s", tree_path)
    _index_cache = tree
    return tree, build_stats


def load_index() -> Optional[CustomTreeIndex]:
    global _index_cache
    if _index_cache is not None:
        return _index_cache
    tree_path = settings.index_store_dir / _TREE_FILE
    if not tree_path.exists():
        logger.info("No persisted tree found at: %s", tree_path)
        return None
    try:
        tree = CustomTreeIndex.load(tree_path)
        _index_cache = tree
        logger.info("Custom tree loaded: %d nodes.", len(tree.nodes))
        return tree
    except Exception as exc:
        logger.error("Failed to load tree: %s", exc)
        return None


def get_or_build_index(chunks: List[Dict[str, Any]]) -> CustomTreeIndex:
    existing = load_index()
    if existing is not None:
        return existing
    tree, _ = build_index(chunks)
    return tree


def get_index_node_count() -> Optional[int]:
    tree = load_index()
    return len(tree.nodes) if tree is not None else None


def invalidate_cache() -> None:
    global _index_cache
    _index_cache = None
