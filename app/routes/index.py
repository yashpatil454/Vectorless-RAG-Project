from __future__ import annotations

import json
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.models.schemas import IndexBuildResponse, IndexStatusResponse
from app.services.ingestion_service import ingest_document, list_ingested_documents, load_parsed_chunks
from app.services.tree_index_service import (
    build_index,
    get_index_node_count,
    invalidate_cache,
    load_index,
)
from utils.logging_utils import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/index", tags=["index"])


def _collect_all_chunks() -> list:
    """Gather parsed chunks from all ingested documents."""
    all_chunks = []
    for doc_stem in list_ingested_documents():
        parsed_path = settings.parsed_dir / f"{doc_stem}.json"
        if parsed_path.exists():
            all_chunks.extend(json.loads(parsed_path.read_text(encoding="utf-8")))
    return all_chunks


@router.post("/build", response_model=IndexBuildResponse)
async def build_index_endpoint() -> IndexBuildResponse:
    """Build the TreeIndex from all currently ingested documents.

    If an index already exists on disk, returns a ``already_exists`` status.
    Use ``POST /index/refresh`` to force a rebuild.
    """
    if load_index() is not None:
        node_count = get_index_node_count() or 0
        return IndexBuildResponse(
            status="already_exists",
            node_count=node_count,
            message="Index already exists. Use /index/refresh to rebuild.",
        )

    chunks = _collect_all_chunks()
    if not chunks:
        raise HTTPException(status_code=422, detail="No ingested documents found. Upload a PDF first.")

    try:
        index, build_stats = build_index(chunks)
        node_count = build_stats["node_count"]
    except Exception as exc:
        logger.error("Index build failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Index build failed: {exc}") from exc

    return IndexBuildResponse(
        status="built",
        node_count=node_count,
        message=f"TreeIndex built with {node_count} nodes.",
        tokens_used=build_stats["tokens_used"],
        llm_calls=build_stats["llm_calls"],
    )


@router.post("/refresh", response_model=IndexBuildResponse)
async def refresh_index_endpoint() -> IndexBuildResponse:
    """Force-rebuild the TreeIndex, re-parsing all uploads first.

    Re-running ingestion on each uploaded PDF ensures that the parsed JSON
    files reflect the current section-aware chunking and summary logic before
    the tree index is rebuilt.
    """
    invalidate_cache()

    # Re-parse every existing upload so parsed JSONs pick up section structure + summaries
    upload_dir = settings.upload_dir
    pdf_files = list(upload_dir.glob("*.pdf")) if upload_dir.exists() else []
    if pdf_files:
        logger.info("Re-parsing %d uploaded PDF(s) before rebuild …", len(pdf_files))
        for pdf_path in pdf_files:
            try:
                file_bytes = pdf_path.read_bytes()
                ingest_document(file_bytes=file_bytes, filename=pdf_path.name)
                logger.info("Re-parsed: %s", pdf_path.name)
            except Exception as exc:
                logger.warning("Re-parse failed for '%s': %s — skipping.", pdf_path.name, exc)

    chunks = _collect_all_chunks()
    if not chunks:
        raise HTTPException(status_code=422, detail="No ingested documents found. Upload a PDF first.")

    try:
        index, build_stats = build_index(chunks)
        node_count = build_stats["node_count"]
    except Exception as exc:
        logger.error("Index refresh failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Index refresh failed: {exc}") from exc

    return IndexBuildResponse(
        status="built",
        node_count=node_count,
        message=f"TreeIndex rebuilt with {node_count} nodes.",
        tokens_used=build_stats["tokens_used"],
        llm_calls=build_stats["llm_calls"],
    )


@router.get("/status", response_model=IndexStatusResponse)
async def index_status_endpoint() -> IndexStatusResponse:
    """Return whether a persisted index exists and how many nodes it has."""
    index = load_index()
    if index is None:
        return IndexStatusResponse(index_exists=False)

    node_count = get_index_node_count()
    return IndexStatusResponse(index_exists=True, node_count=node_count)


@router.get("/tree")
async def tree_data_endpoint() -> dict:
    """Return nodes and edges for tree visualization, read from tree.json."""
    from utils.tree_model import CustomTreeIndex

    tree_path = settings.index_store_dir / "tree.json"
    if not tree_path.exists():
        raise HTTPException(status_code=404, detail="No index found. Build an index first.")

    try:
        tree = CustomTreeIndex.load(tree_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load tree: {exc}") from exc

    root_ids = set(tree.root_ids)
    nodes = []
    edges = []

    for nid, node in tree.nodes.items():
        nodes.append({
            "id": nid,
            "is_root": nid in root_ids,
            "node_type": node.node_type,
            "text": node.text[:200],
            "summary": node.summary[:200] if node.summary else "",
            "section": node.metadata.get("section", ""),
            "subsection": node.metadata.get("subsection", ""),
            "doc_name": node.metadata.get("doc_name", ""),
            "page_number": node.metadata.get("page_number", "?"),
            "page_range": node.metadata.get("page_range", [node.metadata.get("page_number", "?")]),
        })
        for child_id in node.children:
            edges.append({"from": nid, "to": child_id})

    return {"nodes": nodes, "edges": edges, "root_count": len(root_ids)}
