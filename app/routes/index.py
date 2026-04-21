from __future__ import annotations

import json
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.models.schemas import IndexBuildResponse, IndexStatusResponse
from app.services.ingestion_service import list_ingested_documents, load_parsed_chunks
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
        index = build_index(chunks)
        node_count = len(index.index_struct.all_nodes)
    except Exception as exc:
        logger.error("Index build failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Index build failed: {exc}") from exc

    return IndexBuildResponse(
        status="built",
        node_count=node_count,
        message=f"TreeIndex built with {node_count} nodes.",
    )


@router.post("/refresh", response_model=IndexBuildResponse)
async def refresh_index_endpoint() -> IndexBuildResponse:
    """Force-rebuild the TreeIndex from all ingested documents."""
    invalidate_cache()

    chunks = _collect_all_chunks()
    if not chunks:
        raise HTTPException(status_code=422, detail="No ingested documents found. Upload a PDF first.")

    try:
        index = build_index(chunks)
        node_count = len(index.index_struct.all_nodes)
    except Exception as exc:
        logger.error("Index refresh failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Index refresh failed: {exc}") from exc

    return IndexBuildResponse(
        status="built",
        node_count=node_count,
        message=f"TreeIndex rebuilt with {node_count} nodes.",
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
    """Return nodes and edges for tree visualization.

    Reads directly from the persisted index_store.json and docstore.json so
    the index does not need to be loaded into memory.
    """
    index_store_path = settings.index_store_dir / "index_store.json"
    docstore_path = settings.index_store_dir / "docstore.json"

    if not index_store_path.exists():
        raise HTTPException(status_code=404, detail="No index found. Build an index first.")

    index_raw = json.loads(index_store_path.read_text(encoding="utf-8"))
    tree_entry = list(index_raw["index_store/data"].values())[0]["__data__"]
    tree_data: dict = tree_entry if isinstance(tree_entry, dict) else json.loads(tree_entry)

    edges_map: dict = tree_data["node_id_to_children_ids"]
    root_ids: set = set(tree_data["root_nodes"].values())

    node_info: dict = {}
    if docstore_path.exists():
        docstore_raw = json.loads(docstore_path.read_text(encoding="utf-8"))
        for nid, entry in docstore_raw.get("docstore/data", {}).items():
            data = entry["__data__"]
            if isinstance(data, str):
                data = json.loads(data)
            meta = data.get("metadata", {})
            node_info[nid] = {
                "text": data.get("text", "")[:200],
                "page_number": meta.get("page_number", "?"),
                "doc_name": meta.get("doc_name", ""),
            }

    nodes = [
        {
            "id": nid,
            "is_root": nid in root_ids,
            **node_info.get(nid, {"text": "", "page_number": "?", "doc_name": ""}),
        }
        for nid in edges_map
    ]

    edge_list = [
        {"from": parent, "to": child}
        for parent, children in edges_map.items()
        for child in children
    ]

    return {"nodes": nodes, "edges": edge_list, "root_count": len(root_ids)}
