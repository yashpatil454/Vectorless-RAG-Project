from __future__ import annotations

from datetime import datetime
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Upload / Ingestion
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    filename: str
    chunk_count: int
    upload_timestamp: str
    message: str


class IndexBuildResponse(BaseModel):
    status: Literal["built", "loaded", "already_exists"]
    node_count: int
    message: str


class IndexStatusResponse(BaseModel):
    index_exists: bool
    node_count: Optional[int] = None


# ---------------------------------------------------------------------------
# Query / Reasoning
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    document_filter: Optional[str] = Field(
        default=None,
        description="Optional document name to restrict retrieval to.",
    )
    tags_filter: Optional[List[str]] = Field(
        default=None,
        description="Optional list of tags to restrict retrieval to.",
    )


class EvidenceNode(BaseModel):
    node_id: str
    summary: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalStep(BaseModel):
    step: int
    action: Literal["metadata_filter", "tree_traversal", "re_retrieve", "sufficiency_check"]
    notes: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    status: Literal["success", "insufficient_context", "error"]
    evidence: List[EvidenceNode] = Field(default_factory=list)
    retrieval_trace: List[RetrievalStep] = Field(default_factory=list)
    tokens_used: int = 0
    llm_calls: int = 0


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

class HistoryEntry(BaseModel):
    id: str
    query: str
    answer: str
    status: Literal["success", "insufficient_context", "error"]
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    evidence_count: int = 0
    tokens_used: int = 0
    llm_calls: int = 0


class HistoryResponse(BaseModel):
    entries: List[HistoryEntry]
    total: int
