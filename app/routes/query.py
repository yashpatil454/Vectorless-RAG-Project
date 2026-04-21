from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.schemas import QueryRequest, QueryResponse
from app.services import history_service, reasoning_service
from utils.logging_utils import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """Run multi-hop reasoning over the tree index and return a structured answer."""
    logger.info("Received query: '%s'", request.query)

    try:
        response = reasoning_service.run_reasoning(
            query=request.query,
            document_filter=request.document_filter,
            tags_filter=request.tags_filter,
        )
    except Exception as exc:
        logger.error("Reasoning failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Reasoning error: {exc}") from exc

    # Persist to history (non-blocking — failure is logged but not surfaced)
    try:
        history_service.add_entry(response)
    except Exception as exc:
        logger.warning("Failed to record history entry: %s", exc)

    return response
