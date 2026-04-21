from __future__ import annotations

from fastapi import APIRouter

from app.models.schemas import HistoryResponse
from app.services import history_service
from utils.logging_utils import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/history", tags=["history"])


@router.get("", response_model=HistoryResponse)
async def get_history(limit: int = 50) -> HistoryResponse:
    """Return the most recent query history entries."""
    entries = history_service.get_history(limit=limit)
    return HistoryResponse(entries=entries, total=len(entries))


@router.delete("", response_model=dict)
async def clear_history() -> dict:
    """Delete all stored query history."""
    count = history_service.clear_history()
    logger.info("Cleared %d history entries.", count)
    return {"deleted": count, "message": f"Cleared {count} history entries."}
