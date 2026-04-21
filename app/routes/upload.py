from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.models.schemas import UploadResponse
from app.services.ingestion_service import ingest_document
from utils.logging_utils import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/upload", tags=["upload"])


@router.post("", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    tags: Optional[str] = Form(
        default=None,
        description="Comma-separated list of tags, e.g. 'finance,2024'",
    ),
) -> UploadResponse:
    """Accept a PDF upload, parse it, and persist the results to disk."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are supported.")

    tag_list: List[str] = []
    if tags:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    try:
        file_bytes = await file.read()
        result = ingest_document(
            file_bytes=file_bytes,
            filename=file.filename,
            tags=tag_list,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Unexpected error during upload: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error during ingestion.") from exc

    return UploadResponse(
        filename=result["filename"],
        chunk_count=result["chunk_count"],
        upload_timestamp=result["upload_timestamp"],
        message=f"Successfully ingested '{result['filename']}' ({result['chunk_count']} pages).",
    )
