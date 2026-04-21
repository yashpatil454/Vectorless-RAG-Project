from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes import history, index, query, upload
from utils.logging_utils import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Tree-RAG API",
    description="Tree-based document QA API using LlamaIndex TreeIndex + LangGraph multi-hop reasoning.",
    version="1.0.0",
)

# Allow the Streamlit frontend (running on any origin in development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all routers
app.include_router(upload.router)
app.include_router(index.router)
app.include_router(query.router)
app.include_router(history.router)


@app.on_event("startup")
async def startup_event() -> None:
    """Ensure all required data directories exist and attempt to load a cached index."""
    settings.ensure_dirs()
    logger.info("Tree-RAG API starting up.")

    # Attempt to warm up the index cache so the first query is fast.
    from app.services.tree_index_service import load_index

    index = load_index()
    if index is not None:
        logger.info("Existing index loaded into cache at startup.")
    else:
        logger.info("No persisted index found at startup. Upload a document and call POST /index/build.")


@app.get("/health", tags=["health"])
async def health() -> dict:
    return {"status": "ok"}
