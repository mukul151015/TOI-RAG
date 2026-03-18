from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from app.schemas import (
    EmbeddingBackfillRequest,
    FeedIngestRequest,
)
from app.services.auth_service import (
    fetch_recent_chat_traces,
    require_authenticated_user,
)
from app.services.embedding_backfill import backfill_embeddings, get_embedding_status
from app.services.ingestion import ingest_feed


router = APIRouter()
UI_PATH = Path(__file__).resolve().parents[1] / "ui" / "index.html"


@router.get("/", include_in_schema=False)
def ui_home():
    return FileResponse(UI_PATH)


@router.get("/health")
def healthcheck():
    return {"ok": True}


@router.post("/ingest/feed")
async def ingest_feed_route(payload: FeedIngestRequest, request: Request):
    require_authenticated_user(request)
    return await ingest_feed(
        feed_url=str(payload.feed_url) if payload.feed_url else None,
        feed_file=payload.feed_file,
        org_id=payload.org_id,
        process_embeddings=payload.process_embeddings,
        resume_from_article_id=payload.resume_from_article_id,
    )


@router.post("/embeddings/backfill")
def embeddings_backfill_route(payload: EmbeddingBackfillRequest, request: Request):
    require_authenticated_user(request)
    return backfill_embeddings(
        start_article_id=payload.start_article_id,
        limit=payload.limit,
        worker_count=payload.worker_count,
        failed_only=payload.failed_only,
    )


@router.get("/embeddings/status")
def embeddings_status_route(request: Request):
    require_authenticated_user(request)
    return get_embedding_status()


@router.get("/debug/chat-traces")
def chat_traces_route(request: Request, limit: int = 20):
    user = require_authenticated_user(request)
    safe_limit = max(1, min(limit, 100))
    return fetch_recent_chat_traces(user_id=user["id"], limit=safe_limit)
