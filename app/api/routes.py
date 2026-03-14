from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.schemas import (
    ChatRequest,
    EmbeddingBackfillRequest,
    FeedIngestRequest,
    QueryRequest,
)
from app.services.chat_service import answer_question
from app.services.embedding_backfill import backfill_embeddings, get_embedding_status
from app.services.ingestion import ingest_feed
from app.services.query_service import run_query
from app.services.repository import fetch_publication_catalog, fetch_section_catalog


router = APIRouter()
UI_PATH = Path(__file__).resolve().parents[1] / "ui" / "index.html"


@router.get("/", include_in_schema=False)
def ui_home():
    return FileResponse(UI_PATH)


@router.get("/health")
def healthcheck():
    return {"ok": True}


@router.post("/ingest/feed")
async def ingest_feed_route(payload: FeedIngestRequest):
    return await ingest_feed(
        feed_url=str(payload.feed_url) if payload.feed_url else None,
        feed_file=payload.feed_file,
        org_id=payload.org_id,
        process_embeddings=payload.process_embeddings,
        resume_from_article_id=payload.resume_from_article_id,
    )


@router.post("/embeddings/backfill")
def embeddings_backfill_route(payload: EmbeddingBackfillRequest):
    return backfill_embeddings(
        start_article_id=payload.start_article_id,
        limit=payload.limit,
        worker_count=payload.worker_count,
        failed_only=payload.failed_only,
    )


@router.get("/embeddings/status")
def embeddings_status_route():
    return get_embedding_status()


@router.post("/query")
def query_route(payload: QueryRequest):
    return run_query(payload.query, payload.issue_date, payload.limit)


@router.post("/chat")
def chat_route(payload: ChatRequest):
    return answer_question(
        payload.question,
        payload.issue_date,
        payload.limit,
        payload.session_filters,
        payload.history,
    )


@router.get("/catalog/editions")
def editions_catalog_route():
    return [row["publication_name"] for row in fetch_publication_catalog()]


@router.get("/catalog/sections")
def sections_catalog_route():
    return fetch_section_catalog()
