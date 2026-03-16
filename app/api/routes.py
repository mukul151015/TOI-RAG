from pathlib import Path

from fastapi import APIRouter, Request, Response
from fastapi.responses import FileResponse

from app.schemas import (
    AuthStatusResponse,
    ChatRequest,
    EmbeddingBackfillRequest,
    FeedIngestRequest,
    LoginRequest,
    QueryRequest,
)
from app.services.auth_service import (
    fetch_recent_chat_traces,
    get_authenticated_session,
    get_authenticated_user,
    login_or_create,
    log_chat_interaction,
    logout,
    require_authenticated_user,
    update_session_context,
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


@router.get("/auth/status", response_model=AuthStatusResponse)
def auth_status_route(request: Request):
    user = get_authenticated_user(request)
    if not user:
        return AuthStatusResponse(authenticated=False)
    return AuthStatusResponse(authenticated=True, email=user["email"])


@router.post("/auth/login")
def auth_login_route(payload: LoginRequest, response: Response):
    user = login_or_create(payload.email, payload.password, response)
    return {"ok": True, "email": user["email"]}


@router.post("/auth/logout")
def auth_logout_route(request: Request, response: Response):
    logout(response, request.cookies.get("toi_rag_session"))
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


@router.post("/query")
def query_route(payload: QueryRequest, request: Request):
    require_authenticated_user(request)
    return run_query(payload.query, payload.issue_date, payload.limit)


@router.post("/chat")
def chat_route(payload: ChatRequest, request: Request):
    session = get_authenticated_session(request)
    if not session:
        require_authenticated_user(request)
    result = answer_question(
        payload.question,
        payload.issue_date,
        payload.limit,
        payload.session_filters,
        payload.history,
        session.get("session_context") if session else None,
    )
    if session:
        update_session_context(session["session_id"], result.session_context)
        log_chat_interaction(
            user_id=session["user_id"],
            session_id=session["session_id"],
            question=payload.question,
            answer=result.answer,
            issue_date=payload.issue_date,
            mode=result.mode,
            session_filters=payload.session_filters,
            citations=result.citations,
            trace_data=result.debug_trace,
        )
    return result


@router.get("/catalog/editions")
def editions_catalog_route(request: Request):
    require_authenticated_user(request)
    return [row["publication_name"] for row in fetch_publication_catalog()]


@router.get("/catalog/sections")
def sections_catalog_route(request: Request):
    require_authenticated_user(request)
    return fetch_section_catalog()


@router.get("/debug/chat-traces")
def chat_traces_route(request: Request, limit: int = 20):
    user = require_authenticated_user(request)
    safe_limit = max(1, min(limit, 100))
    return fetch_recent_chat_traces(user_id=user["id"], limit=safe_limit)
