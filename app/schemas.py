from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl


class FeedIngestRequest(BaseModel):
    feed_url: HttpUrl | None = None
    feed_file: str | None = None
    org_id: str | None = None
    process_embeddings: bool = False
    resume_from_article_id: int | None = None


class EmbeddingBackfillRequest(BaseModel):
    start_article_id: int | None = None
    limit: int | None = Field(default=None, ge=1, le=5000)
    worker_count: int | None = Field(default=None, ge=1, le=16)
    failed_only: bool = False


class EmbeddingBackfillResponse(BaseModel):
    ok: bool
    requested: int
    embedded: int
    skipped_current: int
    skipped_not_searchable: int
    failed: int
    start_article_id: int | None = None
    next_article_id: int | None = None


class EmbeddingStatusResponse(BaseModel):
    counts: dict[str, int]
    first_failed_article_id: int | None = None
    first_pending_article_id: int | None = None


class QueryRequest(BaseModel):
    query: str = Field(min_length=3)
    issue_date: str | None = None
    limit: int = Field(default=10, ge=1, le=50)


class LoginRequest(BaseModel):
    email: str = Field(min_length=5)
    password: str = Field(min_length=6)


class ChatRequest(BaseModel):
    question: str = Field(min_length=3)
    issue_date: str | None = None
    session_filters: dict[str, Any] | None = None
    history: list[dict[str, str]] | None = None
    limit: int = Field(default=6, ge=1, le=20)


class RoutedQuery(BaseModel):
    mode: Literal["sql", "semantic", "hybrid"]
    intent: Literal["lookup", "article_count", "topic_count", "fact_lookup", "author_lookup", "author_count"] = "lookup"
    issue_date: str | None = None
    edition: str | None = None
    section: str | None = None
    author: str | None = None
    semantic_query: str | None = None


class QueryResponse(BaseModel):
    mode: Literal["sql", "semantic", "hybrid"]
    filters: dict[str, Any]
    results: list[dict[str, Any]]


class ChatResponse(BaseModel):
    answer: str
    mode: Literal["sql", "semantic", "hybrid"]
    citations: list[dict[str, Any]]
    session_context: dict[str, Any] | None = None
    debug_trace: dict[str, Any] | None = None


class AuthStatusResponse(BaseModel):
    authenticated: bool
    email: str | None = None
