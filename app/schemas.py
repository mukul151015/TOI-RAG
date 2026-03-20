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


class UserIntent(BaseModel):
    original_question: str
    standalone_question: str
    intent: str
    mode: Literal["sql", "semantic", "hybrid"]
    needs_count: bool = False
    needs_summary: bool = False
    needs_listing: bool = False
    needs_article_text: bool = False
    entities: dict[str, list[str]] = Field(default_factory=dict)
    filters: dict[str, Any] = Field(default_factory=dict)
    ambiguity_note: str | None = None
    reasoning: str | None = None


class RetrievalPlan(BaseModel):
    mode: Literal["sql", "semantic", "hybrid"]
    intent: str
    task_type: str | None = None
    answer_shape: str | None = None
    time_scope: str | None = None
    issue_date: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    edition: str | None = None
    section: str | None = None
    author: str | None = None
    semantic_query: str | None = None
    entity_terms: list[str] = Field(default_factory=list)
    retrieval_tools: list[str] = Field(default_factory=list)
    fallback_order: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    reasoning: str | None = None
    notes: list[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    article_id: str | None = None
    headline: str | None = None
    edition: str | None = None
    section: str | None = None
    issue_date: str | None = None
    excerpt: str | None = None
    score: float = 0.0
    source_type: str = "article"
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceBundle(BaseModel):
    question: str
    mode: Literal["sql", "semantic", "hybrid"]
    plan: RetrievalPlan
    items: list[EvidenceItem] = Field(default_factory=list)
    raw_filters: dict[str, Any] = Field(default_factory=dict)
    retrieval_confidence: float = 0.0
    applied_tools: list[str] = Field(default_factory=list)
    applied_fallbacks: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class DistilledEvidence(BaseModel):
    summary: str = ""
    key_points: list[str] = Field(default_factory=list)
    supporting_article_ids: list[str] = Field(default_factory=list)
    coverage: str | None = None
    notes: list[str] = Field(default_factory=list)


class AnswerDraft(BaseModel):
    answer: str
    mode: Literal["sql", "semantic", "hybrid"]
    citations: list[dict[str, Any]] = Field(default_factory=list)
    grounded: bool = True
    notes: list[str] = Field(default_factory=list)


class VerificationReport(BaseModel):
    grounded: bool
    supported_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    rationale: str | None = None
    answer_accepted: bool = True


class ExecutionStep(BaseModel):
    step_type: str
    tool: str
    status: str
    detail: str | None = None


class ExecutionState(BaseModel):
    plan: RetrievalPlan
    steps: list[ExecutionStep] = Field(default_factory=list)
    applied_tools: list[str] = Field(default_factory=list)
    applied_fallbacks: list[str] = Field(default_factory=list)
    distilled_summary: str | None = None
    trace_id: str | None = None


class EvalCase(BaseModel):
    case_id: str
    category: str
    question: str
    issue_date: str | None = None
    history: list[dict[str, str]] = Field(default_factory=list)
    session_context: dict[str, Any] = Field(default_factory=dict)
    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)
    expected_intent: str | None = None
    expected_tools: list[str] = Field(default_factory=list)
    expected_mode: str | None = None
    expect_abstention: bool = False


class EvalResult(BaseModel):
    case_id: str
    category: str
    passed: bool
    score: float
    notes: list[str] = Field(default_factory=list)


class TraceEnvelope(BaseModel):
    trace_id: str
    question: str
    standalone_question: str | None = None
    planner: dict[str, Any] = Field(default_factory=dict)
    retrieval_plan: dict[str, Any] = Field(default_factory=dict)
    execution_steps: list[dict[str, Any]] = Field(default_factory=list)
    evidence_summary: dict[str, Any] = Field(default_factory=dict)
    distilled_evidence: dict[str, Any] = Field(default_factory=dict)
    verification: dict[str, Any] = Field(default_factory=dict)
    comparison: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None


class QueryResponse(BaseModel):
    mode: Literal["sql", "semantic", "hybrid"]
    filters: dict[str, Any]
    results: list[dict[str, Any]]
    confidence_score: float = 0.0
    retrieval_plan: RetrievalPlan | None = None


class ChatResponse(BaseModel):
    answer: str
    mode: Literal["sql", "semantic", "hybrid"]
    citations: list[dict[str, Any]]
    confidence_score: float = 0.0
    session_context: dict[str, Any] | None = None
    debug_trace: dict[str, Any] | None = None
    verification: VerificationReport | None = None


class AuthStatusResponse(BaseModel):
    authenticated: bool
    email: str | None = None
