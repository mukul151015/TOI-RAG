from __future__ import annotations

from app.schemas import ChatResponse, DistilledEvidence, QueryResponse, TraceEnvelope
from app.services.rag_v3.answer_generator import generate_answer
from app.services.rag_v3.distiller import distill_evidence
from app.services.rag_v3.executor import V3ExecutionError, execute_plan
from app.services.rag_v3.planner import parse_user_intent
from app.services.rag_v3.telemetry import (
    attach_distilled,
    attach_execution,
    attach_failure,
    attach_planner,
    attach_verification,
    create_trace,
)
from app.services.rag_v3.verifier import verify_answer


def execute_query(question: str, issue_date: str | None, limit: int) -> QueryResponse:
    trace = create_trace(question)
    intent, plan = parse_user_intent(question, issue_date)
    attach_planner(trace, standalone_question=intent.standalone_question, planner=intent.model_dump(), retrieval_plan=plan.model_dump())
    try:
        bundle, state = execute_plan(intent.standalone_question, intent, plan, limit, trace.trace_id)
        attach_execution(trace, execution_steps=[step.model_dump() for step in state.steps], bundle=bundle)
    except V3ExecutionError as exc:
        attach_execution(trace, execution_steps=[step.model_dump() for step in exc.state.steps], bundle=None)
        attach_failure(trace, stage=exc.stage, error=exc.original_error)
        return QueryResponse(
            mode=plan.mode,
            filters=intent.filters,
            results=[],
            confidence_score=0.0,
            retrieval_plan=plan,
        )
    return QueryResponse(
        mode=bundle.mode,
        filters=bundle.raw_filters,
        results=[item.metadata for item in bundle.items],
        confidence_score=bundle.retrieval_confidence,
        retrieval_plan=state.plan,
    )


def answer_question(
    question: str,
    issue_date: str | None,
    limit: int,
    session_filters: dict | None = None,
    history: list[dict[str, str]] | None = None,
    session_context: dict | None = None,
) -> ChatResponse:
    trace = create_trace(question)
    intent, plan = parse_user_intent(question, issue_date, history=history, session_context=session_context)
    if session_filters:
        plan = plan.model_copy(update={
            "edition": session_filters.get("edition") or plan.edition,
            "section": session_filters.get("section") or plan.section,
        })
        intent.filters.update({"edition": plan.edition, "section": plan.section})
    attach_planner(trace, standalone_question=intent.standalone_question, planner=intent.model_dump(), retrieval_plan=plan.model_dump())

    if "Unsupported or prompt-injection markers detected." in (intent.reasoning or ""):
        attach_failure(trace, stage="planning", error=ValueError("Unsupported or prompt-injection request."))
        return ChatResponse(
            answer="I can only answer questions grounded in the dataset, and this request looks like an instruction override rather than a news query.",
            mode=plan.mode,
            citations=[],
            confidence_score=0.0,
            session_context=_build_session_context(question, plan, intent.filters),
            debug_trace=trace.model_dump(),
        )

    try:
        bundle, state = execute_plan(intent.standalone_question, intent, plan, limit, trace.trace_id)
        attach_execution(trace, execution_steps=[step.model_dump() for step in state.steps], bundle=bundle)
        distilled = distill_evidence(intent, bundle)
        attach_distilled(trace, distilled)
        draft = generate_answer(intent, bundle, distilled)
        verification = verify_answer(intent, bundle, draft)
        attach_verification(trace, verification)
        answer = draft.answer if verification.answer_accepted else "I couldn't verify a grounded answer from the current evidence."
    except V3ExecutionError as exc:
        attach_execution(trace, execution_steps=[step.model_dump() for step in exc.state.steps], bundle=None)
        attach_failure(trace, stage=exc.stage, error=exc.original_error)
        return ChatResponse(
            answer="I couldn't complete a grounded answer because the retrieval pipeline failed before verification.",
            mode=plan.mode,
            citations=[],
            confidence_score=0.0,
            session_context=_build_session_context(question, plan, intent.filters),
            debug_trace=trace.model_dump(),
        )
    except Exception as exc:
        attach_failure(trace, stage="unexpected", error=exc)
        return ChatResponse(
            answer="I couldn't complete a grounded answer because the pipeline failed unexpectedly.",
            mode=plan.mode,
            citations=[],
            confidence_score=0.0,
            session_context=_build_session_context(question, plan, intent.filters),
            debug_trace=trace.model_dump(),
        )

    return ChatResponse(
        answer=answer,
        mode=draft.mode,
        citations=draft.citations,
        confidence_score=bundle.retrieval_confidence,
        session_context=_build_session_context(question, state.plan, bundle.raw_filters),
        debug_trace=trace.model_dump(),
        verification=verification,
    )


def _build_session_context(question: str, plan, filters: dict) -> dict:
    return {
        "last_question": question,
        "last_mode": plan.mode,
        "last_intent": plan.intent,
        "last_answer_shape": getattr(plan, "answer_shape", None),
        "last_filters": filters,
        "query_focus": plan.semantic_query,
        "last_topic": " ".join(plan.entity_terms[:3]) if plan.entity_terms else plan.semantic_query,
        "last_entity_terms": list(getattr(plan, "entity_terms", []) or []),
        "last_issue_date": plan.issue_date,
        "last_start_date": getattr(plan, "start_date", None),
        "last_end_date": getattr(plan, "end_date", None),
        "last_time_scope": getattr(plan, "time_scope", None),
    }
