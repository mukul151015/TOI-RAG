from __future__ import annotations

from app.schemas import DistilledEvidence, EvidenceBundle, TraceEnvelope, VerificationReport
from app.services.rag_v3.common import make_trace_id


def create_trace(question: str) -> TraceEnvelope:
    return TraceEnvelope(trace_id=make_trace_id(), question=question)


def attach_planner(trace: TraceEnvelope, *, standalone_question: str, planner: dict, retrieval_plan: dict) -> None:
    trace.standalone_question = standalone_question
    trace.planner = planner
    trace.retrieval_plan = retrieval_plan


def attach_execution(trace: TraceEnvelope, *, execution_steps: list[dict], bundle: EvidenceBundle | None) -> None:
    trace.execution_steps = execution_steps
    if bundle is None:
        return
    trace.evidence_summary = {
        "mode": bundle.mode,
        "item_count": len(bundle.items),
        "retrieval_confidence": bundle.retrieval_confidence,
        "applied_tools": bundle.applied_tools,
        "applied_fallbacks": bundle.applied_fallbacks,
        "sample_headlines": [item.headline for item in bundle.items[:5]],
        "raw_filters": bundle.raw_filters,
        "notes": bundle.notes,
    }


def attach_distilled(trace: TraceEnvelope, distilled: DistilledEvidence) -> None:
    trace.distilled_evidence = distilled.model_dump()


def attach_verification(trace: TraceEnvelope, verification: VerificationReport) -> None:
    trace.verification = verification.model_dump()


def attach_failure(trace: TraceEnvelope, *, stage: str, error: Exception) -> None:
    trace.failure = {
        "stage": stage,
        "error_type": type(error).__name__,
        "message": str(error),
    }


def attach_comparison(trace: TraceEnvelope, comparison: dict) -> None:
    trace.comparison = comparison
