from __future__ import annotations

from app.schemas import ExecutionState, ExecutionStep, EvidenceBundle, RetrievalPlan, UserIntent
from app.services.rag_v3.distiller import distill_evidence
from app.services.rag_v3.planner import replan_after_retrieval
from app.services.rag_v3.retrievers import merge_bundles, retrieve_with_tool


class V3ExecutionError(RuntimeError):
    def __init__(self, stage: str, state: ExecutionState, original_error: Exception):
        super().__init__(str(original_error))
        self.stage = stage
        self.state = state
        self.original_error = original_error


def execute_plan(question: str, intent: UserIntent, plan: RetrievalPlan, limit: int, trace_id: str) -> tuple[EvidenceBundle, ExecutionState]:
    state = ExecutionState(plan=plan, trace_id=trace_id)
    bundles: list[EvidenceBundle] = []
    current_plan = plan
    last_retrieval_error: Exception | None = None
    for tool in current_plan.retrieval_tools:
        try:
            bundle = retrieve_with_tool(question, current_plan, tool, limit)
            bundles.append(bundle)
            state.applied_tools.extend(bundle.applied_tools)
            state.steps.append(
                ExecutionStep(
                    step_type="retrieve",
                    tool=tool,
                    status="completed",
                    detail=f"Retrieved {len(bundle.items)} items with confidence {bundle.retrieval_confidence}.",
                )
            )
        except Exception as exc:
            state.steps.append(ExecutionStep(step_type="retrieve", tool=tool, status="failed", detail=str(exc)))
            last_retrieval_error = exc
            continue
    if not bundles and last_retrieval_error is not None:
        raise V3ExecutionError("retrieve", state, last_retrieval_error) from last_retrieval_error
    merged = merge_bundles(question, current_plan, bundles)
    if merged.retrieval_confidence < 0.45 or len(merged.items) < max(1, min(2, limit)):
        replanned = replan_after_retrieval(intent, current_plan, len(merged.items), merged.retrieval_confidence)
        if replanned.retrieval_tools != current_plan.retrieval_tools:
            state.steps.append(ExecutionStep(step_type="replan", tool="planner", status="completed", detail="Expanded retrieval plan after weak evidence."))
            current_plan = replanned
            state.plan = replanned
            replan_error: Exception | None = None
            for tool in replanned.retrieval_tools:
                if tool in merged.applied_tools:
                    continue
                try:
                    bundle = retrieve_with_tool(question, replanned, tool, limit)
                    bundles.append(bundle)
                    state.applied_tools.extend(bundle.applied_tools)
                    state.applied_fallbacks.extend(bundle.applied_tools)
                    state.steps.append(
                        ExecutionStep(
                            step_type="retrieve",
                            tool=tool,
                            status="completed",
                            detail=f"Fallback retrieved {len(bundle.items)} items with confidence {bundle.retrieval_confidence}.",
                        )
                    )
                except Exception as exc:
                    state.steps.append(ExecutionStep(step_type="retrieve", tool=tool, status="failed", detail=str(exc)))
                    replan_error = exc
                    continue
            if not bundles and replan_error is not None:
                raise V3ExecutionError("retrieve", state, replan_error) from replan_error
            merged = merge_bundles(question, replanned, bundles)
    try:
        distilled = distill_evidence(intent, merged)
        state.distilled_summary = distilled.summary
        state.steps.append(ExecutionStep(step_type="distill", tool="distiller", status="completed", detail=f"Distilled {len(merged.items)} evidence items."))
    except Exception as exc:
        state.steps.append(ExecutionStep(step_type="distill", tool="distiller", status="failed", detail=str(exc)))
        raise V3ExecutionError("distill", state, exc) from exc
    return merged, state
