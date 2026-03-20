import json
from pathlib import Path

from app.schemas import EvalCase, EvalResult
from app.services.rag_v3.pipeline import answer_question


def load_eval_cases(path: str | Path) -> list[EvalCase]:
    payload = json.loads(Path(path).read_text())
    return [EvalCase(**item) for item in payload]


def run_eval_cases(cases: list[EvalCase]) -> list[EvalResult]:
    results: list[EvalResult] = []
    for case in cases:
        results.append(run_eval_case(case))
    return results


def run_eval_case(case: EvalCase) -> EvalResult:
    try:
        response = answer_question(
            case.question,
            case.issue_date,
            6,
            history=case.history,
            session_context=case.session_context,
        )
        notes: list[str] = []
        passed = True
        answer_lower = response.answer.lower()
        if case.expected_intent:
            actual_intent = (response.debug_trace or {}).get("planner", {}).get("intent")
            if actual_intent != case.expected_intent:
                passed = False
                notes.append(f"Expected intent {case.expected_intent}, got {actual_intent}.")
        if case.expected_mode:
            actual_mode = response.mode
            if actual_mode != case.expected_mode:
                passed = False
                notes.append(f"Expected mode {case.expected_mode}, got {actual_mode}.")
        if case.expected_tools:
            tools = (response.debug_trace or {}).get("evidence_summary", {}).get("applied_tools", [])
            missing = [tool for tool in case.expected_tools if tool not in tools]
            if missing:
                passed = False
                notes.append(f"Missing expected tools: {missing}")
        if case.expect_abstention:
            abstained = (
                "couldn't verify a grounded answer" in answer_lower
                or "couldn't find enough grounded evidence" in answer_lower
                or "i can only answer questions grounded in the dataset" in answer_lower
                or "does not contain any information" in answer_lower
                or "there is no news about" in answer_lower
                or "no news about" in answer_lower
            )
            if not abstained:
                passed = False
                notes.append("Expected abstention, but answer did not abstain.")
        for value in case.must_include:
            if value.lower() not in answer_lower:
                passed = False
                notes.append(f"Missing required phrase: {value}")
        for value in case.must_not_include:
            if value.lower() in answer_lower:
                passed = False
                notes.append(f"Included banned phrase: {value}")
        failure = (response.debug_trace or {}).get("failure")
        if failure:
            notes.append(
                "Trace failure: "
                f"{failure.get('stage')}::{failure.get('error_type')}::{failure.get('message')}"
            )
        evidence_summary = (response.debug_trace or {}).get("evidence_summary", {})
        if evidence_summary:
            notes.append(
                "Trace evidence: "
                f"tools={evidence_summary.get('applied_tools', [])}, "
                f"items={evidence_summary.get('item_count')}, "
                f"confidence={evidence_summary.get('retrieval_confidence')}"
            )
        return EvalResult(
            case_id=case.case_id,
            category=case.category,
            passed=passed,
            score=1.0 if passed else 0.0,
            notes=notes,
        )
    except Exception as exc:
        return EvalResult(
            case_id=case.case_id,
            category=case.category,
            passed=False,
            score=0.0,
            notes=[f"Exception during eval: {exc}"],
        )


def summarize_eval_results(results: list[EvalResult]) -> dict:
    total = len(results)
    passed = sum(1 for item in results if item.passed)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
    }
