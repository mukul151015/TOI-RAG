import json
from pathlib import Path

from app.services.rag_v3.evals import load_eval_cases, run_eval_case, summarize_eval_results
from app.services.rag_v3.pipeline import answer_question


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cases_path = repo_root / "benchmarks" / "rag_v3_industry_eval_cases.json"
    results_path = repo_root / "benchmarks" / "rag_v3_industry_eval_results.json"
    cases = load_eval_cases(cases_path)
    results = []
    eval_results = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.case_id} :: {case.question}", flush=True)
        response = answer_question(
            case.question,
            case.issue_date,
            6,
            history=case.history,
            session_context=case.session_context,
        )
        result = run_eval_case(case)
        eval_results.append(result)
        results.append(
            {
                "case": case.model_dump(),
                "evaluation": result.model_dump(),
                "actual": {
                    "answer": response.answer,
                    "mode": response.mode,
                    "verification": None if not response.verification else response.verification.model_dump(),
                    "trace": response.debug_trace,
                },
            }
        )
        payload = {
            "summary": summarize_eval_results(eval_results),
            "results": results,
        }
        results_path.write_text(json.dumps(payload, indent=2))
        print(
            f"  -> {'PASS' if result.passed else 'FAIL'} | mode={response.mode} | answer={response.answer[:180]}",
            flush=True,
        )
    print(json.dumps(summarize_eval_results(eval_results), indent=2))


if __name__ == "__main__":
    main()
