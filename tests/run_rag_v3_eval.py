import json
from pathlib import Path

from app.services.rag_v3.evals import load_eval_cases, run_eval_case, summarize_eval_results


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cases_path = repo_root / "benchmarks" / "rag_v3_eval_cases.json"
    results_path = repo_root / "benchmarks" / "rag_v3_eval_results.json"
    cases = load_eval_cases(cases_path)
    results = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.case_id} :: {case.question}", flush=True)
        result = run_eval_case(case)
        results.append(result)
        payload = {
            "summary": summarize_eval_results(results),
            "results": [item.model_dump() for item in results],
        }
        results_path.write_text(json.dumps(payload, indent=2))
        print(
            f"  -> {'PASS' if result.passed else 'FAIL'} | notes={result.notes}",
            flush=True,
        )
    print(json.dumps(summarize_eval_results(results), indent=2))


if __name__ == "__main__":
    main()
