from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from app.db.database import close_pool, ensure_schema, open_pool
from app.services.chat_service import answer_question


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "benchmarks" / "chat_benchmark_cases.txt"
TODAY = date.today().isoformat()
RESULTS_PATH = ROOT / "benchmarks" / f"live_benchmark_results_{TODAY}.txt"
ISSUE_DATE = "2026-03-11"
LIMIT = 10


@dataclass
class Case:
    case_id: str
    category: str
    prompt: str
    follow_up: str
    initial_must_contain: list[str]
    initial_must_not_contain: list[str]
    follow_must_contain: list[str]
    follow_must_not_contain: list[str]


def parse_tokens(raw: str) -> list[str]:
    return [token.strip().lower() for token in raw.split(";") if token.strip()]


def load_cases() -> list[Case]:
    rows = CASES_PATH.read_text(encoding="utf-8").splitlines()
    header, *lines = rows
    cases: list[Case] = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 8:
            raise ValueError(f"Invalid benchmark row: {line}")
        cases.append(
            Case(
                case_id=parts[0],
                category=parts[1],
                prompt=parts[2],
                follow_up=parts[3],
                initial_must_contain=parse_tokens(parts[4]),
                initial_must_not_contain=parse_tokens(parts[5]),
                follow_must_contain=parse_tokens(parts[6]),
                follow_must_not_contain=parse_tokens(parts[7]),
            )
        )
    return cases


def evaluate(answer: str, must_contain: list[str], must_not_contain: list[str]) -> tuple[bool, list[str]]:
    lowered = answer.lower()
    failures: list[str] = []
    for token in must_contain:
        if token not in lowered:
            failures.append(f"missing:{token}")
    for token in must_not_contain:
        if token in lowered:
            failures.append(f"unexpected:{token}")
    return (not failures, failures)


def run_case(case: Case) -> dict:
    history: list[dict[str, str]] = []
    session_context = None

    first = answer_question(case.prompt, ISSUE_DATE, LIMIT, history=history, session_context=session_context)
    first_ok, first_failures = evaluate(
        first.answer,
        case.initial_must_contain,
        case.initial_must_not_contain,
    )

    history.append({"role": "user", "content": case.prompt})
    history.append({"role": "assistant", "content": first.answer})
    session_context = first.session_context

    second = answer_question(case.follow_up, ISSUE_DATE, LIMIT, history=history, session_context=session_context)
    second_ok, second_failures = evaluate(
        second.answer,
        case.follow_must_contain,
        case.follow_must_not_contain,
    )

    return {
        "case_id": case.case_id,
        "category": case.category,
        "initial_ok": first_ok,
        "follow_ok": second_ok,
        "initial_failures": first_failures,
        "follow_failures": second_failures,
        "initial_answer": first.answer,
        "follow_answer": second.answer,
    }


def main() -> None:
    cases = load_cases()
    results: list[dict] = []
    partial_lines = [
        "TOI RAG Live Benchmark Report",
        f"Run date: {TODAY}",
        f"Issue date: {ISSUE_DATE}",
        f"Cases: {len(cases)}",
        f"Prompt turns: {len(cases) * 2}",
        "",
        "Progress log:",
        "",
    ]
    RESULTS_PATH.write_text("\n".join(partial_lines), encoding="utf-8")
    open_pool()
    ensure_schema()
    try:
        for case in cases:
            print(f"Running {case.case_id}: {case.prompt}", flush=True)
            result = run_case(case)
            results.append(result)
            status = "PASS" if result["initial_ok"] and result["follow_ok"] else "FAIL"
            partial_lines.extend(
                [
                    f"{result['case_id']} [{result['category']}] - {status}",
                    f"Initial failures: {', '.join(result['initial_failures']) or 'none'}",
                    f"Follow failures: {', '.join(result['follow_failures']) or 'none'}",
                    "",
                ]
            )
            RESULTS_PATH.write_text("\n".join(partial_lines), encoding="utf-8")
    finally:
        close_pool()

    passed = sum(1 for item in results if item["initial_ok"] and item["follow_ok"])
    failed = len(results) - passed

    lines = [
        "TOI RAG Live Benchmark Report",
        f"Run date: {TODAY}",
        f"Issue date: {ISSUE_DATE}",
        f"Cases: {len(results)}",
        f"Prompt turns: {len(results) * 2}",
        f"Passed cases: {passed}",
        f"Failed cases: {failed}",
        "",
        "Detailed results:",
        "",
    ]

    for item in results:
        status = "PASS" if item["initial_ok"] and item["follow_ok"] else "FAIL"
        lines.extend(
            [
                f"{item['case_id']} [{item['category']}] - {status}",
                f"Initial failures: {', '.join(item['initial_failures']) or 'none'}",
                f"Follow failures: {', '.join(item['follow_failures']) or 'none'}",
                f"Initial answer: {item['initial_answer']}",
                f"Follow answer: {item['follow_answer']}",
                "",
            ]
        )

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote benchmark report to {RESULTS_PATH}")
    print(f"Passed cases: {passed}/{len(results)}")


if __name__ == "__main__":
    main()
