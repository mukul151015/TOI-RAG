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

# Similarity threshold for semantic evaluation of FAIL cases.
SEMANTIC_MATCH_THRESHOLD = 0.72


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


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def semantic_evaluate(
    answer: str,
    must_contain: list[str],
    must_not_contain: list[str],
) -> tuple[bool, list[str]]:
    """Embedding-based evaluation for FAIL cases.

    Embeds the full answer and each required token, then flags a match when
    cosine similarity exceeds ``SEMANTIC_MATCH_THRESHOLD``.  Only called on
    cases that fail the token check, so API cost is minimal.
    """
    try:
        from app.services.openai_client import embed_texts  # late import

        texts_to_embed = [answer] + must_contain
        embeddings = embed_texts(texts_to_embed)
        answer_vec = embeddings[0]
        token_vecs = embeddings[1:]

        failures: list[str] = []
        for token, token_vec in zip(must_contain, token_vecs):
            sim = _cosine_similarity(answer_vec, token_vec)
            if sim < SEMANTIC_MATCH_THRESHOLD:
                failures.append(f"missing:{token}(sem={sim:.2f})")
        # must_not_contain: token-based check is fine for negatives.
        lowered = answer.lower()
        for token in must_not_contain:
            if token in lowered:
                failures.append(f"unexpected:{token}")
        return (not failures, failures)
    except Exception:
        # If semantic eval fails, fall back to token check result.
        return evaluate(answer, must_contain, must_not_contain)


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
    # Semantic eval on FAIL turns only (reduces API cost).
    first_sem_ok, first_sem_failures = first_ok, first_failures
    if not first_ok and (case.initial_must_contain or case.initial_must_not_contain):
        first_sem_ok, first_sem_failures = semantic_evaluate(
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
    second_sem_ok, second_sem_failures = second_ok, second_failures
    if not second_ok and (case.follow_must_contain or case.follow_must_not_contain):
        second_sem_ok, second_sem_failures = semantic_evaluate(
            second.answer,
            case.follow_must_contain,
            case.follow_must_not_contain,
        )

    return {
        "case_id": case.case_id,
        "category": case.category,
        # Token-based results
        "initial_ok": first_ok,
        "follow_ok": second_ok,
        "initial_failures": first_failures,
        "follow_failures": second_failures,
        # Semantic results (only meaningful when token check fails)
        "initial_sem_ok": first_sem_ok,
        "follow_sem_ok": second_sem_ok,
        "initial_sem_failures": first_sem_failures,
        "follow_sem_failures": second_sem_failures,
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
            token_ok = result["initial_ok"] and result["follow_ok"]
            sem_ok = result["initial_sem_ok"] and result["follow_sem_ok"]
            status = "PASS" if token_ok else ("SEM-PASS" if sem_ok else "FAIL")
            partial_lines.extend(
                [
                    f"{result['case_id']} [{result['category']}] - {status}",
                    f"Initial token failures: {', '.join(result['initial_failures']) or 'none'}",
                    f"Follow token failures: {', '.join(result['follow_failures']) or 'none'}",
                    "",
                ]
            )
            RESULTS_PATH.write_text("\n".join(partial_lines), encoding="utf-8")
    finally:
        close_pool()

    token_passed = sum(1 for item in results if item["initial_ok"] and item["follow_ok"])
    sem_passed = sum(1 for item in results if item["initial_sem_ok"] and item["follow_sem_ok"])
    total = len(results)
    token_pass_rate = token_passed / total if total else 0.0
    semantic_pass_rate = sem_passed / total if total else 0.0

    lines = [
        "TOI RAG Live Benchmark Report",
        f"Run date: {TODAY}",
        f"Issue date: {ISSUE_DATE}",
        f"Cases: {total}",
        f"Prompt turns: {total * 2}",
        f"Token pass rate:    {token_passed}/{total} ({token_pass_rate:.1%})",
        f"Semantic pass rate: {sem_passed}/{total} ({semantic_pass_rate:.1%})",
        "",
        "Detailed results:",
        "",
    ]

    for item in results:
        token_ok = item["initial_ok"] and item["follow_ok"]
        sem_ok = item["initial_sem_ok"] and item["follow_sem_ok"]
        status = "PASS" if token_ok else ("SEM-PASS" if sem_ok else "FAIL")
        lines.extend(
            [
                f"{item['case_id']} [{item['category']}] - {status}",
                f"Initial token failures: {', '.join(item['initial_failures']) or 'none'}",
                f"Follow token failures:  {', '.join(item['follow_failures']) or 'none'}",
                f"Initial sem failures:   {', '.join(item['initial_sem_failures']) or 'none'}",
                f"Follow sem failures:    {', '.join(item['follow_sem_failures']) or 'none'}",
                f"Initial answer: {item['initial_answer']}",
                f"Follow answer: {item['follow_answer']}",
                "",
            ]
        )

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote benchmark report to {RESULTS_PATH}")
    print(f"Token pass rate:    {token_passed}/{total} ({token_pass_rate:.1%})")
    print(f"Semantic pass rate: {sem_passed}/{total} ({semantic_pass_rate:.1%})")


if __name__ == "__main__":
    main()
