import re

from app.schemas import ChatResponse
from app.services.openai_client import chat_completion
from app.services.query_router import is_broad_listing_query, is_section_count_query, route_query
from app.services.query_service import run_query


SYSTEM_PROMPT = """You answer questions from the TOI e-paper dataset.
Use only the provided article results. If the answer is not supported by the results, say so clearly.
Mention edition and section when useful.
If the user asks a follow-up question, use the prior conversation only as conversational context, but ground factual claims in the retrieved articles."""


STRUCTURED_RESULT_WINDOW = 5000
HYBRID_RESULT_WINDOW = 5000


def answer_question(
    question: str,
    issue_date: str | None,
    limit: int,
    session_filters: dict | None = None,
    history: list[dict[str, str]] | None = None,
) -> ChatResponse:
    edition = _filter_value(session_filters, "edition")
    section = _filter_value(session_filters, "section")
    routed = route_query(question, issue_date)
    broad_listing = is_broad_listing_query(question)
    count_query = _is_count_query(question)
    requested_article_count = _requested_article_count(question)
    result_window = limit
    if count_query or is_section_count_query(question) or (broad_listing and routed.mode == "sql"):
        result_window = STRUCTURED_RESULT_WINDOW
    elif broad_listing and routed.mode == "hybrid":
        result_window = HYBRID_RESULT_WINDOW

    query_response = run_query(
        question,
        issue_date,
        limit,
        edition=edition,
        section=section,
        result_window=result_window,
    )
    if is_section_count_query(question):
        return _format_section_counts(query_response)
    if count_query:
        return _format_count_answer(question, query_response)
    if broad_listing and query_response.mode in {"sql", "hybrid"}:
        return _format_article_listing(question, query_response, requested_article_count)

    context_lines = []
    citations = []
    for item in query_response.results[:limit]:
        context_lines.append(
            "\n".join(
                [
                    f"Headline: {item.get('headline')}",
                    f"Edition: {item.get('edition')}",
                    f"Section: {item.get('section')}",
                    f"Issue Date: {item.get('issue_date')}",
                    f"Excerpt: {item.get('excerpt') or item.get('matched_chunk')}",
                ]
            )
        )
        citations.append(
            {
                "article_id": item.get("external_article_id"),
                "headline": item.get("headline"),
                "edition": item.get("edition"),
                "section": item.get("section"),
                "issue_date": item.get("issue_date"),
                "reference_text": item.get("excerpt") or item.get("matched_chunk") or "No supporting excerpt available.",
            }
        )

    conversation_context = _format_history(history)
    user_prompt = (
        (f"Conversation so far:\n{conversation_context}\n\n" if conversation_context else "")
        + f"Question: {question}\n\n"
        + "Retrieved articles:\n\n"
        + "\n\n---\n\n".join(context_lines)
    )
    answer = chat_completion(SYSTEM_PROMPT, user_prompt)
    return ChatResponse(answer=answer, mode=query_response.mode, citations=citations)


def _format_section_counts(query_response) -> ChatResponse:
    rows = query_response.results
    if not rows:
        return ChatResponse(answer="No section counts matched the requested issue date.", mode="sql", citations=[])
    top = rows[0]
    top_section = top.get("section") or "Unclassified"
    top_count = top.get("article_count", 0)
    lines = [f"{top_section} had the most articles on March 11 with {top_count} pieces.", "", "Full section ranking:"]
    for index, row in enumerate(rows, start=1):
        section = row.get("section") or "Unclassified"
        lines.append(f"{index}. {section}: {row.get('article_count', 0)}")
    return ChatResponse(answer="\n".join(lines), mode="sql", citations=[])


def _format_article_listing(question: str, query_response, requested_article_count: int | None) -> ChatResponse:
    rows = query_response.results
    if not rows:
        return ChatResponse(
            answer="No matching articles were found for that request.",
            mode=query_response.mode,
            citations=[],
        )

    display_rows = rows[:requested_article_count] if requested_article_count else rows
    citations = []
    lines = [f"I found {len(rows)} matching articles."]
    if requested_article_count:
        lines[0] = f"I found {len(rows)} matching articles. Here are {min(requested_article_count, len(rows))} of them."
    for index, item in enumerate(display_rows, start=1):
        headline = item.get("headline") or "Untitled"
        edition = item.get("edition") or "Unknown edition"
        section = item.get("section") or "Unknown section"
        issue_date = item.get("issue_date") or "Unknown date"
        lines.append(f"{index}. {headline} | {edition} | {section} | {issue_date}")
        citations.append(
            {
                "article_id": item.get("external_article_id"),
                "headline": headline,
                "edition": edition,
                "section": section,
                "issue_date": issue_date,
                "reference_text": item.get("excerpt") or item.get("matched_chunk") or "No supporting excerpt available.",
            }
        )

    answer = "\n".join(lines)
    if query_response.mode == "hybrid":
        answer = (
            f"I found {len(rows)} semantically matched articles after applying the structured filters.\n\n"
            + "\n".join(lines[1:])
        )
    return ChatResponse(answer=answer, mode=query_response.mode, citations=citations)


def _format_count_answer(question: str, query_response) -> ChatResponse:
    count = len(query_response.results)
    filters = query_response.filters
    edition = filters.get("edition")
    section = filters.get("section")
    issue_date = filters.get("issue_date")
    scope_parts = []
    if section:
        scope_parts.append(f"{section} articles")
    else:
        scope_parts.append("articles")
    if edition:
        scope_parts.append(f"in the {edition} scope")
    if issue_date:
        scope_parts.append(f"on {issue_date}")
    scope = " ".join(scope_parts)
    answer = f"There were {count} {scope}."
    if edition and edition in {"Delhi", "Mumbai", "Kolkata", "Chennai", "Ahmedabad", "Bangalore", "Hyderabad", "Lucknow", "Bhopal", "Chandigarh", "Jaipur", "Pune", "Cochin"}:
        answer += " I treated that as the edition family name across matching publications in the dataset."
    return ChatResponse(answer=answer, mode=query_response.mode, citations=[])


def _filter_value(session_filters: dict | None, key: str) -> str | None:
    if not session_filters:
        return None
    value = session_filters.get(key)
    if value in (None, "", "all"):
        return None
    return str(value)


def _format_history(history: list[dict[str, str]] | None) -> str:
    if not history:
        return ""
    lines = []
    for item in history[-8:]:
        role = item.get("role")
        content = (item.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


def _is_count_query(question: str) -> bool:
    lowered = question.lower()
    patterns = [
        r"\bhow many\b",
        r"\bcount\b",
        r"\bnumber of\b",
        r"\bnumbers of\b",
    ]
    return any(re.search(pattern, lowered) for pattern in patterns)


def _requested_article_count(question: str) -> int | None:
    lowered = question.lower()
    match = re.search(r"\b(?:show|give|list)\s+(?:me\s+)?(\d{1,2})\s+articles?\b", lowered)
    if match:
        return int(match.group(1))
    return None
