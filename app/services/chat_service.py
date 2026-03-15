import re
from difflib import SequenceMatcher

from app.schemas import ChatResponse
from app.services.openai_client import chat_completion
from app.services.query_router import is_broad_listing_query, is_section_count_query, route_query
from app.services.query_service import run_query
from app.services.repository import (
    fetch_matching_publications,
    fetch_publication_catalog,
    fetch_sql_article_count,
)


SYSTEM_PROMPT = """You answer questions from the TOI e-paper dataset.
Use only the provided article results. If the answer is not supported by the results, say so clearly.
Mention edition and section when useful.
If the user asks a follow-up question, use the prior conversation only as conversational context, but ground factual claims in the retrieved articles.
Sound natural and concise, like a helpful newsroom analyst, not a robot."""


STRUCTURED_RESULT_WINDOW = 5000
HYBRID_RESULT_WINDOW = 5000


def answer_question(
    question: str,
    issue_date: str | None,
    limit: int,
    session_filters: dict | None = None,
    history: list[dict[str, str]] | None = None,
    session_context: dict | None = None,
) -> ChatResponse:
    edition_clarification = _format_edition_followup_answer(question, session_context)
    if edition_clarification:
        edition_clarification.session_context = session_context
        return edition_clarification
    if _wants_article_text(question):
        candidate = _article_candidate_from_context(question, session_context)
        if candidate:
            response = _format_context_article_text_answer(candidate, session_context)
            response.session_context = session_context
            return response
    retrieval_question = _augment_followup_question(question, history, session_context)
    edition = _filter_value(session_filters, "edition")
    section = _filter_value(session_filters, "section")
    routed = route_query(retrieval_question, issue_date)
    edition = edition or _context_value(session_context, "edition", question)
    section = section or _context_value(session_context, "section", question)
    broad_listing = is_broad_listing_query(question)
    count_query = _is_count_query(question)
    requested_article_count = _requested_article_count(question)
    show_references = _should_show_references(question)
    result_window = limit
    if count_query or is_section_count_query(question) or (broad_listing and routed.mode == "sql"):
        result_window = STRUCTURED_RESULT_WINDOW
    elif broad_listing and routed.mode == "hybrid":
        result_window = HYBRID_RESULT_WINDOW
    elif _should_use_summary_answer(question, routed.mode):
        result_window = max(limit, 24)

    query_response = run_query(
        retrieval_question,
        issue_date,
        limit,
        edition=edition,
        section=section,
        result_window=result_window,
    )
    ambiguous_edition_answer = _format_ambiguous_edition_answer(query_response)
    if ambiguous_edition_answer:
        ambiguous_edition_answer.session_context = _build_session_context(
            question, query_response, session_context
        )
        return ambiguous_edition_answer
    if is_section_count_query(question):
        response = _format_section_counts(query_response)
        response.session_context = _build_session_context(question, query_response, session_context)
        return response
    if count_query:
        response = _format_count_answer(question, query_response)
        response.session_context = _build_session_context(question, query_response, session_context)
        return response
    if _wants_article_text(question):
        response = _format_article_text_answer(query_response)
        response.session_context = _build_session_context(question, query_response, session_context)
        return response
    if broad_listing and query_response.mode in {"sql", "hybrid"}:
        if _wants_exact_article_listing(question) and requested_article_count:
            response = _format_article_listing(question, query_response, requested_article_count)
            response.session_context = _build_session_context(question, query_response, session_context)
            return response
        response = _format_story_summary(question, query_response)
        response.session_context = _build_session_context(question, query_response, session_context)
        return response
    if _should_use_summary_answer(question, query_response.mode) and not show_references:
        response = _format_story_summary(question, query_response)
        response.session_context = _build_session_context(question, query_response, session_context)
        return response

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
        + "Write a concise answer. Focus on the main story themes and avoid edition-by-edition narration unless it is essential.\n\n"
        + "Retrieved articles:\n\n"
        + "\n\n---\n\n".join(context_lines)
    )
    answer = chat_completion(SYSTEM_PROMPT, user_prompt)
    return ChatResponse(
        answer=answer,
        mode=query_response.mode,
        citations=citations if show_references else [],
        session_context=_build_session_context(question, query_response, session_context),
    )


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
        lines[0] = f"I found {len(rows)} matching articles. Here are {min(requested_article_count, len(rows))} worth looking at."
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
            f"I found {len(rows)} semantically matched articles after applying the filters you asked for.\n\n"
            + "\n".join(lines[1:])
        )
    return ChatResponse(answer=answer, mode=query_response.mode, citations=citations)


def _format_story_summary(question: str, query_response) -> ChatResponse:
    story_groups = _group_unique_stories(query_response.results)
    if not story_groups:
        return ChatResponse(
            answer="I couldn't find any matching stories for that request.",
            mode=query_response.mode,
            citations=[],
        )

    context_blocks = []
    for index, story in enumerate(story_groups[:10], start=1):
        context_blocks.append(
            "\n".join(
                [
                    f"Headline: {story['headline']}",
                    f"Section: {story['section']}",
                    f"Edition count: {story['count']}",
                    f"Representative excerpt: {story['excerpt']}",
                ]
            )
        )

    prompt = (
        f"Question: {question}\n\n"
        "Write a concise answer that reads like a human analyst. "
        "Summarize only the unique stories below. "
        "Merge repeated editions of the same story into one point. "
        "Do not use numbered labels like Story 1, Story 2, or numbered theme lists. "
        "Do not output a raw article list, citation list, or repetitive edition-by-edition breakdown. "
        "Avoid mentioning specific editions unless the user explicitly asked for editions. "
        "If the user asked for all or broad coverage, mention the total number of matching articles and then summarize the dominant unique stories. "
        "If one story clearly dominates, say that directly. "
        "Keep the answer grounded only in the story summaries below.\n\n"
        f"Total matching articles: {len(query_response.results)}\n\n"
        + "\n\n---\n\n".join(context_blocks)
    )
    answer = chat_completion(SYSTEM_PROMPT, prompt)
    return ChatResponse(answer=answer, mode=query_response.mode, citations=[])


def _format_article_text_answer(query_response) -> ChatResponse:
    rows = query_response.results
    if not rows:
        return ChatResponse(
            answer="I couldn't find a relevant article for that request.",
            mode=query_response.mode,
            citations=[],
        )
    item = rows[0]
    headline = item.get("headline") or "Untitled"
    section = item.get("section") or "Unknown section"
    issue_date = item.get("issue_date") or "Unknown date"
    text = item.get("excerpt") or item.get("matched_chunk") or "No article text is available for this result."
    answer = (
        f"Here is one relevant article excerpt:\n\n"
        f"{headline}\n"
        f"{section} | {issue_date}\n\n"
        f"{text}"
    )
    citations = [
        {
            "article_id": item.get("external_article_id"),
            "headline": headline,
            "edition": item.get("edition"),
            "section": section,
            "issue_date": issue_date,
            "reference_text": text,
        }
    ]
    return ChatResponse(answer=answer, mode=query_response.mode, citations=citations)


def _format_context_article_text_answer(candidate: dict, session_context: dict | None) -> ChatResponse:
    headline = candidate.get("headline") or "Untitled"
    section = candidate.get("section") or "Unknown section"
    issue_date = candidate.get("issue_date") or "Unknown date"
    text = candidate.get("reference_text") or "No article text is available for this result."
    answer = (
        f"Here is one relevant article excerpt:\n\n"
        f"{headline}\n"
        f"{section} | {issue_date}\n\n"
        f"{text}"
    )
    citation = {
        "article_id": candidate.get("article_id"),
        "headline": headline,
        "edition": candidate.get("edition"),
        "section": section,
        "issue_date": issue_date,
        "reference_text": text,
    }
    return ChatResponse(
        answer=answer,
        mode=(session_context or {}).get("last_mode", "semantic"),
        citations=[citation],
    )


def _format_count_answer(question: str, query_response) -> ChatResponse:
    filters = query_response.filters
    edition = filters.get("edition")
    section = filters.get("section")
    issue_date = filters.get("issue_date")
    if query_response.mode == "sql":
        count = fetch_sql_article_count(issue_date, edition, section)
    else:
        count = len(query_response.results)
    scope_parts = []
    if section:
        scope_parts.append(f"{section} articles")
    else:
        scope_parts.append("articles")
    if edition:
        scope_parts.append(f"in {edition}")
    if issue_date:
        scope_parts.append(f"on {issue_date}")
    scope = " ".join(scope_parts)
    answer = f"There were {count} {scope}."
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


def _should_use_summary_answer(question: str, mode: str) -> bool:
    if mode not in {"semantic", "hybrid"}:
        return False
    if _wants_exact_article_listing(question) or _should_show_references(question):
        return False
    lowered = question.lower()
    patterns = [
        r"\bfind\b.*\barticles?\b",
        r"\bfind\b.*\bnews\b",
        r"\brelated to\b",
        r"\bnews about\b",
        r"\btell me news about\b",
        r"\bwhat was written about\b",
        r"\bwhat happened in\b",
        r"\bwhat is the news about\b",
        r"\bwhich stories\b",
    ]
    return any(re.search(pattern, lowered) for pattern in patterns)


def _requested_article_count(question: str) -> int | None:
    lowered = question.lower()
    match = re.search(r"\b(?:show|give|list)\s+(?:me\s+)?(\d{1,2})\s+articles?\b", lowered)
    if match:
        return int(match.group(1))
    return None


def _wants_article_text(question: str) -> bool:
    lowered = question.lower()
    patterns = [
        r"\btext of\b.*\barticle\b",
        r"\barticle text\b",
        r"\btext of any article\b",
        r"\bany one article\b",
        r"\bany one of article\b",
        r"\bany article\b",
        r"\bshow me text\b",
        r"\bshow the text\b",
        r"\bshow an article\b",
        r"\bshow any one\b.*\barticle\b",
        r"\bshow any\b.*\barticle\b",
        r"\bgive me an article\b",
        r"\bgive me the article\b",
        r"\bgive me article\b",
        r"\bgive any one\b.*\barticle\b",
        r"\bgive any\b.*\barticle\b",
        r"\bshow article text\b",
        r"\bfull article\b",
        r"\bexcerpt\b",
    ]
    return any(re.search(pattern, lowered) for pattern in patterns)


def _should_show_references(question: str) -> bool:
    lowered = question.lower()
    if _wants_article_text(question):
        return True
    patterns = [
        r"\bshow\b.*\barticles?\b",
        r"\bgive\b.*\bthe article\b",
        r"\bgive\b.*\ban article\b",
        r"\bgive\b.*\barticle\b",
        r"\bgive\b.*\barticles?\b",
        r"\blist\b.*\barticles?\b",
        r"\bwhich articles\b",
        r"\breference\b",
        r"\breferences\b",
        r"\bsource\b",
        r"\bsources\b",
        r"\bcitation\b",
        r"\bcitations\b",
        r"\bexample\b",
        r"\bexamples\b",
    ]
    return any(re.search(pattern, lowered) for pattern in patterns)


def _wants_exact_article_listing(question: str) -> bool:
    lowered = question.lower()
    if _wants_article_text(question):
        return False
    patterns = [
        r"\bgive me\b.*\bthe article\b",
        r"\bgive me\b.*\ban article\b",
        r"\bgive me\b.*\barticle\b",
        r"\bshow me\b.*\barticles?\b",
        r"\bshow\b.*\barticles?\b",
        r"\bgive me\b.*\barticles?\b",
        r"\blist\b.*\barticles?\b",
        r"\bwhich articles\b",
        r"\bshow sources\b",
        r"\bshow references\b",
        r"\bshow citations\b",
    ]
    return any(re.search(pattern, lowered) for pattern in patterns)


def _format_ambiguous_edition_answer(query_response) -> ChatResponse | None:
    edition = query_response.filters.get("edition")
    issue_date = query_response.filters.get("issue_date")
    if not edition:
        return None
    publication_names = {row["publication_name"] for row in fetch_publication_catalog()}
    if edition in publication_names:
        return None
    matches = fetch_matching_publications(issue_date, edition)
    if len(matches) <= 1:
        return None
    date_text = f" for {issue_date}" if issue_date else ""
    lines = [
        f"I don't see a single exact edition named {edition} in the dataset{date_text}.",
        f"Under the TOI{edition} publication family{date_text}, I found these editions instead:",
    ]
    for row in matches:
        lines.append(f"- {row['publication_name']}: {row['article_count']} articles")
    lines.append("Ask for one of those exact editions and I’ll give you a precise answer.")
    return ChatResponse(answer="\n".join(lines), mode=query_response.mode, citations=[])


def _format_edition_followup_answer(question: str, session_context: dict | None) -> ChatResponse | None:
    if not session_context:
        return None
    matches = session_context.get("ambiguous_publications")
    edition = session_context.get("ambiguous_edition")
    issue_date = session_context.get("issue_date")
    if not edition or not isinstance(matches, list) or not matches:
        return None
    if not _wants_edition_clarification(question):
        return None
    date_text = f" for {issue_date}" if issue_date else ""
    lines = [
        f"There isn't a single exact edition named {edition} in the dataset{date_text}.",
        f"These are the available editions under the TOI{edition} publication family{date_text}:",
    ]
    for row in matches:
        publication_name = row.get("publication_name") or "Unknown publication"
        article_count = row.get("article_count", 0)
        lines.append(f"- {publication_name}: {article_count} articles")
    lines.append("Ask for one of these exact editions and I’ll give you the precise result.")
    return ChatResponse(answer="\n".join(lines), mode=str(session_context.get("last_mode") or "sql"), citations=[])


def _group_unique_stories(rows: list[dict]) -> list[dict]:
    grouped: dict[str, dict] = {}
    for row in rows:
        headline = (row.get("headline") or "").strip()
        if not headline:
            continue
        key = _normalize_headline(headline)
        if not key:
            continue
        story = grouped.setdefault(
            key,
            {
                "headline": headline,
                "section": row.get("section") or "Unknown section",
                "editions": [],
                "count": 0,
                "excerpt": row.get("excerpt") or row.get("matched_chunk") or "",
            },
        )
        story["count"] += 1
        edition = row.get("edition") or "Unknown edition"
        if edition not in story["editions"]:
            story["editions"].append(edition)
        if len((row.get("excerpt") or row.get("matched_chunk") or "")) > len(story["excerpt"]):
            story["excerpt"] = row.get("excerpt") or row.get("matched_chunk") or story["excerpt"]
    return sorted(grouped.values(), key=lambda item: (item["count"], item["headline"]), reverse=True)


def _normalize_headline(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _augment_followup_question(
    question: str,
    history: list[dict[str, str]] | None,
    session_context: dict | None = None,
) -> str:
    if _is_generic_article_request(question) and session_context and session_context.get("last_topic"):
        return f"{question} {session_context['last_topic']}"
    if not _is_referential_followup(question):
        return question
    if not history:
        return _augment_with_session_story(question, session_context)
    title = _best_history_title_match(question, history, session_context)
    if not title:
        return _augment_with_session_story(question, session_context)
    return f"{question} {title}"


def _best_history_title_match(
    question: str,
    history: list[dict[str, str]] | None,
    session_context: dict | None = None,
) -> str | None:
    candidates = _extract_history_titles(history)
    candidates.extend(_session_story_titles(session_context))
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = _normalize_headline(candidate)
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(candidate)
    candidates = deduped
    if not candidates:
        return None
    topic = _extract_followup_topic(question)
    if not topic:
        return None
    scored: list[tuple[float, str]] = []
    normalized_topic = _normalize_headline(topic)
    for candidate in candidates:
        normalized_candidate = _normalize_headline(candidate)
        overlap = _token_overlap_score(normalized_topic, normalized_candidate)
        similarity = SequenceMatcher(None, normalized_topic, normalized_candidate).ratio()
        score = overlap + similarity
        if score >= 0.95 or (overlap >= 0.35 and similarity >= 0.45):
            scored.append((score, candidate))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _extract_followup_topic(question: str) -> str | None:
    lowered = question.lower()
    patterns = [
        r"(?:regarding|about|on)\s+(.+)",
        r"(?:the article|article)\s+(?:on|about|regarding)?\s*(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            topic = match.group(1)
            topic = re.sub(r"\b(?:what you have shared above as|what you shared above|you have shared above|you shared above|what you mentioned above|you mentioned above)\b", " ", topic)
            topic = re.sub(r"\b(?:what|which|that|those|above|shared|mentioned|have|you|as)\b", " ", topic)
            topic = re.sub(r"\s+", " ", topic).strip(" ,.?")
            if topic:
                return topic
    return question


def _extract_history_titles(history: list[dict[str, str]] | None) -> list[str]:
    if not history:
        return []
    titles: list[str] = []
    seen: set[str] = set()
    quoted_pattern = r"[\"“”']([^\"“”']{8,160})[\"“”']"
    title_case_pattern = r"(?:[A-Z][A-Za-z’'&.-]+(?:\s+[A-Z][A-Za-z’'&.-]+){2,8})"
    for item in reversed(history[-8:]):
        content = (item.get("content") or "").strip()
        if not content:
            continue
        candidates = re.findall(quoted_pattern, content)
        candidates.extend(re.findall(title_case_pattern, content))
        for candidate in candidates:
            cleaned = candidate.strip(" ,.-")
            if len(cleaned) < 8:
                continue
            normalized = _normalize_headline(cleaned)
            if normalized and normalized not in seen:
                seen.add(normalized)
                titles.append(cleaned)
    return titles


def _token_overlap_score(left: str, right: str) -> float:
    left_tokens = {token for token in left.split() if len(token) >= 4}
    right_tokens = {token for token in right.split() if len(token) >= 4}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens), 1)


def _augment_with_session_story(question: str, session_context: dict | None) -> str:
    titles = _session_story_titles(session_context)
    if not titles:
        return question
    if not _is_referential_followup(question):
        return question
    return f"{question} {titles[0]}"


def _session_story_titles(session_context: dict | None) -> list[str]:
    if not session_context:
        return []
    titles = session_context.get("story_titles") or []
    if isinstance(titles, list):
        return [str(title) for title in titles if title]
    return []


def _article_candidate_from_context(question: str, session_context: dict | None) -> dict | None:
    if not session_context:
        return None
    raw_candidates = session_context.get("article_candidates") or []
    story_candidates = session_context.get("story_candidates") or []
    candidates = raw_candidates if _is_generic_article_request(question) else story_candidates
    if not isinstance(candidates, list) or not candidates:
        return None
    if _is_generic_article_request(question):
        ranked = _rank_context_article_candidates(candidates, session_context)
        return ranked[0] if ranked else None
    topic = _extract_followup_topic(question) if not _is_generic_article_request(question) else None
    if not topic or _is_referential_followup(question) and topic in {"that", "those", "this article", "that article", "that story"}:
        return candidates[0]
    normalized_topic = _normalize_headline(topic)
    scored: list[tuple[float, dict]] = []
    for candidate in candidates:
        headline = str(candidate.get("headline") or "")
        normalized_candidate = _normalize_headline(headline)
        overlap = _token_overlap_score(normalized_topic, normalized_candidate)
        similarity = SequenceMatcher(None, normalized_topic, normalized_candidate).ratio()
        score = overlap + similarity
        if score >= 0.7 or (overlap >= 0.2 and similarity >= 0.3):
            scored.append((score, candidate))
    if not scored:
        return candidates[0] if _is_generic_article_request(question) else None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _rank_context_article_candidates(candidates: list[dict], session_context: dict | None) -> list[dict]:
    preferred_section = (session_context or {}).get("section")
    last_topic = str((session_context or {}).get("last_topic") or "")
    last_question = str((session_context or {}).get("last_question") or "").lower()
    ranked: list[tuple[tuple[float, float], dict]] = []
    normalized_topic = _normalize_headline(last_topic) if last_topic else ""
    for candidate in candidates:
        headline = str(candidate.get("headline") or "")
        normalized_headline = _normalize_headline(headline)
        topic_score = 0.0
        if normalized_topic and normalized_headline:
            topic_score = _token_overlap_score(normalized_topic, normalized_headline) + SequenceMatcher(
                None, normalized_topic, normalized_headline
            ).ratio()
        section_score = _section_priority_score(candidate.get("section"), preferred_section, last_question)
        ranked.append(((section_score, topic_score), candidate))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [candidate for _, candidate in ranked]


def _section_priority_score(section: str | None, preferred_section: str | None, last_question: str) -> float:
    normalized_section = str(section or "").lower()
    if preferred_section and normalized_section == str(preferred_section).lower():
        return 5.0
    if "iran" in last_question or "war" in last_question or "conflict" in last_question:
        if normalized_section == "world":
            return 4.5
        if normalized_section == "nation":
            return 3.5
        if normalized_section == "business":
            return 2.5
        if normalized_section in {"edit", "feature"}:
            return 0.5
    if "budget" in last_question or "middle class" in last_question or "inflation" in last_question:
        if normalized_section == "business":
            return 4.5
        if normalized_section == "nation":
            return 2.5
        return 1.0
    if "world cup" in last_question or "sports" in last_question or "cricket" in last_question:
        if normalized_section == "sports":
            return 4.5
        return 0.5
    return 1.0


def _context_value(session_context: dict | None, key: str, question: str) -> str | None:
    if not session_context:
        return None
    if not _should_apply_context(question):
        return None
    value = session_context.get(key)
    return str(value) if value else None


def _build_session_context(question: str, query_response, prior_context: dict | None) -> dict:
    base = dict(prior_context or {})
    filters = query_response.filters or {}
    if filters.get("edition"):
        base["edition"] = filters["edition"]
    if filters.get("section"):
        base["section"] = filters["section"]
    if filters.get("issue_date"):
        base["issue_date"] = filters["issue_date"]
    base["last_mode"] = query_response.mode
    base["last_question"] = question
    story_titles = [story["headline"] for story in _group_unique_stories(query_response.results)[:5]]
    story_candidates = []
    article_candidates = []
    for item in query_response.results[:8]:
        article_candidates.append(
            {
                "article_id": item.get("external_article_id"),
                "headline": item.get("headline"),
                "edition": item.get("edition"),
                "section": item.get("section"),
                "issue_date": item.get("issue_date"),
                "reference_text": item.get("excerpt") or item.get("matched_chunk"),
            }
        )
    for story in _group_unique_stories(query_response.results)[:20]:
        story_candidates.append(
            {
                "headline": story["headline"],
                "edition": story["editions"][0] if story["editions"] else None,
                "section": story["section"],
                "issue_date": filters.get("issue_date"),
                "reference_text": story["excerpt"],
            }
        )
    if story_titles:
        base["story_titles"] = story_titles
        base["last_topic"] = story_titles[0]
    if story_candidates:
        base["story_candidates"] = story_candidates
    if article_candidates:
        base["article_candidates"] = article_candidates
    edition = filters.get("edition")
    issue_date = filters.get("issue_date")
    if edition:
        publication_names = {row["publication_name"] for row in fetch_publication_catalog()}
        if edition not in publication_names:
            matches = fetch_matching_publications(issue_date, edition)
            if len(matches) > 1:
                base["ambiguous_edition"] = edition
                base["ambiguous_publications"] = matches
    return base


def _is_referential_followup(question: str) -> bool:
    lowered = question.lower()
    patterns = [
        r"\bthat\b",
        r"\bthose\b",
        r"\bthese\b",
        r"\bthe one\b",
        r"\bthe same\b",
        r"\babove\b",
        r"\byou shared\b",
        r"\byou mentioned\b",
        r"\bwhat about\b",
        r"\bregarding that\b",
        r"\babout that\b",
        r"\bthis article\b",
        r"\bthat article\b",
        r"\bthat story\b",
        r"\bthose stories\b",
        r"\bit\b",
    ]
    return any(re.search(pattern, lowered) for pattern in patterns)


def _should_apply_context(question: str) -> bool:
    lowered = question.lower().strip()
    if not lowered:
        return False
    if _is_generic_article_request(question) or _wants_exact_article_listing(question) or _wants_article_text(question):
        return True
    if _is_referential_followup(question):
        return True
    explicit_reset_patterns = [
        r"\bnews about\b",
        r"\barticles about\b",
        r"\bstories about\b",
        r"\bfind\b",
        r"\bshow me\b",
        r"\blist\b",
        r"\bwhich\b",
        r"\bwhat\b",
        r"\bhow many\b",
        r"\bcount\b",
        r"\bwho\b",
        r"\bwhen\b",
    ]
    if any(re.search(pattern, lowered) for pattern in explicit_reset_patterns):
        return False
    if len(re.findall(r"[a-z0-9]+", lowered)) >= 4:
        return False
    return True


def _is_generic_article_request(question: str) -> bool:
    lowered = question.lower()
    if re.search(r"\b(regarding|about|on)\b", lowered):
        return False
    patterns = [
        r"\bany one article\b",
        r"\bany one of article\b",
        r"\bany article\b",
        r"\bshow any one\b.*\barticle\b",
        r"\bgive any one\b.*\barticle\b",
        r"\bgive any\b.*\barticle\b",
    ]
    return any(re.search(pattern, lowered) for pattern in patterns)


def _wants_edition_clarification(question: str) -> bool:
    lowered = question.lower()
    patterns = [
        r"\bwhat exact editions\b",
        r"\bwhich exact editions\b",
        r"\bwhat editions are available\b",
        r"\bwhich editions are available\b",
        r"\bavailable editions\b",
        r"\bexact editions\b",
    ]
    return any(re.search(pattern, lowered) for pattern in patterns)
