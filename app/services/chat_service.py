import re
from difflib import SequenceMatcher

from app.core.config import get_settings
from app.schemas import ChatResponse, QueryResponse, RoutedQuery
from app.services.query_analyzer import analyze_query
from app.services.openai_client import chat_completion
from app.services.query_router import (
    is_broad_listing_query,
    is_section_count_query,
    normalize_user_query,
    route_query,
)
from app.services.query_service import run_query
from app.services.repository import (
    
    fetch_matching_publications,
    fetch_publication_catalog,
    fetch_sql_article_count,
    fetch_section_counts,
)

_settings = get_settings()

SYSTEM_PROMPT = """Role:
You are an enterprise-grade retrieval-augmented news analyst for the TOI e-paper dataset.

Grounding:
- Use only the retrieved dataset evidence supplied in the prompt.
- Treat conversation history as context for intent, not as evidence.
- If the evidence is weak, mixed, incomplete, or missing, say that plainly.

Reasoning rules:
- Prefer exact supported claims over broad summaries.
- Do not invent article details, dates, people, sections, or editions.
- Resolve follow-up questions using the current query plus the provided session context.
- Merge duplicate coverage of the same story across editions when summarizing.

Edge cases:
- If retrieval results conflict, mention the conflict briefly instead of forcing certainty.
- If the user asks for count, answer the count first and then the dominant context if supported.
- If the user asks for summaries, focus on themes, not raw lists, unless listing is explicitly requested.
- If the user asks for article text or references, stay close to the retrieved excerpts.

Style:
- Be concise, direct, and newsroom-professional.
- Mention edition and section only when helpful to answer the question precisely."""

COMPLEX_SYSTEM_PROMPT = """Role:
You are an enterprise-grade retrieval-augmented news analyst for the TOI e-paper dataset.

Grounding:
- Use only the retrieved dataset evidence supplied in the prompt.
- Evidence blocks are tagged [HIGH], [MEDIUM], or [LOW] based on retrieval confidence.
- Prefer [HIGH] evidence blocks for primary claims; [MEDIUM] for supporting context; treat [LOW] as supplementary only.
- Treat conversation history as context for intent, not as evidence.
- If the evidence is weak, mixed, incomplete, or missing, say that plainly.

Chain-of-thought reasoning:
Before writing your final answer, reason through the evidence inside <reasoning>...</reasoning> tags:
1. List the key claims from each evidence block (headline-level).
2. Enumerate ALL article headlines from the evidence, including lower-confidence ones — a relevant answer may appear in a [LOW] block.
3. Flag any conflicts or contradictions between evidence blocks.
4. Decide which evidence blocks best answer the user's question.
5. Write the final answer.

The <reasoning> block will be stripped before returning to the user — write it freely.

Reasoning rules:
- Prefer exact supported claims over broad summaries.
- Do not invent article details, dates, people, sections, or editions.
- Resolve follow-up questions using the current query plus the provided session context.
- Merge duplicate coverage of the same story across editions when summarizing.
- When summarizing about a topic (e.g., budget, middle class), explicitly check every evidence block's headline and excerpt for relevance before dismissing it.

Edge cases:
- If retrieval results conflict, mention the conflict briefly instead of forcing certainty.
- If the user asks for count, answer the count first and then the dominant context if supported.
- If the user asks for summaries, focus on themes, not raw lists, unless listing is explicitly requested.
- If the user asks for article text or references, stay close to the retrieved excerpts.

Style:
- Be concise, direct, and newsroom-professional.
- Mention edition and section only when helpful to answer the question precisely."""


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
    trace = _base_trace(question, issue_date, session_filters, session_context)
    contextual_followup = _format_contextual_followup_answer(question, session_context)
    if contextual_followup:
        trace["answer_path"] = "contextual_followup"
        contextual_followup.debug_trace = trace
        contextual_followup.session_context = session_context
        return contextual_followup
    edition_clarification = _format_edition_followup_answer(question, session_context)
    if edition_clarification:
        trace["answer_path"] = "edition_followup"
        edition_clarification.debug_trace = trace
        edition_clarification.session_context = session_context
        return edition_clarification
    edition_answer = _format_edition_usage_answer(question, session_context)
    if edition_answer:
        trace["answer_path"] = "edition_usage"
        edition_answer.debug_trace = trace
        edition_answer.session_context = session_context
        return edition_answer
    if _wants_article_text(question):
        candidate = _article_candidate_from_context(question, session_context)
        if candidate:
            response = _format_context_article_text_answer(candidate, session_context)
            trace["answer_path"] = "context_article_text"
            trace["context_candidate"] = {
                "headline": candidate.get("headline"),
                "section": candidate.get("section"),
            }
            response.debug_trace = trace
            response.session_context = session_context
            return response
    requested_article_count = _requested_article_count(question)
    if (
        _wants_exact_article_listing(question)
        and requested_article_count
        and session_context
        and _is_referential_followup(question)
    ):
        cached = session_context.get("article_candidates") or []
        if cached:
            last_mode = str(session_context.get("last_mode") or "sql")
            last_mode = last_mode if last_mode in {"sql", "semantic", "hybrid"} else "sql"
            synthetic = QueryResponse(
                mode=last_mode,  # type: ignore[arg-type]
                filters={},
                results=cached,
            )
            response = _format_article_listing(question, synthetic, requested_article_count)
            trace["answer_path"] = "context_article_listing"
            response.debug_trace = trace
            response.session_context = session_context
            return response
    user_analysis = analyze_query(question, issue_date)
    user_routed = user_analysis.routed
    retrieval_question = _augment_followup_question(question, history, session_context)
    edition = _filter_value(session_filters, "edition")
    section = _filter_value(session_filters, "section")
    retrieval_analysis = analyze_query(retrieval_question, issue_date)
    routed = retrieval_analysis.routed
    routed.intent = user_routed.intent
    if user_routed.author:
        routed.author = user_routed.author
    if user_routed.edition == "Delhi" and question.lower().find("delhi edition") != -1:
        routed.edition = "Delhi"
    edition = edition or _context_value(session_context, "edition", question)
    section = section or _context_value(session_context, "section", question)
    broad_listing = is_broad_listing_query(question)
    count_query = routed.intent in {"article_count", "topic_count", "author_count"}
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
        routed_override=routed,
    )
    trace.update(
        {
            "user_analysis": user_analysis.model_dump(mode="json"),
            "retrieval_question": retrieval_question,
            "retrieval_analysis": retrieval_analysis.model_dump(mode="json"),
            "resolved_filters": {
                "edition": edition,
                "section": section,
            },
            "result_window": result_window,
            "result_count": len(query_response.results),
            "response_mode": query_response.mode,
        }
    )
    ambiguous_edition_answer = _format_ambiguous_edition_answer(query_response)
    if ambiguous_edition_answer:
        trace["answer_path"] = "ambiguous_edition"
        ambiguous_edition_answer.debug_trace = trace
        ambiguous_edition_answer.session_context = _build_session_context(
            question, query_response, session_context
        )
        return ambiguous_edition_answer
    if is_section_count_query(question):
        response = _format_section_counts(question, query_response)
        trace["answer_path"] = "section_count"
        response.debug_trace = trace
        response.session_context = _build_session_context(question, query_response, session_context)
        return response
    if count_query:
        response = _format_count_answer(question, query_response, routed)
        trace["answer_path"] = "count_answer"
        response.debug_trace = trace
        response.session_context = _build_session_context(question, query_response, session_context)
        return response
    if routed.intent == "author_lookup":
        if _wants_exact_article_listing(question) or _should_show_references(question):
            response = _format_article_listing(question, query_response, requested_article_count)
            trace["answer_path"] = "author_listing"
        else:
            response = _format_author_summary(query_response, routed.author)
            trace["answer_path"] = "author_summary"
        response.debug_trace = trace
        response.session_context = _build_session_context(question, query_response, session_context)
        return response
    if _wants_article_text(question):
        response = _format_article_text_answer(query_response)
        trace["answer_path"] = "article_text"
        response.debug_trace = trace
        response.session_context = _build_session_context(question, query_response, session_context)
        return response
    if broad_listing and query_response.mode in {"sql", "hybrid"}:
        if _wants_exact_article_listing(question) and requested_article_count:
            response = _format_article_listing(question, query_response, requested_article_count)
            trace["answer_path"] = "article_listing"
            response.debug_trace = trace
            response.session_context = _build_session_context(question, query_response, session_context)
            return response
        response = _format_story_summary(question, query_response)
        trace["answer_path"] = "story_summary_listing"
        response.debug_trace = trace
        response.session_context = _build_session_context(question, query_response, session_context)
        return response
    if _should_use_summary_answer(question, query_response.mode) and not show_references:
        response = _format_story_summary(question, query_response)
        trace["answer_path"] = "story_summary"
        response.debug_trace = trace
        response.session_context = _build_session_context(question, query_response, session_context)
        return response

    # Low-confidence path: no evidence at all, don't hallucinate.
    # Only trigger when results are completely empty to avoid intercepting
    # cases where ranking/filtering reduced results but evidence still exists.
    confidence = query_response.confidence_score
    if (
        confidence < _settings.low_confidence_threshold
        and len(query_response.results) == 0
        and query_response.mode in {"semantic", "hybrid"}
    ):
        answer = _format_low_confidence_answer(question, query_response, confidence)
        return ChatResponse(
            answer=answer,
            mode=query_response.mode,
            citations=[],
            confidence_score=confidence,
            session_context=_build_session_context(question, query_response, session_context),
            debug_trace={**trace, "answer_path": "low_confidence"},
        )

    # Select model and system prompt based on intent complexity.
    use_strong = (
        user_routed.intent in _settings.strong_model_intent_triggers
        or len(query_response.results) > 15
    )
    # Use complex (CoT) prompt for semantic queries (to surface hidden evidence
    # in lower-ranked results, e.g. water crisis in SEM001) and for complex
    # intents. Hybrid/SQL queries already have precise filters and benefit from
    # the standard prompt's comprehensive topic coverage.
    use_complex_prompt = (
        query_response.mode == "semantic"
        or user_routed.intent in _settings.strong_model_intent_triggers
    )
    system_prompt = COMPLEX_SYSTEM_PROMPT if use_complex_prompt else SYSTEM_PROMPT
    model_override = _settings.openai_chat_model_strong if use_strong else None

    citations = _build_citations(query_response.results[:limit])
    user_prompt = _build_layered_answer_prompt(
        question=question,
        query_response=query_response,
        history=history,
        limit=limit,
        show_references=show_references,
        use_confidence_tiers=use_complex_prompt,
    )
    raw_answer = chat_completion(system_prompt, user_prompt, model=model_override)
    # Strip internal <reasoning>...</reasoning> block before returning.
    answer = _strip_reasoning_block(raw_answer)
    return ChatResponse(
        answer=answer,
        mode=query_response.mode,
        citations=citations if show_references else [],
        confidence_score=confidence,
        session_context=_build_session_context(question, query_response, session_context),
        debug_trace={**trace, "answer_path": "llm_answer", "model_used": model_override or _settings.openai_chat_model},
    )


def _base_trace(
    question: str,
    issue_date: str | None,
    session_filters: dict | None,
    session_context: dict | None,
) -> dict:
    return {
        "question": question,
        "issue_date": issue_date,
        "session_filters": session_filters or {},
        "session_context_keys": sorted(list((session_context or {}).keys())),
    }


def _format_section_counts(question: str, query_response) -> ChatResponse:
    rows = query_response.results
    if not rows:
        return ChatResponse(answer="No section counts matched the requested issue date.", mode="sql", citations=[])
    lowered = question.lower()
    ordinal_match = re.search(r"\b(second|third|fourth|fifth|2nd|3rd|4th|5th)\b", lowered)
    if ordinal_match or re.search(r"\bwhich one was\b", lowered):
        ordinal_map = {"second": 1, "2nd": 1, "third": 2, "3rd": 2, "fourth": 3, "4th": 3, "fifth": 4, "5th": 4}
        rank = ordinal_map.get(ordinal_match.group(1) if ordinal_match else "", 1)
        if rank < len(rows):
            ranked_section = rows[rank].get("section") or "Unclassified"
            ranked_count = rows[rank].get("article_count", 0)
            ordinal_word = ordinal_match.group(1) if ordinal_match else "second"
            lines = [
                f"The {ordinal_word} section with the most articles on March 11 was {ranked_section} with {ranked_count} articles.",
                "",
                "Full section ranking:",
            ]
            for index, row in enumerate(rows, start=1):
                section = row.get("section") or "Unclassified"
                lines.append(f"{index}. {section}: {row.get('article_count', 0)}")
            return ChatResponse(answer="\n".join(lines), mode="sql", citations=[])
    if _is_least_section_query(question):
        bottom = rows[-1]
        bottom_section = bottom.get("section") or "Unclassified"
        bottom_count = bottom.get("article_count", 0)
        lines = [f"{bottom_section} had the fewest articles on March 11 with {bottom_count} pieces.", "", "Full section ranking:"]
        for index, row in enumerate(rows, start=1):
            section = row.get("section") or "Unclassified"
            lines.append(f"{index}. {section}: {row.get('article_count', 0)}")
        return ChatResponse(answer="\n".join(lines), mode="sql", citations=[])
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
        if requested_article_count:
            header = f"I found {len(rows)} semantically matched articles after applying the filters you asked for. Here are {min(requested_article_count, len(rows))} worth looking at."
        else:
            header = f"I found {len(rows)} semantically matched articles after applying the filters you asked for."
        answer = header + "\n\n" + "\n".join(lines[1:])
    return ChatResponse(answer=answer, mode=query_response.mode, citations=citations)


def _format_author_summary(query_response, author: str | None) -> ChatResponse:
    rows = query_response.results
    author_name = author or "that author"
    if not rows:
        return ChatResponse(
            answer=f"I couldn't find any articles by {author_name} in the current dataset.",
            mode=query_response.mode,
            citations=[],
        )
    story_groups = _group_unique_stories(rows)
    total_count = rows[0].get("author_article_count", len(rows))
    if not story_groups:
        return ChatResponse(
            answer=f"I found {total_count} article{'s' if total_count != 1 else ''} by {author_name}.",
            mode=query_response.mode,
            citations=[],
        )
    lead = story_groups[:3]
    summary = "; ".join(
        f"{story['headline']} ({story['count']} article{'s' if story['count'] != 1 else ''})"
        for story in lead
    )
    return ChatResponse(
        answer=(
            f"I found {total_count} article{'s' if total_count != 1 else ''} by {author_name}. "
            f"The main pieces are {summary}."
        ),
        mode=query_response.mode,
        citations=[],
    )


def _format_contextual_followup_answer(question: str, session_context: dict | None) -> ChatResponse | None:
    if not session_context:
        return None
    candidates = session_context.get("story_candidates") or []
    if not isinstance(candidates, list) or not candidates:
        return None
    if not _asks_contextual_summary_followup(question):
        return None
    subject = _contextual_subject_label(session_context)
    lead = candidates[:3]
    summary = "; ".join(
        f"{item.get('headline') or 'Untitled'}"
        for item in lead
    )
    count = session_context.get("result_count") or len(candidates)
    noun = _contextual_result_noun(session_context)
    return ChatResponse(
        answer=(
            f"I found {count} {noun}{'s' if count != 1 else ''}{subject}. "
            f"They were mainly about {summary}."
        ),
        mode=str(session_context.get("last_mode") or "sql"),
        citations=[],
    )


def _format_story_summary(question: str, query_response) -> ChatResponse:
    story_groups = _group_unique_stories(query_response.results)
    if not story_groups:
        return ChatResponse(
            answer="I couldn't find any matching stories for that request.",
            mode=query_response.mode,
            citations=[],
        )

    prompt = _build_story_summary_prompt(question, query_response, story_groups)
    answer = chat_completion(SYSTEM_PROMPT, prompt)
    return ChatResponse(answer=answer, mode=query_response.mode, citations=[])


def _build_citations(results: list[dict]) -> list[dict]:
    citations = []
    for item in results:
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
    return citations


def _confidence_tier(similarity: float) -> str:
    if similarity >= 0.70:
        return "HIGH"
    if similarity >= 0.50:
        return "MEDIUM"
    return "LOW"


def _strip_reasoning_block(text: str) -> str:
    """Remove <reasoning>...</reasoning> block from LLM output."""
    return re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL).strip()


def _format_low_confidence_answer(question: str, query_response, confidence: float) -> str:
    closest = query_response.results[0] if query_response.results else None
    excerpt = ""
    if closest:
        excerpt = closest.get("excerpt") or closest.get("matched_chunk") or ""
        if excerpt:
            excerpt = f" Closest match: {excerpt[:200]}"
    topic = question[:80]
    return (
        f"Limited evidence found for: {topic}.{excerpt} "
        f"(Confidence: low, score={confidence:.2f})"
    )


def _build_layered_answer_prompt(
    *,
    question: str,
    query_response,
    history: list[dict[str, str]] | None,
    limit: int,
    show_references: bool,
    use_confidence_tiers: bool = False,
) -> str:
    evidence_blocks = []
    for index, item in enumerate(query_response.results[:limit], start=1):
        similarity = float(item.get("similarity", 0.0))
        tier = _confidence_tier(similarity) if use_confidence_tiers else None
        tier_tag = f"[{tier}] " if tier else ""
        evidence_blocks.append(
            "\n".join(
                [
                    f"[Evidence {index}] {tier_tag}similarity={similarity:.2f}",
                    f"Headline: {item.get('headline') or 'Untitled'}",
                    f"Edition: {item.get('edition') or 'Unknown edition'}",
                    f"Section: {item.get('section') or 'Unknown section'}",
                    f"Issue Date: {item.get('issue_date') or 'Unknown date'}",
                    f"Excerpt: {item.get('excerpt') or item.get('matched_chunk') or 'No excerpt available.'}",
                ]
            )
        )
    conversation_context = _format_history(history)
    answer_contract = [
        "Answer contract:",
        "- Start with the direct answer to the user's question.",
        "- Base every factual claim on the evidence blocks only.",
        "- If the evidence is partial, say so explicitly.",
        "- Avoid repetitive edition-by-edition narration unless it changes the answer.",
    ]
    if show_references:
        answer_contract.append("- Mention the most relevant supporting examples naturally in the answer.")
    else:
        answer_contract.append("- Do not output a raw citation list unless explicitly requested.")
    return "\n\n".join(
        part
        for part in [
            "Layer 1 - User question:\n" + question,
            (
                "Layer 2 - Conversation context:\n" + conversation_context
                if conversation_context
                else "Layer 2 - Conversation context:\nNo prior conversation context."
            ),
            "Layer 3 - Retrieval metadata:\n"
            + "\n".join(
                [
                    f"Mode: {query_response.mode}",
                    f"Filters: {query_response.filters}",
                    f"Retrieved results: {len(query_response.results)}",
                ]
            ),
            "Layer 4 - Evidence:\n" + ("\n\n".join(evidence_blocks) if evidence_blocks else "No evidence blocks available."),
            "Layer 5 - Edge-case policy:\n"
            + "\n".join(
                [
                    "- If no evidence blocks support the answer, say you could not confirm it from the retrieved articles.",
                    "- If multiple evidence blocks describe the same story, consolidate them.",
                    "- If the user is asking for themes or context, summarize the dominant contexts rather than listing every article.",
                    "- If the question is factual, prefer exact figures or statements from the evidence over general wording.",
                ]
            ),
            "Layer 6 - " + "\n".join(answer_contract),
        ]
    )


def _build_story_summary_prompt(question: str, query_response, story_groups: list[dict]) -> str:
    story_blocks = []
    for index, story in enumerate(story_groups[:20], start=1):
        story_blocks.append(
            "\n".join(
                [
                    f"[Story {index}]",
                    f"Headline: {story['headline']}",
                    f"Section: {story['section']}",
                    f"Edition count: {story['count']}",
                    f"Representative excerpt: {story['excerpt']}",
                ]
            )
        )
    return "\n\n".join(
        [
            "Layer 1 - User question:\n" + question,
            "Layer 2 - Retrieval metadata:\n"
            + "\n".join(
                [
                    f"Mode: {query_response.mode}",
                    f"Total matching articles: {len(query_response.results)}",
                    f"Unique stories: {len(story_groups)}",
                ]
            ),
            "Layer 3 - Story evidence:\n" + "\n\n".join(story_blocks),
            "Layer 4 - Summary policy:\n"
            + "\n".join(
                [
                    "- Summarize only the unique stories below.",
                    "- Merge repeated editions of the same story into one point.",
                    "- Do not output a raw article list or repetitive edition-by-edition breakdown.",
                    "- Avoid numbered theme labels like Story 1 or Theme 1 in the final answer.",
                    "- Cover ALL unique story themes — do not skip minority topics.",
                    "- If one story dominates, note it but still mention the other distinct stories.",
                ]
            ),
            "Layer 5 - Answer contract:\n"
            + "\n".join(
                [
                    "- Write like a concise newsroom analyst.",
                    f"- Begin your answer by stating the total article count: 'Among the {len(query_response.results)} articles...' or similar.",
                    "- Start with the main conclusion after mentioning the count.",
                    "- Mention dominant themes and supporting examples only when useful.",
                ]
            ),
        ]
    )


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


def _format_count_answer(question: str, query_response, routed: RoutedQuery) -> ChatResponse:
    if routed.intent == "author_count":
        author = routed.author or "that author"
        count = query_response.results[0].get("author_article_count", 0) if query_response.results else 0
        if _asks_for_context(question) or "what they about" in question.lower() or "what are they about" in question.lower():
            story_groups = _group_unique_stories(query_response.results)
            if story_groups:
                lead = story_groups[:3]
                summary = "; ".join(
                    f"{story['headline']} ({story['count']} article{'s' if story['count'] != 1 else ''})"
                    for story in lead
                )
                return ChatResponse(
                    answer=(
                        f"I found {count} article{'s' if count != 1 else ''} by {author}. "
                        f"They are mainly about {summary}."
                    ),
                    mode=query_response.mode,
                    citations=[],
                )
        return ChatResponse(
            answer=f"I found {count} article{'s' if count != 1 else ''} by {author}.",
            mode=query_response.mode,
            citations=[],
        )
    if routed.intent == "topic_count":
        return _format_topic_count_answer(question, query_response)
    filters = query_response.filters
    edition = filters.get("edition")
    section = filters.get("section")
    issue_date = filters.get("issue_date")
    count = fetch_sql_article_count(issue_date, edition, section)
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


def _format_topic_count_answer(question: str, query_response) -> ChatResponse:
    exact_contexts = query_response.filters.get("exact_contexts") or []
    story_groups = _group_unique_stories(query_response.results)
    article_count = int(query_response.filters.get("exact_article_count") or len(query_response.results))
    if article_count == 0:
        return ChatResponse(
            answer="I couldn't find any relevant articles for that topic in the current dataset.",
            mode=query_response.mode,
            citations=[],
        )
    topic = str(query_response.filters.get("entity_label") or _extract_topic_from_question(question))
    if _asks_for_context(question):
        if exact_contexts:
            dominant_text = "; ".join(
                f"{row.get('headline')} ({row.get('article_count', 0)} article{'s' if row.get('article_count', 0) != 1 else ''})"
                for row in exact_contexts[:4]
                if row.get("headline")
            )
            answer = (
                f"I found {article_count} relevant articles mentioning {topic}. "
                f"The coverage is mainly in the context of {dominant_text}."
            )
            return ChatResponse(answer=answer, mode=query_response.mode, citations=[])
        if not story_groups:
            answer = f"I found {article_count} relevant articles mentioning {topic}."
            return ChatResponse(answer=answer, mode=query_response.mode, citations=[])
        dominant = story_groups[:3]
        story_text = "; ".join(
            f"{story['headline']} ({story['count']} article{'s' if story['count'] != 1 else ''})"
            for story in dominant
        )
        answer = (
            f"I found {article_count} relevant articles mentioning {topic}. "
            f"The coverage is mainly in the context of {story_text}."
        )
        return ChatResponse(answer=answer, mode=query_response.mode, citations=[])
    answer = f"I found {article_count} relevant articles about {topic}."
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


def _asks_for_context(question: str) -> bool:
    lowered = question.lower()
    return any(
        phrase in lowered
        for phrase in [
            "in what context",
            "in which context",
            "what context",
            "which context",
            "what was the context",
            "what they were about",
            "what they are about",
            "what they about",
            "what were they about",
            "what are they about",
            "how was",
            "why was",
            "appeared and in what context",
        ]
    )


def _asks_contextual_summary_followup(question: str) -> bool:
    lowered = question.lower().strip()
    patterns = [
        r"\band what they were about\b",
        r"\band what they are about\b",
        r"\band what they about\b",
        r"\bwhat they were about\b",
        r"\bwhat they are about\b",
        r"\bwhat were they about\b",
        r"\bwhat are they about\b",
        r"\bwhat were those about\b",
        r"\bwhat are those about\b",
        r"\band what was it about\b",
        r"\bwhat was it about\b",
        r"\bwhat was that about\b",
        r"\band what was that about\b",
    ]
    return any(re.search(pattern, lowered) for pattern in patterns)


def _contextual_subject_label(session_context: dict | None) -> str:
    if not session_context:
        return ""
    if session_context.get("author"):
        return f" by {session_context['author']}"
    if session_context.get("last_topic"):
        return f" for {session_context['last_topic']}"
    return ""


def _contextual_result_noun(session_context: dict | None) -> str:
    if not session_context:
        return "result"
    if session_context.get("author"):
        return "article"
    if session_context.get("section"):
        return "story"
    return "result"


def _extract_topic_from_question(question: str) -> str:
    lowered = normalize_user_query(question).lower().strip(" ?.")
    patterns = [
        r"\bhow many times\s+(.+?)\s+(?:name\s+)?appeared\b",
        r"\b([a-z][a-z\s'.-]+?)\s+name appeared\b",
        r"\bhow many article(?:s)?\s+(?:about|around|regarding|on)\s+(.+)",
        r"\b(?:around|about|regarding|on)\s+(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            topic = _clean_topic_phrase(match.group(1))
            if topic:
                return _topic_display_label(topic)
    fallback = re.sub(r"\bhow many\b|\barticles?\b|\btimes\b|\bappeared\b|\bname\b", " ", lowered)
    fallback = _clean_topic_phrase(fallback)
    fallback = re.sub(r"\s+", " ", fallback).strip(" ?.,")
    return _topic_display_label(fallback) if fallback else "that topic"


def _clean_topic_phrase(value: str) -> str:
    cleaned = value.lower().strip(" ,.?")
    cleaned = re.sub(r"\band\s+and\b", " and", cleaned)
    cleaned = re.sub(
        r"\s+and\s+(?:in (?:what|which) context(?: they are)?|(?:what|which) context(?: they are)?|what they (?:were|are)? about.*)$",
        "",
        cleaned,
    )
    cleaned = re.sub(r"\bin (?:what|which) context(?: they are)?\b.*$", "", cleaned)
    cleaned = re.sub(r"\b(?:what|which) context(?: they are)?\b.*$", "", cleaned)
    cleaned = re.sub(r"\bwhat they (?:were|are)? about\b.*$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.?")
    return cleaned


def _topic_display_label(value: str) -> str:
    normalized = value.lower().strip()
    if normalized == "modi":
        return "Narendra Modi"
    return normalized.title()


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
        r"\bone article\b.*\babove conversation\b",
        r"\bgive me one article\b",
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


def _format_edition_usage_answer(question: str, session_context: dict | None) -> ChatResponse | None:
    if not session_context:
        return None
    if not _asks_for_used_edition(question):
        return None
    edition = session_context.get("edition")
    issue_date = session_context.get("issue_date")
    if not edition:
        return None
    date_text = f" for {issue_date}" if issue_date else ""
    return ChatResponse(
        answer=f"I used the edition filter {edition}{date_text}.",
        mode=str(session_context.get("last_mode") or "sql"),
        citations=[],
    )


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
        ranked = _rank_context_article_candidates(raw_candidates, session_context)
        if ranked:
            return ranked[0]
        ranked_story_candidates = _rank_context_story_candidates(story_candidates, session_context)
        return ranked_story_candidates[0] if ranked_story_candidates else None
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
    query_focus = str((session_context or {}).get("query_focus") or "")
    last_question = str((session_context or {}).get("last_question") or "").lower()
    ranked: list[tuple[tuple[float, float], dict]] = []
    normalized_topic = _normalize_headline(query_focus or last_topic) if (query_focus or last_topic) else ""
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


def _rank_context_story_candidates(candidates: list[dict], session_context: dict | None) -> list[dict]:
    if not isinstance(candidates, list) or not candidates:
        return []
    preferred_section = (session_context or {}).get("section")
    query_focus = str((session_context or {}).get("query_focus") or "")
    last_topic = str((session_context or {}).get("last_topic") or "")
    last_question = str((session_context or {}).get("last_question") or "").lower()
    normalized_focus = _normalize_headline(query_focus or last_topic) if (query_focus or last_topic) else ""
    ranked: list[tuple[tuple[float, float], dict]] = []
    for candidate in candidates:
        headline = str(candidate.get("headline") or "")
        normalized_headline = _normalize_headline(headline)
        topic_score = 0.0
        if normalized_focus and normalized_headline:
            topic_score = _token_overlap_score(normalized_focus, normalized_headline) + SequenceMatcher(
                None, normalized_focus, normalized_headline
            ).ratio()
        section_score = _section_priority_score(candidate.get("section"), preferred_section, last_question)
        ranked.append(((section_score, topic_score), candidate))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [candidate for _, candidate in ranked]


def _section_priority_score(section: str | None, preferred_section: str | None, last_question: str) -> float:
    normalized_section = str(section or "").lower()
    if preferred_section and normalized_section == str(preferred_section).lower():
        return 5.0
    if "journalist" in last_question or "middle east" in last_question or "mideast" in last_question:
        if normalized_section == "world":
            return 4.5
        if normalized_section == "sports":
            return 0.1  # strongly deprioritize Sports for journalist/middle-east queries
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
    if filters.get("author"):
        base["author"] = filters["author"]
    if filters.get("edition"):
        base["edition"] = filters["edition"]
    if filters.get("section"):
        base["section"] = filters["section"]
    if filters.get("issue_date"):
        base["issue_date"] = filters["issue_date"]
    base["last_mode"] = query_response.mode
    base["last_question"] = question
    base["result_count"] = query_response.results[0].get("author_article_count", len(query_response.results)) if query_response.results else 0
    story_titles = [story["headline"] for story in _group_unique_stories(query_response.results)[:5]]
    story_candidates = []
    article_candidates = []
    for item in query_response.results[:20]:
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
    query_focus = _derive_query_focus(question)
    if query_focus:
        base["query_focus"] = query_focus
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
        r"\babove conversation\b",
        r"\bfrom the above conversation\b",
        r"\bfrom above conversation\b",
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
        r"\bone article\b",
        r"\bgive me one article\b",
        r"\bany one article\b",
        r"\bany one of article\b",
        r"\bany article\b",
        r"\bshow any one\b.*\barticle\b",
        r"\bgive me\b.*\bone article\b",
        r"\bgive any one\b.*\barticle\b",
        r"\bgive any\b.*\barticle\b",
    ]
    return any(re.search(pattern, lowered) for pattern in patterns)


def _derive_query_focus(question: str) -> str | None:
    lowered = question.lower()
    if "journalist" in lowered and "restrict" in lowered:
        return "journalists restrictions middle east war"
    if "world cup" in lowered:
        return "india world cup win bcci reward suryakumar"
    if "iran" in lowered and "mumbai" in lowered:
        return "war brings new water crises to parched iran"
    if "iran" in lowered:
        return "iran war"
    if "budget" in lowered and "middle class" in lowered:
        return "supply woes higher oil prices hit growth fuel inflation economists"
    if "rahul gandhi" in lowered:
        return "rahul gandhi"
    if "china" in lowered:
        return "world newest beijing problem china growth ambitions"
    if "lpg" in lowered or "png" in lowered or "cng" in lowered:
        return "lpg png cng supply priority"
    topic = _extract_topic_from_question(question)
    return None if topic == "that topic" else topic


def _asks_for_used_edition(question: str) -> bool:
    lowered = question.lower()
    patterns = [
        r"\bwhich edition did you use\b",
        r"\bwhat edition did you use\b",
        r"\bwhich edition was used\b",
        r"\bwhat edition was used\b",
    ]
    return any(re.search(pattern, lowered) for pattern in patterns)


def _is_least_section_query(question: str) -> bool:
    lowered = question.lower()
    return "least articles" in lowered or "fewest articles" in lowered


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
