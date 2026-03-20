from __future__ import annotations

from datetime import date
from difflib import SequenceMatcher, get_close_matches
import re

from app.schemas import RetrievalPlan, UserIntent
from app.services.openai_client import chat_completion
from app.services.query_analyzer import expand_person_alias_terms
from app.services.rag_v3.common import dedupe_preserve_order, normalize_text, parse_calendar_date, safe_load_json
from app.services.rag_v3.query_rewriter import rewrite_question
from app.services.repository import fetch_author_catalog, fetch_publication_catalog, fetch_section_catalog


PLANNER_SYSTEM_PROMPT = """You are the planner for a production news RAG system.
Return strict JSON with keys:
- task_type: one of count|summary|list|article_text|compare|ranking
- answer_shape: one of count_only|count_plus_summary|list|article_text|summary|compare|ranking
- intent
- mode: one of sql|semantic|hybrid
- edition: optional publication/edition name if the user is asking about a specific edition
- section: optional section name if the user is asking about a specific section/page
- author: optional author name if the user is asking about a specific author/writer
- needs_count (boolean)
- needs_summary (boolean)
- needs_listing (boolean)
- needs_article_text (boolean)
- query_focus: short search-focused rewrite for retrieval
- entity_terms: list of entity or topic terms to search for
- retrieval_tools: list drawn from structured_count, structured_articles, semantic_chunks, headline_keyword, story_clusters
- fallback_order: same tool vocabulary
- reasoning
Use deterministic filters provided in the prompt. Do not invent filters or facts. Return JSON only.
Use article_text only when the user explicitly asks for full article text/body/excerpt.
Questions like "what was written in the delhi edition" or "what was written in the world section" are summary requests.
Questions like "which year had the most coverage of Modi" are ranking requests."""

INJECTION_MARKERS = [
    "ignore previous instructions",
    "ignore all instructions",
    "system prompt",
    "developer message",
    "reveal prompt",
]


def parse_user_intent(
    question: str,
    issue_date: str | None,
    history: list[dict[str, str]] | None = None,
    session_context: dict | None = None,
) -> tuple[UserIntent, RetrievalPlan]:
    rewrite = rewrite_question(question, history=history, session_context=session_context)
    standalone_question = rewrite["standalone_question"]
    time_filters = _resolve_time_filters(standalone_question, issue_date, session_context or {})
    raw_planner_payload = _llm_plan(standalone_question, time_filters, history, session_context)
    planner_payload = _normalized_planner_payload(raw_planner_payload, standalone_question)
    filters = _resolve_filters(
        standalone_question,
        issue_date,
        session_context=session_context,
        planner_payload=raw_planner_payload,
    )

    task_type = planner_payload["task_type"]
    answer_shape = planner_payload["answer_shape"]
    needs_count = bool(planner_payload["needs_count"])
    needs_summary = bool(planner_payload["needs_summary"])
    needs_listing = bool(planner_payload["needs_listing"])
    needs_article_text = bool(planner_payload["needs_article_text"])
    entity_terms = _resolve_entity_terms(planner_payload, standalone_question)
    mode = _resolve_mode(planner_payload.get("mode"), needs_count, needs_summary, needs_listing, entity_terms, filters)
    if task_type == "ranking":
        mode = "sql"
    normalized_intent = planner_payload["intent"]
    if filters.get("author"):
        normalized_intent = "author_count" if needs_count else "author_lookup"
    retrieval_tools = _select_tools(
        planner_payload.get("retrieval_tools"),
        needs_count,
        needs_summary,
        needs_listing,
        needs_article_text,
        mode,
        task_type=task_type,
        has_entities=bool(entity_terms),
        has_author=bool(filters.get("author")),
    )
    fallback_order = _select_fallbacks(planner_payload.get("fallback_order"), retrieval_tools, needs_count, mode)
    query_focus = planner_payload.get("query_focus") or _emergency_query_focus(standalone_question, entity_terms)
    unsupported = _is_unsupported_or_injected(question)
    reasoning = planner_payload.get("reasoning") or f"Planned {answer_shape} for task {task_type}."

    intent = UserIntent(
        original_question=question,
        standalone_question=standalone_question,
        intent=str(normalized_intent),
        mode=mode,
        needs_count=needs_count,
        needs_summary=needs_summary,
        needs_listing=needs_listing,
        needs_article_text=needs_article_text,
        entities={
            "entity_terms": entity_terms,
            "query_focus": [query_focus] if query_focus else [],
            "rewriter_reasoning": [rewrite["reasoning"]],
        },
        filters=filters,
        ambiguity_note=filters.get("ambiguity_note"),
        reasoning=f"{reasoning} {'Unsupported or prompt-injection markers detected.' if unsupported else ''}".strip(),
    )
    plan = RetrievalPlan(
        mode=mode,
        intent=intent.intent,
        task_type=task_type,
        answer_shape=answer_shape,
        time_scope=filters.get("time_scope"),
        issue_date=filters.get("issue_date"),
        start_date=filters.get("start_date"),
        end_date=filters.get("end_date"),
        edition=filters.get("edition"),
        section=filters.get("section"),
        author=filters.get("author"),
        semantic_query=query_focus,
        entity_terms=entity_terms,
        retrieval_tools=retrieval_tools,
        fallback_order=fallback_order,
        confidence=0.75 if "structured_count" in retrieval_tools else 0.6,
        reasoning=reasoning,
        notes=dedupe_preserve_order([
            "Deterministic filters are code-resolved.",
            "Counts must come from structured retrieval.",
            "Summary generation is grounded on retrieved evidence only.",
            "Unsupported requests should abstain." if unsupported else "",
        ]),
    )
    return intent, plan


def replan_after_retrieval(intent: UserIntent, plan: RetrievalPlan, evidence_count: int, retrieval_confidence: float) -> RetrievalPlan:
    if plan.task_type == "ranking":
        return plan
    if retrieval_confidence >= 0.45 and evidence_count >= 2:
        return plan
    updated = plan.model_copy(deep=True)
    for tool in plan.fallback_order:
        if tool not in updated.retrieval_tools:
            updated.retrieval_tools.append(tool)
            updated.notes.append(f"Added fallback tool after weak retrieval: {tool}.")
            break
    updated.confidence = max(0.25, min(plan.confidence, retrieval_confidence))
    return updated


def _llm_plan(standalone_question: str, filters: dict, history: list[dict[str, str]] | None, session_context: dict | None) -> dict:
    prompt = (
        f"Question: {standalone_question}\n"
        f"Deterministic filters: {filters}\n"
        f"History: {(history or [])[-6:]}\n"
        f"Session context: {session_context or {}}\n"
        "Return JSON only."
    )
    try:
        return safe_load_json(chat_completion(PLANNER_SYSTEM_PROMPT, prompt, timeout=25.0))
    except Exception:
        return {}


def _normalized_planner_payload(payload: dict, question: str) -> dict:
    emergency = _emergency_plan_defaults(question)
    answer_shape = payload.get("answer_shape") if payload.get("answer_shape") in {"count_only", "count_plus_summary", "list", "article_text", "summary", "compare", "ranking"} else emergency["answer_shape"]
    proposed_task_type = payload.get("task_type") if payload.get("task_type") in {"count", "summary", "list", "article_text", "compare", "ranking"} else emergency["task_type"]
    task_type = _normalize_task_type(proposed_task_type, answer_shape, payload)
    normalized_intent = _normalize_intent(payload.get("intent"), task_type, answer_shape, emergency["intent"])
    needs_count = payload.get("needs_count") if isinstance(payload.get("needs_count"), bool) else answer_shape in {"count_only", "count_plus_summary"}
    needs_summary = bool(payload.get("needs_summary")) or answer_shape in {"count_plus_summary", "summary", "compare", "ranking"}
    needs_listing = bool(payload.get("needs_listing")) or answer_shape == "list"
    needs_article_text = bool(payload.get("needs_article_text")) or answer_shape == "article_text"
    if answer_shape == "ranking" or task_type == "ranking":
        needs_count = False
        needs_summary = True
        needs_listing = False
        needs_article_text = False
    if not _question_requests_count(question) and answer_shape == "count_only" and _question_prefers_summary_lookup(question):
        task_type = "summary"
        answer_shape = "summary"
        normalized_intent = "lookup"
        needs_count = False
        needs_summary = True
    if not _question_requests_list(question) and answer_shape == "list" and _question_prefers_summary_lookup(question):
        task_type = "summary"
        answer_shape = "summary"
        normalized_intent = "lookup"
        needs_listing = False
        needs_summary = True
    if not _question_requests_article_text(question) and answer_shape == "article_text":
        task_type = "summary"
        answer_shape = "summary"
        normalized_intent = "lookup"
        needs_article_text = False
        needs_summary = True
    return {
        "task_type": task_type,
        "answer_shape": answer_shape,
        "intent": normalized_intent,
        "needs_count": needs_count,
        "needs_summary": needs_summary,
        "needs_listing": needs_listing,
        "needs_article_text": needs_article_text,
        "query_focus": payload.get("query_focus") or emergency["query_focus"],
        "entity_terms": payload.get("entity_terms") or emergency["entity_terms"],
        "retrieval_tools": payload.get("retrieval_tools") or emergency["retrieval_tools"],
        "fallback_order": payload.get("fallback_order") or emergency["fallback_order"],
        "mode": payload.get("mode"),
        "reasoning": payload.get("reasoning") or emergency["reasoning"],
    }


def _resolve_filters(
    question: str,
    issue_date: str | None,
    session_context: dict | None = None,
    planner_payload: dict | None = None,
) -> dict:
    session_context = session_context or {}
    planner_payload = planner_payload or {}
    publications = fetch_publication_catalog()
    sections = fetch_section_catalog()
    authors = fetch_author_catalog()
    time_filters = _resolve_time_filters(question, issue_date, session_context)
    publication_names = [row["publication_name"] for row in publications]
    edition = _resolve_catalog_value(planner_payload.get("edition"), publication_names)
    if not edition and _should_extract_edition_filter(question):
        edition = _best_catalog_match(question, publication_names)
    if not edition and _should_extract_edition_filter(question):
        edition = _extract_filter_phrase(question, "edition")
    section = _resolve_catalog_value(planner_payload.get("section"), sections)
    if not section and _should_extract_section_filter(question):
        section = _best_catalog_match(question, sections)
    if not section and _should_extract_section_filter(question):
        section = _extract_filter_phrase(question, "section")
    author = _resolve_catalog_value(planner_payload.get("author"), authors)
    if not author and _should_extract_author_filter(question):
        author = _best_catalog_match(question, authors)
    ambiguity_note = None
    if edition and sum(1 for row in publications if "delhi" in row["publication_name"].lower() and "delhi" in question.lower()) > 1:
        ambiguity_note = "Edition mention may map to multiple Delhi-family publications."
    return {
        "issue_date": time_filters.get("issue_date"),
        "start_date": time_filters.get("start_date"),
        "end_date": time_filters.get("end_date"),
        "time_scope": time_filters.get("time_scope"),
        "edition": edition,
        "section": section,
        "author": author,
        "ambiguity_note": ambiguity_note,
    }


def _resolve_time_filters(question: str, issue_date: str | None, session_context: dict[str, object]) -> dict[str, str | None]:
    lowered = normalize_text(question)
    if _looks_like_all_time_query(lowered):
        return {
            "time_scope": "all_time",
            "issue_date": None,
            "start_date": None,
            "end_date": None,
        }
    explicit_date = _extract_date(question)
    if explicit_date:
        return {
            "time_scope": "single_issue",
            "issue_date": explicit_date,
            "start_date": explicit_date,
            "end_date": explicit_date,
        }
    year_range = _extract_year_range(lowered)
    if year_range:
        start_date, end_date = year_range
        return {
            "time_scope": "date_range",
            "issue_date": None,
            "start_date": start_date,
            "end_date": end_date,
        }
    prior_issue_date = str(session_context.get("last_issue_date") or "") or None
    prior_start_date = str(session_context.get("last_start_date") or "") or None
    prior_end_date = str(session_context.get("last_end_date") or "") or None
    prior_time_scope = str(session_context.get("last_time_scope") or "") or None
    if prior_time_scope == "date_range" and (prior_start_date or prior_end_date):
        return {
            "time_scope": "date_range",
            "issue_date": None,
            "start_date": prior_start_date,
            "end_date": prior_end_date,
        }
    if issue_date or prior_issue_date:
        resolved = issue_date or prior_issue_date
        return {
            "time_scope": "single_issue",
            "issue_date": resolved,
            "start_date": resolved,
            "end_date": resolved,
        }
    return {
        "time_scope": "all_time",
        "issue_date": None,
        "start_date": None,
        "end_date": None,
    }


def _best_catalog_match(question: str, catalog: list[str]) -> str | None:
    lowered = normalize_text(question)
    best: str | None = None
    best_score = 0.0
    for candidate in catalog:
        candidate_lower = candidate.lower()
        if candidate_lower in lowered:
            return candidate
        tokens = [token for token in re.split(r"[^a-z0-9]+", candidate_lower) if len(token) >= 4]
        if tokens and any(token in lowered for token in tokens):
            score = max(SequenceMatcher(a=token, b=lowered).ratio() for token in tokens)
            if score > best_score:
                best = candidate
                best_score = score
    if best:
        return best
    lowered_tokens = [token for token in re.findall(r"[a-z0-9]+", lowered) if len(token) >= 4]
    for token in lowered_tokens:
        match = get_close_matches(token, catalog, n=1, cutoff=0.92)
        if match:
            return match[0]
    return None


def _resolve_catalog_value(value: object, catalog: list[str]) -> str | None:
    if not value or not catalog:
        return None
    candidate = str(value).strip()
    if not candidate:
        return None
    if candidate in catalog:
        return candidate
    matched = _best_catalog_match(candidate, catalog)
    if matched:
        return matched
    return None


def _extract_filter_phrase(question: str, kind: str) -> str | None:
    lowered = normalize_text(question)
    if kind == "edition":
        patterns = [
            r"\b([a-z0-9][a-z0-9\s&'.-]{1,40})\s+edition\b",
            r"\bpublished in\s+([a-z0-9][a-z0-9\s&'.-]{1,40})\b",
        ]
    else:
        patterns = [
            r"\b([a-z0-9][a-z0-9\s&'.-]{1,40})\s+section\b",
            r"\b([a-z0-9][a-z0-9\s&'.-]{1,40})\s+page\b",
        ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if not match:
            continue
        value = re.sub(r"\s+", " ", match.group(1)).strip(" .,-")
        for marker in [" in ", " from ", " for ", " about "]:
            if marker in value:
                value = value.rsplit(marker, 1)[-1].strip(" .,-")
        value = re.sub(r"^(the|a|an)\s+", "", value).strip()
        if value:
            return value.title() if len(value.split()) <= 4 else value
    return None


def _should_extract_author_filter(question: str) -> bool:
    lowered = normalize_text(question)
    cues = [
        "written by",
        "wrote",
        "write about",
        "write on",
        "author",
        "by ",
        "reported by",
        "column by",
        "articles by",
    ]
    return any(cue in lowered for cue in cues)


def _should_extract_edition_filter(question: str) -> bool:
    lowered = normalize_text(question)
    if any(cue in lowered for cue in [" edition", "published in"]):
        return True
    publications = fetch_publication_catalog()
    publication_names = [row["publication_name"] for row in publications]
    return _contains_catalog_phrase(lowered, publication_names, require_context={"from", "in", "edition", "published"})


def _should_extract_section_filter(question: str) -> bool:
    lowered = normalize_text(question)
    if " section" in lowered or "front page" in lowered:
        return True
    return _contains_catalog_phrase(lowered, fetch_section_catalog(), require_context={"section", "page"})


def _extract_date(question: str) -> str | None:
    match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", question)
    if match:
        return match.group(1)
    month_date = re.search(
        r"\b(?:on\s+)?([A-Za-z]{3,9}\s+\d{1,2},?\s+20\d{2}|\d{1,2}\s+[A-Za-z]{3,9}\s+20\d{2})\b",
        question,
    )
    if month_date:
        return parse_calendar_date(month_date.group(1))
    return None


def _looks_like_all_time_query(lowered: str) -> bool:
    return any(
        phrase in lowered
        for phrase in [
            "in the archive",
            "in our archive",
            "over the years",
            "all time",
            "ever written",
            "across the archive",
        ]
    )


def _extract_year_range(lowered: str) -> tuple[str, str] | None:
    year_match = re.search(r"\b(in|during|throughout|across)\s+(20\d{2})\b", lowered)
    if year_match:
        year = year_match.group(2)
        return f"{year}-01-01", f"{year}-12-31"
    current_year = date.today().year
    if "this year" in lowered:
        return f"{current_year}-01-01", f"{current_year}-12-31"
    if "last year" in lowered:
        year = current_year - 1
        return f"{year}-01-01", f"{year}-12-31"
    return None


def _resolve_entity_terms(planner_payload: dict, question: str) -> list[str]:
    entity_terms = planner_payload.get("entity_terms") or []
    if isinstance(entity_terms, str):
        entity_terms = [entity_terms]
    resolved: list[str] = []
    for term in entity_terms:
        if not term:
            continue
        expanded = expand_person_alias_terms(term) if _looks_like_person_name(term) else [term]
        for value in expanded:
            if value and value.lower() not in {existing.lower() for existing in resolved}:
                resolved.append(value)
        parts = [part.strip() for part in str(term).split() if part.strip()]
        if len(parts) >= 2 and _looks_like_person_name(term):
            surname = parts[-1]
            if surname.lower() not in {existing.lower() for existing in resolved}:
                resolved.append(surname)
    resolved.extend(_generic_entity_candidates(question))
    if resolved:
        return dedupe_preserve_order(resolved)
    return _emergency_entity_terms(question)


def _resolve_mode(preferred: str | None, needs_count: bool, needs_summary: bool, needs_listing: bool, entity_terms: list[str], filters: dict) -> str:
    if filters.get("author"):
        return "sql"
    if preferred in {"sql", "semantic", "hybrid"}:
        if needs_count and preferred == "semantic":
            return "hybrid"
        if preferred == "semantic" and needs_summary and entity_terms:
            return "hybrid"
        return preferred
    if needs_count and (needs_summary or entity_terms):
        return "hybrid"
    if needs_count or filters.get("author") or filters.get("edition") or filters.get("section"):
        return "sql"
    if needs_summary and entity_terms:
        return "hybrid"
    if needs_listing:
        return "hybrid"
    return "semantic"


def _select_tools(
    proposed,
    needs_count: bool,
    needs_summary: bool,
    needs_listing: bool,
    needs_article_text: bool,
    mode: str,
    *,
    task_type: str,
    has_entities: bool,
    has_author: bool,
) -> list[str]:
    valid = {"structured_count", "structured_articles", "semantic_chunks", "headline_keyword", "story_clusters"}
    tools = [tool for tool in (proposed or []) if tool in valid]
    if task_type == "ranking":
        return ["structured_count"]
    if not tools:
        if needs_count:
            tools.append("structured_count")
        if needs_listing or needs_article_text:
            tools.append("structured_articles")
        elif has_author:
            tools.append("structured_articles")
        if needs_summary:
            if (mode == "hybrid" or has_entities or has_author) and "structured_articles" not in tools:
                tools.append("structured_articles")
            tools.extend(["semantic_chunks", "story_clusters"])
        elif mode != "sql":
            tools.extend(["headline_keyword", "semantic_chunks"])
    if needs_count and "structured_count" not in tools:
        tools.insert(0, "structured_count")
    if needs_summary and not needs_count and (mode == "hybrid" or has_entities or has_author) and "structured_articles" not in tools:
        tools.insert(0, "structured_articles")
    if has_author and mode == "sql":
        ordered: list[str] = []
        if needs_count:
            ordered.append("structured_count")
        if needs_summary or needs_listing or needs_article_text:
            ordered.append("structured_articles")
        if not ordered:
            ordered.append("structured_articles")
        return dedupe_preserve_order(ordered)
    if needs_listing and "structured_articles" not in tools:
        tools.append("structured_articles")
    if needs_summary and "story_clusters" not in tools:
        tools.append("story_clusters")
    if mode != "sql" and "semantic_chunks" not in tools and needs_summary:
        tools.append("semantic_chunks")
    return dedupe_preserve_order(tools)


def _select_fallbacks(proposed, retrieval_tools: list[str], needs_count: bool, mode: str) -> list[str]:
    valid = {"structured_count", "structured_articles", "semantic_chunks", "headline_keyword", "story_clusters"}
    fallback = [tool for tool in (proposed or []) if tool in valid]
    if not fallback:
        fallback = ["headline_keyword", "semantic_chunks", "structured_articles"]
        if needs_count:
            fallback.insert(0, "structured_count")
        elif mode == "sql":
            fallback.insert(0, "structured_articles")
    return [tool for tool in dedupe_preserve_order(fallback) if tool not in retrieval_tools] + [tool for tool in retrieval_tools if tool not in fallback]


def _emergency_query_focus(question: str, entity_terms: list[str]) -> str:
    if entity_terms:
        return " ".join(entity_terms[:4])
    lowered = normalize_text(question)
    lowered = re.sub(r"\b(how|many|what|show|list|articles|article|the|a|an|context|key|points)\b", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip() or normalize_text(question)


def _is_unsupported_or_injected(question: str) -> bool:
    lowered = question.lower()
    return any(marker in lowered for marker in INJECTION_MARKERS)


def _emergency_entity_terms(question: str) -> list[str]:
    candidates = _generic_entity_candidates(question)
    if candidates:
        return candidates
    matches = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", question)
    return dedupe_preserve_order(matches)


def _emergency_plan_defaults(question: str) -> dict:
    lowered = normalize_text(question)
    count_like = any(token in lowered for token in ["how many", "count", "number of", "amny"])
    ranking_like = _question_requests_ranking(question)
    compare_like = _question_requests_compare(question)
    article_text_like = any(token in lowered for token in ["full text", "article text", "show article"])
    list_like = any(token in lowered for token in ["show me", "list", "which articles"])
    summary_like = any(token in lowered for token in ["key points", "context", "summary", "what were they about", "what was the key points", "abput"])
    if ranking_like:
        return {
            "task_type": "ranking",
            "answer_shape": "ranking",
            "intent": "lookup",
            "query_focus": question,
            "entity_terms": _emergency_entity_terms(question),
            "retrieval_tools": ["structured_count"],
            "fallback_order": ["structured_articles"],
            "reasoning": "Emergency fallback inferred ranking request.",
        }
    if article_text_like:
        return {
            "task_type": "article_text",
            "answer_shape": "article_text",
            "intent": "lookup",
            "query_focus": question,
            "entity_terms": _emergency_entity_terms(question),
            "retrieval_tools": ["structured_articles", "semantic_chunks"],
            "fallback_order": ["headline_keyword", "story_clusters"],
            "reasoning": "Emergency fallback inferred article_text request.",
        }
    if compare_like:
        return {
            "task_type": "compare",
            "answer_shape": "compare",
            "intent": "lookup",
            "query_focus": _emergency_query_focus(question, _emergency_entity_terms(question)),
            "entity_terms": _emergency_entity_terms(question),
            "retrieval_tools": ["structured_articles", "semantic_chunks", "story_clusters"],
            "fallback_order": ["headline_keyword", "structured_count"],
            "reasoning": "Emergency fallback inferred compare request.",
        }
    if list_like:
        return {
            "task_type": "list",
            "answer_shape": "list",
            "intent": "lookup",
            "query_focus": question,
            "entity_terms": _emergency_entity_terms(question),
            "retrieval_tools": ["structured_articles", "headline_keyword"],
            "fallback_order": ["semantic_chunks", "story_clusters"],
            "reasoning": "Emergency fallback inferred list request.",
        }
    if count_like:
        return {
            "task_type": "count",
            "answer_shape": "count_plus_summary" if summary_like else "count_only",
            "intent": "topic_count",
            "query_focus": _emergency_query_focus(question, _emergency_entity_terms(question)),
            "entity_terms": _emergency_entity_terms(question),
            "retrieval_tools": ["structured_count", "story_clusters", "semantic_chunks"] if summary_like else ["structured_count"],
            "fallback_order": ["headline_keyword", "structured_articles", "semantic_chunks"],
            "reasoning": "Emergency fallback inferred count request.",
        }
    return {
        "task_type": "summary",
        "answer_shape": "summary",
        "intent": "lookup",
        "query_focus": _emergency_query_focus(question, _emergency_entity_terms(question)),
        "entity_terms": _emergency_entity_terms(question),
        "retrieval_tools": ["semantic_chunks", "story_clusters", "headline_keyword"],
        "fallback_order": ["structured_articles", "structured_count"],
        "reasoning": "Emergency fallback inferred summary request.",
    }


def _normalize_intent(value: str | None, task_type: str, answer_shape: str, emergency_intent: str) -> str:
    allowed = {"lookup", "article_count", "topic_count", "fact_lookup", "author_lookup", "author_count"}
    if value in allowed and not (
        value == "lookup" and (task_type == "count" or answer_shape in {"count_only", "count_plus_summary"})
    ):
        return value
    lowered = normalize_text(value or "")
    if "author" in lowered and "count" in lowered:
        return "author_count"
    if "author" in lowered:
        return "author_lookup"
    if task_type == "count" and answer_shape == "count_only":
        return "article_count"
    if task_type == "count":
        return "topic_count"
    if task_type == "article_text":
        return "fact_lookup"
    return emergency_intent


def _normalize_task_type(task_type: str, answer_shape: str, payload: dict) -> str:
    if answer_shape == "compare":
        return "compare"
    if answer_shape == "ranking":
        return "ranking"
    if answer_shape in {"count_only", "count_plus_summary"}:
        return "count"
    if answer_shape == "list":
        return "list"
    if answer_shape == "article_text":
        return "article_text"
    if isinstance(payload.get("needs_count"), bool) and payload.get("needs_count"):
        return "count"
    return task_type


def _question_requests_count(question: str) -> bool:
    lowered = normalize_text(question)
    return any(phrase in lowered for phrase in ["how many", "count", "number of", "amny"])


def _question_requests_ranking(question: str) -> bool:
    lowered = normalize_text(question)
    ranking_terms = {"most", "least", "top", "highest", "lowest"}
    subject_terms = {"section", "sections", "edition", "editions", "front page", "author", "authors", "writer", "writers"}
    if "which year" in lowered or "which years" in lowered:
        return True
    return any(term in lowered for term in ranking_terms) and any(term in lowered for term in subject_terms)


def _question_requests_compare(question: str) -> bool:
    lowered = normalize_text(question)
    return (
        lowered.startswith("compare ")
        or " compare " in lowered
        or " vs " in lowered
        or " versus " in lowered
        or "difference between " in lowered
    )


def _question_prefers_summary_lookup(question: str) -> bool:
    lowered = normalize_text(question)
    return any(
        phrase in lowered
        for phrase in [
            "is there any news about",
            "is there any update about",
            "any news about",
            "what was written about",
            "news about",
        ]
    )


def _question_requests_list(question: str) -> bool:
    lowered = normalize_text(question)
    return any(phrase in lowered for phrase in ["show me", "list", "which articles", "matching articles"])


def _question_requests_article_text(question: str) -> bool:
    lowered = normalize_text(question)
    return any(
        phrase in lowered
        for phrase in [
            "full text",
            "article text",
            "show article",
            "show me the article",
            "full article",
            "article body",
        ]
    )


def _contains_catalog_phrase(question: str, catalog: list[str], require_context: set[str]) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", question))
    if not tokens.intersection(require_context):
        return False
    normalized_question = normalize_text(question)
    for candidate in catalog:
        candidate_tokens = [token for token in re.findall(r"[a-z0-9]+", normalize_text(candidate)) if len(token) >= 4]
        if candidate_tokens and any(token in normalized_question for token in candidate_tokens):
            return True
    return False


def _generic_entity_candidates(question: str) -> list[str]:
    lowered = normalize_text(question)
    stop_words = {
        "how", "many", "what", "who", "show", "list", "is", "there", "any", "news", "about",
        "on", "the", "a", "an", "were", "was", "of", "for", "and", "context", "key", "points",
        "summary", "regarding", "around", "related", "to", "in", "from", "me", "please", "tell",
        "give", "find", "articles", "article", "stories", "story", "coverage",
        "can", "them", "their", "performance", "performances", "tournament",
    }
    candidates: list[str] = []
    patterns = [
        r"\b(?:about|on|around|regarding|related to)\s+([a-z0-9][a-z0-9\s&'.-]{2,80})",
        r"\b(?:news|articles|stories|coverage)\s+(?:about|on|around)\s+([a-z0-9][a-z0-9\s&'.-]{2,80})",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if not match:
            continue
        fragment = re.split(r"\b(?:with|and|in|from|for)\b", match.group(1), maxsplit=1)[0]
        normalized = _normalize_entity_fragment(fragment, stop_words)
        if normalized:
            candidates.extend(_entity_variants(normalized))
    capitalized = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", question)
    for value in capitalized:
        normalized = _normalize_entity_fragment(value, stop_words)
        if normalized:
            candidates.extend(_entity_variants(normalized))
    compare_patterns = [
        r"\bcompare\s+([a-z0-9][a-z0-9\s&'.-]{1,50})\s+and\s+([a-z0-9][a-z0-9\s&'.-]{1,50})",
        r"\bdifference between\s+([a-z0-9][a-z0-9\s&'.-]{1,50})\s+and\s+([a-z0-9][a-z0-9\s&'.-]{1,50})",
        r"\b([a-z0-9][a-z0-9\s&'.-]{1,50})\s+vs\s+([a-z0-9][a-z0-9\s&'.-]{1,50})",
        r"\b([a-z0-9][a-z0-9\s&'.-]{1,50})\s+versus\s+([a-z0-9][a-z0-9\s&'.-]{1,50})",
    ]
    for pattern in compare_patterns:
        match = re.search(pattern, lowered)
        if not match:
            continue
        for fragment in match.groups():
            normalized = _normalize_entity_fragment(fragment, stop_words)
            if normalized:
                candidates.extend(_entity_variants(normalized))
    return dedupe_preserve_order(candidates)


def _normalize_entity_fragment(fragment: str, stop_words: set[str]) -> str | None:
    cleaned = re.sub(r"[^a-z0-9\s&'.-]", " ", fragment.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,-")
    if not cleaned:
        return None
    parts = [part for part in cleaned.split() if part not in stop_words]
    if not parts:
        return None
    while len(parts) > 1 and parts[-1] in {"issue", "issues", "case", "cases", "report", "reports"}:
        parts.pop()
    if not parts:
        return None
    if len(parts) > 4:
        parts = parts[:4]
    return " ".join(parts)


def _entity_variants(value: str) -> list[str]:
    variants = [value]
    parts = [part for part in value.split() if part]
    if len(parts) >= 2 and _looks_like_person_name(value):
        variants.append(parts[-1])
    if len(parts) <= 3 and _looks_like_person_name(value):
        variants.extend(expand_person_alias_terms(" ".join(part.title() for part in parts)))
    return dedupe_preserve_order(variants)


def _looks_like_person_name(value: str) -> bool:
    parts = [part for part in re.findall(r"[A-Za-z]+", str(value or "")) if part]
    if not (2 <= len(parts) <= 3):
        return False
    generic_tails = {
        "tournament",
        "cup",
        "war",
        "section",
        "edition",
        "news",
        "coverage",
        "victory",
        "performance",
        "performances",
        "final",
    }
    return parts[-1].lower() not in generic_tails
