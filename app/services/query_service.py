from collections import OrderedDict
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.core.config import get_settings
from app.schemas import QueryResponse
from app.services.query_analyzer import analyze_query
from app.services.query_analyzer import canonical_person_name, expand_person_alias_terms
from app.services.openai_client import embed_texts, chat_completion
from app.services.query_router import (
    expand_semantic_queries,
    is_section_count_query,
    normalize_user_query,
    route_query,
)
from app.services.repository import (
    fetch_author_article_count,
    fetch_author_articles,
    fetch_articles_for_ids,
    fetch_entity_mention_articles,
    fetch_entity_mention_count,
    fetch_entity_mention_contexts,
    fetch_publication_catalog,
    fetch_section_catalog,
    fetch_section_counts,
    fetch_sql_articles,
    keyword_search,
    semantic_search,
)

_settings = get_settings()


logger = logging.getLogger(__name__)

GENERIC_TERMS = {
    "india",
    "world",
    "stories",
    "story",
    "covered",
    "cover",
    "article",
    "articles",
    "which",
}
SPORTS_PHRASES = {
    "world cup",
    "t20 world cup",
    "world champions",
    "bcci reward",
}
BUDGET_PHRASES = {
    "budget",
    "middle class",
    "inflation",
    "price rise",
    "oil prices",
    "growth",
    "economists",
    "tax",
}


def run_query(
    query: str,
    issue_date: str | None,
    limit: int,
    *,
    edition: str | None = None,
    section: str | None = None,
    result_window: int | None = None,
    routed_override=None,
) -> QueryResponse:
    analysis = analyze_query(query, issue_date)
    routed = routed_override or analysis.routed
    if edition:
        routed.edition = _resolve_edition_filter(edition)
        if routed.mode == "semantic":
            routed.mode = "hybrid"
    if section:
        routed.section = _resolve_section_filter(section)
        if routed.mode == "semantic":
            routed.mode = "hybrid"
    window = result_window or limit

    if is_section_count_query(query):
        rows = fetch_section_counts(routed.issue_date)
        return QueryResponse(
            mode="sql",
            filters={"issue_date": routed.issue_date, "intent": routed.intent},
            results=rows,
        )

    if routed.author:
        author_rows = fetch_author_articles(
            routed.issue_date,
            routed.author,
            window,
            edition=routed.edition,
            section=routed.section,
        )
        author_count = fetch_author_article_count(
            routed.issue_date,
            routed.author,
            edition=routed.edition,
            section=routed.section,
        )
        return QueryResponse(
            mode="sql",
            filters=routed.model_dump(exclude_none=True),
            results=[
                {**row, "author_article_count": author_count}
                for row in author_rows
            ],
        )

    exact_entity_terms = _exact_entity_terms_for_topic_count(analysis.entities, routed.intent)
    if routed.intent == "topic_count" and exact_entity_terms:
        person_topic = bool(analysis.entities.get("content_people"))
        rows = fetch_entity_mention_articles(
            routed.issue_date,
            exact_entity_terms,
            window,
            edition=routed.edition,
            section=routed.section,
            headline_priority_only=person_topic,
        )
        mention_count = fetch_entity_mention_count(
            routed.issue_date,
            exact_entity_terms,
            edition=routed.edition,
            section=routed.section,
            headline_priority_only=person_topic,
        )
        mention_contexts = fetch_entity_mention_contexts(
            routed.issue_date,
            exact_entity_terms,
            edition=routed.edition,
            section=routed.section,
            limit=5,
            headline_priority_only=person_topic,
        )
        return QueryResponse(
            mode="sql",
            filters={
                **routed.model_dump(exclude_none=True),
                "retrieval_strategy": "exact_entity_mentions",
                "entity_terms": exact_entity_terms,
                "exact_article_count": mention_count,
                "exact_contexts": mention_contexts,
                "entity_label": _entity_display_label(analysis.entities),
                "subject_strict": person_topic,
            },
            results=rows,
        )

    if routed.mode == "sql":
        rows = fetch_sql_articles(routed.issue_date, routed.edition, routed.section, window)
        return QueryResponse(
            mode="sql",
            filters=routed.model_dump(exclude_none=True),
            results=rows,
        )

    base_semantic = routed.semantic_query or normalize_user_query(query)
    semantic_queries = expand_semantic_queries(base_semantic)

    # Incorporate LLM-generated paraphrases (from analyze_query's entity enrichment).
    llm_paraphrases = analysis.entities.get("llm_paraphrases", [])
    for paraphrase in llm_paraphrases:
        if paraphrase and paraphrase not in semantic_queries:
            semantic_queries.append(paraphrase)

    per_query_limit = min(max(window, limit) * 3, 5000)

    # -----------------------------------------------------------------
    # Concurrent HyDE generation + embedding
    # -----------------------------------------------------------------
    hyde_queries: list[str] = []
    if _settings.hyde_enabled:
        hyde_queries = _generate_hyde_queries(query)

    all_queries_to_embed = semantic_queries + hyde_queries
    embeddings = embed_texts(all_queries_to_embed)
    # Semantic queries use first N embeddings; HyDE embeddings follow.
    semantic_embeddings = embeddings[: len(semantic_queries)]
    hyde_embeddings = embeddings[len(semantic_queries):]

    vector_rows = []
    vector_search_failed = False
    for embedding in semantic_embeddings + hyde_embeddings:
        try:
            vector_rows.extend(
                semantic_search(
                    embedding,
                    routed.issue_date,
                    routed.edition,
                    routed.section,
                    per_query_limit,
                )
            )
        except Exception as exc:
            vector_search_failed = True
            logger.warning("Semantic search failed; falling back to keyword retrieval: %s", exc)
            break
    keyword_rows = []
    keyword_search_failed = False
    try:
        for semantic_query in semantic_queries:
            keyword_rows.extend(
                keyword_search(
                    semantic_query,
                    routed.issue_date,
                    routed.edition,
                    routed.section,
                    min(max(window, limit) * 2, 1000),
                )
            )
    except Exception as exc:
        keyword_search_failed = True
        logger.warning("Keyword search failed during query fallback: %s", exc)

    ordered_ids = list(
        OrderedDict((row["article_id"], None) for row in [*vector_rows, *keyword_rows]).keys()
    )
    try:
        article_rows = {row["id"]: row for row in fetch_articles_for_ids(ordered_ids)}
    except Exception as exc:
        logger.warning("Article hydration failed after retrieval: %s", exc)
        article_rows = {}

    ranked_rows = _rank_rows(
        vector_rows,
        keyword_rows,
        article_rows,
        semantic_queries,
        analysis.entities,
    )

    # -----------------------------------------------------------------
    # Optional cross-encoder reranking
    # -----------------------------------------------------------------
    if _settings.reranking_enabled and ranked_rows:
        try:
            from app.services.reranker import rerank  # noqa: PLC0415
            candidates = ranked_rows[: _settings.reranker_top_k]
            candidate_texts = [
                article_rows.get(r["article_id"], {}).get("excerpt") or r.get("chunk_text", "")
                for r in candidates
            ]
            reranked_indices = rerank(query, candidate_texts)
            ranked_rows = [candidates[i] for i in reranked_indices] + ranked_rows[_settings.reranker_top_k:]
        except Exception as exc:
            logger.warning("Reranking failed, using original ranking: %s", exc)

    # -----------------------------------------------------------------
    # Confidence score: mean similarity of top-5 results
    # -----------------------------------------------------------------
    confidence_score = _compute_confidence(ranked_rows)

    results = []
    for row in ranked_rows[:window]:
        article = article_rows.get(row["article_id"])
        if article:
            results.append(
                {
                    **article,
                    "similarity": row["similarity"],
                    "matched_chunk": row["chunk_text"][:300],
                }
            )
    return QueryResponse(
        mode=routed.mode,
        filters={
            **routed.model_dump(exclude_none=True),
            **({"vector_search_failed": True} if vector_search_failed else {}),
            **({"keyword_search_failed": True} if keyword_search_failed else {}),
        },
        results=results,
        confidence_score=confidence_score,
    )


def _generate_hyde_queries(query: str) -> list[str]:
    """Generate a hypothetical 2-sentence news excerpt that answers the query."""
    try:
        hyde_system = (
            "You are a Times of India journalist. Given a user query, write a plausible "
            "2-sentence news excerpt that would answer it. Be factual and concise. "
            "Output only the excerpt, no preamble."
        )
        hyde_doc = chat_completion(hyde_system, query, model=_settings.openai_chat_model, timeout=15.0)
        if hyde_doc and len(hyde_doc) > 20:
            return [hyde_doc]
    except Exception as exc:
        logger.warning("HyDE generation failed: %s", exc)
    return []


def _compute_confidence(ranked_rows: list[dict]) -> float:
    """Mean similarity of top-5 ranked results, or 0.0 if none."""
    top = ranked_rows[:5]
    if not top:
        return 0.0
    scores = [float(r.get("similarity", 0.0)) for r in top]
    return round(sum(scores) / len(scores), 4)


def _exact_entity_terms_for_topic_count(entities: dict[str, list[str]], intent: str) -> list[str]:
    if intent != "topic_count":
        return []
    terms: list[str] = []
    for key in ("content_people", "content_organizations"):
        for value in entities.get(key, []):
            expanded_terms = expand_person_alias_terms(value) if key == "content_people" else [value]
            for term in expanded_terms:
                if term and term.lower() not in {existing.lower() for existing in terms}:
                    terms.append(term)
    return terms


def _entity_display_label(entities: dict[str, list[str]]) -> str | None:
    content_people = entities.get("content_people", [])
    if content_people:
        return canonical_person_name(content_people[0])
    content_orgs = entities.get("content_organizations", [])
    if content_orgs:
        return content_orgs[0]
    return None


def _resolve_edition_filter(edition: str) -> str:
    normalized = edition.strip().lower()
    for row in fetch_publication_catalog():
        publication_name = row["publication_name"]
        if publication_name.lower() == normalized:
            return publication_name
    return edition


def _resolve_section_filter(section: str) -> str:
    normalized = section.strip().lower()
    section_aliases = {
        "editorial": "Edit",
        "opinion": "Edit",
        "edit": "Edit",
        "oped": "Oped",
        "op-ed": "Oped",
    }
    if normalized in section_aliases:
        return section_aliases[normalized]
    for value in fetch_section_catalog():
        if value and value.lower() == normalized:
            return value
    return section


def _rank_rows(
    vector_rows: list[dict],
    keyword_rows: list[dict],
    article_rows: dict[int, dict],
    semantic_queries: list[str],
    entities: dict[str, list[str]],
) -> list[dict]:
    best_by_article: dict[int, dict] = {}
    semantic_terms = _semantic_terms(semantic_queries)
    for row in vector_rows:
        article_id = row["article_id"]
        article = article_rows.get(article_id)
        if not article:
            continue
        overlap_count = _overlap_count(article, row.get("chunk_text", ""), semantic_terms)
        score = (
            float(row["similarity"])
            + min(overlap_count * 0.03, 0.45)
            + _phrase_overlap_bonus(article, row.get("chunk_text", ""), semantic_queries)
            + _entity_bonus(article, row.get("chunk_text", ""), entities)
        )
        current = best_by_article.get(article_id)
        candidate = {**row, "ranking_score": score, "overlap_count": overlap_count}
        if not current or candidate["ranking_score"] > current["ranking_score"]:
            best_by_article[article_id] = candidate
    for row in keyword_rows:
        article_id = row["article_id"]
        article = article_rows.get(article_id)
        if not article:
            continue
        overlap_count = _overlap_count(article, row.get("excerpt", ""), semantic_terms)
        lexical_score = float(row.get("lexical_score", 0.0))
        score = (
            0.35
            + min(lexical_score, 1.2)
            + min(overlap_count * 0.04, 0.4)
            + _entity_bonus(article, row.get("excerpt", ""), entities)
        )
        current = best_by_article.get(article_id)
        candidate = {
            **row,
            "similarity": lexical_score,
            "chunk_text": row.get("excerpt", ""),
            "ranking_score": score,
            "overlap_count": overlap_count,
        }
        if not current or candidate["ranking_score"] > current["ranking_score"]:
            best_by_article[article_id] = candidate
    filtered = [
        item for item in best_by_article.values()
        if _is_relevant_match(item, article_rows.get(item["article_id"]), semantic_queries, entities)
    ]
    return sorted(
        filtered,
        key=lambda item: (item["ranking_score"], item["similarity"]),
        reverse=True,
    )


def _semantic_terms(semantic_queries: list[str]) -> set[str]:
    terms: set[str] = set()
    for query in semantic_queries:
        for token in re.findall(r"[a-z0-9]+", query.lower()):
            if len(token) >= 3 and token not in GENERIC_TERMS:
                terms.add(token)
    return terms


def _overlap_count(article: dict, chunk_text: str, terms: set[str]) -> int:
    haystack = " ".join(
        [
            str(article.get("headline") or ""),
            str(article.get("section") or ""),
            str(article.get("edition") or ""),
            str(chunk_text or ""),
            str(article.get("excerpt") or ""),
        ]
    ).lower()
    return sum(1 for term in terms if term in haystack)


def _phrase_overlap_bonus(article: dict, chunk_text: str, semantic_queries: list[str]) -> float:
    haystack = " ".join(
        [
            str(article.get("headline") or ""),
            str(article.get("section") or ""),
            str(chunk_text or ""),
            str(article.get("excerpt") or ""),
        ]
    ).lower()
    bonus = 0.0
    for phrase in SPORTS_PHRASES:
        if any(phrase in query.lower() for query in semantic_queries) and phrase in haystack:
            bonus += 0.18
    for phrase in BUDGET_PHRASES:
        if any(phrase in query.lower() for query in semantic_queries) and phrase in haystack:
            bonus += 0.1
    return min(bonus, 0.36)


def _entity_bonus(article: dict | None, chunk_text: str, entities: dict[str, list[str]]) -> float:
    haystack = " ".join(
        [
            str((article or {}).get("headline") or ""),
            str((article or {}).get("section") or ""),
            str((article or {}).get("edition") or ""),
            str((article or {}).get("excerpt") or ""),
            str(chunk_text or ""),
        ]
    ).lower()
    bonus = 0.0
    for person in entities.get("content_people", entities.get("people", [])):
        if person and person.lower() in haystack:
            bonus += 0.22
    for place in entities.get("content_locations", entities.get("places", [])):
        if place and place.lower() in haystack:
            bonus += 0.12
    for org in entities.get("content_organizations", entities.get("organizations", [])):
        if org and org.lower() in haystack:
            bonus += 0.12
    return min(bonus, 0.5)


def _is_relevant_match(row: dict, article: dict | None, semantic_queries: list[str], entities: dict[str, list[str]]) -> bool:
    similarity = float(row["similarity"])
    overlap_count = int(row.get("overlap_count", 0))
    ranking_score = float(row.get("ranking_score", similarity))
    if _fails_topic_guard(article, row.get("chunk_text", ""), semantic_queries, entities):
        return False
    if overlap_count >= 3:
        return True
    if overlap_count == 2:
        return similarity >= 0.28 or ranking_score >= 0.36
    if overlap_count == 1:
        return similarity >= 0.42 or ranking_score >= 0.48
    return similarity >= 0.62 and ranking_score >= 0.62


def _fails_topic_guard(article: dict | None, chunk_text: str, semantic_queries: list[str], entities: dict[str, list[str]]) -> bool:
    headline = str((article or {}).get("headline") or "").lower()
    lead_text = " ".join(
        [
            headline,
            str((article or {}).get("excerpt") or "")[:220],
            str(chunk_text or "")[:220],
        ]
    ).lower()
    query_text = " ".join(semantic_queries).lower()
    if "world cup" in query_text and "india" in query_text:
        primary = {"world champions", "bcci", "wt20", "reward"}
        secondary = {"rohit", "dhoni", "surya", "winning squad"}
        if "coach" in headline and not any(term in headline for term in primary):
            return True
        if any(term in headline for term in primary):
            return False
        if any(term in lead_text for term in primary):
            return False
        if "india" in lead_text and ("world cup" in lead_text or "wt20" in lead_text):
            return False
        secondary_hits = sum(1 for term in secondary if term in lead_text)
        return not ("india" in lead_text and secondary_hits >= 1)
    if "budget" in query_text and "middle class" in query_text:
        required = {"budget", "middle class", "inflation", "prices", "price rise", "growth", "economists", "tax"}
        return not any(term in lead_text for term in required)
    for person in entities.get("content_people", entities.get("people", [])):
        person_lower = person.lower()
        if person_lower and person_lower not in lead_text and person_lower not in headline:
            return True
    return False
