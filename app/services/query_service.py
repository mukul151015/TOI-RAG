from collections import OrderedDict
import re

from app.schemas import QueryResponse
from app.services.openai_client import embed_texts
from app.services.query_router import (
    expand_semantic_queries,
    is_section_count_query,
    normalize_user_query,
    route_query,
)
from app.services.repository import (
    fetch_articles_for_ids,
    fetch_publication_catalog,
    fetch_section_catalog,
    fetch_section_counts,
    fetch_sql_articles,
    semantic_search,
)


def run_query(
    query: str,
    issue_date: str | None,
    limit: int,
    *,
    edition: str | None = None,
    section: str | None = None,
    result_window: int | None = None,
) -> QueryResponse:
    routed = route_query(query, issue_date)
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
            filters={"issue_date": routed.issue_date},
            results=rows,
        )

    if routed.mode == "sql":
        rows = fetch_sql_articles(routed.issue_date, routed.edition, routed.section, window)
        return QueryResponse(
            mode="sql",
            filters=routed.model_dump(exclude_none=True),
            results=rows,
        )

    semantic_queries = expand_semantic_queries(routed.semantic_query or normalize_user_query(query))
    embeddings = embed_texts(semantic_queries)
    vector_rows = []
    per_query_limit = min(max(window, limit) * 3, 5000)
    for embedding in embeddings:
        vector_rows.extend(
            semantic_search(
                embedding,
                routed.issue_date,
                routed.edition,
                routed.section,
                per_query_limit,
            )
        )

    ordered_ids = list(OrderedDict((row["article_id"], None) for row in vector_rows).keys())
    article_rows = {row["id"]: row for row in fetch_articles_for_ids(ordered_ids)}
    ranked_rows = _rank_vector_rows(vector_rows, article_rows, semantic_queries)
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
        filters=routed.model_dump(exclude_none=True),
        results=results,
    )


def _resolve_edition_filter(edition: str) -> str:
    normalized = edition.strip().lower()
    for row in fetch_publication_catalog():
        publication_name = row["publication_name"]
        if publication_name.lower() == normalized:
            return publication_name
    return edition


def _resolve_section_filter(section: str) -> str:
    normalized = section.strip().lower()
    for value in fetch_section_catalog():
        if value and value.lower() == normalized:
            return value
    return section


def _rank_vector_rows(
    vector_rows: list[dict],
    article_rows: dict[int, dict],
    semantic_queries: list[str],
) -> list[dict]:
    best_by_article: dict[int, dict] = {}
    semantic_terms = _semantic_terms(semantic_queries)
    for row in vector_rows:
        article_id = row["article_id"]
        article = article_rows.get(article_id)
        if not article:
            continue
        overlap_count = _overlap_count(article, row.get("chunk_text", ""), semantic_terms)
        score = float(row["similarity"]) + min(overlap_count * 0.03, 0.45)
        current = best_by_article.get(article_id)
        candidate = {**row, "ranking_score": score, "overlap_count": overlap_count}
        if not current or candidate["ranking_score"] > current["ranking_score"]:
            best_by_article[article_id] = candidate
    filtered = [
        item for item in best_by_article.values()
        if _is_relevant_match(item)
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
            if len(token) >= 3:
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


def _is_relevant_match(row: dict) -> bool:
    similarity = float(row["similarity"])
    overlap_count = int(row.get("overlap_count", 0))
    ranking_score = float(row.get("ranking_score", similarity))
    if overlap_count >= 3:
        return True
    if overlap_count == 2:
        return similarity >= 0.28 or ranking_score >= 0.36
    if overlap_count == 1:
        return similarity >= 0.42 or ranking_score >= 0.48
    return similarity >= 0.62 and ranking_score >= 0.62
