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
    keyword_search,
    semantic_search,
)

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
    keyword_rows = []
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

    ordered_ids = list(
        OrderedDict((row["article_id"], None) for row in [*vector_rows, *keyword_rows]).keys()
    )
    article_rows = {row["id"]: row for row in fetch_articles_for_ids(ordered_ids)}
    ranked_rows = _rank_rows(vector_rows, keyword_rows, article_rows, semantic_queries)
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
        score = 0.35 + min(lexical_score, 1.2) + min(overlap_count * 0.04, 0.4)
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
        if _is_relevant_match(item, article_rows.get(item["article_id"]), semantic_queries)
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


def _is_relevant_match(row: dict, article: dict | None, semantic_queries: list[str]) -> bool:
    similarity = float(row["similarity"])
    overlap_count = int(row.get("overlap_count", 0))
    ranking_score = float(row.get("ranking_score", similarity))
    if _fails_topic_guard(article, row.get("chunk_text", ""), semantic_queries):
        return False
    if overlap_count >= 3:
        return True
    if overlap_count == 2:
        return similarity >= 0.28 or ranking_score >= 0.36
    if overlap_count == 1:
        return similarity >= 0.42 or ranking_score >= 0.48
    return similarity >= 0.62 and ranking_score >= 0.62


def _fails_topic_guard(article: dict | None, chunk_text: str, semantic_queries: list[str]) -> bool:
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
    return False
