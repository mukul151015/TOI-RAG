from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
import re

from app.core.config import get_settings
from app.schemas import EvidenceBundle, EvidenceItem, RetrievalPlan
from app.services.rag_v3.common import headline_key, normalize_issue_date
from app.services.rag_v3.reranker import rerank_items
from app.services.repository import (
    fetch_articles_for_ids,
    fetch_author_article_count,
    fetch_author_article_count_in_range,
    fetch_author_counts,
    fetch_author_counts_in_range,
    fetch_author_articles,
    fetch_author_articles_in_range,
    fetch_entity_mention_articles,
    fetch_entity_mention_contexts,
    fetch_entity_mention_count,
    fetch_publication_counts,
    fetch_publication_counts_in_range,
    fetch_section_counts,
    fetch_section_counts_in_range,
    fetch_sql_article_count,
    fetch_sql_article_count_in_range,
    fetch_sql_articles,
    fetch_sql_articles_in_range,
    fetch_entity_mention_articles_in_range,
    fetch_entity_mention_contexts_in_range,
    fetch_entity_mention_year_counts_in_range,
    fetch_entity_mention_count_in_range,
    keyword_search,
    semantic_search,
)
from app.services.openai_client import chat_completion, embed_texts


_settings = get_settings()
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
    "what",
    "were",
    "about",
    "there",
    "many",
}


def _plan_date_window(plan: RetrievalPlan) -> tuple[str | None, str | None]:
    if getattr(plan, "time_scope", None) == "single_issue":
        issue_date = plan.issue_date or plan.start_date or plan.end_date
        return issue_date, issue_date
    start_date = getattr(plan, "start_date", None)
    end_date = getattr(plan, "end_date", None)
    if start_date or end_date:
        return start_date, end_date
    return None, None


def retrieve_with_tool(question: str, plan: RetrievalPlan, tool: str, limit: int) -> EvidenceBundle:
    if tool == "structured_count":
        return _structured_count(question, plan, limit)
    if tool == "structured_articles":
        return _structured_articles(question, plan, limit)
    if tool == "semantic_chunks":
        return _semantic_chunks(question, plan, limit)
    if tool == "headline_keyword":
        return _headline_keyword(question, plan, limit)
    if tool == "story_clusters":
        return _story_clusters(question, plan, limit)
    raise ValueError(f"Unknown retrieval tool: {tool}")


def merge_bundles(question: str, plan: RetrievalPlan, bundles: list[EvidenceBundle]) -> EvidenceBundle:
    merged_items: list[EvidenceItem] = []
    applied_tools: list[str] = []
    applied_fallbacks: list[str] = []
    notes: list[str] = []
    raw_filters: dict = {}
    seen: set[str] = set()
    mode = plan.mode
    confidence_scores: list[float] = []
    for bundle in bundles:
        raw_filters.update(bundle.raw_filters)
        applied_tools.extend(bundle.applied_tools)
        applied_fallbacks.extend(bundle.applied_fallbacks)
        notes.extend(bundle.notes)
        confidence_scores.append(bundle.retrieval_confidence)
        for item in bundle.items:
            key = headline_key(item.headline) or str(item.article_id or "")
            if key in seen:
                continue
            seen.add(key)
            merged_items.append(item)
    rerankable = any(item.source_type in {"semantic_chunks", "headline_keyword", "story_clusters"} for item in merged_items)
    reranked = False
    if rerankable and "structured_count" not in applied_tools:
        merged_items, reranked = rerank_items(plan.semantic_query or question, merged_items)
        if reranked:
            notes.append("Applied cross-encoder reranking to merged evidence.")
    if any(tool.startswith("structured") for tool in applied_tools) and any(
        tool in {"semantic_chunks", "headline_keyword", "story_clusters"} for tool in applied_tools
    ):
        mode = "hybrid"
    elif any(tool.startswith("structured") for tool in applied_tools):
        mode = "sql"
    elif any(tool in {"semantic_chunks", "headline_keyword", "story_clusters"} for tool in applied_tools):
        mode = "semantic"
    retrieval_confidence = round(sum(confidence_scores) / len(confidence_scores), 4) if confidence_scores else 0.0
    return EvidenceBundle(
        question=question,
        mode=mode,
        plan=plan,
        items=merged_items,
        raw_filters=raw_filters,
        retrieval_confidence=retrieval_confidence,
        applied_tools=applied_tools,
        applied_fallbacks=applied_fallbacks,
        notes=notes,
    )


def _structured_count(question: str, plan: RetrievalPlan, limit: int) -> EvidenceBundle:
    start_date, end_date = _plan_date_window(plan)
    if plan.task_type == "ranking":
        ranking_kind = _ranking_kind(question)
        if ranking_kind == "year":
            rows = fetch_entity_mention_year_counts_in_range(
                start_date,
                end_date,
                plan.entity_terms,
                edition=plan.edition,
                section=plan.section,
                headline_priority_only=False,
            )
            top_rows = rows[:limit]
            items = [
                _item_from_row(
                    {
                        "id": f"year:{row.get('year')}",
                        "external_article_id": f"year:{row.get('year')}",
                        "headline": str(row.get("year") or "Unknown year"),
                        "issue_date": plan.issue_date,
                        "excerpt": f"{row.get('year') or 'Unknown year'} had {int(row.get('article_count') or 0)} relevant articles.",
                        "year_article_count": int(row.get("article_count") or 0),
                    },
                    source_type="structured_count",
                )
                for row in top_rows
            ]
            return _bundle(
                question,
                plan,
                items,
                count=int(top_rows[0].get("article_count") or 0) if top_rows else 0,
                contexts=[],
                tool="structured_count",
                confidence=1.0,
                extra_filters={"year_counts": rows, "ranking_kind": "year"},
            )
        if ranking_kind == "author":
            rows = fetch_author_counts(plan.issue_date) if start_date == end_date else fetch_author_counts_in_range(start_date, end_date)
            top_rows = rows[:limit]
            items = [
                _item_from_row(
                    {
                        "id": f"author:{row.get('author')}",
                        "external_article_id": f"author:{row.get('author')}",
                        "headline": row.get("author") or "Unknown author",
                        "author": row.get("author"),
                        "issue_date": plan.issue_date,
                        "excerpt": f"{row.get('author') or 'Unknown author'} wrote {int(row.get('article_count') or 0)} articles.",
                        "author_article_count": int(row.get("article_count") or 0),
                    },
                    source_type="structured_count",
                )
                for row in top_rows
            ]
            return _bundle(
                question,
                plan,
                items,
                count=int(top_rows[0].get("article_count") or 0) if top_rows else 0,
                contexts=[],
                tool="structured_count",
                confidence=1.0,
                extra_filters={"author_counts": rows, "ranking_kind": "author"},
            )
        if ranking_kind == "edition":
            rows = (
                fetch_publication_counts(plan.issue_date)
                if start_date == end_date
                else fetch_publication_counts_in_range(start_date, end_date)
            )
            top_rows = rows[:limit]
            items = [
                _item_from_row(
                    {
                        "id": f"publication:{row.get('publication_name')}",
                        "external_article_id": f"publication:{row.get('publication_name')}",
                        "headline": row.get("publication_name") or "Unknown edition",
                        "edition": row.get("publication_name"),
                        "issue_date": plan.issue_date,
                        "excerpt": f"{row.get('publication_name') or 'Unknown edition'} had {int(row.get('article_count') or 0)} articles.",
                        "publication_article_count": int(row.get("article_count") or 0),
                    },
                    source_type="structured_count",
                )
                for row in top_rows
            ]
            return _bundle(
                question,
                plan,
                items,
                count=int(top_rows[0].get("article_count") or 0) if top_rows else 0,
                contexts=[],
                tool="structured_count",
                confidence=1.0,
                extra_filters={"publication_counts": rows, "ranking_kind": "edition"},
            )
        rows = fetch_section_counts(plan.issue_date) if start_date == end_date else fetch_section_counts_in_range(start_date, end_date)
        top_rows = rows[:limit]
        items = [
            _item_from_row(
                {
                    "id": f"section:{row.get('section')}",
                    "external_article_id": f"section:{row.get('section')}",
                    "headline": row.get("section") or "Unknown section",
                    "section": row.get("section"),
                    "issue_date": plan.issue_date,
                    "excerpt": f"{row.get('section') or 'Unknown section'} had {int(row.get('article_count') or 0)} articles.",
                    "section_article_count": int(row.get("article_count") or 0),
                },
                source_type="structured_count",
            )
            for row in top_rows
        ]
        return _bundle(
            question,
            plan,
            items,
            count=int(top_rows[0].get("article_count") or 0) if top_rows else 0,
            contexts=[],
            tool="structured_count",
            confidence=1.0,
            extra_filters={"section_counts": rows, "ranking_kind": "section"},
        )
    if plan.intent in {"author_count", "author_lookup"} and plan.author:
        if start_date == end_date:
            rows = fetch_author_articles(plan.issue_date, plan.author, limit, edition=plan.edition, section=plan.section)
            count = fetch_author_article_count(plan.issue_date, plan.author, edition=plan.edition, section=plan.section)
        else:
            rows = fetch_author_articles_in_range(start_date, end_date, plan.author, limit, edition=plan.edition, section=plan.section)
            count = fetch_author_article_count_in_range(start_date, end_date, plan.author, edition=plan.edition, section=plan.section)
        items = [_item_from_row({**row, "author_article_count": count}, source_type="structured_count") for row in rows]
        return _bundle(question, plan, items, count=count, contexts=[], tool="structured_count", confidence=1.0)
    if plan.entity_terms:
        # Topic/person counts use broad mentions so users get archive-level coverage, not just headline-only matches.
        if start_date == end_date:
            rows = fetch_entity_mention_articles(
                plan.issue_date,
                plan.entity_terms,
                limit,
                edition=plan.edition,
                section=plan.section,
                headline_priority_only=False,
            )
            count = fetch_entity_mention_count(
                plan.issue_date,
                plan.entity_terms,
                edition=plan.edition,
                section=plan.section,
                headline_priority_only=False,
            )
            contexts = fetch_entity_mention_contexts(
                plan.issue_date,
                plan.entity_terms,
                edition=plan.edition,
                section=plan.section,
                limit=5,
                headline_priority_only=False,
            )
        else:
            rows = fetch_entity_mention_articles_in_range(
                start_date,
                end_date,
                plan.entity_terms,
                limit,
                edition=plan.edition,
                section=plan.section,
                headline_priority_only=False,
            )
            count = fetch_entity_mention_count_in_range(
                start_date,
                end_date,
                plan.entity_terms,
                edition=plan.edition,
                section=plan.section,
                headline_priority_only=False,
            )
            contexts = fetch_entity_mention_contexts_in_range(
                start_date,
                end_date,
                plan.entity_terms,
                edition=plan.edition,
                section=plan.section,
                limit=5,
                headline_priority_only=False,
            )
        items = [_item_from_row(row, source_type="structured_count") for row in rows]
        return _bundle(
            question,
            plan,
            items,
            count=count,
            contexts=contexts,
            tool="structured_count",
            confidence=1.0,
            extra_filters={"count_scope": "broad_mentions"},
        )
    if start_date == end_date:
        rows = fetch_sql_articles(plan.issue_date, plan.edition, plan.section, limit)
        count = fetch_sql_article_count(plan.issue_date, plan.edition, plan.section)
    else:
        rows = fetch_sql_articles_in_range(start_date, end_date, plan.edition, plan.section, limit)
        count = fetch_sql_article_count_in_range(start_date, end_date, plan.edition, plan.section)
    items = [_item_from_row(row, source_type="structured_count") for row in rows]
    return _bundle(question, plan, items, count=count, contexts=[], tool="structured_count", confidence=1.0)


def _structured_articles(question: str, plan: RetrievalPlan, limit: int) -> EvidenceBundle:
    start_date, end_date = _plan_date_window(plan)
    if plan.author:
        rows = (
            fetch_author_articles(plan.issue_date, plan.author, limit, edition=plan.edition, section=plan.section)
            if start_date == end_date
            else fetch_author_articles_in_range(start_date, end_date, plan.author, limit, edition=plan.edition, section=plan.section)
        )
    elif plan.entity_terms:
        rows = (
            fetch_entity_mention_articles(
                plan.issue_date,
                plan.entity_terms,
                limit,
                edition=plan.edition,
                section=plan.section,
                headline_priority_only=False,
            )
            if start_date == end_date
            else fetch_entity_mention_articles_in_range(
                start_date,
                end_date,
                plan.entity_terms,
                limit,
                edition=plan.edition,
                section=plan.section,
                headline_priority_only=False,
            )
        )
    else:
        rows = (
            fetch_sql_articles(plan.issue_date, plan.edition, plan.section, limit)
            if start_date == end_date
            else fetch_sql_articles_in_range(start_date, end_date, plan.edition, plan.section, limit)
        )
    items = [_item_from_row(row, source_type="structured_articles") for row in rows]
    return _bundle(question, plan, items, tool="structured_articles", confidence=0.95)


def _ranking_kind(question: str) -> str:
    lowered = question.lower()
    if "year" in lowered or "years" in lowered:
        return "year"
    if any(token in lowered for token in ["author", "authors", "writer", "writers"]):
        return "author"
    if "edition" in lowered:
        return "edition"
    return "section"


def _semantic_chunks(question: str, plan: RetrievalPlan, limit: int) -> EvidenceBundle:
    start_date, end_date = _plan_date_window(plan)
    semantic_queries = _semantic_queries(question, plan)
    embeddings = embed_texts(semantic_queries)
    vector_rows = []
    vector_failed = False
    for embedding in embeddings:
        try:
            vector_rows.extend(
                semantic_search(
                    embedding,
                    plan.issue_date,
                    plan.edition,
                    plan.section,
                    min(max(limit * 4, 20), 200),
                    start_date=start_date,
                    end_date=end_date,
                )
            )
        except Exception:
            vector_failed = True
            break
    article_ids = [row["article_id"] for row in vector_rows]
    article_rows = {row["id"]: row for row in fetch_articles_for_ids(article_ids)}
    ranked_rows = _rank_native_semantic_rows(vector_rows, article_rows, semantic_queries, plan.entity_terms)
    items = [
        _item_from_row(
            {
                **article_rows[row["article_id"]],
                "similarity": row["similarity"],
                "matched_chunk": row.get("chunk_text"),
            },
            source_type="semantic_chunks",
        )
        for row in ranked_rows[:limit]
        if row["article_id"] in article_rows
    ]
    confidence = _confidence_from_rows(ranked_rows)
    bundle = _bundle(question, plan, items, tool="semantic_chunks", confidence=confidence)
    bundle.mode = "semantic" if items else plan.mode
    bundle.raw_filters.update(
        {
            "time_scope": plan.time_scope,
            "issue_date": plan.issue_date,
            "start_date": plan.start_date,
            "end_date": plan.end_date,
            "edition": plan.edition,
            "section": plan.section,
            "vector_search_failed": vector_failed,
        }
    )
    return bundle


def _headline_keyword(question: str, plan: RetrievalPlan, limit: int) -> EvidenceBundle:
    start_date, end_date = _plan_date_window(plan)
    rows = []
    for semantic_query in _semantic_queries(question, plan):
        rows.extend(
            keyword_search(
                semantic_query,
                plan.issue_date,
                plan.edition,
                plan.section,
                limit,
                start_date=start_date,
                end_date=end_date,
            )
        )
    article_rows = {row["id"]: row for row in fetch_articles_for_ids([row["article_id"] for row in rows])} if rows else {}
    items = []
    for row in rows:
        article = article_rows.get(row["article_id"])
        if not article:
            continue
        items.append(_item_from_row({**article, "similarity": row.get("similarity", 0.45)}, source_type="headline_keyword"))
    return _bundle(question, plan, items, tool="headline_keyword", confidence=0.45 if items else 0.0)


def _story_clusters(question: str, plan: RetrievalPlan, limit: int) -> EvidenceBundle:
    source_bundle = _structured_articles(question, plan, max(limit * 2, 10))
    grouped: dict[str, list[EvidenceItem]] = defaultdict(list)
    for item in source_bundle.items:
        grouped[headline_key(item.headline) or str(item.article_id)].append(item)
    cluster_items: list[EvidenceItem] = []
    for cluster in grouped.values():
        item = cluster[0]
        item.metadata = {
            **item.metadata,
            "cluster_size": len(cluster),
            "cluster_headlines": [member.headline for member in cluster],
        }
        cluster_items.append(item)
    cluster_items.sort(key=lambda item: int((item.metadata or {}).get("cluster_size", 1)), reverse=True)
    return _bundle(question, plan, cluster_items[:limit], tool="story_clusters", confidence=0.7 if cluster_items else 0.0)


def _semantic_queries(question: str, plan: RetrievalPlan) -> list[str]:
    queries = [plan.semantic_query or question]
    if plan.entity_terms:
        queries.append(" ".join(plan.entity_terms[:4]))
    if _should_use_hyde(plan, question):
        hyde = _generate_hyde_query(question)
        if hyde:
            queries.append(hyde)
    deduped: list[str] = []
    seen: set[str] = set()
    for value in queries:
        cleaned = re.sub(r"\s+", " ", value).strip()
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            deduped.append(cleaned)
    return deduped


def _should_use_hyde(plan: RetrievalPlan, question: str) -> bool:
    if not _settings.hyde_enabled:
        return False
    if plan.task_type in {"count", "ranking", "article_text"}:
        return False
    if plan.author or plan.section or plan.edition:
        return False
    if len(plan.entity_terms) >= 2:
        return False
    lowered = question.lower()
    if any(token in lowered for token in ["about", "news about", "written about", "is there any news about"]):
        return False
    return True


def _generate_hyde_query(question: str) -> str | None:
    try:
        hyde_system = (
            "You are a Times of India journalist. Given a user query, write a plausible "
            "2-sentence news excerpt that would answer it. Be factual and concise. "
            "Output only the excerpt."
        )
        result = chat_completion(hyde_system, question, model=_settings.openai_chat_model, timeout=12.0)
        if result and len(result.strip()) > 20:
            return result.strip()
    except Exception:
        return None
    return None


def _rank_native_semantic_rows(vector_rows: list[dict], article_rows: dict[int, dict], semantic_queries: list[str], entity_terms: list[str]) -> list[dict]:
    best_by_article: dict[int, dict] = {}
    semantic_terms = _semantic_terms(semantic_queries, entity_terms)
    for row in vector_rows:
        article_id = row["article_id"]
        article = article_rows.get(article_id)
        if not article:
            continue
        overlap_count = _overlap_count(article, row.get("chunk_text", ""), semantic_terms)
        entity_bonus = _entity_bonus(article, row.get("chunk_text", ""), entity_terms)
        phrase_bonus = _phrase_overlap_bonus(article, row.get("chunk_text", ""), semantic_queries)
        similarity = float(row.get("similarity", 0.0))
        ranking_score = similarity + min(overlap_count * 0.02, 0.18) + min(entity_bonus, 0.22) + min(phrase_bonus, 0.12)
        candidate = {**row, "ranking_score": ranking_score, "overlap_count": overlap_count}
        current = best_by_article.get(article_id)
        if not current or candidate["ranking_score"] > current["ranking_score"]:
            best_by_article[article_id] = candidate
    filtered = [
        item
        for item in best_by_article.values()
        if _is_relevant_match(item, article_rows.get(item["article_id"]), semantic_queries, entity_terms)
    ]
    ranked = sorted(filtered, key=lambda item: (item["ranking_score"], item.get("similarity", 0.0)), reverse=True)
    reranked = _rerank_semantic_candidates(" ".join(semantic_queries[:2]), ranked, article_rows)
    return reranked if reranked else ranked


def _confidence_from_rows(rows: list[dict]) -> float:
    top = rows[:5]
    if not top:
        return 0.0
    scores = [float(row.get("similarity", 0.0)) for row in top]
    return round(sum(scores) / len(scores), 4)


def _semantic_terms(semantic_queries: list[str], entity_terms: list[str]) -> set[str]:
    terms: set[str] = set()
    for query in [*semantic_queries, *entity_terms]:
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
    return sum(1 for term in terms if _contains_term_match(haystack, term))


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
    for query in semantic_queries:
        query_lower = query.lower()
        if query_lower and query_lower in haystack:
            bonus += 0.18
    return min(bonus, 0.36)


def _entity_bonus(article: dict | None, chunk_text: str, entity_terms: list[str]) -> float:
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
    for term in entity_terms:
        if term and _contains_term_match(haystack, term.lower()):
            bonus += 0.18
    return min(bonus, 0.5)


def _is_relevant_match(row: dict, article: dict | None, semantic_queries: list[str], entity_terms: list[str]) -> bool:
    similarity = float(row.get("similarity", 0.0))
    overlap_count = int(row.get("overlap_count", 0))
    ranking_score = float(row.get("ranking_score", similarity))
    if _fails_topic_guard(article, row.get("chunk_text", ""), semantic_queries, entity_terms) and similarity < 0.55:
        return False
    if similarity >= 0.68:
        return True
    if ranking_score >= 0.72:
        return True
    if overlap_count >= 2 and similarity >= 0.32:
        return True
    if overlap_count >= 1 and entity_terms and similarity >= 0.42:
        return True
    return False


def _fails_topic_guard(article: dict | None, chunk_text: str, semantic_queries: list[str], entity_terms: list[str]) -> bool:
    if not entity_terms:
        return False
    lead_text = " ".join(
        [
            str((article or {}).get("headline") or ""),
            str((article or {}).get("excerpt") or "")[:220],
            str(chunk_text or "")[:220],
        ]
    ).lower()
    return not any(_contains_term_match(lead_text, term.lower()) for term in entity_terms)


def _contains_term_match(haystack: str, term: str) -> bool:
    if not haystack or not term:
        return False
    if term in haystack:
        return True
    candidate = term.strip().lower()
    if len(candidate) < 5:
        return False
    hay_tokens = {token for token in re.findall(r"[a-z0-9]+", haystack.lower()) if len(token) >= 4}
    term_tokens = [token for token in re.findall(r"[a-z0-9]+", candidate) if len(token) >= 4]
    if not term_tokens:
        return False
    for needle in term_tokens:
        if needle in hay_tokens:
            return True
        for token in hay_tokens:
            if token[0] != needle[0]:
                continue
            if abs(len(token) - len(needle)) > 2:
                continue
            if SequenceMatcher(a=token, b=needle).ratio() >= 0.83:
                return True
    return False


def _rerank_semantic_candidates(query: str, rows: list[dict], article_rows: dict[int, dict]) -> list[dict]:
    if not _settings.reranking_enabled or len(rows) < 2:
        return rows
    try:
        from app.services.reranker import rerank  # noqa: PLC0415

        candidate_texts = []
        for row in rows:
            article = article_rows.get(row["article_id"], {})
            candidate_texts.append(
                " ".join(
                    part
                    for part in [
                        str(article.get("headline") or ""),
                        str(article.get("excerpt") or ""),
                        str(row.get("chunk_text") or ""),
                        str(article.get("section") or ""),
                        str(article.get("edition") or ""),
                    ]
                    if part
                )
            )
        reranked_indices = rerank(query, candidate_texts)
        return [rows[index] for index in reranked_indices if 0 <= index < len(rows)]
    except Exception:
        return rows


def _item_from_row(row: dict, *, source_type: str) -> EvidenceItem:
    metadata = dict(row)
    metadata["issue_date"] = normalize_issue_date(metadata.get("issue_date"))
    return EvidenceItem(
        article_id=str(row.get("external_article_id") or row.get("id") or row.get("article_id") or ""),
        headline=row.get("headline"),
        edition=row.get("edition"),
        section=row.get("section"),
        issue_date=normalize_issue_date(row.get("issue_date")),
        excerpt=row.get("excerpt") or row.get("matched_chunk"),
        score=float(row.get("similarity") or row.get("score") or 0.0),
        source_type=source_type,
        metadata=metadata,
    )


def _bundle(
    question: str,
    plan: RetrievalPlan,
    items: list[EvidenceItem],
    *,
    tool: str,
    confidence: float,
    count: int | None = None,
    contexts: list[str] | None = None,
    extra_filters: dict | None = None,
) -> EvidenceBundle:
    raw_filters = {
        "time_scope": plan.time_scope,
        "issue_date": plan.issue_date,
        "start_date": plan.start_date,
        "end_date": plan.end_date,
        "edition": plan.edition,
        "section": plan.section,
        "author": plan.author,
    }
    if count is not None:
        raw_filters["exact_article_count"] = count
    if contexts is not None:
        raw_filters["exact_contexts"] = contexts
    if extra_filters:
        raw_filters.update(extra_filters)
    return EvidenceBundle(
        question=question,
        mode="sql" if tool.startswith("structured") or tool == "story_clusters" else "semantic",
        plan=plan,
        items=items,
        raw_filters=raw_filters,
        retrieval_confidence=confidence,
        applied_tools=[tool],
        applied_fallbacks=[],
        notes=[f"Retrieved with tool: {tool}."],
    )
