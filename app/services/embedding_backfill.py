from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from openai import APIConnectionError

from app.core.config import get_settings
from app.db.database import get_connection
from app.schemas import EmbeddingBackfillResponse, EmbeddingStatusResponse
from app.services.chunking import chunk_text
from app.services.openai_client import embed_texts
from app.services.repository import (
    _article_embedding_is_current,
    _mark_article_status,
    _replace_article_chunks,
    fetch_articles_for_embedding,
    fetch_embedding_status_summary,
)


settings = get_settings()
logger = logging.getLogger(__name__)


def backfill_embeddings(
    start_article_id: int | None = None,
    limit: int | None = None,
    worker_count: int | None = None,
    failed_only: bool = False,
) -> EmbeddingBackfillResponse:
    batch_limit = limit or settings.embedding_backfill_batch_size
    workers = worker_count or settings.embedding_worker_count
    current_start_id = start_article_id
    total_requested = 0
    results = {
        "embedded": 0,
        "skipped_current": 0,
        "skipped_not_searchable": 0,
        "failed": 0,
    }
    next_article_id = current_start_id

    while True:
        articles = fetch_articles_for_embedding(current_start_id, batch_limit, failed_only)
        if not articles:
            break

        total_requested += len(articles)
        next_article_id = articles[-1]["id"] + 1
        article_groups = _partition_articles(articles, workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_process_embedding_group, group) for group in article_groups if group]
            for future in as_completed(futures):
                group_results = future.result()
                for key, value in group_results.items():
                    results[key] += value
        current_start_id = next_article_id

    response = EmbeddingBackfillResponse(
        ok=True,
        requested=total_requested,
        embedded=results["embedded"],
        skipped_current=results["skipped_current"],
        skipped_not_searchable=results["skipped_not_searchable"],
        failed=results["failed"],
        start_article_id=start_article_id,
        next_article_id=next_article_id if total_requested > 0 else None,
    )
    logger.info("Embedding backfill completed result=%s", response.model_dump())
    return response


def get_embedding_status() -> EmbeddingStatusResponse:
    summary = fetch_embedding_status_summary()
    return EmbeddingStatusResponse(**summary)


def _partition_articles(articles: list[dict], workers: int) -> list[list[dict]]:
    if workers <= 1 or len(articles) <= 1:
        return [articles]
    groups: list[list[dict]] = [[] for _ in range(min(workers, len(articles)))]
    for idx, article in enumerate(articles):
        groups[idx % len(groups)].append(article)
    return groups


def _process_embedding_group(articles: list[dict]) -> dict[str, int]:
    results = {
        "embedded": 0,
        "skipped_current": 0,
        "skipped_not_searchable": 0,
        "failed": 0,
    }
    with get_connection() as conn:
        with conn.cursor() as cur:
            embeddable: list[tuple[dict, list[str]]] = []
            for article in articles:
                article_id = article["id"]
                headline = article["headline"]
                cleaned_text = article["cleaned_text"]
                embedding_source_hash = article["embedding_source_hash"]

                if not headline or not cleaned_text:
                    _mark_article_status(
                        cur,
                        article_id,
                        processing_status="processed",
                        embedding_status="skipped",
                        last_error="missing_headline_or_cleaned_text",
                    )
                    logger.info(
                        "Skipping embedding for article id=%s reason=not_searchable_backfill",
                        article_id,
                    )
                    results["skipped_not_searchable"] += 1
                    continue

                if _article_embedding_is_current(cur, article_id, embedding_source_hash):
                    _mark_article_status(
                        cur,
                        article_id,
                        processing_status="processed",
                        embedding_status="embedded",
                        clear_last_error=True,
                    )
                    logger.info("Skipping article id=%s reason=already_embedded", article_id)
                    results["skipped_current"] += 1
                    continue

                chunks = chunk_text(
                    f"{headline}\n{cleaned_text}",
                    settings.chunk_size,
                    settings.chunk_overlap,
                )
                if not chunks:
                    _mark_article_status(
                        cur,
                        article_id,
                        processing_status="processed",
                        embedding_status="skipped",
                        last_error="no_chunks",
                    )
                    logger.info(
                        "Skipping embedding for article id=%s reason=no_chunks_backfill",
                        article_id,
                    )
                    results["skipped_not_searchable"] += 1
                    continue

                _mark_article_status(
                    cur,
                    article_id,
                    processing_status="processed",
                    embedding_status="running",
                )
                embeddable.append((article, chunks))

            if not embeddable:
                return results

            # Batch OpenAI calls across multiple articles/chunks while preserving chunk order.
            for batch_start in range(0, len(embeddable), settings.embedding_batch_size):
                batch_articles = embeddable[batch_start : batch_start + settings.embedding_batch_size]
                flat_texts: list[str] = []
                chunk_counts: list[int] = []
                for _, chunks in batch_articles:
                    flat_texts.extend(chunks)
                    chunk_counts.append(len(chunks))

                try:
                    flat_embeddings = embed_texts(flat_texts)
                except APIConnectionError as exc:
                    for article, _ in batch_articles:
                        article_id = article["id"]
                        _mark_article_status(
                            cur,
                            article_id,
                            processing_status="processed",
                            embedding_status="failed",
                            last_error=str(exc),
                        )
                        logger.warning(
                            "Embedding failed for article id=%s reason=openai_connection_error error=%s",
                            article_id,
                            exc,
                        )
                        results["failed"] += 1
                    continue

                offset = 0
                for (article, chunks), chunk_count in zip(batch_articles, chunk_counts):
                    article_id = article["id"]
                    article_embeddings = flat_embeddings[offset : offset + chunk_count]
                    offset += chunk_count
                    chunk_rows = [
                        {
                            "chunk_index": idx,
                            "chunk_text": chunk,
                            "embedding": embedding,
                            "token_count": max(1, len(chunk) // 4),
                        }
                        for idx, (chunk, embedding) in enumerate(zip(chunks, article_embeddings))
                    ]
                    _replace_article_chunks(cur, article_id, chunk_rows)
                    _mark_article_status(
                        cur,
                        article_id,
                        processing_status="processed",
                        embedding_status="embedded",
                        clear_last_error=True,
                    )
                    logger.info(
                        "Embedded article id=%s chunks=%s via_backfill=true",
                        article_id,
                        len(chunk_rows),
                    )
                    results["embedded"] += 1
            conn.commit()
    return results
