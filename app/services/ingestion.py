from datetime import datetime
import json
import logging
from pathlib import Path
from urllib.parse import urlparse

import httpx
from openai import APIConnectionError

from app.core.config import get_settings
from app.db.database import get_connection
from app.services.chunking import chunk_text
from app.services.openai_client import embed_texts
from app.services.parser import parse_doc
from app.services.repository import (
    _ensure_author,
    _ensure_organization,
    _ensure_section,
    _article_embedding_is_current,
    _complete_ingestion_run,
    _fail_ingestion_run,
    _insert_rule_counts,
    _link_article_author,
    _mark_article_status,
    _should_skip_article_processing,
    _update_ingestion_checkpoint,
    _upsert_article,
    _upsert_article_body,
    _upsert_publication,
    _upsert_publication_issue,
    complete_ingestion_run,
    ensure_author,
    ensure_organization,
    ensure_section,
    get_latest_ingestion_run,
    get_resume_ingestion_run,
    get_article_resume_point,
    insert_rule_counts,
    insert_ingestion_run,
    link_article_author,
    mark_article_status,
    replace_article_chunks,
    upsert_article,
    upsert_article_body,
    upsert_issue,
    upsert_publication,
    upsert_publication_issue,
)


settings = get_settings()
logger = logging.getLogger(__name__)


def _split_author(value: str) -> tuple[str | None, str | None]:
    if "@" in value:
        return value.split("@", 1)[0].replace(".", " ").title(), value
    return value, None


async def ingest_feed(
    feed_url: str | None = None,
    feed_file: str | None = None,
    org_id: str | None = None,
    process_embeddings: bool = False,
    resume_from_article_id: int | None = None,
) -> dict:
    target_url = feed_url or settings.default_feed_url
    target_file = Path(feed_file or settings.default_feed_file)
    target_org = org_id or settings.default_org_id

    source_key = str(target_file.resolve()) if target_file.exists() else target_url
    latest_run = get_latest_ingestion_run(target_org, source_key)
    payload = None
    issue_name = None
    issue_date = None
    resume_run = None

    if latest_run and latest_run.get("raw_payload"):
        payload = latest_run["raw_payload"]
        issue_name = payload["issueName"]
        issue_date = datetime.fromisoformat(payload["fromDate"]).date()
        resume_run = get_resume_ingestion_run(target_org, source_key, issue_date)
        if resume_run:
            logger.info(
                "Using stored payload from ingestion_run_id=%s for resume",
                latest_run["id"],
            )

    if payload is None:
        if target_file.exists():
            payload = json.loads(target_file.read_text(encoding="utf-8"))
            source_key = str(target_file.resolve())
            logger.info("Loaded feed payload from local file=%s", source_key)
        else:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(target_url)
                response.raise_for_status()
                payload = response.json()
            source_key = target_url

        issue_name = payload["issueName"]
        issue_date = datetime.fromisoformat(payload["fromDate"]).date()
        resume_run = get_resume_ingestion_run(target_org, source_key, issue_date)

    logger.info("Starting ingest for org=%s source=%s", target_org, source_key)

    ingestion_run_id = insert_ingestion_run(target_org, source_key, issue_date, payload)
    issue_id = upsert_issue(target_org, issue_date, issue_name)

    stats_map = payload.get("publicationStats", {})
    docs_map = payload.get("rawDataByPublication", {})

    inserted_articles = 0
    embedded_articles = 0
    embedding_failures = 0
    searchable_articles = 0
    processed_since_checkpoint = 0
    section_cache: dict[tuple[str, str | None, str | None, str | None], int] = {}
    author_cache: dict[tuple[str | None, str | None], int] = {}
    pending_checkpoint: dict[str, str | int | None] | None = None
    start_publication_id = resume_run["checkpoint_publication_id"] if resume_run else None
    start_doc_index = resume_run["checkpoint_doc_index"] if resume_run else None
    manual_resume_external_article_id = None

    if resume_from_article_id is not None:
        manual_resume = get_article_resume_point(resume_from_article_id)
        if not manual_resume:
            raise ValueError(f"resume_from_article_id={resume_from_article_id} not found")
        start_publication_id = manual_resume["publication_id"]
        start_doc_index = None
        manual_resume_external_article_id = manual_resume["external_article_id"]
        logger.info(
            "Manual resume requested from article_id=%s publication=%s external_article_id=%s",
            resume_from_article_id,
            start_publication_id,
            manual_resume_external_article_id,
        )

    resume_mode = start_publication_id is not None and start_doc_index is not None
    manual_resume_mode = start_publication_id is not None and manual_resume_external_article_id is not None

    if resume_mode:
        logger.info(
            "Resuming ingest run_id=%s from publication=%s doc_index=%s",
            ingestion_run_id,
            start_publication_id,
            start_doc_index,
        )
    elif manual_resume_mode:
        logger.info(
            "Resuming ingest run_id=%s from article_id=%s publication=%s external_article_id=%s",
            ingestion_run_id,
            resume_from_article_id,
            start_publication_id,
            manual_resume_external_article_id,
        )

    def flush_checkpoint(force: bool = False) -> None:
        nonlocal processed_since_checkpoint, pending_checkpoint
        if not pending_checkpoint:
            return
        if not force and processed_since_checkpoint < settings.ingestion_checkpoint_interval:
            return
        _update_ingestion_checkpoint(
            cur,
            ingestion_run_id,
            checkpoint_publication_id=str(pending_checkpoint["checkpoint_publication_id"]),
            checkpoint_doc_index=int(pending_checkpoint["checkpoint_doc_index"]),
            last_processed_article_id=(
                int(pending_checkpoint["last_processed_article_id"])
                if pending_checkpoint["last_processed_article_id"] is not None
                else None
            ),
        )
        conn.commit()
        processed_since_checkpoint = 0
        pending_checkpoint = None

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                _ensure_organization(cur, target_org)
                for publication_id, publication_docs in docs_map.items():
                    if (resume_mode or manual_resume_mode) and publication_id != start_publication_id:
                        continue

                    docs = publication_docs.get("docs", [])
                    if docs:
                        publication_name = docs[0]["publication_name"]
                    else:
                        publication_name = publication_id

                    _upsert_publication(cur, target_org, publication_id, publication_name)
                    publication_issue_id = _upsert_publication_issue(
                        cur, issue_id, publication_id, stats_map.get(publication_id, {})
                    )

                    debug = stats_map.get(publication_id, {}).get("debug", {})
                    _insert_rule_counts(
                        cur, publication_issue_id, debug.get("acceptCounts", {}), "accept"
                    )
                    _insert_rule_counts(
                        cur, publication_issue_id, debug.get("rejectCounts", {}), "reject"
                    )
                    logger.info(
                        "Processing publication=%s docs=%s",
                        publication_id,
                        len(docs),
                    )

                    starting_index = start_doc_index if resume_mode and publication_id == start_publication_id else 0
                    found_manual_resume_article = not manual_resume_mode
                    for doc_index, raw_doc in enumerate(docs):
                        if doc_index < starting_index:
                            continue
                        if manual_resume_mode and publication_id == start_publication_id and not found_manual_resume_article:
                            if raw_doc.get("article_id") != manual_resume_external_article_id:
                                continue
                            found_manual_resume_article = True
                            logger.info(
                                "Matched manual resume point publication=%s doc_index=%s external_article_id=%s",
                                publication_id,
                                doc_index,
                                manual_resume_external_article_id,
                            )

                        parsed_doc = parse_doc(raw_doc)
                        section_key = (
                            parsed_doc.publication_id,
                            parsed_doc.zone,
                            parsed_doc.pagegroup,
                            parsed_doc.layoutdesk,
                        )
                        section_id = section_cache.get(section_key)
                        if section_id is None:
                            section_id = _ensure_section(
                                cur,
                                parsed_doc.publication_id,
                                parsed_doc.zone,
                                parsed_doc.pagegroup,
                                parsed_doc.layoutdesk,
                            )
                            section_cache[section_key] = section_id
                        article_id = _upsert_article(cur, publication_issue_id, section_id, parsed_doc)
                        inserted_articles += 1
                        logger.info(
                            "Upserted article id=%s external_article_id=%s headline=%s searchable=%s",
                            article_id,
                            parsed_doc.external_article_id,
                            parsed_doc.headline,
                            parsed_doc.is_searchable,
                        )
                        processed_since_checkpoint += 1

                        if _should_skip_article_processing(
                            cur,
                            article_id, parsed_doc.embedding_source_hash
                        ):
                            logger.info(
                                "Skipping article id=%s reason=already_processed",
                                article_id,
                            )
                            pending_checkpoint = {
                                "checkpoint_publication_id": publication_id,
                                "checkpoint_doc_index": doc_index + 1,
                                "last_processed_article_id": article_id,
                            }
                            flush_checkpoint()
                            continue

                        if parsed_doc.body_text:
                            _upsert_article_body(
                                cur,
                                article_id,
                                parsed_doc.body_text,
                                parsed_doc.cleaned_body_text,
                            )

                        for byline in parsed_doc.bylines:
                            display_name, email = _split_author(byline)
                            author_key = (display_name, email)
                            author_id = author_cache.get(author_key)
                            if author_id is None:
                                author_id = _ensure_author(cur, display_name, email)
                                author_cache[author_key] = author_id
                            _link_article_author(cur, article_id, author_id)

                        if not parsed_doc.is_searchable:
                            _mark_article_status(
                                cur,
                                article_id,
                                processing_status="processed",
                                embedding_status="skipped",
                                clear_last_error=True,
                            )
                            logger.info(
                                "Skipping embedding for article id=%s reason=not_searchable",
                                article_id,
                            )
                            pending_checkpoint = {
                                "checkpoint_publication_id": publication_id,
                                "checkpoint_doc_index": doc_index + 1,
                                "last_processed_article_id": article_id,
                            }
                            flush_checkpoint()
                            continue

                        searchable_articles += 1

                        if not process_embeddings:
                            logger.info(
                                "Leaving article id=%s for backfill reason=metadata_only_ingest",
                                article_id,
                            )
                            pending_checkpoint = {
                                "checkpoint_publication_id": publication_id,
                                "checkpoint_doc_index": doc_index + 1,
                                "last_processed_article_id": article_id,
                            }
                            flush_checkpoint()
                            continue

                        if _article_embedding_is_current(
                            cur,
                            article_id, parsed_doc.embedding_source_hash
                        ):
                            _mark_article_status(
                                cur,
                                article_id,
                                processing_status="processed",
                                embedding_status="embedded",
                                clear_last_error=True,
                            )
                            logger.info(
                                "Skipping embedding for article id=%s reason=duplicate_unchanged",
                                article_id,
                            )
                            pending_checkpoint = {
                                "checkpoint_publication_id": publication_id,
                                "checkpoint_doc_index": doc_index + 1,
                                "last_processed_article_id": article_id,
                            }
                            flush_checkpoint()
                            continue

                        chunks = chunk_text(
                            parsed_doc.embedding_text,
                            settings.chunk_size,
                            settings.chunk_overlap,
                            settings.chunk_overlap_sentences,
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
                                "Skipping embedding for article id=%s reason=no_chunks",
                                article_id,
                            )
                            pending_checkpoint = {
                                "checkpoint_publication_id": publication_id,
                                "checkpoint_doc_index": doc_index + 1,
                                "last_processed_article_id": article_id,
                            }
                            flush_checkpoint()
                            continue

                        try:
                            embeddings = embed_texts(chunks)
                        except APIConnectionError as exc:
                            embedding_failures += 1
                            _mark_article_status(
                                cur,
                                article_id,
                                processing_status="processed",
                                embedding_status="failed",
                                last_error=str(exc),
                            )
                            _fail_ingestion_run(cur, ingestion_run_id, str(exc))
                            logger.warning(
                                "Skipping embedding for article id=%s reason=openai_connection_error error=%s",
                                article_id,
                                exc,
                            )
                            raise
                        chunk_rows = [
                            {
                                "chunk_index": idx,
                                "chunk_text": chunk,
                                "embedding": embedding,
                                "token_count": max(1, len(chunk) // 4),
                            }
                            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
                        ]
                        replace_article_chunks(article_id, chunk_rows)
                        _mark_article_status(
                            cur,
                            article_id,
                            processing_status="processed",
                            embedding_status="embedded",
                            clear_last_error=True,
                        )
                        embedded_articles += 1
                        logger.info(
                            "Embedded article id=%s chunks=%s",
                            article_id,
                            len(chunk_rows),
                        )
                        pending_checkpoint = {
                            "checkpoint_publication_id": publication_id,
                            "checkpoint_doc_index": doc_index + 1,
                            "last_processed_article_id": article_id,
                        }
                        flush_checkpoint()

                    if resume_mode and publication_id == start_publication_id:
                        resume_mode = False
                        start_doc_index = None
                    if manual_resume_mode and publication_id == start_publication_id:
                        manual_resume_mode = False
                        manual_resume_external_article_id = None
                    flush_checkpoint(force=True)

                flush_checkpoint(force=True)
                _complete_ingestion_run(cur, ingestion_run_id)
    except Exception as exc:
        with get_connection() as conn:
            with conn.cursor() as cur:
                _fail_ingestion_run(cur, ingestion_run_id, str(exc))
        raise

    result = {
        "ok": True,
        "source": source_key if source_key != target_url else urlparse(target_url).geturl(),
        "ingestion_run_id": ingestion_run_id,
        "issue_id": issue_id,
        "inserted_articles": inserted_articles,
        "searchable_articles": searchable_articles,
        "process_embeddings": process_embeddings,
        "embedded_articles": embedded_articles,
        "embedding_failures": embedding_failures,
    }
    logger.info("Ingest completed result=%s", result)
    return result
