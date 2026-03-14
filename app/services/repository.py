import json
from datetime import date
from functools import lru_cache
from typing import Any

from app.db.database import get_cursor


def _ensure_organization(cur, org_id: str) -> None:
    cur.execute(
        """
        insert into organizations (id, name)
        values (%s, %s)
        on conflict (id) do update set name = excluded.name
        """,
        (org_id, org_id.upper()),
    )


def ensure_organization(org_id: str) -> None:
    with get_cursor() as cur:
        _ensure_organization(cur, org_id)


def _upsert_publication(cur, org_id: str, publication_id: str, publication_name: str) -> None:
    cur.execute(
        """
        insert into publications (id, organization_id, publication_name)
        values (%s, %s, %s)
        on conflict (id) do update
        set organization_id = excluded.organization_id,
            publication_name = excluded.publication_name
        """,
        (publication_id, org_id, publication_name),
    )


def upsert_issue(org_id: str, issue_date: date, issue_name: str) -> int:
    with get_cursor() as cur:
        cur.execute(
            """
            insert into issues (organization_id, issue_date, issue_name)
            values (%s, %s, %s)
            on conflict (organization_id, issue_date, issue_name)
            do update set issue_name = excluded.issue_name
            returning id
            """,
            (org_id, issue_date, issue_name),
        )
        row = cur.fetchone()
        return row["id"]


def upsert_publication(org_id: str, publication_id: str, publication_name: str) -> None:
    with get_cursor() as cur:
        _upsert_publication(cur, org_id, publication_id, publication_name)


def _upsert_publication_issue(
    cur,
    issue_id: int,
    publication_id: str,
    stats: dict[str, Any],
) -> int:
    cur.execute(
        """
        insert into publication_issues (
          issue_id, publication_id, num_found, fetched, in_house, converted
        )
        values (%s, %s, %s, %s, %s, %s)
        on conflict (issue_id, publication_id) do update
        set num_found = excluded.num_found,
            fetched = excluded.fetched,
            in_house = excluded.in_house,
            converted = excluded.converted
        returning id
        """,
        (
            issue_id,
            publication_id,
            stats.get("numFound", 0),
            stats.get("fetched", 0),
            stats.get("inHouse", 0),
            stats.get("converted", 0),
        ),
    )
    return cur.fetchone()["id"]


def insert_ingestion_run(
    org_id: str,
    source_url: str,
    issue_date: date | None,
    raw_payload: dict[str, Any],
) -> int:
    existing_run = get_resume_ingestion_run(org_id, source_url, issue_date)
    if existing_run:
        with get_cursor() as cur:
            cur.execute(
                """
                update ingestion_runs
                set status = 'running',
                    last_error = null,
                    raw_payload = %s::jsonb
                where id = %s
                returning id
                """,
                (json.dumps(raw_payload), existing_run["id"]),
            )
            return cur.fetchone()["id"]

    with get_cursor() as cur:
        cur.execute(
            """
            insert into ingestion_runs (
              organization_id, source_url, issue_date, status, raw_payload
            )
            values (%s, %s, %s, 'running', %s::jsonb)
            returning id
            """,
            (org_id, source_url, issue_date, json.dumps(raw_payload)),
        )
        return cur.fetchone()["id"]


def get_resume_ingestion_run(
    org_id: str,
    source_url: str,
    issue_date: date | None,
) -> dict[str, Any] | None:
    with get_cursor() as cur:
        cur.execute(
            """
            select *
            from ingestion_runs
            where organization_id = %s
              and source_url = %s
              and issue_date is not distinct from %s
              and status in ('running', 'failed')
            order by id desc
            limit 1
            """,
            (org_id, source_url, issue_date),
        )
        return cur.fetchone()


def get_latest_ingestion_run(
    org_id: str,
    source_url: str,
) -> dict[str, Any] | None:
    with get_cursor() as cur:
        cur.execute(
            """
            select *
            from ingestion_runs
            where organization_id = %s
              and source_url = %s
            order by id desc
            limit 1
            """,
            (org_id, source_url),
        )
        return cur.fetchone()


def update_ingestion_checkpoint(
    ingestion_run_id: int,
    *,
    checkpoint_publication_id: str,
    checkpoint_doc_index: int,
    last_processed_article_id: int | None,
) -> None:
    with get_cursor() as cur:
        cur.execute(
            """
            update ingestion_runs
            set checkpoint_publication_id = %s,
                checkpoint_doc_index = %s,
                last_processed_article_id = %s,
                last_error = null
            where id = %s
            """,
            (
                checkpoint_publication_id,
                checkpoint_doc_index,
                last_processed_article_id,
                ingestion_run_id,
            ),
        )


def _update_ingestion_checkpoint(
    cur,
    ingestion_run_id: int,
    *,
    checkpoint_publication_id: str,
    checkpoint_doc_index: int,
    last_processed_article_id: int | None,
) -> None:
    cur.execute(
        """
        update ingestion_runs
        set checkpoint_publication_id = %s,
            checkpoint_doc_index = %s,
            last_processed_article_id = %s,
            last_error = null
        where id = %s
        """,
        (
            checkpoint_publication_id,
            checkpoint_doc_index,
            last_processed_article_id,
            ingestion_run_id,
        ),
    )


def complete_ingestion_run(ingestion_run_id: int) -> None:
    with get_cursor() as cur:
        cur.execute(
            """
            update ingestion_runs
            set status = 'completed',
                checkpoint_publication_id = null,
                checkpoint_doc_index = null,
                last_error = null
            where id = %s
            """,
            (ingestion_run_id,),
        )


def _complete_ingestion_run(cur, ingestion_run_id: int) -> None:
    cur.execute(
        """
        update ingestion_runs
        set status = 'completed',
            checkpoint_publication_id = null,
            checkpoint_doc_index = null,
            last_error = null
        where id = %s
        """,
        (ingestion_run_id,),
    )


def fail_ingestion_run(ingestion_run_id: int, error_message: str) -> None:
    with get_cursor() as cur:
        cur.execute(
            """
            update ingestion_runs
            set status = 'failed',
                last_error = %s
            where id = %s
            """,
            (error_message[:2000], ingestion_run_id),
        )


def _fail_ingestion_run(cur, ingestion_run_id: int, error_message: str) -> None:
    cur.execute(
        """
        update ingestion_runs
        set status = 'failed',
            last_error = %s
        where id = %s
        """,
        (error_message[:2000], ingestion_run_id),
    )


def upsert_publication_issue(
    issue_id: int,
    publication_id: str,
    stats: dict[str, Any],
) -> int:
    with get_cursor() as cur:
        return _upsert_publication_issue(cur, issue_id, publication_id, stats)


def _insert_rule_counts(
    cur,
    publication_issue_id: int,
    counts: dict[str, int],
    rule_kind: str,
) -> None:
    if not counts:
        return
    for rule_name, rule_count in counts.items():
        cur.execute(
            """
            insert into publication_issue_rule_counts (
              publication_issue_id, rule_kind, rule_name, rule_count
            )
            values (%s, %s, %s, %s)
            on conflict (publication_issue_id, rule_kind, rule_name) do update
            set rule_count = excluded.rule_count
            """,
            (publication_issue_id, rule_kind, rule_name, rule_count),
        )


def insert_rule_counts(
    publication_issue_id: int,
    counts: dict[str, int],
    rule_kind: str,
) -> None:
    with get_cursor() as cur:
        _insert_rule_counts(cur, publication_issue_id, counts, rule_kind)


def _ensure_section(
    cur,
    publication_id: str,
    zone: str | None,
    pagegroup: str | None,
    layoutdesk: str | None,
) -> int:
    cur.execute(
        """
        insert into sections (publication_id, zone, pagegroup, layoutdesk)
        values (%s, %s, %s, %s)
        on conflict (publication_id, zone, pagegroup, layoutdesk)
        do update set publication_id = excluded.publication_id
        returning id
        """,
        (publication_id, zone, pagegroup, layoutdesk),
    )
    return cur.fetchone()["id"]


def ensure_section(
    publication_id: str,
    zone: str | None,
    pagegroup: str | None,
    layoutdesk: str | None,
) -> int:
    with get_cursor() as cur:
        return _ensure_section(cur, publication_id, zone, pagegroup, layoutdesk)


def _upsert_article(
    cur,
    publication_issue_id: int,
    section_id: int | None,
    parsed_doc: Any,
) -> int:
    cur.execute(
        """
        insert into articles (
          publication_issue_id, external_article_id, external_doc_id, section_id,
          pageno, headline, deck, label, location, article_filename, status,
          is_searchable,
          processing_status, embedding_status, last_error,
          embedding_source_hash,
          issue_timestamp, updated_at, raw_json
        )
        values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
        on conflict (publication_issue_id, external_article_id)
        do update set
          external_doc_id = excluded.external_doc_id,
          section_id = excluded.section_id,
          pageno = excluded.pageno,
          headline = excluded.headline,
          deck = excluded.deck,
          label = excluded.label,
          location = excluded.location,
          article_filename = excluded.article_filename,
          status = excluded.status,
          is_searchable = excluded.is_searchable,
          processing_status = excluded.processing_status,
          embedding_status = case
            when articles.embedding_source_hash is not distinct from excluded.embedding_source_hash
              then articles.embedding_status
            else excluded.embedding_status
          end,
          last_error = case
            when articles.embedding_source_hash is not distinct from excluded.embedding_source_hash
              then articles.last_error
            else excluded.last_error
          end,
          embedding_source_hash = excluded.embedding_source_hash,
          issue_timestamp = excluded.issue_timestamp,
          updated_at = excluded.updated_at,
          raw_json = excluded.raw_json
        returning id
        """,
        (
            publication_issue_id,
            parsed_doc.external_article_id,
            parsed_doc.external_doc_id,
            section_id,
            parsed_doc.pageno,
            parsed_doc.headline,
            parsed_doc.deck,
            parsed_doc.label,
            parsed_doc.location,
            parsed_doc.article_filename,
            parsed_doc.status,
            parsed_doc.is_searchable,
            "processed",
            "pending" if parsed_doc.is_searchable else "skipped",
            None,
            parsed_doc.embedding_source_hash,
            parsed_doc.issue_date,
            parsed_doc.updated_at,
            json.dumps(parsed_doc.raw_json),
        ),
    )
    return cur.fetchone()["id"]


def upsert_article(
    publication_issue_id: int,
    section_id: int | None,
    parsed_doc: Any,
) -> int:
    with get_cursor() as cur:
        return _upsert_article(cur, publication_issue_id, section_id, parsed_doc)


def _upsert_article_body(cur, article_id: int, body_text: str, cleaned_text: str | None) -> None:
    cur.execute(
        """
        insert into article_bodies (article_id, body_text, cleaned_text)
        values (%s, %s, %s)
        on conflict (article_id) do update
        set body_text = excluded.body_text,
            cleaned_text = excluded.cleaned_text
        """,
        (article_id, body_text, cleaned_text),
    )


def mark_article_status(
    article_id: int,
    *,
    processing_status: str | None = None,
    embedding_status: str | None = None,
    last_error: str | None = None,
    clear_last_error: bool = False,
) -> None:
    assignments: list[str] = []
    params: list[Any] = []

    if processing_status is not None:
        assignments.append("processing_status = %s")
        params.append(processing_status)
    if embedding_status is not None:
        assignments.append("embedding_status = %s")
        params.append(embedding_status)
    if last_error is not None:
        assignments.append("last_error = %s")
        params.append(last_error)
    elif clear_last_error:
        assignments.append("last_error = null")

    if not assignments:
        return

    params.append(article_id)
    with get_cursor() as cur:
        cur.execute(
            f"""
            update articles
            set {", ".join(assignments)}
            where id = %s
            """,
            params,
        )


def _mark_article_status(
    cur,
    article_id: int,
    *,
    processing_status: str | None = None,
    embedding_status: str | None = None,
    last_error: str | None = None,
    clear_last_error: bool = False,
) -> None:
    assignments: list[str] = []
    params: list[Any] = []

    if processing_status is not None:
        assignments.append("processing_status = %s")
        params.append(processing_status)
    if embedding_status is not None:
        assignments.append("embedding_status = %s")
        params.append(embedding_status)
    if last_error is not None:
        assignments.append("last_error = %s")
        params.append(last_error)
    elif clear_last_error:
        assignments.append("last_error = null")

    if not assignments:
        return

    params.append(article_id)
    cur.execute(
        f"""
        update articles
        set {", ".join(assignments)}
        where id = %s
        """,
        params,
    )


def upsert_article_body(article_id: int, body_text: str, cleaned_text: str | None) -> None:
    with get_cursor() as cur:
        _upsert_article_body(cur, article_id, body_text, cleaned_text)


def _ensure_author(cur, display_name: str | None, email: str | None) -> int:
    cur.execute(
        """
        insert into authors (display_name, email)
        values (%s, %s)
        on conflict (email) do update
        set display_name = coalesce(excluded.display_name, authors.display_name)
        returning id
        """,
        (display_name, email),
    )
    return cur.fetchone()["id"]


def ensure_author(display_name: str | None, email: str | None) -> int:
    with get_cursor() as cur:
        return _ensure_author(cur, display_name, email)


def _link_article_author(cur, article_id: int, author_id: int) -> None:
    cur.execute(
        """
        insert into article_authors (article_id, author_id)
        values (%s, %s)
        on conflict do nothing
        """,
        (article_id, author_id),
    )


def link_article_author(article_id: int, author_id: int) -> None:
    with get_cursor() as cur:
        _link_article_author(cur, article_id, author_id)


def replace_article_chunks(article_id: int, chunks: list[dict[str, Any]]) -> None:
    with get_cursor() as cur:
        _replace_article_chunks(cur, article_id, chunks)


def _replace_article_chunks(cur, article_id: int, chunks: list[dict[str, Any]]) -> None:
    cur.execute("delete from article_chunks where article_id = %s", (article_id,))
    for item in chunks:
        cur.execute(
            """
            insert into article_chunks (
              article_id, chunk_index, chunk_text, embedding, token_count
            )
            values (%s, %s, %s, %s::vector, %s)
            """,
            (
                article_id,
                item["chunk_index"],
                item["chunk_text"],
                json.dumps(item["embedding"]),
                item["token_count"],
            ),
        )


def article_embedding_is_current(article_id: int, embedding_source_hash: str | None) -> bool:
    if not embedding_source_hash:
        return False
    with get_cursor() as cur:
        cur.execute(
            """
            select
              a.embedding_source_hash,
              exists (
                select 1 from article_chunks c where c.article_id = a.id
              ) as has_chunks
            from articles a
            where a.id = %s
            """,
            (article_id,),
        )
        row = cur.fetchone()
        if not row:
            return False
        return row["embedding_source_hash"] == embedding_source_hash and row["has_chunks"]


def _article_embedding_is_current(cur, article_id: int, embedding_source_hash: str | None) -> bool:
    if not embedding_source_hash:
        return False
    cur.execute(
        """
        select
          a.embedding_source_hash,
          exists (
            select 1 from article_chunks c where c.article_id = a.id
          ) as has_chunks
        from articles a
        where a.id = %s
        """,
        (article_id,),
    )
    row = cur.fetchone()
    if not row:
        return False
    return row["embedding_source_hash"] == embedding_source_hash and row["has_chunks"]


def should_skip_article_processing(article_id: int, embedding_source_hash: str | None) -> bool:
    if not embedding_source_hash:
        return False
    with get_cursor() as cur:
        cur.execute(
            """
            select
              processing_status,
              embedding_status,
              embedding_source_hash,
              exists (
                select 1 from article_chunks c where c.article_id = a.id
              ) as has_chunks
            from articles a
            where a.id = %s
            """,
            (article_id,),
        )
        row = cur.fetchone()
        if not row:
            return False
        return (
            row["processing_status"] == "processed"
            and row["embedding_status"] == "embedded"
            and row["embedding_source_hash"] == embedding_source_hash
            and row["has_chunks"]
        )


def _should_skip_article_processing(cur, article_id: int, embedding_source_hash: str | None) -> bool:
    if not embedding_source_hash:
        return False
    cur.execute(
        """
        select
          processing_status,
          embedding_status,
          embedding_source_hash,
          exists (
            select 1 from article_chunks c where c.article_id = a.id
          ) as has_chunks
        from articles a
        where a.id = %s
        """,
        (article_id,),
    )
    row = cur.fetchone()
    if not row:
        return False
    return (
        row["processing_status"] == "processed"
        and row["embedding_status"] == "embedded"
        and row["embedding_source_hash"] == embedding_source_hash
        and row["has_chunks"]
    )


def fetch_articles_for_embedding(
    start_article_id: int | None,
    limit: int,
    failed_only: bool,
) -> list[dict[str, Any]]:
    with get_cursor() as cur:
        cur.execute(
            """
            select
              a.id,
              a.external_article_id,
              a.headline,
              a.embedding_source_hash,
              a.embedding_status,
              ab.cleaned_text
            from articles a
            join article_bodies ab on ab.article_id = a.id
            where a.is_searchable = true
              and ab.cleaned_text is not null
              and (%s::bigint is null or a.id >= %s::bigint)
              and (
                (%s = true and a.embedding_status = 'failed')
                or
                (%s = false and a.embedding_status in ('pending', 'failed'))
              )
            order by a.id
            limit %s
            """,
            (start_article_id, start_article_id, failed_only, failed_only, limit),
        )
        return cur.fetchall()


def fetch_embedding_status_summary() -> dict[str, Any]:
    with get_cursor() as cur:
        cur.execute(
            """
            select embedding_status, count(*) as count
            from articles
            group by embedding_status
            """
        )
        counts = {row["embedding_status"]: row["count"] for row in cur.fetchall()}

        cur.execute(
            """
            select min(id) as first_failed_article_id
            from articles
            where embedding_status = 'failed'
            """
        )
        failed_row = cur.fetchone()

        cur.execute(
            """
            select min(id) as first_pending_article_id
            from articles
            where is_searchable = true
              and embedding_status = 'pending'
            """
        )
        pending_row = cur.fetchone()

    return {
        "counts": counts,
        "first_failed_article_id": failed_row["first_failed_article_id"],
        "first_pending_article_id": pending_row["first_pending_article_id"],
    }


def get_article_resume_point(article_id: int) -> dict[str, Any] | None:
    with get_cursor() as cur:
        cur.execute(
            """
            select
              a.id,
              a.external_article_id,
              p.id as publication_id
            from articles a
            join publication_issues pi on pi.id = a.publication_issue_id
            join publications p on p.id = pi.publication_id
            where a.id = %s
            """,
            (article_id,),
        )
        return cur.fetchone()


@lru_cache(maxsize=1)
def fetch_publication_catalog() -> list[dict[str, Any]]:
    with get_cursor() as cur:
        cur.execute(
            """
            select id, publication_name
            from publications
            order by publication_name
            """
        )
        return cur.fetchall()


@lru_cache(maxsize=1)
def fetch_section_catalog() -> list[str]:
    with get_cursor() as cur:
        cur.execute(
            """
            select distinct normalized_section
            from sections
            where normalized_section is not null
            order by normalized_section
            """
        )
        return [row["normalized_section"] for row in cur.fetchall()]


def fetch_sql_articles(issue_date: str | None, edition: str | None, section: str | None, limit: int):
    with get_cursor() as cur:
        cur.execute(
            """
            select
              a.id,
              a.external_article_id,
              a.headline,
              s.normalized_section as section,
              p.publication_name as edition,
              i.issue_date,
              left(ab.body_text, 280) as excerpt
            from articles a
            join publication_issues pi on pi.id = a.publication_issue_id
            join publications p on p.id = pi.publication_id
            join issues i on i.id = pi.issue_id
            left join sections s on s.id = a.section_id
            left join article_bodies ab on ab.article_id = a.id
            where (%s::date is null or i.issue_date = %s::date)
              and (%s::text is null or p.publication_name ilike '%%' || %s::text || '%%')
              and (%s::text is null or s.normalized_section ilike '%%' || %s::text || '%%')
            order by i.issue_date desc, p.publication_name, a.pageno nulls last, a.id
            limit %s
            """,
            (issue_date, issue_date, edition, edition, section, section, limit),
        )
        return cur.fetchall()


def fetch_section_counts(issue_date: str | None):
    with get_cursor() as cur:
        cur.execute(
            """
            select
              s.normalized_section as section,
              count(*) as article_count
            from articles a
            join publication_issues pi on pi.id = a.publication_issue_id
            join issues i on i.id = pi.issue_id
            left join sections s on s.id = a.section_id
            where (%s::date is null or i.issue_date = %s::date)
            group by s.normalized_section
            order by article_count desc, section nulls last
            """,
            (issue_date, issue_date),
        )
        return cur.fetchall()


def semantic_search(
    embedding: list[float],
    issue_date: str | None,
    edition: str | None,
    section: str | None,
    limit: int,
):
    with get_cursor() as cur:
        cur.execute(
            """
            select * from match_article_chunks_filtered(
              %s::vector, %s::date, %s, %s, %s
            )
            """,
            (json.dumps(embedding), issue_date, edition, section, limit),
        )
        return cur.fetchall()


def fetch_articles_for_ids(article_ids: list[int]) -> list[dict[str, Any]]:
    if not article_ids:
        return []
    with get_cursor() as cur:
        cur.execute(
            """
            select
              a.id,
              a.external_article_id,
              a.headline,
              s.normalized_section as section,
              p.publication_name as edition,
              i.issue_date,
              left(ab.body_text, 400) as excerpt
            from articles a
            join publication_issues pi on pi.id = a.publication_issue_id
            join publications p on p.id = pi.publication_id
            join issues i on i.id = pi.issue_id
            left join sections s on s.id = a.section_id
            left join article_bodies ab on ab.article_id = a.id
            where a.id = any(%s)
            """,
            (article_ids,),
        )
        return cur.fetchall()
