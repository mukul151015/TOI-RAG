import json
from datetime import date
from functools import lru_cache
import re
from typing import Any

from app.db.database import get_cursor


_publication_catalog_fallback: list[dict[str, Any]] = []
_section_catalog_fallback: list[str] = []
_author_catalog_fallback: list[str] = []


def _is_plausible_author_name(value: str | None) -> bool:
    name = str(value or "").strip()
    if not name:
        return False
    if len(name) > 80:
        return False
    if name.startswith(("–", "—", "-")):
        return False
    if re.search(r"\d", name):
        return False
    alpha_tokens = re.findall(r"[A-Za-z]+", name)
    if len(alpha_tokens) >= 2:
        return True
    return bool(re.fullmatch(r"[A-Z]{2,6}", name))


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
    global _publication_catalog_fallback
    try:
        with get_cursor() as cur:
            cur.execute(
                """
                select id, publication_name
                from publications
                order by publication_name
                """
            )
            rows = cur.fetchall()
            _publication_catalog_fallback = rows
            return rows
    except Exception:
        return list(_publication_catalog_fallback)


@lru_cache(maxsize=1)
def fetch_section_catalog() -> list[str]:
    global _section_catalog_fallback
    try:
        with get_cursor() as cur:
            cur.execute(
                """
                select distinct normalized_section
                from sections
                where normalized_section is not null
                order by normalized_section
                """
            )
            rows = [row["normalized_section"] for row in cur.fetchall()]
            _section_catalog_fallback = rows
            return rows
    except Exception:
        return list(_section_catalog_fallback)


@lru_cache(maxsize=1)
def fetch_author_catalog() -> list[str]:
    global _author_catalog_fallback
    try:
        with get_cursor() as cur:
            cur.execute(
                """
                select distinct display_name
                from authors
                where display_name is not null
                  and btrim(display_name) <> ''
                order by display_name
                """
            )
            rows = [row["display_name"] for row in cur.fetchall() if _is_plausible_author_name(row["display_name"])]
            _author_catalog_fallback = rows
            return rows
    except Exception:
        return list(_author_catalog_fallback)


def fetch_sql_articles(issue_date: str | None, edition: str | None, section: str | None, limit: int):
    return fetch_sql_articles_in_range(issue_date, issue_date, edition, section, limit)


def fetch_sql_articles_in_range(start_date: str | None, end_date: str | None, edition: str | None, section: str | None, limit: int):
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
            where (%s::date is null or i.issue_date >= %s::date)
              and (%s::date is null or i.issue_date <= %s::date)
              and (%s::text is null or p.publication_name ilike '%%' || %s::text || '%%')
              and (%s::text is null or s.normalized_section ilike '%%' || %s::text || '%%')
              and coalesce(nullif(a.headline, ''), nullif(ab.cleaned_text, '')) is not null
              and (
                %s::text is distinct from 'Sports'
                or lower(coalesce(a.headline, '') || ' ' || left(coalesce(ab.cleaned_text, ''), 220)) ~
                  '(cricket|football|golf|tennis|hockey|ipl|bcci|coach|match|cup|trophy|champion|squad|player|olympic|medal|surya|rohit|dhoni|goal|league|wt20)'
              )
            order by i.issue_date desc, a.id desc
            limit %s
            """,
            (start_date, start_date, end_date, end_date, edition, edition, section, section, section, limit),
        )
        return cur.fetchall()


def fetch_sql_article_count(issue_date: str | None, edition: str | None, section: str | None) -> int:
    return fetch_sql_article_count_in_range(issue_date, issue_date, edition, section)


def fetch_sql_article_count_in_range(start_date: str | None, end_date: str | None, edition: str | None, section: str | None) -> int:
    with get_cursor() as cur:
        cur.execute(
            """
            select count(*) as article_count
            from articles a
            join publication_issues pi on pi.id = a.publication_issue_id
            join publications p on p.id = pi.publication_id
            join issues i on i.id = pi.issue_id
            left join sections s on s.id = a.section_id
            where (%s::date is null or i.issue_date >= %s::date)
              and (%s::date is null or i.issue_date <= %s::date)
              and (%s::text is null or p.publication_name ilike '%%' || %s::text || '%%')
              and (%s::text is null or s.normalized_section ilike '%%' || %s::text || '%%')
            """,
            (start_date, start_date, end_date, end_date, edition, edition, section, section),
        )
        return int(cur.fetchone()["article_count"])


def fetch_author_articles(
    issue_date: str | None,
    author: str,
    limit: int,
    *,
    edition: str | None = None,
    section: str | None = None,
):
    return fetch_author_articles_in_range(issue_date, issue_date, author, limit, edition=edition, section=section)


def fetch_author_articles_in_range(
    start_date: str | None,
    end_date: str | None,
    author: str,
    limit: int,
    *,
    edition: str | None = None,
    section: str | None = None,
):
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
              left(ab.body_text, 400) as excerpt,
              au.display_name as author
            from articles a
            join publication_issues pi on pi.id = a.publication_issue_id
            join publications p on p.id = pi.publication_id
            join issues i on i.id = pi.issue_id
            join article_authors aa on aa.article_id = a.id
            join authors au on au.id = aa.author_id
            left join sections s on s.id = a.section_id
            left join article_bodies ab on ab.article_id = a.id
            where (%s::date is null or i.issue_date >= %s::date)
              and (%s::date is null or i.issue_date <= %s::date)
              and (%s::text is null or p.publication_name ilike '%%' || %s::text || '%%')
              and (%s::text is null or s.normalized_section ilike '%%' || %s::text || '%%')
              and lower(au.display_name) = lower(%s)
            order by i.issue_date desc, a.id desc
            limit %s
            """,
            (start_date, start_date, end_date, end_date, edition, edition, section, section, author, limit),
        )
        return cur.fetchall()


def fetch_author_article_count(
    issue_date: str | None,
    author: str,
    *,
    edition: str | None = None,
    section: str | None = None,
) -> int:
    return fetch_author_article_count_in_range(issue_date, issue_date, author, edition=edition, section=section)


def fetch_author_article_count_in_range(
    start_date: str | None,
    end_date: str | None,
    author: str,
    *,
    edition: str | None = None,
    section: str | None = None,
) -> int:
    with get_cursor() as cur:
        cur.execute(
            """
            select count(*) as article_count
            from articles a
            join publication_issues pi on pi.id = a.publication_issue_id
            join publications p on p.id = pi.publication_id
            join issues i on i.id = pi.issue_id
            join article_authors aa on aa.article_id = a.id
            join authors au on au.id = aa.author_id
            left join sections s on s.id = a.section_id
            where (%s::date is null or i.issue_date >= %s::date)
              and (%s::date is null or i.issue_date <= %s::date)
              and (%s::text is null or p.publication_name ilike '%%' || %s::text || '%%')
              and (%s::text is null or s.normalized_section ilike '%%' || %s::text || '%%')
              and lower(au.display_name) = lower(%s)
            """,
            (start_date, start_date, end_date, end_date, edition, edition, section, section, author),
        )
        return int(cur.fetchone()["article_count"])


def fetch_author_counts(issue_date: str | None):
    return fetch_author_counts_in_range(issue_date, issue_date)


def fetch_author_counts_in_range(start_date: str | None, end_date: str | None):
    with get_cursor() as cur:
        cur.execute(
            """
            select
              au.display_name as author,
              count(*) as article_count
            from articles a
            join publication_issues pi on pi.id = a.publication_issue_id
            join issues i on i.id = pi.issue_id
            join article_authors aa on aa.article_id = a.id
            join authors au on au.id = aa.author_id
            where (%s::date is null or i.issue_date >= %s::date)
              and (%s::date is null or i.issue_date <= %s::date)
            group by au.display_name
            order by article_count desc, au.display_name asc
            """,
            (start_date, start_date, end_date, end_date),
        )
        return [row for row in cur.fetchall() if _is_plausible_author_name(row.get("author"))]


def fetch_entity_mention_articles(
    issue_date: str | None,
    entity_terms: list[str],
    limit: int,
    *,
    edition: str | None = None,
    section: str | None = None,
    headline_priority_only: bool = False,
):
    return fetch_entity_mention_articles_in_range(
        issue_date,
        issue_date,
        entity_terms,
        limit,
        edition=edition,
        section=section,
        headline_priority_only=headline_priority_only,
    )


def fetch_entity_mention_articles_in_range(
    start_date: str | None,
    end_date: str | None,
    entity_terms: list[str],
    limit: int,
    *,
    edition: str | None = None,
    section: str | None = None,
    headline_priority_only: bool = False,
):
    if not entity_terms:
        return []
    normalized_terms = [term.lower() for term in entity_terms if term]
    with get_cursor() as cur:
        cur.execute(
            """
            with scored_articles as (
              select
                a.id,
                a.external_article_id,
                a.headline,
                s.normalized_section as section,
                p.publication_name as edition,
                i.issue_date,
                left(ab.body_text, 400) as excerpt,
                max(
                  case
                    when lower(coalesce(a.headline, '')) like '%%' || term || '%%' then 6
                    when lower(left(coalesce(ab.cleaned_text, ab.body_text, ''), 320)) like '%%' || term || '%%' then 4
                    when lower(coalesce(ab.cleaned_text, ab.body_text, '')) like '%%' || term || '%%' then 1
                    else 0
                  end
                ) as mention_score
              from articles a
              join publication_issues pi on pi.id = a.publication_issue_id
              join publications p on p.id = pi.publication_id
              join issues i on i.id = pi.issue_id
              left join sections s on s.id = a.section_id
              left join article_bodies ab on ab.article_id = a.id
              cross join unnest(%s::text[]) term
              where (%s::date is null or i.issue_date >= %s::date)
                and (%s::date is null or i.issue_date <= %s::date)
                and (%s::text is null or p.publication_name ilike '%%' || %s::text || '%%')
                and (%s::text is null or s.normalized_section ilike '%%' || %s::text || '%%')
                and lower(
                  coalesce(a.headline, '') || ' ' || coalesce(ab.body_text, '') || ' ' || coalesce(ab.cleaned_text, '')
                ) like '%%' || term || '%%'
              group by
                a.id,
                a.external_article_id,
                a.headline,
                s.normalized_section,
                p.publication_name,
                i.issue_date,
                left(ab.body_text, 400)
            )
            select
              id,
              external_article_id,
              headline,
              section,
              edition,
              issue_date,
              excerpt
            from scored_articles
            where (%s = false or mention_score >= 6)
            order by mention_score desc, issue_date desc, id desc
            limit %s
            """,
            (
                normalized_terms,
                start_date,
                start_date,
                end_date,
                end_date,
                edition,
                edition,
                section,
                section,
                headline_priority_only,
                limit,
            ),
        )
        return cur.fetchall()


def fetch_entity_mention_count(
    issue_date: str | None,
    entity_terms: list[str],
    *,
    edition: str | None = None,
    section: str | None = None,
    headline_priority_only: bool = False,
) -> int:
    return fetch_entity_mention_count_in_range(
        issue_date,
        issue_date,
        entity_terms,
        edition=edition,
        section=section,
        headline_priority_only=headline_priority_only,
    )


def fetch_entity_mention_count_in_range(
    start_date: str | None,
    end_date: str | None,
    entity_terms: list[str],
    *,
    edition: str | None = None,
    section: str | None = None,
    headline_priority_only: bool = False,
) -> int:
    if not entity_terms:
        return 0
    normalized_terms = [term.lower() for term in entity_terms if term]
    with get_cursor() as cur:
        cur.execute(
            """
            with scored_articles as (
              select
                a.id,
                max(
                  case
                    when lower(coalesce(a.headline, '')) like '%%' || term || '%%' then 6
                    when lower(left(coalesce(ab.cleaned_text, ab.body_text, ''), 320)) like '%%' || term || '%%' then 4
                    when lower(coalesce(ab.cleaned_text, ab.body_text, '')) like '%%' || term || '%%' then 1
                    else 0
                  end
                ) as mention_score
              from articles a
              join publication_issues pi on pi.id = a.publication_issue_id
              join publications p on p.id = pi.publication_id
              join issues i on i.id = pi.issue_id
              left join sections s on s.id = a.section_id
              left join article_bodies ab on ab.article_id = a.id
              cross join unnest(%s::text[]) term
              where (%s::date is null or i.issue_date >= %s::date)
                and (%s::date is null or i.issue_date <= %s::date)
                and (%s::text is null or p.publication_name ilike '%%' || %s::text || '%%')
                and (%s::text is null or s.normalized_section ilike '%%' || %s::text || '%%')
                and lower(
                  coalesce(a.headline, '') || ' ' || coalesce(ab.body_text, '') || ' ' || coalesce(ab.cleaned_text, '')
                ) like '%%' || term || '%%'
              group by a.id
            )
            select count(*) as article_count
            from scored_articles
            where mention_score >= case when %s then 6 else 4 end
            """,
            (
                normalized_terms,
                start_date,
                start_date,
                end_date,
                end_date,
                edition,
                edition,
                section,
                section,
                headline_priority_only,
            ),
        )
        return int(cur.fetchone()["article_count"])


def fetch_entity_mention_contexts(
    issue_date: str | None,
    entity_terms: list[str],
    *,
    edition: str | None = None,
    section: str | None = None,
    limit: int = 5,
    headline_priority_only: bool = False,
) -> list[dict[str, Any]]:
    return fetch_entity_mention_contexts_in_range(
        issue_date,
        issue_date,
        entity_terms,
        edition=edition,
        section=section,
        limit=limit,
        headline_priority_only=headline_priority_only,
    )


def fetch_entity_mention_contexts_in_range(
    start_date: str | None,
    end_date: str | None,
    entity_terms: list[str],
    *,
    edition: str | None = None,
    section: str | None = None,
    limit: int = 5,
    headline_priority_only: bool = False,
) -> list[dict[str, Any]]:
    if not entity_terms:
        return []
    normalized_terms = [term.lower() for term in entity_terms if term]
    with get_cursor() as cur:
        cur.execute(
            """
            with scored_articles as (
              select
                a.id,
                a.headline,
                s.normalized_section as section,
                coalesce(ab.body_text, ab.cleaned_text, '') as article_text,
                max(
                  case
                    when lower(coalesce(a.headline, '')) like '%%' || term || '%%' then 6
                    when lower(left(coalesce(ab.cleaned_text, ab.body_text, ''), 320)) like '%%' || term || '%%' then 4
                    when lower(coalesce(ab.cleaned_text, ab.body_text, '')) like '%%' || term || '%%' then 1
                    else 0
                  end
                ) as mention_score
              from articles a
              join publication_issues pi on pi.id = a.publication_issue_id
              join publications p on p.id = pi.publication_id
              join issues i on i.id = pi.issue_id
              left join sections s on s.id = a.section_id
              left join article_bodies ab on ab.article_id = a.id
              cross join unnest(%s::text[]) term
              where (%s::date is null or i.issue_date >= %s::date)
                and (%s::date is null or i.issue_date <= %s::date)
                and (%s::text is null or p.publication_name ilike '%%' || %s::text || '%%')
                and (%s::text is null or s.normalized_section ilike '%%' || %s::text || '%%')
                and a.headline is not null
                and lower(
                  coalesce(a.headline, '') || ' ' || coalesce(ab.body_text, '') || ' ' || coalesce(ab.cleaned_text, '')
                ) like '%%' || term || '%%'
              group by a.id, a.headline, s.normalized_section, coalesce(ab.body_text, ab.cleaned_text, '')
            )
            select
              headline,
              count(*) as article_count,
              min(section) as section,
              left(min(article_text), 300) as excerpt
            from scored_articles
            where mention_score >= case when %s then 6 else 4 end
            group by headline
            order by max(mention_score) desc, article_count desc, headline
            limit %s
            """,
            (
                normalized_terms,
                start_date,
                start_date,
                end_date,
                end_date,
                edition,
                edition,
                section,
                section,
                headline_priority_only,
                limit,
            ),
        )
        return cur.fetchall()


def fetch_entity_mention_year_counts_in_range(
    start_date: str | None,
    end_date: str | None,
    entity_terms: list[str],
    *,
    edition: str | None = None,
    section: str | None = None,
    headline_priority_only: bool = False,
) -> list[dict[str, Any]]:
    if not entity_terms:
        return []
    normalized_terms = [term.lower() for term in entity_terms if term]
    with get_cursor() as cur:
        cur.execute(
            """
            with scored_articles as (
              select
                a.id,
                extract(year from i.issue_date)::int as coverage_year,
                max(
                  case
                    when lower(coalesce(a.headline, '')) like '%%' || term || '%%' then 6
                    when lower(left(coalesce(ab.cleaned_text, ab.body_text, ''), 320)) like '%%' || term || '%%' then 4
                    when lower(coalesce(ab.cleaned_text, ab.body_text, '')) like '%%' || term || '%%' then 1
                    else 0
                  end
                ) as mention_score
              from articles a
              join publication_issues pi on pi.id = a.publication_issue_id
              join publications p on p.id = pi.publication_id
              join issues i on i.id = pi.issue_id
              left join sections s on s.id = a.section_id
              left join article_bodies ab on ab.article_id = a.id
              cross join unnest(%s::text[]) term
              where (%s::date is null or i.issue_date >= %s::date)
                and (%s::date is null or i.issue_date <= %s::date)
                and (%s::text is null or p.publication_name ilike '%%' || %s::text || '%%')
                and (%s::text is null or s.normalized_section ilike '%%' || %s::text || '%%')
                and lower(
                  coalesce(a.headline, '') || ' ' || coalesce(ab.body_text, '') || ' ' || coalesce(ab.cleaned_text, '')
                ) like '%%' || term || '%%'
              group by a.id, extract(year from i.issue_date)
            )
            select
              coverage_year as year,
              count(*) as article_count
            from scored_articles
            where mention_score >= case when %s then 6 else 4 end
            group by coverage_year
            order by article_count desc, coverage_year desc
            """,
            (
                normalized_terms,
                start_date,
                start_date,
                end_date,
                end_date,
                edition,
                edition,
                section,
                section,
                headline_priority_only,
            ),
        )
        return cur.fetchall()


def fetch_matching_publications(issue_date: str | None, edition_term: str) -> list[dict[str, Any]]:
    return fetch_matching_publications_in_range(issue_date, issue_date, edition_term)


def fetch_matching_publications_in_range(start_date: str | None, end_date: str | None, edition_term: str) -> list[dict[str, Any]]:
    with get_cursor() as cur:
        cur.execute(
            """
            select
              p.publication_name,
              count(*) as article_count
            from articles a
            join publication_issues pi on pi.id = a.publication_issue_id
            join publications p on p.id = pi.publication_id
            join issues i on i.id = pi.issue_id
            where (%s::date is null or i.issue_date >= %s::date)
              and (%s::date is null or i.issue_date <= %s::date)
              and p.publication_name ilike '%%' || %s::text || '%%'
            group by p.publication_name
            order by article_count desc, p.publication_name
            """,
            (start_date, start_date, end_date, end_date, edition_term),
        )
        return cur.fetchall()


def fetch_section_counts(issue_date: str | None):
    return fetch_section_counts_in_range(issue_date, issue_date)


def fetch_section_counts_in_range(start_date: str | None, end_date: str | None):
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
            where (%s::date is null or i.issue_date >= %s::date)
              and (%s::date is null or i.issue_date <= %s::date)
            group by s.normalized_section
            order by article_count desc, section nulls last
            """,
            (start_date, start_date, end_date, end_date),
        )
        return cur.fetchall()


def fetch_publication_counts(issue_date: str | None):
    return fetch_publication_counts_in_range(issue_date, issue_date)


def fetch_publication_counts_in_range(start_date: str | None, end_date: str | None):
    with get_cursor() as cur:
        cur.execute(
            """
            select
              p.publication_name,
              count(*) as article_count
            from articles a
            join publication_issues pi on pi.id = a.publication_issue_id
            join publications p on p.id = pi.publication_id
            join issues i on i.id = pi.issue_id
            where (%s::date is null or i.issue_date >= %s::date)
              and (%s::date is null or i.issue_date <= %s::date)
            group by p.publication_name
            order by article_count desc, p.publication_name
            """,
            (start_date, start_date, end_date, end_date),
        )
        return cur.fetchall()


def semantic_search(
    embedding: list[float],
    issue_date: str | None,
    edition: str | None,
    section: str | None,
    limit: int,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
):
    candidate_limit = max(limit * 40, 400)
    if not issue_date and not start_date and not end_date and not edition and not section:
        candidate_limit = max(limit * 10, 200)
    effective_start_date = start_date if start_date is not None else issue_date
    effective_end_date = end_date if end_date is not None else issue_date
    with get_cursor() as cur:
        cur.execute(
            """
            with candidates as (
              select
                c.id,
                c.article_id,
                c.chunk_text,
                c.embedding <=> %s::vector as distance
              from article_chunks c
              order by c.embedding <=> %s::vector
              limit %s
            )
            select
              c.id as chunk_id,
              a.id as article_id,
              c.chunk_text,
              1 - c.distance as similarity,
              a.headline,
              s.normalized_section as section,
              p.publication_name
            from candidates c
            join articles a on a.id = c.article_id
            join publication_issues pi on pi.id = a.publication_issue_id
            join publications p on p.id = pi.publication_id
            join issues i on i.id = pi.issue_id
            left join sections s on s.id = a.section_id
            where (%s::date is null or i.issue_date >= %s::date)
              and (%s::date is null or i.issue_date <= %s::date)
              and (%s::text is null or p.publication_name ilike '%%' || %s::text || '%%')
              and (%s::text is null or s.normalized_section ilike '%%' || %s::text || '%%')
            order by c.distance
            limit %s
            """,
            (
                json.dumps(embedding),
                json.dumps(embedding),
                candidate_limit,
                effective_start_date,
                effective_start_date,
                effective_end_date,
                effective_end_date,
                edition,
                edition,
                section,
                section,
                limit,
            ),
        )
        return cur.fetchall()


def keyword_search(
    query_text: str,
    issue_date: str | None,
    edition: str | None,
    section: str | None,
    limit: int,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
):
    effective_start_date = start_date if start_date is not None else issue_date
    effective_end_date = end_date if end_date is not None else issue_date
    with get_cursor() as cur:
        cur.execute(
            """
            with q as (
              select websearch_to_tsquery('english', %s) as tsq
            )
            select
              a.id as article_id,
              a.external_article_id,
              a.headline,
              s.normalized_section as section,
              p.publication_name as edition,
              i.issue_date,
              left(ab.body_text, 400) as excerpt,
              ts_rank_cd(
                setweight(to_tsvector('english', coalesce(a.headline, '')), 'A') ||
                setweight(coalesce(ab.body_tsv, to_tsvector('english', '')), 'B'),
                q.tsq
              ) as lexical_score
            from q
            join articles a on true
            join publication_issues pi on pi.id = a.publication_issue_id
            join publications p on p.id = pi.publication_id
            join issues i on i.id = pi.issue_id
            left join sections s on s.id = a.section_id
            left join article_bodies ab on ab.article_id = a.id
            where (%s::date is null or i.issue_date >= %s::date)
              and (%s::date is null or i.issue_date <= %s::date)
              and (%s::text is null or p.publication_name ilike '%%' || %s::text || '%%')
              and (%s::text is null or s.normalized_section ilike '%%' || %s::text || '%%')
              and (
                setweight(to_tsvector('english', coalesce(a.headline, '')), 'A') ||
                setweight(coalesce(ab.body_tsv, to_tsvector('english', '')), 'B')
              ) @@ q.tsq
            order by lexical_score desc, a.id
            limit %s
            """,
            (
                query_text,
                effective_start_date,
                effective_start_date,
                effective_end_date,
                effective_end_date,
                edition,
                edition,
                section,
                section,
                limit,
            ),
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
