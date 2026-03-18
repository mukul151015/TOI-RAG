drop function if exists auth_validate_session(text);
drop function if exists auth_get_session(text);
drop function if exists keyword_search_filtered(text, date, text, text, integer);
drop function if exists fetch_sql_articles_filtered(date, text, text, integer);
drop function if exists fetch_sql_articles_filtered(date, text, text, integer, text);
drop function if exists fetch_sql_article_count_filtered(date, text, text);
drop function if exists fetch_articles_by_ids(bigint[]);
drop function if exists fetch_matching_pubs(date, text);
drop function if exists fetch_section_counts_by_date(date);
drop function if exists fetch_author_article_count_filtered(date, text, text, text);
drop function if exists fetch_entity_mention_articles_filtered(date, text[], text, text, integer, boolean);
drop function if exists fetch_entity_mention_count_filtered(date, text[], text, text, boolean);
drop function if exists fetch_entity_mention_contexts_filtered(date, text[], text, text, integer, boolean);
drop function if exists match_article_chunks_filtered(jsonb, date, text, text, integer);
drop function if exists match_article_chunks_filtered(text, date, text, text, integer);

create or replace function auth_validate_session(p_token text)
returns table (user_id bigint, email text)
language sql
security definer
set search_path = public
as $$
  select s.user_id, u.email
  from user_sessions s
  join app_users u on u.id = s.user_id
  where s.session_token = p_token
    and s.expires_at > now();
$$;

create or replace function auth_get_session(p_token text)
returns table (session_id bigint, user_id bigint, email text, session_context jsonb)
language sql
security definer
set search_path = public
as $$
  select s.id as session_id, s.user_id, u.email, coalesce(s.session_context, '{}'::jsonb)
  from user_sessions s
  join app_users u on u.id = s.user_id
  where s.session_token = p_token
    and s.expires_at > now();
$$;

create or replace function keyword_search_filtered(
  query_text text,
  issue_dt date default null,
  publication_filter text default null,
  section_filter text default null,
  match_count integer default 20
)
returns table (
  article_id bigint,
  external_article_id text,
  headline text,
  section text,
  edition text,
  issue_date date,
  excerpt text,
  lexical_score real
)
language sql
stable
as $$
  with q as (
    select websearch_to_tsquery('english', query_text) as tsq
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
    )::real as lexical_score
  from q
  join articles a on true
  join publication_issues pi on pi.id = a.publication_issue_id
  join publications p on p.id = pi.publication_id
  join issues i on i.id = pi.issue_id
  left join sections s on s.id = a.section_id
  left join article_bodies ab on ab.article_id = a.id
  where (issue_dt is null or i.issue_date = issue_dt)
    and (publication_filter is null or p.publication_name ilike '%' || publication_filter || '%')
    and (section_filter is null or s.normalized_section ilike '%' || section_filter || '%')
    and (
      setweight(to_tsvector('english', coalesce(a.headline, '')), 'A') ||
      setweight(coalesce(ab.body_tsv, to_tsvector('english', '')), 'B')
    ) @@ q.tsq
  order by lexical_score desc, a.id
  limit match_count;
$$;

create or replace function fetch_sql_articles_filtered(
  issue_dt date default null,
  publication_filter text default null,
  section_filter text default null,
  result_limit integer default 20,
  author_filter text default null
)
returns table (
  id bigint,
  external_article_id text,
  headline text,
  section text,
  edition text,
  issue_date date,
  excerpt text
)
language sql
stable
as $$
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
  left join article_authors aa on aa.article_id = a.id
  left join authors au on au.id = aa.author_id
  where (issue_dt is null or i.issue_date = issue_dt)
    and (publication_filter is null or p.publication_name ilike '%' || publication_filter || '%')
    and (section_filter is null or s.normalized_section ilike '%' || section_filter || '%')
    and (author_filter is null or lower(au.display_name) = lower(author_filter))
    and coalesce(nullif(a.headline, ''), nullif(ab.cleaned_text, '')) is not null
    and (
      section_filter is distinct from 'Sports'
      or lower(coalesce(a.headline, '') || ' ' || left(coalesce(ab.cleaned_text, ''), 220)) ~
        '(cricket|football|golf|tennis|hockey|ipl|bcci|coach|match|cup|trophy|champion|squad|player|olympic|medal|rohit|dhoni|goal|league|wt20)'
    )
  order by i.issue_date desc, a.id desc
  limit result_limit;
$$;

create or replace function fetch_sql_article_count_filtered(
  issue_dt date default null,
  publication_filter text default null,
  section_filter text default null
)
returns table (article_count bigint)
language sql
stable
as $$
  select count(*)::bigint as article_count
  from articles a
  join publication_issues pi on pi.id = a.publication_issue_id
  join publications p on p.id = pi.publication_id
  join issues i on i.id = pi.issue_id
  left join sections s on s.id = a.section_id
  where (issue_dt is null or i.issue_date = issue_dt)
    and (publication_filter is null or p.publication_name ilike '%' || publication_filter || '%')
    and (section_filter is null or s.normalized_section ilike '%' || section_filter || '%');
$$;

create or replace function fetch_articles_by_ids(article_ids bigint[])
returns table (
  id bigint,
  external_article_id text,
  headline text,
  section text,
  edition text,
  issue_date date,
  excerpt text
)
language sql
stable
as $$
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
  where a.id = any(article_ids);
$$;

create or replace function fetch_matching_pubs(issue_dt date default null, edition_term text default null)
returns table (publication_name text, article_count bigint)
language sql
stable
as $$
  select
    p.publication_name,
    count(*)::bigint as article_count
  from articles a
  join publication_issues pi on pi.id = a.publication_issue_id
  join publications p on p.id = pi.publication_id
  join issues i on i.id = pi.issue_id
  where (issue_dt is null or i.issue_date = issue_dt)
    and (edition_term is null or p.publication_name ilike '%' || edition_term || '%')
  group by p.publication_name
  order by article_count desc, p.publication_name;
$$;

create or replace function fetch_section_counts_by_date(issue_dt date default null)
returns table (section text, article_count bigint)
language sql
stable
as $$
  select
    s.normalized_section as section,
    count(*)::bigint as article_count
  from articles a
  join publication_issues pi on pi.id = a.publication_issue_id
  join issues i on i.id = pi.issue_id
  left join sections s on s.id = a.section_id
  where (issue_dt is null or i.issue_date = issue_dt)
  group by s.normalized_section
  order by article_count desc, section nulls last;
$$;

create or replace function fetch_author_article_count_filtered(
  issue_dt date default null,
  publication_filter text default null,
  section_filter text default null,
  author_filter text default null
)
returns table (article_count bigint)
language sql
stable
as $$
  select count(*)::bigint as article_count
  from articles a
  join publication_issues pi on pi.id = a.publication_issue_id
  join publications p on p.id = pi.publication_id
  join issues i on i.id = pi.issue_id
  join article_authors aa on aa.article_id = a.id
  join authors au on au.id = aa.author_id
  left join sections s on s.id = a.section_id
  where (issue_dt is null or i.issue_date = issue_dt)
    and (publication_filter is null or p.publication_name ilike '%' || publication_filter || '%')
    and (section_filter is null or s.normalized_section ilike '%' || section_filter || '%')
    and (author_filter is null or lower(au.display_name) = lower(author_filter));
$$;

create or replace function fetch_entity_mention_articles_filtered(
  issue_dt date default null,
  entity_terms text[] default null,
  publication_filter text default null,
  section_filter text default null,
  result_limit integer default 20,
  headline_priority_only boolean default false
)
returns table (
  id bigint,
  external_article_id text,
  headline text,
  section text,
  edition text,
  issue_date date,
  excerpt text
)
language sql
stable
as $$
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
          when lower(coalesce(a.headline, '')) like '%' || term || '%' then 6
          when lower(left(coalesce(ab.cleaned_text, ab.body_text, ''), 320)) like '%' || term || '%' then 4
          when lower(coalesce(ab.cleaned_text, ab.body_text, '')) like '%' || term || '%' then 1
          else 0
        end
      ) as mention_score
    from articles a
    join publication_issues pi on pi.id = a.publication_issue_id
    join publications p on p.id = pi.publication_id
    join issues i on i.id = pi.issue_id
    left join sections s on s.id = a.section_id
    left join article_bodies ab on ab.article_id = a.id
    cross join unnest(entity_terms) term
    where (issue_dt is null or i.issue_date = issue_dt)
      and (publication_filter is null or p.publication_name ilike '%' || publication_filter || '%')
      and (section_filter is null or s.normalized_section ilike '%' || section_filter || '%')
      and lower(
        coalesce(a.headline, '') || ' ' || coalesce(ab.body_text, '') || ' ' || coalesce(ab.cleaned_text, '')
      ) like '%' || term || '%'
    group by
      a.id, a.external_article_id, a.headline, s.normalized_section,
      p.publication_name, i.issue_date, left(ab.body_text, 400)
  )
  select id, external_article_id, headline, section, edition, issue_date, excerpt
  from scored_articles
  where mention_score >= case when headline_priority_only then 6 else 4 end
  order by mention_score desc, issue_date desc, id desc
  limit result_limit;
$$;

create or replace function fetch_entity_mention_count_filtered(
  issue_dt date default null,
  entity_terms text[] default null,
  publication_filter text default null,
  section_filter text default null,
  headline_priority_only boolean default false
)
returns table (article_count bigint)
language sql
stable
as $$
  with scored_articles as (
    select
      a.id,
      max(
        case
          when lower(coalesce(a.headline, '')) like '%' || term || '%' then 6
          when lower(left(coalesce(ab.cleaned_text, ab.body_text, ''), 320)) like '%' || term || '%' then 4
          when lower(coalesce(ab.cleaned_text, ab.body_text, '')) like '%' || term || '%' then 1
          else 0
        end
      ) as mention_score
    from articles a
    join publication_issues pi on pi.id = a.publication_issue_id
    join publications p on p.id = pi.publication_id
    join issues i on i.id = pi.issue_id
    left join sections s on s.id = a.section_id
    left join article_bodies ab on ab.article_id = a.id
    cross join unnest(entity_terms) term
    where (issue_dt is null or i.issue_date = issue_dt)
      and (publication_filter is null or p.publication_name ilike '%' || publication_filter || '%')
      and (section_filter is null or s.normalized_section ilike '%' || section_filter || '%')
      and lower(
        coalesce(a.headline, '') || ' ' || coalesce(ab.body_text, '') || ' ' || coalesce(ab.cleaned_text, '')
      ) like '%' || term || '%'
    group by a.id
  )
  select count(*)::bigint as article_count
  from scored_articles
  where mention_score >= case when headline_priority_only then 6 else 4 end;
$$;

create or replace function fetch_entity_mention_contexts_filtered(
  issue_dt date default null,
  entity_terms text[] default null,
  publication_filter text default null,
  section_filter text default null,
  result_limit integer default 5,
  headline_priority_only boolean default false
)
returns table (
  headline text,
  article_count bigint,
  section text,
  excerpt text
)
language sql
stable
as $$
  with scored_articles as (
    select
      a.id,
      a.headline,
      s.normalized_section as section,
      coalesce(ab.body_text, ab.cleaned_text, '') as article_text,
      max(
        case
          when lower(coalesce(a.headline, '')) like '%' || term || '%' then 6
          when lower(left(coalesce(ab.cleaned_text, ab.body_text, ''), 320)) like '%' || term || '%' then 4
          when lower(coalesce(ab.cleaned_text, ab.body_text, '')) like '%' || term || '%' then 1
          else 0
        end
      ) as mention_score
    from articles a
    join publication_issues pi on pi.id = a.publication_issue_id
    join publications p on p.id = pi.publication_id
    join issues i on i.id = pi.issue_id
    left join sections s on s.id = a.section_id
    left join article_bodies ab on ab.article_id = a.id
    cross join unnest(entity_terms) term
    where (issue_dt is null or i.issue_date = issue_dt)
      and (publication_filter is null or p.publication_name ilike '%' || publication_filter || '%')
      and (section_filter is null or s.normalized_section ilike '%' || section_filter || '%')
      and a.headline is not null
      and lower(
        coalesce(a.headline, '') || ' ' || coalesce(ab.body_text, '') || ' ' || coalesce(ab.cleaned_text, '')
      ) like '%' || term || '%'
    group by a.id, a.headline, s.normalized_section, coalesce(ab.body_text, ab.cleaned_text, '')
  )
  select
    headline,
    count(*)::bigint as article_count,
    min(section) as section,
    left(min(article_text), 300) as excerpt
  from scored_articles
  where mention_score >= case when headline_priority_only then 6 else 4 end
  group by headline
  order by max(mention_score) desc, article_count desc, headline
  limit result_limit;
$$;

create or replace function match_article_chunks_filtered(
  query_embedding text,
  issue_dt date default null,
  publication_filter text default null,
  section_filter text default null,
  match_count integer default 20
)
returns table (
  chunk_id bigint,
  article_id bigint,
  chunk_text text,
  similarity real,
  headline text,
  section text,
  publication_name text
)
language sql
stable
as $$
  with candidates as (
    select
      c.id,
      c.article_id,
      c.chunk_text,
      c.embedding <=> query_embedding::vector as distance
    from article_chunks c
    order by c.embedding <=> query_embedding::vector
    limit greatest(match_count * 40, 400)
  )
  select
    c.id as chunk_id,
    a.id as article_id,
    c.chunk_text,
    (1 - c.distance)::real as similarity,
    a.headline,
    s.normalized_section as section,
    p.publication_name
  from candidates c
  join articles a on a.id = c.article_id
  join publication_issues pi on pi.id = a.publication_issue_id
  join publications p on p.id = pi.publication_id
  join issues i on i.id = pi.issue_id
  left join sections s on s.id = a.section_id
  where (issue_dt is null or i.issue_date = issue_dt)
    and (publication_filter is null or p.publication_name ilike '%' || publication_filter || '%')
    and (section_filter is null or s.normalized_section ilike '%' || section_filter || '%')
  order by c.distance
  limit match_count;
$$;
