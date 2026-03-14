create extension if not exists vector;

create table if not exists organizations (
  id text primary key,
  name text not null,
  created_at timestamptz not null default now()
);

create table if not exists issues (
  id bigserial primary key,
  organization_id text not null references organizations(id),
  issue_date date not null,
  issue_name text not null,
  created_at timestamptz not null default now(),
  unique (organization_id, issue_date, issue_name)
);

create table if not exists publications (
  id text primary key,
  organization_id text not null references organizations(id),
  publication_name text not null,
  created_at timestamptz not null default now()
);

create table if not exists ingestion_runs (
  id bigserial primary key,
  organization_id text not null references organizations(id),
  source_url text not null,
  issue_date date,
  status text not null default 'running',
  checkpoint_publication_id text,
  checkpoint_doc_index int,
  last_processed_article_id bigint,
  last_error text,
  raw_payload jsonb not null,
  created_at timestamptz not null default now()
);

alter table ingestion_runs
  add column if not exists status text not null default 'running';

alter table ingestion_runs
  add column if not exists checkpoint_publication_id text;

alter table ingestion_runs
  add column if not exists checkpoint_doc_index int;

alter table ingestion_runs
  add column if not exists last_processed_article_id bigint;

alter table ingestion_runs
  add column if not exists last_error text;

create table if not exists publication_issues (
  id bigserial primary key,
  issue_id bigint not null references issues(id) on delete cascade,
  publication_id text not null references publications(id),
  num_found int not null default 0,
  fetched int not null default 0,
  in_house int not null default 0,
  converted int not null default 0,
  created_at timestamptz not null default now(),
  unique (issue_id, publication_id)
);

create table if not exists publication_issue_rule_counts (
  id bigserial primary key,
  publication_issue_id bigint not null references publication_issues(id) on delete cascade,
  rule_kind text not null check (rule_kind in ('accept', 'reject')),
  rule_name text not null,
  rule_count int not null default 0,
  created_at timestamptz not null default now(),
  unique (publication_issue_id, rule_kind, rule_name)
);

create table if not exists sections (
  id bigserial primary key,
  publication_id text not null references publications(id),
  zone text,
  pagegroup text,
  layoutdesk text,
  normalized_section text generated always as (
    coalesce(nullif(pagegroup, ''), nullif(zone, ''), nullif(layoutdesk, ''))
  ) stored,
  created_at timestamptz not null default now(),
  unique nulls not distinct (publication_id, zone, pagegroup, layoutdesk)
);

create table if not exists articles (
  id bigserial primary key,
  publication_issue_id bigint not null references publication_issues(id) on delete cascade,
  external_article_id text not null,
  external_doc_id text,
  section_id bigint references sections(id),
  pageno int,
  headline text,
  deck text,
  label text,
  location text,
  article_filename text,
  status text,
  is_searchable boolean not null default false,
  processing_status text not null default 'pending',
  embedding_status text not null default 'pending',
  last_error text,
  embedding_source_hash text,
  issue_timestamp timestamptz,
  updated_at timestamptz,
  raw_json jsonb not null,
  created_at timestamptz not null default now(),
  unique (publication_issue_id, external_article_id)
);

alter table articles
  add column if not exists is_searchable boolean not null default false;

alter table articles
  add column if not exists embedding_source_hash text;

alter table articles
  add column if not exists processing_status text not null default 'pending';

alter table articles
  add column if not exists embedding_status text not null default 'pending';

alter table articles
  add column if not exists last_error text;

create table if not exists article_bodies (
  article_id bigint primary key references articles(id) on delete cascade,
  body_text text not null,
  cleaned_text text,
  body_tsv tsvector generated always as (
    to_tsvector('english', coalesce(cleaned_text, body_text, ''))
  ) stored,
  created_at timestamptz not null default now()
);

alter table article_bodies
  add column if not exists cleaned_text text;

create table if not exists authors (
  id bigserial primary key,
  display_name text,
  email text unique,
  created_at timestamptz not null default now()
);

create table if not exists article_authors (
  article_id bigint not null references articles(id) on delete cascade,
  author_id bigint not null references authors(id) on delete cascade,
  created_at timestamptz not null default now(),
  primary key (article_id, author_id)
);

create table if not exists article_chunks (
  id bigserial primary key,
  article_id bigint not null references articles(id) on delete cascade,
  chunk_index int not null,
  chunk_text text not null,
  embedding vector(512) not null,
  token_count int,
  created_at timestamptz not null default now(),
  unique (article_id, chunk_index)
);

create index if not exists idx_issues_date on issues(issue_date);
create index if not exists idx_articles_pub_issue on articles(publication_issue_id);
create index if not exists idx_articles_section on articles(section_id);
create index if not exists idx_article_bodies_tsv on article_bodies using gin (body_tsv);
create index if not exists idx_articles_headline_tsv on articles using gin (
  to_tsvector('english', coalesce(headline, ''))
);
create index if not exists idx_article_chunks_embedding on article_chunks
using hnsw (embedding vector_cosine_ops);

create or replace function match_article_chunks_filtered(
  query_embedding vector(512),
  issue_dt date default null,
  publication_filter text default null,
  section_filter text default null,
  match_count int default 10
)
returns table (
  chunk_id bigint,
  article_id bigint,
  chunk_text text,
  similarity float,
  headline text,
  section text,
  publication_name text
)
language sql
as $$
  select
    c.id as chunk_id,
    a.id as article_id,
    c.chunk_text,
    1 - (c.embedding <=> query_embedding) as similarity,
    a.headline,
    s.normalized_section as section,
    p.publication_name
  from article_chunks c
  join articles a on a.id = c.article_id
  join publication_issues pi on pi.id = a.publication_issue_id
  join publications p on p.id = pi.publication_id
  join issues i on i.id = pi.issue_id
  left join sections s on s.id = a.section_id
  where (issue_dt is null or i.issue_date = issue_dt)
    and (publication_filter is null or p.publication_name ilike '%' || publication_filter || '%')
    and (section_filter is null or s.normalized_section ilike '%' || section_filter || '%')
  order by c.embedding <=> query_embedding
  limit match_count;
$$;
