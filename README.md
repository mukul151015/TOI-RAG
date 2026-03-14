# TOI RAG FastAPI

Starter project for TOI E-Paper RAG using:

- FastAPI
- Supabase Postgres
- `pgvector`
- OpenAI embeddings with `dimensions=512`

## SSL note

If your company network injects its own SSL certificate, OpenAI calls may fail with
`CERTIFICATE_VERIFY_FAILED`.

Options:

- preferred: set `OPENAI_CA_BUNDLE_PATH` to your corporate CA bundle path
- temporary local workaround: set `OPENAI_VERIFY_SSL=false`

## What is included

- Normalized schema for TOI feed metadata
- Feed ingestion endpoint
- Article chunk embedding pipeline
- SQL, semantic, and hybrid query endpoint
- Chat endpoint on top of retrieved articles

## Setup

1. Create a Supabase project.
2. Enable `pgvector`.
3. Run [sql/schema.sql](/Users/mukulkumar/Documents/Rag/sql/schema.sql) in the Supabase SQL editor.
4. Copy `.env.example` to `.env` and fill the values.
5. Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

6. Start the API:

```bash
uvicorn app.main:app --reload
```

## Endpoints

- `GET /health`
- `POST /ingest/feed`
- `GET /embeddings/status`
- `POST /embeddings/backfill`
- `POST /query`
- `POST /chat`

### Ingest example

```bash
curl -X POST http://localhost:8000/ingest/feed \
  -H "Content-Type: application/json" \
  -d '{"feed_url":"https://embed-epaper.indiatimes.com/api/rss-feeds-json/toi/11_03_2026","org_id":"toi"}'
```

### Query example

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Find articles about the Iran conflict published in the Mumbai edition","issue_date":"2026-03-11"}'
```

### Chat example

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"Summarize the Iran conflict coverage in the Mumbai edition","issue_date":"2026-03-11"}'
```

### Recommended ingest flow

1. Metadata-first ingest:

```bash
curl -X POST http://localhost:8000/ingest/feed \
  -H "Content-Type: application/json" \
  -d '{"feed_file":"data/toi_11_03_2026.json","org_id":"toi","process_embeddings":false}'
```

2. Check pending/failed embedding status:

```bash
curl http://localhost:8000/embeddings/status
```

3. Resume embeddings from the last failed/pending article id using worker threads:

```bash
curl -X POST http://localhost:8000/embeddings/backfill \
  -H "Content-Type: application/json" \
  -d '{"start_article_id":1468,"worker_count":4,"limit":100,"failed_only":false}'
```

For local testing, put the exact TOI payload into [data/toi_11_03_2026.json](/Users/mukulkumar/Documents/Rag/data/toi_11_03_2026.json). The ingest endpoint will prefer that file when present.

## Notes

- `pagegroup` is treated as the primary normalized section when present.
- Article metadata is kept relationally for SQL filters.
- Semantic retrieval runs on `article_chunks`.
- Hybrid retrieval applies SQL filters before vector ranking.
