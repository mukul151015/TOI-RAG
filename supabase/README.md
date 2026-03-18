# Supabase Edge Migration

This folder contains a zero-dependency Supabase Edge Functions port of the FastAPI API layer.

## Functions

- `auth-login`
- `auth-status`
- `auth-logout`
- `catalog-editions`
- `catalog-sections`
- `chat`

Shared runtime code lives in `supabase/functions/_shared/`.

## SQL setup

Run `supabase/sql/edge_rpc.sql` in the Supabase SQL Editor before deploying the functions.

## Secrets

- `OPENAI_API_KEY`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`

`SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` are provided automatically in Supabase Edge Functions. `OPENAI_API_KEY` must be added in the Supabase dashboard.

## Deploy

```bash
supabase functions deploy auth-login --no-verify-jwt
supabase functions deploy auth-status --no-verify-jwt
supabase functions deploy auth-logout --no-verify-jwt
supabase functions deploy catalog-editions --no-verify-jwt
supabase functions deploy catalog-sections --no-verify-jwt
supabase functions deploy chat --no-verify-jwt
```

## Remaining app migration

The frontend should point at these function URLs with bearer-token auth, and the overlapping FastAPI auth/catalog/chat routes should stay removed or be thin proxies only.
