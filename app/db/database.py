


from contextlib import contextmanager
import logging
from pathlib import Path
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import psycopg
from psycopg.errors import CannotConnectNow, InternalError_, QueryCanceled
from psycopg.rows import dict_row
from psycopg_pool import PoolTimeout

from app.core.config import get_settings


settings = get_settings()
logger = logging.getLogger(__name__)
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "sql" / "schema.sql"
RUNTIME_SCHEMA_SQL = [
    """
    create table if not exists app_users (
      id bigserial primary key,
      email text not null unique,
      password_hash text not null,
      created_at timestamptz not null default now()
    )
    """,
    """
    create table if not exists user_sessions (
      id bigserial primary key,
      user_id bigint not null references app_users(id) on delete cascade,
      session_token text not null unique,
      session_context jsonb not null default '{}'::jsonb,
      expires_at timestamptz not null,
      created_at timestamptz not null default now()
    )
    """,
    """
    alter table user_sessions
      add column if not exists session_context jsonb not null default '{}'::jsonb
    """,
    """
    create table if not exists chat_interactions (
      id bigserial primary key,
      user_id bigint not null references app_users(id) on delete cascade,
      session_id bigint not null references user_sessions(id) on delete cascade,
      user_question text not null,
      system_answer text not null,
      issue_date date,
      mode text,
      session_filters jsonb,
      citations jsonb,
      trace_data jsonb,
      created_at timestamptz not null default now()
    )
    """,
    """
    alter table chat_interactions
      add column if not exists trace_data jsonb
    """,
    """
    create index if not exists idx_user_sessions_token on user_sessions(session_token)
    """,
    """
    create index if not exists idx_user_sessions_expires_at on user_sessions(expires_at)
    """,
    """
    create index if not exists idx_chat_interactions_user on chat_interactions(user_id, created_at desc)
    """,
    """
    create index if not exists idx_chat_interactions_session on chat_interactions(session_id, created_at desc)
    """,
]


def _conninfo_with_timeout(conninfo: str, connect_timeout_seconds: int) -> str:
    if "connect_timeout=" in conninfo:
        return conninfo
    if conninfo.startswith(("postgresql://", "postgres://")):
        parts = urlsplit(conninfo)
        query_params = dict(parse_qsl(parts.query, keep_blank_values=True))
        query_params.setdefault("connect_timeout", str(connect_timeout_seconds))
        return urlunsplit(
            (parts.scheme, parts.netloc, parts.path, urlencode(query_params), parts.fragment)
        )
    return f"{conninfo} connect_timeout={connect_timeout_seconds}"


def _strip_hostaddr_query(conninfo: str) -> str:
    return conninfo


def _direct_conninfo() -> str:
    configured = settings.supabase_db_direct_dsn or settings.supabase_db_dsn
    return _conninfo_with_timeout(
        _strip_hostaddr_query(configured),
        settings.db_connect_timeout_seconds,
    )


DIRECT_CONNINFO = _direct_conninfo()


def _direct_connection():
    return psycopg.connect(
        DIRECT_CONNINFO,
        row_factory=dict_row,
        prepare_threshold=None,
    )


def _is_retryable_db_error(exc: Exception) -> bool:
    if isinstance(exc, (CannotConnectNow, PoolTimeout, psycopg.OperationalError)):
        return True
    message = str(exc).lower()
    return (
        "unable to check out connection from the pool due to timeout" in message
        or "the database system is starting up" in message
    )


def _retry_delay_seconds(attempt: int) -> float:
    return settings.db_retry_base_delay_seconds * attempt


def _acquire_connection():
    last_error: Exception | None = None
    for attempt in range(1, settings.db_retry_attempts + 1):
        try:
            return _direct_connection()
        except Exception as exc:
            last_error = exc
            if attempt == settings.db_retry_attempts or not _is_retryable_db_error(exc):
                raise
            delay = _retry_delay_seconds(attempt)
            logger.warning(
                "Retrying database checkout attempt=%s/%s delay=%ss error=%s",
                attempt,
                settings.db_retry_attempts,
                delay,
                exc,
            )
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def open_pool() -> None:
    logger.info("Using retried direct database connections for request-time access")


def close_pool() -> None:
    logger.info("No request-time pool to close")


# ── Schema helpers ────────────────────────────────────────────────────────────

def ensure_schema() -> None:
    logger.info("Ensuring database schema from %s", SCHEMA_PATH)
    try:
        with _direct_connection() as conn:
            logger.info("Database connection acquired for schema ensure")
            if _schema_is_current(conn):
                logger.info("Database schema already current; skipping bootstrap SQL")
                return
            schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
            with conn.cursor() as cur:
                cur.execute(schema_sql)
            conn.commit()
        logger.info("Database schema ensure complete")
    except QueryCanceled:
        logger.warning("Schema ensure timed out; assuming existing schema and continuing startup")
    except Exception as exc:
        logger.warning("Schema ensure failed: %s; continuing startup", exc)


def ensure_runtime_schema() -> None:
    logger.info("Ensuring lightweight runtime schema")
    try:
        with _direct_connection() as conn:
            logger.info("Database connection acquired for runtime schema ensure")
            with conn.cursor() as cur:
                for statement in RUNTIME_SCHEMA_SQL:
                    cur.execute(statement)
            conn.commit()
        logger.info("Lightweight runtime schema ensure complete")
    except QueryCanceled:
        logger.warning("Runtime schema ensure timed out; continuing startup with existing schema")
    except InternalError_ as exc:
        logger.warning("Runtime schema ensure internal error: %s; continuing startup", exc)
    except Exception as exc:
        logger.warning("Runtime schema ensure failed: %s; continuing startup", exc)


def _schema_is_current(conn) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            select exists (
              select 1
              from information_schema.columns
              where table_name = 'user_sessions'
                and column_name = 'session_context'
            ) as has_session_context,
            exists (
              select 1
              from information_schema.columns
              where table_name = 'chat_interactions'
                and column_name = 'trace_data'
            ) as has_trace_data,
            exists (
              select 1
              from pg_proc
              where proname = 'match_article_chunks_filtered'
            ) as has_match_function
            """
        )
        row = cur.fetchone()
    return bool(
        row
        and row["has_session_context"]
        and row["has_trace_data"]
        and row["has_match_function"]
    )


# ── Request-time DB access ────────────────────────────────────────────────────

@contextmanager
def get_cursor():
    conn = _acquire_connection()
    try:
        with conn.cursor() as cur:
            yield cur
        conn.commit()
    except Exception:
        if not conn.closed:
            conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def get_connection():
    conn = _acquire_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        if not conn.closed:
            conn.rollback()
        raise
    finally:
        conn.close()
