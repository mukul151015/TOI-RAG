from contextlib import contextmanager
from pathlib import Path

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.core.config import get_settings


settings = get_settings()
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "sql" / "schema.sql"
pool = ConnectionPool(
    conninfo=settings.supabase_db_dsn,
    min_size=1,
    max_size=8,
    kwargs={"row_factory": dict_row, "prepare_threshold": None},
    open=False,
)


def open_pool() -> None:
    if pool.closed:
        pool.open()


def close_pool() -> None:
    if not pool.closed:
        pool.close()


def ensure_schema() -> None:
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(schema_sql)
        conn.commit()


@contextmanager
def get_cursor():
    with pool.connection() as conn:
        with conn.cursor() as cur:
            yield cur
        conn.commit()


@contextmanager
def get_connection():
    with pool.connection() as conn:
        yield conn
        conn.commit()
