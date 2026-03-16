from __future__ import annotations

from datetime import UTC, datetime, timedelta
import hashlib
import hmac
import json
import secrets

from psycopg import Error as PsycopgError
from psycopg.errors import UndefinedColumn
from fastapi import HTTPException, Request, Response, status

from app.core.config import get_settings
from app.db.database import get_cursor


settings = get_settings()
SESSION_COOKIE = "toi_rag_session"
SESSION_DAYS = 7
DEV_USER = {"id": 0, "email": "local-dev@toi"}
DEV_SESSION = {"session_id": 0, "user_id": 0, "email": "local-dev@toi", "session_context": {}}


def _dev_auth_enabled() -> bool:
    return settings.app_env == "development" and settings.auth_bypass_in_dev


def login_or_create(email: str, password: str, response: Response) -> dict[str, str]:
    normalized_email = email.strip().lower()
    if not normalized_email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email is required")

    with get_cursor() as cur:
        cur.execute(
            """
            select id, email, password_hash
            from app_users
            where email = %s
            """,
            (normalized_email,),
        )
        user = cur.fetchone()

        if not user:
            password_hash = _hash_password(password)
            cur.execute(
                """
                insert into app_users (email, password_hash)
                values (%s, %s)
                returning id, email
                """,
                (normalized_email, password_hash),
            )
            user = cur.fetchone()
        else:
            if not _verify_password(password, user["password_hash"]):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid email or password",
                )

        token = secrets.token_urlsafe(32)
        expires_at = datetime.now(UTC) + timedelta(days=SESSION_DAYS)
        cur.execute(
            """
            insert into user_sessions (user_id, session_token, expires_at)
            values (%s, %s, %s)
            """,
            (user["id"], token, expires_at),
        )

    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        max_age=SESSION_DAYS * 24 * 60 * 60,
        expires=SESSION_DAYS * 24 * 60 * 60,
        httponly=True,
        samesite="lax",
    )
    return {"email": normalized_email}


def logout(response: Response, session_token: str | None) -> None:
    if session_token:
        with get_cursor() as cur:
            cur.execute(
                """
                delete from user_sessions
                where session_token = %s
                """,
                (session_token,),
            )
    response.delete_cookie(SESSION_COOKIE)


def get_authenticated_user(request: Request) -> dict | None:
    if _dev_auth_enabled():
        return DEV_USER
    session_token = request.cookies.get(SESSION_COOKIE)
    if not session_token:
        return None
    try:
        with get_cursor() as cur:
            cur.execute(
                """
                select u.id, u.email
                from user_sessions s
                join app_users u on u.id = s.user_id
                where s.session_token = %s
                  and s.expires_at > now()
                """,
                (session_token,),
            )
            return cur.fetchone()
    except PsycopgError:
        if _dev_auth_enabled():
            return DEV_USER
        raise


def get_authenticated_session(request: Request) -> dict | None:
    if _dev_auth_enabled():
        return DEV_SESSION
    session_token = request.cookies.get(SESSION_COOKIE)
    if not session_token:
        return None
    try:
        with get_cursor() as cur:
            cur.execute(
                """
                select
                  s.id as session_id,
                  s.user_id,
                  u.email,
                  s.session_context
                from user_sessions s
                join app_users u on u.id = s.user_id
                where s.session_token = %s
                  and s.expires_at > now()
                """,
                (session_token,),
            )
            return cur.fetchone()
    except PsycopgError:
        if _dev_auth_enabled():
            return DEV_SESSION
        raise


def update_session_context(session_id: int, session_context: dict | None) -> None:
    if session_context is None or session_id <= 0:
        return
    with get_cursor() as cur:
        cur.execute(
            """
            update user_sessions
            set session_context = %s::jsonb
            where id = %s
            """,
            (json.dumps(session_context, default=str), session_id),
        )


def require_authenticated_user(request: Request) -> dict:
    user = get_authenticated_user(request)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    return user


def log_chat_interaction(
    *,
    user_id: int,
    session_id: int,
    question: str,
    answer: str,
    issue_date: str | None,
    mode: str | None,
    session_filters: dict | None,
    citations: list[dict] | None,
    trace_data: dict | None,
) -> None:
    if session_id <= 0:
        return
    with get_cursor() as cur:
        params = (
            user_id,
            session_id,
            question,
            answer,
            issue_date,
            mode,
            json.dumps(session_filters or {}, default=str),
            json.dumps(citations or [], default=str),
            json.dumps(trace_data or {}, default=str),
        )
        try:
            cur.execute(
                """
                insert into chat_interactions (
                  user_id,
                  session_id,
                  user_question,
                  system_answer,
                  issue_date,
                  mode,
                  session_filters,
                  citations,
                  trace_data
                )
                values (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb)
                """,
                params,
            )
        except UndefinedColumn:
            cur.execute(
                """
                insert into chat_interactions (
                  user_id,
                  session_id,
                  user_question,
                  system_answer,
                  issue_date,
                  mode,
                  session_filters,
                  citations
                )
                values (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                """,
                params[:-1],
            )


def fetch_recent_chat_traces(*, user_id: int | None = None, limit: int = 20) -> list[dict]:
    with get_cursor() as cur:
        cur.execute(
            """
            select
              ci.id,
              ci.user_question,
              ci.system_answer,
              ci.mode,
              ci.issue_date,
              ci.session_filters,
              ci.citations,
              ci.trace_data,
              ci.created_at,
              u.email
            from chat_interactions ci
            join app_users u on u.id = ci.user_id
            where (%s::bigint is null or ci.user_id = %s::bigint)
            order by ci.created_at desc
            limit %s
            """,
            (user_id, user_id, limit),
        )
        return cur.fetchall()


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return f"{salt.hex()}${digest.hex()}"


def _verify_password(password: str, password_hash: str) -> bool:
    try:
        salt_hex, digest_hex = password_hash.split("$", 1)
    except ValueError:
        return False
    salt = bytes.fromhex(salt_hex)
    expected = bytes.fromhex(digest_hex)
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return hmac.compare_digest(actual, expected)
