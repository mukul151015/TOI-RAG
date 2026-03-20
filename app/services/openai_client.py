import atexit
import logging
import time
from functools import lru_cache

import httpx
from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError

from app.core.config import get_settings


settings = get_settings()
logger = logging.getLogger(__name__)

# Embedding cache: keyed by (model, normalized_texts_tuple).
# Max 1000 entries; TTL is effectively the process lifetime (news data is
# per-issue so content doesn't change within a single run).
_EMBED_CACHE_MAX = 1000
_CHAT_CACHE_MAX = 512
_http_client: httpx.Client | None = None
_client: OpenAI | None = None


def _build_http_client() -> httpx.Client:
    verify: bool | str = settings.openai_verify_ssl
    if settings.openai_ca_bundle_path:
        verify = settings.openai_ca_bundle_path
    return httpx.Client(verify=verify, timeout=60.0)


def get_client() -> OpenAI:
    global _client, _http_client
    if _client is None:
        _http_client = _build_http_client()
        _client = OpenAI(api_key=settings.openai_api_key, http_client=_http_client)
    return _client


def close_openai_client() -> None:
    global _client, _http_client
    try:
        if _http_client is not None:
            _http_client.close()
    finally:
        _client = None
        _http_client = None


atexit.register(close_openai_client)


@lru_cache(maxsize=_EMBED_CACHE_MAX)
def _embed_texts_cached(model: str, dimensions: int, texts_tuple: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
    """Internal cached embedding call.  Args must be hashable (tuples)."""
    last_error: Exception | None = None
    for attempt in range(1, settings.openai_max_retries + 1):
        try:
            response = get_client().embeddings.create(
                model=model,
                input=list(texts_tuple),
                dimensions=dimensions,
            )
            return tuple(tuple(item.embedding) for item in response.data)
        except (APIConnectionError, APITimeoutError, RateLimitError, httpx.ConnectError) as exc:
            last_error = exc
            if attempt == settings.openai_max_retries:
                break
            delay = settings.openai_retry_base_delay * attempt
            logger.warning(
                "OpenAI embedding retry attempt=%s/%s delay=%ss error=%s",
                attempt,
                settings.openai_max_retries,
                delay,
                exc,
            )
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts, using an in-process LRU cache to avoid
    redundant API calls for repeated queries within the same process."""
    normalized = tuple(t.strip() for t in texts)
    result = _embed_texts_cached(
        settings.openai_embedding_model,
        settings.embedding_dimensions,
        normalized,
    )
    return [list(vec) for vec in result]


def chat_completion(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
    timeout: float = 60.0,
) -> str:
    """Call the chat model.  Pass ``model`` to override the default.

    ``timeout`` (seconds) prevents indefinite hangs when the API is slow.
    Callers that don't need the answer (e.g. HyDE, LLM query analysis) should
    pass a shorter value so a single slow request doesn't stall the pipeline.
    """
    chosen_model = model or settings.openai_chat_model
    return _chat_completion_cached(chosen_model, system_prompt, user_prompt, timeout)


@lru_cache(maxsize=_CHAT_CACHE_MAX)
def _chat_completion_cached(
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: float,
) -> str:
    response = get_client().responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        timeout=timeout,
    )
    return response.output_text
