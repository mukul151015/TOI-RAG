import logging
import time

import httpx
from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError

from app.core.config import get_settings


settings = get_settings()
logger = logging.getLogger(__name__)


def _build_http_client() -> httpx.Client:
    verify: bool | str = settings.openai_verify_ssl
    if settings.openai_ca_bundle_path:
        verify = settings.openai_ca_bundle_path
    return httpx.Client(verify=verify, timeout=60.0)


client = OpenAI(api_key=settings.openai_api_key, http_client=_build_http_client())


def embed_texts(texts: list[str]) -> list[list[float]]:
    last_error: Exception | None = None
    for attempt in range(1, settings.openai_max_retries + 1):
        try:
            response = client.embeddings.create(
                model=settings.openai_embedding_model,
                input=texts,
                dimensions=settings.embedding_dimensions,
            )
            return [item.embedding for item in response.data]
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


def chat_completion(system_prompt: str, user_prompt: str) -> str:
    response = client.responses.create(
        model=settings.openai_chat_model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.output_text
