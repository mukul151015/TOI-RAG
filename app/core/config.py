from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "TOI RAG API"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    supabase_db_dsn: str
    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4.1-mini"
    openai_verify_ssl: bool = True
    openai_ca_bundle_path: str | None = None
    openai_max_retries: int = 4
    openai_retry_base_delay: float = 1.5
    embedding_worker_count: int = 4
    embedding_backfill_batch_size: int = 100
    ingestion_checkpoint_interval: int = 100
    embedding_dimensions: int = 512
    embedding_batch_size: int = 32
    chunk_size: int = 1800
    chunk_overlap: int = 250

    default_org_id: str = "toi"
    default_feed_url: str = (
        "https://embed-epaper.indiatimes.com/api/rss-feeds-json/toi/11_03_2026"
    )
    default_feed_file: str = "data/toi_11_03_2026.json"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
