from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "TOI RAG API"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    auth_bypass_in_dev: bool = True

    supabase_db_dsn: str
    supabase_db_direct_dsn: str | None = None
    db_auto_ensure_schema: bool = False
    db_auto_ensure_runtime_schema: bool = True
    db_connect_timeout_seconds: int = 5
    db_pool_timeout_seconds: float = 8.0
    db_pool_min_size: int = 1
    db_pool_max_size: int = 4
    db_retry_attempts: int = 3
    db_retry_base_delay_seconds: float = 1.0
    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4.1-mini"
    openai_chat_model_strong: str = "gpt-4.1"
    strong_model_intent_triggers: list = ["fact_lookup", "topic_count"]
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
    chunk_overlap_sentences: int = 2

    # Feature flags
    hyde_enabled: bool = True
    llm_query_analysis_enabled: bool = True
    reranking_enabled: bool = False  # requires sentence_transformers package; set True to enable
    reranker_top_k: int = 30
    low_confidence_threshold: float = 0.40
    rag_v3_enabled: bool = True

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
