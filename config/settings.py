"""
Centralized application configuration.

All runtime parameters (API keys, model names, RAG hyperparameters,
storage paths) are loaded from environment variables via pydantic-settings.
This avoids magic strings/numbers scattered across the codebase and makes
the system's behavior tunable without code changes — a deliberate Code
Quality and System Design choice.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ── LLM / OpenRouter ──────────────────────────────────────────
    openrouter_api_key: str = Field(..., alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        "https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL"
    )

    coordinator_model: str = Field(
        "meta-llama/llama-3.1-8b-instruct:free", alias="COORDINATOR_MODEL"
    )
    general_agent_model: str = Field(
        "meta-llama/llama-3.1-8b-instruct:free", alias="GENERAL_AGENT_MODEL"
    )
    retriever_agent_model: str = Field(
        "meta-llama/llama-3.1-8b-instruct:free", alias="RETRIEVER_AGENT_MODEL"
    )

    # ── RAG ───────────────────────────────────────────────────────
    embedding_model: str = Field("all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    chunk_size: int = Field(500, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(50, alias="CHUNK_OVERLAP")
    top_k_retrieval: int = Field(3, alias="TOP_K_RETRIEVAL")
    similarity_threshold: float = Field(0.3, alias="SIMILARITY_THRESHOLD")

    # ── Storage ───────────────────────────────────────────────────
    duckdb_path: str = Field("storage/vectors.duckdb", alias="DUCKDB_PATH")
    knowledge_base_dir: str = Field(
        "data/knowledge_base", alias="KNOWLEDGE_BASE_DIR"
    )

    # ── Logging ───────────────────────────────────────────────────
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    log_file: str = Field("logs/app.log", alias="LOG_FILE")

    # ── App ───────────────────────────────────────────────────────
    app_env: str = Field("development", alias="APP_ENV")


# Singleton instance — imported everywhere else as `from config.settings import settings`
settings = Settings()