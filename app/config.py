"""Environment-backed configuration for the API and indexer."""

from __future__ import annotations

import re
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ADDRESS_PATTERN = re.compile(r"^0x[a-fA-F0-9]{40}$")


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or a local ``.env`` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    database_url: str = "postgresql://zora_app:zora_app@localhost:5432/zora_analytics"
    read_only_database_url: str = (
        "postgresql://zora_reader:zora_reader@localhost:5432/zora_analytics"
    )

    zora_explorer_base_url: str = "https://explorer.zora.energy/api/v2"
    tracked_token_address: str = "0x7777777d57c1c6e472fa379b7b3b6c6ba3835073"
    tracked_chain: str = "zora"

    sync_interval_minutes: int = Field(default=15, ge=1, le=1440)
    sync_on_startup: bool = True
    enable_scheduler: bool = True
    admin_sync_token: SecretStr = SecretStr("replace-me")

    explorer_timeout_seconds: float = Field(default=20.0, gt=0, le=120)
    explorer_max_retries: int = Field(default=4, ge=0, le=10)
    explorer_backoff_seconds: float = Field(default=0.5, ge=0, le=30)
    explorer_max_pages: int = Field(default=10_000, ge=1)
    explorer_max_transfer_pages: int = Field(default=50_000, ge=1)
    allow_empty_holder_snapshot: bool = False
    sync_transfers: bool = True

    sql_max_rows: int = Field(default=1000, ge=1, le=10_000)
    sql_statement_timeout_ms: int = Field(default=5000, ge=100, le=60_000)

    llm_provider: Literal["anthropic", "openai", "claude_code"] = "anthropic"
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-sonnet-5"
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-5.6-terra"
    openai_reasoning_effort: Literal["none", "low", "medium", "high"] = "low"
    claude_code_command: str = "claude"
    claude_code_model: str = "sonnet"
    claude_code_timeout_seconds: float = Field(default=180.0, gt=0, le=900)
    llm_max_output_tokens: int = Field(default=2048, ge=128, le=32_768)
    llm_timeout_seconds: float = Field(default=60.0, gt=0, le=300)
    llm_input_cost_per_million: Decimal | None = Field(default=None, ge=0)
    llm_output_cost_per_million: Decimal | None = Field(default=None, ge=0)

    llm_provider_retry_attempts: int = Field(default=2, ge=0, le=6)
    llm_provider_backoff_seconds: float = Field(default=0.5, ge=0, le=30)
    llm_circuit_failure_threshold: int = Field(default=4, ge=1, le=100)
    llm_circuit_reset_seconds: float = Field(default=30.0, gt=0, le=3600)

    answer_cache_enabled: bool = True
    answer_cache_ttl_seconds: int = Field(default=3600, ge=1, le=2_592_000)

    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str | None = None
    otel_service_name: str = "zora-analytics-agent"

    retrieval_enabled: bool = True
    embeddings_provider: Literal["fastembed", "openai"] = "fastembed"
    fastembed_model: str = "BAAI/bge-small-en-v1.5"
    fastembed_cache_dir: Path = Path(".cache/fastembed")
    fastembed_local_files_only: bool = True
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=384, ge=32, le=4096)
    retrieval_dense_candidates: int = Field(default=20, ge=1, le=200)
    retrieval_sparse_candidates: int = Field(default=20, ge=1, le=200)
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    retrieval_rrf_k: int = Field(default=60, ge=1, le=1000)
    retrieval_chunk_tokens: int = Field(default=220, ge=40, le=1000)
    retrieval_chunk_overlap_tokens: int = Field(default=40, ge=0, le=400)
    reranker_provider: Literal["none", "fastembed"] = "none"
    fastembed_reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    mcp_host: str = "127.0.0.1"
    mcp_port: int = Field(default=8001, ge=1, le=65_535)

    @field_validator("tracked_token_address")
    @classmethod
    def normalize_token_address(cls, value: str) -> str:
        if not ADDRESS_PATTERN.fullmatch(value):
            raise ValueError("TRACKED_TOKEN_ADDRESS must be a 20-byte 0x-prefixed address")
        return value.lower()

    @field_validator("zora_explorer_base_url")
    @classmethod
    def strip_base_url(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("otel_service_name")
    @classmethod
    def require_otel_service_name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("OTEL_SERVICE_NAME cannot be empty")
        return stripped

    @field_validator("admin_sync_token")
    @classmethod
    def require_admin_token(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("ADMIN_SYNC_TOKEN cannot be empty")
        return value

    @property
    def selected_llm_model(self) -> str:
        """Return the model configured for the selected provider."""

        if self.llm_provider == "anthropic":
            return self.anthropic_model
        if self.llm_provider == "openai":
            return self.openai_model
        return self.claude_code_model


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide validated settings object."""

    return Settings()
