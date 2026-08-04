from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    app_name: str = "Arabic Fact Checker"
    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"
    database_url: str = "sqlite+aiosqlite:///./factchecker.db"
    redis_url: str | None = None
    frontend_origin: str = "http://localhost:3000"

    max_input_chars: int = Field(default=50_000, ge=100, le=500_000)
    request_timeout_seconds: float = Field(default=15.0, ge=1, le=60)
    max_document_bytes: int = Field(default=5_000_000, ge=10_000, le=20_000_000)

    search_provider: str = "local"
    search_api_key: str | None = None
    brave_search_api_key: str | None = None
    google_cse_id: str | None = None
    searxng_base_url: str | None = None
    local_search_fixture: Path = Path("../datasets/fixtures/search_results.json")

    enable_neural_models: bool = False
    verifier_model: str = "models/verifier"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    vector_db: Literal["faiss", "qdrant", "memory"] = "faiss"
    qdrant_url: str = "http://qdrant:6333"

    model_confidence_threshold: float = Field(default=0.70, ge=0, le=1)
    model_margin_threshold: float = Field(default=0.20, ge=0, le=1)
    evidence_quality_threshold: float = Field(default=0.65, ge=0, le=1)
    rate_limit_per_minute: int = Field(default=20, ge=1, le=10_000)

    cache_enabled: bool = True
    search_cache_ttl_seconds: int = 900
    page_cache_ttl_seconds: int = 86_400
    fact_check_cache_ttl_seconds: int = 3_600

    source_reliability_path: Path = Path("../configs/source_reliability.yaml")
    evidence_scoring_path: Path = Path("../configs/evidence_scoring.yaml")

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        return value.upper()

    def public_dict(self) -> dict[str, object]:
        return {
            "app_name": self.app_name,
            "app_env": self.app_env,
            "search_provider": self.search_provider,
            "neural_models_enabled": self.enable_neural_models,
            "verifier_model": self.verifier_model,
            "embedding_model": self.embedding_model,
            "vector_db": self.vector_db,
            "thresholds": {
                "model_confidence": self.model_confidence_threshold,
                "model_margin": self.model_margin_threshold,
                "evidence_quality": self.evidence_quality_threshold,
            },
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
