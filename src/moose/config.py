from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")

    MOOSE_LLM_PROVIDER: str = "ollama"
    MOOSE_MODEL: str = "llama3"
    MOOSE_API_KEY: str | None = None
    MOOSE_MAX_RETRIES: int = 2
    MOOSE_WORKER_COUNT: int = 4
    MOOSE_QUEUE_MAXSIZE: int = 1000
    MOOSE_OLLAMA_HOST: str = "http://localhost:11434"
    MOOSE_OLLAMA_TOKEN: str | None = None
    MOOSE_OPENROUTER_API_KEY: str | None = None
    MOOSE_OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    MOOSE_MONGO_URL: str | None = None
    MOOSE_MONGO_DB: str = "moose"
    MOOSE_MONGO_TIMEOUT_SECS: float = 2.0
    MOOSE_MAX_TASKS_PER_PROMPT: int = 10
    MOOSE_MAX_CHARS_PER_PROMPT: int = 12000
    MOOSE_TIMEOUT_SECS: float = 60.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
