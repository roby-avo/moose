from __future__ import annotations

from moose.config import Settings
from moose.llm.base import LLMClient
from moose.llm.ollama import OllamaClient
from moose.llm.openrouter import OpenRouterClient


def create_client(settings: Settings) -> LLMClient:
    provider = settings.MOOSE_LLM_PROVIDER.lower()
    if provider == "ollama":
        return OllamaClient(
            host=settings.MOOSE_OLLAMA_HOST,
            token=settings.MOOSE_OLLAMA_TOKEN,
            model=settings.MOOSE_MODEL,
            timeout=settings.MOOSE_TIMEOUT_SECS,
        )
    if provider == "openrouter":
        return OpenRouterClient(
            api_key=settings.MOOSE_OPENROUTER_API_KEY,
            base_url=settings.MOOSE_OPENROUTER_BASE_URL,
            model=settings.MOOSE_MODEL,
            timeout=settings.MOOSE_TIMEOUT_SECS,
        )
    raise ValueError(f"Unsupported LLM provider: {settings.MOOSE_LLM_PROVIDER}")
