from __future__ import annotations

from moose.config import Settings
from moose.llm.base import LLMClient

def create_client(settings: Settings) -> LLMClient:
    provider = settings.MOOSE_LLM_PROVIDER.lower()
    if provider == "ollama":
        from moose.llm.ollama import OllamaClient

        return OllamaClient(
            host=settings.MOOSE_OLLAMA_HOST,
            token=settings.MOOSE_OLLAMA_TOKEN,
            model=settings.MOOSE_MODEL,
            timeout=settings.MOOSE_TIMEOUT_SECS,
        )
    if provider == "openrouter":
        from moose.llm.openrouter import OpenRouterClient

        return OpenRouterClient(
            api_key=settings.MOOSE_OPENROUTER_API_KEY,
            base_url=settings.MOOSE_OPENROUTER_BASE_URL,
            model=settings.MOOSE_MODEL,
            timeout=settings.MOOSE_TIMEOUT_SECS,
        )
    if provider == "deepinfra":
        from moose.llm.deepinfra import DeepInfraClient

        return DeepInfraClient(
            api_key=settings.MOOSE_DEEPINFRA_API_KEY,
            base_url=settings.MOOSE_DEEPINFRA_BASE_URL,
            model=settings.MOOSE_MODEL,
            timeout=settings.MOOSE_TIMEOUT_SECS,
        )

    if provider == "deepseek":
        from moose.llm.deepseek import DeepSeekClient

        return DeepSeekClient(
            api_key=settings.MOOSE_DEEPSEEK_API_KEY,
            base_url=settings.MOOSE_DEEPSEEK_BASE_URL,
            model=settings.MOOSE_MODEL,
            timeout=settings.MOOSE_TIMEOUT_SECS,
        )
    raise ValueError(f"Unsupported LLM provider: {settings.MOOSE_LLM_PROVIDER}")
