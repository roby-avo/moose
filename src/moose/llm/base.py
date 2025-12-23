from __future__ import annotations

from abc import ABC, abstractmethod

class LLMClient(ABC):
    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        raise NotImplementedError

    async def close(self) -> None:
        return None
