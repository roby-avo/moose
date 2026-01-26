from __future__ import annotations

import httpx
from moose.llm.base import LLMClient


class DeepSeekClient(LLMClient):
    def __init__(self, api_key: str | None, base_url: str, model: str, timeout: float) -> None:
        super().__init__(provider="deepseek", model=model)
        if not api_key:
            raise ValueError("DeepSeek API key is required")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout, base_url=self._base_url)

    async def generate(self, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        resp = await self._client.post("/chat/completions", json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("DeepSeek response missing choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not content:
            raise RuntimeError("DeepSeek response missing content")
        return content

    async def close(self) -> None:
        await self._client.aclose()