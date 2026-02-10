from __future__ import annotations

from types import SimpleNamespace

import pytest

from moose.ner import (
    make_cell_task_id,
    run_tabular_ner,
    run_text_ner,
)


class _FakeLLM:
    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._response


@pytest.mark.asyncio
async def test_run_text_ner_uses_single_prompt_for_all_tasks():
    tasks = [{"task_id": "t1", "text": "Alice"}, {"task_id": "t2", "text": "Bob"}]
    llm = _FakeLLM('[{"task_id":"t1","entities":[]},{"task_id":"t2","entities":[]}]')
    settings = SimpleNamespace(MOOSE_MAX_RETRIES=0)
    out = await run_text_ner(tasks, "coarse", llm, settings=settings)
    assert len(llm.calls) == 1
    assert [item["task_id"] for item in out["results"]] == ["t1", "t2"]


@pytest.mark.asyncio
async def test_run_text_ner_prompt_keeps_raw_text_unchanged():
    tasks = [{"task_id": "t1", "text": "hello\x00world"}]
    llm = _FakeLLM('[{"task_id":"t1","entities":[]}]')
    settings = SimpleNamespace(MOOSE_MAX_RETRIES=0)
    await run_text_ner(tasks, "coarse", llm, settings=settings)
    assert len(llm.calls) == 1
    assert "\\u0000" in llm.calls[0]


@pytest.mark.asyncio
async def test_run_tabular_ner_prompt_keeps_raw_cell_text_unchanged():
    table_task_id = "t1"
    cell_task_id = make_cell_task_id(table_task_id, 0, "name")
    tasks = [
        {
            "task_id": table_task_id,
            "table_id": "tbl",
            "sampled_rows": [{"name": "  Alice\x00"}],
            "target_columns": ["name"],
            "strings_only": True,
            "skip_structured_literals": False,
        }
    ]
    llm = _FakeLLM(f'[{{"task_id":"{cell_task_id}","entities":[]}}]')
    settings = SimpleNamespace(MOOSE_MAX_RETRIES=0)
    out = await run_tabular_ner(tasks, "coarse", llm, settings=settings)
    assert len(llm.calls) == 1
    assert '"text": "  Alice\\u0000"' in llm.calls[0]
    assert out["results"][0]["rows"][0]["cells"][0]["entities"] == []
