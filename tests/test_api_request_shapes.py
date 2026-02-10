from __future__ import annotations

import pytest
from pydantic import ValidationError

from moose_api.main import (
    SchemaCPARequest,
    SchemaNERRequest,
    SchemaTabularNERRequest,
    SchemaTabularRequest,
)


_LLM = {"provider": "openrouter", "model": "test-model"}


def test_schema_ner_accepts_single_payload_shape() -> None:
    request = SchemaNERRequest.model_validate({"text": "Alice works at Moose.", "llm": _LLM})
    assert request.text == "Alice works at Moose."


def test_schema_ner_rejects_legacy_tasks_shape() -> None:
    with pytest.raises(ValidationError):
        SchemaNERRequest.model_validate(
            {
                "text": "Alice works at Moose.",
                "tasks": [{"task_id": "t1", "text": "Alice works at Moose."}],
                "llm": _LLM,
            }
        )


def test_schema_tabular_accepts_single_payload_shape() -> None:
    request = SchemaTabularRequest.model_validate(
        {"sampled_rows": [{"name": "Alice"}], "llm": _LLM}
    )
    assert request.sampled_rows == [{"name": "Alice"}]


def test_schema_tabular_rejects_legacy_tasks_shape() -> None:
    with pytest.raises(ValidationError):
        SchemaTabularRequest.model_validate(
            {
                "sampled_rows": [{"name": "Alice"}],
                "tasks": [{"task_id": "t1", "sampled_rows": [{"name": "Alice"}]}],
                "llm": _LLM,
            }
        )


def test_schema_tabular_ner_accepts_single_payload_shape() -> None:
    request = SchemaTabularNERRequest.model_validate(
        {
            "sampled_rows": [{"notes": "email alice@example.com"}],
            "target_columns": ["notes"],
            "llm": _LLM,
        }
    )
    assert request.target_columns == ["notes"]


def test_schema_tabular_ner_rejects_legacy_tasks_shape() -> None:
    with pytest.raises(ValidationError):
        SchemaTabularNERRequest.model_validate(
            {
                "sampled_rows": [{"notes": "email alice@example.com"}],
                "target_columns": ["notes"],
                "tasks": [
                    {
                        "task_id": "t1",
                        "sampled_rows": [{"notes": "email alice@example.com"}],
                        "target_columns": ["notes"],
                    }
                ],
                "llm": _LLM,
            }
        )


def test_schema_cpa_accepts_single_payload_shape() -> None:
    request = SchemaCPARequest.model_validate(
        {
            "sampled_rows": [{"subject": "Alice", "target": "Bob"}],
            "subject_column": "subject",
            "llm": _LLM,
        }
    )
    assert request.subject_column == "subject"


def test_schema_cpa_rejects_legacy_tasks_shape() -> None:
    with pytest.raises(ValidationError):
        SchemaCPARequest.model_validate(
            {
                "sampled_rows": [{"subject": "Alice", "target": "Bob"}],
                "subject_column": "subject",
                "tasks": [
                    {
                        "task_id": "t1",
                        "sampled_rows": [{"subject": "Alice", "target": "Bob"}],
                        "subject_column": "subject",
                    }
                ],
                "llm": _LLM,
            }
        )
