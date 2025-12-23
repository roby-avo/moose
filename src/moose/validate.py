from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, TypeAdapter


class NEREntityModel(BaseModel):
    start: int
    end: int
    text: str
    scores: dict[str, float]


class NERTaskModel(BaseModel):
    task_id: str
    entities: list[NEREntityModel]


class TableColumnModel(BaseModel):
    column: str
    scores: dict[str, float]


class TableTaskModel(BaseModel):
    task_id: str
    table_id: str
    columns: list[TableColumnModel]


def extract_json(text: str) -> Any:
    text = text.strip()
    if not text:
        raise ValueError("Empty response")
    first_curly = text.find("{")
    first_square = text.find("[")
    if first_curly == -1 and first_square == -1:
        raise ValueError("No JSON object found")
    start = min(i for i in (first_curly, first_square) if i != -1)
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(text[start:])
    return obj


def _validate_scores(scores: dict[str, float], allowed_types: set[str]) -> None:
    missing = allowed_types.difference(scores.keys())
    if missing:
        raise ValueError(f"Missing score keys: {sorted(missing)}")
    for key, value in scores.items():
        if key not in allowed_types:
            raise ValueError(f"Unexpected score key: {key}")
        if value < 0:
            raise ValueError("Scores must be non-negative")
    if all(value <= 0 for value in scores.values()):
        raise ValueError("At least one score must be > 0")


def validate_ner_response(tasks: list[dict], raw_text: str, allowed_types: set[str]) -> list[NERTaskModel]:
    data = extract_json(raw_text)
    adapter = TypeAdapter(list[NERTaskModel])
    parsed = adapter.validate_python(data)

    task_lookup = {t["task_id"]: t["text"] for t in tasks}
    task_ids = set(task_lookup)
    seen_ids = {item.task_id for item in parsed}
    if task_ids != seen_ids:
        raise ValueError("Task IDs mismatch in NER response")

    for item in parsed:
        text = task_lookup[item.task_id]
        for entity in item.entities:
            if entity.start < 0 or entity.end > len(text) or entity.start >= entity.end:
                raise ValueError("Invalid entity offsets")
            if text[entity.start : entity.end] != entity.text:
                raise ValueError("Entity text does not match offsets")
            _validate_scores(entity.scores, allowed_types)

    return parsed


def validate_table_response(
    tasks: list[dict], raw_text: str, allowed_types: set[str]
) -> list[TableTaskModel]:
    data = extract_json(raw_text)
    adapter = TypeAdapter(list[TableTaskModel])
    parsed = adapter.validate_python(data)

    task_lookup = {t["task_id"]: t for t in tasks}
    task_ids = set(task_lookup)
    seen_ids = {item.task_id for item in parsed}
    if task_ids != seen_ids:
        raise ValueError("Task IDs mismatch in table response")

    for item in parsed:
        task = task_lookup[item.task_id]
        if item.table_id != task["table_id"]:
            raise ValueError("table_id mismatch in table response")
        columns = set()
        for row in task["sampled_rows"]:
            columns.update(row.keys())
        output_columns = [col.column for col in item.columns]
        if set(output_columns) != columns:
            raise ValueError("Column names mismatch in table response")
        if len(output_columns) != len(set(output_columns)):
            raise ValueError("Duplicate columns in table response")
        for column in item.columns:
            _validate_scores(column.scores, allowed_types)

    return parsed
