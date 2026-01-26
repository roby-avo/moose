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


# -------------------
# NEW: CPA models
# -------------------
class CPARelationshipModel(BaseModel):
    target_column: str
    scores: dict[str, float]


class CPATaskModel(BaseModel):
    task_id: str
    table_id: str
    subject_column: str
    relationships: list[CPARelationshipModel]


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


def _validate_scores(scores: dict[str, float], allowed_types: set[str], require_all: bool = True) -> None:
    if require_all:
        missing = allowed_types.difference(scores.keys())
        if missing:
            raise ValueError(f"Missing score keys: {sorted(missing)}")
    for key, value in scores.items():
        if key not in allowed_types:
            raise ValueError(f"Unexpected score key: {key}")
        if value < 0:
            raise ValueError("Scores must be non-negative")
    if not scores or all(value <= 0 for value in scores.values()):
        raise ValueError("At least one score must be > 0")


def _normalize_scores(
    scores: dict[str, float],
    allowed_types: set[str],
    type_aliases: dict[str, str] | None = None,
    type_alias_prefixes: dict[str, str] | None = None,
) -> dict[str, float]:
    if not type_aliases and not type_alias_prefixes:
        return scores
    normalized = dict(scores)
    if type_aliases:
        for alias, canonical in type_aliases.items():
            if alias in normalized and canonical in allowed_types:
                value = normalized.pop(alias)
                current = normalized.get(canonical)
                normalized[canonical] = value if current is None else max(current, value)
    if type_alias_prefixes:
        for alias_prefix, canonical_prefix in type_alias_prefixes.items():
            for key in list(normalized.keys()):
                if key in allowed_types:
                    continue
                if key.startswith(alias_prefix):
                    candidate = canonical_prefix + key[len(alias_prefix) :]
                    if candidate in allowed_types:
                        value = normalized.pop(key)
                        current = normalized.get(candidate)
                        normalized[candidate] = value if current is None else max(current, value)
    return normalized


def _normalize_type_id(
    value: str,
    allowed_types: set[str],
    type_aliases: dict[str, str] | None = None,
    type_alias_prefixes: dict[str, str] | None = None,
) -> str | None:
    if value in allowed_types:
        return value
    if type_aliases and value in type_aliases:
        candidate = type_aliases[value]
        if candidate in allowed_types:
            return candidate
    if type_alias_prefixes:
        for alias_prefix, canonical_prefix in type_alias_prefixes.items():
            if value.startswith(alias_prefix):
                candidate = canonical_prefix + value[len(alias_prefix) :]
                if candidate in allowed_types:
                    return candidate
    return None


def validate_type_selection_response(
    raw_text: str,
    allowed_types: set[str],
    type_aliases: dict[str, str] | None = None,
    type_alias_prefixes: dict[str, str] | None = None,
) -> list[str]:
    data = extract_json(raw_text)
    if not isinstance(data, list):
        raise ValueError("Type selection must be a JSON array.")
    selected: list[str] = []
    seen: set[str] = set()
    for item in data:
        if not isinstance(item, str):
            raise ValueError("Type selection items must be strings.")
        normalized = _normalize_type_id(
            item.strip(),
            allowed_types,
            type_aliases=type_aliases,
            type_alias_prefixes=type_alias_prefixes,
        )
        if not normalized:
            continue
        if normalized not in seen:
            selected.append(normalized)
            seen.add(normalized)
    return selected


def _repair_offsets(full_text: str, start: int, end: int, ent_text: str) -> tuple[int, int] | None:
    ent_text = (ent_text or "").strip()
    if not ent_text:
        return None

    matches: list[int] = []
    pos = full_text.find(ent_text)
    while pos != -1:
        matches.append(pos)
        pos = full_text.find(ent_text, pos + 1)

    if not matches:
        return None

    best = min(matches, key=lambda i: abs(i - start))
    return best, best + len(ent_text)


def validate_ner_response_with_warnings(
    tasks: list[dict],
    raw_text: str,
    allowed_types: set[str],
    require_all_scores: bool = True,
    type_aliases: dict[str, str] | None = None,
    type_alias_prefixes: dict[str, str] | None = None,
    strict_offsets: bool = False,
) -> tuple[list[NERTaskModel], list[dict[str, Any]]]:
    data = extract_json(raw_text)
    adapter = TypeAdapter(list[NERTaskModel])
    parsed = adapter.validate_python(data)

    task_lookup = {t["task_id"]: t["text"] for t in tasks}
    task_ids = set(task_lookup)
    seen_ids = {item.task_id for item in parsed}
    if task_ids != seen_ids:
        raise ValueError("Task IDs mismatch in NER response")

    warnings: list[dict[str, Any]] = []

    for item in parsed:
        text = task_lookup[item.task_id]
        kept: list[NEREntityModel] = []

        for entity in item.entities:
            original = {"start": entity.start, "end": entity.end, "text": entity.text}

            offsets_valid = entity.start >= 0 and entity.end <= len(text) and entity.start < entity.end
            if not offsets_valid:
                if strict_offsets:
                    raise ValueError("Invalid entity offsets")

                repaired = _repair_offsets(text, entity.start, entity.end, entity.text)
                if repaired is None:
                    warnings.append(
                        {
                            "task_id": item.task_id,
                            "code": "entity_dropped_invalid_offsets",
                            "original": original,
                        }
                    )
                    continue

                entity.start, entity.end = repaired
                warnings.append(
                    {
                        "task_id": item.task_id,
                        "code": "offsets_repaired_from_invalid",
                        "original": original,
                        "final": {"start": entity.start, "end": entity.end},
                    }
                )

            slice_text = text[entity.start : entity.end]
            if slice_text != entity.text:
                repaired = _repair_offsets(text, entity.start, entity.end, entity.text)
                if repaired is not None:
                    entity.start, entity.end = repaired
                    slice_text = text[entity.start : entity.end]
                    warnings.append(
                        {
                            "task_id": item.task_id,
                            "code": "offsets_repaired_text_mismatch",
                            "original": original,
                            "final": {"start": entity.start, "end": entity.end},
                        }
                    )
                else:
                    if strict_offsets:
                        raise ValueError("Entity text does not match offsets")
                    warnings.append(
                        {
                            "task_id": item.task_id,
                            "code": "text_overwritten_to_match_offsets",
                            "original": original,
                            "final": {"start": entity.start, "end": entity.end, "text": slice_text},
                        }
                    )

                entity.text = slice_text

            normalized = _normalize_scores(
                entity.scores,
                allowed_types,
                type_aliases=type_aliases,
                type_alias_prefixes=type_alias_prefixes,
            )
            if normalized is not entity.scores:
                entity.scores.clear()
                entity.scores.update(normalized)
            _validate_scores(entity.scores, allowed_types, require_all=require_all_scores)

            kept.append(entity)

        item.entities = kept

    return parsed, warnings


def validate_ner_response(
    tasks: list[dict],
    raw_text: str,
    allowed_types: set[str],
    require_all_scores: bool = True,
    type_aliases: dict[str, str] | None = None,
    type_alias_prefixes: dict[str, str] | None = None,
) -> list[NERTaskModel]:
    parsed, _warnings = validate_ner_response_with_warnings(
        tasks,
        raw_text,
        allowed_types,
        require_all_scores=require_all_scores,
        type_aliases=type_aliases,
        type_alias_prefixes=type_alias_prefixes,
        strict_offsets=False,
    )
    return parsed


def validate_table_response(
    tasks: list[dict],
    raw_text: str,
    allowed_types: set[str],
    require_all_scores: bool = True,
    type_aliases: dict[str, str] | None = None,
    type_alias_prefixes: dict[str, str] | None = None,
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
            normalized = _normalize_scores(
                column.scores,
                allowed_types,
                type_aliases=type_aliases,
                type_alias_prefixes=type_alias_prefixes,
            )
            if normalized is not column.scores:
                column.scores.clear()
                column.scores.update(normalized)
            _validate_scores(column.scores, allowed_types, require_all=require_all_scores)

    return parsed


# -------------------
# NEW: CPA validator
# -------------------
def validate_cpa_response(
    tasks: list[dict],
    raw_text: str,
    allowed_types: set[str],
    require_all_scores: bool = True,
    type_aliases: dict[str, str] | None = None,
    type_alias_prefixes: dict[str, str] | None = None,
) -> list[CPATaskModel]:
    """
    Validate CPA output where each task expects:
      - task_id
      - table_id
      - subject_column
      - relationships over exactly target_columns for this task
    tasks input MUST include "target_columns" list for each task for validation.
    """
    data = extract_json(raw_text)
    adapter = TypeAdapter(list[CPATaskModel])
    parsed = adapter.validate_python(data)

    task_lookup = {t["task_id"]: t for t in tasks}
    expected_ids = set(task_lookup)
    got_ids = {p.task_id for p in parsed}
    if expected_ids != got_ids:
        raise ValueError("Task IDs mismatch in CPA response")

    for item in parsed:
        expected = task_lookup[item.task_id]
        if item.table_id != expected["table_id"]:
            raise ValueError("table_id mismatch in CPA response")
        if item.subject_column != expected["subject_column"]:
            raise ValueError("subject_column mismatch in CPA response")

        expected_targets = expected.get("target_columns")
        if not isinstance(expected_targets, list) or not all(isinstance(x, str) for x in expected_targets):
            raise ValueError("CPA validation requires target_columns list in tasks input.")
        expected_target_set = set(expected_targets)

        got_targets = [rel.target_column for rel in item.relationships]
        if len(got_targets) != len(set(got_targets)):
            raise ValueError("Duplicate target_column entries in CPA response")
        if set(got_targets) != expected_target_set:
            raise ValueError("Target columns mismatch in CPA response")

        for rel in item.relationships:
            normalized = _normalize_scores(
                rel.scores,
                allowed_types,
                type_aliases=type_aliases,
                type_alias_prefixes=type_alias_prefixes,
            )
            if normalized is not rel.scores:
                rel.scores.clear()
                rel.scores.update(normalized)
            _validate_scores(rel.scores, allowed_types, require_all=require_all_scores)

    return parsed