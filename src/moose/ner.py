from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import quote, unquote

from moose.config import Settings, get_settings
from moose.prob import choose_argmax
from moose.prompts import (
    build_table_prompt,
    build_text_ner_prompt,
    build_tabular_cell_ner_prompt,
)
from moose.schema import SchemaConfig, get_schema_config
from moose.validate import (
    extract_json,
    validate_ner_response_with_warnings,
    validate_table_response,
)


def make_cell_task_id(table_task_id: str, row_index: int, column: str) -> str:
    return f"{table_task_id}:row{row_index}:col={quote(column, safe='')}"


def parse_cell_task_id(cell_task_id: str) -> tuple[str, int, str]:
    try:
        table_task_id, rest = cell_task_id.split(":row", 1)
        row_str, col_str = rest.split(":col=", 1)
        row_index = int(row_str)
        column = unquote(col_str)
        return table_task_id, row_index, column
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid cell task_id format: {cell_task_id}") from exc


_STRUCTURED_RE = re.compile(
    r"""^(
        \d{4}(-\d{2}(-\d{2})?)?              # 2024 or 2024-01 or 2024-01-17
        | \$?\d+(,\d{3})*(\.\d+)?            # money/number-ish
        | \d+(\.\d+)?                        # numeric-ish
        | [0-9a-fA-F]{8,}                    # hex-ish ids
        | [A-Z0-9_-]{8,}                     # token-ish ids
    )$""",
    re.VERBOSE,
)

def _looks_like_structured_literal(value: str) -> bool:
    v = value.strip()
    if not v:
        return True
    # Very short values like "NY" are ambiguous; don’t auto-skip those
    if len(v) <= 2:
        return False
    return bool(_STRUCTURED_RE.match(v))


_STI_NE_COARSE_TYPES = {"NE:PERSON", "NE:ORGANIZATION", "NE:LOCATION", "NE:OTHER"}


def _sti_bucket_for_fine_type(fine_coarse_type: str | None) -> str:
    if fine_coarse_type == "PERSON":
        return "NE:PERSON"
    if fine_coarse_type == "ORGANIZATION":
        return "NE:ORGANIZATION"
    if fine_coarse_type == "LOCATION":
        return "NE:LOCATION"
    return "NE:OTHER"


def _project_rows_to_columns(sampled_rows: list[dict[str, Any]], columns: list[str]) -> list[dict[str, Any]]:
    return [{column: row.get(column) for column in columns} for row in sampled_rows]


def _build_fine_type_buckets() -> dict[str, list[str]]:
    fine_schema = get_schema_config("fine")
    fine_coarse_mapping = fine_schema.coarse_mapping or {}
    buckets: dict[str, list[str]] = {
        "NE:PERSON": [],
        "NE:ORGANIZATION": [],
        "NE:LOCATION": [],
        "NE:OTHER": [],
    }
    for fine_type_id in fine_schema.load_type_ids():
        coarse = fine_coarse_mapping.get(fine_type_id)
        buckets[_sti_bucket_for_fine_type(coarse)].append(fine_type_id)
    return buckets


def _coerce_text_ner_output_with_task_ids(raw_text: str, tasks: list[dict[str, Any]]) -> str:
    data = extract_json(raw_text)
    if isinstance(data, dict):
        if len(tasks) != 1:
            raise ValueError("NER response object shape is only valid for single-task requests.")
        entities = data.get("entities")
        if not isinstance(entities, list):
            raise ValueError("NER response object must include an entities array.")
        coerced = [{"task_id": tasks[0]["task_id"], "entities": entities}]
        return json.dumps(coerced, ensure_ascii=True)
    if not isinstance(data, list):
        raise ValueError("NER response must be a JSON array.")
    if all(isinstance(item, dict) and isinstance(item.get("task_id"), str) for item in data):
        return json.dumps(data, ensure_ascii=True)
    if len(data) != len(tasks):
        raise ValueError("NER response length mismatch.")

    coerced: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError("Each NER response item must be an object.")
        entities = item.get("entities")
        if not isinstance(entities, list):
            raise ValueError("Each NER response item must include an entities array.")
        coerced.append({"task_id": tasks[index]["task_id"], "entities": entities})

    return json.dumps(coerced, ensure_ascii=True)


def _coerce_table_output_with_task_ids(raw_text: str, tasks: list[dict[str, Any]]) -> str:
    data = extract_json(raw_text)
    if not isinstance(data, list):
        raise ValueError("Table response must be a JSON array.")
    if all(isinstance(item, dict) and isinstance(item.get("task_id"), str) for item in data):
        return json.dumps(data, ensure_ascii=True)
    if len(data) != len(tasks):
        raise ValueError("Table response length mismatch.")

    coerced: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError("Each table response item must be an object.")
        columns = item.get("columns")
        if not isinstance(columns, list):
            raise ValueError("Each table response item must include a columns array.")
        table_id = item.get("table_id")
        expected_table_id = tasks[index]["table_id"]
        if not isinstance(table_id, str):
            table_id = expected_table_id
        coerced.append(
            {
                "task_id": tasks[index]["task_id"],
                "table_id": table_id,
                "columns": columns,
            }
        )

    return json.dumps(coerced, ensure_ascii=True)


def _coerce_cell_ner_output_with_task_ids(raw_text: str, tasks: list[dict[str, Any]]) -> str:
    data = extract_json(raw_text)
    if not isinstance(data, list):
        raise ValueError("Tabular cell NER response must be a JSON array.")
    if all(isinstance(item, dict) and isinstance(item.get("task_id"), str) for item in data):
        return json.dumps(data, ensure_ascii=True)
    if len(data) != len(tasks):
        raise ValueError("Tabular cell NER response length mismatch.")

    expected_by_coord: dict[tuple[str | None, int, str], str] = {}
    expected_ids: set[str] = set()
    for task in tasks:
        row_index = task.get("row_index")
        column = task.get("column")
        task_id = task.get("task_id")
        if not isinstance(row_index, int) or not isinstance(column, str) or not isinstance(task_id, str):
            raise ValueError("Invalid internal cell task metadata.")
        table_id = task.get("table_id")
        key = (table_id if isinstance(table_id, str) else None, row_index, column)
        expected_by_coord[key] = task_id
        expected_ids.add(task_id)

    coerced: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError("Each tabular cell NER response item must be an object.")
        entities = item.get("entities")
        if not isinstance(entities, list):
            raise ValueError("Each tabular cell NER response item must include an entities array.")

        task_id = item.get("task_id")
        if isinstance(task_id, str):
            resolved_task_id = task_id
        else:
            row_index = item.get("row_index")
            column = item.get("column")
            table_id = item.get("table_id")
            table_key = table_id if isinstance(table_id, str) else None
            if isinstance(row_index, int) and isinstance(column, str):
                resolved_task_id = expected_by_coord.get((table_key, row_index, column))
                if resolved_task_id is None:
                    resolved_task_id = expected_by_coord.get((None, row_index, column))
            else:
                resolved_task_id = tasks[index]["task_id"]
            if not isinstance(resolved_task_id, str):
                raise ValueError("Unknown cell identifier in tabular cell NER response.")

        if resolved_task_id in seen_ids:
            raise ValueError("Duplicate cell output in tabular cell NER response.")
        seen_ids.add(resolved_task_id)
        coerced.append({"task_id": resolved_task_id, "entities": entities})

    if seen_ids != expected_ids:
        raise ValueError("Tabular cell NER response items mismatch.")

    return json.dumps(coerced, ensure_ascii=True)


async def _run_with_retries(llm_client, prompt: str, validator, max_retries: int) -> Any:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        if attempt == 0:
            response = await llm_client.generate(prompt)
        else:
            correction = (
                "\n\nThe previous output was invalid: "
                f"{last_error}. Return ONLY valid JSON following the schema."
            )
            response = await llm_client.generate(prompt + correction)
        try:
            return validator(response)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise ValueError(f"LLM output invalid after {max_retries} retries: {last_error}")


async def run_text_ner(
    tasks: list[dict],
    schema: str,
    llm_client,
    settings: Settings | None = None,
) -> dict:
    settings = settings or get_settings()
    schema_config = get_schema_config(schema)
    if not schema_config.supports_text:
        raise ValueError(f"Schema '{schema}' does not support text annotation.")

    type_ids = schema_config.load_type_ids()
    require_all_scores = schema_config.require_all_scores

    task_lookup = {task["task_id"]: task for task in tasks}
    results_by_id: dict[str, dict] = {}
    all_warnings: list[dict[str, Any]] = []

    selected_type_ids = type_ids
    type_set = set(selected_type_ids)
    prompt = build_text_ner_prompt(schema_config, tasks, selected_type_ids)

    def validator(raw_text: str):
        normalized_raw = _coerce_text_ner_output_with_task_ids(raw_text, tasks)
        return validate_ner_response_with_warnings(
            [{"task_id": t["task_id"], "text": t["text"]} for t in tasks],
            normalized_raw,
            type_set,
            require_all_scores=require_all_scores,
            type_aliases=schema_config.type_aliases,
            type_alias_prefixes=schema_config.type_alias_prefixes,
            strict_offsets=False,
        )

    parsed, warnings = await _run_with_retries(llm_client, prompt, validator, settings.MOOSE_MAX_RETRIES)
    all_warnings.extend(warnings)

    for item in parsed:
        original_text = task_lookup[item.task_id]["text"]
        entities = []
        for entity in item.entities:
            scores = {type_id: float(entity.scores.get(type_id, 0)) for type_id in selected_type_ids}
            type_id, confidence, _distribution = choose_argmax(scores)
            output = {
                "start": entity.start,
                "end": entity.end,
                "text": original_text[entity.start : entity.end],
                "type_id": type_id,
                "confidence": confidence,
            }
            if schema_config.coarse_mapping:
                output["coarse_type_id"] = schema_config.coarse_mapping.get(type_id)
            entities.append(output)
        results_by_id[item.task_id] = {"task_id": item.task_id, "entities": entities}

    ordered = [results_by_id[task["task_id"]] for task in tasks]
    response: dict[str, Any] = {"results": ordered}
    if all_warnings:
        response["warnings"] = all_warnings
    return response


async def _run_table_annotate_once(
    tasks: list[dict[str, Any]],
    schema_config: SchemaConfig,
    llm_client,
    settings: Settings,
    selected_type_ids: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    type_ids = schema_config.load_type_ids() if selected_type_ids is None else selected_type_ids
    type_set = set(type_ids)
    prompt = build_table_prompt(schema_config, tasks, type_ids)

    def validator(raw_text: str):
        normalized_raw = _coerce_table_output_with_task_ids(raw_text, tasks)
        return validate_table_response(
            tasks,
            normalized_raw,
            type_set,
            require_all_scores=schema_config.require_all_scores,
            type_aliases=schema_config.type_aliases,
            type_alias_prefixes=schema_config.type_alias_prefixes,
        )

    parsed = await _run_with_retries(llm_client, prompt, validator, settings.MOOSE_MAX_RETRIES)

    task_lookup = {task["task_id"]: task for task in tasks}
    results_by_id: dict[str, dict[str, Any]] = {}
    for item in parsed:
        task = task_lookup[item.task_id]
        columns: list[dict[str, Any]] = []
        for column in item.columns:
            scores = {type_id: float(column.scores.get(type_id, 0)) for type_id in type_ids}
            type_id, confidence, _distribution = choose_argmax(scores)
            output = {"column": column.column, "type_id": type_id, "confidence": confidence}
            if schema_config.coarse_mapping:
                output["coarse_type_id"] = schema_config.coarse_mapping.get(type_id)
            columns.append(output)
        results_by_id[item.task_id] = {
            "task_id": item.task_id,
            "table_id": task["table_id"],
            "columns": columns,
        }
    return results_by_id


async def _augment_sti_ne_columns_with_fine_types(
    tasks: list[dict[str, Any]],
    results_by_id: dict[str, dict[str, Any]],
    llm_client,
    settings: Settings,
) -> None:
    fine_schema = get_schema_config("fine")
    fine_types_by_bucket = _build_fine_type_buckets()

    bucket_tasks: dict[str, list[dict[str, Any]]] = {
        "NE:PERSON": [],
        "NE:ORGANIZATION": [],
        "NE:LOCATION": [],
        "NE:OTHER": [],
    }
    bucket_columns: dict[str, dict[str, set[str]]] = {
        "NE:PERSON": {},
        "NE:ORGANIZATION": {},
        "NE:LOCATION": {},
        "NE:OTHER": {},
    }

    for task in tasks:
        task_id = task["task_id"]
        task_result = results_by_id.get(task_id)
        if not isinstance(task_result, dict):
            continue
        columns = task_result.get("columns")
        if not isinstance(columns, list):
            continue

        per_bucket: dict[str, list[str]] = {}
        for col in columns:
            col_name = col.get("column")
            if not isinstance(col_name, str):
                continue
            coarse_type = col.get("coarse_type_id")
            if not isinstance(coarse_type, str):
                continue
            if coarse_type not in _STI_NE_COARSE_TYPES:
                continue
            per_bucket.setdefault(coarse_type, []).append(col_name)

        for bucket, cols in per_bucket.items():
            projected_rows = _project_rows_to_columns(task["sampled_rows"], cols)
            bucket_tasks[bucket].append(
                {
                    "task_id": task_id,
                    "table_id": task["table_id"],
                    "sampled_rows": projected_rows,
                }
            )
            bucket_columns[bucket][task_id] = set(cols)

    for bucket, projected_tasks in bucket_tasks.items():
        if not projected_tasks:
            continue
        selected_fine_types = fine_types_by_bucket.get(bucket, [])
        if not selected_fine_types:
            continue
        try:
            fine_results_by_id = await _run_table_annotate_once(
                projected_tasks,
                fine_schema,
                llm_client,
                settings,
                selected_type_ids=selected_fine_types,
            )
        except Exception:  # noqa: BLE001
            # Keep STI response stable even if enrichment fails.
            continue

        for task_id, projected_cols in bucket_columns[bucket].items():
            fine_task = fine_results_by_id.get(task_id, {})
            fine_columns = fine_task.get("columns", [])
            if not isinstance(fine_columns, list):
                continue

            fine_by_col: dict[str, dict[str, Any]] = {}
            for fine_col in fine_columns:
                fine_col_name = fine_col.get("column")
                if isinstance(fine_col_name, str):
                    fine_by_col[fine_col_name] = fine_col

            original_task = results_by_id.get(task_id, {})
            original_columns = original_task.get("columns", [])
            if not isinstance(original_columns, list):
                continue
            for original_col in original_columns:
                col_name = original_col.get("column")
                if not isinstance(col_name, str) or col_name not in projected_cols:
                    continue
                fine_match = fine_by_col.get(col_name)
                if not fine_match:
                    continue
                fine_type_id = fine_match.get("type_id")
                fine_confidence = fine_match.get("confidence")
                if isinstance(fine_type_id, str):
                    original_col["fine_type_id"] = fine_type_id
                if isinstance(fine_confidence, (int, float)):
                    original_col["fine_confidence"] = float(fine_confidence)


async def run_table_annotate(
    tasks: list[dict],
    schema: str,
    llm_client,
    settings: Settings | None = None,
) -> dict:
    settings = settings or get_settings()
    schema_config = get_schema_config(schema)
    if not schema_config.supports_table:
        raise ValueError(f"Schema '{schema}' does not support tabular annotation.")

    results_by_id = await _run_table_annotate_once(tasks, schema_config, llm_client, settings)

    if schema_config.name == "sti":
        await _augment_sti_ne_columns_with_fine_types(tasks, results_by_id, llm_client, settings)

    ordered = [results_by_id[task["task_id"]] for task in tasks]
    return {"results": ordered}


async def run_tabular_ner(
    tasks: list[dict],
    schema: str,
    llm_client,
    settings: Settings | None = None,
) -> dict:
    settings = settings or get_settings()
    schema_config = get_schema_config(schema)
    if not schema_config.supports_text:
        raise ValueError(f"Schema '{schema}' does not support text annotation.")

    # Explode into cell tasks
    cell_tasks: list[dict[str, Any]] = []
    cell_index: dict[str, dict[str, Any]] = {}  # cell_task_id -> metadata
    pre_warnings: list[dict[str, Any]] = []

    # Also prepare an “empty but structured” output scaffold up front,
    # so even if we send nothing to the LLM we still return stable results.
    results_scaffold: list[dict[str, Any]] = []

    for table_task in tasks:
        table_task_id = table_task["task_id"]
        table_id = table_task["table_id"]
        sampled_rows = table_task["sampled_rows"]
        target_columns = table_task.get("target_columns")
        if not target_columns:
            raise ValueError(f"tabular/ner requires target_columns for task_id={table_task_id}")

        strings_only = bool(table_task.get("strings_only", True))
        skip_structured = bool(table_task.get("skip_structured_literals", True))

        rows_out: list[dict[str, Any]] = []
        for row_index, row in enumerate(sampled_rows):
            cells_out: list[dict[str, Any]] = []
            for col in target_columns:
                # Create scaffold entry no matter what
                cells_out.append({"column": col, "entities": []})

                if col not in row:
                    pre_warnings.append(
                        {
                            "task_id": make_cell_task_id(table_task_id, row_index, col),
                            "code": "missing_target_column",
                            "table_task_id": table_task_id,
                            "table_id": table_id,
                            "row_index": row_index,
                            "column": col,
                        }
                    )

                value = row.get(col)
                if value is None:
                    text = ""
                else:
                    if strings_only:
                        text = value if isinstance(value, str) else ""
                    else:
                        text = value if isinstance(value, str) else str(value)

                if isinstance(text, str) and skip_structured and _looks_like_structured_literal(text.strip()):
                    text_for_task = ""
                else:
                    text_for_task = text if isinstance(text, str) else ""

                cell_task_id = make_cell_task_id(table_task_id, row_index, col)
                cell_index[cell_task_id] = {
                    "table_task_id": table_task_id,
                    "table_id": table_id,
                    "row_index": row_index,
                    "column": col,
                }

                if text_for_task:
                    cell_tasks.append(
                        {
                            "task_id": cell_task_id,
                            "text": text_for_task,
                            "table_id": table_id,
                            "row_index": row_index,
                            "column": col,
                        }
                    )

            rows_out.append({"row_index": row_index, "cells": cells_out})

        results_scaffold.append({"task_id": table_task_id, "table_id": table_id, "rows": rows_out})

    # If there is nothing to process, return scaffold + any pre_warnings
    if not cell_tasks:
        response: dict[str, Any] = {"results": results_scaffold}
        if pre_warnings:
            response["warnings"] = pre_warnings
        return response

    type_ids = schema_config.load_type_ids()
    require_all_scores = schema_config.require_all_scores

    results_by_cell_id: dict[str, list[dict[str, Any]]] = {}
    all_warnings: list[dict[str, Any]] = []

    selected_type_ids = type_ids
    type_set = set(selected_type_ids)

    prompt = build_tabular_cell_ner_prompt(schema_config, cell_tasks, selected_type_ids)

    batch_text_by_id = {t["task_id"]: t["text"] for t in cell_tasks}

    def validator(raw_text: str):
        base_tasks = [{"task_id": t["task_id"], "text": t["text"]} for t in cell_tasks]
        normalized_raw = _coerce_cell_ner_output_with_task_ids(raw_text, cell_tasks)
        return validate_ner_response_with_warnings(
            base_tasks,
            normalized_raw,
            type_set,
            require_all_scores=require_all_scores,
            type_aliases=schema_config.type_aliases,
            type_alias_prefixes=schema_config.type_alias_prefixes,
            strict_offsets=False,
        )

    parsed, warnings = await _run_with_retries(llm_client, prompt, validator, settings.MOOSE_MAX_RETRIES)
    all_warnings.extend(warnings)

    for item in parsed:
        text_for_cell = batch_text_by_id.get(item.task_id, "")
        entities_out: list[dict[str, Any]] = []

        for entity in item.entities:
            scores = {type_id: float(entity.scores.get(type_id, 0)) for type_id in selected_type_ids}
            type_id, confidence, _distribution = choose_argmax(scores)
            output = {
                "start": entity.start,
                "end": entity.end,
                "text": text_for_cell[entity.start : entity.end],
                "type_id": type_id,
                "confidence": confidence,
            }
            if schema_config.coarse_mapping:
                output["coarse_type_id"] = schema_config.coarse_mapping.get(type_id)
            entities_out.append(output)

        results_by_cell_id[item.task_id] = entities_out

    # Fill scaffold with extracted entities
    for table_task_out in results_scaffold:
        table_task_id = table_task_out["task_id"]
        for row in table_task_out["rows"]:
            row_index = row["row_index"]
            for cell in row["cells"]:
                col = cell["column"]
                cell_id = make_cell_task_id(table_task_id, row_index, col)
                cell["entities"] = results_by_cell_id.get(cell_id, [])

    # Rewrite warnings to include metadata
    warnings_out: list[dict[str, Any]] = []
    for w in all_warnings:
        cell_id = w.get("task_id")
        if isinstance(cell_id, str) and cell_id in cell_index:
            warnings_out.append({**w, **cell_index[cell_id]})
        else:
            warnings_out.append(w)

    # Include pre-warnings (missing columns, etc.)
    warnings_out = pre_warnings + warnings_out

    response: dict[str, Any] = {"results": results_scaffold}
    if warnings_out:
        response["warnings"] = warnings_out
    return response
