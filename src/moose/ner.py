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
    build_type_selection_prompt,
)
from moose.schema import get_schema_config
from moose.validate import (
    validate_ner_response_with_warnings,
    validate_table_response,
    validate_type_selection_response,
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


def _estimate_task_size(task: dict) -> int:
    return len(json.dumps(task, ensure_ascii=True))


def _batch_tasks(tasks: list[dict], max_tasks: int, max_chars: int) -> list[list[dict]]:
    batches: list[list[dict]] = []
    current: list[dict] = []
    current_chars = 0
    for task in tasks:
        task_size = _estimate_task_size(task)
        if current and (len(current) + 1 > max_tasks or current_chars + task_size > max_chars):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(task)
        current_chars += task_size
    if current:
        batches.append(current)
    return batches


def _batch_type_ids_for_prompt(schema_config, tasks: list[dict], type_ids: list[str], mode: str, max_chars: int) -> list[list[str]]:
    batches: list[list[str]] = []
    current: list[str] = []
    for type_id in type_ids:
        candidate = current + [type_id]
        prompt = build_type_selection_prompt(schema_config, tasks, candidate, mode)
        if current and len(prompt) > max_chars:
            batches.append(current)
            current = [type_id]
        else:
            current = candidate
    if current:
        batches.append(current)
    return batches


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


async def _select_type_ids(schema_config, tasks: list[dict], type_ids: list[str], llm_client, settings: Settings, mode: str) -> list[str]:
    if not schema_config.prefilter_types:
        return type_ids

    selected: set[str] = set()
    type_batches = _batch_type_ids_for_prompt(
        schema_config,
        tasks,
        type_ids,
        mode,
        settings.MOOSE_MAX_CHARS_PER_PROMPT,
    )
    for type_batch in type_batches:
        prompt = build_type_selection_prompt(schema_config, tasks, type_batch, mode)

        def validator(raw_text: str):
            return validate_type_selection_response(
                raw_text,
                set(type_batch),
                type_aliases=schema_config.type_aliases,
                type_alias_prefixes=schema_config.type_alias_prefixes,
            )

        extracted = await _run_with_retries(llm_client, prompt, validator, settings.MOOSE_MAX_RETRIES)
        selected.update(extracted)

    if not selected:
        return type_ids
    return [type_id for type_id in type_ids if type_id in selected]


async def run_text_ner(
    tasks: list[dict],
    schema: str,
    llm_client,
    include_scores: bool = False,
    strict_offsets: bool = False,
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

    batches = _batch_tasks(
        tasks,
        max_tasks=settings.MOOSE_MAX_TASKS_PER_PROMPT,
        max_chars=settings.MOOSE_MAX_CHARS_PER_PROMPT,
    )

    for batch in batches:
        selected_type_ids = await _select_type_ids(schema_config, batch, type_ids, llm_client, settings, mode="text")
        type_set = set(selected_type_ids)
        prompt = build_text_ner_prompt(schema_config, batch, selected_type_ids)

        def validator(raw_text: str):
            return validate_ner_response_with_warnings(
                [{"task_id": t["task_id"], "text": t["text"]} for t in batch],
                raw_text,
                type_set,
                require_all_scores=require_all_scores,
                type_aliases=schema_config.type_aliases,
                type_alias_prefixes=schema_config.type_alias_prefixes,
                strict_offsets=strict_offsets,
            )

        parsed, warnings = await _run_with_retries(llm_client, prompt, validator, settings.MOOSE_MAX_RETRIES)
        all_warnings.extend(warnings)

        for item in parsed:
            text = task_lookup[item.task_id]["text"]
            entities = []
            for entity in item.entities:
                scores = {type_id: float(entity.scores.get(type_id, 0)) for type_id in selected_type_ids}
                type_id, confidence, distribution = choose_argmax(scores)
                output = {
                    "start": entity.start,
                    "end": entity.end,
                    "text": text[entity.start : entity.end],
                    "type_id": type_id,
                    "confidence": confidence,
                }
                if schema_config.coarse_mapping:
                    output["coarse_type_id"] = schema_config.coarse_mapping.get(type_id)
                if include_scores:
                    output["distribution"] = distribution
                entities.append(output)
            results_by_id[item.task_id] = {"task_id": item.task_id, "entities": entities}

    ordered = [results_by_id[task["task_id"]] for task in tasks]
    response: dict[str, Any] = {"results": ordered}
    if all_warnings:
        response["warnings"] = all_warnings
    return response


async def run_table_annotate(
    tasks: list[dict],
    schema: str,
    llm_client,
    include_scores: bool = False,
    settings: Settings | None = None,
) -> dict:
    settings = settings or get_settings()
    schema_config = get_schema_config(schema)
    if not schema_config.supports_table:
        raise ValueError(f"Schema '{schema}' does not support tabular annotation.")

    type_ids = schema_config.load_type_ids()
    require_all_scores = schema_config.require_all_scores

    task_lookup = {task["task_id"]: task for task in tasks}
    results_by_id: dict[str, dict] = {}

    batches = _batch_tasks(
        tasks,
        max_tasks=settings.MOOSE_MAX_TASKS_PER_PROMPT,
        max_chars=settings.MOOSE_MAX_CHARS_PER_PROMPT,
    )

    for batch in batches:
        selected_type_ids = await _select_type_ids(schema_config, batch, type_ids, llm_client, settings, mode="table")
        type_set = set(selected_type_ids)
        prompt = build_table_prompt(schema_config, batch, selected_type_ids)

        def validator(raw_text: str):
            return validate_table_response(
                batch,
                raw_text,
                type_set,
                require_all_scores=require_all_scores,
                type_aliases=schema_config.type_aliases,
                type_alias_prefixes=schema_config.type_alias_prefixes,
            )

        parsed = await _run_with_retries(llm_client, prompt, validator, settings.MOOSE_MAX_RETRIES)

        for item in parsed:
            task = task_lookup[item.task_id]
            columns = []
            for column in item.columns:
                scores = {type_id: float(column.scores.get(type_id, 0)) for type_id in selected_type_ids}
                type_id, confidence, distribution = choose_argmax(scores)
                output = {"column": column.column, "type_id": type_id, "confidence": confidence}
                if schema_config.coarse_mapping:
                    output["coarse_type_id"] = schema_config.coarse_mapping.get(type_id)
                if include_scores:
                    output["distribution"] = distribution
                columns.append(output)
            results_by_id[item.task_id] = {"task_id": item.task_id, "table_id": task["table_id"], "columns": columns}

    ordered = [results_by_id[task["task_id"]] for task in tasks]
    return {"results": ordered}


async def run_tabular_ner(
    tasks: list[dict],
    schema: str,
    llm_client,
    include_scores: bool = False,
    strict_offsets: bool = False,
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

                if isinstance(text, str):
                    text = text.strip()

                if isinstance(text, str) and skip_structured and _looks_like_structured_literal(text):
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

    batches = _batch_tasks(
        cell_tasks,
        max_tasks=settings.MOOSE_MAX_TASKS_PER_PROMPT,
        max_chars=settings.MOOSE_MAX_CHARS_PER_PROMPT,
    )

    for batch in batches:
        selected_type_ids = await _select_type_ids(schema_config, batch, type_ids, llm_client, settings, mode="text")
        type_set = set(selected_type_ids)

        prompt = build_tabular_cell_ner_prompt(schema_config, batch, selected_type_ids)

        batch_text_by_id = {t["task_id"]: t["text"] for t in batch}

        def validator(raw_text: str):
            base_tasks = [{"task_id": t["task_id"], "text": t["text"]} for t in batch]
            return validate_ner_response_with_warnings(
                base_tasks,
                raw_text,
                type_set,
                require_all_scores=require_all_scores,
                type_aliases=schema_config.type_aliases,
                type_alias_prefixes=schema_config.type_alias_prefixes,
                strict_offsets=strict_offsets,
            )

        parsed, warnings = await _run_with_retries(llm_client, prompt, validator, settings.MOOSE_MAX_RETRIES)
        all_warnings.extend(warnings)

        for item in parsed:
            text_for_cell = batch_text_by_id.get(item.task_id, "")
            entities_out: list[dict[str, Any]] = []

            for entity in item.entities:
                scores = {type_id: float(entity.scores.get(type_id, 0)) for type_id in selected_type_ids}
                type_id, confidence, distribution = choose_argmax(scores)
                output = {
                    "start": entity.start,
                    "end": entity.end,
                    "text": text_for_cell[entity.start : entity.end],
                    "type_id": type_id,
                    "confidence": confidence,
                }
                if schema_config.coarse_mapping:
                    output["coarse_type_id"] = schema_config.coarse_mapping.get(type_id)
                if include_scores:
                    output["distribution"] = distribution
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
