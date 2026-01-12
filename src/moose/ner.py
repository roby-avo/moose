from __future__ import annotations

import json
from typing import Any

from moose.config import Settings, get_settings
from moose.prob import choose_argmax
from moose.prompts import (
    build_table_prompt,
    build_text_ner_prompt,
    build_type_selection_prompt,
)
from moose.schema import get_schema_config
from moose.validate import (
    validate_ner_response,
    validate_table_response,
    validate_type_selection_response,
)


def _estimate_task_size(task: dict) -> int:
    return len(json.dumps(task, ensure_ascii=True))


def _batch_tasks(
    tasks: list[dict], max_tasks: int, max_chars: int
) -> list[list[dict]]:
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


def _batch_type_ids_for_prompt(
    schema_config,
    tasks: list[dict],
    type_ids: list[str],
    mode: str,
    max_chars: int,
) -> list[list[str]]:
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


async def _select_type_ids(
    schema_config,
    tasks: list[dict],
    type_ids: list[str],
    llm_client,
    settings: Settings,
    mode: str,
) -> list[str]:
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

        extracted = await _run_with_retries(
            llm_client, prompt, validator, settings.MOOSE_MAX_RETRIES
        )
        selected.update(extracted)
    if not selected:
        return type_ids
    return [type_id for type_id in type_ids if type_id in selected]


async def _run_with_retries(
    llm_client,
    prompt: str,
    validator,
    max_retries: int,
) -> Any:
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
    include_scores: bool = False,
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

    batches = _batch_tasks(
        tasks,
        max_tasks=settings.MOOSE_MAX_TASKS_PER_PROMPT,
        max_chars=settings.MOOSE_MAX_CHARS_PER_PROMPT,
    )

    for batch in batches:
        selected_type_ids = await _select_type_ids(
            schema_config,
            batch,
            type_ids,
            llm_client,
            settings,
            mode="text",
        )
        type_set = set(selected_type_ids)
        prompt = build_text_ner_prompt(schema_config, batch, selected_type_ids)

        def validator(raw_text: str):
            return validate_ner_response(
                batch,
                raw_text,
                type_set,
                require_all_scores=require_all_scores,
                type_aliases=schema_config.type_aliases,
                type_alias_prefixes=schema_config.type_alias_prefixes,
            )

        parsed = await _run_with_retries(
            llm_client, prompt, validator, settings.MOOSE_MAX_RETRIES
        )

        for item in parsed:
            text = task_lookup[item.task_id]["text"]
            entities = []
            for entity in item.entities:
                scores = {
                    type_id: float(entity.scores.get(type_id, 0))
                    for type_id in selected_type_ids
                }
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
    return {"results": ordered}


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
        selected_type_ids = await _select_type_ids(
            schema_config,
            batch,
            type_ids,
            llm_client,
            settings,
            mode="table",
        )
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

        parsed = await _run_with_retries(
            llm_client, prompt, validator, settings.MOOSE_MAX_RETRIES
        )

        for item in parsed:
            task = task_lookup[item.task_id]
            columns = []
            for column in item.columns:
                scores = {
                    type_id: float(column.scores.get(type_id, 0))
                    for type_id in selected_type_ids
                }
                type_id, confidence, distribution = choose_argmax(scores)
                output = {
                    "column": column.column,
                    "type_id": type_id,
                    "confidence": confidence,
                }
                if schema_config.coarse_mapping:
                    output["coarse_type_id"] = schema_config.coarse_mapping.get(type_id)
                if include_scores:
                    output["distribution"] = distribution
                columns.append(output)
            results_by_id[item.task_id] = {
                "task_id": item.task_id,
                "table_id": task["table_id"],
                "columns": columns,
            }

    ordered = [results_by_id[task["task_id"]] for task in tasks]
    return {"results": ordered}
