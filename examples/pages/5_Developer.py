from __future__ import annotations

import json
import re
from typing import Any

import streamlit as st

from moose.prompts import (
    build_table_prompt,
    build_tabular_cell_ner_prompt,
    build_text_ner_prompt,
)
from moose.schema import get_schema_config
from moose_ui.api import api_get
from moose_ui.config import sidebar
from moose_ui.metadata import (
    fetch_assets,
    fetch_policy_packs,
    fetch_privacy_profiles,
    fetch_schemas,
    schemas_supporting,
)
from moose_ui.samples import DEFAULT_TABLE_SAMPLE, DEFAULT_TEXT_SAMPLE

_SINGLE_TABLE_ID = "__single_table__"
_STRUCTURED_RE = re.compile(
    r"""^(
        \d{4}(-\d{2}(-\d{2})?)?
        | \$?\d+(,\d{3})*(\.\d+)?
        | \d+(\.\d+)?
        | [0-9a-fA-F]{8,}
        | [A-Z0-9_-]{8,}
    )$""",
    re.VERBOSE,
)


def _looks_like_structured_literal(value: str) -> bool:
    v = value.strip()
    if not v:
        return True
    if len(v) <= 2:
        return False
    return bool(_STRUCTURED_RE.match(v))


def _infer_table_columns(sampled_rows: Any) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    if not isinstance(sampled_rows, list):
        return columns
    for row in sampled_rows:
        if not isinstance(row, dict):
            continue
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    return columns


def _build_cell_tasks(
    table_id: str,
    sampled_rows: list[dict[str, Any]],
    target_columns: list[str],
    strings_only: bool,
    skip_structured_literals: bool,
) -> list[dict[str, Any]]:
    cell_tasks: list[dict[str, Any]] = []
    for row_index, row in enumerate(sampled_rows):
        if not isinstance(row, dict):
            continue
        for col in target_columns:
            value = row.get(col)
            if value is None:
                text = ""
            elif strings_only:
                text = value if isinstance(value, str) else ""
            else:
                text = value if isinstance(value, str) else str(value)

            if isinstance(text, str) and skip_structured_literals and _looks_like_structured_literal(
                text.strip()
            ):
                continue
            if not isinstance(text, str) or not text:
                continue

            cell_tasks.append(
                {
                    "table_id": table_id,
                    "row_index": row_index,
                    "column": col,
                    "text": text,
                }
            )
    return cell_tasks


def _show_prompt(prompt: str, *, schema: str, task_count: int, type_count: int) -> None:
    st.success(
        f"Prompt ready ({len(prompt):,} chars, schema={schema}, tasks={task_count}, type_ids={type_count:,})."
    )
    st.download_button(
        "Download prompt",
        data=prompt,
        file_name=f"prompt_{schema}.txt",
        mime="text/plain",
    )
    st.text_area("Generated prompt", value=prompt, height=420)

st.title("Developer")

cfg = sidebar()
if not cfg.get("developer_mode"):
    st.info("Enable Developer mode in the sidebar to use this page.")
    st.stop()

if not cfg.get("api_key"):
    st.error("Moose API key required.")
    st.stop()

api_tab, prompt_tab = st.tabs(["API Tools", "Prompt Debugger"])

with api_tab:
    st.subheader("Health")
    if st.button("GET /health"):
        st.json(api_get(cfg["base_url"], cfg["api_key"], "/health"))

    st.subheader("Schemas")
    if st.button("GET /schemas?include_type_count=true"):
        st.json(fetch_schemas(cfg["base_url"], cfg["api_key"]))

    st.subheader("Policy packs")
    if st.button("GET /policy-packs"):
        st.json(fetch_policy_packs(cfg["base_url"], cfg["api_key"]))

    st.subheader("Privacy profiles")
    if st.button("GET /privacy/profiles"):
        st.json(fetch_privacy_profiles(cfg["base_url"], cfg["api_key"]))

    st.subheader("Assets index")
    if st.button("GET /assets"):
        st.json(fetch_assets(cfg["base_url"], cfg["api_key"]))

    st.subheader("OpenAPI routes")
    if st.button("GET /openapi.json"):
        st.json(api_get(cfg["base_url"], cfg["api_key"], "/openapi.json"))

with prompt_tab:
    st.caption("Build the exact prompt template from local Moose prompt builders for a concrete input.")

    try:
        schemas = fetch_schemas(cfg["base_url"], cfg["api_key"])
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to fetch schemas: {exc}")
        st.stop()

    mode = st.radio(
        "Operation",
        ["Text NER", "Column Typing", "Cell NER"],
        horizontal=True,
    )

    if mode == "Text NER":
        schema_names = schemas_supporting(schemas, text=True)
        if not schema_names:
            st.error("No text-capable schemas available.")
            st.stop()
        preferred = (
            "dpv_pd"
            if "dpv_pd" in schema_names
            else ("dpv" if "dpv" in schema_names else schema_names[0])
        )
        schema = st.selectbox(
            "Schema",
            schema_names,
            index=schema_names.index(preferred),
            key="prompt_debug_text_schema",
        )
        text = st.text_area(
            "Text",
            value=DEFAULT_TEXT_SAMPLE,
            height=180,
            key="prompt_debug_text_value",
        )
        if st.button("Build prompt", key="prompt_debug_build_text"):
            if not text:
                st.error("Provide non-empty text.")
                st.stop()
            schema_config = get_schema_config(schema)
            type_ids = schema_config.load_type_ids()
            tasks = [{"text": text}]
            prompt = build_text_ner_prompt(schema_config, tasks, type_ids)
            _show_prompt(prompt, schema=schema, task_count=len(tasks), type_count=len(type_ids))

    elif mode == "Column Typing":
        schema_names = schemas_supporting(schemas, table=True)
        if not schema_names:
            st.error("No table-capable schemas available.")
            st.stop()
        preferred = (
            "sti"
            if "sti" in schema_names
            else ("dpv_pd" if "dpv_pd" in schema_names else schema_names[0])
        )
        schema = st.selectbox(
            "Schema",
            schema_names,
            index=schema_names.index(preferred),
            key="prompt_debug_table_schema",
        )
        table_id = st.text_input("table_id", value="debug-table", key="prompt_debug_table_id")
        sampled_rows_raw = st.text_area(
            "sampled_rows JSON",
            value=DEFAULT_TABLE_SAMPLE,
            height=220,
            key="prompt_debug_table_rows",
        )
        try:
            sampled_rows = json.loads(sampled_rows_raw)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Invalid JSON: {exc}")
            sampled_rows = None

        if st.button("Build prompt", key="prompt_debug_build_table"):
            if not isinstance(sampled_rows, list) or not all(
                isinstance(row, dict) for row in sampled_rows
            ):
                st.error("sampled_rows must be a JSON array of objects.")
                st.stop()
            schema_config = get_schema_config(schema)
            type_ids = schema_config.load_type_ids()
            tasks = [
                {
                    "table_id": table_id or _SINGLE_TABLE_ID,
                    "sampled_rows": sampled_rows,
                }
            ]
            prompt = build_table_prompt(schema_config, tasks, type_ids)
            _show_prompt(prompt, schema=schema, task_count=len(tasks), type_count=len(type_ids))

    else:
        schema_names = schemas_supporting(schemas, text=True)
        if not schema_names:
            st.error("No text-capable schemas available for cell NER.")
            st.stop()
        preferred = (
            "dpv_pd"
            if "dpv_pd" in schema_names
            else ("dpv" if "dpv" in schema_names else schema_names[0])
        )
        schema = st.selectbox(
            "Schema",
            schema_names,
            index=schema_names.index(preferred),
            key="prompt_debug_cell_schema",
        )
        table_id = st.text_input("table_id", value="debug-table", key="prompt_debug_cell_table_id")
        sampled_rows_raw = st.text_area(
            "sampled_rows JSON",
            value=DEFAULT_TABLE_SAMPLE,
            height=220,
            key="prompt_debug_cell_rows",
        )

        sampled_rows: Any = None
        columns: list[str] = []
        try:
            sampled_rows = json.loads(sampled_rows_raw)
            columns = _infer_table_columns(sampled_rows)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Invalid JSON: {exc}")

        target_columns = st.multiselect(
            "target_columns",
            options=columns,
            default=columns[:2],
            key="prompt_debug_cell_targets",
        )
        col1, col2 = st.columns([1, 1])
        with col1:
            strings_only = st.checkbox("strings_only", value=True, key="prompt_debug_cell_strings")
        with col2:
            skip_structured = st.checkbox(
                "skip_structured_literals",
                value=True,
                key="prompt_debug_cell_skip_structured",
            )

        if st.button("Build prompt", key="prompt_debug_build_cell"):
            if not isinstance(sampled_rows, list) or not all(
                isinstance(row, dict) for row in sampled_rows
            ):
                st.error("sampled_rows must be a JSON array of objects.")
                st.stop()
            if not target_columns:
                st.error("Select at least one target column.")
                st.stop()

            resolved_table_id = table_id or _SINGLE_TABLE_ID
            cell_tasks = _build_cell_tasks(
                table_id=resolved_table_id,
                sampled_rows=sampled_rows,
                target_columns=target_columns,
                strings_only=strings_only,
                skip_structured_literals=skip_structured,
            )
            if not cell_tasks:
                st.warning(
                    "No cells remained after current filters (strings_only/skip_structured_literals)."
                )
                st.stop()

            schema_config = get_schema_config(schema)
            type_ids = schema_config.load_type_ids()
            prompt = build_tabular_cell_ner_prompt(schema_config, cell_tasks, type_ids)
            _show_prompt(prompt, schema=schema, task_count=len(cell_tasks), type_count=len(type_ids))
