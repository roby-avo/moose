from __future__ import annotations

import json
from typing import Any

import streamlit as st

from moose_ui.config import build_llm_headers, sidebar, validate_common
from moose_ui.metadata import fetch_schemas, schemas_supporting
from moose_ui.samples import DEFAULT_CPA_TABLE_SAMPLE, DEFAULT_TABLE_SAMPLE
from moose_ui.submit import submit_and_render_job


def _infer_table_columns(sampled_rows: Any) -> list[str]:
    cols: list[str] = []
    seen: set[str] = set()
    if not isinstance(sampled_rows, list):
        return cols
    for row in sampled_rows:
        if not isinstance(row, dict):
            continue
        for k in row.keys():
            if k not in seen:
                cols.append(k)
                seen.add(k)
    return cols


st.title("Tables")

cfg = sidebar()
err = validate_common(cfg)
if err:
    st.warning(err)

if not cfg.get("api_key"):
    st.info("Enter your Moose API key in the sidebar.")
    st.stop()

st.caption(f"Provider: {cfg['provider']} | Model: {cfg['model']}")

# Metadata
try:
    schemas = fetch_schemas(cfg["base_url"], cfg["api_key"])
except Exception as exc:  # noqa: BLE001
    st.error(f"Failed to fetch schemas: {exc}")
    st.stop()

table_schema_names = schemas_supporting(schemas, table=True)
text_schema_names = schemas_supporting(schemas, text=True)
cpa_schema_names = schemas_supporting(schemas, cpa=True)

operation = st.radio("Operation", ["Column typing", "Cell NER", "CPA"], horizontal=True)

default_rows = DEFAULT_CPA_TABLE_SAMPLE if operation == "CPA" else DEFAULT_TABLE_SAMPLE

colA, colB = st.columns([1, 1])
with colA:
    if st.button("Load sample"):
        st.session_state["tables_sampled_rows_raw"] = default_rows
with colB:
    auto_poll = st.checkbox("Auto-poll", value=cfg["auto_poll_default"])

if "tables_sampled_rows_raw" not in st.session_state:
    st.session_state["tables_sampled_rows_raw"] = default_rows

table_id = st.text_input("table_id", value="table-1")
sampled_rows_raw = st.text_area("sampled_rows JSON", height=220, key="tables_sampled_rows_raw")

sampled_rows: Any = None
columns: list[str] = []
json_error = None
try:
    sampled_rows = json.loads(sampled_rows_raw)
    columns = _infer_table_columns(sampled_rows)
except Exception as exc:  # noqa: BLE001
    json_error = str(exc)

if json_error:
    st.error(f"Invalid JSON: {json_error}")
else:
    st.caption(f"Detected columns: {columns or '(none)'}")

# ------------------------
# Column typing
# ------------------------
if operation == "Column typing":
    if not table_schema_names:
        st.error("No table-capable schemas returned from API.")
        st.stop()

    st.subheader("Column typing (one label per column)")
    st.caption("Tip: STI is fast for structural patterns. DPV-PD is fast privacy typing. DPV is deep/slow.")

    # Recommend defaults: sti -> dpv_pd -> dpv -> first
    recommended = "sti" if "sti" in table_schema_names else ("dpv_pd" if "dpv_pd" in table_schema_names else ("dpv" if "dpv" in table_schema_names else table_schema_names[0]))
    schema = st.selectbox("Schema", table_schema_names, index=table_schema_names.index(recommended))

    include_scores = st.checkbox("include_scores", value=False)

    if st.button("Run column typing", type="primary"):
        if err:
            st.error(err)
            st.stop()
        if not isinstance(sampled_rows, list) or not all(isinstance(r, dict) for r in sampled_rows):
            st.error("sampled_rows must be a JSON array of objects.")
            st.stop()

        payload = {
            "tasks": [{"task_id": "table-1", "table_id": table_id, "sampled_rows": sampled_rows}],
            "include_scores": include_scores,
            "llm": {"provider": cfg["provider"], "model": cfg["model"]},
        }

        submit_and_render_job(
            cfg=cfg,
            path=f"/schemas/{schema}/tabular/annotate",
            payload=payload,
            headers=build_llm_headers(cfg),
            label=f"Tabular typing ({schema})",
            auto_poll=auto_poll,
        )

# ------------------------
# Cell NER
# ------------------------
elif operation == "Cell NER":
    if not text_schema_names:
        st.error("No text-capable schemas returned from API (required for cell NER).")
        st.stop()

    st.subheader("Cell NER (entities inside selected columns)")
    st.caption("Use for notes/comments/description-like columns. For structured columns, column typing is usually enough.")

    default_schema = "dpv_pd" if "dpv_pd" in text_schema_names else ("dpv" if "dpv" in text_schema_names else text_schema_names[0])
    schema = st.selectbox("NER schema", text_schema_names, index=text_schema_names.index(default_schema))

    if not columns:
        st.warning("No columns detected yet. Fix JSON above to enable column selection.")
        target_cols = []
    else:
        target_cols = st.multiselect("target_columns", options=columns, default=columns[:2])

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        strings_only = st.checkbox("strings_only", value=True)
    with col2:
        skip_structured_literals = st.checkbox("skip_structured_literals", value=True)
    with col3:
        include_scores = st.checkbox("include_scores", value=False)
    with col4:
        strict_offsets = st.checkbox("strict_offsets", value=False)

    if st.button("Run cell NER", type="primary"):
        if err:
            st.error(err)
            st.stop()
        if not isinstance(sampled_rows, list) or not all(isinstance(r, dict) for r in sampled_rows):
            st.error("sampled_rows must be a JSON array of objects.")
            st.stop()
        if not target_cols:
            st.error("Select at least one target column.")
            st.stop()

        payload = {
            "tasks": [
                {
                    "task_id": "table-1",
                    "table_id": table_id,
                    "sampled_rows": sampled_rows,
                    "target_columns": target_cols,
                    "strings_only": strings_only,
                    "skip_structured_literals": skip_structured_literals,
                }
            ],
            "include_scores": include_scores,
            "strict_offsets": strict_offsets,
            "llm": {"provider": cfg["provider"], "model": cfg["model"]},
        }

        submit_and_render_job(
            cfg=cfg,
            path=f"/schemas/{schema}/tabular/ner",
            payload=payload,
            headers=build_llm_headers(cfg),
            label=f"Tabular NER ({schema})",
            auto_poll=auto_poll,
        )

# ------------------------
# CPA
# ------------------------
else:
    if not cpa_schema_names:
        st.error("No CPA-capable schemas returned from API.")
        st.stop()

    st.subheader("CPA (subject-column relationships)")
    st.caption(
        "CPA-lite (schema=cpa) is fast and generic. schemaorg_cpa_v1 is deeper and supports moose:NONE. "
        "For schemaorg_cpa_v1, subject_class is inferred by default (CTA) and domain filtering reduces candidates."
    )

    # Prefer cpa then schemaorg_cpa_v1
    default_cpa_schema = "cpa" if "cpa" in cpa_schema_names else cpa_schema_names[0]
    schema = st.selectbox("CPA schema", cpa_schema_names, index=cpa_schema_names.index(default_cpa_schema))

    if columns:
        subject_column = st.selectbox("subject_column", options=columns)
        available_targets = [c for c in columns if c != subject_column]
        target_columns = st.multiselect("target_columns (optional)", options=available_targets, default=[])
    else:
        subject_column = st.text_input("subject_column", value="")
        target_columns = []

    include_scores = st.checkbox("include_scores", value=False)

    # New: CPA flags
    debug_flag = st.checkbox("debug", value=cfg.get("show_debug", False))
    use_sti_signature_cache = st.checkbox("use_sti_signature_cache", value=True)

    # Optional subject_class override only in advanced mode
    subject_class = None
    if cfg.get("advanced_mode", False) and schema.startswith("schemaorg_"):
        subject_class = st.text_input("subject_class override (optional)", value="", help="Example: schema:Book").strip() or None

    if st.button("Run CPA", type="primary"):
        if err:
            st.error(err)
            st.stop()
        if not isinstance(sampled_rows, list) or not all(isinstance(r, dict) for r in sampled_rows):
            st.error("sampled_rows must be a JSON array of objects.")
            st.stop()
        if not subject_column:
            st.error("subject_column is required.")
            st.stop()

        task: dict[str, Any] = {
            "task_id": "cpa-1",
            "table_id": table_id,
            "sampled_rows": sampled_rows,
            "subject_column": subject_column,
            "debug": debug_flag,
            "use_sti_signature_cache": use_sti_signature_cache,
        }

        if subject_class:
            task["subject_class"] = subject_class
        if target_columns:
            task["target_columns"] = target_columns

        payload = {
            "tasks": [task],
            "include_scores": include_scores,
            "llm": {"provider": cfg["provider"], "model": cfg["model"]},
        }

        submit_and_render_job(
            cfg=cfg,
            path=f"/schemas/{schema}/tabular/cpa",
            payload=payload,
            headers=build_llm_headers(cfg),
            label=f"CPA ({schema})",
            auto_poll=auto_poll,
        )