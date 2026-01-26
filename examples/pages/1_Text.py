from __future__ import annotations

import streamlit as st

from moose_ui.config import build_llm_headers, sidebar, validate_common
from moose_ui.metadata import fetch_schemas, schemas_supporting
from moose_ui.samples import DEFAULT_TEXT_SAMPLE
from moose_ui.submit import submit_and_render_job

st.title("Text")

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

schema_names = schemas_supporting(schemas, text=True)
if not schema_names:
    st.error("No text-capable schemas returned from API.")
    st.stop()

# Prefer dpv_pd then dpv
preferred = "dpv_pd" if "dpv_pd" in schema_names else ("dpv" if "dpv" in schema_names else schema_names[0])
schema = st.selectbox("Schema", schema_names, index=schema_names.index(preferred))

raw_text = st.text_area("Text (one sentence per line)", value=DEFAULT_TEXT_SAMPLE, height=160)

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    include_scores = st.checkbox("include_scores", value=False)
with col2:
    strict_offsets = st.checkbox("strict_offsets (fail-fast)", value=False)
with col3:
    auto_poll = st.checkbox("Auto-poll", value=cfg["auto_poll_default"])

if st.button("Run NER", type="primary"):
    if err:
        st.error(err)
        st.stop()

    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    tasks = [{"task_id": f"t{i+1}", "text": ln} for i, ln in enumerate(lines)]
    if not tasks:
        st.error("Provide at least one non-empty line.")
        st.stop()

    payload = {
        "tasks": tasks,
        "include_scores": include_scores,
        "strict_offsets": strict_offsets,
        "llm": {"provider": cfg["provider"], "model": cfg["model"]},
    }

    submit_and_render_job(
        cfg=cfg,
        path=f"/schemas/{schema}/ner",
        payload=payload,
        headers=build_llm_headers(cfg),
        label=f"Text NER ({schema})",
        auto_poll=auto_poll,
    )