from __future__ import annotations

import json
import os
from typing import Any

import httpx
import streamlit as st

from moose.prompts import build_text_ner_prompt
from moose.schema import get_types

DEFAULT_BASE_URL = os.getenv("MOOSE_API_BASE_URL", "http://localhost:8000")
DEFAULT_SAMPLE = (
    "Hi, my name is David Johnson and I'm originally from Liverpool.\n"
    "My credit card number is 4095-2609-9393-4932 and my crypto wallet id is "
    "16Yeky6GMjeNkAiNcBY7ZhrLoMSgg1BoyZ.\n"
    "On 11/10/2024 I visited www.microsoft.com and sent an email to "
    "test@presidio.site, from IP 192.168.0.1."
)
DEFAULT_TABLE_SAMPLE = json.dumps(
    [
        {
            "name": "Alice Smith",
            "email": "alice@example.com",
            "age": "29",
            "ip_address": "192.168.0.1",
        },
        {
            "name": "Bob Jones",
            "email": "bob@example.com",
            "age": "41",
            "ip_address": "10.0.0.5",
        },
    ],
    ensure_ascii=True,
    indent=2,
)


def _request_headers(api_key: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = {"X-API-Key": api_key}
    if extra:
        headers.update(extra)
    return headers


def _fetch_models(
    base_url: str,
    api_key: str,
    provider: str,
    llm_api_key: str | None,
    llm_endpoint: str | None,
) -> dict[str, Any]:
    extra_headers: dict[str, str] = {}
    if llm_api_key:
        extra_headers["X-LLM-API-Key"] = llm_api_key
    if llm_endpoint:
        extra_headers["X-LLM-Endpoint"] = llm_endpoint
    params = {"provider": provider}
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(
            f"{base_url.rstrip('/')}/models",
            headers=_request_headers(api_key, extra_headers),
            params=params,
        )
        resp.raise_for_status()
        return resp.json()


def _submit_ner(
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    extra_headers: dict[str, str] | None,
    endpoint: str = "/ner",
) -> dict[str, Any]:
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            f"{base_url.rstrip('/')}{endpoint}",
            headers=_request_headers(api_key, extra_headers),
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()


def _submit_tabular(
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    extra_headers: dict[str, str] | None,
    endpoint: str = "/tabular/annotate",
) -> dict[str, Any]:
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            f"{base_url.rstrip('/')}{endpoint}",
            headers=_request_headers(api_key, extra_headers),
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()


def _poll_job(base_url: str, api_key: str, job_id: str) -> dict[str, Any]:
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(
            f"{base_url.rstrip('/')}/jobs/{job_id}",
            headers=_request_headers(api_key),
        )
        resp.raise_for_status()
        return resp.json()


def _build_tasks(raw_text: str) -> list[dict[str, str]]:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    tasks = []
    for idx, line in enumerate(lines, start=1):
        tasks.append({"task_id": f"t{idx}", "text": line})
    return tasks


def _extract_models(models_payload: dict[str, Any], provider: str) -> list[str]:
    if provider == "ollama":
        return models_payload.get("ollama", {}).get("models", []) or []
    models = models_payload.get("openrouter", {}).get("models", []) or []
    return [model.get("id", "") for model in models if model.get("id")]


st.set_page_config(page_title="Moose API Demo", layout="wide")
st.title("Moose API Streamlit Demo")

with st.sidebar:
    st.header("API Settings")
    base_url = st.text_input("Moose API base URL", value=DEFAULT_BASE_URL)
    api_key = st.text_input("Moose API key", type="password")
    provider = st.selectbox("LLM provider", ["openrouter", "ollama"])
    llm_api_key = st.text_input("LLM API key (optional for Ollama)", type="password")
    llm_endpoint = st.text_input("LLM endpoint override (optional)")

    fetch_models = st.button("Fetch available models")

if "models" not in st.session_state:
    st.session_state["models"] = None
if "last_job_id" not in st.session_state:
    st.session_state["last_job_id"] = ""
if "last_prompt" not in st.session_state:
    st.session_state["last_prompt"] = ""
if "debug_prompt" not in st.session_state:
    st.session_state["debug_prompt"] = ""

models_error = None
if fetch_models:
    if not api_key:
        models_error = "Moose API key is required to fetch models."
    elif provider == "openrouter" and not llm_api_key:
        models_error = "OpenRouter API key is required to fetch models."
    else:
        try:
            st.session_state["models"] = _fetch_models(
                base_url=base_url,
                api_key=api_key,
                provider=provider,
                llm_api_key=llm_api_key,
                llm_endpoint=llm_endpoint,
            )
        except Exception as exc:  # noqa: BLE001
            models_error = str(exc)

if models_error:
    st.error(models_error)

models_payload = st.session_state.get("models") or {}
model_options = _extract_models(models_payload, provider)

with st.sidebar:
    if model_options:
        selected_model = st.selectbox("Model from API", model_options)
    else:
        selected_model = ""
        st.caption("Fetch models to populate the list.")
    model_override = st.text_input("Model override (optional)", value="")

model_in_use = model_override or selected_model

tab_models, tab_ner, tab_tabular, tab_prompt, tab_jobs = st.tabs(
    ["Models", "NER", "Tabular", "Prompt", "Jobs"]
)

with tab_models:
    st.subheader("Available Models")
    provider_error = models_payload.get(provider, {}).get("error") if models_payload else None
    if provider_error:
        st.error(provider_error)
    if provider == "ollama":
        st.write(model_options or "No Ollama models returned.")
    else:
        st.write(model_options or "No OpenRouter models returned.")
    if models_payload:
        with st.expander("Raw response"):
            st.json(models_payload)

with tab_ner:
    st.subheader("NER Test")
    if model_in_use:
        st.caption(f"Using model: {model_in_use}")
    else:
        st.caption("Using server defaults (no model override).")
    with st.form("ner_form"):
        raw_text = st.text_area(
            "One sentence per line",
            value=DEFAULT_SAMPLE,
            height=140,
            key="ner_raw_text",
        )
        schema = st.selectbox("Schema", ["coarse", "fine", "dpv"], key="ner_schema")
        endpoint_mode = st.selectbox(
            "Endpoint",
            ["Auto (use schema)", "/ner", "/dpv/ner"],
            key="ner_endpoint_mode",
        )
        include_scores = st.checkbox("Include scores", key="ner_include_scores")
        submit = st.form_submit_button("Submit NER job")

    if submit:
        tasks = _build_tasks(raw_text)
        if not tasks:
            st.error("Provide at least one task line.")
            st.session_state["last_prompt"] = ""
        else:
            type_ids = get_types(schema)
            st.session_state["last_prompt"] = build_text_ner_prompt(
                schema, tasks, type_ids
            )
            if not api_key:
                st.error("Moose API key is required to submit jobs.")
            elif provider == "openrouter" and not llm_api_key:
                st.error("OpenRouter API key is required to submit jobs.")
            elif not model_in_use:
                st.error("Model is required. Fetch models or provide a model override.")
            else:
                llm_payload: dict[str, Any] = {"provider": provider, "model": model_in_use}
                endpoint = "/dpv/ner" if schema == "dpv" else "/ner"
                if endpoint_mode != "Auto (use schema)":
                    endpoint = endpoint_mode
                payload: dict[str, Any] = {
                    "tasks": tasks,
                    "include_scores": include_scores,
                    "llm": llm_payload,
                }
                if endpoint == "/ner":
                    payload["schema"] = schema
                extra_headers: dict[str, str] = {}
                if llm_api_key:
                    extra_headers["X-LLM-API-Key"] = llm_api_key
                if llm_endpoint:
                    extra_headers["X-LLM-Endpoint"] = llm_endpoint
                try:
                    response = _submit_ner(
                        base_url,
                        api_key,
                        payload,
                        extra_headers,
                        endpoint=endpoint,
                    )
                    st.success("Job submitted.")
                    st.json(response)
                    st.session_state["last_job_id"] = response.get("job_id", "")
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
    if st.session_state.get("last_prompt"):
        with st.expander("Generated prompt"):
            st.code(st.session_state["last_prompt"], language="text")

with tab_tabular:
    st.subheader("Tabular Typing Test")
    if model_in_use:
        st.caption(f"Using model: {model_in_use}")
    else:
        st.caption("Using server defaults (no model override).")
    with st.form("tabular_form"):
        table_id = st.text_input("Table ID", value="table-1", key="tabular_table_id")
        sampled_rows_raw = st.text_area(
            "Sampled rows JSON",
            value=DEFAULT_TABLE_SAMPLE,
            height=160,
            key="tabular_rows_raw",
        )
        schema = st.selectbox("Schema", ["coarse", "fine", "dpv"], key="tabular_schema")
        endpoint_mode = st.selectbox(
            "Endpoint",
            ["Auto (use schema)", "/tabular/annotate", "/dpv/tabular/annotate"],
            key="tabular_endpoint_mode",
        )
        include_scores = st.checkbox("Include scores", key="tabular_include_scores")
        submit = st.form_submit_button("Submit Tabular job")

    if submit:
        try:
            sampled_rows = json.loads(sampled_rows_raw)
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc}")
            sampled_rows = None
        if sampled_rows is None:
            st.stop()
        if not isinstance(sampled_rows, list) or not all(
            isinstance(row, dict) for row in sampled_rows
        ):
            st.error("Sampled rows must be a JSON array of objects.")
        elif not api_key:
            st.error("Moose API key is required to submit jobs.")
        elif provider == "openrouter" and not llm_api_key:
            st.error("OpenRouter API key is required to submit jobs.")
        elif not model_in_use:
            st.error("Model is required. Fetch models or provide a model override.")
        else:
            llm_payload: dict[str, Any] = {"provider": provider, "model": model_in_use}
            tasks = [
                {
                    "task_id": "table-1",
                    "table_id": table_id,
                    "sampled_rows": sampled_rows,
                }
            ]
            endpoint = "/dpv/tabular/annotate" if schema == "dpv" else "/tabular/annotate"
            if endpoint_mode != "Auto (use schema)":
                endpoint = endpoint_mode
            payload: dict[str, Any] = {
                "tasks": tasks,
                "include_scores": include_scores,
                "llm": llm_payload,
            }
            if endpoint == "/tabular/annotate":
                payload["schema"] = schema
            extra_headers: dict[str, str] = {}
            if llm_api_key:
                extra_headers["X-LLM-API-Key"] = llm_api_key
            if llm_endpoint:
                extra_headers["X-LLM-Endpoint"] = llm_endpoint
            try:
                response = _submit_tabular(
                    base_url,
                    api_key,
                    payload,
                    extra_headers,
                    endpoint=endpoint,
                )
                st.success("Job submitted.")
                st.json(response)
                st.session_state["last_job_id"] = response.get("job_id", "")
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

with tab_prompt:
    st.subheader("Prompt Debug")
    prompt_text = st.text_area(
        "One sentence per line",
        value=DEFAULT_SAMPLE,
        height=160,
        key="prompt_raw_text",
    )
    prompt_schema = st.selectbox("Schema", ["coarse", "fine", "dpv"], key="prompt_schema")
    generate_prompt = st.button("Generate prompt")
    if generate_prompt:
        tasks = _build_tasks(prompt_text)
        if not tasks:
            st.error("Provide at least one task line.")
            st.session_state["debug_prompt"] = ""
        else:
            type_ids = get_types(prompt_schema)
            st.session_state["debug_prompt"] = build_text_ner_prompt(
                prompt_schema, tasks, type_ids
            )
    if st.session_state.get("debug_prompt"):
        st.code(st.session_state["debug_prompt"], language="text")

with tab_jobs:
    st.subheader("Poll Job")
    job_id = st.text_input("Job ID", value=st.session_state.get("last_job_id", ""))
    poll = st.button("Poll status")
    if poll:
        if not api_key:
            st.error("Moose API key is required to poll jobs.")
        elif not job_id:
            st.error("Provide a job id.")
        else:
            try:
                job = _poll_job(base_url, api_key, job_id)
                st.json(job)
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
