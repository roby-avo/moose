from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

DEFAULT_BASE_URL = os.getenv("MOOSE_API_BASE_URL", "http://localhost:8000")
DEFAULT_SAMPLE = (
    "Hi, my name is David Johnson and I'm originally from Liverpool.\n"
    "My credit card number is 4095-2609-9393-4932 and my crypto wallet id is "
    "16Yeky6GMjeNkAiNcBY7ZhrLoMSgg1BoyZ.\n"
    "On 11/10/2024 I visited www.microsoft.com and sent an email to "
    "test@presidio.site, from IP 192.168.0.1."
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
    ollama_token: str | None,
    openrouter_api_key: str | None,
) -> dict[str, Any]:
    extra_headers: dict[str, str] = {}
    if openrouter_api_key:
        extra_headers["X-OpenRouter-API-Key"] = openrouter_api_key
    if ollama_token:
        extra_headers["X-Ollama-Token"] = ollama_token
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
) -> dict[str, Any]:
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            f"{base_url.rstrip('/')}/ner",
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
    if provider == "openrouter":
        models = models_payload.get("openrouter", {}).get("models", []) or []
        return [model.get("id", "") for model in models if model.get("id")]
    return []


st.set_page_config(page_title="Moose API Demo", layout="wide")
st.title("Moose API Streamlit Demo")

with st.sidebar:
    st.header("API Settings")
    base_url = st.text_input("Moose API base URL", value=DEFAULT_BASE_URL)
    api_key = st.text_input("Moose API key", type="password")
    provider = st.selectbox("LLM provider", ["ollama", "openrouter"])
    openrouter_api_key = None
    ollama_token = None
    if provider == "openrouter":
        openrouter_api_key = st.text_input("OpenRouter API key", type="password")
    if provider == "ollama":
        ollama_token = st.text_input("Ollama token (optional)", type="password")

    fetch_models = st.button("Fetch available models")

if "models" not in st.session_state:
    st.session_state["models"] = None
if "last_job_id" not in st.session_state:
    st.session_state["last_job_id"] = ""

models_error = None
if fetch_models:
    if not api_key:
        models_error = "Moose API key is required to fetch models."
    else:
        try:
            st.session_state["models"] = _fetch_models(
                base_url=base_url,
                api_key=api_key,
                provider=provider,
                ollama_token=ollama_token,
                openrouter_api_key=openrouter_api_key,
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

tab_models, tab_ner, tab_jobs = st.tabs(["Models", "NER", "Jobs"])

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
        raw_text = st.text_area("One sentence per line", value=DEFAULT_SAMPLE, height=140)
        schema = st.selectbox("Schema", ["coarse", "fine"])
        include_scores = st.checkbox("Include scores")
        submit = st.form_submit_button("Submit NER job")

    if submit:
        if not api_key:
            st.error("Moose API key is required to submit jobs.")
        else:
            tasks = _build_tasks(raw_text)
            if not tasks:
                st.error("Provide at least one task line.")
            else:
                llm_payload: dict[str, Any] = {"provider": provider}
                if model_in_use:
                    llm_payload["model"] = model_in_use
                payload: dict[str, Any] = {
                    "schema": schema,
                    "tasks": tasks,
                    "include_scores": include_scores,
                    "llm": llm_payload,
                }
                extra_headers: dict[str, str] = {}
                if openrouter_api_key:
                    extra_headers["X-OpenRouter-API-Key"] = openrouter_api_key
                if ollama_token:
                    extra_headers["X-Ollama-Token"] = ollama_token
                try:
                    response = _submit_ner(base_url, api_key, payload, extra_headers)
                    st.success("Job submitted.")
                    st.json(response)
                    st.session_state["last_job_id"] = response.get("job_id", "")
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))

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
