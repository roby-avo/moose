from __future__ import annotations

import os
from typing import Any

import streamlit as st

DEFAULT_BASE_URL = os.getenv("MOOSE_API_BASE_URL", "http://localhost:8000")

PROVIDERS = ["openrouter", "ollama", "deepinfra", "deepseek"]
PROVIDERS_REQUIRE_KEY = {"openrouter", "deepinfra", "deepseek"}


def sidebar() -> dict[str, Any]:
    st.sidebar.header("Connection")
    base_url = st.sidebar.text_input("Moose API base URL", value=st.session_state.get("base_url", DEFAULT_BASE_URL))
    api_key = st.sidebar.text_input("Moose API key", type="password", value=st.session_state.get("api_key", ""))

    st.sidebar.divider()
    st.sidebar.header("LLM")
    provider = st.sidebar.selectbox(
        "Provider",
        PROVIDERS,
        index=PROVIDERS.index(st.session_state.get("provider", "deepinfra")) if st.session_state.get("provider") in PROVIDERS else 0,
    )

    default_model = st.session_state.get("model", "")
    if not default_model:
        default_model = {
            "deepseek": "deepseek-chat",
            "deepinfra": "Qwen/Qwen3-Next-80B-A3B-Instruct",
            "ollama": "llama3",
            "openrouter": "anthropic/claude-3.5-sonnet",
        }.get(provider, "")

    model = st.sidebar.text_input("Model", value=default_model)

    label = "LLM API key"
    if provider not in PROVIDERS_REQUIRE_KEY:
        label = "LLM API key (optional)"
    llm_api_key = st.sidebar.text_input(label, type="password", value=st.session_state.get("llm_api_key", ""))

    st.sidebar.divider()
    st.sidebar.header("UI")
    auto_poll_default = st.sidebar.checkbox("Auto-poll by default", value=st.session_state.get("auto_poll_default", True))
    advanced_mode = st.sidebar.checkbox("Advanced mode", value=st.session_state.get("advanced_mode", False))

    show_legal_refs = st.sidebar.checkbox("Show legal refs", value=st.session_state.get("show_legal_refs", True))
    show_legal_detail = st.sidebar.checkbox("Show legal ref details", value=st.session_state.get("show_legal_detail", True))
    show_debug = st.sidebar.checkbox("Show debug output", value=st.session_state.get("show_debug", False))
    show_raw = st.sidebar.checkbox("Show raw JSON", value=st.session_state.get("show_raw", False))

    with st.sidebar.expander("Advanced provider options", expanded=False):
        llm_endpoint = st.text_input("LLM endpoint override (optional)", value=st.session_state.get("llm_endpoint", ""))
        developer_mode = st.checkbox("Developer mode", value=st.session_state.get("developer_mode", False))

    # persist state
    st.session_state.update(
        {
            "base_url": base_url,
            "api_key": api_key,
            "provider": provider,
            "model": model,
            "llm_api_key": llm_api_key,
            "llm_endpoint": llm_endpoint,
            "developer_mode": developer_mode,
            "auto_poll_default": auto_poll_default,
            "advanced_mode": advanced_mode,
            "show_legal_refs": show_legal_refs,
            "show_legal_detail": show_legal_detail,
            "show_debug": show_debug,
            "show_raw": show_raw,
        }
    )

    return {
        "base_url": base_url,
        "api_key": api_key,
        "provider": provider,
        "model": model,
        "llm_api_key": llm_api_key,
        "llm_endpoint": llm_endpoint,
        "developer_mode": developer_mode,
        "auto_poll_default": auto_poll_default,
        "advanced_mode": advanced_mode,
        "show_legal_refs": show_legal_refs,
        "show_legal_detail": show_legal_detail,
        "show_debug": show_debug,
        "show_raw": show_raw,
    }


def validate_common(cfg: dict[str, Any]) -> str | None:
    if not cfg.get("api_key"):
        return "Moose API key is required."
    provider = cfg.get("provider")
    if provider in PROVIDERS_REQUIRE_KEY and not cfg.get("llm_api_key"):
        return f"LLM API key is required for provider '{provider}'."
    if not cfg.get("model"):
        return "Model is required."
    return None


def build_llm_headers(cfg: dict[str, Any]) -> dict[str, str]:
    h: dict[str, str] = {}
    if cfg.get("llm_api_key"):
        h["X-LLM-API-Key"] = cfg["llm_api_key"]
    if cfg.get("llm_endpoint"):
        h["X-LLM-Endpoint"] = cfg["llm_endpoint"]
    return h