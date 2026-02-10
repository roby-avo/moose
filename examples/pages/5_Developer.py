from __future__ import annotations

import streamlit as st

from moose_ui.api import api_get
from moose_ui.config import sidebar
from moose_ui.metadata import fetch_assets, fetch_policy_packs, fetch_privacy_profiles, fetch_schemas

st.title("Developer")

cfg = sidebar()
if not cfg.get("developer_mode"):
    st.info("Enable Developer mode in the sidebar to use this page.")
    st.stop()

if not cfg.get("api_key"):
    st.error("Moose API key required.")
    st.stop()

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