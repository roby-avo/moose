from __future__ import annotations

import time
from typing import Any

import streamlit as st


def add_job_history(job_id: str, label: str) -> None:
    st.session_state.setdefault("job_history", [])
    st.session_state["job_history"].insert(0, {"job_id": job_id, "label": label, "ts": int(time.time())})
    st.session_state["job_history"] = st.session_state["job_history"][:50]
    st.session_state["last_job_id"] = job_id


def get_job_history() -> list[dict[str, Any]]:
    return st.session_state.get("job_history", []) or []


def get_last_job_id() -> str:
    return st.session_state.get("last_job_id", "") or ""