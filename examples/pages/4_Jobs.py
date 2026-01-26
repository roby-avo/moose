# examples/pages/4_Jobs.py

from __future__ import annotations

import streamlit as st

from moose_ui.api import poll_job, wait_for_job
from moose_ui.config import sidebar
from moose_ui.render import render_job
from moose_ui.state import get_job_history, get_last_job_id

st.title("Jobs")

cfg = sidebar()

if not cfg.get("api_key"):
    st.info("Enter your Moose API key in the sidebar.")
    st.stop()

history = get_job_history()

st.subheader("Recent jobs (this session)")
selected_job_id = ""
if history:
    labels = [f"{h['job_id']} — {h['label']}" for h in history]
    choice = st.selectbox("Pick a job", labels, index=0)
    selected_job_id = choice.split(" — ", 1)[0]
else:
    st.caption("No jobs yet.")

st.subheader("Poll")

default_job_id = selected_job_id or get_last_job_id()
job_id = st.text_input("Job ID", value=default_job_id)

col1, col2 = st.columns([1, 1])
with col1:
    poll_once = st.button("Poll once")
with col2:
    wait_done = st.button("Wait until done (auto-poll)")

def _render(job: dict) -> None:
    render_job(
        job,
        show_raw=cfg.get("show_raw", False),
        show_legal_refs=cfg.get("show_legal_refs", True),
        show_legal_detail=cfg.get("show_legal_detail", True),
        show_debug=cfg.get("show_debug", False),
    )

if poll_once:
    if not job_id.strip():
        st.error("Provide a job id.")
    else:
        try:
            job = poll_job(cfg["base_url"], cfg["api_key"], job_id.strip())
            _render(job)
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))

if wait_done:
    if not job_id.strip():
        st.error("Provide a job id.")
    else:
        try:
            job = wait_for_job(cfg["base_url"], cfg["api_key"], job_id.strip())
            _render(job)
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))