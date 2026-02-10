from __future__ import annotations

from typing import Any

import streamlit as st

from .api import api_post, wait_for_job
from .render import render_job
from .state import add_job_history


def submit_and_render_job(
    *,
    cfg: dict[str, Any],
    path: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    label: str,
    auto_poll: bool,
) -> None:
    resp = api_post(cfg["base_url"], cfg["api_key"], path, payload, headers)
    job_id = resp.get("job_id", "")
    add_job_history(job_id, label)

    st.success("Job submitted.")
    st.json(resp)

    if auto_poll and job_id:
        final = wait_for_job(cfg["base_url"], cfg["api_key"], job_id)
        render_job(
            final,
            show_raw=cfg.get("show_raw", False),
            show_legal_refs=cfg.get("show_legal_refs", True),
            show_legal_detail=cfg.get("show_legal_detail", True),
            show_debug=cfg.get("show_debug", False),
        )