from __future__ import annotations

from typing import Any

import streamlit as st


def _expander_json(title: str, data: Any, *, show: bool) -> None:
    if not show:
        return
    with st.expander(title, expanded=False):
        st.json(data)


def render_job(job: dict[str, Any], *, show_raw: bool, show_legal_refs: bool, show_legal_detail: bool, show_debug: bool) -> None:
    st.markdown("## Job")
    st.markdown(
        f"**job_id:** `{job.get('job_id')}`  \n"
        f"**status:** `{job.get('status')}`  \n"
        f"**created_at:** `{job.get('created_at', '')}`  \n"
        f"**updated_at:** `{job.get('updated_at', '')}`"
    )

    if job.get("status") == "failed":
        st.error(job.get("error") or "Job failed (no error message).")
        _expander_json("Raw job JSON", job, show=show_raw)
        return

    if job.get("status") != "completed":
        _expander_json("Raw job JSON", job, show=show_raw)
        return

    result = job.get("result") or {}
    _render_result(result, show_raw=show_raw, show_legal_refs=show_legal_refs, show_legal_detail=show_legal_detail, show_debug=show_debug)
    _expander_json("Raw job JSON", job, show=show_raw)


def _detect_kind(result: dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return "unknown"
    if "action_catalog" in result and "results" in result:
        rs = result.get("results")
        if isinstance(rs, list) and rs and isinstance(rs[0], dict) and "findings" in rs[0]:
            return "privacy"
    rs = result.get("results")
    if not isinstance(rs, list) or not rs:
        return "unknown"
    first = rs[0]
    if not isinstance(first, dict):
        return "unknown"
    if "entities" in first:
        return "text_ner"
    if "columns" in first:
        return "tabular_annotate"
    if "relationships" in first:
        return "cpa"
    if "rows" in first:
        return "tabular_ner"
    return "unknown"


def _render_result(result: dict[str, Any], *, show_raw: bool, show_legal_refs: bool, show_legal_detail: bool, show_debug: bool) -> None:
    kind = _detect_kind(result)
    if kind == "privacy":
        render_privacy_result(result, show_legal_refs=show_legal_refs, show_legal_detail=show_legal_detail, show_raw=show_raw)
        return
    if kind == "cpa":
        render_cpa_result(result, show_debug=show_debug, show_raw=show_raw)
        return

    # fallback: keep your existing simple behavior
    st.subheader("Result")
    st.json(result) if show_raw else st.write(result)


def render_cpa_result(result: dict[str, Any], *, show_debug: bool, show_raw: bool) -> None:
    st.subheader("CPA relationships")
    for item in result.get("results", []):
        st.markdown(
            f"**task_id:** `{item.get('task_id')}`  \n"
            f"**table_id:** `{item.get('table_id')}`  \n"
            f"**subject_column:** `{item.get('subject_column')}`"
        )
        rels = [
            {
                "target_column": r.get("target_column"),
                "relation_id": r.get("relation_id"),
                "confidence": r.get("confidence"),
            }
            for r in item.get("relationships", []) or []
        ]
        if rels:
            st.dataframe(rels, use_container_width=True)
        else:
            st.info("No relationships returned.")

        if show_debug and "debug" in item:
            _expander_json("Debug", item.get("debug"), show=True)

    _expander_json("Raw CPA result", result, show=show_raw)


def render_privacy_result(result: dict[str, Any], *, show_legal_refs: bool, show_legal_detail: bool, show_raw: bool) -> None:
    st.subheader("Privacy findings")
    st.caption(f"profile: {result.get('profile')} | policy_pack: {result.get('policy_pack')}")

    for task in result.get("results", []) or []:
        st.markdown(f"### {task.get('kind')} task `{task.get('task_id')}`")

        findings = task.get("findings", []) or []
        if not findings:
            st.info("No findings.")
            continue

        table_rows = []
        for f in findings:
            row = {
                "severity": f.get("severity"),
                "status": f.get("status"),
                "rule_id": f.get("rule_id"),
                "issue": f.get("issue"),
                "recommended_actions": ", ".join(f.get("recommended_actions") or []),
                "rationale": f.get("rationale"),
            }
            if show_legal_refs:
                # show short labels if detail available, otherwise IDs
                if show_legal_detail and f.get("legal_refs_detail"):
                    row["legal_refs"] = "; ".join([d.get("label") or d.get("id") for d in f["legal_refs_detail"]])
                else:
                    row["legal_refs"] = "; ".join(f.get("legal_refs") or [])
            table_rows.append(row)

        st.dataframe(table_rows, use_container_width=True)

        with st.expander("Details (findings)", expanded=False):
            for f in findings:
                st.markdown(f"**{f.get('rule_id')}** — {f.get('issue')}")
                st.caption(f"severity={f.get('severity')} status={f.get('status')}")
                st.write(f.get("rationale"))

                if f.get("recommended_actions"):
                    st.write("Actions:", f.get("recommended_actions"))

                _expander_json("Evidence", f.get("evidence"), show=True)

                if show_legal_refs:
                    if show_legal_detail:
                        _expander_json("Legal refs (detail)", f.get("legal_refs_detail"), show=True)
                    else:
                        _expander_json("Legal refs (ids)", f.get("legal_refs"), show=True)

                st.divider()

        if "extraction" in task:
            _expander_json("Extraction (raw)", task.get("extraction"), show=show_raw)

    if result.get("warnings"):
        _expander_json("Warnings", result.get("warnings"), show=True)

    _expander_json("Raw privacy result", result, show=show_raw)