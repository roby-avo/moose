from __future__ import annotations

from moose.prompts import (
    build_cpa_prompt,
    build_table_prompt,
    build_tabular_cell_ner_prompt,
    build_text_ner_prompt,
    build_type_selection_prompt,
)
from moose.schema import get_schema_config


def test_text_prompt_has_no_task_id_field() -> None:
    schema = get_schema_config("coarse")
    prompt = build_text_ner_prompt(schema, [{"task_id": "t1", "text": "Alice"}], schema.load_type_ids())
    assert "task_id" not in prompt


def test_table_prompt_has_no_task_id_field() -> None:
    schema = get_schema_config("coarse")
    prompt = build_table_prompt(
        schema,
        [{"task_id": "t1", "table_id": "tbl", "sampled_rows": [{"name": "Alice"}]}],
        schema.load_type_ids(),
    )
    assert "task_id" not in prompt


def test_tabular_cell_prompt_has_no_task_id_field() -> None:
    schema = get_schema_config("coarse")
    prompt = build_tabular_cell_ner_prompt(
        schema,
        [{"task_id": "t1", "table_id": "tbl", "row_index": 0, "column": "name", "text": "Alice"}],
        schema.load_type_ids(),
    )
    assert "task_id" not in prompt


def test_type_selection_prompt_has_no_task_id_field() -> None:
    schema = get_schema_config("cpa")
    prompt = build_type_selection_prompt(
        schema,
        [
            {
                "task_id": "sel-1",
                "table_id": "tbl",
                "subject_column": "subject",
                "target_column": "target",
                "sampled_rows": [{"subject": "A", "target": "B"}],
            }
        ],
        ["CPA:NONE"],
        mode="cpa",
    )
    assert "task_id" not in prompt


def test_cpa_prompt_has_no_task_id_field() -> None:
    schema = get_schema_config("cpa")
    prompt = build_cpa_prompt(
        schema,
        {
            "task_id": "cpa-1",
            "table_id": "tbl",
            "sampled_rows": [{"subject": "A", "target": "B"}],
            "subject_column": "subject",
            "target_columns": ["target"],
        },
        ["CPA:NONE", "CPA:OTHER"],
    )
    assert "task_id" not in prompt


def test_dpv_pd_text_prompt_follows_contract_sections() -> None:
    schema = get_schema_config("dpv_pd")
    prompt = build_text_ner_prompt(
        schema,
        [{"task_id": "t1", "text": "We process email addresses and use an AI system."}],
        schema.load_type_ids(),
    )
    assert "TASK" in prompt
    assert "OUTPUT (API CONTRACT)" in prompt
    assert "ALLOWED LABEL SPACE" in prompt
    assert "SPAN RULES (NER BEST PRACTICES)" in prompt
    assert "SCORING RULES" in prompt
    assert "INPUT" in prompt
    assert "OUTPUT" in prompt
    assert "personal data categories (DPV-PD)" in prompt
    assert "AI-related concepts/activities (DPV-AI)" in prompt
