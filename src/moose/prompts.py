from __future__ import annotations

import json
from typing import Any

from moose.schema import SchemaConfig


def _format_intro(text: str) -> str:
    return f"{text.rstrip()}\n"


def _escape_md_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\n", " ").replace("\r", " ")
    # Markdown table delimiter is '|'
    text = text.replace("|", "\\|")
    return text


def table_to_markdown(sampled_rows: list[dict[str, Any]], columns: list[str], max_rows: int = 5) -> str:
    """
    Render a markdown table with given columns, using up to max_rows sampled rows.
    """
    cols = list(columns)
    header = "| " + " | ".join(_escape_md_cell(c) for c in cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for row in sampled_rows[:max_rows]:
        lines.append("| " + " | ".join(_escape_md_cell(row.get(c)) for c in cols) + " |")
    return "\n".join(lines)

def build_cpa_prompt(schema: SchemaConfig, task: dict[str, Any], relation_ids: list[str], max_rows: int = 5) -> str:
    subject = task["subject_column"]
    targets: list[str] = task["target_columns"]
    table_md = table_to_markdown(task["sampled_rows"], [subject] + targets, max_rows=max_rows)

    none_label = "moose:NONE" if "moose:NONE" in relation_ids else ("CPA:NONE" if "CPA:NONE" in relation_ids else None)
    other_label = "moose:OTHER" if "moose:OTHER" in relation_ids else ("CPA:OTHER" if "CPA:OTHER" in relation_ids else None)

    intro = _format_intro(schema.cpa_intro)
    allowed = ", ".join(relation_ids)

    if schema.require_all_scores:
        score_rule = "- Scores must include ALL relationship labels as keys.\n"
    else:
        score_rule = "- Scores may include ONLY a subset of relationship labels (missing labels are treated as 0).\n"

    rules_lines = [
        "- Output relationships for EXACTLY the provided target columns (no extra, no missing).",
        "- Scores must be non-negative floats; at least one score must be > 0 per target column.",
    ]
    if none_label:
        rules_lines.append(f"- Choose {none_label} if there is no relationship.")
    if other_label:
        rules_lines.append(f"- Choose {other_label} if there is a relationship but it is not represented in the label set.")

    return "\n".join(
        [
            intro.rstrip(),
            "Task: Predict the semantic relationship (predicate) between the SUBJECT column and each TARGET column.",
            "Return ONLY valid JSON.",
            f"Allowed relationship labels: {allowed}",
            "",
            "Output format (JSON array):",
            "[",
            "  {",
            '    "task_id": "...",',
            '    "table_id": "...",',
            '    "subject_column": "...",',
            '    "relationships": [',
            "      {",
            '        "target_column": "...",',
            '        "scores": {"<label>": 1.0}',
            "      }",
            "    ]",
            "  }",
            "]",
            "",
            "Rules:",
            *rules_lines,
            score_rule.rstrip(),
            "No extra text around the JSON.",
            "",
            "Input:",
            f'Task ID: {task["task_id"]}',
            f'Table ID: {task["table_id"]}',
            f"Subject column: {subject}",
            f"Target columns: {json.dumps(targets, ensure_ascii=True)}",
            "",
            "Sample table (markdown):",
            table_md,
        ]
    )

TYPE_SELECT_INTRO = "You are a type inventory selector for semantic typing."


def build_text_ner_prompt(schema: SchemaConfig, tasks: list[dict], type_ids: list[str]) -> str:
    payload = [{"task_id": t["task_id"], "text": t["text"]} for t in tasks]
    types = ", ".join(type_ids)
    intro = _format_intro(schema.text_intro)
    if schema.require_all_scores:
        score_rule = "- Scores must be non-negative floats for every allowed type_id (include all keys).\n"
    else:
        score_rule = (
            "- Scores must be non-negative floats for selected type_ids only; "
            "omit unrelated types (missing keys treated as 0).\n"
        )
    return "".join(
        [
            intro,
            f"Schema: {schema.name}\n",
            f"Allowed type_ids: {types}\n",
            "Return ONLY valid JSON.\n",
            "Output format (JSON array):\n",
            "[\n",
            "  {\n",
            "    \"task_id\": \"...\",\n",
            "    \"entities\": [\n",
            "      {\n",
            "        \"start\": 0,\n",
            "        \"end\": 0,\n",
            "        \"text\": \"exact substring\",\n",
            "        \"scores\": {\"NER:PERSON\": 0.1, \"NER:OTHER\": 0.2}\n",
            "      }\n",
            "    ]\n",
            "  }\n",
            "]\n",
            "Rules:\n",
            "- Offsets are 0-based, end-exclusive.\n",
            "- The text must match the exact substring from the input text.\n",
            score_rule,
            "- At least one score must be > 0 per entity.\n",
            "No extra text around the JSON.\n\n",
            "Input tasks JSON:\n",
            f"{json.dumps(payload, ensure_ascii=True)}",
        ]
    )


def build_tabular_cell_ner_prompt(schema: SchemaConfig, tasks: list[dict], type_ids: list[str]) -> str:
    """
    Each task is ONE TABLE CELL. Offsets are relative to that cell's text.
    Expected task keys:
      - task_id: synthetic id like "<table_task_id>:row<idx>:col=<encoded_col>"
      - text: cell string
      - table_id, row_index, column are included for context (optional)
    """
    payload = [
        {
            "task_id": t["task_id"],
            "table_id": t.get("table_id"),
            "row_index": t.get("row_index"),
            "column": t.get("column"),
            "text": t["text"],
        }
        for t in tasks
    ]

    types = ", ".join(type_ids)
    intro = _format_intro(schema.text_intro)
    if schema.require_all_scores:
        score_rule = "- Scores must be non-negative floats for every allowed type_id (include all keys).\n"
    else:
        score_rule = (
            "- Scores must be non-negative floats for selected type_ids only; "
            "omit unrelated types (missing keys treated as 0).\n"
        )

    return "".join(
        [
            intro,
            "You are a high-precision NER engine operating over TABLE CELLS.\n",
            "Each task corresponds to one cell's text. Offsets are relative to that cell's text.\n",
            f"Schema: {schema.name}\n",
            f"Allowed type_ids: {types}\n",
            "Return ONLY valid JSON.\n",
            "Output format (JSON array):\n",
            "[\n",
            "  {\n",
            "    \"task_id\": \"...\",\n",
            "    \"entities\": [\n",
            "      {\n",
            "        \"start\": 0,\n",
            "        \"end\": 0,\n",
            "        \"text\": \"exact substring from the cell text\",\n",
            "        \"scores\": {\"NER:PERSON\": 0.1, \"NER:OTHER\": 0.2}\n",
            "      }\n",
            "    ]\n",
            "  }\n",
            "]\n",
            "Rules:\n",
            "- Offsets are 0-based, end-exclusive.\n",
            "- entity.text MUST equal cell_text[start:end] exactly.\n",
            score_rule,
            "- At least one score must be > 0 per entity.\n",
            "No extra text around the JSON.\n\n",
            "Input cell tasks JSON:\n",
            f"{json.dumps(payload, ensure_ascii=True)}",
        ]
    )


def build_type_selection_prompt(schema: SchemaConfig, tasks: list[dict], type_ids: list[str], mode: str) -> str:
    if mode == "text":
        payload = [{"task_id": t["task_id"], "text": t["text"]} for t in tasks]
        mode_hint = "text"
    elif mode == "table":
        payload = [
            {
                "task_id": t["task_id"],
                "table_id": t["table_id"],
                "sampled_rows": t["sampled_rows"],
            }
            for t in tasks
        ]
        mode_hint = "tabular"
    elif mode == "cpa":
        payload = [
            {
                "task_id": t["task_id"],
                "table_id": t["table_id"],
                "subject_column": t["subject_column"],
                "target_column": t["target_column"],
                "sampled_rows": t["sampled_rows"],
            }
            for t in tasks
        ]
        mode_hint = "cpa"
    else:
        raise ValueError(f"Unknown selection mode: {mode}")

    types = ", ".join(type_ids)
    intro = _format_intro(TYPE_SELECT_INTRO)
    return "".join(
        [
            intro,
            f"Schema: {schema.name}\n",
            f"Input mode: {mode_hint}\n",
            f"Allowed type_ids (subset): {types}\n",
            "Return ONLY valid JSON.\n",
            "Output format (JSON array):\n",
            "[\"type_id\", \"type_id\"]\n",
            "Rules:\n",
            "- Only include type_ids from the allowed list.\n",
            "- Be recall-oriented: include all types that could apply.\n",
            "- Return unique type_ids only.\n",
            "- If none apply, return an empty list [].\n",
            "No extra text around the JSON.\n\n",
            "Input tasks JSON:\n",
            f"{json.dumps(payload, ensure_ascii=True)}",
        ]
    )


def build_table_prompt(schema: SchemaConfig, tasks: list[dict], type_ids: list[str]) -> str:
    payload = [
        {
            "task_id": t["task_id"],
            "table_id": t["table_id"],
            "sampled_rows": t["sampled_rows"],
        }
        for t in tasks
    ]
    types = ", ".join(type_ids)
    intro = _format_intro(schema.table_intro)
    if schema.require_all_scores:
        score_rule = "- Scores must be non-negative floats for every allowed type_id (include all keys).\n"
    else:
        score_rule = (
            "- Scores must be non-negative floats for selected type_ids only; "
            "omit unrelated types (missing keys treated as 0).\n"
        )
    return "".join(
        [
            intro,
            f"Schema: {schema.name}\n",
            f"Allowed type_ids: {types}\n",
            "Return ONLY valid JSON.\n",
            "Output format (JSON array):\n",
            "[\n",
            "  {\n",
            "    \"task_id\": \"...\",\n",
            "    \"table_id\": \"...\",\n",
            "    \"columns\": [\n",
            "      {\"column\": \"name\", \"scores\": {\"NER:PERSON\": 0.1, \"NER:OTHER\": 0.2}}\n",
            "    ]\n",
            "  }\n",
            "]\n",
            "Rules:\n",
            "- Return one entry per observed column name from the sampled_rows union.\n",
            score_rule,
            "- At least one score must be > 0 per column.\n",
            "No extra text around the JSON.\n\n",
            "Input tasks JSON:\n",
            f"{json.dumps(payload, ensure_ascii=True)}",
        ]
    )