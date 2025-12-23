from __future__ import annotations

import json


def build_text_ner_prompt(schema: str, tasks: list[dict], type_ids: list[str]) -> str:
    payload = [{"task_id": t["task_id"], "text": t["text"]} for t in tasks]
    types = ", ".join(type_ids)
    return (
        "You are a high-precision NER engine.\n"
        f"Schema: {schema}\n"
        f"Allowed type_ids: {types}\n"
        "Return ONLY valid JSON.\n"
        "Output format (JSON array):\n"
        "[\n"
        "  {\n"
        "    \"task_id\": \"...\",\n"
        "    \"entities\": [\n"
        "      {\n"
        "        \"start\": 0,\n"
        "        \"end\": 0,\n"
        "        \"text\": \"exact substring\",\n"
        "        \"scores\": {\"NER:PERSON\": 0.1, \"NER:OTHER\": 0.2}\n"
        "      }\n"
        "    ]\n"
        "  }\n"
        "]\n"
        "Rules:\n"
        "- Offsets are 0-based, end-exclusive.\n"
        "- The text must match the exact substring from the input text.\n"
        "- Scores must be non-negative floats for every allowed type_id (include all keys).\n"
        "- At least one score must be > 0 per entity.\n"
        "No extra text around the JSON.\n\n"
        "Input tasks JSON:\n"
        f"{json.dumps(payload, ensure_ascii=True)}"
    )


def build_table_prompt(schema: str, tasks: list[dict], type_ids: list[str]) -> str:
    payload = [
        {
            "task_id": t["task_id"],
            "table_id": t["table_id"],
            "sampled_rows": t["sampled_rows"],
        }
        for t in tasks
    ]
    types = ", ".join(type_ids)
    return (
        "You are a semantic typing engine for tabular data.\n"
        f"Schema: {schema}\n"
        f"Allowed type_ids: {types}\n"
        "Return ONLY valid JSON.\n"
        "Output format (JSON array):\n"
        "[\n"
        "  {\n"
        "    \"task_id\": \"...\",\n"
        "    \"table_id\": \"...\",\n"
        "    \"columns\": [\n"
        "      {\"column\": \"name\", \"scores\": {\"NER:PERSON\": 0.1, \"NER:OTHER\": 0.2}}\n"
        "    ]\n"
        "  }\n"
        "]\n"
        "Rules:\n"
        "- Return one entry per observed column name from the sampled_rows union.\n"
        "- Scores must be non-negative floats for every allowed type_id (include all keys).\n"
        "- At least one score must be > 0 per column.\n"
        "No extra text around the JSON.\n\n"
        "Input tasks JSON:\n"
        f"{json.dumps(payload, ensure_ascii=True)}"
    )
