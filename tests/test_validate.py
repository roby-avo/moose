from moose.validate import (
    extract_json,
    validate_ner_response,
    validate_ner_response_with_warnings,
    validate_table_response,
)


def test_extract_json_with_prefix():
    text = "Here you go: [{\"task_id\":\"t1\",\"entities\":[]}]"
    data = extract_json(text)
    assert isinstance(data, list)
    assert data[0]["task_id"] == "t1"


def test_validate_ner_response():
    tasks = [{"task_id": "t1", "text": "Roberto Avogadro"}]
    allowed = {"PERSON", "MISC"}
    raw = (
        "[{\"task_id\":\"t1\",\"entities\":[{\"start\":0,\"end\":7,"
        "\"text\":\"Roberto\",\"scores\":{\"PERSON\":1.0,\"MISC\":0.1}}]}]"
    )
    parsed = validate_ner_response(tasks, raw, allowed)
    assert parsed[0].entities[0].text == "Roberto"


def test_validate_ner_response_sparse_drops_unknown_score_keys_with_warning():
    tasks = [{"task_id": "t1", "text": "email addresses"}]
    allowed = {"dpv-pd:EmailAddress"}
    raw = (
        '[{"task_id":"t1","entities":[{"start":0,"end":15,"text":"email addresses",'
        '"scores":{"dpv-pd:EmailAddress":1.0,"dpv-pd:ThirdPartyAudience":0.8}}]}]'
    )

    parsed, warnings = validate_ner_response_with_warnings(
        tasks,
        raw,
        allowed,
        require_all_scores=False,
    )

    assert len(parsed) == 1
    assert len(parsed[0].entities) == 1
    assert set(parsed[0].entities[0].scores.keys()) == {"dpv-pd:EmailAddress"}
    assert any(w.get("code") == "unknown_score_keys_dropped" for w in warnings)


def test_validate_table_response_remaps_column_name_variants():
    tasks = [
        {
            "task_id": "t1",
            "table_id": "table-1",
            "sampled_rows": [
                {"patient_name": "Alice", "insurance_id": "DE-1"},
                {"patient_name": "Bob", "insurance_id": "DE-2"},
            ],
        }
    ]
    allowed = {"HC:FULL_NAME"}
    raw = (
        '[{"task_id":"t1","table_id":"table-1","columns":['
        '{"column":"patient name","scores":{"HC:FULL_NAME":1.0}},'
        '{"column":"insurance-id","scores":{"HC:FULL_NAME":0.2}}]}]'
    )

    parsed = validate_table_response(tasks, raw, allowed, require_all_scores=False)
    assert [c.column for c in parsed[0].columns] == ["patient_name", "insurance_id"]


def test_validate_table_response_rejects_ambiguous_column_mapping():
    tasks = [
        {
            "task_id": "t1",
            "table_id": "table-1",
            "sampled_rows": [
                {"a_b": "x", "a-b": "y"},
            ],
        }
    ]
    allowed = {"HC:FULL_NAME"}
    raw = (
        '[{"task_id":"t1","table_id":"table-1","columns":['
        '{"column":"a b","scores":{"HC:FULL_NAME":1.0}},'
        '{"column":"a-b","scores":{"HC:FULL_NAME":0.2}}]}]'
    )

    try:
        validate_table_response(tasks, raw, allowed, require_all_scores=False)
    except ValueError as exc:
        assert "Column names mismatch" in str(exc)
    else:
        raise AssertionError("Expected ValueError for ambiguous column remap")


def test_validate_table_response_repairs_all_zero_scores() -> None:
    tasks = [
        {
            "task_id": "t1",
            "table_id": "table-1",
            "sampled_rows": [{"patient_name": "Alice"}],
        }
    ]
    allowed = {"HC:FULL_NAME", "HC:PHONE"}
    raw = (
        '[{"task_id":"t1","table_id":"table-1","columns":['
        '{"column":"patient_name","scores":{"HC:FULL_NAME":0.0,"HC:PHONE":0.0}}]}]'
    )

    parsed = validate_table_response(tasks, raw, allowed, require_all_scores=False)
    scores = parsed[0].columns[0].scores
    assert scores["HC:FULL_NAME"] == 0.5
    assert scores["HC:PHONE"] == 0.5


def test_validate_table_response_seeds_scores_when_empty_sparse() -> None:
    tasks = [
        {
            "task_id": "t1",
            "table_id": "table-1",
            "sampled_rows": [{"patient_name": "Alice"}],
        }
    ]
    allowed = {"HC:FULL_NAME", "HC:PHONE", "HC:EMAIL"}
    raw = '[{"task_id":"t1","table_id":"table-1","columns":[{"column":"patient_name","scores":{}}]}]'

    parsed = validate_table_response(tasks, raw, allowed, require_all_scores=False)
    scores = parsed[0].columns[0].scores
    assert set(scores.keys()).issubset(allowed)
    assert sum(scores.values()) > 0


def test_validate_table_response_drops_unknown_and_repairs_sparse() -> None:
    tasks = [
        {
            "task_id": "t1",
            "table_id": "table-1",
            "sampled_rows": [{"patient_name": "Alice"}],
        }
    ]
    allowed = {"HC:FULL_NAME", "HC:PHONE"}
    raw = (
        '[{"task_id":"t1","table_id":"table-1","columns":['
        '{"column":"patient_name","scores":{"OTHER:TYPE":0.0}}]}]'
    )

    parsed = validate_table_response(tasks, raw, allowed, require_all_scores=False)
    scores = parsed[0].columns[0].scores
    assert set(scores.keys()).issubset(allowed)
    assert sum(scores.values()) > 0
