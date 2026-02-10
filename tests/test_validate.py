from moose.validate import (
    extract_json,
    validate_ner_response,
    validate_ner_response_with_warnings,
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
