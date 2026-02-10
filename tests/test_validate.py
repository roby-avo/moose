from moose.validate import extract_json, validate_ner_response


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
