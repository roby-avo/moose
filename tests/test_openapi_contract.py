from __future__ import annotations

from moose_api.main import app


def _resolve_component(schema: dict, spec: dict) -> dict:
    ref = schema.get("$ref")
    if not isinstance(ref, str):
        return schema
    prefix = "#/components/schemas/"
    if not ref.startswith(prefix):
        raise AssertionError(f"Unexpected schema ref: {ref}")
    name = ref[len(prefix) :]
    return spec["components"]["schemas"][name]


def test_openapi_has_current_core_post_endpoints() -> None:
    spec = app.openapi()
    paths = set(spec.get("paths", {}))
    expected = {
        "/ner",
        "/schemas/{schema}/ner",
        "/tabular/annotate",
        "/schemas/{schema}/tabular/annotate",
        "/tabular/cpa",
        "/schemas/{schema}/tabular/cpa",
        "/privacy/analyze",
        "/schemas/ingest/preview",
        "/jobs/{job_id}/privacy-report",
        "/privacy/reports/schema",
    }
    assert expected.issubset(paths)


def test_openapi_request_models_exclude_removed_flags() -> None:
    spec = app.openapi()
    endpoint_to_removed = {
        "/ner": {"include_scores", "strict_offsets"},
        "/schemas/{schema}/ner": {"include_scores", "strict_offsets"},
        "/tabular/annotate": {"include_scores"},
        "/schemas/{schema}/tabular/annotate": {"include_scores"},
        "/tabular/cpa": {"include_scores"},
        "/schemas/{schema}/tabular/cpa": {"include_scores"},
    }

    for path, removed in endpoint_to_removed.items():
        op = spec["paths"][path]["post"]
        schema = op["requestBody"]["content"]["application/json"]["schema"]
        resolved = _resolve_component(schema, spec)
        properties = set((resolved.get("properties") or {}).keys())
        assert properties.isdisjoint(removed), (path, sorted(properties.intersection(removed)))


def test_openapi_llm_api_key_uses_authorize_security_scheme() -> None:
    spec = app.openapi()
    paths = spec["paths"]

    llm_paths = [
        "/ner",
        "/schemas/{schema}/ner",
        "/tabular/annotate",
        "/schemas/{schema}/tabular/annotate",
        "/tabular/cpa",
        "/schemas/{schema}/tabular/cpa",
        "/privacy/analyze",
        "/models",
    ]

    for path in llm_paths:
        op = paths[path]["post"] if path != "/models" else paths[path]["get"]
        params = op.get("parameters", [])
        header_names = {
            p.get("name")
            for p in params
            if isinstance(p, dict) and p.get("in") == "header"
        }
        assert "X-LLM-API-Key" not in header_names, (path, sorted(header_names))

        security = op.get("security", [])
        security_keys = set()
        for item in security:
            if isinstance(item, dict):
                security_keys.update(item.keys())
        assert "X-LLM-API-Key" in security_keys, (path, security)
