from __future__ import annotations

import json
from pathlib import Path

import pytest

import moose.schema as schema_mod
from moose_api.main import get_assets_index


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


@pytest.mark.asyncio
async def test_assets_index_includes_user_schemas(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(schema_mod, "DATA_DIR", data_dir)

    _write_json(
        data_dir / "assets_index.json",
        {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "registries": {"vocabularies": "vocabularies.json"},
            "assets": {},
        },
    )
    _write_json(
        data_dir / "user_vocabularies.json",
        [
            {
                "name": "acme_schema",
                "label": "Acme Schema",
                "description": "Acme custom types",
                "type_source": "user/acme_schema/type_ids.json",
                "score_mode": "sparse",
                "supports_text": True,
                "supports_table": True,
                "supports_cpa": False,
                "prefilter_types": False,
            }
        ],
    )
    _write_json(data_dir / "user" / "acme_schema" / "type_ids.json", ["ACME:EMAIL", "ACME:PHONE"])
    _write_json(
        data_dir / "user" / "acme_schema" / "manifest.json",
        {
            "name": "acme_schema",
            "generated_at": "2026-03-05T10:00:00+00:00",
            "source": {"type": "url", "url": "https://example.com/acme.json"},
            "counts": {"types": 2},
        },
    )

    payload = await get_assets_index()
    assert payload["registries"]["user_vocabularies"] == "user_vocabularies.json"
    assert "user_schemas" in payload["assets"]
    user_schemas = payload["assets"]["user_schemas"]
    assert isinstance(user_schemas, list)
    assert len(user_schemas) == 1
    assert user_schemas[0]["name"] == "acme_schema"
    assert user_schemas[0]["type_count"] == 2
    assert user_schemas[0]["source"] == {"type": "url", "url": "https://example.com/acme.json"}
