from __future__ import annotations

from moose.schema import get_schema_config


def test_dpv_pd_schema_includes_pd_and_ai_type_ids() -> None:
    type_ids = get_schema_config("dpv_pd").load_type_ids()
    type_set = set(type_ids)

    assert "dpv-pd:Accent" in type_set
    assert "dpv-ai:AISystem" in type_set
    assert len(type_ids) == len(type_set)
