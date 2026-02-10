from __future__ import annotations

from moose.schema import COARSE_TYPES, FINE_TO_COARSE, get_schema_config


def test_fine_schema_uses_person_not_human() -> None:
    schema = get_schema_config("fine")
    type_ids = schema.load_type_ids()

    assert "PERSON" in type_ids
    assert "HUMAN" not in type_ids
    assert FINE_TO_COARSE["PERSON"] == "PERSON"


def test_fine_schema_alias_and_sparse_scores() -> None:
    schema = get_schema_config("fine")
    assert schema.require_all_scores is False
    assert schema.type_aliases is not None
    assert schema.type_aliases.get("HUMAN") == "PERSON"


def test_ner_schema_excludes_relation_like_labels() -> None:
    schema = get_schema_config("fine")
    type_ids = schema.load_type_ids()

    assert "RELATION" not in COARSE_TYPES
    assert "PROPERTY" not in type_ids
    assert "ENTITY" not in type_ids
    assert "PROPERTY" not in FINE_TO_COARSE
    assert "ENTITY" not in FINE_TO_COARSE


def test_ner_schemas_are_loaded_from_vocab_registry() -> None:
    coarse_schema = get_schema_config("coarse")
    fine_schema = get_schema_config("fine")

    assert coarse_schema.data_path is not None
    assert fine_schema.data_path is not None
    assert coarse_schema.type_ids is None
    assert fine_schema.type_ids is None
