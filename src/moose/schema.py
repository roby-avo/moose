from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

COARSE_TYPES = [
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "EVENT",
    "WORK",
    "PRODUCT",
    "CONCEPT",
    "MISC",
]

FINE_TYPES = [
    "PERSON",
    "FICTIONAL_CHARACTER",
    "COMPANY",
    "NONPROFIT_ORG",
    "GOVERNMENT_ORG",
    "EDUCATIONAL_ORG",
    "SPORTS_TEAM",
    "COUNTRY",
    "CITY",
    "REGION",
    "LANDMARK",
    "CELESTIAL_BODY",
    "CONFLICT",
    "SPORT_EVENT",
    "EVENT_GENERIC",
    "FILM",
    "BOOK",
    "MUSIC_WORK",
    "SOFTWARE",
    "INTERNET_MEME",
    "DEVICE",
    "MEDICATION",
    "FOOD_BEVERAGE",
    "PRODUCT_GENERIC",
    "LANGUAGE",
    "LAW",
    "SCIENTIFIC_THEORY",
    "BIOLOGICAL_TAXON",
    "ANATOMY",
]

FINE_TO_COARSE = {
    "PERSON": "PERSON",
    "FICTIONAL_CHARACTER": "PERSON",
    "COMPANY": "ORGANIZATION",
    "NONPROFIT_ORG": "ORGANIZATION",
    "GOVERNMENT_ORG": "ORGANIZATION",
    "EDUCATIONAL_ORG": "ORGANIZATION",
    "SPORTS_TEAM": "ORGANIZATION",
    "COUNTRY": "LOCATION",
    "CITY": "LOCATION",
    "REGION": "LOCATION",
    "LANDMARK": "LOCATION",
    "CELESTIAL_BODY": "LOCATION",
    "CONFLICT": "EVENT",
    "SPORT_EVENT": "EVENT",
    "EVENT_GENERIC": "EVENT",
    "FILM": "WORK",
    "BOOK": "WORK",
    "MUSIC_WORK": "WORK",
    "SOFTWARE": "WORK",
    "INTERNET_MEME": "WORK",
    "DEVICE": "PRODUCT",
    "MEDICATION": "PRODUCT",
    "FOOD_BEVERAGE": "PRODUCT",
    "PRODUCT_GENERIC": "PRODUCT",
    "LANGUAGE": "CONCEPT",
    "LAW": "CONCEPT",
    "SCIENTIFIC_THEORY": "CONCEPT",
    "BIOLOGICAL_TAXON": "CONCEPT",
    "ANATOMY": "CONCEPT",
}

DATA_DIR = Path(__file__).resolve().parent / "data"
VOCAB_REGISTRY_PATH = DATA_DIR / "vocabularies.json"
USER_VOCAB_REGISTRY_PATH = DATA_DIR / "user_vocabularies.json"

DEFAULT_TEXT_INTRO = "You are a high-precision NER engine."
DEFAULT_TABLE_INTRO = "You are a semantic typing engine for tabular data."
DEFAULT_CPA_INTRO = "You are a column relationship prediction engine."


@dataclass(frozen=True)
class SchemaConfig:
    name: str
    label: str
    description: str | None
    require_all_scores: bool
    text_intro: str
    table_intro: str
    cpa_intro: str
    type_ids: tuple[str, ...] | None = None
    data_path: Path | None = None
    coarse_mapping: dict[str, str] | None = None
    type_aliases: dict[str, str] | None = None
    type_alias_prefixes: dict[str, str] | None = None
    prefilter_types: bool = False
    supports_text: bool = True
    supports_table: bool = True
    supports_cpa: bool = False

    def load_type_ids(self) -> list[str]:
        if self.type_ids is not None:
            return list(self.type_ids)
        if self.data_path is None:
            raise ValueError(f"Schema {self.name} has no type source configured.")
        return list(_load_vocab_file(self.data_path))


def _extract_type_ids(data: Any) -> list[str]:
    if isinstance(data, dict):
        for key in ("types", "type_ids", "items", "data"):
            if key in data:
                data = data[key]
                break
    if isinstance(data, list):
        type_ids: list[str] = []
        for item in data:
            if isinstance(item, str):
                type_ids.append(item)
            elif isinstance(item, dict):
                type_id = item.get("id")
                if isinstance(type_id, str):
                    type_ids.append(type_id)
        return type_ids
    return []


@lru_cache
def _load_vocab_file(path: Path) -> tuple[str, ...]:
    if not path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    type_ids = _extract_type_ids(data)
    if not type_ids:
        raise ValueError(f"No type ids found in vocabulary file: {path}")
    return tuple(type_ids)


def _normalize_intro(value: Any, fallback: str) -> str:
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
    return fallback


def _parse_score_mode(entry: dict[str, Any]) -> bool:
    require_all = entry.get("require_all_scores")
    if isinstance(require_all, bool):
        return require_all
    score_mode = entry.get("score_mode", "dense")
    if score_mode not in {"dense", "sparse"}:
        raise ValueError(f"Invalid score_mode: {score_mode}")
    return score_mode == "dense"


def _parse_alias_mapping(value: Any, label: str) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    mapping: dict[str, str] = {}
    for key, val in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{label} keys must be non-empty strings.")
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"{label} values must be non-empty strings.")
        mapping[key.strip()] = val.strip()
    return mapping or None


def _parse_coarse_mapping(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("coarse_mapping must be a JSON object.")
    mapping: dict[str, str] = {}
    for key, val in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("coarse_mapping keys must be non-empty strings.")
        if not isinstance(val, str) or not val.strip():
            raise ValueError("coarse_mapping values must be non-empty strings.")
        mapping[key.strip()] = val.strip()
    return mapping or None


def _parse_bool_field(entry: dict[str, Any], key: str, default: bool) -> bool:
    value = entry.get(key)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean.")
    return value


def _load_vocab_registry_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Vocabulary registry must be a JSON array.")
    for entry in data:
        if not isinstance(entry, dict):
            raise ValueError("Vocabulary registry entries must be JSON objects.")
    return data


@lru_cache
def _schema_registry() -> dict[str, SchemaConfig]:
    registry: dict[str, SchemaConfig] = {
        "coarse": SchemaConfig(
            name="coarse",
            label="Coarse",
            description="Coarse-grained NER schema.",
            require_all_scores=True,
            text_intro=DEFAULT_TEXT_INTRO,
            table_intro=DEFAULT_TABLE_INTRO,
            cpa_intro=DEFAULT_CPA_INTRO,
            type_ids=tuple(COARSE_TYPES),
            supports_cpa=False,
        ),
        "fine": SchemaConfig(
            name="fine",
            label="Fine",
            description="Fine-grained NER schema with coarse mapping.",
            require_all_scores=False,
            text_intro=DEFAULT_TEXT_INTRO,
            table_intro=DEFAULT_TABLE_INTRO,
            cpa_intro=DEFAULT_CPA_INTRO,
            type_ids=tuple(FINE_TYPES),
            coarse_mapping=FINE_TO_COARSE,
            type_aliases={"HUMAN": "PERSON"},
            supports_cpa=False,
        ),
    }

    seen_registry_names: set[str] = set()
    registry_paths = [VOCAB_REGISTRY_PATH, USER_VOCAB_REGISTRY_PATH]
    for registry_path in registry_paths:
        for entry in _load_vocab_registry_entries(registry_path):
            name = entry.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Vocabulary entry missing a valid name.")
            name = name.strip()
            if name in seen_registry_names:
                raise ValueError(f"Duplicate schema name: {name}")
            seen_registry_names.add(name)
            source = entry.get("type_source") or entry.get("data_path") or entry.get("path")
            if not isinstance(source, str) or not source.strip():
                raise ValueError(f"Vocabulary {name} missing type_source path.")
            data_path = Path(source.strip())
            if not data_path.is_absolute():
                data_path = DATA_DIR / data_path
            label = entry.get("label")
            description = entry.get("description")
            registry[name] = SchemaConfig(
                name=name,
                label=label.strip() if isinstance(label, str) and label.strip() else name,
                description=description.strip() if isinstance(description, str) else None,
                require_all_scores=_parse_score_mode(entry),
                text_intro=_normalize_intro(entry.get("text_intro"), DEFAULT_TEXT_INTRO),
                table_intro=_normalize_intro(entry.get("table_intro"), DEFAULT_TABLE_INTRO),
                cpa_intro=_normalize_intro(entry.get("cpa_intro"), DEFAULT_CPA_INTRO),
                data_path=data_path,
                type_aliases=_parse_alias_mapping(entry.get("type_aliases"), "type_aliases"),
                type_alias_prefixes=_parse_alias_mapping(
                    entry.get("type_alias_prefixes"), "type_alias_prefixes"
                ),
                coarse_mapping=_parse_coarse_mapping(entry.get("coarse_mapping")),
                prefilter_types=_parse_bool_field(entry, "prefilter_types", False),
                supports_text=_parse_bool_field(entry, "supports_text", True),
                supports_table=_parse_bool_field(entry, "supports_table", True),
                supports_cpa=_parse_bool_field(entry, "supports_cpa", False),
            )
    return registry


def list_schema_names() -> list[str]:
    return sorted(_schema_registry())


def reload_schema_registry() -> list[str]:
    _load_vocab_file.cache_clear()
    _schema_registry.cache_clear()
    return list_schema_names()


def get_schema_config(schema: str) -> SchemaConfig:
    registry = _schema_registry()
    if schema not in registry:
        # In multi-worker deployments, one worker may ingest a schema while another
        # still has a stale in-memory cache. Refresh once before failing.
        reload_schema_registry()
        registry = _schema_registry()
    if schema not in registry:
        known = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Unknown schema '{schema}'. Available schemas: {known}")
    return registry[schema]


def get_types(schema: str) -> list[str]:
    config = get_schema_config(schema)
    return config.load_type_ids()


def get_coarse_type(type_id: str) -> str | None:
    if type_id in COARSE_TYPES:
        return type_id
    return FINE_TO_COARSE.get(type_id)
