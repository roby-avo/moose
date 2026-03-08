#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _pick_records(data: Any) -> list[Any]:
    if isinstance(data, dict):
        for key in ("types", "type_ids", "items", "data"):
            if key in data:
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError("Schema input must be a JSON array or object containing one of: types, type_ids, items, data.")
    return data


def _normalize_aliases(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        alias = item.strip()
        if not alias or alias in seen:
            continue
        out.append(alias)
        seen.add(alias)
    return out


def normalize_types(
    raw: Any,
    *,
    id_field: str,
    label_field: str,
    description_field: str,
    aliases_field: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    records = _pick_records(raw)
    normalized: list[dict[str, Any]] = []
    type_ids: list[str] = []
    seen_ids: set[str] = set()

    for index, item in enumerate(records):
        if isinstance(item, str):
            type_id = item.strip()
            label = type_id
            description = None
            aliases: list[str] = []
        elif isinstance(item, dict):
            raw_type_id = item.get(id_field)
            if not isinstance(raw_type_id, str) or not raw_type_id.strip():
                raise ValueError(f"Item at index {index} is missing a valid '{id_field}' field.")
            type_id = raw_type_id.strip()
            raw_label = item.get(label_field)
            label = raw_label.strip() if isinstance(raw_label, str) and raw_label.strip() else type_id
            raw_desc = item.get(description_field)
            description = raw_desc.strip() if isinstance(raw_desc, str) and raw_desc.strip() else None
            aliases = _normalize_aliases(item.get(aliases_field))
        else:
            raise ValueError(f"Unsupported item type at index {index}: {type(item).__name__}")

        if type_id in seen_ids:
            raise ValueError(f"Duplicate type id: {type_id}")
        seen_ids.add(type_id)

        obj: dict[str, Any] = {"id": type_id, "label": label}
        if description:
            obj["description"] = description
        if aliases:
            obj["aliases"] = aliases
        normalized.append(obj)
        type_ids.append(type_id)

    if not type_ids:
        raise ValueError("No type ids were extracted from input.")
    return normalized, type_ids


def upsert_registry_entry(registry_path: Path, entry: dict[str, Any]) -> list[dict[str, Any]]:
    if registry_path.exists():
        data = read_json(registry_path)
        if not isinstance(data, list):
            raise ValueError(f"Registry must be a JSON array: {registry_path}")
        registry = [item for item in data if isinstance(item, dict)]
    else:
        registry = []

    replaced = False
    for idx, item in enumerate(registry):
        if item.get("name") == entry["name"]:
            registry[idx] = {**item, **entry}
            replaced = True
            break
    if not replaced:
        registry.append(entry)

    return sorted(registry, key=lambda x: str(x.get("name", "")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest a user-provided schema into Moose registry files.")
    parser.add_argument("--name", required=True, help="Schema name, e.g. customer_pii")
    parser.add_argument("--input", required=True, help="Path to source JSON schema file")
    parser.add_argument("--label", default=None, help="Display label (defaults to name)")
    parser.add_argument("--description", default=None, help="Schema description")
    parser.add_argument("--score-mode", choices=["dense", "sparse"], default="sparse")
    parser.add_argument("--prefilter-types", action="store_true")
    parser.add_argument("--supports-cpa", action="store_true")
    parser.add_argument("--disable-text", action="store_true", help="Disable text annotation support")
    parser.add_argument("--disable-table", action="store_true", help="Disable table annotation support")
    parser.add_argument("--text-intro", default=None, help="Optional custom text prompt intro")
    parser.add_argument("--table-intro", default=None, help="Optional custom table prompt intro")
    parser.add_argument("--cpa-intro", default=None, help="Optional custom CPA prompt intro")
    parser.add_argument("--id-field", default="id", help="ID field name when input uses object items")
    parser.add_argument("--label-field", default="label", help="Label field name when input uses object items")
    parser.add_argument("--description-field", default="description", help="Description field name for object items")
    parser.add_argument("--aliases-field", default="aliases", help="Aliases field name for object items")
    parser.add_argument("--data-dir", default=None, help="Path to src/moose/data (auto-detected by default)")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not NAME_RE.match(args.name):
        raise SystemExit("Schema name must match: ^[A-Za-z0-9][A-Za-z0-9_-]*$")

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir).resolve() if args.data_dir else (repo_root / "src" / "moose" / "data")
    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    raw = read_json(input_path)
    types, type_ids = normalize_types(
        raw,
        id_field=args.id_field,
        label_field=args.label_field,
        description_field=args.description_field,
        aliases_field=args.aliases_field,
    )

    schema_dir = data_dir / "user" / args.name
    types_path = schema_dir / "types.json"
    type_ids_path = schema_dir / "type_ids.json"
    manifest_path = schema_dir / "manifest.json"
    user_registry_path = data_dir / "user_vocabularies.json"

    registry_entry: dict[str, Any] = {
        "name": args.name,
        "label": args.label or args.name,
        "description": args.description or f"User-ingested schema '{args.name}'.",
        "type_source": f"user/{args.name}/type_ids.json",
        "score_mode": args.score_mode,
        "supports_text": not bool(args.disable_text),
        "supports_table": not bool(args.disable_table),
        "supports_cpa": bool(args.supports_cpa),
        "prefilter_types": bool(args.prefilter_types),
    }
    if args.text_intro:
        registry_entry["text_intro"] = args.text_intro
    if args.table_intro:
        registry_entry["table_intro"] = args.table_intro
    if args.cpa_intro:
        registry_entry["cpa_intro"] = args.cpa_intro

    manifest = {
        "name": args.name,
        "label": registry_entry["label"],
        "description": registry_entry["description"],
        "source_file": str(input_path),
        "paths": {
            "types": str(types_path.relative_to(data_dir).as_posix()),
            "type_ids": str(type_ids_path.relative_to(data_dir).as_posix()),
        },
        "counts": {"types": len(type_ids)},
    }

    updated_registry = upsert_registry_entry(user_registry_path, registry_entry)

    if args.dry_run:
        print("[DRY-RUN] schema ingestion summary")
        print(f"  name:            {args.name}")
        print(f"  input:           {input_path}")
        print(f"  data_dir:        {data_dir}")
        print(f"  type_count:      {len(type_ids)}")
        print(f"  schema_dir:      {schema_dir}")
        print(f"  user_registry:   {user_registry_path}")
        return 0

    write_json(types_path, types)
    write_json(type_ids_path, type_ids)
    write_json(manifest_path, manifest)
    write_json(user_registry_path, updated_registry)

    print("Ingested schema successfully:")
    print(f"  name:            {args.name}")
    print(f"  type_count:      {len(type_ids)}")
    print(f"  types:           {types_path}")
    print(f"  type_ids:        {type_ids_path}")
    print(f"  manifest:        {manifest_path}")
    print(f"  user_registry:   {user_registry_path}")
    print("Next: call reload_schema_registry() in-process (or restart API) to activate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
