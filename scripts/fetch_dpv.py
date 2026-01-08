#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

VERSION = "2.2"
TREE_URL = "https://api.github.com/repos/w3c/dpv/git/trees/master?recursive=1"
RAW_BASE = "https://raw.githubusercontent.com/w3c/dpv/master/"

PREF_LABEL = "http://www.w3.org/2004/02/skos/core#prefLabel"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"


def _fetch_json(url: str) -> Any:
    req = Request(url, headers={"User-Agent": "moose-dpv-fetch"})
    with urlopen(req) as resp:  # noqa: S310
        return json.load(resp)


def _shorten_iri(iri: str) -> str:
    for base in ("https://w3id.org/dpv/", "http://w3id.org/dpv/"):
        if iri.startswith(base):
            rest = iri[len(base) :]
            if rest.startswith("#"):
                return f"dpv:{rest[1:]}"
            if "#" in rest:
                namespace, local = rest.split("#", 1)
                return f"dpv-{namespace}:{local}"
            return f"dpv:{rest}"
    for base in ("https://w3id.org/dpv#", "http://w3id.org/dpv#"):
        if iri.startswith(base):
            return f"dpv:{iri[len(base):]}"
    return iri


def _pick_label(entry: dict[str, Any]) -> str | None:
    for key in (PREF_LABEL, RDFS_LABEL):
        values = entry.get(key)
        if not values:
            continue
        if isinstance(values, dict):
            values = [values]
        if isinstance(values, list):
            for item in values:
                if isinstance(item, dict):
                    if item.get("@language") == "en" and "@value" in item:
                        return item["@value"]
            for item in values:
                if isinstance(item, dict) and "@value" in item:
                    return item["@value"]
    return None


def _iter_entries(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        graph = data.get("@graph")
        if isinstance(graph, list):
            return [item for item in graph if isinstance(item, dict)]
    return []


def _list_jsonld_paths() -> list[str]:
    tree = _fetch_json(TREE_URL)
    paths: list[str] = []
    for item in tree.get("tree", []):
        path = item.get("path", "")
        if not path.startswith(f"{VERSION}/"):
            continue
        if not path.endswith(".jsonld"):
            continue
        if "-owl" in path:
            continue
        paths.append(path)
    return sorted(set(paths))


def main() -> int:
    paths = _list_jsonld_paths()
    if not paths:
        print("No JSON-LD files found; check repository structure.")
        return 1

    entries: dict[str, dict[str, str]] = {}
    for path in paths:
        url = f"{RAW_BASE}{path}"
        data = _fetch_json(url)
        for entry in _iter_entries(data):
            iri = entry.get("@id")
            if not isinstance(iri, str) or iri.startswith("_:"):
                continue
            label = _pick_label(entry)
            if not label:
                continue
            short_id = _shorten_iri(iri)
            if short_id in entries:
                continue
            entries[short_id] = {"id": short_id, "label": label, "iri": iri}

    ordered = [entries[key] for key in sorted(entries)]
    out_path = Path(__file__).resolve().parents[1] / "src" / "moose" / "data" / "dpv_full.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ordered, ensure_ascii=True, indent=2, sort_keys=True))
    print(f"Wrote {len(ordered)} entries to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
