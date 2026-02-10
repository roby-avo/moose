from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from moose.schema import DATA_DIR

LEGAL_DIR = DATA_DIR / "legal"


def list_legal_sources() -> list[str]:
    """
    Discover available legal sources under moose/data/legal/<source>/.
    Example: ["gdprtext", "hipaa", ...]
    """
    if not LEGAL_DIR.exists():
        return []
    out: list[str] = []
    for p in LEGAL_DIR.iterdir():
        if p.is_dir() and not p.name.startswith("."):
            out.append(p.name)
    return sorted(set(out))


def _source_dir(source: str) -> Path:
    return LEGAL_DIR / source


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache
def load_legal_manifest(source: str) -> dict[str, Any] | None:
    """
    Optional manifest for a legal source:
      legal/<source>/manifest.json

    We don't require any schema here, but this gives a place to extend later with:
      - iri_prefixes for normalization
      - version
      - source urls
    """
    path = _source_dir(source) / "manifest.json"
    if not path.exists():
        return None
    data = _read_json(path)
    return data if isinstance(data, dict) else None


def normalize_legal_ref(source: str, ref: str) -> str:
    """
    Normalize a legal reference into the canonical ID used in our artifacts.

    This function is intentionally conservative: it tries a few safe conversions,
    especially for GDPRtEXT, and otherwise returns the input unchanged.

    Supported for gdprtext:
      - "gdprtext:R17" -> "gdprtext:R17"
      - "R17" -> "gdprtext:R17"  (best-effort convenience)
      - "https://w3id.org/GDPRtEXT#R17" -> "gdprtext:R17"
      - "http://w3id.org/GDPRtEXT#R17" -> "gdprtext:R17"

    If you later add HIPAA and its assets have different IRI bases, you can
    extend this by putting an iri-prefix mapping in legal/<source>/manifest.json.
    """
    if not isinstance(ref, str):
        return str(ref)
    r = ref.strip()
    if not r:
        return r

    # Already a CURIE-like ID
    if ":" in r and not r.startswith("http"):
        return r

    # --- GDPRtEXT-specific normalization (current observed IDs: gdprtext:R17) ---
    if source == "gdprtext":
        # IRI -> id
        for prefix in (
            "https://w3id.org/GDPRtEXT#",
            "http://w3id.org/GDPRtEXT#",
            "https://w3id.org/GDPRtEXT/",
            "http://w3id.org/GDPRtEXT/",
        ):
            if r.startswith(prefix):
                local = r[len(prefix) :]
                # handle possible paths like ".../gdpr#A5"
                if "#" in local:
                    local = local.split("#", 1)[1]
                local = local.strip("/").strip()
                if local:
                    return f"gdprtext:{local}"

        # raw local IDs like "R17", "A32"
        if re.fullmatch(r"[RA]\d+", r, flags=re.IGNORECASE):
            return f"gdprtext:{r.upper()}"

    # Future: use manifest iri-prefix mapping if provided
    manifest = load_legal_manifest(source)
    if manifest:
        iri_prefixes = manifest.get("iri_prefixes")
        if isinstance(iri_prefixes, dict) and r.startswith("http"):
            # example mapping:
            #   "https://example.org/law#" -> "law:"
            for iri_prefix, curie_prefix in iri_prefixes.items():
                if isinstance(iri_prefix, str) and isinstance(curie_prefix, str) and r.startswith(iri_prefix):
                    local = r[len(iri_prefix) :].strip()
                    if local:
                        return f"{curie_prefix}{local}"

    return r


def _iter_candidate_concept_files(source_dir: Path) -> list[Path]:
    """
    In each legal source directory, we support (any subset of):
      - concepts.json
      - articles.json
      - recitals.json
    We'll load what exists and merge them. concepts.json is preferred.
    """
    candidates = []
    for name in ("concepts.json", "articles.json", "recitals.json"):
        p = source_dir / name
        if p.exists():
            candidates.append(p)
    return candidates


@lru_cache
def load_legal_concepts_index(source: str) -> dict[str, dict[str, Any]]:
    """
    Load legal concepts and index them by "id".

    This is designed to be robust even when:
      - articles.json is empty (like your current gdprtext build)
      - only recitals.json exists
      - concepts.json exists and contains everything

    Return value:
      { "<id>": <raw concept dict> }
    """
    base = _source_dir(source)
    if not base.exists():
        raise FileNotFoundError(f"Legal source directory not found: {base}")

    files = _iter_candidate_concept_files(base)
    if not files:
        raise FileNotFoundError(
            f"No legal concept files found in {base}. Expected one of: concepts.json, articles.json, recitals.json"
        )

    index: dict[str, dict[str, Any]] = {}

    for path in files:
        data = _read_json(path)
        if not isinstance(data, list):
            # ignore unexpected structure
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            cid = item.get("id")
            if isinstance(cid, str) and cid.strip():
                # Keep first occurrence; later files shouldn't override concepts.json
                index.setdefault(cid, item)

    return index


def get_legal_concept(source: str, concept_id: str) -> dict[str, Any] | None:
    """
    Return the raw concept dict for concept_id, or None if missing.
    """
    cid = normalize_legal_ref(source, concept_id)
    idx = load_legal_concepts_index(source)
    return idx.get(cid)


def resolve_legal_refs(
    source: str,
    refs: list[str],
    *,
    include_text: bool = False,
    include_raw: bool = False,
) -> list[dict[str, Any]]:
    """
    Resolve a list of legal reference IDs to lightweight display metadata.

    - Returns ONLY the refs that can be resolved.
    - Unknown refs are silently skipped (caller can still include IDs in legal_refs).
    - include_text: include "text" field if present
    - include_raw: return full raw concept objects (not recommended for normal API responses)
    """
    if not refs:
        return []

    idx = load_legal_concepts_index(source)
    out: list[dict[str, Any]] = []

    for ref in refs:
        cid = normalize_legal_ref(source, ref)
        c = idx.get(cid)
        if not c:
            continue

        if include_raw:
            out.append(c)
            continue

        item: dict[str, Any] = {
            "id": c.get("id"),
            "label": c.get("label"),
            "iri": c.get("iri"),
            "kind": c.get("kind"),
            "number": c.get("number"),
        }
        if include_text:
            item["text"] = c.get("text") or c.get("definition")
        out.append(item)

    return out

def resolve_legal_refs_detail(source: str, ref_ids: list[str]) -> list[dict[str, Any]]:
    """
    Convenience wrapper used by privacy layer.
    Returns lightweight detail objects for refs that exist in the legal index.
    """
    return resolve_legal_refs(source, ref_ids, include_text=False, include_raw=False)


def filter_existing_legal_refs(source: str, ref_ids: list[str]) -> list[str]:
    """
    Return only those legal ref IDs that exist in the legal index (after normalization).
    Useful if you want legal_refs to contain only resolvable IDs.
    """
    if not ref_ids:
        return []
    idx = load_legal_concepts_index(source)
    out: list[str] = []
    for r in ref_ids:
        rid = normalize_legal_ref(source, r)
        if rid in idx:
            out.append(rid)
    return out