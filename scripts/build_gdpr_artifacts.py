#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx


# -----------------------
# Source configuration
# -----------------------
REPO_OWNER = "coolharsh55"
REPO_NAME = "GDPRtEXT"
BRANCH = "master"

TREE_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/git/trees/{BRANCH}?recursive=1"
RAW_BASE = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}/"


# -----------------------
# Common RDF/JSON-LD constants (full IRIs)
# -----------------------
SKOS_PREF_LABEL = "http://www.w3.org/2004/02/skos/core#prefLabel"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
SKOS_ALT_LABEL = "http://www.w3.org/2004/02/skos/core#altLabel"
SKOS_DEFINITION = "http://www.w3.org/2004/02/skos/core#definition"
RDFS_COMMENT = "http://www.w3.org/2000/01/rdf-schema#comment"

SKOS_BROADER = "http://www.w3.org/2004/02/skos/core#broader"
RDFS_SUBCLASS_OF = "http://www.w3.org/2000/01/rdf-schema#subClassOf"

SKOS_IN_SCHEME = "http://www.w3.org/2004/02/skos/core#inScheme"
SKOS_TOP_CONCEPT_OF = "http://www.w3.org/2004/02/skos/core#topConceptOf"

DCT_DESCRIPTION = "http://purl.org/dc/terms/description"
DCT_IDENTIFIER = "http://purl.org/dc/terms/identifier"
RDF_VALUE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#value"
SCHEMA_TEXT = "http://schema.org/text"


# -----------------------
# Optional DeepSeek augmentation (keywords)
# -----------------------
def _deepseek_client():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return None
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def augment_keywords_with_deepseek(
    concepts: list[dict[str, Any]],
    out_dir: Path,
    batch_size: int = 25,
    model: str = "deepseek-chat",
) -> dict[str, list[str]]:
    """
    OPTIONAL: produce non-authoritative keywords for search/matching.

    Writes:
      out_dir/llm/keywords.json
      out_dir/llm/cache.json
    """
    client = _deepseek_client()
    if client is None:
        raise RuntimeError("DeepSeek keywords requested but DEEPSEEK_API_KEY/openai not available.")

    llm_dir = out_dir / "llm"
    llm_dir.mkdir(parents=True, exist_ok=True)
    keywords_path = llm_dir / "keywords.json"
    cache_path = llm_dir / "cache.json"

    keywords: dict[str, list[str]] = {}
    if keywords_path.exists():
        keywords = json.loads(keywords_path.read_text(encoding="utf-8"))

    cache: dict[str, Any] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    pending = [c for c in concepts if c["id"] not in keywords]

    system_prompt = (
        "Generate lightweight keywords for GDPR/GDPRtEXT concepts.\n"
        "Return ONLY JSON: {\"items\":[{\"id\":\"...\",\"keywords\":[...]}]}.\n"
        "Rules:\n"
        "- keywords must be lowercase\n"
        "- 3 to 8 keywords per item\n"
        "- keywords should come from the label/definition/text, not invented\n"
        "- short phrases allowed (<=3 words)\n"
    )

    def chunks(lst: list[Any], n: int) -> Iterable[list[Any]]:
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    for batch in chunks(pending, batch_size):
        payload = [{"id": c["id"], "label": c.get("label"), "definition": c.get("definition")} for c in batch]

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps({"items": payload}, ensure_ascii=True)},
            ],
            response_format={"type": "json_object"},
        )

        data = json.loads(resp.choices[0].message.content)
        items = data.get("items", [])
        if not isinstance(items, list):
            continue

        for item in items:
            if not isinstance(item, dict):
                continue
            cid = item.get("id")
            kws = item.get("keywords")
            if not isinstance(cid, str) or not isinstance(kws, list):
                continue
            cleaned = []
            seen = set()
            for k in kws:
                if not isinstance(k, str):
                    continue
                kk = k.strip().lower()
                if not kk or kk in seen:
                    continue
                cleaned.append(kk)
                seen.add(kk)
            if cleaned:
                keywords[cid] = cleaned
                cache[cid] = {"keywords": cleaned}

        # incremental save
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        keywords_path.write_text(json.dumps(keywords, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    return keywords


# -----------------------
# HTTP helpers
# -----------------------
def fetch_json(url: str, github_token: str | None = None) -> Any:
    headers = {"User-Agent": "moose-gdpr-build"}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    with httpx.Client(timeout=60.0, follow_redirects=True, headers=headers) as client:
        r = client.get(url)
        r.raise_for_status()
        return r.json()


def list_repo_paths(version_filter: str | None, github_token: str | None) -> list[str]:
    """
    GDPRtEXT does not have a simple "version directory" like DPV.
    We keep a --version-filter option to filter by path substring if needed.
    """
    tree = fetch_json(TREE_URL, github_token=github_token)
    paths: list[str] = []
    for item in tree.get("tree", []):
        path = item.get("path", "")
        if not isinstance(path, str):
            continue
        if version_filter and version_filter not in path:
            continue
        paths.append(path)
    return sorted(set(paths))


# -----------------------
# JSON-LD parsing helpers
# -----------------------
def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def iter_entries(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        graph = data.get("@graph")
        if isinstance(graph, list):
            return [x for x in graph if isinstance(x, dict)]
    return []


def pick_lang_value(values: Any, lang: str = "en") -> str | None:
    if values is None:
        return None
    if isinstance(values, str):
        return values.strip() or None
    if isinstance(values, dict):
        values = [values]
    if not isinstance(values, list):
        return None

    # prefer requested language
    for item in values:
        if isinstance(item, dict) and item.get("@language") == lang and isinstance(item.get("@value"), str):
            v = item["@value"].strip()
            if v:
                return v

    # fallback any @value
    for item in values:
        if isinstance(item, dict) and isinstance(item.get("@value"), str):
            v = item["@value"].strip()
            if v:
                return v

    # fallback string items
    for item in values:
        if isinstance(item, str) and item.strip():
            return item.strip()

    return None


def extract_labels(entry: dict[str, Any]) -> tuple[str | None, list[str]]:
    label = pick_lang_value(entry.get(SKOS_PREF_LABEL)) or pick_lang_value(entry.get(RDFS_LABEL))
    alt_values = entry.get(SKOS_ALT_LABEL)
    alt_labels: list[str] = []
    if alt_values:
        for item in as_list(alt_values):
            s = pick_lang_value(item)
            if s:
                alt_labels.append(s)

    # dedupe
    seen = set()
    dedup = []
    for a in alt_labels:
        if a not in seen and a != label:
            dedup.append(a)
            seen.add(a)
    return label, dedup


def extract_definition(entry: dict[str, Any]) -> str | None:
    return pick_lang_value(entry.get(SKOS_DEFINITION)) or pick_lang_value(entry.get(RDFS_COMMENT))


def extract_types(entry: dict[str, Any]) -> list[str]:
    t = entry.get("@type")
    if isinstance(t, str):
        return [t]
    if isinstance(t, list):
        return [x for x in t if isinstance(x, str)]
    return []


def extract_id_refs(entry: dict[str, Any], key: str, shorten) -> list[str]:
    out: list[str] = []
    for item in as_list(entry.get(key)):
        if isinstance(item, str):
            out.append(shorten(item))
        elif isinstance(item, dict) and isinstance(item.get("@id"), str):
            out.append(shorten(item["@id"]))
    # dedupe preserve order
    seen = set()
    dedup = []
    for x in out:
        if x not in seen:
            dedup.append(x)
            seen.add(x)
    return dedup


# -----------------------
# GDPRtEXT ID normalization
# -----------------------
def shorten_gdpr_iri(iri: str) -> str:
    """
    Convert GDPRtEXT IRIs into compact IDs.

    Common patterns seen in GDPRtEXT:
      https://w3id.org/GDPRtEXT#X
      https://w3id.org/GDPRtEXT/gdpr#A5
      https://w3id.org/GDPRtEXT/terms#PersonalData
    We normalize to:
      gdprtext:X
      gdprtext-gdpr:A5
      gdprtext-terms:PersonalData
    """
    if not isinstance(iri, str):
        return str(iri)

    for base in ("https://w3id.org/GDPRtEXT/", "http://w3id.org/GDPRtEXT/"):
        if iri.startswith(base):
            rest = iri[len(base) :]
            if rest.startswith("#"):
                return f"gdprtext:{rest[1:]}"
            if "#" in rest:
                namespace, local = rest.split("#", 1)
                namespace = namespace.strip("/").replace("/", "_")
                return f"gdprtext-{namespace}:{local}"
            # fallback
            return f"gdprtext:{rest.strip('/')}"
    for base in ("https://w3id.org/GDPRtEXT#", "http://w3id.org/GDPRtEXT#"):
        if iri.startswith(base):
            return f"gdprtext:{iri[len(base):]}"
    return iri


def classify_concept(label: str | None, concept_id: str) -> tuple[str, int | None]:
    """
    Identify article/recital by label patterns.
    """
    if label:
        m = re.match(r"^\s*Article\s+(\d+)\b", label, flags=re.IGNORECASE)
        if m:
            return "article", int(m.group(1))
        m = re.match(r"^\s*Recital\s+(\d+)\b", label, flags=re.IGNORECASE)
        if m:
            return "recital", int(m.group(1))

    # fallback: id pattern
    m = re.search(r":A(\d+)\b", concept_id)
    if m:
        return "article", int(m.group(1))
    m = re.search(r":R(\d+)\b", concept_id)
    if m:
        return "recital", int(m.group(1))

    return "concept", None


def extract_text(entry: dict[str, Any]) -> str | None:
    """
    Try common fields where GDPRtEXT might store the legal text.
    """
    for key in (SCHEMA_TEXT, RDF_VALUE, DCT_DESCRIPTION, RDFS_COMMENT):
        t = pick_lang_value(entry.get(key))
        if t:
            return t
    return None


@dataclass
class GDPRArtifacts:
    concepts: dict[str, dict[str, Any]]
    concept_parents: dict[str, list[str]]
    concept_ancestors: dict[str, list[str]]
    schemes: dict[str, dict[str, Any]]
    module_index: dict[str, dict[str, Any]]


def compute_ancestors(parents: dict[str, list[str]]) -> dict[str, list[str]]:
    memo: dict[str, list[str]] = {}
    visiting: set[str] = set()

    def dfs(n: str) -> list[str]:
        if n in memo:
            return memo[n]
        if n in visiting:
            return []
        visiting.add(n)
        out: list[str] = []
        for p in parents.get(n, []):
            if p not in out:
                out.append(p)
            for ap in dfs(p):
                if ap not in out:
                    out.append(ap)
        visiting.remove(n)
        memo[n] = out
        return out

    for node in parents.keys():
        dfs(node)
    return memo


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def build_gdpr_artifacts(
    out_dir: Path,
    cache_raw: bool,
    include_paths_regex: str | None,
    github_token: str | None,
) -> GDPRArtifacts:
    paths = list_repo_paths(version_filter=None, github_token=github_token)

    # Prefer jsonld/json; if none exist, you can extend script to parse ttl via rdflib later
    jsonld_paths = [p for p in paths if p.endswith(".jsonld") or p.endswith(".json")]
    if include_paths_regex:
        rx = re.compile(include_paths_regex)
        jsonld_paths = [p for p in jsonld_paths if rx.search(p)]

    if not jsonld_paths:
        raise RuntimeError(
            "No .jsonld/.json files found in GDPRtEXT repository. "
            "You may need to add Turtle parsing via rdflib if repo only provides TTL."
        )

    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    concepts: dict[str, dict[str, Any]] = {}
    parents: dict[str, list[str]] = defaultdict(list)
    schemes: dict[str, dict[str, Any]] = {}
    module_index: dict[str, dict[str, Any]] = defaultdict(lambda: {"files": [], "concept_count": 0})

    for path in jsonld_paths:
        module = Path(path).stem
        module_index[module]["files"].append(path)

        raw_path = raw_dir / path.replace("/", "__")
        if cache_raw and raw_path.exists():
            data = json.loads(raw_path.read_text(encoding="utf-8"))
        else:
            url = f"{RAW_BASE}{path}"
            data = fetch_json(url, github_token=github_token)
            if cache_raw:
                write_json(raw_path, data)

        for entry in iter_entries(data):
            iri = entry.get("@id")
            if not isinstance(iri, str) or iri.startswith("_:"):
                continue

            cid = shorten_gdpr_iri(iri)
            label, alt_labels = extract_labels(entry)
            definition = extract_definition(entry)
            types = extract_types(entry)

            broader = extract_id_refs(entry, SKOS_BROADER, shorten=shorten_gdpr_iri)
            subclass = extract_id_refs(entry, RDFS_SUBCLASS_OF, shorten=shorten_gdpr_iri)
            in_scheme = extract_id_refs(entry, SKOS_IN_SCHEME, shorten=shorten_gdpr_iri)
            top_of = extract_id_refs(entry, SKOS_TOP_CONCEPT_OF, shorten=shorten_gdpr_iri)

            text = extract_text(entry)
            kind, number = classify_concept(label, cid)

            if cid not in concepts:
                concepts[cid] = {
                    "id": cid,
                    "iri": iri,
                    "label": label or cid,
                    "definition": definition,
                    "alt_labels": alt_labels,
                    "types": types,
                    "schemes": sorted(set(in_scheme + top_of)),
                    "modules": [module],
                    "kind": kind,
                    "number": number,
                    "text": text,
                }
                module_index[module]["concept_count"] += 1
            else:
                # merge modules
                if module not in concepts[cid]["modules"]:
                    concepts[cid]["modules"].append(module)

                # merge missing fields
                if not concepts[cid].get("definition") and definition:
                    concepts[cid]["definition"] = definition
                if not concepts[cid].get("text") and text:
                    concepts[cid]["text"] = text

                # merge alt labels
                existing_alt = set(concepts[cid].get("alt_labels") or [])
                for a in alt_labels:
                    if a not in existing_alt and a != concepts[cid].get("label"):
                        concepts[cid].setdefault("alt_labels", []).append(a)
                        existing_alt.add(a)

                # merge types
                existing_types = set(concepts[cid].get("types") or [])
                for t in types:
                    if t not in existing_types:
                        concepts[cid].setdefault("types", []).append(t)
                        existing_types.add(t)

                # merge schemes
                existing_schemes = set(concepts[cid].get("schemes") or [])
                for s in in_scheme + top_of:
                    if s not in existing_schemes:
                        concepts[cid].setdefault("schemes", []).append(s)
                        existing_schemes.add(s)

            for p in broader + subclass:
                if p not in parents[cid]:
                    parents[cid].append(p)

            for s in in_scheme + top_of:
                schemes.setdefault(s, {"id": s, "label": s, "iri": None})

    # ensure every concept has parents list
    for cid in concepts.keys():
        parents.setdefault(cid, [])

    ancestors = compute_ancestors(dict(parents))

    module_index_final: dict[str, dict[str, Any]] = {}
    for module, meta in module_index.items():
        module_index_final[module] = {
            "module": module,
            "concept_count": meta["concept_count"],
            "files": sorted(set(meta["files"])),
        }

    return GDPRArtifacts(
        concepts=concepts,
        concept_parents=dict(parents),
        concept_ancestors=ancestors,
        schemes=schemes,
        module_index=module_index_final,
    )


def build_subsets(
    artifacts: GDPRArtifacts,
    out_dir: Path,
    subset_name: str,
    only_kinds: list[str] | None = None,
) -> None:
    subset_dir = out_dir / "subsets" / subset_name
    subset_dir.mkdir(parents=True, exist_ok=True)

    subset_concepts = {}
    for cid, c in artifacts.concepts.items():
        if only_kinds and c.get("kind") not in only_kinds:
            continue
        subset_concepts[cid] = c

    subset_parents: dict[str, list[str]] = {}
    for cid in subset_concepts.keys():
        subset_parents[cid] = [p for p in artifacts.concept_parents.get(cid, []) if p in subset_concepts]

    subset_anc = compute_ancestors(subset_parents)

    subset_schemes: dict[str, dict[str, Any]] = {}
    for cid, c in subset_concepts.items():
        for s in c.get("schemes") or []:
            subset_schemes[s] = {"id": s, "label": s, "iri": None}

    write_json(subset_dir / "concepts.json", [subset_concepts[k] for k in sorted(subset_concepts.keys())])
    write_json(subset_dir / "concept_parents.json", dict(sorted(subset_parents.items())))
    write_json(subset_dir / "concept_ancestors.json", dict(sorted(subset_anc.items())))
    write_json(subset_dir / "schemes.json", [subset_schemes[k] for k in sorted(subset_schemes.keys())])


def main() -> int:
    parser = argparse.ArgumentParser(description="Build structured GDPR artifacts (GDPRtEXT) for Moose.")
    parser.add_argument(
        "--out-dir",
        default="src/moose/data/policies/gdpr",
        help="Output directory (requested: src/moose/data/policies/gdpr)",
    )
    parser.add_argument("--no-cache-raw", action="store_true", help="Do not cache raw downloaded files")
    parser.add_argument(
        "--include-paths-regex",
        default=None,
        help="Optional regex to restrict downloaded JSON/JSON-LD file paths (debugging/size control).",
    )
    parser.add_argument("--build-subsets", action="store_true", help="Write curated subsets for articles/recitals.")
    parser.add_argument("--augment-keywords", action="store_true", help="Generate llm/keywords.json using DeepSeek.")
    parser.add_argument("--deepseek-model", default="deepseek-chat", help="DeepSeek model name.")

    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    github_token = os.getenv("GITHUB_TOKEN")

    artifacts = build_gdpr_artifacts(
        out_dir=out_dir,
        cache_raw=not bool(args.no_cache_raw),
        include_paths_regex=args.include_paths_regex,
        github_token=github_token,
    )

    # Write main artifacts
    concepts_list = [artifacts.concepts[k] for k in sorted(artifacts.concepts.keys())]
    write_json(out_dir / "concepts.json", concepts_list)
    write_json(out_dir / "concept_parents.json", dict(sorted(artifacts.concept_parents.items())))
    write_json(out_dir / "concept_ancestors.json", dict(sorted(artifacts.concept_ancestors.items())))
    write_json(out_dir / "schemes.json", [artifacts.schemes[k] for k in sorted(artifacts.schemes.keys())])
    write_json(out_dir / "module_index.json", dict(sorted(artifacts.module_index.items())))

    # Convenience outputs
    articles = [c for c in concepts_list if c.get("kind") == "article"]
    recitals = [c for c in concepts_list if c.get("kind") == "recital"]
    write_json(out_dir / "articles.json", articles)
    write_json(out_dir / "recitals.json", recitals)

    print("Wrote GDPR artifacts (GDPRtEXT):")
    print(f"  out_dir:     {out_dir}")
    print(f"  concepts:    {len(concepts_list)}")
    print(f"  articles:    {len(articles)}")
    print(f"  recitals:    {len(recitals)}")

    if args.build_subsets:
        build_subsets(artifacts, out_dir=out_dir, subset_name="articles_only", only_kinds=["article"])
        build_subsets(artifacts, out_dir=out_dir, subset_name="recitals_only", only_kinds=["recital"])
        print(f"  subsets:     {out_dir / 'subsets'}")

    if args.augment_keywords:
        kws = augment_keywords_with_deepseek(
            concepts=concepts_list,
            out_dir=out_dir,
            model=args.deepseek_model,
        )
        print(f"  llm keywords: {len(kws)} -> {out_dir / 'llm' / 'keywords.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())