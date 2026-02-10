#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx


# -----------------------
# Source configuration
# -----------------------
DEFAULT_VERSION = "2.2"
TREE_URL = "https://api.github.com/repos/w3c/dpv/git/trees/master?recursive=1"
RAW_BASE = "https://raw.githubusercontent.com/w3c/dpv/master/"


# -----------------------
# RDF-ish constants (full IRIs, because DPV JSON-LD uses them)
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


# -----------------------
# Optional DeepSeek augmentation
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
    batch_size: int = 30,
    model: str = "deepseek-chat",
) -> dict[str, list[str]]:
    """
    OPTIONAL offline augmentation:
      - produces keywords for search/matching (not authoritative ontology data)
      - uses caching to avoid re-calling on subsequent runs

    Output files:
      out_dir/llm/keywords.json  (id -> keywords)
      out_dir/llm/cache.json     (id -> payload hash or just stored keywords)
    """
    client = _deepseek_client()
    if client is None:
        raise RuntimeError(
            "DeepSeek augmentation requested but DEEPSEEK_API_KEY or openai package not available."
        )

    llm_dir = out_dir / "llm"
    llm_dir.mkdir(parents=True, exist_ok=True)
    cache_path = llm_dir / "cache.json"
    keywords_path = llm_dir / "keywords.json"

    cache: dict[str, Any] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    keywords: dict[str, list[str]] = {}
    if keywords_path.exists():
        keywords = json.loads(keywords_path.read_text(encoding="utf-8"))

    # pick candidates that do not yet have keywords
    pending = []
    for c in concepts:
        cid = c["id"]
        if cid in keywords:
            continue
        pending.append(c)

    if not pending:
        return keywords

    system_prompt = (
        "You generate lightweight search keywords for DPV concepts.\n"
        "Return ONLY JSON with the shape:\n"
        '{ "items": [ { "id": "...", "keywords": ["k1","k2",...]} ] }\n'
        "Rules:\n"
        "- keywords must be lowercase\n"
        "- 3 to 8 keywords per item\n"
        "- single words or short phrases (<= 3 words)\n"
        "- do not include the dpv prefix (e.g. 'dpv-pd:')\n"
        "- do not hallucinate unrelated topics\n"
        "- use label + definition when present\n"
    )

    def chunk(it: list[Any], n: int) -> Iterable[list[Any]]:
        for i in range(0, len(it), n):
            yield it[i : i + n]

    for batch in chunk(pending, batch_size):
        payload = []
        for c in batch:
            payload.append(
                {
                    "id": c["id"],
                    "label": c.get("label"),
                    "definition": c.get("definition"),
                    "alt_labels": c.get("alt_labels", []),
                }
            )

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
            kw_list = [k.strip().lower() for k in kws if isinstance(k, str) and k.strip()]
            # dedupe while preserving order
            seen = set()
            dedup = []
            for k in kw_list:
                if k not in seen:
                    dedup.append(k)
                    seen.add(k)
            if dedup:
                keywords[cid] = dedup
                cache[cid] = {"keywords": dedup}

        # incremental save (safer if interrupted)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        keywords_path.write_text(json.dumps(keywords, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    return keywords


# -----------------------
# HTTP helpers
# -----------------------
def fetch_json(url: str, github_token: str | None = None) -> Any:
    headers = {"User-Agent": "moose-dpv-build"}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    with httpx.Client(timeout=60.0, follow_redirects=True, headers=headers) as client:
        r = client.get(url)
        r.raise_for_status()
        return r.json()


# -----------------------
# DPV ID normalization
# -----------------------
def shorten_dpv_iri(iri: str) -> str:
    """
    Convert DPV IRIs into compact IDs consistent with your existing dpv_full.json convention:

    - https://w3id.org/dpv#X         -> dpv:X
    - https://w3id.org/dpv/pd#X      -> dpv-pd:X
    - https://w3id.org/dpv/ai#AGI    -> dpv-ai:AGI
    - https://w3id.org/dpv/tech#Foo  -> dpv-tech:Foo
    """
    if not isinstance(iri, str):
        return str(iri)

    for base in ("https://w3id.org/dpv/", "http://w3id.org/dpv/"):
        if iri.startswith(base):
            rest = iri[len(base) :]
            # handle https://w3id.org/dpv/#X or dpv/ #X
            if rest.startswith("#"):
                return f"dpv:{rest[1:]}"
            # handle namespace: dpv/<namespace>#<local>
            if "#" in rest:
                namespace, local = rest.split("#", 1)
                return f"dpv-{namespace}:{local}"
            # handle /dpv/<local> (rare)
            return f"dpv:{rest}"

    for base in ("https://w3id.org/dpv#", "http://w3id.org/dpv#"):
        if iri.startswith(base):
            return f"dpv:{iri[len(base):]}"

    return iri


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
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        graph = data.get("@graph")
        if isinstance(graph, list):
            return [item for item in graph if isinstance(item, dict)]
    return []


def pick_lang_value(values: Any, lang: str = "en") -> str | None:
    """
    DPV uses JSON-LD literals like:
      [{"@language":"en","@value":"..."}]
    """
    if values is None:
        return None
    if isinstance(values, dict):
        values = [values]
    if isinstance(values, str):
        return values.strip() or None
    if not isinstance(values, list):
        return None

    # prefer requested language
    for item in values:
        if isinstance(item, dict) and item.get("@language") == lang and isinstance(item.get("@value"), str):
            v = item["@value"].strip()
            if v:
                return v

    # fallback: any @value
    for item in values:
        if isinstance(item, dict) and isinstance(item.get("@value"), str):
            v = item["@value"].strip()
            if v:
                return v

    # fallback: first string
    for item in values:
        if isinstance(item, str) and item.strip():
            return item.strip()

    return None


def extract_labels(entry: dict[str, Any]) -> tuple[str | None, list[str]]:
    """
    Return (label, alt_labels)
    """
    label = pick_lang_value(entry.get(SKOS_PREF_LABEL)) or pick_lang_value(entry.get(RDFS_LABEL))
    alt_values = entry.get(SKOS_ALT_LABEL)
    alt_labels: list[str] = []
    if alt_values:
        if isinstance(alt_values, dict):
            alt_values = [alt_values]
        if isinstance(alt_values, list):
            for item in alt_values:
                s = pick_lang_value(item)
                if s:
                    alt_labels.append(s)

    # dedupe alt_labels
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


def extract_id_refs(entry: dict[str, Any], key: str) -> list[str]:
    out: list[str] = []
    for item in as_list(entry.get(key)):
        if isinstance(item, str):
            out.append(shorten_dpv_iri(item))
        elif isinstance(item, dict) and isinstance(item.get("@id"), str):
            out.append(shorten_dpv_iri(item["@id"]))
    # dedupe preserve order
    seen = set()
    dedup = []
    for x in out:
        if x not in seen:
            dedup.append(x)
            seen.add(x)
    return dedup


@dataclass
class DPVArtifacts:
    concepts: dict[str, dict[str, Any]]            # id -> concept object
    concept_parents: dict[str, list[str]]          # id -> parents
    concept_ancestors: dict[str, list[str]]        # id -> ancestors
    schemes: dict[str, dict[str, Any]]             # scheme_id -> metadata
    module_index: dict[str, dict[str, Any]]        # module_name -> info


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


def list_jsonld_paths(version: str, github_token: str | None) -> list[str]:
    tree = fetch_json(TREE_URL, github_token=github_token)
    paths: list[str] = []
    for item in tree.get("tree", []):
        path = item.get("path", "")
        if not isinstance(path, str):
            continue
        if not path.startswith(f"{version}/"):
            continue
        if not path.endswith(".jsonld"):
            continue
        # we’ll allow --include-owl to override this later
        paths.append(path)
    return sorted(set(paths))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def build_dpv_artifacts(
    version: str,
    include_owl: bool,
    cache_raw: bool,
    out_dir: Path,
    github_token: str | None,
) -> DPVArtifacts:
    paths = list_jsonld_paths(version, github_token=github_token)

    if not include_owl:
        paths = [p for p in paths if "-owl" not in p]

    if not paths:
        raise RuntimeError("No JSON-LD files found; check DPV repository/version.")

    concepts: dict[str, dict[str, Any]] = {}
    parents: dict[str, list[str]] = defaultdict(list)
    schemes: dict[str, dict[str, Any]] = {}
    module_index: dict[str, dict[str, Any]] = defaultdict(lambda: {"files": [], "concept_count": 0})

    raw_dir = out_dir / "raw" / version
    raw_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        module = Path(path).stem  # e.g. dpv-pd, dpv, dpv-ai, etc.
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

            cid = shorten_dpv_iri(iri)
            label, alt_labels = extract_labels(entry)
            definition = extract_definition(entry)
            types = extract_types(entry)

            broader = extract_id_refs(entry, SKOS_BROADER)
            subclass_of = extract_id_refs(entry, RDFS_SUBCLASS_OF)
            in_scheme = extract_id_refs(entry, SKOS_IN_SCHEME)
            top_of = extract_id_refs(entry, SKOS_TOP_CONCEPT_OF)

            # Build/merge concept record
            if cid not in concepts:
                concepts[cid] = {
                    "id": cid,
                    "iri": iri,
                    "label": label or cid,
                    "definition": definition,
                    "alt_labels": alt_labels,
                    "types": types,
                    "modules": [module],
                    "schemes": sorted(set(in_scheme + top_of)),
                }
                module_index[module]["concept_count"] += 1
            else:
                # merge modules
                if module not in concepts[cid]["modules"]:
                    concepts[cid]["modules"].append(module)

                # merge label/definition if missing
                if (not concepts[cid].get("label")) and label:
                    concepts[cid]["label"] = label
                if (not concepts[cid].get("definition")) and definition:
                    concepts[cid]["definition"] = definition

                # merge alt labels
                existing_alt = set(concepts[cid].get("alt_labels") or [])
                for a in alt_labels:
                    if a not in existing_alt:
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

            # Parents: combine SKOS broader + rdfs:subClassOf
            for p in broader + subclass_of:
                if p not in parents[cid]:
                    parents[cid].append(p)

            # Track schemes seen
            for s in in_scheme + top_of:
                schemes.setdefault(s, {"id": s, "label": s, "iri": None})

    # Ensure every concept has parents list
    for cid in concepts.keys():
        parents.setdefault(cid, [])

    # Ancestor closure
    ancestors = compute_ancestors(dict(parents))

    # Sort module file lists
    module_index_final: dict[str, dict[str, Any]] = {}
    for module, meta in module_index.items():
        module_index_final[module] = {
            "module": module,
            "concept_count": meta["concept_count"],
            "files": sorted(set(meta["files"])),
        }

    return DPVArtifacts(
        concepts=concepts,
        concept_parents=dict(parents),
        concept_ancestors=ancestors,
        schemes=schemes,
        module_index=module_index_final,
    )


def write_artifacts(out_dir: Path, artifacts: DPVArtifacts) -> None:
    concepts_list = [artifacts.concepts[k] for k in sorted(artifacts.concepts.keys())]
    write_json(out_dir / "concepts.json", concepts_list)
    write_json(out_dir / "concept_parents.json", dict(sorted(artifacts.concept_parents.items())))
    write_json(out_dir / "concept_ancestors.json", dict(sorted(artifacts.concept_ancestors.items())))
    write_json(out_dir / "module_index.json", dict(sorted(artifacts.module_index.items())))
    write_json(out_dir / "schemes.json", [artifacts.schemes[k] for k in sorted(artifacts.schemes.keys())])


def write_legacy_dpv_full(out_path: Path, artifacts: DPVArtifacts) -> None:
    """
    Backward compatible file that matches your old dpv_full.json format:
      [{id, label, iri}, ...]
    """
    entries: list[dict[str, str]] = []
    for cid in sorted(artifacts.concepts.keys()):
        c = artifacts.concepts[cid]
        entries.append({"id": cid, "label": c.get("label") or cid, "iri": c.get("iri") or ""})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(entries, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


def build_subset(
    artifacts: DPVArtifacts,
    out_dir: Path,
    subset_name: str,
    module_allowlist: list[str] | None = None,
    id_prefix_allowlist: list[str] | None = None,
) -> None:
    """
    Deterministic curated subset builder.

    Examples:
      - module_allowlist=["dpv-pd"]  (personal data only)
      - id_prefix_allowlist=["dpv-pd:"]  (same idea)
    """
    subset_dir = out_dir / "subsets" / subset_name
    subset_dir.mkdir(parents=True, exist_ok=True)

    def allowed(cid: str, c: dict[str, Any]) -> bool:
        if module_allowlist:
            if not any(m in module_allowlist for m in (c.get("modules") or [])):
                return False
        if id_prefix_allowlist:
            if not any(cid.startswith(pfx) for pfx in id_prefix_allowlist):
                return False
        return True

    subset_concepts = {cid: c for cid, c in artifacts.concepts.items() if allowed(cid, c)}
    subset_parents: dict[str, list[str]] = {}
    for cid in subset_concepts.keys():
        subset_parents[cid] = [p for p in artifacts.concept_parents.get(cid, []) if p in subset_concepts]

    subset_anc = compute_ancestors(subset_parents)

    # subset schemes and module index (best-effort)
    schemes: dict[str, dict[str, Any]] = {}
    module_index: dict[str, dict[str, Any]] = defaultdict(lambda: {"concept_count": 0, "files": []})
    for cid, c in subset_concepts.items():
        for s in c.get("schemes") or []:
            schemes[s] = {"id": s, "label": s, "iri": None}
        for m in c.get("modules") or []:
            module_index[m]["concept_count"] += 1

    concepts_list = [subset_concepts[k] for k in sorted(subset_concepts.keys())]
    write_json(subset_dir / "concepts.json", concepts_list)
    write_json(subset_dir / "concept_parents.json", dict(sorted(subset_parents.items())))
    write_json(subset_dir / "concept_ancestors.json", dict(sorted(subset_anc.items())))
    write_json(subset_dir / "schemes.json", [schemes[k] for k in sorted(schemes.keys())])
    write_json(
        subset_dir / "module_index.json",
        {k: {"module": k, "concept_count": v["concept_count"], "files": []} for k, v in sorted(module_index.items())},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build structured DPV artifacts for Moose.")
    parser.add_argument("--version", default=DEFAULT_VERSION, help="DPV version directory (e.g. 2.2)")
    parser.add_argument("--out-dir", default="src/moose/data/dpv", help="Output directory")
    parser.add_argument("--include-owl", action="store_true", help="Include *-owl.jsonld files")
    parser.add_argument("--no-cache-raw", action="store_true", help="Do not cache raw JSON-LD files")
    parser.add_argument(
        "--write-legacy-dpv-full",
        action="store_true",
        help="Also write src/moose/data/dpv_full.json for backward compatibility",
    )
    parser.add_argument(
        "--build-subset-pd",
        action="store_true",
        help="Also build a dpv-pd-only subset under subsets/dpv_pd_only/",
    )

    # Optional DeepSeek augmentation
    parser.add_argument("--augment-keywords", action="store_true", help="Generate llm/keywords.json using DeepSeek.")
    parser.add_argument("--deepseek-model", default="deepseek-chat", help="DeepSeek model (default deepseek-chat)")

    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    github_token = os.getenv("GITHUB_TOKEN")

    artifacts = build_dpv_artifacts(
        version=args.version,
        include_owl=bool(args.include_owl),
        cache_raw=not bool(args.no_cache_raw),
        out_dir=out_dir,
        github_token=github_token,
    )
    write_artifacts(out_dir, artifacts)

    print("Wrote DPV artifacts:")
    print(f"  out_dir: {out_dir}")
    print(f"  concepts: {len(artifacts.concepts)}")

    if args.write_legacy_dpv_full:
        legacy_path = out_dir.parents[0] / "dpv_full.json"  # src/moose/data/dpv_full.json if out_dir is src/moose/data/dpv
        write_legacy_dpv_full(legacy_path, artifacts)
        print(f"  legacy: {legacy_path}")

    if args.build_subset_pd:
        build_subset(
            artifacts,
            out_dir=out_dir,
            subset_name="dpv_pd_only",
            id_prefix_allowlist=["dpv-pd:"],
        )
        print(f"  subset: {out_dir / 'subsets' / 'dpv_pd_only'}")

    if args.augment_keywords:
        concepts_list = [artifacts.concepts[k] for k in sorted(artifacts.concepts.keys())]
        kws = augment_keywords_with_deepseek(
            concepts=concepts_list,
            out_dir=out_dir,
            model=args.deepseek_model,
        )
        print(f"  llm keywords: {len(kws)} entries -> {out_dir / 'llm' / 'keywords.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())