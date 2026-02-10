#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def move_or_copy_dir(src: Path, dst: Path, mode: str, dry_run: bool) -> None:
    if not src.exists():
        return
    if dst.exists():
        # If destination exists, merge/copy in and optionally remove source.
        # We prefer "dirs_exist_ok=True" semantics for copy; for move we do a merge-copy then delete src.
        if mode == "copy":
            if dry_run:
                print(f"[DRY] copy tree merge {src} -> {dst}")
                return
            shutil.copytree(src, dst, dirs_exist_ok=True)
            return

        if mode == "move":
            if dry_run:
                print(f"[DRY] move tree merge {src} -> {dst} (copy then delete src)")
                return
            shutil.copytree(src, dst, dirs_exist_ok=True)
            shutil.rmtree(src)
            return

        raise ValueError(f"Unknown mode: {mode}")

    # Destination does not exist
    if mode == "copy":
        if dry_run:
            print(f"[DRY] copy tree {src} -> {dst}")
            return
        shutil.copytree(src, dst)
        return

    if mode == "move":
        if dry_run:
            print(f"[DRY] move tree {src} -> {dst}")
            return
        ensure_dir(dst.parent)
        shutil.move(str(src), str(dst))
        return

    raise ValueError(f"Unknown mode: {mode}")


def make_manifest_dpv(dpv_dir: Path) -> dict[str, Any]:
    concepts_path = dpv_dir / "concepts.json"
    version = None
    raw_dir = dpv_dir / "raw"
    if raw_dir.exists():
        # dpv/raw/<version>
        children = [p.name for p in raw_dir.iterdir() if p.is_dir()]
        if len(children) == 1:
            version = children[0]
        elif children:
            version = sorted(children)[-1]

    n_concepts = None
    if concepts_path.exists():
        n_concepts = len(read_json(concepts_path))

    subsets = {}
    subsets_dir = dpv_dir / "subsets"
    if subsets_dir.exists():
        for sub in sorted([p for p in subsets_dir.iterdir() if p.is_dir()]):
            sub_concepts = sub / "concepts.json"
            subsets[sub.name] = {
                "concept_count": len(read_json(sub_concepts)) if sub_concepts.exists() else None
            }

    return {
        "name": "dpv",
        "label": "W3C Data Privacy Vocabulary (DPV)",
        "version": version,
        "generated_at": utc_now_iso(),
        "paths": {
            "concepts": str(concepts_path.relative_to(dpv_dir.parent)),
            "concept_parents": str((dpv_dir / "concept_parents.json").relative_to(dpv_dir.parent)),
            "concept_ancestors": str((dpv_dir / "concept_ancestors.json").relative_to(dpv_dir.parent)),
        },
        "counts": {"concepts": n_concepts},
        "subsets": subsets,
        "source": {"repo": "https://github.com/w3c/dpv", "format": "jsonld"},
    }


def make_manifest_schemaorg(schema_dir: Path) -> dict[str, Any]:
    classes = schema_dir / "classes.json"
    props = schema_dir / "properties.json"

    subsets = {}
    subsets_dir = schema_dir / "subsets"
    if subsets_dir.exists():
        for sub in sorted([p for p in subsets_dir.iterdir() if p.is_dir()]):
            subsets[sub.name] = {
                "class_count": len(read_json(sub / "classes.json")) if (sub / "classes.json").exists() else None,
                "property_count": len(read_json(sub / "properties.json")) if (sub / "properties.json").exists() else None,
            }

    return {
        "name": "schemaorg",
        "label": "schema.org",
        "version": "latest",
        "generated_at": utc_now_iso(),
        "counts": {
            "classes": len(read_json(classes)) if classes.exists() else None,
            "properties": len(read_json(props)) if props.exists() else None,
        },
        "subsets": subsets,
        "source": {
            "url": "https://schema.org/version/latest/schemaorg-current-https.jsonld",
            "format": "jsonld",
        },
    }


def make_manifest_gdprtext(gdpr_dir: Path) -> dict[str, Any]:
    concepts = gdpr_dir / "concepts.json"
    articles = gdpr_dir / "articles.json"
    recitals = gdpr_dir / "recitals.json"

    return {
        "name": "gdprtext",
        "label": "GDPRtEXT (GDPR articles/recitals as linked data)",
        "version": "unknown",
        "generated_at": utc_now_iso(),
        "counts": {
            "concepts": len(read_json(concepts)) if concepts.exists() else None,
            "articles": len(read_json(articles)) if articles.exists() else None,
            "recitals": len(read_json(recitals)) if recitals.exists() else None,
        },
        "source": {"repo": "https://github.com/coolharsh55/GDPRtEXT", "format": "jsonld/json"},
    }


def generate_schemaorg_label_sets(schemaorg_dir: Path, subset_name: str, dry_run: bool) -> None:
    subset_dir = schemaorg_dir / "subsets" / subset_name
    classes_path = subset_dir / "classes.json"
    props_path = subset_dir / "properties.json"

    if not classes_path.exists() or not props_path.exists():
        raise FileNotFoundError(f"Missing schema.org subset files under {subset_dir}")

    classes = read_json(classes_path)
    props = read_json(props_path)
    if not isinstance(classes, list) or not isinstance(props, list):
        raise ValueError("schemaorg subset classes/properties must be JSON arrays")

    # CTA label set: classes + moose:OTHER
    cta_labels = [
        {"id": "moose:OTHER", "iri": None, "label": "Other / Unknown Class"},
        *classes,
    ]

    # CPA label set: properties + moose:NONE + moose:OTHER
    cpa_labels = [
        {"id": "moose:NONE", "iri": None, "label": "No Relationship"},
        {"id": "moose:OTHER", "iri": None, "label": "Other Relationship"},
        *props,
    ]

    cta_out = subset_dir / "cta_labels.json"
    cpa_out = subset_dir / "cpa_labels.json"

    if dry_run:
        print(f"[DRY] would write {cta_out} ({len(cta_labels)} labels)")
        print(f"[DRY] would write {cpa_out} ({len(cpa_labels)} labels)")
        return

    write_json(cta_out, cta_labels)
    write_json(cpa_out, cpa_labels)


def write_privacy_profiles(pipelines_dir: Path, dry_run: bool, force: bool) -> None:
    out_path = pipelines_dir / "privacy_profiles.json"
    if out_path.exists() and not force:
        print(f"[SKIP] {out_path} exists (use --force to overwrite)")
        return

    profiles = {
        "default_profile": "balanced",
        "profiles": {
            "fast": {
                "description": "Fast, low-cost privacy scan. Uses small vocabularies and rules only.",
                "defaults": {
                    "policy_pack": "gdpr_basic",
                    "analysis_mode": "rules",
                    "text_schema": "dpv_pd",
                    "table_schema": "sti",
                    "scan_schema": "dpv_pd",
                    "include_extraction": True
                },
                "escalation": []
            },
            "balanced": {
                "description": "Balanced accuracy/cost. Uses dpv_pd + assessor; escalates selectively.",
                "defaults": {
                    "policy_pack": "gdpr_basic",
                    "analysis_mode": "hybrid",
                    "text_schema": "dpv_pd",
                    "table_schema": "sti",
                    "scan_schema": "dpv_pd",
                    "include_extraction": True
                },
                "escalation": [
                    {
                        "when": "uncertain_findings",
                        "text_schema": "dpv",
                        "max_extra_passes": 1,
                        "note": "Escalate only tasks that remain uncertain after rules+assessor."
                    }
                ]
            },
            "deep": {
                "description": "Deep semantic privacy analysis (slow). Uses full DPV by default.",
                "defaults": {
                    "policy_pack": "gdpr_basic",
                    "analysis_mode": "hybrid",
                    "text_schema": "dpv",
                    "table_schema": "sti",
                    "scan_schema": "dpv",
                    "include_extraction": True
                },
                "escalation": []
            }
        }
    }

    if dry_run:
        print(f"[DRY] would write {out_path}")
        return
    write_json(out_path, profiles)


def update_vocabularies_json(data_dir: Path, dry_run: bool, backup: bool = True) -> None:
    vocab_path = data_dir / "vocabularies.json"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing {vocab_path}")

    vocab = read_json(vocab_path)
    if not isinstance(vocab, list):
        raise ValueError("vocabularies.json must be a JSON array")

    def upsert(entry: dict[str, Any]) -> None:
        for i, e in enumerate(vocab):
            if isinstance(e, dict) and e.get("name") == entry["name"]:
                vocab[i] = {**e, **entry}
                return
        vocab.append(entry)

    # Update dpv to use dpv/concepts.json (one source of truth)
    upsert(
        {
            "name": "dpv",
            "label": "DPV (Full)",
            "description": "Data Privacy Vocabulary (W3C) – full concept set (deep/slow).",
            "type_source": "dpv/concepts.json",
            "score_mode": "sparse",
            "text_intro": "You are a DPV annotation engine.",
            "table_intro": "You are a DPV classification engine for tabular data.",
            "prefilter_types": True,
            # IMPORTANT: remove the old dpv-pd -> dpv prefix alias (it is backwards/unsafe)
            "type_alias_prefixes": {}
        }
    )

    # Add dpv_pd (fast default)
    upsert(
        {
            "name": "dpv_pd",
            "label": "DPV Personal Data (Fast)",
            "description": "DPV personal data subset for fast privacy detection.",
            "type_source": "dpv/subsets/dpv_pd_only/concepts.json",
            "score_mode": "sparse",
            "text_intro": "You are a DPV personal data annotation engine.",
            "table_intro": "You are a DPV personal data classification engine for tabular data.",
            "prefilter_types": True,
            "type_alias_prefixes": {}
        }
    )

    # Add schemaorg CTA and CPA schemas based on curated subset
    schema_sub = "schemaorg/subsets/curated_v1"
    upsert(
        {
            "name": "schemaorg_cta_v1",
            "label": "schema.org CTA (curated_v1)",
            "description": "schema.org class typing (CTA) using curated_v1 class subset.",
            "type_source": f"{schema_sub}/cta_labels.json",
            "score_mode": "sparse",
            "supports_text": False,
            "supports_table": True,
            "table_intro": (
                "You are a schema.org class typing engine for tables. "
                "Assign one schema.org class per column based on header and sample values. "
                "If no class applies, choose moose:OTHER."
            ),
            "prefilter_types": True,
            "type_alias_prefixes": {
                "https://schema.org/": "schema:",
                "http://schema.org/": "schema:"
            }
        }
    )

    upsert(
        {
            "name": "schemaorg_cpa_v1",
            "label": "schema.org CPA (curated_v1)",
            "description": "schema.org predicate CPA using curated_v1 property subset (+ moose:NONE/OTHER).",
            "type_source": f"{schema_sub}/cpa_labels.json",
            "score_mode": "sparse",
            "supports_text": False,
            "supports_table": False,
            "supports_cpa": True,
            "cpa_intro": (
                "You are a column relationship prediction engine. "
                "Predict the schema.org property relating the SUBJECT column to each TARGET column. "
                "Choose ONLY from the provided labels. "
                "If no relationship exists, choose moose:NONE. "
                "If a relationship exists but is not represented, choose moose:OTHER."
            ),
            "prefilter_types": True,
            "type_alias_prefixes": {
                "https://schema.org/": "schema:",
                "http://schema.org/": "schema:"
            }
        }
    )

    # stable ordering by schema name
    vocab_sorted = sorted(vocab, key=lambda x: (x.get("name") if isinstance(x, dict) else ""))

    if dry_run:
        print(f"[DRY] would update {vocab_path} (schemas: {len(vocab_sorted)})")
        return

    if backup:
        backup_path = vocab_path.with_suffix(".json.bak")
        backup_path.write_text(vocab_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[OK] backup written: {backup_path}")

    write_json(vocab_path, vocab_sorted)
    print(f"[OK] updated: {vocab_path}")


def relpath_from_data_dir(data_dir: Path, path: Path) -> str:
    """
    Return a posix-style relative path from data_dir.
    """
    return path.resolve().relative_to(data_dir.resolve()).as_posix()


def list_policy_packs_in_dir(policies_dir: Path) -> list[dict[str, Any]]:
    """
    Policy packs are JSON files directly under policies/.
    Excludes directories like policies/gdprtext/ or policies/gdpr/ etc.
    """
    packs: list[dict[str, Any]] = []
    if not policies_dir.exists():
        return packs

    for p in sorted(policies_dir.glob("*.json")):
        try:
            data = read_json(p)
            name = data.get("name") if isinstance(data, dict) else None
            if not isinstance(name, str) or not name.strip():
                name = p.stem
            packs.append(
                {
                    "name": name,
                    "path": relpath_from_data_dir(policies_dir.parent, p),
                    "label": data.get("label") if isinstance(data, dict) else None,
                    "version": data.get("version") if isinstance(data, dict) else None,
                }
            )
        except Exception:
            # If a pack is malformed, still list it by filename so you can debug it.
            packs.append({"name": p.stem, "path": relpath_from_data_dir(policies_dir.parent, p)})

    return packs


def build_assets_index(data_dir: Path) -> dict[str, Any]:
    """
    Create a global discovery index for all assets under src/moose/data.
    """
    # Discover ontology manifests
    ontologies: dict[str, dict[str, Any]] = {}

    dpv_manifest = data_dir / "dpv" / "manifest.json"
    if dpv_manifest.exists():
        ontologies["dpv"] = {"manifest": relpath_from_data_dir(data_dir, dpv_manifest)}

    schemaorg_manifest = data_dir / "schemaorg" / "manifest.json"
    if schemaorg_manifest.exists():
        ontologies["schemaorg"] = {"manifest": relpath_from_data_dir(data_dir, schemaorg_manifest)}

    # Discover legal asset manifests
    legal_assets: dict[str, dict[str, Any]] = {}
    legal_dir = data_dir / "legal"
    if legal_dir.exists():
        for sub in sorted([p for p in legal_dir.iterdir() if p.is_dir()]):
            manifest = sub / "manifest.json"
            if manifest.exists():
                legal_assets[sub.name] = {"manifest": relpath_from_data_dir(data_dir, manifest)}

    # Pipelines
    pipelines: dict[str, Any] = {}
    privacy_profiles = data_dir / "pipelines" / "privacy_profiles.json"
    if privacy_profiles.exists():
        pipelines["privacy_profiles"] = relpath_from_data_dir(data_dir, privacy_profiles)

    # Policy packs
    policies_dir = data_dir / "policies"
    policy_packs = list_policy_packs_in_dir(policies_dir)

    return {
        "generated_at": utc_now_iso(),
        # keep it stable and explicit
        "data_root": "src/moose/data",
        "registries": {
            "vocabularies": "vocabularies.json",
            "pipelines": pipelines,
        },
        "assets": {
            "ontologies": ontologies,
            "legal": legal_assets,
            "policy_packs": policy_packs,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Organize Moose data assets into a modular structure.")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to src/moose/data (default: auto-detect from repo root).",
    )
    parser.add_argument("--apply", action="store_true", help="Actually apply changes (default is dry-run).")
    parser.add_argument("--copy", action="store_true", help="Copy instead of move for GDPRtEXT relocation.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing generated files (profiles/manifests).")
    parser.add_argument("--update-vocabularies", action="store_true", help="Update vocabularies.json entries.")
    parser.add_argument("--schemaorg-subset", default="curated_v1", help="schema.org subset name (default curated_v1)")
    args = parser.parse_args()

    dry_run = not args.apply
    mode = "copy" if args.copy else "move"

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir).resolve() if args.data_dir else (repo_root / "src" / "moose" / "data")
    if not data_dir.exists():
        raise SystemExit(f"data dir not found: {data_dir}")

    # 1) Create new directories
    pipelines_dir = data_dir / "pipelines"
    legal_dir = data_dir / "legal"
    gdprtext_dir = legal_dir / "gdprtext"

    if dry_run:
        print(f"[DRY] data_dir={data_dir}")
        print(f"[DRY] ensure {pipelines_dir}")
        print(f"[DRY] ensure {gdprtext_dir}")
    else:
        ensure_dir(pipelines_dir)
        ensure_dir(gdprtext_dir)

    # 2) Move GDPRtEXT artifacts out of policies/
    gdpr_src = data_dir / "policies" / "gdpr"
    if gdpr_src.exists():
        move_or_copy_dir(gdpr_src, gdprtext_dir, mode=mode, dry_run=dry_run)
        if not dry_run and mode == "move":
            # Leave a note so it's obvious what happened
            ensure_dir(gdpr_src)
            (gdpr_src / "MOVED_TO_LEGAL_GDPRTEXT.txt").write_text(
                "This folder was moved to src/moose/data/legal/gdprtext\n",
                encoding="utf-8",
            )
    else:
        print(f"[INFO] GDPRtEXT source folder not found (skipping): {gdpr_src}")

    # 3) Generate schema.org CTA/CPA label sets
    schemaorg_dir = data_dir / "schemaorg"
    generate_schemaorg_label_sets(schemaorg_dir, subset_name=args.schemaorg_subset, dry_run=dry_run)

    # 4) Write privacy profiles
    write_privacy_profiles(pipelines_dir, dry_run=dry_run, force=bool(args.force))

    # 5) Write manifests
    if not dry_run or args.force:
        # DPV manifest
        dpv_dir = data_dir / "dpv"
        if dpv_dir.exists():
            m = make_manifest_dpv(dpv_dir)
            if dry_run:
                print(f"[DRY] would write {dpv_dir / 'manifest.json'}")
            else:
                write_json(dpv_dir / "manifest.json", m)

        # schema.org manifest
        if schemaorg_dir.exists():
            m = make_manifest_schemaorg(schemaorg_dir)
            if dry_run:
                print(f"[DRY] would write {schemaorg_dir / 'manifest.json'}")
            else:
                write_json(schemaorg_dir / "manifest.json", m)

        # GDPRtEXT manifest
        if gdprtext_dir.exists():
            m = make_manifest_gdprtext(gdprtext_dir)
            if dry_run:
                print(f"[DRY] would write {gdprtext_dir / 'manifest.json'}")
            else:
                write_json(gdprtext_dir / "manifest.json", m)

    # 6) Update vocabularies.json
    if args.update_vocabularies:
        update_vocabularies_json(data_dir, dry_run=dry_run, backup=True)

    # 7) Write global assets index
    assets_index_path = data_dir / "assets_index.json"
    if dry_run:
        print(f"[DRY] would write {assets_index_path}")
    else:
        index = build_assets_index(data_dir)
        write_json(assets_index_path, index)
        print(f"[OK] wrote: {assets_index_path}")

    print("[DONE]" if not dry_run else "[DRY-RUN DONE]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())