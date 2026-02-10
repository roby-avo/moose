from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from moose.schema import DATA_DIR

PIPELINES_DIR = DATA_DIR / "pipelines"
PRIVACY_PROFILES_PATH = PIPELINES_DIR / "privacy_profiles.json"


@lru_cache
def load_privacy_profiles() -> dict[str, Any]:
    """
    Load pipelines/privacy_profiles.json.

    Expected shape:
    {
      "default_profile": "...",
      "profiles": {
        "fast": {"defaults": {...}, "escalation": [...]},
        "balanced": {...},
        ...
      }
    }
    """
    if not PRIVACY_PROFILES_PATH.exists():
        raise FileNotFoundError(f"Privacy profiles not found: {PRIVACY_PROFILES_PATH}")
    data = json.loads(PRIVACY_PROFILES_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("privacy_profiles.json must be a JSON object")
    return data


def list_privacy_profiles() -> list[str]:
    data = load_privacy_profiles()
    profiles = data.get("profiles", {})
    if not isinstance(profiles, dict):
        return []
    out = [k for k in profiles.keys() if isinstance(k, str) and k.strip()]
    return sorted(set(out))


def get_default_profile_name() -> str:
    data = load_privacy_profiles()
    default_profile = data.get("default_profile")
    if isinstance(default_profile, str) and default_profile.strip():
        return default_profile.strip()
    return "balanced"


def get_privacy_profile(profile: str | None) -> dict[str, Any]:
    """
    Return {"name": <profile_name>, ...profile_config...}
    """
    data = load_privacy_profiles()
    profiles = data.get("profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError("privacy_profiles.json missing 'profiles' object")

    if not profile:
        profile = get_default_profile_name()

    if profile not in profiles:
        known = ", ".join(sorted([k for k in profiles.keys() if isinstance(k, str)]))
        raise ValueError(f"Unknown profile '{profile}'. Available: {known}")

    cfg = profiles[profile]
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid profile definition for '{profile}' (must be a JSON object).")

    return {"name": profile, **cfg}


def resolve_privacy_defaults(profile: str | None, overrides: dict[str, Any]) -> dict[str, Any]:
    """
    Merge profile.defaults + overrides, where overrides wins when not None.

    Returns a dict with resolved runtime values plus:
      - _profile (name)
      - _profile_config (full config)
    """
    p = get_privacy_profile(profile)
    defaults = p.get("defaults", {})
    if not isinstance(defaults, dict):
        defaults = {}

    resolved: dict[str, Any] = dict(defaults)

    for key, value in overrides.items():
        if value is not None:
            resolved[key] = value

    resolved["_profile"] = p.get("name")
    resolved["_profile_config"] = p
    return resolved