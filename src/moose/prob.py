from __future__ import annotations

import math
from typing import Any


def normalize_scores(scores: dict[str, float], *, clamp_negative: bool = True) -> dict[str, float]:
    cleaned: dict[str, float] = {}
    for k, v in scores.items():
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if clamp_negative and fv < 0:
            fv = 0.0
        cleaned[k] = fv

    total = sum(cleaned.values())
    if total <= 0:
        if not cleaned:
            return {}
        uniform = 1.0 / len(cleaned)
        return {k: uniform for k in cleaned}
    return {k: (v / total) for k, v in cleaned.items()}


def choose_argmax(scores: dict[str, float]) -> tuple[str, float, dict[str, float]]:
    normalized = normalize_scores(scores)
    if not normalized:
        raise ValueError("No scores provided")
    type_id = max(normalized, key=normalized.get)
    return type_id, normalized[type_id], normalized


def choose_argmax_with_metrics(scores: dict[str, float]) -> tuple[str, float, dict[str, float], dict[str, Any]]:
    type_id, conf, dist = choose_argmax(scores)
    sorted_items = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
    top = sorted_items[0][1] if sorted_items else 0.0
    second = sorted_items[1][1] if len(sorted_items) > 1 else 0.0
    margin = top - second

    entropy = 0.0
    for p in dist.values():
        if p > 0:
            entropy -= p * math.log(p)

    metrics = {
        "candidate_count": len(dist),
        "nonzero_count": sum(1 for p in dist.values() if p > 0),
        "margin": margin,
        "entropy": entropy,
        "effective_candidates": math.exp(entropy) if entropy > 0 else 1.0,
    }
    return type_id, conf, dist, metrics