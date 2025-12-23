from __future__ import annotations


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    total = sum(value for value in scores.values() if value is not None)
    if total <= 0:
        if not scores:
            return {}
        uniform = 1.0 / len(scores)
        return {key: uniform for key in scores}
    return {key: (value / total) for key, value in scores.items()}


def choose_argmax(scores: dict[str, float]) -> tuple[str, float, dict[str, float]]:
    normalized = normalize_scores(scores)
    if not normalized:
        raise ValueError("No scores provided")
    type_id = max(normalized, key=normalized.get)
    return type_id, normalized[type_id], normalized
